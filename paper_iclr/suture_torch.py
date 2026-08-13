"""PyTorch/Hugging Face adapter for the SUTURE reference metrics.

The NumPy implementation in :mod:`suture_metrics` intentionally knows nothing
about a transformer implementation.  This module is the real-model boundary:
it captures the host trajectory with the model's own forward pass, evaluates a
donor block at the corresponding host state, and obtains all adjoints from the
same retained trajectory.

The adapter supports decoder-only Hugging Face models whose sequential blocks
are exposed through one of the common paths (Qwen/Llama ``model.layers``,
GPT-2 ``transformer.h``, GPT-NeoX ``gpt_neox.layers`` and decoder-style
``decoder.layers``).  It does not guess an architecture: unsupported models
raise ``AdapterError`` with the discovered paths.

Selection is protected by a hard phase guard.  ``build_grafted_model`` raises
while a ``selection_phase`` is active, which makes an accidental merged-model
build during score computation a testable invariant rather than a convention.
The returned graft model is a zero-copy routed evaluation wrapper: it reuses
the host and donor block parameters and dispatches selected blocks to the
donor.  It is still counted as one merged-model evaluation in experiment
artifacts, while avoiding a second 1B--2B parameter allocation on a small GPU.
"""

from __future__ import annotations

import contextlib
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn


class AdapterError(RuntimeError):
    """Raised when an adapter contract cannot be satisfied."""


@dataclass
class TorchPrompt:
    """One batched prompt and the token ids used by the two readouts.

    ``input_ids`` is ``[T]`` or ``[B, T]``.  A utility readout uses the
    probability of ``utility_token_ids`` at the final position.  A risk
    readout uses the target-language probability minus the donor-language
    probability at that same position.  The latter is deliberately explicit:
    the adapter never infers language anchors from text or silently substitutes
    a token.
    """

    input_ids: Tensor
    attention_mask: Optional[Tensor] = None
    utility_token_ids: Optional[Tensor | Sequence[int] | int] = None
    risk_target_token_ids: Optional[Tensor | Sequence[int] | int] = None
    risk_donor_token_ids: Optional[Tensor | Sequence[int] | int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.input_ids, Tensor):
            self.input_ids = torch.as_tensor(self.input_ids, dtype=torch.long)
        if self.input_ids.ndim not in (1, 2):
            raise AdapterError("input_ids must have shape [T] or [B,T]")
        if self.input_ids.dtype not in (torch.int64, torch.int32):
            self.input_ids = self.input_ids.to(dtype=torch.long)
        if self.attention_mask is not None and not isinstance(self.attention_mask, Tensor):
            self.attention_mask = torch.as_tensor(self.attention_mask, dtype=torch.long)
        if self.attention_mask is not None and self.attention_mask.shape != self.input_ids.shape:
            raise AdapterError("attention_mask must have the same shape as input_ids")

    @property
    def batch_size(self) -> int:
        return int(self.input_ids.shape[0] if self.input_ids.ndim == 2 else 1)

    def model_inputs(self, device: torch.device) -> Dict[str, Tensor]:
        ids = self.input_ids.to(device)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        mask = self.attention_mask
        if mask is None:
            mask = torch.ones_like(ids, dtype=torch.long)
        else:
            mask = mask.to(device)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
        return {
            "input_ids": ids,
            "attention_mask": mask,
            "use_cache": False,
            "output_attentions": False,
            "output_hidden_states": False,
            "return_dict": True,
        }

    def token_ids(self, value: Tensor | Sequence[int] | int | None, device: torch.device) -> Tensor:
        if value is None:
            raise AdapterError("prompt is missing a readout token id")
        result = value if isinstance(value, Tensor) else torch.as_tensor(value, dtype=torch.long)
        result = result.to(device=device, dtype=torch.long)
        if result.ndim == 0:
            result = result.reshape(1, 1)
        elif result.ndim == 1:
            # A length-B vector means one token per example.  A length-K
            # vector means a shared K-token set for a single example/batch.
            if result.numel() == self.batch_size:
                result = result.reshape(self.batch_size, 1)
            else:
                result = result.reshape(1, -1)
        elif result.ndim != 2:
            raise AdapterError("readout token ids must be scalar, [B], or [B,K]")
        if result.shape[0] not in (1, self.batch_size):
            raise AdapterError(
                f"readout token batch {result.shape[0]} does not match prompt batch {self.batch_size}"
            )
        if result.shape[0] == 1 and self.batch_size > 1:
            result = result.expand(self.batch_size, -1)
        return result


@dataclass(frozen=True)
class ArchitectureSpec:
    """Resolved decoder stack paths used for provenance and diagnostics."""

    layer_path: str
    n_layers: int
    final_norm_path: str
    output_head_path: str


@dataclass
class _Trajectory:
    prompt: TorchPrompt
    hidden_inputs: List[Tensor]
    hidden_outputs: List[Tensor]
    layer_kwargs: List[Dict[str, Any]]
    logits: Tensor
    spec: ArchitectureSpec


def _get_path(root: Any, path: str) -> Any:
    value = root
    for part in path.split("."):
        if not hasattr(value, part):
            raise AttributeError(path)
        value = getattr(value, part)
    return value


def _find_first_path(root: Any, paths: Iterable[str]) -> Tuple[str, Any]:
    for path in paths:
        try:
            value = _get_path(root, path)
        except AttributeError:
            continue
        if value is not None:
            return path, value
    raise AdapterError(
        "could not locate a decoder block stack; tried: " + ", ".join(paths)
    )


def _unwrap_base(model: nn.Module) -> nn.Module:
    """Return the model containing the decoder blocks.

    PEFT models expose ``get_base_model``.  Plain causal-LM classes expose a
    ``base_model_prefix`` and one of the common attributes below.
    """

    if hasattr(model, "get_base_model"):
        try:
            base = model.get_base_model()
            if isinstance(base, nn.Module):
                return base
        except Exception:
            pass
    prefix = getattr(model, "base_model_prefix", None)
    if prefix and hasattr(model, prefix):
        candidate = getattr(model, prefix)
        if isinstance(candidate, nn.Module):
            return candidate
    for path in ("model", "transformer", "gpt_neox", "decoder"):
        candidate = getattr(model, path, None)
        if isinstance(candidate, nn.Module):
            return candidate
    return model


def _resolve_spec(model: nn.Module) -> Tuple[nn.Module, ArchitectureSpec, nn.ModuleList]:
    base = _unwrap_base(model)
    layer_path, layers = _find_first_path(
        base,
        (
            "layers",
            "h",
            "model.layers",
            "transformer.h",
            "gpt_neox.layers",
            "decoder.layers",
        ),
    )
    if not isinstance(layers, (nn.ModuleList, list, tuple)) or len(layers) == 0:
        raise AdapterError(f"resolved {layer_path}, but it is not a non-empty layer list")
    if not all(isinstance(layer, nn.Module) for layer in layers):
        raise AdapterError(f"all entries in {layer_path} must be torch modules")

    norm_path, _ = _find_first_path(
        base,
        ("norm", "ln_f", "model.norm", "transformer.ln_f", "gpt_neox.final_layer_norm"),
    )
    head_path, _ = _find_first_path(model, ("lm_head", "embed_out", "output_projection"))
    spec = ArchitectureSpec(
        layer_path=layer_path,
        n_layers=len(layers),
        final_norm_path=norm_path,
        output_head_path=head_path,
    )
    return base, spec, nn.ModuleList(layers)


def _first_tensor(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
        return output[0]
    if hasattr(output, "last_hidden_state") and isinstance(output.last_hidden_state, Tensor):
        return output.last_hidden_state
    raise AdapterError(f"decoder block returned unsupported output type {type(output)!r}")


def _to_numpy(tensor: Tensor) -> np.ndarray:
    """Convert model states to NumPy while supporting CUDA bfloat16."""

    return tensor.detach().float().cpu().numpy()


def _filter_layer_kwargs(layer: nn.Module, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep only kwargs accepted by a decoder block.

    Transformers has renamed cache and rotary-position arguments across
    releases.  Filtering against the installed layer signature lets the
    adapter fail closed for required inputs while avoiding version-specific
    keyword errors for optional inputs.
    """

    try:
        signature = inspect.signature(layer.forward)
    except (TypeError, ValueError):
        return dict(kwargs)
    parameters = signature.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in parameters}


def _call_layer(layer: nn.Module, hidden: Tensor, kwargs: Mapping[str, Any]) -> Any:
    filtered = _filter_layer_kwargs(layer, kwargs)
    filtered.pop("hidden_states", None)
    try:
        return layer(hidden, **filtered)
    except TypeError as exc:
        # A layer with a version-specific optional argument should not make a
        # valid architecture unusable, but do not silently retry arbitrary
        # errors: only remove the named optional cache/position fields.
        optional = (
            "position_embeddings",
            "position_ids",
            "cache_position",
            "past_key_value",
            "past_key_values",
            "use_cache",
            "output_attentions",
        )
        retry = dict(filtered)
        changed = False
        for key in optional:
            if key in retry and key in str(exc):
                retry.pop(key)
                changed = True
        if not changed:
            raise
        return layer(hidden, **retry)


class HFResidualAdapter:
    """Concrete ``suture_metrics.ModelAdapter`` for a pair of HF decoders."""

    def __init__(
        self,
        host_model: nn.Module,
        donor_model: nn.Module,
        *,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
        host_adapter_name: Optional[str] = None,
        donor_adapter_name: Optional[str] = None,
    ) -> None:
        self.host_model = host_model
        self.donor_model = donor_model
        self.host_adapter_name = host_adapter_name
        self.donor_adapter_name = donor_adapter_name
        self.device = torch.device(device)
        self.host_model.to(self.device).eval()
        self.donor_model.to(self.device).eval()
        # Selection differentiates with respect to retained hidden states, not
        # model parameters.  Freezing parameters avoids storing useless
        # parameter-gradient edges while the hook still marks each state as a
        # differentiable input.
        seen_parameters = set()
        for module in (self.host_model, self.donor_model):
            for parameter in module.parameters():
                if id(parameter) not in seen_parameters:
                    parameter.requires_grad_(False)
                    seen_parameters.add(id(parameter))
        self.host_base, self.spec, self.host_layers = _resolve_spec(self.host_model)
        self.donor_base, donor_spec, self.donor_layers = _resolve_spec(self.donor_model)
        if self.spec.n_layers != donor_spec.n_layers:
            raise AdapterError(
                f"host/donor layer count mismatch: {self.spec.n_layers} vs {donor_spec.n_layers}"
            )
        if self.spec.layer_path != donor_spec.layer_path:
            raise AdapterError(
                f"host/donor layer paths differ: {self.spec.layer_path} vs {donor_spec.layer_path}"
            )
        if self.host_model is self.donor_model and (
            self.host_adapter_name is None or self.donor_adapter_name is None
        ):
            raise AdapterError(
                "a shared PEFT model requires both host_adapter_name and donor_adapter_name"
            )
        self._current: Optional[_Trajectory] = None
        self._selection_active = False
        self.merged_model_builds = 0

    @property
    def n_layers(self) -> int:
        return self.spec.n_layers

    @property
    def architecture(self) -> Dict[str, Any]:
        return {
            "layer_path": self.spec.layer_path,
            "n_layers": self.spec.n_layers,
            "final_norm_path": self.spec.final_norm_path,
            "output_head_path": self.spec.output_head_path,
            "host_class": type(self.host_model).__name__,
            "donor_class": type(self.donor_model).__name__,
            "host_adapter_name": self.host_adapter_name,
            "donor_adapter_name": self.donor_adapter_name,
        }

    @contextlib.contextmanager
    def _active_adapter(self, model: nn.Module, name: Optional[str]) -> Iterator[None]:
        """Temporarily select a PEFT adapter when the pair shares a base."""

        if name is None:
            yield
            return
        setter = getattr(model, "set_adapter", None)
        if setter is None:
            raise AdapterError(
                f"adapter name {name!r} was supplied, but {type(model).__name__} "
                "does not expose set_adapter()"
            )
        previous = getattr(model, "active_adapter", None)
        if previous is None:
            previous = getattr(model, "active_adapters", None)
        setter(name)
        try:
            yield
        finally:
            if previous is not None:
                try:
                    setter(previous)
                except (TypeError, ValueError):
                    # PEFT releases differ on whether a one-element active
                    # adapter is exposed as a string or a list.
                    if isinstance(previous, (list, tuple)) and len(previous) == 1:
                        setter(previous[0])
                    else:
                        raise

    def _coerce_prompt(self, prompt: TorchPrompt | Mapping[str, Any]) -> TorchPrompt:
        if isinstance(prompt, TorchPrompt):
            return prompt
        if isinstance(prompt, Mapping):
            return TorchPrompt(**dict(prompt))
        raise AdapterError(f"prompt must be TorchPrompt or mapping, got {type(prompt)!r}")

    def _capture(self, prompt: TorchPrompt | Mapping[str, Any]) -> _Trajectory:
        prompt = self._coerce_prompt(prompt)
        hidden_inputs: List[Optional[Tensor]] = [None] * self.n_layers
        hidden_outputs: List[Optional[Tensor]] = [None] * self.n_layers
        layer_kwargs: List[Optional[Dict[str, Any]]] = [None] * self.n_layers
        handles = []

        def make_pre(index: int):
            def pre(module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
                hidden = args[0] if args else kwargs.get("hidden_states")
                if not isinstance(hidden, Tensor):
                    raise AdapterError(f"layer {index} did not receive a tensor hidden state")
                if not hidden.requires_grad:
                    hidden = hidden.detach().requires_grad_(True)
                else:
                    hidden.retain_grad()
                hidden_inputs[index] = hidden
                layer_kwargs[index] = dict(kwargs)
                return (hidden,), kwargs
            return pre

        def make_post(index: int):
            def post(
                module: nn.Module,
                args: Tuple[Any, ...],
                kwargs: Dict[str, Any],
                output: Any,
            ):
                hidden = _first_tensor(output)
                if not hidden.requires_grad:
                    hidden = hidden.detach().requires_grad_(True)
                else:
                    hidden.retain_grad()
                hidden_outputs[index] = hidden
                return output
            return post

        for index, layer in enumerate(self.host_layers):
            try:
                handles.append(
                    layer.register_forward_pre_hook(make_pre(index), with_kwargs=True)
                )
                handles.append(
                    layer.register_forward_hook(make_post(index), with_kwargs=True)
                )
            except TypeError as exc:
                for handle in handles:
                    handle.remove()
                raise AdapterError(
                    "the installed PyTorch version lacks with_kwargs hooks; "
                    "the adapter cannot guarantee identical host/donor inputs"
                ) from exc

        try:
            with self._active_adapter(self.host_model, self.host_adapter_name):
                with torch.enable_grad():
                    outputs = self.host_model(**prompt.model_inputs(self.device))
            logits = getattr(outputs, "logits", None)
            if not isinstance(logits, Tensor):
                raise AdapterError("host model forward did not return .logits")
        finally:
            for handle in handles:
                handle.remove()

        if any(value is None for value in hidden_inputs + hidden_outputs + layer_kwargs):
            raise AdapterError("one or more decoder layers were not captured during host forward")
        trajectory = _Trajectory(
            prompt=prompt,
            hidden_inputs=[value for value in hidden_inputs if value is not None],
            hidden_outputs=[value for value in hidden_outputs if value is not None],
            layer_kwargs=[value for value in layer_kwargs if value is not None],
            logits=logits,
            spec=self.spec,
        )
        self._current = trajectory
        return trajectory

    def residual_states(self, prompt: TorchPrompt | Mapping[str, Any]) -> List[np.ndarray]:
        trajectory = self._capture(prompt)
        return [
            _to_numpy(trajectory.hidden_inputs[0]),
            *[
                _to_numpy(hidden)
                for hidden in trajectory.hidden_outputs
            ],
        ]

    def _require_current(self) -> _Trajectory:
        if self._current is None:
            raise AdapterError("no host trajectory is active; call residual_states first")
        return self._current

    def host_block(self, l: int, x: np.ndarray) -> np.ndarray:
        trajectory = self._require_current()
        if not 0 <= l < self.n_layers:
            raise AdapterError(f"layer index out of range: {l}")
        expected = _to_numpy(trajectory.hidden_inputs[l])
        actual = np.asarray(x)
        if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=1e-4, atol=1e-5):
            raise AdapterError("host_block received a state from a different host trajectory")
        return _to_numpy(trajectory.hidden_outputs[l]) - _to_numpy(trajectory.hidden_inputs[l])

    def donor_block(self, l: int, x: np.ndarray) -> np.ndarray:
        trajectory = self._require_current()
        if not 0 <= l < self.n_layers:
            raise AdapterError(f"layer index out of range: {l}")
        expected = _to_numpy(trajectory.hidden_inputs[l])
        actual = np.asarray(x)
        if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=1e-4, atol=1e-5):
            raise AdapterError("donor_block received a state from a different host trajectory")
        hidden = torch.as_tensor(
            actual,
            device=self.device,
            dtype=trajectory.hidden_inputs[l].dtype,
        )
        with self._active_adapter(self.donor_model, self.donor_adapter_name):
            with torch.no_grad():
                output = _call_layer(self.donor_layers[l], hidden, trajectory.layer_kwargs[l])
        return _to_numpy(_first_tensor(output) - hidden)

    def _readout(self, logits: Tensor, prompt: TorchPrompt, readout: str) -> Tensor:
        last = logits[:, -1, :]
        # Keep float64/float32 precision for numerical adjoint checks.  Half
        # precision logits are promoted only when necessary for stable
        # log-softmax evaluation.
        readout_logits = (
            last.float()
            if last.dtype in (torch.float16, torch.bfloat16)
            else last
        )
        log_probs = torch.log_softmax(readout_logits, dim=-1)
        if readout in ("utility", "util"):
            ids = prompt.token_ids(prompt.utility_token_ids, self.device)
            selected = log_probs.gather(1, ids)
            return torch.logsumexp(selected, dim=1).mean()
        if readout == "risk":
            target = prompt.token_ids(prompt.risk_target_token_ids, self.device)
            donor = prompt.token_ids(prompt.risk_donor_token_ids, self.device)
            target_lp = torch.logsumexp(log_probs.gather(1, target), dim=1)
            donor_lp = torch.logsumexp(log_probs.gather(1, donor), dim=1)
            return (target_lp - donor_lp).mean()
        raise AdapterError(f"unknown readout {readout!r}; expected utility or risk")

    def adjoints(
        self,
        prompt: TorchPrompt | Mapping[str, Any],
        readout: str,
    ) -> List[np.ndarray]:
        trajectory = self._require_current()
        prompt = self._coerce_prompt(prompt)
        if prompt is not trajectory.prompt and prompt.input_ids.shape != trajectory.prompt.input_ids.shape:
            raise AdapterError("adjoints requested for a prompt different from the active trajectory")
        phi = self._readout(trajectory.logits, trajectory.prompt, readout)
        gradients = torch.autograd.grad(
            phi,
            tuple(trajectory.hidden_outputs),
            retain_graph=False,
            allow_unused=False,
        )
        self._current = None
        return [_to_numpy(gradient) for gradient in gradients]

    def adjoint_check(
        self,
        prompt: TorchPrompt | Mapping[str, Any],
        *,
        layer: int,
        readout: str,
        delta_scale: float = 1e-3,
        seed: int = 0,
    ) -> Dict[str, Any]:
        """Finite-difference check of ``d phi / d x_{layer+1}``.

        The tail is replayed with the exact per-layer kwargs captured from the
        original host forward.  This checks the quantity used by SUTURE rather
        than a gradient through a separately reconstructed trajectory.
        """

        trajectory = self._capture(prompt)
        if not 0 <= layer < self.n_layers:
            raise AdapterError(f"layer index out of range: {layer}")
        phi = self._readout(trajectory.logits, trajectory.prompt, readout)
        gradient = torch.autograd.grad(
            phi,
            trajectory.hidden_outputs[layer],
            retain_graph=True,
            allow_unused=False,
        )[0]
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        delta = torch.randn(
            trajectory.hidden_outputs[layer].shape,
            generator=generator,
            device=self.device,
            dtype=trajectory.hidden_outputs[layer].dtype,
        )
        delta = delta / delta.norm().clamp_min(torch.finfo(delta.dtype).eps) * delta_scale
        direction = delta / delta.norm().clamp_min(torch.finfo(delta.dtype).eps)
        step_scales = np.asarray(
            [0.5, 1.0, 2.0, 4.0],
            dtype=float,
        ) * float(delta_scale)
        finite_slopes: List[float] = []

        with torch.no_grad():
            for step in step_scales:
                perturbation = direction * float(step)
                plus = self._tail_readout(
                    trajectory,
                    layer + 1,
                    trajectory.hidden_outputs[layer] + perturbation,
                    readout,
                )
                minus = self._tail_readout(
                    trajectory,
                    layer + 1,
                    trajectory.hidden_outputs[layer] - perturbation,
                    readout,
                )
                finite_slopes.append(float((plus - minus) / (2.0 * step)))
        # Central differences have an O(h^2) error.  Extrapolating the
        # directional slopes to h=0 removes that leading truncation term and
        # makes the check meaningful when the selected readout derivative is
        # small relative to the model's hidden-state scale.
        coefficients = np.polynomial.polynomial.polyfit(
            step_scales * step_scales,
            np.asarray(finite_slopes, dtype=float),
            deg=2,
        )
        finite_slope = float(coefficients[0])
        finite_difference = finite_slope * float(delta_scale)
        directional = (gradient * delta).sum()
        denominator = max(abs(float(finite_difference)), abs(float(directional)), 1e-12)
        self._current = None
        return {
            "layer": float(layer),
            "finite_difference": float(finite_difference),
            "directional_derivative": float(directional),
            "relative_error": abs(float(finite_difference - directional)) / denominator,
            "delta_norm": float(delta.norm()),
            "finite_difference_step_scales": step_scales.tolist(),
            "finite_difference_extrapolation": "quadratic_in_step_squared",
        }

    def jacobian_gap_proxy(
        self,
        prompt: TorchPrompt | Mapping[str, Any],
        *,
        layer: int,
        delta_scale: float = 1e-3,
        seed: int = 0,
    ) -> float:
        """Estimate ``||J_d-J_h||`` along a few finite-difference directions."""

        trajectory = self._capture(prompt)
        if not 0 <= layer < self.n_layers:
            raise AdapterError(f"layer index out of range: {layer}")
        hidden = trajectory.hidden_inputs[layer]
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        delta = torch.randn(
            hidden.shape,
            generator=generator,
            device=self.device,
            dtype=hidden.dtype,
        )
        delta = delta / delta.norm().clamp_min(torch.finfo(delta.dtype).eps) * delta_scale
        kwargs = trajectory.layer_kwargs[layer]
        with torch.no_grad():
            with self._active_adapter(self.host_model, self.host_adapter_name):
                host_plus = _first_tensor(_call_layer(self.host_layers[layer], hidden + delta, kwargs))
                host_minus = _first_tensor(_call_layer(self.host_layers[layer], hidden - delta, kwargs))
            with self._active_adapter(self.donor_model, self.donor_adapter_name):
                donor_plus = _first_tensor(_call_layer(self.donor_layers[layer], hidden + delta, kwargs))
                donor_minus = _first_tensor(_call_layer(self.donor_layers[layer], hidden - delta, kwargs))
        host_direction = (host_plus - host_minus) / (2.0 * delta.norm())
        donor_direction = (donor_plus - donor_minus) / (2.0 * delta.norm())
        self._current = None
        return float((donor_direction - host_direction).norm().detach().cpu())

    def _tail_readout(
        self,
        trajectory: _Trajectory,
        start: int,
        hidden: Tensor,
        readout: str,
    ) -> Tensor:
        value = hidden
        with self._active_adapter(self.host_model, self.host_adapter_name):
            with torch.no_grad():
                for index in range(start, self.n_layers):
                    value = _first_tensor(
                        _call_layer(self.host_layers[index], value, trajectory.layer_kwargs[index])
                    )
                norm = _get_path(self.host_base, self.spec.final_norm_path)
                head = _get_path(self.host_model, self.spec.output_head_path)
                logits = head(norm(value))
        return self._readout(logits, trajectory.prompt, readout)

    @contextlib.contextmanager
    def selection_phase(self) -> Iterator[None]:
        """Reject merged-model construction until selection has finished."""

        if self._selection_active:
            raise AdapterError("selection_phase cannot be nested")
        self._selection_active = True
        try:
            yield
        finally:
            self._selection_active = False

    @contextlib.contextmanager
    def _route(self, graft: Sequence[int]) -> Iterator[None]:
        selected = set(int(index) for index in graft)
        if any(index < 0 or index >= self.n_layers for index in selected):
            raise AdapterError(f"graft contains an out-of-range layer: {sorted(selected)}")
        handles = []
        for index in sorted(selected):
            host_layer = self.host_layers[index]
            donor_layer = self.donor_layers[index]

            try:
                if host_layer is donor_layer and self.host_model is self.donor_model:
                    # A shared PEFT model contains one physical layer with
                    # multiple adapter branches.  Calling that layer from a
                    # forward hook re-enters the hook recursively.  Switch
                    # the active branch around the layer's own forward call
                    # instead, then restore the host branch for the next
                    # layer.
                    setter = getattr(self.host_model, "set_adapter", None)
                    if setter is None or self.donor_adapter_name is None:
                        raise AdapterError(
                            "shared host/donor layers require a PEFT set_adapter() "
                            "and a donor adapter name"
                        )

                    def use_donor(
                        module: nn.Module,
                        args: Tuple[Any, ...],
                        kwargs: Dict[str, Any],
                    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
                        setter(self.donor_adapter_name)
                        return args, kwargs

                    def restore_host(
                        module: nn.Module,
                        args: Tuple[Any, ...],
                        kwargs: Dict[str, Any],
                        output: Any,
                    ) -> Any:
                        setter(self.host_adapter_name)
                        return output

                    handles.append(
                        host_layer.register_forward_pre_hook(use_donor, with_kwargs=True)
                    )
                    handles.append(
                        host_layer.register_forward_hook(restore_host, with_kwargs=True)
                    )
                else:
                    def replace(
                        module: nn.Module,
                        args: Tuple[Any, ...],
                        kwargs: Dict[str, Any],
                        output: Any,
                        donor: nn.Module = donor_layer,
                    ) -> Any:
                        hidden = args[0] if args else kwargs.get("hidden_states")
                        if not isinstance(hidden, Tensor):
                            raise AdapterError("graft hook did not receive a hidden-state tensor")
                        with self._active_adapter(self.donor_model, self.donor_adapter_name):
                            return _call_layer(donor, hidden, kwargs)

                    handles.append(
                        host_layer.register_forward_hook(replace, with_kwargs=True)
                    )
            except TypeError as exc:
                raise AdapterError("PyTorch with_kwargs hooks are required for graft routing") from exc
        try:
            yield
        finally:
            if self.host_model is self.donor_model and self.host_adapter_name is not None:
                setter = getattr(self.host_model, "set_adapter", None)
                if setter is not None:
                    setter(self.host_adapter_name)
            for handle in handles:
                handle.remove()

    def build_grafted_model(self, graft: Sequence[int]) -> "RoutedGraftModel":
        if self._selection_active:
            raise AdapterError(
                "merged-model construction was requested during selection; "
                "selection must use scores only"
            )
        normalized = tuple(sorted(set(int(index) for index in graft)))
        if any(index < 0 or index >= self.n_layers for index in normalized):
            raise AdapterError(f"graft contains an out-of-range layer: {normalized}")
        self.merged_model_builds += 1
        return RoutedGraftModel(self, normalized)

    def score_selection(
        self,
        utility_probe: Sequence[TorchPrompt],
        risk_probe: Sequence[TorchPrompt],
    ):
        """Convenience wrapper that enforces the no-build selection boundary."""

        try:
            from paper_iclr.suture_metrics import suture_scores
        except ModuleNotFoundError:
            from suture_metrics import suture_scores

        with self.selection_phase():
            return suture_scores(self, utility_probe, risk_probe)


class RoutedGraftModel:
    """Zero-copy merged-model evaluation route."""

    def __init__(self, adapter: HFResidualAdapter, graft: Tuple[int, ...]) -> None:
        super().__init__()
        self.adapter = adapter
        self.graft = graft

    def forward(self, **inputs: Tensor) -> Any:
        with self.adapter._active_adapter(self.adapter.host_model, self.adapter.host_adapter_name):
            with self.adapter._route(self.graft):
                return self.adapter.host_model(**inputs)

    def readout(self, prompt: TorchPrompt, readout: str) -> float:
        logits = self.logits(prompt)
        with torch.no_grad():
            value = self.adapter._readout(logits, prompt, readout)
        return float(value.detach().cpu())

    def logits(self, prompt: TorchPrompt) -> Tensor:
        with torch.no_grad():
            outputs = self.forward(**prompt.model_inputs(self.adapter.device))
            logits = getattr(outputs, "logits", None)
            if not isinstance(logits, Tensor):
                raise AdapterError("routed model forward did not return .logits")
        return logits

    def next_token_accuracy(self, prompt: TorchPrompt) -> float:
        logits = self.logits(prompt)[:, -1, :]
        ids = prompt.token_ids(prompt.utility_token_ids, self.adapter.device)
        if ids.shape[1] != 1:
            raise AdapterError("next_token_accuracy requires exactly one utility token per item")
        return float((logits.argmax(dim=-1) == ids[:, 0]).float().mean().cpu())

    def metadata(self) -> Dict[str, Any]:
        return {
            "graft": list(self.graft),
            "n_layers": self.adapter.n_layers,
            "routing": "host/donor zero-copy forward hooks",
            "merged_model_count": 1,
        }


def save_adapter_metadata(adapter: HFResidualAdapter, path: str, **extra: Any) -> None:
    """Write architecture and build-accounting metadata beside a run."""

    payload = {
        "architecture": adapter.architecture,
        "merged_model_builds": adapter.merged_model_builds,
        **extra,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


__all__ = [
    "AdapterError",
    "ArchitectureSpec",
    "HFResidualAdapter",
    "RoutedGraftModel",
    "TorchPrompt",
    "save_adapter_metadata",
]
