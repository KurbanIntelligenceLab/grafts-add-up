"""Run the frozen local powered-LLM reviewer comparison.

The comparison uses the valid Spanish Qwen2.5-Coder-1.5B pair because the
separate Qwen3 follow-up stopped at the fake-donor P1 gate.  Every method sees
the same fixed candidate windows and is compared with the archived canonical
free-generation outcomes; no new test-set evaluation or post-hoc tuning is
performed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from suture.suture_metrics import GraftScores, spearman
from suture.suture_torch import (
    HFResidualAdapter,
    _call_layer,
    _first_tensor,
    _get_path,
)
from suture.tier_a_config import refuse_frozen_write, repo_root, sha256_file
from suture.tier_a_gate import _load_records, _risk_prompts, _utility_prompts


CONTRACT_PATH = Path("configs/reviewer_llm_comparison_v1.json")
DEFAULT_OUTPUT = Path("results/reviewer_followup/llm_comparison")
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MODEL_REVISION = "2e1fd397ee46e1388853d2af2c993145b0f1098a"


def _default_snapshot() -> Path:
    configured = os.environ.get("SUTURE_QWEN25_MODEL_PATH")
    if configured:
        return Path(configured).expanduser()
    hf_home = Path(
        os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    ).expanduser()
    return (
        hf_home
        / "hub"
        / f"models--{MODEL_ID.replace('/', '--')}"
        / "snapshots"
        / MODEL_REVISION
    )


def _adapter_path(root: Path, name: str) -> Path:
    direct = root / "adapter_config.json"
    nested = root / name / "adapter_config.json"
    if direct.is_file():
        return root
    if nested.is_file():
        return nested.parent
    raise RuntimeError(f"adapter_config.json not found below {root}")


def _load_pair(
    *,
    host_dir: Path,
    donor_dir: Path,
    device: str,
    snapshot: Path,
) -> Tuple[Any, Any]:
    target = torch.device(device)
    dtype = torch.bfloat16 if target.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    host_adapter = _adapter_path(host_dir.resolve(), "host")
    donor_adapter = _adapter_path(donor_dir.resolve(), "donor")
    for label, path in (("host", host_adapter), ("donor", donor_adapter)):
        config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
        if config.get("base_model_name_or_path") != MODEL_ID:
            raise RuntimeError(f"{label} adapter is not bound to {MODEL_ID}")
    base = AutoModelForCausalLM.from_pretrained(
        str(SNAPSHOT),
        local_files_only=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(target).eval()
    base.config.use_cache = False
    shared = PeftModel.from_pretrained(
        base,
        str(host_adapter),
        adapter_name="host",
        is_trainable=False,
    )
    shared.load_adapter(str(donor_adapter), adapter_name="donor", is_trainable=False)
    shared.set_adapter("host")
    return shared, tokenizer


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr, y_arr = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return 0.0
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _layer_terms(
    adapter: HFResidualAdapter,
    prompts: Sequence[Any],
    *,
    readout: str,
) -> Dict[str, np.ndarray]:
    utility_scores: List[np.ndarray] = []
    injection_norms: List[np.ndarray] = []
    adjoint_norms: List[np.ndarray] = []
    for prompt in prompts:
        states = adapter.residual_states(prompt)
        injections = []
        for layer in range(adapter.n_layers):
            donor_update = adapter.donor_block(layer, states[layer])
            host_update = adapter.host_block(layer, states[layer])
            injections.append(donor_update - host_update)
        adjoints = adapter.adjoints(prompt, readout)
        utility_scores.append(
            np.asarray(
                [
                    float(np.vdot(adjoints[layer].ravel(), injections[layer].ravel()))
                    for layer in range(adapter.n_layers)
                ]
            )
        )
        injection_norms.append(
            np.asarray([float(np.linalg.norm(value)) for value in injections])
        )
        adjoint_norms.append(
            np.asarray([float(np.linalg.norm(value)) for value in adjoints])
        )
    return {
        "scores": np.mean(utility_scores, axis=0),
        "injection_norm": np.mean(injection_norms, axis=0),
        "adjoint_norm": np.mean(adjoint_norms, axis=0),
    }


def _tail_value(
    adapter: HFResidualAdapter,
    trajectory: Any,
    start: int,
    hidden: torch.Tensor,
    readout: str,
) -> torch.Tensor:
    value = hidden
    with adapter._active_adapter(adapter.host_model, adapter.host_adapter_name):
        for index in range(start, adapter.n_layers):
            value = _first_tensor(
                _call_layer(
                    adapter.host_layers[index],
                    value,
                    trajectory.layer_kwargs[index],
                )
            )
        norm = _get_path(adapter.host_base, adapter.spec.final_norm_path)
        head = _get_path(adapter.host_model, adapter.spec.output_head_path)
        logits = head(norm(value))
    return adapter._readout(logits, trajectory.prompt, readout)


def _path_integrated_scores(
    adapter: HFResidualAdapter,
    prompts: Sequence[Any],
    *,
    readout: str,
    nodes: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> np.ndarray:
    per_prompt: List[np.ndarray] = []
    for prompt in prompts:
        trajectory = adapter._capture(prompt)
        layer_values: List[float] = []
        for layer in range(adapter.n_layers):
            hidden_in = trajectory.hidden_inputs[layer]
            hidden_out = trajectory.hidden_outputs[layer]
            with adapter._active_adapter(adapter.donor_model, adapter.donor_adapter_name):
                with torch.no_grad():
                    donor_out = _first_tensor(
                        _call_layer(
                            adapter.donor_layers[layer],
                            hidden_in,
                            trajectory.layer_kwargs[layer],
                        )
                    )
            delta = (donor_out - hidden_in) - (hidden_out - hidden_in)
            base = hidden_out.detach()
            directional: List[float] = []
            for alpha in nodes:
                state = (base + float(alpha) * delta.detach()).detach()
                state.requires_grad_(True)
                value = _tail_value(adapter, trajectory, layer + 1, state, readout)
                gradient = torch.autograd.grad(
                    value,
                    state,
                    retain_graph=False,
                    allow_unused=False,
                )[0]
                directional.append(float((gradient * delta).sum().detach().cpu()))
            layer_values.append(float(np.trapezoid(directional, np.asarray(nodes))))
        adapter._current = None
        per_prompt.append(np.asarray(layer_values, dtype=float))
    return np.mean(per_prompt, axis=0)


def _parameter_movement(adapter: HFResidualAdapter) -> np.ndarray:
    movement = np.zeros(adapter.n_layers, dtype=float)
    found = 0
    parameters = dict(adapter.host_model.named_parameters())
    for donor_name, donor_parameter in adapter.donor_model.named_parameters():
        if ".donor." not in donor_name:
            continue
        host_name = donor_name.replace(".donor.", ".host.")
        host_parameter = parameters.get(host_name)
        if host_parameter is None:
            continue
        match = re.search(r"(?:layers|h)\.(\d+)\.", donor_name)
        if match is None:
            continue
        layer = int(match.group(1))
        if layer >= adapter.n_layers:
            continue
        movement[layer] += float(
            torch.linalg.vector_norm(
                donor_parameter.detach().float() - host_parameter.detach().float()
            ).cpu()
        )
        found += 1
    if found == 0:
        raise RuntimeError("could not locate paired host/donor LoRA parameters")
    return movement


def _load_ground_truth(path: Path, windows: Sequence[Tuple[int, int]]) -> List[Dict[str, Any]]:
    rows: Dict[Tuple[int, int], Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[(int(row["start"]), int(row["end"]))] = row
    missing = [window for window in windows if window not in rows]
    if missing:
        raise RuntimeError(f"archived ground truth is missing candidate windows: {missing}")
    return [rows[window] for window in windows]


def _method_summary(
    scores: Mapping[str, Sequence[float]],
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    free = np.asarray([float(row["measured_exact_match"]) for row in rows])
    teacher = np.asarray(
        [float(row["measured_teacher_forced_sequence_logprob"]) for row in rows]
    )
    summary: Dict[str, Any] = {}
    for name, values in scores.items():
        values_arr = np.asarray(values, dtype=float)
        best = int(np.argmax(values_arr))
        summary[name] = {
            "spearman_free_generation_exact_match": float(spearman(values_arr, free)),
            "pearson_free_generation_exact_match": _pearson(values_arr, free),
            "spearman_teacher_forced_sequence_logprob": float(
                spearman(values_arr, teacher)
            ),
            "selected_window": [
                int(rows[best]["start"]),
                int(rows[best]["end"]),
            ],
            "selected_free_generation_exact_match": float(free[best]),
            "selected_teacher_forced_sequence_logprob": float(teacher[best]),
        }
    return summary


def _write_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"immutable comparison artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(
    *,
    output_dir: Path,
    device: str,
    prompt_limit: int = 8,
    max_length: int = 128,
    model_path: Path | None = None,
) -> Dict[str, Any]:
    root = repo_root()
    contract_path = root / CONTRACT_PATH
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    windows = [tuple(window) for window in contract["inputs"]["candidate_windows"]]
    data_dir = root / contract["inputs"]["data_dir"]
    rows = _load_ground_truth(
        root / contract["inputs"]["free_generation_ground_truth"],
        windows,
    )
    snapshot = model_path or _default_snapshot()
    shared, tokenizer = _load_pair(
        host_dir=root / contract["inputs"]["host_adapter"],
        donor_dir=root / contract["inputs"]["donor_adapter"],
        device=device,
        snapshot=snapshot,
    )
    utility_records = _load_records(data_dir / "P_util.jsonl")[:prompt_limit]
    risk_records = _load_records(data_dir / "P_risk.jsonl")[:prompt_limit]
    utility_prompts = _utility_prompts(
        tokenizer,
        utility_records,
        batch_size=4,
        max_length=max_length,
    )
    risk_prompts = _risk_prompts(
        tokenizer,
        risk_records,
        language="es",
        batch_size=4,
        max_length=max_length,
    )
    adapter = HFResidualAdapter(
        shared,
        shared,
        device=device,
        host_adapter_name="host",
        donor_adapter_name="donor",
    )
    scores: GraftScores = adapter.score_selection(utility_prompts, risk_prompts)
    utility_terms = _layer_terms(adapter, utility_prompts, readout="utility")
    risk_terms = _layer_terms(adapter, risk_prompts, readout="risk")
    movement = _parameter_movement(adapter)
    path_scores = _path_integrated_scores(
        adapter,
        utility_prompts,
        readout="utility",
    )

    method_scores = {
        "suture": np.asarray(
            [scores.predict(tuple(range(start, end + 1)))[0] for start, end in windows]
        ),
        "injection_only": np.asarray(
            [
                -float(utility_terms["injection_norm"][start : end + 1].sum())
                for start, end in windows
            ]
        ),
        "adjoint_only": np.asarray(
            [
                float(utility_terms["adjoint_norm"][start : end + 1].sum())
                for start, end in windows
            ]
        ),
        "parameter_movement": np.asarray(
            [-float(movement[start : end + 1].sum()) for start, end in windows]
        ),
        "risk_readout": np.asarray(
            [
                float(risk_terms["scores"][start : end + 1].sum())
                for start, end in windows
            ]
        ),
        "normalized_suture": np.asarray(
            [
                float(
                    (
                        scores.utility[start : end + 1]
                        / (utility_terms["injection_norm"][start : end + 1] + 1e-12)
                    ).sum()
                )
                for start, end in windows
            ]
        ),
        "path_integrated_state": np.asarray(
            [float(path_scores[start : end + 1].sum()) for start, end in windows]
        ),
    }
    summary = _method_summary(method_scores, rows)
    candidate_rows = []
    for index, row in enumerate(rows):
        candidate_rows.append(
            {
                "start": int(row["start"]),
                "end": int(row["end"]),
                "free_generation_exact_match": float(row["measured_exact_match"]),
                "teacher_forced_sequence_logprob": float(
                    row["measured_teacher_forced_sequence_logprob"]
                ),
                "method_scores": {
                    name: float(values[index])
                    for name, values in method_scores.items()
                },
            }
        )
    payload = {
        "contract_version": contract["contract_version"],
        "contract_sha256": sha256_file(contract_path),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "device": device,
        "prompt_limit": prompt_limit,
        "max_length": max_length,
        "n_layers": adapter.n_layers,
        "n_candidate_windows": len(windows),
        "candidate_windows": [list(window) for window in windows],
        "methods": list(method_scores),
        "summary": summary,
        "candidate_rows": candidate_rows,
        "path_integral": {
            "nodes": [0.0, 0.25, 0.5, 0.75, 1.0],
            "quadrature": "trapezoidal",
            "interpretation": "state-space local path diagnostic, not a parameter-space certified bound",
        },
        "qwen3_followup": {
            "status": "STOP_P1_FAKE",
            "artifact": "results/reviewer_followup/qwen3_1_7b/repair_01/plumbing_diagnostic.json",
            "e1_authorized": False,
        },
        "canonical_artifacts_modified": False,
        "ground_truth_source": "archived canonical v1 free-generation sweep; no new generation in this comparison",
    }
    output_dir = output_dir.resolve()
    refuse_frozen_write(output_dir)
    _write_immutable(output_dir / "comparison.json", payload)
    _write_immutable(
        output_dir / "run_manifest.json",
        {
            "contract": str(CONTRACT_PATH),
            "contract_sha256": sha256_file(contract_path),
            "command": "python -m suture.reviewer_llm_comparison",
            "device": device,
            "prompt_limit": prompt_limit,
            "max_length": max_length,
            "canonical_artifacts_modified": False,
        },
    )
    del adapter, shared, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prompt-limit", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Local Qwen2.5-Coder snapshot; defaults to SUTURE_QWEN25_MODEL_PATH or HF_HOME.",
    )
    args = parser.parse_args()
    report = run(
        output_dir=args.output_dir,
        device=args.device,
        prompt_limit=args.prompt_limit,
        max_length=args.max_length,
        model_path=args.model_path,
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
