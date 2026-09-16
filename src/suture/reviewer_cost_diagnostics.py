"""Benchmark local streamed scoring costs and directional smoothness proxies.

The measurements are engineering diagnostics for the reviewer response.  They
do not estimate a worst-case Lipschitz constant and do not authorize an E1
ranking claim.  Qwen2.5-Coder-1.5B is measured only as the frozen v1 local
comparison; Qwen3-1.7B is measured from the frozen v2 pair.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from suture.suture_torch import HFResidualAdapter
from suture.tier_a_config import refuse_frozen_write, repo_root, sha256_file
from suture.tier_a_gate import _load_records, _risk_prompts, _utility_prompts
from suture.tier_a_preflight import pretrained_load_args


DEFAULT_OUTPUT = Path("results/reviewer_followup/cost")
QWEN25_MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
QWEN25_MODEL_REVISION = "2e1fd397ee46e1388853d2af2c993145b0f1098a"


def _default_qwen25_snapshot() -> Path:
    configured = os.environ.get("SUTURE_QWEN25_MODEL_PATH")
    if configured:
        return Path(configured).expanduser()
    hf_home = Path(
        os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    ).expanduser()
    return (
        hf_home
        / "hub"
        / f"models--{QWEN25_MODEL_ID.replace('/', '--')}"
        / "snapshots"
        / QWEN25_MODEL_REVISION
    )


def _adapter_path(root: Path, name: str) -> Path:
    direct = root / "adapter_config.json"
    nested = root / name / "adapter_config.json"
    if direct.is_file():
        return root
    if nested.is_file():
        return nested.parent
    raise RuntimeError(f"adapter_config.json not found below {root}")


def _sync(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def _load_pair(
    *,
    snapshot: Path | None,
    host_dir: Path,
    donor_dir: Path,
    device: str,
    model_id: str,
    model_revision: str | None,
) -> Tuple[Any, Any]:
    target = torch.device(device)
    dtype = torch.bfloat16 if target.type == "cuda" else torch.float32
    if snapshot is None:
        source = pretrained_load_args(repo_root())
        model_path = source["pretrained_model_name_or_path"]
    else:
        model_path = str(snapshot.resolve())
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    host_adapter = _adapter_path(host_dir.resolve(), "host")
    donor_adapter = _adapter_path(donor_dir.resolve(), "donor")
    host_cfg = json.loads((host_adapter / "adapter_config.json").read_text(encoding="utf-8"))
    donor_cfg = json.loads((donor_adapter / "adapter_config.json").read_text(encoding="utf-8"))
    for label, cfg in (("host", host_cfg), ("donor", donor_cfg)):
        if cfg.get("base_model_name_or_path") != model_id:
            raise RuntimeError(
                f"{label} adapter is bound to {cfg.get('base_model_name_or_path')!r}, "
                f"expected {model_id!r}"
            )
        if model_revision is not None and cfg.get("revision") not in {None, model_revision}:
            raise RuntimeError(f"{label} adapter revision mismatch")
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
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


def _timed(
    fn: Callable[[], Any],
    *,
    device: str,
    repeats: int = 3,
) -> Dict[str, Any]:
    """Warm up, then time synchronized repetitions and report peak allocation."""

    fn()
    _sync(device)
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(repeats):
        _sync(device)
        started = time.perf_counter()
        fn()
        _sync(device)
        samples.append(time.perf_counter() - started)
    peak = None
    if torch.device(device).type == "cuda":
        peak = float(torch.cuda.max_memory_allocated() / (1024**3))
    return {
        "repeats": repeats,
        "warmup": True,
        "seconds": [float(value) for value in samples],
        "median_seconds": float(statistics.median(samples)),
        "p95_seconds": float(np.percentile(samples, 95)),
        "peak_allocated_gib": peak,
    }


def _measure_model(
    *,
    label: str,
    model_id: str,
    model_revision: str | None,
    snapshot: Path | None,
    host_dir: Path,
    donor_dir: Path,
    data_dir: Path,
    device: str,
    prompt_limit: int,
    max_length: int,
) -> Dict[str, Any]:
    shared, tokenizer = _load_pair(
        snapshot=snapshot,
        host_dir=host_dir,
        donor_dir=donor_dir,
        device=device,
        model_id=model_id,
        model_revision=model_revision,
    )
    utility_records = _load_records(data_dir / "P_util.jsonl")[:prompt_limit]
    risk_records = _load_records(data_dir / "P_risk.jsonl")[:prompt_limit]
    utility_prompts = _utility_prompts(
        tokenizer,
        utility_records,
        batch_size=prompt_limit,
        max_length=max_length,
    )
    risk_prompts = _risk_prompts(
        tokenizer,
        risk_records,
        language="es",
        batch_size=prompt_limit,
        max_length=max_length,
    )
    if not utility_prompts or not risk_prompts:
        raise RuntimeError(f"{label} produced no benchmark prompts")
    prompt = utility_prompts[0]
    adapter = HFResidualAdapter(
        shared,
        shared,
        device=device,
        host_adapter_name="host",
        donor_adapter_name="donor",
    )
    candidate = adapter.build_grafted_model((0, 1))

    def host_forward() -> Any:
        with adapter._active_adapter(adapter.host_model, adapter.host_adapter_name):
            with torch.inference_mode():
                return adapter.host_model(**prompt.model_inputs(adapter.device))

    def donor_stream() -> None:
        states = adapter.residual_states(prompt)
        for layer in range(adapter.n_layers):
            adapter.donor_block(layer, states[layer])

    def candidate_forward() -> Any:
        return candidate.forward(**prompt.model_inputs(adapter.device))

    def backward() -> Any:
        adapter.residual_states(prompt)
        return adapter.adjoints(prompt, "utility")

    def score_acquisition() -> Any:
        return adapter.score_selection(utility_prompts, risk_prompts)

    timings = {
        "host_forward": _timed(host_forward, device=device),
        "streamed_one_donor_all_layers": _timed(donor_stream, device=device),
        "backward_adjoint": _timed(backward, device=device),
        "candidate_forward_two_layers": _timed(candidate_forward, device=device),
        "score_acquisition": _timed(score_acquisition, device=device),
    }
    scores = score_acquisition()
    smoothness = {}
    mid = adapter.n_layers // 2
    smoothness["directional_jacobian_gap_proxy"] = adapter.jacobian_gap_proxy(
        prompt,
        layer=mid,
        delta_scale=1e-3,
        seed=17,
    )
    smoothness["adjoint_relative_error_proxy"] = adapter.adjoint_check(
        prompt,
        layer=mid,
        readout="utility",
        delta_scale=1e-3,
        seed=17,
    )
    injection = np.asarray(scores.injection_norm, dtype=float)
    result = {
        "label": label,
        "model_id": model_id,
        "model_revision": model_revision,
        "n_layers": adapter.n_layers,
        "prompt_limit": prompt_limit,
        "max_length": max_length,
        "dtype": str(next(shared.parameters()).dtype),
        "n_parameters": int(sum(parameter.numel() for parameter in shared.parameters())),
        "timings": timings,
        "memory_note": "peak_allocated_gib is per timed stage after warmup; CUDA allocator state is not a model-independent capacity bound",
        "smoothness": {
            "interpretation": "directional local proxies only; not global or worst-case Lipschitz constants",
            **smoothness,
        },
        "score_diagnostics": {
            "mean_injection_norm": float(injection.mean()),
            "max_injection_norm": float(injection.max()),
            "one_donor_only": True,
            "multiple_donor_models_available": False,
        },
    }
    del candidate, adapter, shared, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run(
    *,
    output_dir: Path,
    device: str,
    prompt_limit: int = 2,
    max_length: int = 128,
    qwen25_model_path: Path | None = None,
) -> Dict[str, Any]:
    root = repo_root()
    output_dir = output_dir.resolve()
    refuse_frozen_write(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    qwen3_root = root / "results/v2/qwen3_1_7b"
    qwen25_root = root / "results/tier_a/es"
    qwen3 = _measure_model(
        label="Qwen3-1.7B-v2",
        model_id="Qwen/Qwen3-1.7B",
        model_revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        snapshot=None,
        host_dir=qwen3_root / "es/adapters/host/checkpoint-1024/host",
        donor_dir=qwen3_root / "shared/donor/checkpoint-0512/donor",
        data_dir=qwen3_root / "es/data",
        device=device,
        prompt_limit=prompt_limit,
        max_length=max_length,
    )
    qwen25_snapshot = qwen25_model_path or _default_qwen25_snapshot()
    qwen25 = _measure_model(
        label="Qwen2.5-Coder-1.5B-v1",
        model_id=QWEN25_MODEL_ID,
        model_revision=QWEN25_MODEL_REVISION,
        snapshot=qwen25_snapshot,
        host_dir=qwen25_root / "adapters/host/host",
        donor_dir=qwen25_root / "adapters/donor/donor",
        data_dir=qwen25_root,
        device=device,
        prompt_limit=prompt_limit,
        max_length=max_length,
    )
    payload = {
        "protocol": "reviewer_followup_local",
        "device": device,
        "prompt_limit": prompt_limit,
        "max_length": max_length,
        "canonical_artifacts_modified": False,
        "donor_scaling": {
            "measured_donors": 1,
            "multiple_donor_models_available": False,
            "status": "not_measured_because_only_one_real_donor_pair_is_available",
            "controlled_synthetic_scaling_artifact": "results/reviewer_followup/controlled/controlled_summary.json",
        },
        "models": [qwen3, qwen25],
        "interpretation": (
            "All timings are local warmup/repeated diagnostics. Directional "
            "Jacobian and adjoint values are local proxies, not certified "
            "global smoothness constants."
        ),
    }
    report_path = output_dir / "cost_diagnostics.json"
    if report_path.is_file():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"immutable cost report differs: {report_path}")
    else:
        report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "command": "python -m suture.reviewer_cost_diagnostics",
        "output": str(report_path),
        "canonical_artifacts_modified": False,
        "qwen3_snapshot": str((root / "models/Qwen3-1.7B-70d244cc").resolve()),
        "qwen3_preflight_sha256": sha256_file(
            qwen3_root / "preflight/preflight_report.json"
        ),
    }
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError(f"immutable cost manifest differs: {manifest_path}")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prompt-limit", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--qwen25-model-path",
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
        qwen25_model_path=args.qwen25_model_path,
    )
    print(json.dumps({"models": [row["label"] for row in report["models"]]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
