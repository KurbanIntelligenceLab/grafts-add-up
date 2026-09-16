"""Acquire and certify the exact Qwen3-1.7B snapshot.  Fail closed.

Scientific stages must call ``require_certified_snapshot`` and then load with
``local_files_only=True``.  If this module cannot load the pinned revision, it
records a setup stop.  It never substitutes Qwen2.5 or any other checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping

try:
    from suture.data_manifest import ManifestError
    from suture.tier_a_config import (
        ATTN_IMPLEMENTATION,
        HIDDEN_SIZE,
        MODEL_ARCHITECTURE,
        MODEL_ID,
        MODEL_REVISION,
        MODEL_TYPE,
        N_LAYERS,
        N_PARAMETERS,
        N_PARAMETERS_UNIQUE,
        TORCH_DTYPE,
        VOCAB_SIZE,
        ConfigError,
        refuse_forbidden_model,
        refuse_v1_write,
        repo_root,
        results_root,
        sha256_file,
        snapshot_dir,
    )
except ModuleNotFoundError:
    from data_manifest import ManifestError
    from tier_a_config import (
        ATTN_IMPLEMENTATION,
        HIDDEN_SIZE,
        MODEL_ARCHITECTURE,
        MODEL_ID,
        MODEL_REVISION,
        MODEL_TYPE,
        N_LAYERS,
        N_PARAMETERS,
        N_PARAMETERS_UNIQUE,
        TORCH_DTYPE,
        VOCAB_SIZE,
        ConfigError,
        refuse_forbidden_model,
        refuse_v1_write,
        repo_root,
        results_root,
        sha256_file,
        snapshot_dir,
    )


class PreflightError(ConfigError):
    """Raised when the exact Qwen3 snapshot cannot be certified."""


def _diagnose(exc: BaseException) -> List[str]:
    text = f"{type(exc).__name__}: {exc}"
    hints = [text]
    lowered = text.lower()
    if "401" in text or "unauthorized" in lowered or "token" in lowered:
        hints.append("Check Hugging Face authentication (`huggingface-cli login`).")
    if "offline" in lowered or "local_files_only" in lowered or "not found" in lowered:
        hints.append("The pinned snapshot is not in the local cache; run acquire first.")
    if "connect" in lowered or "timed out" in lowered or "network" in lowered:
        hints.append("Network/connectivity failure while contacting Hugging Face.")
    if "no space" in lowered or "disk" in lowered or "errno 28" in lowered:
        hints.append("Disk space is insufficient for the ~4 GiB snapshot.")
    if "qwen3" in lowered and "keyerror" in lowered:
        hints.append("Upgrade transformers to >=4.51 so the qwen3 architecture is registered.")
    if "out of memory" in lowered or "cuda" in lowered and "memory" in lowered:
        hints.append("VRAM is insufficient; keep BF16/LoRA and batch size 1, but do not change the model.")
    hints.append("Do not fall back to Qwen2.5 or any other checkpoint.")
    return hints


def acquire_snapshot(*, root: Path, allow_download: bool) -> Dict[str, Any]:
    refuse_forbidden_model(MODEL_ID, MODEL_REVISION)
    target = snapshot_dir(root)
    target.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=str(target),
        local_files_only=not allow_download,
    )
    config_path = Path(path) / "config.json"
    if not config_path.is_file():
        raise PreflightError(f"acquired snapshot is missing config.json: {path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("model_type") != MODEL_TYPE:
        raise PreflightError(
            f"acquired model_type is {payload.get('model_type')!r}, expected {MODEL_TYPE}"
        )
    return {
        "path": str(Path(path).resolve()),
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "config_sha256": sha256_file(config_path),
        "allow_download": bool(allow_download),
    }


def inspect_config(snapshot: Path) -> Dict[str, Any]:
    config_path = snapshot / "config.json"
    tokenizer_path = snapshot / "tokenizer_config.json"
    if not config_path.is_file():
        raise PreflightError(f"missing {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tokenizer_cfg = (
        json.loads(tokenizer_path.read_text(encoding="utf-8"))
        if tokenizer_path.is_file()
        else {}
    )
    architectures = list(config.get("architectures") or [])
    errors = []
    if config.get("model_type") != MODEL_TYPE:
        errors.append(f"model_type={config.get('model_type')!r}")
    if MODEL_ARCHITECTURE not in architectures:
        errors.append(f"architectures={architectures!r}")
    if int(config.get("num_hidden_layers", -1)) != N_LAYERS:
        errors.append(f"num_hidden_layers={config.get('num_hidden_layers')}")
    if int(config.get("hidden_size", -1)) != HIDDEN_SIZE:
        errors.append(f"hidden_size={config.get('hidden_size')}")
    if int(config.get("vocab_size", -1)) != VOCAB_SIZE:
        errors.append(f"vocab_size={config.get('vocab_size')}")
    declared_dtype = str(config.get("torch_dtype") or "").lower()
    if declared_dtype and declared_dtype != TORCH_DTYPE:
        errors.append(f"torch_dtype={config.get('torch_dtype')}")
    if errors:
        raise PreflightError("snapshot config does not match the v2 pin: " + "; ".join(errors))
    return {
        "model_type": config["model_type"],
        "architectures": architectures,
        "num_hidden_layers": int(config["num_hidden_layers"]),
        "hidden_size": int(config["hidden_size"]),
        "vocab_size": int(config["vocab_size"]),
        "torch_dtype": declared_dtype or TORCH_DTYPE,
        "config_sha256": sha256_file(config_path),
        "tokenizer_config_sha256": (
            sha256_file(tokenizer_path) if tokenizer_path.is_file() else None
        ),
        "chat_template_present": bool(tokenizer_cfg.get("chat_template")),
    }


def load_certified_model(
    *,
    root: Path,
    device: str,
    local_files_only: bool = True,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    refuse_forbidden_model(MODEL_ID, MODEL_REVISION)
    snapshot = snapshot_dir(root)
    if not (snapshot / "config.json").is_file():
        raise PreflightError(
            f"Qwen3 snapshot is not acquired at {snapshot}. "
            "Run `python -m suture.tier_a_preflight acquire` first."
        )
    inspect_config(snapshot)
    target = torch.device(device)
    dtype = torch.bfloat16 if target.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot),
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot),
        local_files_only=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=ATTN_IMPLEMENTATION,
    ).to(target)
    n_params = int(sum(parameter.numel() for parameter in model.parameters()))
    tied_extra = 0
    if bool(getattr(model.config, "tie_word_embeddings", False)):
        embed = model.get_input_embeddings()
        if embed is not None:
            tied_extra = int(embed.weight.numel())
    reported = n_params + tied_extra
    if n_params != N_PARAMETERS_UNIQUE or reported != N_PARAMETERS:
        raise PreflightError(
            f"parameter count unique={n_params} including_tied_lm_head={reported}; "
            f"expected unique={N_PARAMETERS_UNIQUE} including_tied={N_PARAMETERS}"
        )
    model_type = str(getattr(model.config, "model_type", ""))
    if model_type != MODEL_TYPE:
        raise PreflightError(f"loaded model_type {model_type!r} is not {MODEL_TYPE}")
    class_name = type(model).__name__
    if class_name != MODEL_ARCHITECTURE:
        raise PreflightError(
            f"loaded class {class_name!r} is not {MODEL_ARCHITECTURE}"
        )
    return model, tokenizer, snapshot


def smoke_forward(model, tokenizer, device: str) -> Dict[str, Any]:
    import copy

    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from suture.suture_torch import HFResidualAdapter, TorchPrompt
    from suture.tier_a_config import apply_qwen3_chat

    target = torch.device(device)
    layers = list(model.model.layers)
    if len(layers) != N_LAYERS:
        raise PreflightError(f"loaded {len(layers)} layers, expected {N_LAYERS}")
    text = apply_qwen3_chat(tokenizer, "What is 2 + 2?", enable_thinking=False)
    encoded = tokenizer(text, return_tensors="pt").to(target)
    with torch.inference_mode():
        uncached = model.generate(
            **encoded,
            max_new_tokens=8,
            do_sample=False,
            use_cache=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        cached = model.generate(
            **encoded,
            max_new_tokens=8,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        logits = model(**encoded).logits
    if not torch.equal(uncached, cached):
        raise PreflightError("cache/no-cache greedy tokens disagree on the base snapshot")
    if not torch.isfinite(logits).all():
        raise PreflightError("base logits contain non-finite values")
    completion = tokenizer.decode(
        uncached[0, encoded["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )

    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        max_position_embeddings=16,
        bos_token_id=1,
        eos_token_id=2,
    )
    torch.manual_seed(7)
    host = Qwen3ForCausalLM(config)
    donor = copy.deepcopy(host)
    with torch.no_grad():
        donor.model.layers[1].self_attn.q_proj.weight.add_(0.01)
    fixture = HFResidualAdapter(host, donor, device="cpu")
    prompt = TorchPrompt(
        input_ids=torch.tensor([[1, 5, 7, 9]], dtype=torch.long),
        utility_token_ids=torch.tensor([10]),
        risk_target_token_ids=torch.tensor([11]),
        risk_donor_token_ids=torch.tensor([12]),
    )
    states = fixture.residual_states(prompt)
    if len(states) != 4:
        raise PreflightError("fixture residual trajectory length is wrong")
    grafted = fixture.build_grafted_model((1,))
    generated = grafted.generate(
        input_ids=prompt.input_ids,
        max_new_tokens=2,
        do_sample=False,
        use_cache=False,
        pad_token_id=2,
    )
    generated_cache = grafted.generate(
        input_ids=prompt.input_ids,
        max_new_tokens=2,
        do_sample=False,
        use_cache=True,
        pad_token_id=2,
    )
    peak = None
    if target.type == "cuda":
        peak = float(torch.cuda.max_memory_allocated() / (1024 ** 3))
    return {
        "n_layers": len(layers),
        "generation_preview": completion[:200],
        "peak_vram_gib": peak,
        "finite_logits": True,
        "cache_no_cache_token_match": True,
        "fixture_layers": fixture.n_layers,
        "fixture_residual_len": len(states),
        "fixture_graft_tokens": int(generated.shape[1]),
        "fixture_cache_shape_match": tuple(generated.shape) == tuple(generated_cache.shape),
        "merged_model_builds": fixture.merged_model_builds,
    }


def write_report(path: Path, payload: Mapping[str, Any]) -> None:
    refuse_v1_write(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def run_preflight(
    *,
    root: Path,
    device: str,
    allow_download: bool,
    load_weights: bool,
) -> Dict[str, Any]:
    started = datetime.now(timezone.utc)
    out_dir = results_root(root) / "preflight"
    out_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {
        "schema_version": 1,
        "contract_version": 2,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "fallback": None,
        "device": device,
        "started_at": started.isoformat(),
        "status": "RUNNING",
    }
    try:
        acquired = acquire_snapshot(root=root, allow_download=allow_download)
        inspected = inspect_config(Path(acquired["path"]))
        report["acquire"] = acquired
        report["config"] = inspected
        if load_weights:
            import torch

            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            model, tokenizer, snapshot = load_certified_model(
                root=root,
                device=device,
                local_files_only=True,
            )
            smoke = smoke_forward(model, tokenizer, device)
            report["load"] = {
                "snapshot": str(snapshot),
                "class_name": type(model).__name__,
                "dtype": str(next(model.parameters()).dtype),
                "n_parameters_unique": int(sum(p.numel() for p in model.parameters())),
                "n_parameters_including_tied_lm_head": N_PARAMETERS,
            }
            report["smoke"] = smoke
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            report["status"] = "PASS_PREFLIGHT"
        else:
            report["status"] = "PASS_ACQUIRE"
            report["note"] = "weights not loaded; run with --load-weights before science"
        report["ended_at"] = datetime.now(timezone.utc).isoformat()
        write_report(out_dir / "preflight_report.json", report)
        return report
    except Exception as exc:
        report["status"] = "STOP_SETUP"
        report["error"] = _diagnose(exc)
        report["traceback"] = traceback.format_exc()
        report["ended_at"] = datetime.now(timezone.utc).isoformat()
        write_report(out_dir / "preflight_report.json", report)
        raise PreflightError(
            "Qwen3 preflight failed; refusing to continue on any other model. "
            + " | ".join(_diagnose(exc))
        ) from exc


def pretrained_load_args(root: Path | None = None) -> Dict[str, Any]:
    """Return from_pretrained kwargs for the certified local Qwen3 snapshot."""

    snapshot = require_certified_snapshot(root)
    return {
        "pretrained_model_name_or_path": str(snapshot),
        "local_files_only": True,
    }


def require_certified_snapshot(root: Path | None = None) -> Path:
    """Fail closed unless a passing preflight report exists for this pin."""

    base = root or repo_root()
    report_path = results_root(base) / "preflight" / "preflight_report.json"
    if not report_path.is_file():
        raise PreflightError(
            f"missing preflight report at {report_path}. "
            "Run `python -m suture.tier_a_preflight --load-weights` first."
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS_PREFLIGHT":
        raise PreflightError(f"preflight did not pass: {report.get('status')}")
    if report.get("model_id") != MODEL_ID or report.get("model_revision") != MODEL_REVISION:
        raise PreflightError("preflight report is for a different model pin")
    snapshot = snapshot_dir(base)
    inspect_config(snapshot)
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--load-weights", action="store_true")
    parser.add_argument("command", nargs="?", choices=["acquire", "preflight", "train-smoke"], default="preflight")
    args = parser.parse_args()
    root = args.root or repo_root()
    if args.command == "train-smoke":
        metadata = train_smoke(root=root, device=args.device)
        print(json.dumps({"status": metadata.get("status"), "loss_last": metadata.get("loss_last")}, indent=2))
        return 0
    allow = bool(args.allow_download or args.command == "acquire")
    load = bool(args.load_weights or args.command == "preflight")
    if args.command == "acquire" and not args.load_weights:
        load = False
    report = run_preflight(
        root=root,
        device=args.device,
        allow_download=allow,
        load_weights=load,
    )
    print(json.dumps(report, indent=2, sort_keys=True)[:4000])
    return 0 if report.get("status") in {"PASS_PREFLIGHT", "RUNNING"} or not load else 0


def train_smoke(*, root: Path, device: str) -> Dict[str, Any]:
    """One AdamW/LoRA step on the certified snapshot.  Not a scientific expert."""

    from suture.data_manifest import write_manifest
    from suture.tier_a_train import train_role

    require_certified_snapshot(root)
    smoke_dir = results_root(root) / "preflight" / "train_smoke"
    refuse_v1_write(smoke_dir, root=root)
    data_dir = smoke_dir / "data"
    write_manifest(
        data_dir / "P_util.jsonl",
        [
            {
                "source_id": "smoke:gsm8k:0",
                "source_question": "What is 1+1?",
                "answer_number": "2",
                "language": "en",
            }
        ],
        manifest_name="P_util",
        metadata={"purpose": "preflight_train_smoke"},
    )
    write_manifest(
        data_dir / "host_train.jsonl",
        [
            {
                "source_id": "smoke:host:0",
                "prompt": "Say hello in Spanish.",
                "completion": "Hola.",
                "language": "es",
            }
        ],
        manifest_name="host_train",
        metadata={"purpose": "preflight_train_smoke"},
    )
    metadata = train_role(
        role="donor",
        data_dir=data_dir,
        output_dir=smoke_dir / "adapter",
        language="en",
        seed=0,
        steps=1,
        batch_size=1,
        grad_accumulation=1,
        max_length=64,
        learning_rate=1e-5,
        lora_rank=8,
        device=device,
        checkpoint_steps=(),
    )
    from suture.tier_a_config import assert_adapter_matches_pin

    assert_adapter_matches_pin(smoke_dir / "adapter")
    metadata["status"] = "PASS_TRAIN_SMOKE"
    write_report(smoke_dir / "train_smoke.json", metadata)
    return metadata


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PreflightError, ConfigError, ManifestError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
