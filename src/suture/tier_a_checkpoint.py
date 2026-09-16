"""Select a matched LoRA checkpoint from frozen expert-validation sets only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pyarrow  # noqa: F401  — must load before torch on Windows
import torch

try:
    from suture.data_manifest import ManifestError, read_manifest
    from suture.tier_a_config import (
        CHECKPOINT_SELECTION_RULE,
        MIN_DONOR_EM,
        MIN_DONOR_GAIN,
        MIN_HOST_LOGPROB_GAIN,
        MODEL_ID,
        MODEL_REVISION,
        refuse_v1_write,
        refuse_frozen_write,
    )
    from suture.tier_a_readiness import (
        _adapter_path,
        _donor_eval,
        _load_model,
        _mean_completion_logprob,
        _records,
    )
except ModuleNotFoundError:
    from data_manifest import ManifestError, read_manifest
    from tier_a_config import (
        CHECKPOINT_SELECTION_RULE,
        MIN_DONOR_EM,
        MIN_DONOR_GAIN,
        MIN_HOST_LOGPROB_GAIN,
        MODEL_ID,
        MODEL_REVISION,
        refuse_v1_write,
        refuse_frozen_write,
    )
    from tier_a_readiness import (
        _adapter_path,
        _donor_eval,
        _load_model,
        _mean_completion_logprob,
        _records,
    )


def _checkpoints(root: Path, role: str) -> List[Path]:
    found = []
    for config in sorted(root.rglob("adapter_config.json")):
        parent = config.parent
        found.append(parent)
    unique = []
    seen = set()
    for path in found:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    numbered = [path for path in unique if "checkpoint-" in path.as_posix()]
    finals = [path for path in unique if "checkpoint-" not in path.as_posix()]
    if numbered:
        return numbered
    return finals


def select_checkpoint(
    *,
    role: str,
    adapter_root: Path,
    validation_path: Path,
    output: Path,
    device: str,
    max_length: int = 128,
    max_new_tokens: int = 128,
) -> Dict[str, Any]:
    refuse_frozen_write(output)
    records = _records(validation_path)
    target = torch.device(device)
    base, tokenizer = _load_model(device=target)
    if role == "donor":
        base_metric = _donor_eval(
            base,
            tokenizer,
            records,
            device=target,
            batch_size=1,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
        )
        base_value = float(base_metric["exact_match"])
    else:
        base_metric = _mean_completion_logprob(
            base, tokenizer, records, device=target, max_length=max_length
        )
        base_value = float(base_metric["mean_nats_per_token"])
    del base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[checkpoint:{role}] base={base_value:.4f} n={len(records)}", flush=True)
    rows = []
    selected = None
    for checkpoint in _checkpoints(adapter_root, role):
        print(f"[checkpoint:{role}] evaluating {checkpoint}", flush=True)
        model, tok = _load_model(
            device=target,
            adapter_dir=_adapter_path(checkpoint, role),
            adapter_name=role,
        )
        if role == "donor":
            metric = _donor_eval(
                model,
                tok,
                records,
                device=target,
                batch_size=1,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
            )
            value = float(metric["exact_match"])
            gain = value - base_value
            passed = value >= MIN_DONOR_EM and gain >= MIN_DONOR_GAIN
        else:
            metric = _mean_completion_logprob(
                model, tok, records, device=target, max_length=max_length
            )
            value = float(metric["mean_nats_per_token"])
            gain = value - base_value
            passed = gain >= MIN_HOST_LOGPROB_GAIN
        row = {
            "path": str(checkpoint),
            "metric": metric,
            "value": value,
            "gain": gain,
            "passed": passed,
        }
        rows.append(row)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    try:
        from suture.tier_a_eval import IncompleteDenominatorError, select_best_passing_by_gain
    except ModuleNotFoundError:
        from tier_a_eval import IncompleteDenominatorError, select_best_passing_by_gain
    for row in rows:
        metric_n = int((row.get("metric") or {}).get("n") or 0)
        if metric_n != len(records):
            raise IncompleteDenominatorError(
                f"{role} checkpoint {row['path']} scored n={metric_n}, declared {len(records)}"
            )
    selected = select_best_passing_by_gain(rows) if rows else None
    report = {
        "role": role,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "selection_rule": CHECKPOINT_SELECTION_RULE,
        "base": base_metric,
        "checkpoints": rows,
        "selected": selected,
        "gate": (
            "PASS_CHECKPOINT"
            if selected and selected.get("passed")
            else "STOP_CONTINUE_OR_FAIL"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("donor", "host"), required=True)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    report = select_checkpoint(
        role=args.role,
        adapter_root=args.adapter_root,
        validation_path=args.validation,
        output=args.output,
        device=args.device,
    )
    print(json.dumps({"gate": report["gate"], "selected": report["selected"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
