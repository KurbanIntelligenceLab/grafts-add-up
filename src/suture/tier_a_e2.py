"""Measure the SUTURE linearisation regime across expert checkpoints.

E2 is deliberately diagnostic: it does not read MGSM_test and does not tune
the E1 contract.  It varies training amount and graft size, then records the
first-order prediction, realised teacher-forced utility change, a Jacobian-gap
proxy, and the remainder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

try:
    from suture.data_manifest import ManifestError, read_manifest
    from suture.suture_metrics import spearman
    from suture.suture_torch import HFResidualAdapter
    from suture.tier_a_config import MODEL_ID, MODEL_REVISION, refuse_v1_write
    from suture.tier_a_gate import (
        _load_shared_pair,
        _predicted_for_sets,
        _risk_prompts,
        _utility_prompts,
    )
except ModuleNotFoundError:
    from data_manifest import ManifestError, read_manifest
    from suture_metrics import spearman
    from suture_torch import HFResidualAdapter
    from tier_a_config import MODEL_ID, MODEL_REVISION, refuse_v1_write
    from tier_a_gate import (
        _load_shared_pair,
        _predicted_for_sets,
        _risk_prompts,
        _utility_prompts,
    )


def _checkpoint_dirs(root: Path) -> List[Tuple[str, Path]]:
    candidates: Dict[str, Path] = {}
    for config in root.rglob("adapter_config.json"):
        adapter_dir = config.parent
        checkpoint_dir = (
            adapter_dir
            if adapter_dir.name.startswith("checkpoint-")
            else adapter_dir.parent
        )
        label = (
            checkpoint_dir.name
            if checkpoint_dir.name.startswith("checkpoint-")
            else "final"
        )
        candidates[label] = checkpoint_dir
    if not candidates:
        raise ManifestError(f"no adapter checkpoints found below {root}")
    return sorted(candidates.items(), key=lambda item: item[0])


def _grafts(
    n_layers: int,
    sizes: Sequence[int],
    count: int,
    seed: int,
) -> List[Tuple[int, ...]]:
    rng = np.random.default_rng(seed)
    result: List[Tuple[int, ...]] = []
    for size in sizes:
        width = min(int(size), n_layers)
        starts = np.arange(n_layers - width + 1)
        if len(starts) > count:
            starts = rng.choice(starts, size=count, replace=False)
        for start in sorted(int(value) for value in starts):
            result.append(tuple(range(start, start + width)))
    return result


def run(
    *,
    language: str,
    data_dir: Path,
    host_root: Path,
    donor_root: Path,
    output_dir: Path,
    device: str,
    probe_limit: int | None = None,
    graft_sizes: Sequence[int] = (1, 2, 4, 8, 16),
    grafts_per_size: int = 8,
    max_length: int = 256,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    refuse_v1_write(output_dir)
    utility_records = list(read_manifest(data_dir / "P_util.jsonl")[1])
    risk_records = list(read_manifest(data_dir / "P_risk.jsonl")[1])
    if probe_limit is not None:
        utility_records = utility_records[:probe_limit]
        risk_records = risk_records[:probe_limit]
    host_checkpoints = _checkpoint_dirs(host_root)
    donor_checkpoints = dict(_checkpoint_dirs(donor_root))
    if {name for name, _ in host_checkpoints} - set(donor_checkpoints):
        raise ManifestError("host and donor checkpoint names do not match")

    output_path = output_dir / "e2_regime.jsonl"
    existing: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    if output_path.is_file():
        with output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    existing[
                        (
                            str(row["checkpoint"]),
                            int(row["graft_size"]),
                            int(row["start"]),
                        )
                    ] = row

    all_rows: List[Dict[str, Any]] = []
    with output_path.open("a", encoding="utf-8") as handle:
        for checkpoint, host_path in host_checkpoints:
            donor_path = donor_checkpoints[checkpoint]
            shared, tokenizer = _load_shared_pair(host_path, donor_path, device)
            adapter = HFResidualAdapter(
                shared,
                shared,
                device=device,
                host_adapter_name="host",
                donor_adapter_name="donor",
            )
            utility_prompts = _utility_prompts(
                tokenizer,
                utility_records,
                batch_size=1,
                max_length=max_length,
            )
            risk_prompts = _risk_prompts(
                tokenizer,
                risk_records,
                language=language,
                batch_size=1,
                max_length=max_length,
            )
            scores = adapter.score_selection(utility_prompts, risk_prompts)
            for graft in _grafts(
                adapter.n_layers,
                graft_sizes,
                grafts_per_size,
                seed=int.from_bytes(
                    hashlib.sha256(
                        f"{language}:{checkpoint}".encode("utf-8")
                    ).digest()[:4],
                    "big",
                ),
            ):
                key = (checkpoint, len(graft), graft[0])
                if key in existing:
                    all_rows.append(existing[key])
                    continue
                started = time.perf_counter()
                baseline = adapter.build_grafted_model(())
                baseline_value = float(
                    np.mean(
                        [
                            baseline.readout(prompt, "utility")
                            for prompt in utility_prompts
                        ]
                    )
                )
                model = adapter.build_grafted_model(graft)
                measured_value = float(
                    np.mean(
                        [
                            model.readout(prompt, "utility")
                            for prompt in utility_prompts
                        ]
                    )
                ) - baseline_value
                predicted_value = float(scores.predict(graft)[0])
                epsilon = float(scores.epsilon(graft))
                jvp = float(
                    adapter.jacobian_gap_proxy(
                        utility_prompts[0],
                        layer=graft[len(graft) // 2],
                        seed=int.from_bytes(
                            hashlib.sha256(
                                f"{checkpoint}:{graft}".encode("utf-8")
                            ).digest()[:4],
                            "big",
                        ),
                    )
                )
                row = {
                    "protocol_version": 1,
                    "language": language,
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "checkpoint": checkpoint,
                    "graft": list(graft),
                    "graft_size": len(graft),
                    "start": graft[0],
                    "predicted_utility": predicted_value,
                    "measured_utility": measured_value,
                    "epsilon_S": epsilon,
                    "jacobian_gap_proxy": jvp,
                    "remainder": measured_value - predicted_value,
                    "relative_magnitude_error": abs(
                        measured_value - predicted_value
                    )
                    / max(abs(measured_value), 1e-12),
                    "wall_clock_seconds": time.perf_counter() - started,
                    "n_probe_utility": len(utility_records),
                    "n_probe_risk": len(risk_records),
                }
                existing[key] = row
                all_rows.append(row)
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
            del adapter, shared

    by_checkpoint: Dict[str, Dict[str, Any]] = {}
    for checkpoint in sorted({str(row["checkpoint"]) for row in all_rows}):
        rows = [row for row in all_rows if row["checkpoint"] == checkpoint]
        predicted = np.asarray([row["predicted_utility"] for row in rows])
        measured = np.asarray([row["measured_utility"] for row in rows])
        by_checkpoint[checkpoint] = {
            "n": len(rows),
            "spearman": float(spearman(predicted, measured)),
            "mean_relative_magnitude_error": float(
                np.mean([row["relative_magnitude_error"] for row in rows])
            ),
            "mean_remainder": float(np.mean([row["remainder"] for row in rows])),
            "mean_epsilon_S": float(np.mean([row["epsilon_S"] for row in rows])),
            "mean_jacobian_gap_proxy": float(
                np.mean([row["jacobian_gap_proxy"] for row in rows])
            ),
        }
    report = {
        "protocol_version": 1,
        "language": language,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "probe_limit": probe_limit,
        "graft_sizes": list(graft_sizes),
        "grafts_per_size": grafts_per_size,
        "checkpoints": by_checkpoint,
        "output": str(output_path),
        "test_manifest_read": "MGSM_test was not read",
    }
    with (output_dir / "e2_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", required=True, choices=("es", "zh", "sw"))
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--host-root", required=True, type=Path)
    parser.add_argument("--donor-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--probe-limit", type=int)
    parser.add_argument("--grafts-per-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()
    report = run(
        language=args.language,
        data_dir=args.data_dir,
        host_root=args.host_root,
        donor_root=args.donor_root,
        output_dir=args.output_dir,
        device=args.device,
        probe_limit=args.probe_limit,
        grafts_per_size=args.grafts_per_size,
        max_length=args.max_length,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
