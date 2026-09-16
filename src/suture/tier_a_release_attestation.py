"""Hash immutable Tier-A evidence without rewriting it."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List


CANONICAL_PATHS = [
    "results/tier_a/es/e1_canonical_seed0/e1_config.json",
    "results/tier_a/es/e1_canonical_seed0/e1_gate.json",
    "results/tier_a/es/e1_canonical_seed0/e1_selection_scores.json",
    "results/tier_a/es/e1_canonical_seed0/e1_baseline.json",
    "results/tier_a/es/e1_canonical_seed0/e1_window_sweep.jsonl",
    "results/tier_a/es/e1_canonical_seed0/run_manifest.json",
    "results/tier_a/es/P_util.jsonl",
    "results/tier_a/es/P_risk.jsonl",
    "results/tier_a/es/C.jsonl",
    "results/tier_a/es/MGSM_dev.jsonl",
    "results/tier_a/es/MGSM_test.jsonl",
    "results/tier_a/es/readiness/seed0_readiness.json",
    "results/tier_a/zh/readiness/seed0_readiness.json",
    "results/tier_a/sw/readiness/seed0_readiness.json",
    "results/tier_a/es/e2/e2_report.json",
    "results/tier_a/zh/e2/e2_report.json",
    "results/tier_a/sw/e2/e2_report.json",
    "results/tier_a/es/pilots_final/pilots.json",
    "results/tier_a/zh/pilots_final/pilots.json",
    "results/tier_a/sw/pilots_final/pilots.json",
    "configs/experimental_contract_v1.json",
    "results/tier_a/ARTIFACT_INVENTORY.json",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def attest(root: Path, paths: Iterable[str] = CANONICAL_PATHS) -> Dict[str, object]:
    files: List[Dict[str, object]] = []
    missing: List[str] = []
    for relative in paths:
        path = root / relative
        if not path.is_file():
            missing.append(relative.replace("\\", "/"))
            continue
        files.append(
            {
                "path": relative.replace("\\", "/"),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    return {
        "schema_version": 1,
        "kind": "release_attestation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "canonical_e1_rewritten": False,
        "mgsm_test_scored": False,
        "files": files,
        "missing": missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", type=Path)
    parser.add_argument(
        "--output",
        default="results/tier_a/RELEASE_ATTESTATION.json",
        type=Path,
    )
    args = parser.parse_args()
    root = args.root.resolve()
    payload = attest(root)
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(output)
    return 0 if not payload["missing"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
