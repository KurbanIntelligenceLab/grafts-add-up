"""Build disjoint held-out readiness manifests for Tier-A experts.

The existing P_util, P_risk, C, MGSM-dev, and MGSM-test manifests are frozen.
This script creates two additional manifests without reusing their canonical
record hashes:

* readiness_donor: held-out English GSM8K questions for donor exact-match;
* readiness_host: held-out target-language instruction prompts for host
  generation and language-identification checks.

The manifests are intentionally separate from C, which remains reserved for
the drift certificate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

try:
    from suture.data_manifest import (
        ManifestError,
        assert_pairwise_disjoint,
        read_manifest,
        write_manifest,
    )
    from suture.tier_a_data import (
        LANGUAGE_NAMES,
        LocalTranslator,
        _load_gsm8k_train,
        _load_native_instruction_pool,
    )
except ModuleNotFoundError:
    from data_manifest import (
        ManifestError,
        assert_pairwise_disjoint,
        read_manifest,
        write_manifest,
    )
    from tier_a_data import (
        LANGUAGE_NAMES,
        LocalTranslator,
        _load_gsm8k_train,
        _load_native_instruction_pool,
    )


def _records(path: Path) -> List[Dict[str, Any]]:
    _, records = read_manifest(path)
    return records


def _used_ids(data_dir: Path) -> set[str]:
    used: set[str] = set()
    for name in ("P_util", "P_risk", "C", "MGSM_dev", "MGSM_test", "host_train"):
        path = data_dir / f"{name}.jsonl"
        if path.is_file():
            used.update(str(row["source_id"]) for row in _records(path))
    return used


def build(
    *,
    language: str,
    data_dir: str | Path,
    output_dir: str | Path | None = None,
    n_donor: int = 32,
    n_host: int = 32,
    translator_device: str | None = None,
) -> Dict[str, Any]:
    if language not in LANGUAGE_NAMES:
        raise ManifestError(f"language must be one of {sorted(LANGUAGE_NAMES)}")
    data_root = Path(data_dir)
    target = Path(output_dir) if output_dir is not None else data_root
    target.mkdir(parents=True, exist_ok=True)
    used = _used_ids(data_root)

    donor_start = 1024
    donor_pool = _load_gsm8k_train(n_donor, start=donor_start)
    donor_rows = [
        {
            "source_id": row["source_id"],
            "question": row["source_question"],
            "answer_number": row["answer_number"],
            "language": "en",
        }
        for row in donor_pool
        if row["source_id"] not in used
    ]
    if len(donor_rows) != n_donor:
        raise ManifestError("donor readiness pool overlaps a frozen manifest")

    native_pool = _load_native_instruction_pool(language)
    host_pool = [
        row
        for row in native_pool
        if str(row["source_id"]) not in used
    ][:n_host]
    if len(host_pool) != n_host:
        raise ManifestError(
            f"STOP_DATA: only {len(host_pool)} disjoint native host readiness records "
            "available; GSM8K host supplement is forbidden"
        )
    host_rows = [
        {
            "source_id": row["source_id"],
            "prompt": row["prompt"],
            "language": language,
            "source_dataset": row.get("source_dataset", "licensed_target_language_pool"),
        }
        for row in host_pool
    ]
    host_source = "licensed_target_language_pool:held_out"
    translation_model = "dataset-provided"
    translation_revision = None

    donor_path = target / "readiness_donor.jsonl"
    host_path = target / "readiness_host.jsonl"
    donor_summary = write_manifest(
        donor_path,
        donor_rows,
        manifest_name="readiness_donor",
        metadata={
            "language": "en",
            "source_dataset": "openai/gsm8k:main:train:readiness_held_out_range",
            "purpose": "donor exact-match readiness only",
        },
    )
    host_summary = write_manifest(
        host_path,
        host_rows,
        manifest_name="readiness_host",
        metadata={
            "language": language,
            "source_dataset": host_source,
            "translation_model": translation_model,
            "translation_revision": translation_revision,
            "purpose": "host generation and language-ID readiness only",
        },
    )
    all_manifests = {
        "P_util": _records(data_root / "P_util.jsonl"),
        "P_risk": _records(data_root / "P_risk.jsonl"),
        "C": _records(data_root / "C.jsonl"),
        "MGSM_dev": _records(data_root / "MGSM_dev.jsonl"),
        "MGSM_test": _records(data_root / "MGSM_test.jsonl"),
        "host_train": _records(data_root / "host_train.jsonl"),
        "readiness_donor": donor_rows,
        "readiness_host": host_rows,
    }
    disjointness = assert_pairwise_disjoint(all_manifests)
    summary = {
        "language": language,
        "counts": {
            name: len(rows) for name, rows in all_manifests.items()
        },
        "readiness_donor": donor_summary,
        "readiness_host": host_summary,
        "disjointness": disjointness,
    }
    with (target / "readiness_data_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", required=True, choices=sorted(LANGUAGE_NAMES))
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--n-donor", type=int, default=32)
    parser.add_argument("--n-host", type=int, default=32)
    parser.add_argument("--device")
    args = parser.parse_args()
    summary = build(
        language=args.language,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_donor=args.n_donor,
        n_host=args.n_host,
        translator_device=args.device,
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
