"""Classify and hash Tier-A result artifacts without modifying them.

The inventory is written beside ``results/tier_a`` and never touches
``e1_canonical_seed0``.  MGSM-test files may be hashed for integrity, but this
module never reads their records for scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


CANONICAL_E1 = Path("results/tier_a/es/e1_canonical_seed0")
INVENTORY_NAME = "ARTIFACT_INVENTORY.json"
HASH_SUFFIXES = {".json", ".jsonl"}
SKIP_NAMES = {INVENTORY_NAME, "ARTIFACT_INVENTORY.md"}

CLASSIFICATION: Dict[str, Dict[str, Any]] = {
    "es": {
        "status": "canonical_language_panel",
        "mgsm_test_scored": False,
        "directories": {
            "e1_canonical_seed0": {
                "class": "canonical",
                "claim": "primary_e1_stop",
                "reason": "Protocol-v2 free-generation exhaustive Spanish seed-0 E1.",
            },
            "pilots_final": {
                "class": "canonical",
                "claim": "plumbing_p1_p2_p3",
                "reason": "Final float32 P1/P2/P3 on publication adapters.",
            },
            "readiness": {
                "class": "canonical",
                "claim": "expert_readiness",
                "reason": "Seed-0 readiness gate; donor gain is small but strictly positive.",
            },
            "publication_adapters_fixed": {
                "class": "canonical",
                "claim": "publication_experts_seed0",
                "reason": "128-step matched LoRA experts used by canonical E1.",
            },
            "e2": {
                "class": "canonical",
                "claim": "regime_diagnostic",
                "reason": "E2 checkpoint/graft-size regime measurement. Missing run_manifest.",
            },
            "e1": {
                "class": "preliminary",
                "claim": "not_citable_as_e1",
                "reason": "Protocol v1 teacher-forced post-selection MGSM-dev alignment on 16-step adapters.",
            },
            "e1_previous_full_run": {
                "class": "preliminary",
                "claim": "not_citable_as_e1",
                "reason": "Earlier teacher-forced sweep; superseded by canonical seed-0.",
            },
            "adapters": {
                "class": "superseded",
                "claim": "none",
                "reason": "16-step preliminary experts, not the publication recipe.",
            },
            "publication_adapters": {
                "class": "superseded",
                "claim": "none",
                "reason": "Unstable/incomplete publication attempt; donor has extra checkpoint-0256.",
            },
            "publication_adapters_seed0": {
                "class": "superseded",
                "claim": "none",
                "reason": "Incomplete donor-only checkpoint tree.",
            },
            "publication_adapters_stable": {
                "class": "superseded",
                "claim": "none",
                "reason": "Incomplete donor-only checkpoint tree.",
            },
            "pilots": {
                "class": "superseded",
                "claim": "none",
                "reason": "Original bfloat16 P1 plumbing; replaced by pilots_final.",
            },
            "pilots_fixed": {"class": "smoke", "claim": "none", "reason": "P1 debug."},
            "pilots_fixed32": {"class": "smoke", "claim": "none", "reason": "P1 debug."},
            "pilots_float32_fake": {"class": "smoke", "claim": "none", "reason": "P1 debug."},
            "pilots_revalidated": {"class": "smoke", "claim": "none", "reason": "P1 debug."},
            "pilots_revalidated_scale2e3": {"class": "smoke", "claim": "none", "reason": "P1 debug."},
            "pilots_revalidated_scale5e4": {"class": "smoke", "claim": "none", "reason": "P1 debug."},
            "pilots_fake_scale_1e4": {"class": "smoke", "claim": "none", "reason": "P1 debug."},
            "pilots_single_prompt": {"class": "smoke", "claim": "none", "reason": "P1 debug."},
            "pilots_single_prompt_debug": {"class": "smoke", "claim": "none", "reason": "P1 debug."},
            "e1_cache_smoke": {"class": "smoke", "claim": "none", "reason": "Cache plumbing smoke."},
            "e1_cache_smoke_b8": {"class": "smoke", "claim": "none", "reason": "Cache plumbing smoke."},
            "e1_canonical_smoke": {"class": "smoke", "claim": "none", "reason": "Incomplete canonical smoke."},
            "e1_canonical_smoke_b1": {"class": "smoke", "claim": "none", "reason": "Incomplete canonical smoke."},
            "e1_canonical_smoke_batchgen": {"class": "smoke", "claim": "none", "reason": "Incomplete canonical smoke."},
            "e1_canonical_smoke_final": {"class": "smoke", "claim": "none", "reason": "SMOKE_NOT_CANONICAL gate."},
            "e1_canonical_smoke_manualgen": {"class": "smoke", "claim": "none", "reason": "Incomplete canonical smoke."},
            "e1_canonical_smoke_probe8": {"class": "smoke", "claim": "none", "reason": "Incomplete canonical smoke."},
        },
    },
    "zh": {
        "status": "canonical_e1_blocked",
        "mgsm_test_scored": False,
        "directories": {
            "pilots_final": {
                "class": "canonical",
                "claim": "plumbing_p1_p2_p3",
                "reason": "Final float32 P1/P2/P3 on publication adapters.",
            },
            "readiness": {
                "class": "canonical",
                "claim": "expert_readiness",
                "reason": "Seed-0 readiness PASS with the same small donor gain as Spanish.",
            },
            "publication_adapters_fixed": {
                "class": "canonical",
                "claim": "publication_experts_seed0",
                "reason": "128-step matched LoRA experts. Canonical E1 not authorized after Spanish stop.",
            },
            "e2": {
                "class": "canonical",
                "claim": "regime_diagnostic",
                "reason": "E2 regime measurement. Missing run_manifest.",
            },
            "e1": {
                "class": "preliminary",
                "claim": "not_citable_as_e1",
                "reason": "Protocol v1 teacher-forced post-selection alignment; blocked from promotion.",
            },
            "adapters": {
                "class": "superseded",
                "claim": "none",
                "reason": "16-step preliminary experts.",
            },
            "pilots": {
                "class": "superseded",
                "claim": "none",
                "reason": "Original plumbing; replaced by pilots_final.",
            },
        },
    },
    "sw": {
        "status": "canonical_e1_blocked",
        "mgsm_test_scored": False,
        "directories": {
            "pilots_final": {
                "class": "canonical",
                "claim": "plumbing_p1_p2_p3",
                "reason": "Final float32 P1/P2/P3 on publication adapters.",
            },
            "readiness": {
                "class": "canonical",
                "claim": "expert_readiness_failed",
                "reason": "STOP_EXPERT_READINESS: donor_gain is zero.",
            },
            "publication_adapters_fixed": {
                "class": "canonical",
                "claim": "publication_experts_seed0",
                "reason": "128-step matched LoRA experts. Canonical E1 not authorized after Spanish stop.",
            },
            "e2": {
                "class": "canonical",
                "claim": "regime_diagnostic",
                "reason": "E2 regime measurement. Missing run_manifest.",
            },
            "e1": {
                "class": "preliminary",
                "claim": "not_citable_as_e1",
                "reason": "Protocol v1 teacher-forced sweep; readiness already failed.",
            },
            "adapters": {
                "class": "superseded",
                "claim": "none",
                "reason": "16-step preliminary experts.",
            },
            "pilots": {
                "class": "superseded",
                "claim": "none",
                "reason": "Original plumbing; replaced by pilots_final.",
            },
        },
    },
    "es_smoke": {
        "status": "smoke",
        "mgsm_test_scored": False,
        "directories": {},
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_files(directory: Path) -> List[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in HASH_SUFFIXES
        and path.name not in SKIP_NAMES
    )


def _summarize_json(path: Path) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "path": path.as_posix().replace("\\", "/"),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            summary["n_lines"] = sum(1 for line in handle if line.strip())
        return summary
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        summary["parse_error"] = True
        return summary
    if isinstance(payload, Mapping):
        for key in (
            "gate",
            "protocol_version",
            "spearman",
            "donor_gain",
            "mgsm_test_read",
            "test_manifest_read",
            "measured_objective",
            "ranking_prediction_source",
        ):
            if key in payload:
                summary[key] = payload[key]
    return summary


def _classify_directory(language: str, name: str) -> Dict[str, Any]:
    spec = CLASSIFICATION.get(language, {})
    directories = spec.get("directories", {})
    if name in directories:
        return dict(directories[name])
    return {
        "class": "unclassified",
        "claim": "none",
        "reason": "Directory is not in the frozen classification table.",
    }


def inventory(root: Path) -> Dict[str, Any]:
    tier = root / "results" / "tier_a"
    if not tier.is_dir():
        raise FileNotFoundError(f"missing results tree: {tier}")
    languages: Dict[str, Any] = {}
    for language_dir in sorted(path for path in tier.iterdir() if path.is_dir()):
        language = language_dir.name
        spec = CLASSIFICATION.get(language, {})
        entries: List[Dict[str, Any]] = []
        for child in sorted(path for path in language_dir.iterdir() if path.is_dir()):
            classification = _classify_directory(language, child.name)
            files = [_summarize_json(path) for path in _json_files(child)]
            entries.append(
                {
                    "directory": child.relative_to(root).as_posix().replace("\\", "/"),
                    **classification,
                    "n_json_artifacts": len(files),
                    "has_run_manifest": (child / "run_manifest.json").is_file(),
                    "files": files,
                }
            )
        root_files = [
            _summarize_json(path)
            for path in sorted(language_dir.iterdir())
            if path.is_file() and path.suffix.lower() in HASH_SUFFIXES
        ]
        languages[language] = {
            "status": spec.get("status", "unclassified"),
            "mgsm_test_scored": bool(spec.get("mgsm_test_scored", False)),
            "root_files": root_files,
            "directories": entries,
        }
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "canonical_e1": CANONICAL_E1.as_posix().replace("\\", "/"),
        "canonical_e1_immutable": True,
        "mgsm_test_policy": "hashed_for_integrity_never_scored",
        "blocked_by_spanish_stop": [
            "zh canonical E1",
            "sw canonical E1",
            "expert seeds 1 and 2",
            "E3-E10",
            "Tier B",
            "MGSM_test evaluation",
        ],
        "languages": languages,
    }


def write_inventory(root: Path) -> Path:
    payload = inventory(root)
    output = root / "results" / "tier_a" / INVENTORY_NAME
    if CANONICAL_E1.resolve() == output.resolve():
        raise RuntimeError("inventory path must not overwrite canonical E1")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    markdown = root / "docs" / "iclr" / "TIER_A_ARTIFACT_INVENTORY.md"
    lines = [
        "# Tier-A artifact inventory",
        "",
        "This inventory classifies existing result directories. Canonical Spanish",
        "E1 files were not rewritten. MGSM-test records were not scored.",
        "",
        f"- Written: `{payload['created_at']}`",
        f"- Canonical E1: `{payload['canonical_e1']}`",
        f"- MGSM-test policy: `{payload['mgsm_test_policy']}`",
        "- Contract v2 (`results/v2/qwen3_1_7b/`): scientific Qwen3 measurement. es `PASS_READINESS` then `STOP_PLUMBING_P1`; zh/sw `STOP_EXPERT_READINESS` on host LID. No Qwen3 E1.",
        "",
        "| Language | Directory | Class | Claim | Manifest |",
        "|---|---|---|---|---|",
    ]
    for language, body in payload["languages"].items():
        for entry in body["directories"]:
            name = Path(entry["directory"]).name
            lines.append(
                f"| {language} | `{name}` | {entry['class']} | "
                f"{entry['claim']} | {entry['has_run_manifest']} |"
            )
    lines.extend(
        [
            "",
            "Contract v2 (`results/v2/qwen3_1_7b/`):",
            "",
            "| Language | Directory | Class | Claim | Manifest |",
            "|---|---|---|---|---|",
            "| es | `readiness` | canonical | expert_readiness_passed | True |",
            "| es | `pilots` | canonical | plumbing_p1_failed | False |",
            "| zh | `readiness` | canonical | expert_readiness_failed | True |",
            "| sw | `readiness` | canonical | expert_readiness_failed | True |",
            "",
        ]
    )
    markdown.write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", type=Path)
    args = parser.parse_args()
    output = write_inventory(args.root.resolve())
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
