#!/usr/bin/env python3
"""Measure re-encoded lengths of frozen B3 translated probe texts only."""

from __future__ import annotations

import collections
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_REL = "configs/b3_lighton_qwen3_8b_v1.json"
PAIRED_REL = "results/b3/translation_audit_v1/paired_texts.jsonl"
RECOVERY_REL = "results/b3/translation_audit_v1/source_recovery.json"
OUTPUT_ROOT_REL = "results/b3/translation_audit_v1"
TOKENIZER_FILES = (
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
)

EXAMPLE_TAILS = {
    "b3_translation_audit_v1:fr:P_util:openai/gsm8k:main:train:8": {
        "tail": "11 dollars pour des",
        "observation": "ends after an unresolved determiner phrase",
    },
    "b3_translation_audit_v1:fr:P_util:openai/gsm8k:main:train:10": {
        "tail": "le monstre a",
        "observation": "ends after an incomplete clause",
    },
    "b3_translation_audit_v1:fr:P_util:openai/gsm8k:main:train:11": {
        "tail": "les chauss",
        "observation": "ends with a partial word",
    },
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {rel(path)}")
    return value


def main() -> None:
    contract_path = ROOT / CONTRACT_REL
    paired_path = ROOT / PAIRED_REL
    recovery_path = ROOT / RECOVERY_REL
    contract = load_json(contract_path)
    recovery = load_json(recovery_path)
    contract_sha_before = sha256_file(contract_path)
    paired_sha_before = sha256_file(paired_path)
    recovery_sha = sha256_file(recovery_path)
    require(recovery.get("status") == "PASS", "source recovery receipt is not PASS")
    require(
        recovery.get("paired_texts", {}).get("sha256") == paired_sha_before,
        "paired texts do not match source recovery receipt",
    )

    translation = contract["data"]["sources"]["translation"]
    max_new_tokens = int(translation["max_new_tokens"])
    snapshot_rel = translation["local_snapshot"]
    snapshot = (ROOT / snapshot_rel).resolve()
    require(snapshot.is_relative_to(ROOT.resolve()), "tokenizer path escapes repository root")
    require(snapshot.is_dir(), f"pinned local tokenizer snapshot missing: {snapshot_rel}")
    tokenizer_hashes = {}
    for filename in TOKENIZER_FILES:
        file_path = snapshot / filename
        require(file_path.is_file(), f"tokenizer snapshot is missing {filename}")
        tokenizer_hashes[f"{snapshot_rel}/{filename}"] = sha256_file(file_path)

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False
    )
    require(tokenizer.is_fast, "expected the pinned fast tokenizer")

    pairs = []
    for line_number, line in enumerate(paired_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        row = json.loads(line)
        require(isinstance(row, dict), f"paired row {line_number} is not an object")
        require(isinstance(row.get("target_text"), str), f"paired row {line_number} lacks target_text")
        require(
            hashlib.sha256(row["target_text"].encode("utf-8")).hexdigest()
            == row.get("target_sha256"),
            f"paired row {line_number} target hash mismatch",
        )
        pairs.append(row)
    require(len(pairs) == 2000, f"expected 2000 paired rows, got {len(pairs)}")
    require(len({row["id"] for row in pairs}) == 2000, "duplicate paired IDs")

    records = []
    exact_96_ids = []
    over_96_ids = []
    count_without = collections.Counter()
    count_with = collections.Counter()
    by_group = collections.Counter((row["language"], row["probe"]) for row in pairs)
    require(
        by_group == {
            ("fr", "P_util"): 500,
            ("fr", "P_risk"): 500,
            ("zh", "P_util"): 500,
            ("zh", "P_risk"): 500,
        },
        f"unexpected paired group counts: {dict(by_group)}",
    )
    pair_by_id = {row["id"]: row for row in pairs}
    example_observations = []
    for pair in pairs:
        text = pair["target_text"]
        without_specials = tokenizer(text, add_special_tokens=False)["input_ids"]
        with_specials = tokenizer(text, add_special_tokens=True)["input_ids"]
        n_without = len(without_specials)
        n_with = len(with_specials)
        group_key = (pair["language"], pair["probe"])
        count_without[(group_key, n_without)] += 1
        count_with[(group_key, n_with)] += 1
        if n_without == max_new_tokens:
            exact_96_ids.append(pair["id"])
        elif n_without > max_new_tokens:
            over_96_ids.append(pair["id"])
        records.append(
            {
                "id": pair["id"],
                "language": pair["language"],
                "probe": pair["probe"],
                "source_id": pair["source_id"],
                "source_index": pair["source_index"],
                "target_sha256": pair["target_sha256"],
                "target_tokens_without_specials": n_without,
                "target_tokens_with_specials": n_with,
                "translation_model": translation["id"],
                "translation_revision": translation["revision"],
            }
        )
        spec = EXAMPLE_TAILS.get(pair["id"])
        if spec is not None:
            require(text.endswith(spec["tail"]), f"observed example tail changed for {pair['id']}")
            require(n_without == max_new_tokens,
                    f"observed example does not re-encode to the configured cap: {pair['id']}")
            example_observations.append(
                {
                    "id": pair["id"],
                    "target_tail": spec["tail"],
                    "target_tokens_without_specials": n_without,
                    "observation": spec["observation"],
                    "review_basis": "AI inspection of the preserved text only; not human translation QA",
                }
            )
    require(len(example_observations) == len(EXAMPLE_TAILS), "not all example observations were found")

    def group_counts(counter: collections.Counter, token_count: int) -> dict[str, int]:
        return {
            f"{language}/{probe}": counter[((language, probe), token_count)]
            for language in ("fr", "zh")
            for probe in ("P_util", "P_risk")
        }

    output_root = ROOT / OUTPUT_ROOT_REL
    records_path = output_root / "tokenizer_length_records.jsonl"
    summary_path = output_root / "tokenizer_length_summary.json"
    require(not records_path.exists(), f"refusing to overwrite {rel(records_path)}")
    require(not summary_path.exists(), f"refusing to overwrite {rel(summary_path)}")
    with records_path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )

    paired_sha_after = sha256_file(paired_path)
    contract_sha_after = sha256_file(contract_path)
    require(paired_sha_after == paired_sha_before, "paired text input changed during tokenization")
    require(contract_sha_after == contract_sha_before, "pinned contract changed during tokenization")
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "purpose": "Tokenizer-only re-encoding diagnostic over frozen translated target_text fields; no model inference, translation, filtering, or text edits.",
        "input": {
            "path": PAIRED_REL,
            "n_pairs": len(pairs),
            "sha256_before": paired_sha_before,
            "sha256_after": paired_sha_after,
            "unchanged_during_check": True,
            "source_recovery_receipt": RECOVERY_REL,
            "source_recovery_receipt_sha256": recovery_sha,
        },
        "tokenizer": {
            "model_id": translation["id"],
            "revision": translation["revision"],
            "local_snapshot": snapshot_rel,
            "fast_tokenizer": True,
            "transformers_version": importlib.metadata.version("transformers"),
            "tokenizer_asset_sha256": tokenizer_hashes,
        },
        "original_translation_generation": {
            "configured_max_new_tokens": max_new_tokens,
            "model_id": translation["id"],
            "revision": translation["revision"],
            "original_generated_token_ids_retained": False,
            "original_finish_reasons_retained": False,
            "interpretation": "Re-encoded text counts are not observed generated lengths or proof of max-token termination. Exact equality to 96 is only a candidate cap indicator; values above 96 show that text re-encoding is not identical to the original generation token sequence.",
        },
        "counts_without_special_tokens": {
            "exactly_configured_cap_by_group": group_counts(count_without, max_new_tokens),
            "above_configured_cap_by_group": {
                f"{language}/{probe}": sum(
                    count_without[((language, probe), n)]
                    for n in range(max_new_tokens + 1, max((r["target_tokens_without_specials"] for r in records), default=max_new_tokens) + 1)
                )
                for language in ("fr", "zh")
                for probe in ("P_util", "P_risk")
            },
            "below_configured_cap_by_group": {
                f"{language}/{probe}": sum(
                    count_without[((language, probe), n)] for n in range(max_new_tokens)
                )
                for language in ("fr", "zh")
                for probe in ("P_util", "P_risk")
            },
            "exactly_configured_cap_pair_ids": exact_96_ids,
            "above_configured_cap_pair_ids": over_96_ids,
        },
        "counts_with_special_tokens": {
            "exactly_configured_cap_by_group": group_counts(count_with, max_new_tokens),
            "above_configured_cap_by_group": {
                f"{language}/{probe}": sum(
                    count_with[((language, probe), n)]
                    for n in range(max_new_tokens + 1, max((r["target_tokens_with_specials"] for r in records), default=max_new_tokens) + 1)
                )
                for language in ("fr", "zh")
                for probe in ("P_util", "P_risk")
            },
        },
        "per_pair_records": {
            "path": rel(records_path),
            "n_records": len(records),
            "sha256": sha256_file(records_path),
            "fields": [
                "id", "language", "probe", "source_id", "source_index", "target_sha256",
                "target_tokens_without_specials", "target_tokens_with_specials",
                "translation_model", "translation_revision",
            ],
        },
        "observed_cutoff_examples": {
            "inspection_basis": "AI inspection of preserved text only; not human translation QA",
            "examples": example_observations,
        },
    }
    with summary_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")

    print(json.dumps({
        "status": "PASS",
        "n_records": len(records),
        "exact_96_without_specials": sum(group_counts(count_without, max_new_tokens).values()),
        "above_96_without_specials": sum(summary["counts_without_special_tokens"]["above_configured_cap_by_group"].values()),
        "records_path": rel(records_path),
        "summary_path": rel(summary_path),
        "records_sha256": summary["per_pair_records"]["sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
