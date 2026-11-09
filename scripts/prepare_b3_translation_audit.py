#!/usr/bin/env python3
"""Recover paired B3 source and translated probe texts from frozen manifests."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from suture.b3_data import load_risk_source, load_utility_source  # noqa: E402
from suture.b3_lighton import B3Error, load_contract, sha256_file  # noqa: E402
from suture.data_manifest import ManifestError, read_manifest  # noqa: E402


CONFIG_REL = "configs/b3_lighton_qwen3_8b_v1.json"
DATA_ROOT_REL = "results/b3/lighton_qwen3_8b"
OUTPUT_DIR_REL = "results/b3/translation_audit_v1"
LANGUAGE_LABELS = {"fr": "français", "zh": "中文"}
CHAT_PREFIX = "<|im_start|>user\n"
CHAT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise B3Error(message)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def extract_translation(input_text: str, marker: str, *, context: str) -> str:
    """Remove only the exact known Qwen wrapper and final readout marker."""
    suffix = f"\n{marker}{CHAT_SUFFIX}"
    require(input_text.startswith(CHAT_PREFIX), f"{context}: unexpected Qwen chat prefix")
    require(input_text.endswith(suffix), f"{context}: unexpected final marker/chat suffix")
    translated = input_text[len(CHAT_PREFIX) : -len(suffix)]
    require(bool(translated), f"{context}: empty translated text")
    return translated


def source_hash(row: Mapping[str, Any], kind: str) -> str:
    field = "source_question" if kind == "P_util" else "source_text"
    require(isinstance(row.get(field), str) and row[field], f"{kind}: missing {field}")
    return sha256_text(row[field])


def verify_header_and_metadata(
    header: Mapping[str, Any],
    *,
    language: str,
    kind: str,
    contract: Mapping[str, Any],
) -> None:
    require(header.get("_manifest") == f"b3_{language}_{kind}",
            f"{language}/{kind}: unexpected manifest name")
    require(header.get("schema_version") == 1, f"{language}/{kind}: schema version mismatch")
    require(header.get("n_records") == 500, f"{language}/{kind}: expected 500 records")
    metadata = header.get("metadata")
    require(isinstance(metadata, dict), f"{language}/{kind}: missing metadata")
    require(metadata.get("contract_name") == contract["contract_name"],
            f"{language}/{kind}: contract name mismatch")
    require(metadata.get("contract_version") == contract["contract_version"],
            f"{language}/{kind}: contract version mismatch")
    require(metadata.get("language") == language, f"{language}/{kind}: language mismatch")
    require(metadata.get("readout_uses_first_token_only") is True,
            f"{language}/{kind}: readout metadata mismatch")
    translation = contract["data"]["sources"]["translation"]
    manifest_translation = metadata.get("translation_model")
    require(isinstance(manifest_translation, dict),
            f"{language}/{kind}: missing translation provenance")
    for key in ("id", "revision", "method", "max_new_tokens", "batch_size"):
        require(manifest_translation.get(key) == translation.get(key),
                f"{language}/{kind}: translation provenance {key} mismatch")


def verify_record(
    record: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    language: str,
    kind: str,
    context: str,
    translation_spec: Mapping[str, Any],
) -> str:
    source_id = source["source_id"]
    source_index = source["source_index"]
    expected_id = f"b3:{language}:{kind}:{source_id}"
    require(record.get("id") == expected_id, f"{context}: record ID mismatch")
    require(record.get("source_id") == source_id, f"{context}: source ID mismatch")
    require(record.get("source_index") == source_index, f"{context}: source index mismatch")
    require(record.get("language") == language, f"{context}: record language mismatch")
    require(record.get("translation_model") == translation_spec["id"],
            f"{context}: translation model mismatch")
    require(record.get("translation_revision") == translation_spec["revision"],
            f"{context}: translation revision mismatch")

    expected_source_hash = source_hash(source, kind)
    hash_field = "source_question_sha256" if kind == "P_util" else "source_text_sha256"
    require(record.get(hash_field) == expected_source_hash,
            f"{context}: English source SHA-256 mismatch")
    if kind == "P_util":
        require(record.get("utility_texts") == [str(source["answer_number"])],
                f"{context}: GSM8K answer label mismatch")
        marker = "Answer:"
    else:
        require(record.get("source_subset") == source["source_subset"],
                f"{context}: MRB subset mismatch")
        label = LANGUAGE_LABELS[language]
        require(record.get("risk_target_texts") == [label],
                f"{context}: target-language readout mismatch")
        require(record.get("risk_donor_texts") == ["English"],
                f"{context}: donor-language readout mismatch")
        marker = f"Respond in {label}:"
    input_text = record.get("input_text")
    require(isinstance(input_text, str), f"{context}: missing input_text")
    return extract_translation(input_text, marker, context=context)


def manifest_paths() -> dict[tuple[str, str], Path]:
    root = ROOT / DATA_ROOT_REL
    return {
        (language, kind): root / f"{language}_host__en_donor" / "data" / f"{kind}.jsonl"
        for language in ("fr", "zh")
        for kind in ("P_util", "P_risk")
    }


def main() -> None:
    config_path = ROOT / CONFIG_REL
    contract = load_contract(config_path)
    config_sha_before = sha256_file(config_path)
    b3_data_path = ROOT / "src" / "suture" / "b3_data.py"
    b3_data_sha = sha256_file(b3_data_path)
    paths = manifest_paths()
    hashes_before = {relative_path(path): sha256_file(path) for path in paths.values()}

    try:
        utility_rows = load_utility_source(contract)
        risk_rows = load_risk_source(contract)
        loaded_sources = {"P_util": utility_rows, "P_risk": risk_rows}
        for kind, rows in loaded_sources.items():
            require(len(rows) == 500, f"{kind}: pinned source loader did not return 500 rows")
            require(len({row["source_id"] for row in rows}) == 500,
                    f"{kind}: duplicate source IDs in pinned loader output")
            require(len({row["source_index"] for row in rows}) == 500,
                    f"{kind}: duplicate source indices in pinned loader output")

        source_receipts: dict[str, Any] = {}
        for kind, rows in loaded_sources.items():
            pin = contract["data"]["sources"]["utility" if kind == "P_util" else "risk"]
            source_receipts[kind] = {
                "dataset": pin["dataset"],
                "config": pin["config"],
                "split": pin["split"],
                "revision": pin["revision"],
                "selection": pin["selection"],
                "license": pin.get("license"),
                "n_rows": len(rows),
                "ordered_source_ids_sha256": canonical_sha256(
                    [row["source_id"] for row in rows]
                ),
                "ordered_source_indices_sha256": canonical_sha256(
                    [row["source_index"] for row in rows]
                ),
                "canonical_loader_rows_sha256": canonical_sha256(rows),
                "source_text_sha256_list_sha256": canonical_sha256(
                    [source_hash(row, kind) for row in rows]
                ),
            }

        paired_rows: list[dict[str, Any]] = []
        manifest_receipts: dict[str, Any] = {}
        for (language, kind), path in paths.items():
            rel = relative_path(path)
            header, records = read_manifest(path)
            verify_header_and_metadata(
                header, language=language, kind=kind, contract=contract
            )
            sources = loaded_sources[kind]
            require(len(records) == len(sources) == 500,
                    f"{language}/{kind}: manifest and source counts differ")
            before_rows = len(paired_rows)
            for row_index, (source, record) in enumerate(zip(sources, records)):
                context = f"{language}/{kind}[{row_index}]"
                translated = verify_record(
                    record,
                    source,
                    language=language,
                    kind=kind,
                    context=context,
                    translation_spec=contract["data"]["sources"]["translation"],
                )
                english_field = "source_question" if kind == "P_util" else "source_text"
                english = source[english_field]
                paired_rows.append(
                    {
                        "id": f"b3_translation_audit_v1:{language}:{kind}:{source['source_id']}",
                        "language": language,
                        "probe": kind,
                        "source_id": source["source_id"],
                        "source_index": source["source_index"],
                        "source_subset": source.get("source_subset"),
                        "source_text": english,
                        "source_sha256": sha256_text(english),
                        "target_text": translated,
                        "target_sha256": sha256_text(translated),
                        "manifest_record_id": record["id"],
                        "manifest_record_sha256": record["_hash"],
                        "manifest_input_text_sha256": sha256_text(record["input_text"]),
                        "manifest_path": rel,
                    }
                )
            manifest_receipts[rel] = {
                "manifest_name": header["_manifest"],
                "n_records": len(records),
                "file_sha256_before": hashes_before[rel],
                "record_hashes_verified": True,
                "source_ids_indices_and_english_hashes_matched_in_order": True,
                "translated_texts_extracted": len(records),
                "paired_rows_added": len(paired_rows) - before_rows,
            }

        require(len(paired_rows) == 2000, f"expected exactly 2000 pairs, got {len(paired_rows)}")
        pair_ids = [row["id"] for row in paired_rows]
        require(len(set(pair_ids)) == 2000, "duplicate pair IDs")
        for language in ("fr", "zh"):
            for kind in ("P_util", "P_risk"):
                count = sum(row["language"] == language and row["probe"] == kind for row in paired_rows)
                require(count == 500, f"{language}/{kind}: expected 500 pairs, got {count}")

        hashes_after = {relative_path(path): sha256_file(path) for path in paths.values()}
        config_sha_after = sha256_file(config_path)
        require(hashes_before == hashes_after, "a frozen manifest changed during recovery")
        require(config_sha_before == config_sha_after, "pinned config changed during recovery")
        for rel, receipt in manifest_receipts.items():
            receipt["file_sha256_after"] = hashes_after[rel]
            receipt["unchanged_during_recovery"] = True

        output_dir = ROOT / OUTPUT_DIR_REL
        paired_path = output_dir / "paired_texts.jsonl"
        receipt_path = output_dir / "source_recovery.json"
        require(not paired_path.exists(), f"refusing to overwrite {relative_path(paired_path)}")
        require(not receipt_path.exists(), f"refusing to overwrite {relative_path(receipt_path)}")
        output_dir.mkdir(parents=True, exist_ok=True)
        with paired_path.open("x", encoding="utf-8", newline="\n") as handle:
            for row in paired_rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    + "\n"
                )

        receipt = {
            "schema_version": 1,
            "status": "PASS",
            "purpose": "Post-hoc recovery of English source and existing translated B3 probe texts; no translation inference or filtering was performed.",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "contract": {
                "path": CONFIG_REL,
                "contract_name": contract["contract_name"],
                "contract_version": contract["contract_version"],
                "sha256_before": config_sha_before,
                "sha256_after": config_sha_after,
                "unchanged_during_recovery": True,
            },
            "source_loader": {
                "module": "src/suture/b3_data.py",
                "sha256": b3_data_sha,
                "utility_function": "load_utility_source",
                "risk_function": "load_risk_source",
                "dataset_loader_revision_argument": "contract-pinned immutable revision",
            },
            "source_pin_receipts": source_receipts,
            "translation_provenance": {
                key: contract["data"]["sources"]["translation"][key]
                for key in ("id", "revision", "method", "max_new_tokens", "batch_size")
            },
            "frozen_manifests": manifest_receipts,
            "frozen_manifest_hashes_before": hashes_before,
            "frozen_manifest_hashes_after": hashes_after,
            "paired_texts": {
                "path": relative_path(paired_path),
                "n_pairs": len(paired_rows),
                "unique_pair_ids": len(set(pair_ids)),
                "sha256": sha256_file(paired_path),
                "ordering": "FR P_util, FR P_risk, ZH P_util, ZH P_risk; source loader/manifest order within each group",
                "translation_extraction": "exact Qwen chat prefix/suffix and final probe marker removed; no trimming or content rewriting",
            },
        }
        with receipt_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(receipt, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
    except (OSError, ValueError, KeyError, ManifestError, B3Error) as exc:
        raise SystemExit(f"B3 translation audit recovery failed: {exc}") from exc

    print(json.dumps({
        "status": "PASS",
        "pairs": len(paired_rows),
        "groups": {f"{lang}/{kind}": 500 for lang in ("fr", "zh") for kind in ("P_util", "P_risk")},
        "paired_texts": relative_path(paired_path),
        "source_recovery": relative_path(receipt_path),
        "paired_texts_sha256": receipt["paired_texts"]["sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
