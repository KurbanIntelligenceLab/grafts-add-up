"""Post-hoc format-robust scoring for retained B3 free-generation outputs.

This module never edits the frozen generation rows or the original strict EM
fields. A caller must approve the exact parser SHA-256 before any raw files are
read, making parser review a separate step from comparative rescoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from suture.paths import REPO_ROOT


PARSER_VERSION = "b3_generation_scoring_v1"
LANGUAGES = ("fr", "zh")
CANDIDATES = ("host_baseline", "selected", "published_reference")
DEFAULT_CONFIG = REPO_ROOT / "configs" / "b3_generation_followup_v1.json"
STOP_MARKERS = ("<|im_end|>", "<|endoftext|>", "</s>")
COMPLETION_REASONS = {"stop", "eos"}
ANSWER_MARKER_RE = re.compile(
    r"(?P<label>Réponse(?:\s+finale)?|Final\s+answer|Answer|答案|最终答案)\s*[:：]",
    re.IGNORECASE,
)
UNICODE_MINUS = "−‐‑‒–—﹣－"


class ScoringError(RuntimeError):
    """Raised for incomplete or inconsistent retained generation artifacts."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScoringError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_fraction(value: Fraction) -> str:
    value = Fraction(value)
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _normalize_scalar_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = value.replace("\u2044", "/")
    value = value.translate(str.maketrans({character: "-" for character in UNICODE_MINUS}))
    value = "".join(str(unicodedata.decimal(character)) if character.isdecimal() else character for character in value)
    value = value.strip()
    for _ in range(5):
        previous = value
        while value and value[-1] in ".。!！?？;；":
            value = value[:-1].rstrip()
        for wrapper in ("**", "__", "`", "$", "("):
            closing = {"(": ")"}.get(wrapper, wrapper)
            if value.startswith(wrapper) and value.endswith(closing) and len(value) >= len(wrapper) + len(closing):
                value = value[len(wrapper) : len(value) - len(closing)].strip()
                break
        else:
            box = re.fullmatch(r"\\(?:boxed|fbox)\s*\{\s*(.*?)\s*\}", value, flags=re.DOTALL)
            if box:
                value = box.group(1).strip()
        if value == previous:
            break
    return value


def _parse_scalar(text: str) -> Tuple[str, Fraction | None]:
    """Parse one scalar exactly; reject grouping, prose, and operations."""

    value = _normalize_scalar_text(text)
    if not value:
        return "empty_numeric_suffix", None

    fraction_match = re.fullmatch(r"([+-]?\d+)\s*/\s*([+-]?\d+)", value)
    if fraction_match:
        numerator, denominator = int(fraction_match.group(1)), int(fraction_match.group(2))
        if denominator == 0:
            return "zero_fraction_denominator", None
        return "parsed", Fraction(numerator, denominator)
    if "/" in value:
        return "invalid_fraction", None

    if re.fullmatch(r"[+-]?\d+", value):
        return "parsed", Fraction(int(value), 1)

    decimal_match = re.fullmatch(r"([+-]?\d+)([.,])(\d+)", value)
    if decimal_match:
        whole, separator, fraction = decimal_match.groups()
        # With exactly three trailing digits, a single separator can be a
        # decimal mark or a thousands separator. Leave that case unknown.
        if len(fraction) == 3 and len(whole.lstrip("+-")) <= 3:
            return "ambiguous_decimal_or_group_separator", None
        decimal_text = whole + "." + fraction
        try:
            return "parsed", Fraction(Decimal(decimal_text))
        except (InvalidOperation, ValueError, ZeroDivisionError):
            return "invalid_decimal", None

    if re.fullmatch(r"[+-]?\d+[.,]\d+", value):
        return "ambiguous_decimal_or_group_separator", None
    if re.search(r"\d[,.]\d", value):
        return "unsupported_grouping_or_multiple_numbers", None
    if re.search(r"\d", value):
        return "invalid_numeric_suffix", None
    return "invalid_numeric_suffix", None


def _last_nonempty_line(text: str) -> str:
    value = str(text or "")
    for marker in STOP_MARKERS:
        value = value.replace(marker, "")
    return next((line.strip() for line in reversed(value.splitlines()) if line.strip()), "")


def parse_completed_answer(
    generated_text: str,
    *,
    language: str,
    finish_reason: str | None,
    truncated: bool,
) -> Dict[str, Any]:
    """Extract one scalar from the completed final line after ``</think>``."""

    _require(language in LANGUAGES, f"unsupported language: {language}")
    if truncated or finish_reason == "length":
        return {
            "parser_status": "truncated_output",
            "segment_status": "truncated",
            "final_line": None,
            "marker": None,
            "format_status": "not_scored",
            "native_marker_present": False,
            "prediction_canonical": None,
        }
    if finish_reason not in COMPLETION_REASONS:
        return {
            "parser_status": "unknown_finish_reason",
            "segment_status": "unknown_finish_reason",
            "final_line": None,
            "marker": None,
            "format_status": "not_scored",
            "native_marker_present": False,
            "prediction_canonical": None,
        }
    if "</think>" not in str(generated_text or ""):
        return {
            "parser_status": "missing_think_close",
            "segment_status": "missing_think_close",
            "final_line": None,
            "marker": None,
            "format_status": "not_scored",
            "native_marker_present": False,
            "prediction_canonical": None,
        }

    answer_segment = str(generated_text).split("</think>", 1)[1]
    final_line = _last_nonempty_line(answer_segment)
    if not final_line:
        return {
            "parser_status": "missing_final_line",
            "segment_status": "post_think_final_line",
            "final_line": "",
            "marker": None,
            "format_status": "missing",
            "native_marker_present": False,
            "prediction_canonical": None,
        }

    normalized_line = unicodedata.normalize("NFKC", final_line)
    markers = list(ANSWER_MARKER_RE.finditer(normalized_line))
    if markers:
        marker_match = markers[-1]
        marker = marker_match.group("label")
        suffix = normalized_line[marker_match.end() :].strip()
        native = (
            (language == "fr" and marker.casefold().startswith("réponse"))
            or (language == "zh" and marker in {"答案", "最终答案"})
        )
        format_status = "native_answer_marker" if native else "english_or_non_native_marker"
        native_marker_present = bool(native)
    else:
        marker = None
        suffix = normalized_line
        format_status = "standalone_scalar_without_answer_marker"
        native_marker_present = False

    scalar_status, scalar = _parse_scalar(suffix)
    if scalar is not None:
        return {
            "parser_status": "parsed",
            "segment_status": "post_think_final_line",
            "final_line": final_line,
            "marker": marker,
            "format_status": format_status,
            "native_marker_present": native_marker_present,
            "scalar_status": scalar_status,
            "prediction_canonical": _canonical_fraction(scalar),
        }
    normalized_suffix = _normalize_scalar_text(suffix)
    numeric_tokens = re.findall(r"[+-]?\d+(?:[.,]\d+)?", normalized_suffix)
    if len(numeric_tokens) > 1 or re.search(r"\b(?:or|ou|either|versus)\b|或|或者|还是", normalized_suffix, re.IGNORECASE):
        status = "ambiguous_multiple_values"
    elif scalar_status == "ambiguous_decimal_or_group_separator":
        status = "ambiguous_numeric_format"
    else:
        status = scalar_status
    return {
        "parser_status": status,
        "segment_status": "post_think_final_line",
        "final_line": final_line,
        "marker": marker,
        "format_status": format_status,
        "native_marker_present": native_marker_present,
        "scalar_status": scalar_status,
        "prediction_canonical": None,
    }


def score_record(raw: Mapping[str, Any], *, language: str, candidate: str) -> Dict[str, Any]:
    """Rescore one retained raw row without consulting the gold during parsing."""

    _require(str(raw.get("language")) == language, "raw row language does not match its input path")
    _require(str(raw.get("candidate")) == candidate, "raw row candidate does not match its input path")
    _require("exact_match" in raw, "raw row is missing its original exact_match field")
    parser = parse_completed_answer(
        str(raw.get("decoded_generated_text", "")),
        language=language,
        finish_reason=raw.get("finish_reason"),
        truncated=bool(raw.get("truncated")),
    )
    prediction = parser["prediction_canonical"]
    numeric_em: bool | None
    if parser["parser_status"] in {"truncated_output", "unknown_finish_reason", "missing_think_close"}:
        numeric_em = False
        outcome = "incorrect_incomplete_output"
    elif prediction is None:
        numeric_em = None
        outcome = "unknown_unparseable_or_ambiguous"
    else:
        # Gold is read only after the format-only parser has produced a scalar.
        try:
            gold_integer = int(str(raw["ground_truth_answer"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ScoringError(f"parsed record {raw.get('record_id')} lacks an integer gold answer") from exc
        numeric_em = Fraction(prediction) == Fraction(gold_integer, 1)
        outcome = "correct" if numeric_em else "incorrect_numeric_value"
    raw_canonical = _canonical_json(dict(raw))
    return {
        "schema_version": 1,
        "parser_version": PARSER_VERSION,
        "record_id": str(raw.get("record_id", "")),
        "language": language,
        "candidate": candidate,
        "source_raw_row_canonical_sha256": hashlib.sha256(raw_canonical.encode("utf-8")).hexdigest(),
        "strict_original_em": bool(raw["exact_match"]),
        "strict_original_parser_status": str(raw.get("answer_parse", {}).get("status", "missing")),
        "ground_truth_answer": str(raw.get("ground_truth_answer", "")),
        **parser,
        "new_numeric_em": numeric_em,
        "counted_new_numeric_correct": numeric_em is True,
        "new_numeric_em_outcome": outcome,
        "truncated": bool(raw.get("truncated")),
        "finish_reason": raw.get("finish_reason"),
        "request_seed": raw.get("request_seed"),
        # These are copied unchanged from the generation row.
        "reasoning_language_id": raw.get("reasoning_language_id"),
        "reasoning_english_chunk_diagnostic": raw.get("reasoning_english_chunk_diagnostic"),
        "answer_language_id": raw.get("answer_language_id"),
    }


def _read_raw_rows(path: Path, language: str, candidate: str, expected_items: int) -> Tuple[List[Dict[str, Any]], str]:
    _require(path.is_file(), f"missing raw generation file: {path}")
    file_hash = sha256_file(path)
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ScoringError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
            record_id = str(raw.get("record_id", ""))
            _require(record_id and record_id not in seen, f"missing or duplicate record ID in {path}:{line_number}")
            seen.add(record_id)
            rows.append(raw)
    _require(len(rows) == expected_items, f"{path} has {len(rows)} rows, expected {expected_items}")
    return rows, file_hash


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    fraction = position - low
    return float(sorted_values[low] * (1 - fraction) + sorted_values[high] * fraction)


def _paired_bootstrap(values: Sequence[float], *, seed: int, n_resamples: int) -> Dict[str, Any]:
    _require(bool(values), "paired bootstrap needs at least one record")
    rng = random.Random(seed)
    n = len(values)
    observed = sum(values) / n
    means = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples)]
    means.sort()
    return {
        "mean_difference": observed,
        "ci95_percentile": [_percentile(means, 0.025), _percentile(means, 0.975)],
        "n_paired_records": n,
        "bootstrap_resamples": n_resamples,
        "bootstrap_seed": seed,
    }


def _language_summary(rows: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    values = list(rows.values())
    n = len(values)
    new_correct = sum(row["new_numeric_em"] is True for row in values)
    original_correct = sum(bool(row["strict_original_em"]) for row in values)
    statuses = Counter(str(row["parser_status"]) for row in values)
    format_statuses = Counter(str(row["format_status"]) for row in values)
    native_marker_count = sum(bool(row["native_marker_present"]) for row in values)
    lid_values = [row.get("reasoning_language_id") or {} for row in values]
    lid_evaluated = [row for row in lid_values if row.get("status") == "evaluated"]
    lid_target = sum(row.get("target_language_top1") is True for row in lid_evaluated)
    chunk_values = [row.get("reasoning_english_chunk_diagnostic") or {} for row in values]
    chunk_evaluated = [row for row in chunk_values if row.get("status") == "evaluated"]
    chunk_english = sum(row.get("any_english_chunk") is True for row in chunk_evaluated)
    return {
        "n_records": n,
        "strict_original_em": {
            "correct": original_correct,
            "denominator": n,
            "accuracy_all_records": original_correct / n,
        },
        "new_numeric_em": {
            "correct": new_correct,
            "denominator": n,
            "accuracy_all_records": new_correct / n,
            "unknown_count": sum(row["new_numeric_em"] is None for row in values),
            "incorrect_count": n - new_correct - sum(row["new_numeric_em"] is None for row in values),
        },
        "parser_status_counts": dict(sorted(statuses.items())),
        "format_status_counts": dict(sorted(format_statuses.items())),
        "native_answer_marker_count": native_marker_count,
        "native_answer_marker_rate_all_records": native_marker_count / n,
        "truncated_count": sum(bool(row.get("truncated")) for row in values),
        "unchanged_reasoning_language_id": {
            "target_top1_count": lid_target,
            "unknown_count": n - len(lid_evaluated),
            "denominator_all_records": n,
            "target_top1_rate_all_records": lid_target / n,
            "target_top1_rate_evaluable": lid_target / len(lid_evaluated) if lid_evaluated else None,
        },
        "unchanged_english_chunk_diagnostic": {
            "any_english_chunk_count": chunk_english,
            "unknown_count": n - len(chunk_evaluated),
            "denominator_all_records": n,
            "any_english_chunk_rate_all_records": chunk_english / n,
        },
    }


def _load_generation_config(path: Path) -> Tuple[Dict[str, Any], str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScoringError(f"could not read generation protocol config {path}: {exc}") from exc
    _require(isinstance(payload, dict), "generation protocol config must be an object")
    return payload, sha256_file(path)


def _load_raw_audit(path: Path, *, input_root: Path, expected_items_per_candidate: int) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], str]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScoringError(f"could not read raw-output audit report {path}: {exc}") from exc
    _require(isinstance(report, dict) and report.get("status") == "PASS", "raw-output audit report must have status PASS")
    expected_total = expected_items_per_candidate * len(LANGUAGES) * len(CANDIDATES)
    _require(report.get("total_rows") == expected_total, f"raw-output audit total_rows must equal {expected_total}")
    source_files = report.get("source_files")
    expected_keys = {
        f"{language}/{candidate}/generations.jsonl"
        for language in LANGUAGES
        for candidate in CANDIDATES
    }
    _require(isinstance(source_files, dict) and set(source_files) == expected_keys, "raw-output audit source_files keys are incomplete or unexpected")
    normalized: Dict[str, Dict[str, Any]] = {}
    for key, receipt in source_files.items():
        _require(isinstance(receipt, dict), f"raw-output audit receipt for {key} must be an object")
        _require(set(receipt) == {"sha256", "n_records"}, f"raw-output audit receipt for {key} has unexpected fields")
        digest = receipt.get("sha256")
        _require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None, f"raw-output audit SHA-256 for {key} is invalid")
        _require(receipt.get("n_records") == expected_items_per_candidate, f"raw-output audit record count for {key} is invalid")
        normalized[key] = {"sha256": digest, "n_records": expected_items_per_candidate}
    return report, normalized, sha256_file(path)


def score_outputs(
    *,
    input_root: Path,
    output_root: Path,
    raw_audit_report: Path,
    approved_parser_sha256: str,
    config_path: Path = DEFAULT_CONFIG,
    expected_items_per_candidate: int = 250,
    bootstrap_resamples: int | None = None,
) -> Dict[str, Any]:
    """Score all six retained outputs after an exact parser-hash approval."""

    actual_parser_hash = sha256_file(Path(__file__))
    _require(
        approved_parser_sha256.lower() == actual_parser_hash,
        "parser SHA-256 approval does not match this scorer; review/freeze this exact file first",
    )
    _require(not output_root.exists(), f"refusing to overwrite scoring output: {output_root}")
    audit, audited_sources, audit_hash = _load_raw_audit(
        raw_audit_report,
        input_root=input_root,
        expected_items_per_candidate=expected_items_per_candidate,
    )
    protocol, protocol_hash = _load_generation_config(config_path)
    seed = int(protocol["metrics"]["bootstrap_seed"])
    resamples = int(bootstrap_resamples or protocol["metrics"]["bootstrap_resamples"])
    _require(resamples > 0, "bootstrap resamples must be positive")
    scored: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    source_receipts: Dict[str, Dict[str, Any]] = {}

    for language in LANGUAGES:
        scored[language] = {}
        for candidate in CANDIDATES:
            raw_path = input_root / language / candidate / "generations.jsonl"
            raw_rows, raw_hash = _read_raw_rows(raw_path, language, candidate, expected_items_per_candidate)
            receipt_key = f"{language}/{candidate}/generations.jsonl"
            _require(raw_hash == audited_sources[receipt_key]["sha256"], f"raw source hash differs from passing audit receipt: {receipt_key}")
            candidate_rows = [score_record(raw, language=language, candidate=candidate) for raw in raw_rows]
            by_id = {str(row["record_id"]): row for row in candidate_rows}
            _require(len(by_id) == expected_items_per_candidate, f"duplicate scored IDs in {language}/{candidate}")
            scored[language][candidate] = by_id
            source_receipts[f"{language}/{candidate}"] = {
                "path": str(raw_path.resolve()),
                "sha256": raw_hash,
                "n_records": len(raw_rows),
            }

    language_reports: Dict[str, Any] = {}
    comparisons_to_make = (
        ("selected_minus_host", "selected", "host_baseline"),
        ("published_minus_host", "published_reference", "host_baseline"),
        ("selected_minus_published", "selected", "published_reference"),
    )
    for language in LANGUAGES:
        ids = {candidate: set(scored[language][candidate]) for candidate in CANDIDATES}
        host_ids = ids["host_baseline"]
        _require(all(candidate_ids == host_ids for candidate_ids in ids.values()), f"candidate record IDs differ for {language}")
        ordered_ids = sorted(host_ids)
        for record_id in ordered_ids:
            golds = {str(scored[language][candidate][record_id]["ground_truth_answer"]) for candidate in CANDIDATES}
            _require(len(golds) == 1, f"candidate gold records differ for {language}/{record_id}")
        candidates_summary = {
            candidate: _language_summary(scored[language][candidate]) for candidate in CANDIDATES
        }
        comparisons: Dict[str, Any] = {}
        for name, left, right in comparisons_to_make:
            seed_for_pair = seed + int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "big")
            new_diff = [
                float(scored[language][left][record_id]["new_numeric_em"] is True)
                - float(scored[language][right][record_id]["new_numeric_em"] is True)
                for record_id in ordered_ids
            ]
            original_diff = [
                float(scored[language][left][record_id]["strict_original_em"])
                - float(scored[language][right][record_id]["strict_original_em"])
                for record_id in ordered_ids
            ]
            lid_diff = [
                float((scored[language][left][record_id].get("reasoning_language_id") or {}).get("target_language_top1") is True)
                - float((scored[language][right][record_id].get("reasoning_language_id") or {}).get("target_language_top1") is True)
                for record_id in ordered_ids
            ]
            english_diff = [
                float((scored[language][left][record_id].get("reasoning_english_chunk_diagnostic") or {}).get("any_english_chunk") is True)
                - float((scored[language][right][record_id].get("reasoning_english_chunk_diagnostic") or {}).get("any_english_chunk") is True)
                for record_id in ordered_ids
            ]
            comparisons[name] = {
                "new_numeric_em": _paired_bootstrap(new_diff, seed=seed_for_pair, n_resamples=resamples),
                "strict_original_em": _paired_bootstrap(original_diff, seed=seed_for_pair, n_resamples=resamples),
                "unchanged_reasoning_lid": _paired_bootstrap(lid_diff, seed=seed_for_pair + 1, n_resamples=resamples),
                "unchanged_english_chunk_diagnostic": _paired_bootstrap(english_diff, seed=seed_for_pair + 2, n_resamples=resamples),
            }
        language_reports[language] = {
            "candidate_summaries": candidates_summary,
            "paired_comparisons": comparisons,
        }

    _require(sha256_file(raw_audit_report) == audit_hash, "raw-output audit report changed during rescoring")
    for key, receipt in audited_sources.items():
        _require(
            sha256_file(input_root / Path(key)) == receipt["sha256"],
            f"raw source changed after audit validation: {key}",
        )

    # Do not create outputs until every source file and pairing check passes.
    for language in LANGUAGES:
        for candidate in CANDIDATES:
            target = output_root / language / candidate / "scored.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", encoding="utf-8", newline="\n") as handle:
                rows = scored[language][candidate]
                for record_id in sorted(rows):
                    handle.write(_canonical_json(rows[record_id]) + "\n")
            source_receipts[f"{language}/{candidate}"]["scored_path"] = str(target.resolve())
            source_receipts[f"{language}/{candidate}"]["scored_sha256"] = sha256_file(target)

    report = {
        "schema_version": 1,
        "status": "PASS_POSTHOC_FORMAT_ROBUST_RESCORE",
        "study": "b3_generation_followup_v1",
        "scoring_type": "post-hoc parser correction; original strict EM is retained unchanged alongside new numeric EM",
        "parser": {
            "version": PARSER_VERSION,
            "sha256": actual_parser_hash,
            "approval_sha256": approved_parser_sha256.lower(),
            "approval_matches_frozen_file": True,
            "basis": "host-format QA examples and synthetic fixtures; candidate comparisons were not used to select the parser",
            "preregistration": "not externally preregistered; this post-hoc parser correction was frozen before comparative rescoring",
            "rule": "completed non-truncated post-</think> final nonempty line; last explicit native/English marker or standalone scalar; exact Fraction/Decimal comparison",
        },
        "generation_protocol_config": {"path": str(config_path.resolve()), "sha256": protocol_hash},
        "raw_output_audit": {
            "path": str(raw_audit_report.resolve()),
            "sha256": audit_hash,
            "status": audit["status"],
            "total_rows": audit["total_rows"],
        },
        "source_files": source_receipts,
        "bootstrap": {
            "method": "paired record bootstrap with replacement, matching the original deterministic seed derivation",
            "seed": seed,
            "resamples": resamples,
            "accuracy_denominator": "all records; unknown, ambiguous, unparseable, incomplete, and truncated cases count as not correct",
        },
        "languages": language_reports,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "score_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--raw-audit-report", type=Path, required=True)
    parser.add_argument("--approved-parser-sha256", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = score_outputs(
            input_root=args.input_root,
            output_root=args.output_root,
            raw_audit_report=args.raw_audit_report,
            approved_parser_sha256=args.approved_parser_sha256,
            config_path=args.config,
        )
    except ScoringError as exc:
        print(f"STOP_B3_GENERATION_SCORING: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
