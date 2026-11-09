"""Synthetic parser and paired-rescoring fixtures for B3 generation outputs."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from suture import b3_generation_scoring as scoring


def _raw_row(
    *,
    text: str,
    language: str = "fr",
    candidate: str = "host_baseline",
    gold: str = "14",
    finish_reason: str | None = "stop",
    truncated: bool = False,
    original_em: bool = False,
) -> dict:
    return {
        "record_id": "item-0",
        "language": language,
        "candidate": candidate,
        "ground_truth_answer": gold,
        "decoded_generated_text": text,
        "finish_reason": finish_reason,
        "truncated": truncated,
        "exact_match": original_em,
        "answer_parse": {"status": "invalid_integer_answer"},
        "request_seed": 31,
        "reasoning_language_id": {
            "status": "evaluated",
            "target_language_top1": language == "fr",
            "probability": 0.91,
        },
        "reasoning_english_chunk_diagnostic": {
            "status": "evaluated",
            "any_english_chunk": False,
        },
        "answer_language_id": {"status": "insufficient_text", "target_language_top1": None},
    }


class B3GenerationScoringTests(unittest.TestCase):
    def test_prose_before_last_native_marker_parses_only_the_suffix(self):
        result = scoring.parse_completed_answer(
            "Let me calculate. </think> 37 - 23 = 14. Réponse : 14.",
            language="fr",
            finish_reason="stop",
            truncated=False,
        )
        self.assertEqual(result["parser_status"], "parsed")
        self.assertEqual(result["prediction_canonical"], "14")
        self.assertTrue(result["native_marker_present"])

    def test_last_explicit_english_marker_wins_and_format_status_is_separate(self):
        result = scoring.parse_completed_answer(
            "</think> Work says Answer: 12, then final result: Final answer : 14.",
            language="fr",
            finish_reason="stop",
            truncated=False,
        )
        self.assertEqual(result["prediction_canonical"], "14")
        self.assertEqual(result["format_status"], "english_or_non_native_marker")
        self.assertFalse(result["native_marker_present"])

    def test_decimal_fraction_comparison_is_exact_without_integer_coercion(self):
        whole = scoring.score_record(
            _raw_row(text="</think> Réponse : 57.00.", gold="57"),
            language="fr",
            candidate="host_baseline",
        )
        decimal = scoring.score_record(
            _raw_row(text="</think> Answer: 2,4", gold="2"),
            language="fr",
            candidate="host_baseline",
        )
        fraction = scoring.score_record(
            _raw_row(text="</think> Réponse: -3/4", gold="-1"),
            language="fr",
            candidate="host_baseline",
        )
        self.assertTrue(whole["new_numeric_em"])
        self.assertEqual(whole["prediction_canonical"], "57")
        self.assertFalse(decimal["new_numeric_em"])
        self.assertEqual(decimal["prediction_canonical"], "12/5")
        self.assertFalse(fraction["new_numeric_em"])
        self.assertEqual(fraction["prediction_canonical"], "-3/4")

    def test_ambiguous_grouping_and_multiple_values_are_unknown(self):
        grouped = scoring.parse_completed_answer(
            "</think> Réponse: 1,234", language="fr", finish_reason="stop", truncated=False
        )
        alternatives = scoring.parse_completed_answer(
            "</think> Answer: 14 or 15", language="fr", finish_reason="stop", truncated=False
        )
        bare_ambiguous = scoring.parse_completed_answer(
            "</think> 14 and 15", language="fr", finish_reason="stop", truncated=False
        )
        self.assertEqual(grouped["parser_status"], "ambiguous_numeric_format")
        self.assertIsNone(grouped["prediction_canonical"])
        self.assertEqual(alternatives["parser_status"], "ambiguous_multiple_values")
        self.assertEqual(bare_ambiguous["parser_status"], "ambiguous_multiple_values")

    def test_operations_units_and_prose_suffixes_are_not_scanned_for_numbers(self):
        operation = scoring.parse_completed_answer(
            "</think> Réponse: 37 - 23 = 14", language="fr", finish_reason="stop", truncated=False
        )
        unit = scoring.parse_completed_answer(
            "</think> Réponse: 14 dollars", language="fr", finish_reason="stop", truncated=False
        )
        no_marker_prose = scoring.parse_completed_answer(
            "</think> The answer is approximately 14", language="fr", finish_reason="stop", truncated=False
        )
        self.assertIsNone(operation["prediction_canonical"])
        self.assertIsNone(unit["prediction_canonical"])
        self.assertIsNone(no_marker_prose["prediction_canonical"])

    def test_unicode_minus_markdown_wrappers_and_standalone_scalar(self):
        boxed = scoring.parse_completed_answer(
            "</think> **\\boxed{−14.00}.**", language="fr", finish_reason="stop", truncated=False
        )
        bare = scoring.score_record(
            _raw_row(text="</think> `14`", original_em=False),
            language="fr",
            candidate="host_baseline",
        )
        self.assertEqual(boxed["prediction_canonical"], "-14")
        self.assertFalse(boxed["native_marker_present"])
        self.assertTrue(bare["new_numeric_em"])
        self.assertFalse(bare["strict_original_em"])
        self.assertEqual(bare["format_status"], "standalone_scalar_without_answer_marker")
        bold_terminal = scoring.parse_completed_answer(
            "</think> **18**.", language="fr", finish_reason="stop", truncated=False
        )
        parenthesized_box = scoring.parse_completed_answer(
            "</think> (\\boxed{18})", language="fr", finish_reason="stop", truncated=False
        )
        self.assertEqual(bold_terminal["prediction_canonical"], "18")
        self.assertEqual(parenthesized_box["prediction_canonical"], "18")

    def test_truncated_missing_think_and_unknown_finish_are_incorrect(self):
        rows = (
            _raw_row(text="</think> Réponse: 14", truncated=True, finish_reason="length", original_em=True),
            _raw_row(text="Réponse: 14", finish_reason="stop"),
            _raw_row(text="</think> Réponse: 14", finish_reason=None),
        )
        scored = [scoring.score_record(row, language="fr", candidate="host_baseline") for row in rows]
        self.assertTrue(all(item["new_numeric_em"] is False for item in scored))
        self.assertEqual(scored[0]["strict_original_em"], True)
        self.assertEqual(scored[0]["new_numeric_em_outcome"], "incorrect_incomplete_output")
        self.assertEqual(scored[1]["parser_status"], "missing_think_close")
        self.assertEqual(scored[2]["parser_status"], "unknown_finish_reason")

    def test_gold_is_not_consulted_until_a_scalar_has_been_extracted(self):
        ambiguous = _raw_row(text="</think> Réponse: 1,234", gold="not-an-integer")
        scored = scoring.score_record(ambiguous, language="fr", candidate="host_baseline")
        self.assertIsNone(scored["new_numeric_em"])
        parsed = _raw_row(text="</think> Réponse: 14", gold="not-an-integer")
        with self.assertRaises(scoring.ScoringError):
            scoring.score_record(parsed, language="fr", candidate="host_baseline")

    def test_freeze_guard_checks_hash_before_reading_any_raw_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(scoring.ScoringError, "approval does not match"):
                scoring.score_outputs(
                    input_root=Path(temporary) / "missing-inputs",
                    output_root=Path(temporary) / "scores",
                    raw_audit_report=Path(temporary) / "missing-audit.json",
                    approved_parser_sha256="0" * 64,
                )

    def test_score_outputs_writes_separate_sidecars_and_paired_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "raw"
            output_root = root / "scored"
            audit_path = root / "audit_report.json"
            item_ids = ("item-0", "item-1")
            source_files = {}
            for language in scoring.LANGUAGES:
                for candidate_index, candidate in enumerate(scoring.CANDIDATES):
                    path = input_root / language / candidate / "generations.jsonl"
                    path.parent.mkdir(parents=True)
                    lines = []
                    for index, record_id in enumerate(item_ids):
                        good = (index + candidate_index) % 2 == 0
                        gold = str(14 + index)
                        answer = gold if good else str(99 + index)
                        raw = _raw_row(
                            text=f"</think> Résolution. Réponse : {answer}.",
                            language=language,
                            candidate=candidate,
                            gold=gold,
                            original_em=False,
                        )
                        raw["record_id"] = record_id
                        raw["reasoning_language_id"]["target_language_top1"] = language == "fr"
                        lines.append(json.dumps(raw, ensure_ascii=False))
                    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                    source_files[f"{language}/{candidate}/generations.jsonl"] = {
                        "sha256": scoring.sha256_file(path),
                        "n_records": 2,
                    }
            audit_path.write_text(json.dumps({"status": "PASS", "total_rows": 12, "source_files": source_files}), encoding="utf-8")
            bad_audit = root / "bad_audit.json"
            bad_sources = dict(source_files)
            bad_sources["fr/host_baseline/generations.jsonl"] = {"sha256": "0" * 64, "n_records": 2}
            bad_audit.write_text(json.dumps({"status": "PASS", "total_rows": 12, "source_files": bad_sources}), encoding="utf-8")
            with self.assertRaisesRegex(scoring.ScoringError, "differs from passing audit receipt"):
                scoring.score_outputs(
                    input_root=input_root,
                    output_root=root / "bad-scored",
                    raw_audit_report=bad_audit,
                    approved_parser_sha256=scoring.sha256_file(Path(scoring.__file__)),
                    expected_items_per_candidate=2,
                    bootstrap_resamples=20,
                )
            report = scoring.score_outputs(
                input_root=input_root,
                output_root=output_root,
                raw_audit_report=audit_path,
                approved_parser_sha256=scoring.sha256_file(Path(scoring.__file__)),
                expected_items_per_candidate=2,
                bootstrap_resamples=20,
            )
            self.assertEqual(report["status"], "PASS_POSTHOC_FORMAT_ROBUST_RESCORE")
            self.assertEqual(report["raw_output_audit"]["status"], "PASS")
            self.assertEqual(report["languages"]["fr"]["candidate_summaries"]["host_baseline"]["new_numeric_em"]["correct"], 1)
            self.assertEqual(report["languages"]["fr"]["paired_comparisons"]["selected_minus_host"]["new_numeric_em"]["n_paired_records"], 2)
            self.assertTrue((output_root / "fr" / "host_baseline" / "scored.jsonl").is_file())
            self.assertTrue((output_root / "score_report.json").is_file())


if __name__ == "__main__":
    unittest.main()
