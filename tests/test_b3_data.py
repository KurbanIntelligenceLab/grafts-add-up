"""Offline tests for B3 probe-source and record construction."""

from __future__ import annotations

import unittest

from suture.b3_data import (
    LANGUAGE_LABELS,
    _answer_number,
    _clean_translation,
    _measurement_records,
    _risk_records,
    _utility_records,
)


class _NoChatTemplateTokenizer:
    chat_template = None


class B3DataTests(unittest.TestCase):
    def test_answer_parser_is_strict_and_uses_final_marker(self) -> None:
        self.assertEqual(_answer_number("work\n#### 17"), "17")
        self.assertEqual(_answer_number("#### 2\n#### -3.5"), "-3.5")
        with self.assertRaises(Exception):
            _answer_number("no final answer")

    def test_translation_cleanup_removes_wrappers(self) -> None:
        self.assertEqual(_clean_translation("```text\nBonjour\n```"), "Bonjour")
        self.assertEqual(_clean_translation("Translation: Bonjour"), "Bonjour")

    def test_records_use_frozen_language_readouts_and_disjoint_ids(self) -> None:
        tokenizer = _NoChatTemplateTokenizer()
        translation_spec = {"id": "translator", "revision": "a" * 40}
        utility = _utility_records(
            [
                {
                    "source_id": "gsm8k:0",
                    "source_index": 0,
                    "source_question": "How many?",
                    "answer_number": "17",
                }
            ],
            ["Combien ?"],
            tokenizer,
            "fr",
            translation_spec,
        )
        risk = _risk_records(
            [
                {
                    "source_id": "mrb:0",
                    "source_index": 0,
                    "source_subset": "translation-en-fr",
                    "source_text": "Solve this.",
                }
            ],
            ["Résous ceci."],
            tokenizer,
            "fr",
            translation_spec,
        )
        self.assertEqual(LANGUAGE_LABELS["fr"], "français")
        self.assertEqual(utility[0]["utility_texts"], ["17"])
        self.assertEqual(risk[0]["risk_target_texts"], ["français"])
        self.assertEqual(risk[0]["risk_donor_texts"], ["English"])
        self.assertTrue({utility[0]["id"]}.isdisjoint({risk[0]["id"]}))

    def test_measurement_records_use_numeric_answer_targets(self) -> None:
        records = _measurement_records(
            [
                {
                    "source_id": "mgsm-rev2:fr:test:0",
                    "source_index": 0,
                    "question": "Combien font deux plus deux ?",
                    "answer_number": "4",
                }
            ],
            _NoChatTemplateTokenizer(),
            "fr",
            {
                "dataset": "lightonai/mgsm-rev2",
                "revision": "a" * 40,
            },
        )
        self.assertEqual(records[0]["answer_number"], "4")
        self.assertEqual(records[0]["dataset"], "lightonai/mgsm-rev2")
        self.assertIn("Combien font deux plus deux ?", records[0]["input_text"])


if __name__ == "__main__":
    unittest.main()
