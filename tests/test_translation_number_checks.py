"""Focused tests for conservative EN to FR/ZH numeral consistency flags."""

from __future__ import annotations

import unittest

from suture.translation_number_checks import audit_numbers


def _values(mentions: list[dict]) -> list[str]:
    return [str(mention["normalized_value"]) for mention in mentions]


class TranslationNumberChecksTests(unittest.TestCase):
    def test_locale_grouping_decimal_fraction_and_currency(self) -> None:
        result = audit_numbers(
            "$1,234.50 plus 1/2 and 1/2",
            "1\u202f234,50 € plus 1/2 et 1/2",
            "fr",
        )
        self.assertEqual(result["status"], "match")
        self.assertEqual(result["strict_digit_comparison"]["status"], "match")
        self.assertEqual(_values(result["source"]["digit_mentions"]), ["2469/2", "1/2", "1/2"])
        self.assertEqual(_values(result["target"]["digit_mentions"]), ["2469/2", "1/2", "1/2"])
        self.assertEqual(result["source"]["digit_mentions"][0]["marker"], "$")
        self.assertEqual(result["target"]["digit_mentions"][0]["marker"], "€")

    def test_fullwidth_digits_and_grouping_are_normalized(self) -> None:
        result = audit_numbers("1,234.5", "１\u202f２３４,５", "fr")
        self.assertEqual(result["supported_number_comparison"]["status"], "match")
        self.assertEqual(_values(result["target"]["digit_mentions"]), ["2469/2"])

    def test_common_english_and_french_cardinals_normalize(self) -> None:
        result = audit_numbers(
            "one hundred and five, twenty-one, seventy-one",
            "cent cinq, vingt et un, soixante et onze",
            "fr",
        )
        self.assertEqual(result["status"], "match")
        self.assertEqual(result["strict_digit_comparison"]["status"], "match")
        self.assertEqual(
            _values(result["source"]["supported_number_words"]), ["105", "21", "71"]
        )
        self.assertEqual(
            _values(result["target"]["supported_number_words"]), ["105", "21", "71"]
        )

    def test_chinese_number_words_and_percent_normalize(self) -> None:
        result = audit_numbers(
            "There are 2,003 items and 30 percent remain.",
            "共有两千零三项，百分之三十仍保留。",
            "zh",
        )
        self.assertEqual(result["status"], "match")
        self.assertEqual(result["strict_digit_comparison"]["status"], "mismatch")
        self.assertEqual(
            _values(result["source"]["digit_mentions"]), ["2003", "30"]
        )
        self.assertEqual(
            _values(result["target"]["supported_number_words"]), ["2003", "30"]
        )

    def test_multisets_preserve_duplicates_and_changed_values_flag(self) -> None:
        same = audit_numbers("2 plus 2", "deux plus deux", "fr")
        self.assertEqual(same["status"], "match")
        self.assertEqual(same["strict_digit_comparison"]["status"], "mismatch")
        self.assertEqual(
            same["supported_number_comparison"]["source_multiset"],
            [{"normalized_value": "2", "count": 2}],
        )

        changed = audit_numbers("3", "4", "fr")
        self.assertEqual(changed["status"], "mismatch")
        self.assertEqual(
            changed["supported_number_comparison"]["missing_from_target"],
            [{"normalized_value": "3", "count": 1}],
        )
        self.assertEqual(
            changed["supported_number_comparison"]["extra_in_target"],
            [{"normalized_value": "4", "count": 1}],
        )

    def test_articles_and_unsupported_fraction_words_require_review(self) -> None:
        article = audit_numbers("one dog", "un chien", "fr")
        self.assertEqual(article["status"], "review-needed")
        self.assertEqual(article["target"]["supported_number_words"], [])
        self.assertEqual(
            article["target"]["article_candidates_not_counted"][0]["reason"],
            "article_not_counted_as_one",
        )

        fraction_word = audit_numbers("one half", "1/2", "fr")
        self.assertEqual(fraction_word["status"], "review-needed")
        self.assertEqual(
            fraction_word["unhandled_source_spans"][0]["reason"],
            "unsupported_ordinal_or_fraction_word",
        )

    def test_dates_times_versions_and_ranges_are_review_needed(self) -> None:
        dated = audit_numbers(
            "Released 2024-02-29 at 12:30, version v1.2.3.",
            "发行于2024-02-29 12:30，版本v1.2.3。",
            "zh",
        )
        self.assertEqual(dated["status"], "review-needed")
        reasons = {
            reason
            for group in (dated["ambiguous_source_spans"], dated["ambiguous_target_spans"])
            for span in group
            for reason in span["reasons"]
        }
        self.assertIn("iso_or_year_first_date", reasons)
        self.assertIn("time", reasons)
        self.assertIn("version", reasons)

        written_range = audit_numbers("from three to five", "de trois à cinq", "fr")
        self.assertEqual(written_range["status"], "review-needed")
        self.assertTrue(written_range["ambiguous_source_spans"])
        self.assertTrue(written_range["ambiguous_target_spans"])

        zh_range = audit_numbers("3-5", "三到五", "zh")
        self.assertEqual(zh_range["status"], "review-needed")
        self.assertTrue(zh_range["ambiguous_source_spans"])
        self.assertTrue(zh_range["ambiguous_target_spans"])

    def test_invalid_language_and_non_string_inputs_fail_explicitly(self) -> None:
        with self.assertRaises(ValueError):
            audit_numbers("3", "3", "de")
        with self.assertRaises(TypeError):
            audit_numbers("3", 3, "fr")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
