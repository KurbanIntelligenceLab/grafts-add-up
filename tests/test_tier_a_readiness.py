"""Unit tests for fail-closed Tier-A expert readiness helpers."""

from __future__ import annotations

import unittest

from suture.tier_a_readiness import _number_matches, readiness_gate


class TierAReadinessTests(unittest.TestCase):
    def test_number_match_uses_final_answer_number(self) -> None:
        self.assertTrue(_number_matches("The answer is 12. #### 7", "7"))
        self.assertFalse(_number_matches("The answer is 12.", "7"))

    def test_number_match_handles_decimal_answers(self) -> None:
        self.assertTrue(_number_matches("Result: 3.50", "3.5"))
        self.assertFalse(_number_matches("No numeric answer", "0"))

    def test_readiness_gate_outcomes(self) -> None:
        self.assertEqual(
            readiness_gate(
                donor_gain=0.03125,
                host_nonempty_rate=1.0,
                host_target_language_rate=1.0,
                min_nonempty_rate=0.9,
                min_target_language_rate=0.5,
            ),
            "PASS_READINESS",
        )
        self.assertEqual(
            readiness_gate(
                donor_gain=0.0,
                host_nonempty_rate=1.0,
                host_target_language_rate=1.0,
                min_nonempty_rate=0.9,
                min_target_language_rate=0.5,
            ),
            "STOP_EXPERT_READINESS",
        )
        self.assertEqual(
            readiness_gate(
                donor_gain=0.1,
                host_nonempty_rate=0.5,
                host_target_language_rate=1.0,
                min_nonempty_rate=0.9,
                min_target_language_rate=0.5,
            ),
            "STOP_EXPERT_READINESS",
        )
        self.assertEqual(
            readiness_gate(
                donor_gain=0.1,
                host_nonempty_rate=1.0,
                host_target_language_rate=0.2,
                min_nonempty_rate=0.9,
                min_target_language_rate=0.5,
            ),
            "STOP_EXPERT_READINESS",
        )


if __name__ == "__main__":
    unittest.main()
