"""Unit tests for item-aligned Tier-A statistics."""

from __future__ import annotations

import unittest

from suture.tier_a_stats import analyze_canonical_run, holm_adjust, paired_bootstrap


class TierAStatsTests(unittest.TestCase):
    def test_paired_bootstrap_preserves_item_alignment(self) -> None:
        result = paired_bootstrap(
            {"a": 0.0, "b": 1.0, "c": 0.0},
            {"a": 1.0, "b": 1.0, "c": 0.0},
            repetitions=200,
            seed=4,
        )
        self.assertAlmostEqual(result["observed_regret"], 1 / 3)
        self.assertEqual(result["n_items"], 3)

    def test_holm_adjustment_is_monotonic(self) -> None:
        result = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.2})
        self.assertLessEqual(result["a"], result["b"])
        self.assertLessEqual(result["b"], result["c"])

    def test_run_does_not_average_p_values_across_missing_seeds(self) -> None:
        from suture.tier_a_stats import run as stats_run
        from pathlib import Path

        root = Path("results/tier_a")
        if not (root / "es" / "e1_canonical_seed0" / "e1_gate.json").is_file():
            self.skipTest("canonical Spanish E1 artifacts are absent")
        summary = stats_run(root=root, languages=("es",), expert_seeds=(0,))
        self.assertFalse(summary["p_values_averaged_across_seeds"])
        self.assertEqual(summary["results"]["es"][0]["expert_seeds_measured"], [0])


    def test_analyze_refuses_to_treat_retention_as_capability(self) -> None:
        from pathlib import Path

        canonical = Path("results/tier_a/es/e1_canonical_seed0")
        if not (canonical / "e1_gate.json").is_file():
            self.skipTest("canonical Spanish E1 artifacts are absent")
        summary = analyze_canonical_run(canonical, repetitions=50, seed=0)
        self.assertFalse(summary["p_values_averaged_across_seeds"])
        self.assertEqual(summary["c_set"]["certificate"]["n_calibration"], 2500)
        self.assertGreater(
            summary["c_set"]["certificate"]["upper_bound"],
            summary["c_set"]["selected_empirical_rate"],
        )
        nonempty = summary["target_language_nonempty_check"]
        self.assertTrue(nonempty["not_general_capability_retention"])
        self.assertEqual(nonempty["belebele_or_knowledge"], "unmeasured")
        self.assertEqual(
            nonempty["measured_name"],
            "target_language_nonempty_check",
        )
        self.assertEqual(
            summary["retention"]["alias_of"],
            "target_language_nonempty_check",
        )


if __name__ == "__main__":
    unittest.main()
