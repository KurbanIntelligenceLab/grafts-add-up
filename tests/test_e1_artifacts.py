"""Read-only integrity checks for Spanish canonical E1 artifacts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from suture.suture_metrics import certify_drift
from suture.tier_a_gate import (
    _answer_match,
    _contiguous_windows,
    _seed_generation,
    _write_immutable_json,
    load_protocol_v2_sweep,
)
from suture.tier_a_stats import analyze_canonical_run, paired_bootstrap


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "results" / "tier_a" / "es" / "e1_canonical_seed0"


class AnswerMatchTests(unittest.TestCase):
    def test_uses_the_final_number(self) -> None:
        self.assertTrue(_answer_match("scratch 12 then 7", "7"))
        self.assertFalse(_answer_match("the answer is 12", "7"))
        self.assertTrue(_answer_match("3.50", "3.5"))
        self.assertFalse(_answer_match("no number here", "0"))
        self.assertTrue(_answer_match("1,000", "1000"))


class ImmutableCheckpointTests(unittest.TestCase):
    def test_identical_rewrite_is_a_no_op_and_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "e1_config.json"
            payload = {"protocol_version": 2, "language": "es"}
            _write_immutable_json(path, payload)
            _write_immutable_json(path, payload)
            with self.assertRaises(Exception):
                _write_immutable_json(path, {"protocol_version": 2, "language": "zh"})


class CanonicalE1ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (CANONICAL / "e1_gate.json").is_file():
            raise unittest.SkipTest("canonical Spanish E1 artifacts are absent")
        cls.gate = json.loads((CANONICAL / "e1_gate.json").read_text(encoding="utf-8"))
        cls.sweep = load_protocol_v2_sweep(CANONICAL / "e1_window_sweep.jsonl")
        cls.config = json.loads((CANONICAL / "e1_config.json").read_text(encoding="utf-8"))

    def test_exhaustive_406_windows_and_five_decode_seeds(self) -> None:
        declared = _contiguous_windows(int(self.gate["n_layers"]))
        self.assertEqual(len(declared), 406)
        self.assertEqual(int(self.gate["n_windows"]), 406)
        self.assertEqual(len(self.sweep), 406)
        self.assertEqual(sorted(self.sweep), declared)
        selected = tuple(self.gate["suture_selected"]["graft"])
        best = tuple(range(self.gate["sweep_best_window"][0], self.gate["sweep_best_window"][1] + 1))
        self.assertIn((selected[0], selected[-1]), self.sweep)
        self.assertIn((best[0], best[-1]), self.sweep)
        row = self.sweep[(selected[0], selected[-1])]
        items = row["generation"]["items"]
        self.assertEqual(len(items), 8 * 5)
        source_ids = sorted({item["source_id"] for item in items})
        self.assertEqual(len(source_ids), 8)
        self.assertEqual(self.config["decoding"]["seeds"], [0, 1, 2, 3, 4])
        self.assertEqual(self.gate["selected_certificate"]["decode_seeds"], [0, 1, 2, 3, 4])
        self.assertEqual(self.gate["selection_merged_models_built"], 0)
        self.assertTrue(self.gate["sweep_is_exhaustive"])
        self.assertEqual(self.gate["gate"], "STOP_E2_REGIME_INVESTIGATION")
        self.assertEqual(
            self.gate["ranking_prediction_source"],
            "frozen_P_util_and_P_risk",
        )
        self.assertEqual(
            self.gate["measured_objective"],
            "free_generation_exact_match_on_MGSM_dev",
        )

    def test_resume_index_is_idempotent(self) -> None:
        again = load_protocol_v2_sweep(CANONICAL / "e1_window_sweep.jsonl")
        self.assertEqual(sorted(again), sorted(self.sweep))
        self.assertEqual(self.gate["sweep_windows_completed"], 406)

    def test_seed_helper_is_deterministic(self) -> None:
        _seed_generation(4)
        first = int(__import__("random").random() * 1_000_000)
        _seed_generation(4)
        second = int(__import__("random").random() * 1_000_000)
        self.assertEqual(first, second)


class CanonicalStatsSemanticsTests(unittest.TestCase):
    def test_single_run_bootstrap_does_not_average_missing_seeds(self) -> None:
        if not (CANONICAL / "e1_gate.json").is_file():
            self.skipTest("canonical Spanish E1 artifacts are absent")
        summary = analyze_canonical_run(CANONICAL, repetitions=200, seed=0)
        self.assertEqual(summary["expert_seeds_measured"], [0])
        self.assertFalse(summary["p_values_averaged_across_seeds"])
        self.assertEqual(summary["bootstrap"]["n_items"], 8)
        self.assertEqual(
            summary["retention"]["measured_name"],
            "target_language_nonempty_check",
        )
        self.assertEqual(summary["retention"]["belebele_or_knowledge"], "unmeasured")
        self.assertEqual(summary["c_set"]["status"], "empirical_plus_clopper_pearson")
        self.assertIn("upper_bound", summary["c_set"]["certificate"])
        self.assertEqual(
            summary["timing"]["selection_timing_status"],
            "unknown_after_resume",
        )
        self.assertEqual(
            summary["same_probe_versus_cross_set"]["headline_ranking"],
            "cross_set_P_util_P_risk_to_MGSM_dev",
        )
        self.assertFalse(
            summary["same_probe_versus_cross_set"]["cross_set_failure_is_bookkeeping"]
        )

    def test_certify_drift_is_not_the_raw_rate(self) -> None:
        certificate = certify_drift([0, 1, 1, 0, 0], delta=0.05)
        self.assertGreater(certificate["upper_bound"], certificate["empirical_rate"])
        self.assertAlmostEqual(certificate["empirical_rate"], 0.4)

    def test_bootstrap_item_ids_are_aligned(self) -> None:
        result = paired_bootstrap(
            {"x": 0.0, "y": 1.0},
            {"y": 1.0, "x": 1.0, "z": 0.0},
            repetitions=50,
            seed=1,
        )
        self.assertEqual(result["item_ids"], ["x", "y"])
        self.assertAlmostEqual(result["observed_regret"], 0.5)


if __name__ == "__main__":
    unittest.main()
