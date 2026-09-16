"""Fail-closed guards for the Qwen3 contract-v2 pin."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from suture.tier_a_config import (
    MODEL_ID,
    MODEL_REVISION,
    ConfigError,
    frozen_mgsm_indices,
    refuse_forbidden_model,
    refuse_v1_write,
    sha256_file,
)
from suture.tier_a_readiness import readiness_gate
from suture.tier_a_stats import classify_v2_panel, holm_adjust
from suture.run_manifest import assert_resume_compatible


class Qwen3PinTests(unittest.TestCase):
    def test_pin_is_exact_qwen3(self) -> None:
        self.assertEqual(MODEL_ID, "Qwen/Qwen3-1.7B")
        self.assertEqual(MODEL_REVISION, "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")

    def test_qwen25_is_refused(self) -> None:
        with self.assertRaises(ConfigError):
            refuse_forbidden_model("Qwen/Qwen2.5-Coder-1.5B-Instruct")
        with self.assertRaises(ConfigError):
            refuse_forbidden_model(MODEL_ID, "2e1fd397ee46e1388853d2af2c993145b0f1098a")

    def test_v1_canonical_write_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical = root / "results" / "tier_a" / "es" / "e1_canonical_seed0"
            canonical.mkdir(parents=True)
            with self.assertRaises(ConfigError):
                refuse_v1_write(canonical / "e1_gate.json", root=root)
            with self.assertRaises(ConfigError):
                refuse_v1_write(root / "results" / "tier_a" / "es" / "new.json", root=root)
            ok = root / "results" / "v2" / "qwen3_1_7b" / "es" / "e1"
            ok.mkdir(parents=True)
            refuse_v1_write(ok / "e1_gate.json", root=root)

    def test_mgsm_split_is_disjoint_and_language_independent(self) -> None:
        split = frozen_mgsm_indices()
        self.assertEqual(len(split["dev"]), 64)
        self.assertEqual(len(split["test"]), 186)
        self.assertEqual(len(set(split["dev"]) | set(split["test"])), 250)
        self.assertFalse(set(split["dev"]) & set(split["test"]))
        again = frozen_mgsm_indices()
        self.assertEqual(split["dev"], again["dev"])
        self.assertEqual(split["test"], again["test"])

    def test_v2_contract_hash_is_stable_and_version_2(self) -> None:
        path = Path("configs/experimental_contract_v2.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["contract_version"], 2)
        self.assertEqual(payload["base_model"]["id"], MODEL_ID)
        self.assertEqual(payload["base_model"]["revision"], MODEL_REVISION)
        self.assertIsNone(payload["base_model"]["fallback"])
        self.assertTrue(sha256_file(path))

    def test_v1_contract_untouched(self) -> None:
        payload = json.loads(
            Path("configs/experimental_contract_v1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["contract_version"], 1)

    def test_readiness_v1_compat_and_v2_thresholds(self) -> None:
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
                donor_gain=0.12,
                host_nonempty_rate=1.0,
                host_target_language_rate=0.95,
                min_nonempty_rate=0.9,
                min_target_language_rate=0.9,
                donor_em=0.15,
                host_logprob_gain=0.06,
                belebele_drop=0.01,
                min_donor_gain=0.10,
                min_donor_em=0.20,
                min_host_logprob_gain=0.05,
                max_belebele_drop=0.05,
            ),
            "STOP_EXPERT_READINESS",
        )
        self.assertEqual(
            readiness_gate(
                donor_gain=0.12,
                host_nonempty_rate=1.0,
                host_target_language_rate=0.95,
                min_nonempty_rate=0.9,
                min_target_language_rate=0.9,
                donor_em=0.25,
                host_logprob_gain=0.06,
                belebele_drop=0.01,
                min_donor_gain=0.10,
                min_donor_em=0.20,
                min_host_logprob_gain=0.05,
                max_belebele_drop=0.05,
            ),
            "PASS_READINESS",
        )

    def test_resume_guard(self) -> None:
        with self.assertRaises(ValueError):
            assert_resume_compatible(
                {"model": {"id": MODEL_ID, "revision": "abc"}, "contract_sha256": "1"},
                {"model_id": MODEL_ID, "model_revision": MODEL_REVISION, "contract_sha256": "1"},
            )

    def test_classification_and_holm(self) -> None:
        self.assertEqual(classify_v2_panel({}, ["es", "zh", "sw"]), "expert_or_setup_failure")
        self.assertEqual(
            classify_v2_panel(
                {
                    "es": {"gate": "PASS_E1"},
                    "zh": {"gate": "STOP_E2_REGIME_INVESTIGATION"},
                    "sw": {"gate": "PASS_E1"},
                },
                [],
            ),
            "language_dependent_applicability",
        )
        adjusted = holm_adjust({"es": 0.01, "zh": 0.04, "sw": 0.20})
        self.assertLessEqual(adjusted["es"], adjusted["zh"])
        self.assertLessEqual(adjusted["zh"], 1.0)

    def test_paper_block_retains_v1_negative_and_qwen3_pin(self) -> None:
        from suture.tier_a_stats import render_v2_paper_block

        text = render_v2_paper_block(
            {
                "classification": "persistent_cross_set_failure",
                "results": {
                    "es": {
                        "spearman_cross_set_exact_match": -0.02,
                        "pearson_cross_set_exact_match": 0.01,
                        "accuracy_regret": 0.04,
                        "selected_exact_match": 0.10,
                        "baseline_exact_match": 0.08,
                        "selected_drift_rate": 0.2,
                        "gate": "STOP_E2_REGIME_INVESTIGATION",
                        "readiness_gains": {"donor_em": 0.22, "donor_gain": 0.12},
                    }
                },
                "blocked_languages": ["zh"],
                "holm_adjusted_p_values": {"es": 0.4},
            }
        )
        self.assertIn("Qwen/Qwen3-1.7B", text)
        self.assertIn("70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", text)
        self.assertIn("$-0.018$", text)
        self.assertIn("regret $0.05$", text)
        self.assertIn("e1\\_canonical\\_seed0", text)
        self.assertIn("persistent\\_cross\\_set\\_failure", text)
        self.assertNotIn("Qwen2.5-Coder", text)


if __name__ == "__main__":
    unittest.main()
