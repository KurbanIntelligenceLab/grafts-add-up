"""Schema checks for the frozen Tier-A artifact inventory."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "results" / "tier_a" / "ARTIFACT_INVENTORY.json"
ALLOWED = {"canonical", "preliminary", "smoke", "failed", "superseded"}


class ArtifactInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not INVENTORY.is_file():
            raise unittest.SkipTest("artifact inventory is absent")
        cls.payload = json.loads(INVENTORY.read_text(encoding="utf-8"))

    def test_canonical_e1_is_immutable_and_mgsm_test_unscored(self) -> None:
        self.assertTrue(self.payload["canonical_e1_immutable"])
        self.assertEqual(
            self.payload["canonical_e1"].replace("\\", "/"),
            "results/tier_a/es/e1_canonical_seed0",
        )
        self.assertIn("MGSM_test evaluation", self.payload["blocked_by_spanish_stop"])
        self.assertFalse(self.payload["languages"]["es"]["mgsm_test_scored"])

    def test_directory_classes_are_from_the_declared_set(self) -> None:
        seen = []
        for language, block in self.payload["languages"].items():
            for entry in block["directories"]:
                self.assertIn(entry["class"], ALLOWED, msg=f"{language}:{entry['directory']}")
                self.assertTrue(entry["reason"])
                seen.append(entry["class"])
        self.assertIn("canonical", seen)

    def test_canonical_directory_is_not_marked_overwriteable(self) -> None:
        for entry in self.payload["languages"]["es"]["directories"]:
            if entry["directory"].replace("\\", "/").endswith("e1_canonical_seed0"):
                self.assertEqual(entry["class"], "canonical")
                self.assertEqual(entry["claim"], "primary_e1_stop")
                return
        self.fail("canonical Spanish E1 directory missing from inventory")


if __name__ == "__main__":
    unittest.main()
