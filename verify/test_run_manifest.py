"""Tests for immutable, stage-specific run provenance."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from paper_iclr.run_manifest import write_manifest


class RunManifestTests(unittest.TestCase):
    def test_manifest_records_stage_and_refuses_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            run_dir = root / "run"
            data_dir.mkdir()
            run_dir.mkdir()
            (data_dir / "P_util.jsonl").write_text("{}\n", encoding="utf-8")
            (run_dir / "e1_gate.json").write_text(
                json.dumps(
                    {
                        "selection_merged_models_built": 0,
                        "sweep_is_exhaustive": True,
                        "merged_models_built": 3,
                    }
                ),
                encoding="utf-8",
            )

            output = write_manifest(
                root=root,
                run_dir=run_dir,
                data_dir=data_dir,
                run_kind="e1",
                language="sw",
                command="python gate.py e1",
                seed=0,
                started_at="2026-08-13T00:00:00+00:00",
                ended_at="2026-08-13T00:01:00+00:00",
            )
            payload = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(payload["schema_version"], 2)
            self.assertEqual(payload["stage"], "e1")
            self.assertEqual(payload["seed"], 0)
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["duration_seconds"], 60.0)
            self.assertEqual(payload["selection_contract"]["e1_merged_models_built"], 3)
            self.assertNotIn("run_manifest.json", payload["files"]["outputs"])

            with self.assertRaises(FileExistsError):
                write_manifest(
                    root=root,
                    run_dir=run_dir,
                    data_dir=data_dir,
                    run_kind="e1",
                    language="sw",
                    command="python gate.py e1",
                )


if __name__ == "__main__":
    unittest.main()
