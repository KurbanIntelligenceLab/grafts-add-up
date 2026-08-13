"""Unit tests for the fail-closed data-integrity contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from paper_iclr.data_manifest import (
    ManifestError,
    assert_pairwise_disjoint,
    read_manifest,
    record_hash,
    write_manifest,
)


class DataManifestTests(unittest.TestCase):
    def test_hash_is_stable_and_round_trips_utf8(self) -> None:
        row = {"source_id": "x:1", "text": "¿Cuánto? 中文"}
        digest = record_hash(row)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "P.jsonl"
            write_manifest(path, [row], manifest_name="P", metadata={"language": "es"})
            header, records = read_manifest(path)
        self.assertEqual(header["_manifest"], "P")
        self.assertEqual(records[0]["_hash"], digest)

    def test_pairwise_overlap_fails_closed(self) -> None:
        row = {"source_id": "same", "text": "prompt"}
        with self.assertRaises(ManifestError):
            assert_pairwise_disjoint({"P_util": [row], "C": [row]})

    def test_bad_hash_fails_closed(self) -> None:
        with self.assertRaises(ManifestError):
            write_manifest(
                "verify/.manifest_test_should_not_exist.jsonl",
                [{"source_id": "x", "_hash": "wrong"}],
                manifest_name="bad",
            )


if __name__ == "__main__":
    unittest.main()
