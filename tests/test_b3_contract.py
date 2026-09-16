"""Offline tests for the isolated LightOn Qwen3-8B B3 preparation gate."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from suture.b3_lighton import (
    B3Error,
    _candidate_windows,
    _windows,
    contract_path,
    inspect_snapshot,
    load_contract,
    run_measure,
    run_preflight,
    validate_contract,
    validate_pair_snapshots,
)
from suture.suture_metrics import GraftScores


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _fake_snapshot(root: Path, *, missing_shard: bool = False) -> Path:
    root.mkdir(parents=True)
    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "max_position_embeddings": 40960,
        "vocab_size": 151936,
        "torch_dtype": "bfloat16",
    }
    _write_json(root / "config.json", config)
    _write_json(
        root / "tokenizer_config.json",
        {
            "pad_token": "<|endoftext|>",
            "padding_side": "left",
            "chat_template": "test-template",
        },
    )
    (root / "tokenizer.json").write_text("tokenizer\n", encoding="utf-8")
    (root / "merges.txt").write_text("merges\n", encoding="utf-8")
    (root / "vocab.json").write_text("{}\n", encoding="utf-8")
    _write_json(
        root / "model.safetensors.index.json",
        {"weight_map": {"model.layers.0.self_attn.q_proj.weight": "model-00001.safetensors"}},
    )
    if not missing_shard:
        (root / "model-00001.safetensors").write_bytes(b"fixture")
    return root


class B3ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = load_contract(contract_path())
        self.host_spec = self.contract["models"]["hosts"][0]
        self.donor_spec = self.contract["models"]["donor"]

    def test_contract_has_immutable_pair_pins_and_isolated_results(self) -> None:
        self.assertEqual(self.contract["contract_status"], "frozen_preflight_only")
        self.assertEqual(self.contract["data"]["windows"]["expected_count"], 666)
        self.assertEqual(self.contract["results"]["root"], "results/b3/lighton_qwen3_8b")
        self.assertEqual(len(self.host_spec["revision"]), 40)
        self.assertEqual(len(self.donor_spec["revision"]), 40)
        self.assertEqual(
            self.contract["data"]["sources"]["utility"]["revision"],
            "740312add88f781978c0658806c59bc2815b9866",
        )
        self.assertEqual(
            self.contract["data"]["sources"]["risk"]["revision"],
            "04120fd1f0ef4faed0d6fd4fb632a14476fb0498",
        )
        self.assertEqual(
            self.contract["data"]["sources"]["translation"]["revision"],
            "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        )
        held_out = self.contract["data"]["sources"]["held_out_measurement"]
        self.assertEqual(held_out["dataset"], "lightonai/mgsm-rev2")
        self.assertEqual(held_out["revision"], "1463c6dc7991a8751b8e28e76f6c561b9201eb55")
        self.assertEqual(held_out["n_items"], 250)
        self.assertEqual(
            self.contract["data"]["held_out_measurement"]["candidates"],
            ["host_baseline", "selected", "published_reference"],
        )

    def test_contract_rejects_mutable_revision(self) -> None:
        changed = copy.deepcopy(self.contract)
        changed["models"]["donor"]["revision"] = "main"
        with self.assertRaises(B3Error):
            validate_contract(changed)

    def test_pair_snapshot_preflight_hashes_files_and_tokenizers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            host = _fake_snapshot(root / "host")
            donor = _fake_snapshot(root / "donor")
            pair = validate_pair_snapshots(
                host,
                donor,
                self.host_spec,
                self.donor_spec,
                self.contract,
            )
            self.assertTrue(pair["pair_compatible"])
            self.assertEqual(pair["host"]["architecture"]["n_layers"], 36)
            self.assertIsNotNone(pair["host"]["snapshot_sha256"])
            self.assertEqual(
                pair["host"]["tokenizer"]["chat_template_sha256"],
                pair["donor"]["tokenizer"]["chat_template_sha256"],
            )

    def test_missing_weight_shard_stops_setup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = _fake_snapshot(Path(temporary) / "missing", missing_shard=True)
            with self.assertRaises(B3Error):
                inspect_snapshot(snapshot, self.host_spec, self.contract)

    def test_tokenizer_difference_stops_pair_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            host = _fake_snapshot(root / "host")
            donor = _fake_snapshot(root / "donor")
            (donor / "vocab.json").write_text('{"different": 1}\n', encoding="utf-8")
            with self.assertRaises(B3Error):
                validate_pair_snapshots(
                    host,
                    donor,
                    self.host_spec,
                    self.donor_spec,
                    self.contract,
                )

    def test_preflight_writes_pass_report_without_loading_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            host = _fake_snapshot(root / "host")
            donor = _fake_snapshot(root / "donor")
            output = root / "results"
            report = run_preflight(
                language="fr",
                host_path=host,
                donor_path=donor,
                output_root=output,
                load_models=False,
            )
            self.assertEqual(report["status"], "PASS_PREFLIGHT")
            self.assertTrue(
                (output / "fr_host__en_donor" / "preflight" / "preflight_report.json").is_file()
            )
            self.assertTrue(
                (output / "fr_host__en_donor" / "preflight" / "run_manifest.json").is_file()
            )

    def test_window_enumeration_is_exact(self) -> None:
        scores = GraftScores(
            utility=np.asarray([1.0, 2.0, 3.0]),
            risk=np.asarray([0.0, 0.0, 0.0]),
            injection_norm=np.asarray([1.0, 1.0, 1.0]),
            n_probe_utility=1,
            n_probe_risk=1,
        )
        windows = _windows(scores)
        self.assertEqual(len(windows), 6)
        self.assertEqual((windows[0]["start"], windows[0]["end"]), (0, 0))
        self.assertEqual((windows[-1]["start"], windows[-1]["end"]), (2, 2))

    def test_measurement_candidates_are_baseline_selected_and_published(self) -> None:
        candidates = _candidate_windows(
            self.contract,
            {"selection": {"graft": [2, 3, 4]}},
            "fr",
        )
        self.assertEqual([item["name"] for item in candidates], [
            "host_baseline",
            "selected",
            "published_reference",
        ])
        self.assertEqual(candidates[0]["graft"], [])
        self.assertEqual(candidates[1]["graft"], [2, 3, 4])
        self.assertEqual(candidates[2]["graft"], list(range(13, 23)))

    def test_measurement_stage_isolated_and_accounted(self) -> None:
        class _Tokenizer:
            pad_token_id = 0
            padding_side = "left"

            def __call__(self, text, **kwargs):
                del text, kwargs
                return {
                    "input_ids": torch.tensor([[1, 2]]),
                    "attention_mask": torch.tensor([[1, 1]]),
                }

        class _Routed:
            def __init__(self, graft):
                self.graft = graft

            def sequence_logprob(self, input_ids, target_input_ids, attention_mask):
                del input_ids, attention_mask
                return torch.tensor([-float(len(self.graft) + target_input_ids.shape[1])])

        class _Adapter:
            merged_model_builds = 0

            def __init__(self, host, donor, *, device):
                del host, donor, device

            def build_grafted_model(self, graft):
                type(self).merged_model_builds += 1
                return _Routed(tuple(graft))

        from suture.data_manifest import write_manifest

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            host = _fake_snapshot(root / "host")
            donor = _fake_snapshot(root / "donor")
            output = root / "results"
            data_root = output / "fr_host__en_donor" / "data"
            write_manifest(
                data_root / "P_util.jsonl",
                [{"id": "selection-util", "input_text": "u", "utility_texts": ["1"]}],
                manifest_name="util",
            )
            write_manifest(
                data_root / "P_risk.jsonl",
                [{
                    "id": "selection-risk",
                    "input_text": "r",
                    "risk_target_texts": ["français"],
                    "risk_donor_texts": ["English"],
                }],
                manifest_name="risk",
            )
            held_out = [
                {
                    "id": f"b3:fr:MGSM_rev2_test:{index}",
                    "input_text": f"question {index}",
                    "answer_number": "4",
                    "dataset": self.contract["data"]["sources"]["held_out_measurement"]["dataset"],
                    "dataset_revision": self.contract["data"]["sources"]["held_out_measurement"]["revision"],
                    "language": "fr",
                }
                for index in range(250)
            ]
            measurement_path = data_root / "MGSM_rev2_test.jsonl"
            write_manifest(
                measurement_path,
                held_out,
                manifest_name="b3_fr_MGSM_rev2_test",
                metadata={
                    "contract_name": self.contract["contract_name"],
                    "contract_version": self.contract["contract_version"],
                    "language": "fr",
                    "objective": self.contract["data"]["held_out_measurement"]["objective"],
                    "source": self.contract["data"]["sources"]["held_out_measurement"],
                },
            )
            score_root = output / "fr_host__en_donor" / "score"
            score_root.mkdir(parents=True)
            _write_json(
                score_root / "selection.json",
                {
                    "selection": {"graft": [2, 3]},
                    "merged_model_builds_during_selection": 0,
                },
            )
            with patch("suture.b3_lighton._hardware_report", return_value={"name": "fixture"}), \
                patch("suture.b3_lighton._load_tokenizer", return_value=_Tokenizer()), \
                patch("suture.b3_lighton._load_model", side_effect=["host", "donor"]), \
                patch("suture.suture_torch.HFResidualAdapter", _Adapter):
                result = run_measure(
                    language="fr",
                    host_path=host,
                    donor_path=donor,
                    output_root=output,
                    measurement_path=measurement_path,
                )
            self.assertEqual(result["status"], "PASS_MEASURE")
            self.assertEqual(result["measurement_merged_model_builds"], 3)
            self.assertEqual(len(result["candidates"]), 3)
            self.assertTrue((output / "fr_host__en_donor" / "measure" / "measure.json").is_file())


if __name__ == "__main__":
    unittest.main()
