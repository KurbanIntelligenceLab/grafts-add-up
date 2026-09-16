"""Setup/evaluator guards for contract v2.  CPU-only."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from suture.tier_a_config import (
    ACTIVE_CONTRACT_VERSION,
    CHECKPOINT_SELECTION_RULE,
    MODEL_ID,
    ConfigError,
    refuse_frozen_write,
    results_root,
)
from suture.tier_a_eval import (
    IncompleteDenominatorError,
    align_prompt_completion,
    choose_max_gain_among_passing,
)
from suture.tier_a_text import (
    TextFilterError,
    balanced_round_robin,
    filter_host_rows,
    instruction_pair_from_row,
    looks_like_code,
)


class DummyTok:
    def __call__(self, text, add_special_tokens=True, truncation=False):
        tokens = list(range(1, max(len(text), 1) + 1))
        if not add_special_tokens:
            tokens = tokens[: max(len(text.split()), 1)]
        return {"input_ids": tokens}


class V2SetupTests(unittest.TestCase):
    def test_code_filter_and_language_rows(self) -> None:
        self.assertTrue(looks_like_code("def foo(x):\n    return x"))
        self.assertTrue(looks_like_code("Crea un programa en Python para ordenar una lista"))
        self.assertTrue(
            looks_like_code("Escriba una función JavaScript que sume dos numeros")
        )
        prompt, completion = instruction_pair_from_row(
            {
                "inputs": {
                    "1-instruction": "¿Qué significa DNA?",
                    "2-input": "",
                    "3-output": "ácido desoxirribonucleico.",
                }
            }
        )
        self.assertIn("DNA", prompt)
        self.assertIn("desoxirribonucleico", completion)

    def test_checkpoint_rule_max_gain_among_passing(self) -> None:
        self.assertEqual(CHECKPOINT_SELECTION_RULE, "max_gain_among_passing")
        rows = [
            {"path": "c256", "gain": 0.07, "value": 1.0, "passed": True},
            {"path": "c512", "gain": 0.19, "value": 1.1, "passed": True},
            {"path": "c1024", "gain": 0.42, "value": 1.2, "passed": True},
        ]
        selected = choose_max_gain_among_passing(rows)
        self.assertEqual(selected["path"], "c1024")

    def test_active_root_is_v2_and_legacy_writes_refused(self) -> None:
        self.assertEqual(ACTIVE_CONTRACT_VERSION, 2)
        self.assertEqual(MODEL_ID, "Qwen/Qwen3-1.7B")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            withdrawn = root / "results" / "v3" / "qwen3_1_7b" / "es" / "x.json"
            withdrawn.parent.mkdir(parents=True)
            with self.assertRaises(ConfigError):
                refuse_frozen_write(withdrawn, root=root)
            active = results_root(root)
            self.assertTrue(
                str(active).replace("\\", "/").endswith("results/v2/qwen3_1_7b")
            )

    def test_modules_import(self) -> None:
        for rel in ("src/suture/tier_a_v2.py", "src/suture/tier_a_e1_v2.py"):
            ast.parse(Path(rel).read_text(encoding="utf-8"))
        import suture.tier_a_v2 as v2

        self.assertTrue(hasattr(v2, "main"))
        self.assertTrue(hasattr(v2, "collect_setup_issues"))
        self.assertTrue(hasattr(v2, "require_manifest"))

    def test_panel_status_and_resume_refusal(self) -> None:
        from suture.tier_a_v2 import collect_setup_issues, require_manifest

        with tempfile.TemporaryDirectory() as tmp:
            lang = Path(tmp) / "es"
            ready = lang / "readiness"
            ready.mkdir(parents=True)
            (ready / "readiness.json").write_text(
                json.dumps(
                    {
                        "gate": "STOP_SETUP",
                        "setup_issues": ["incomplete_logprob_denominator"],
                        "host_expert_logprob": {"n": 22},
                        "belebele_host": {"scorer": "generate_parse"},
                    }
                ),
                encoding="utf-8",
            )
            issues = collect_setup_issues(lang)
            self.assertIn("incomplete_logprob_denominator", issues)
            dirty = lang / "data"
            dirty.mkdir()
            (dirty / "data_build_summary.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(SystemExit):
                require_manifest(dirty)
            (dirty / "run_manifest.json").write_text("{}", encoding="utf-8")
            require_manifest(dirty)

    def test_v2_contract_is_version_2(self) -> None:
        payload = json.loads(
            Path("configs/experimental_contract_v2.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(payload["contract_version"], 2)
        self.assertEqual(payload["base_model"]["id"], MODEL_ID)
        self.assertEqual(
            payload["experts"]["readiness"]["belebele_scorer"],
            "multiple_choice_logprob",
        )
        self.assertEqual(payload["data"]["results_root"], "results/v2/qwen3_1_7b")


if __name__ == "__main__":
    unittest.main()
