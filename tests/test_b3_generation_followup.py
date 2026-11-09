"""CPU-only contract and resume tests for the B3 generation follow-up."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from suture.b3_generation_followup import (
    _prepare_record,
    _reasoning_segment,
    _run_identity,
    _fasttext_predict,
    _answer_segment,
    aggregate_language,
    build_prompt_token_ids,
    candidate_spec,
    extract_question,
    language_id,
    load_config,
    parse_final_answer,
    paired_bootstrap,
    run_candidate,
)
from suture.data_manifest import write_manifest


class _NativeLid:
    def __init__(self, label: str = "__label__fr", probability: float = 0.9):
        self.label = label
        self.probability = probability
        self.inputs = []
        self.f = self

    def predict(self, text, k, threshold, on_unicode_error):
        self.inputs.append((text, k, threshold, on_unicode_error))
        return [(self.probability, self.label)]


class _Tokenizer:
    chat_template = "<|im_start|>{{ role }}\n{{ content }}<|im_end|>"
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    unk_token_id = 3

    def __init__(self, decoded: str = "On calcule toutes les étapes pour obtenir le résultat. </think>\nRéponse: 4<|im_end|>"):
        self.decoded = decoded
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return [101, 102, 103]

    def decode(self, token_ids, **kwargs):
        return self.decoded


class _FakeLLM:
    calls = 0
    init_kwargs = None

    def __init__(self, **kwargs):
        type(self).init_kwargs = kwargs

    def generate(self, prompts, sampling_params, use_tqdm):
        type(self).calls += 1
        assert use_tqdm is False
        assert len(prompts) == len(sampling_params)
        return [
            SimpleNamespace(
                outputs=[SimpleNamespace(token_ids=[201, 202], text="raw completion", finish_reason="stop")]
            )
            for _ in prompts
        ]


def _fake_sampling(**kwargs):
    return kwargs


class B3GenerationFollowupTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def test_candidate_windows_match_frozen_selection_and_publication(self):
        fr_selected = candidate_spec(self.config, "fr", "selected")
        fr_published = candidate_spec(self.config, "fr", "published_reference")
        zh_selected = candidate_spec(self.config, "zh", "selected")
        self.assertEqual(fr_selected["graft_layers"], [13, 14, 15])
        self.assertEqual(fr_published["graft_layers"], list(range(13, 23)))
        self.assertEqual(zh_selected["graft_layers"], list(range(20, 32)))

    def test_native_prompts_use_thinking_template_and_localized_instruction(self):
        tokenizer = _Tokenizer()
        ids = build_prompt_token_ids(tokenizer, "fr", "Question ?", self.config)
        self.assertEqual(ids, [101, 102, 103])
        messages, kwargs = tokenizer.calls[0]
        self.assertEqual(messages[0]["role"], "user")
        self.assertTrue(messages[0]["content"].startswith("Résolvez ce problème de mathématiques."))
        self.assertTrue(messages[0]["content"].endswith("\nQuestion ?"))
        self.assertIs(kwargs["enable_thinking"], True)
        self.assertIs(kwargs["add_generation_prompt"], True)
        self.assertIs(kwargs["tokenize"], True)

    def test_manifest_question_parser_is_strict_and_exact(self):
        record = {
            "id": "x",
            "input_text": "<|im_start|>user\nQuestion?\nAnswer:<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        }
        self.assertEqual(extract_question(record), "Question?")
        broken = dict(record, input_text=record["input_text"].replace("Answer:", "Réponse:"))
        with self.assertRaises(Exception):
            extract_question(broken)

    def test_parser_accepts_localized_english_and_bare_boxed_answers_without_reading_reasoning(self):
        localized = parse_final_answer("**Réponse finale:** $\\boxed{0018}$", "fr")
        english = parse_final_answer("Answer: 18", "fr")
        bare = parse_final_answer("\\boxed{18}", "fr")
        ambiguous = parse_final_answer("The steps are 3 + 6 = 9, then add 9 more.", "fr")
        self.assertEqual(localized["prediction"], "18")
        self.assertTrue(localized["strict_localized_format"])
        self.assertEqual(english["prediction"], "18")
        self.assertFalse(english["strict_localized_format"])
        self.assertEqual(bare["prediction"], "18")
        self.assertEqual(bare["format_status"], "bare_final_integer")
        self.assertIsNone(ambiguous["prediction"])
        self.assertEqual(parse_final_answer("Réponse: -018", "fr")["prediction"], "-18")

    def test_reasoning_uses_prefilled_open_think_and_marks_unclosed_output_incomplete(self):
        text, status = _reasoning_segment("Okay, nous calculons ici. </think> Réponse: 4")
        partial, partial_status = _reasoning_segment("Okay, nous calculons ici.")
        self.assertEqual(text, "Okay, nous calculons ici.")
        self.assertEqual(status, "complete")
        self.assertEqual(partial, "Okay, nous calculons ici.")
        self.assertEqual(partial_status, "incomplete_no_think_close")

    def test_native_fasttext_wrapper_uses_newline_and_avoids_numpy_wrapper(self):
        model = _NativeLid()
        self.assertEqual(_fasttext_predict(model, "Un texte assez long."), ("__label__fr", 0.9))
        self.assertEqual(model.inputs[0], ("Un texte assez long.\n", 1, 0.0, "strict"))

    def test_primary_lid_requires_minimum_prose_and_keeps_top1_probability(self):
        model = _NativeLid(probability=0.52)
        result = language_id("Voici un raisonnement assez long pour classifier correctement.", "fr", model, self.config)
        short = language_id("12345", "fr", model, self.config)
        self.assertEqual(result["status"], "evaluated")
        self.assertTrue(result["target_language_top1"])
        self.assertEqual(result["probability"], 0.52)
        self.assertEqual(short["status"], "insufficient_text")

    def test_truncated_output_is_incorrect_and_primary_lid_unknown_but_keeps_diagnostic(self):
        record = {"id": "x", "_hash": "h", "answer_number": "4", "input_text": "<|im_start|>user\nQuestion?\nAnswer:<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"}
        tokenizer = _Tokenizer("On explique suffisamment longtemps pour analyser le problème. </think>\nRéponse: 4")
        lid = _NativeLid()
        generation = SimpleNamespace(outputs=[SimpleNamespace(token_ids=[1, 2], text="", finish_reason="length")])
        row = _prepare_record(
            record,
            [101],
            language="fr",
            candidate="host_baseline",
            expected={"graft_layers": []},
            tokenizer=tokenizer,
            lid_model=lid,
            run_identity="run",
            config=self.config,
            request_seed=13,
            generation_output=generation,
        )
        self.assertFalse(row["exact_match"])
        self.assertTrue(row["truncated"])
        self.assertEqual(row["reasoning_language_id"]["status"], "truncated_output")
        self.assertIsNone(row["reasoning_language_id"]["target_language_top1"])
        self.assertTrue(row["reasoning_language_id"]["diagnostic_target_language_top1"])
        self.assertIsNone(row["reasoning_english_chunk_diagnostic"]["any_english_chunk"])

    def test_truncated_reasoning_number_is_not_misread_as_answer(self):
        answer_text, status = _answer_segment("The reasoning ends with 4.", "length")
        parsed = parse_final_answer(answer_text, "fr")
        self.assertEqual(status, "missing_think_close_before_completion")
        self.assertIsNone(parsed["prediction"])

    def test_paired_bootstrap_is_reproducible(self):
        left = paired_bootstrap([1.0, 0.0, -1.0], seed=7, n_resamples=100)
        right = paired_bootstrap([1.0, 0.0, -1.0], seed=7, n_resamples=100)
        self.assertEqual(left, right)
        self.assertEqual(left["mean_difference"], 0.0)

    def test_run_batches_and_resumes_with_same_per_record_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "model"
            tokenizer_dir = root / "tokenizer"
            model_dir.mkdir()
            tokenizer_dir.mkdir()
            (model_dir / "config.json").write_text(json.dumps({
                "model_type": "qwen3",
                "num_hidden_layers": 36,
                "architectures": ["Qwen3ForCausalLM"],
                "eos_token_id": 2,
            }), encoding="utf-8")
            (model_dir / "generation_config.json").write_text("{}\n", encoding="utf-8")
            for name in ("tokenizer.json", "tokenizer_config.json"):
                (tokenizer_dir / name).write_text("{}\n", encoding="utf-8")
            (tokenizer_dir / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<eos>"}), encoding="utf-8")
            lid_path = root / "lid.176.ftz"
            lid_path.write_bytes(b"fixture")
            config = copy.deepcopy(self.config)
            config["dataset"]["n_items_per_language"] = 2
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
            manifest = root / "mgsm.jsonl"
            records = [
                {
                    "id": f"b3:fr:MGSM_rev2_test:{index}",
                    "input_text": f"<|im_start|>user\nQuestion {index}?\nAnswer:<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                    "answer_number": "4",
                    "language": "fr",
                    "dataset_revision": config["dataset"]["revision"],
                }
                for index in range(2)
            ]
            write_manifest(
                manifest,
                records,
                manifest_name="b3_fr_MGSM_rev2_test",
                metadata={
                    "language": "fr",
                    "source": {"dataset": config["dataset"]["id"], "revision": config["dataset"]["revision"]},
                },
            )
            _FakeLLM.calls = 0
            kwargs = {
                "language": "fr",
                "candidate": "host_baseline",
                "model_dir": model_dir,
                "tokenizer_dir": tokenizer_dir,
                "manifest_path": manifest,
                "lid_path": lid_path,
                "output_root": root / "outputs",
                "config_path": config_path,
                "container_image_digest": "sha256:" + "a" * 64,
                "llm_factory": _FakeLLM,
                "sampling_params_factory": _fake_sampling,
                "tokenizer": _Tokenizer(),
                "lid_model": _NativeLid(),
                "enforce_cuda": False,
            }
            first = run_candidate(**kwargs)
            self.assertEqual(first["status"], "PASS_GENERATION")
            self.assertEqual(first["completed_records"], 2)
            rows = [json.loads(line) for line in (root / "outputs" / "fr" / "host_baseline" / "generations.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["exact_match"] for row in rows))
            self.assertNotEqual(rows[0]["request_seed"], rows[1]["request_seed"])
            self.assertEqual(rows[0]["reasoning_segment_status"], "complete")
            self.assertEqual(rows[0]["reasoning_language_id"]["status"], "evaluated")
            second = run_candidate(**kwargs)
            self.assertTrue(second["resumed_without_generation"])
            self.assertEqual(_FakeLLM.calls, 1)
            changed_model_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
            changed_model_config["local_mutation_for_test"] = True
            (model_dir / "config.json").write_text(json.dumps(changed_model_config), encoding="utf-8")
            with self.assertRaises(Exception):
                run_candidate(**kwargs)

    def test_aggregate_requires_all_source_ids_and_builds_paired_comparisons(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = copy.deepcopy(self.config)
            config["metrics"]["bootstrap_resamples"] = 25
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
            records = [
                {
                    "id": f"b3:fr:MGSM_rev2_test:{index}",
                    "input_text": f"<|im_start|>user\nQuestion {index}?\nAnswer:<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                    "answer_number": str(index + 1),
                    "language": "fr",
                    "dataset_revision": config["dataset"]["revision"],
                }
                for index in range(250)
            ]
            manifest = root / "MGSM_rev2_test.jsonl"
            write_manifest(
                manifest,
                records,
                manifest_name="b3_fr_MGSM_rev2_test",
                metadata={"language": "fr", "source": {"dataset": config["dataset"]["id"], "revision": config["dataset"]["revision"]}},
            )
            manifest_hash = __import__("hashlib").sha256(manifest.read_bytes()).hexdigest()
            output_root = root / "outputs"
            for candidate_index, candidate in enumerate(("host_baseline", "selected", "published_reference")):
                run_identity = f"identity-{candidate}"
                run = {
                    "status": "PASS_GENERATION",
                    "language": "fr",
                    "candidate": candidate,
                    "run_identity": run_identity,
                    "data_manifest": {"path": str(manifest), "sha256": manifest_hash},
                    "followup_config": {"path": str(config_path), "sha256": __import__("hashlib").sha256(config_path.read_bytes()).hexdigest()},
                    "frozen_source_contract": {"sha256": "contract"},
                    "prompt_protocol": {"sha256": "prompt"},
                    "fasttext_lid": {"sha256": "lid"},
                    "container_image_digest": "sha256:" + "a" * 64,
                    "runtime": {
                        "package_versions": {"vllm": "0.11.0"},
                        "torch_cuda_version": "12.8",
                        "cuda_available": True,
                        "gpu_devices": [{"name": "A100", "memory_bytes": 80}],
                    },
                    "tokenizer": {"sha256": "tokenizer"},
                    "checkpoint_decode_behavior": {
                        "model_eos_token_id": 2,
                        "model_bos_token_id": 1,
                        "tokenizer_special_tokens": {"eos_token": "<eos>"},
                    },
                }
                candidate_dir = output_root / "fr" / candidate
                candidate_dir.mkdir(parents=True)
                (candidate_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")
                output_lines = []
                for index, record in enumerate(records):
                    correct = ((index + candidate_index) % (candidate_index + 2)) != 0
                    prediction = str(index + 1) if correct else str(index + 2)
                    output_lines.append(json.dumps({
                        "record_id": record["id"],
                        "run_identity": run_identity,
                        "record_hash": __import__("suture.data_manifest", fromlist=["record_hash"]).record_hash(record),
                        "ground_truth_answer": record["answer_number"],
                        "exact_match": correct,
                        "truncated": False,
                        "answer_parse": {"status": "parsed", "prediction": prediction, "format_status": "localized_label", "strict_localized_format": True},
                        "reasoning_language_id": {"status": "evaluated", "target_language_top1": bool(index % 3), "probability": 0.9},
                        "reasoning_english_chunk_diagnostic": {"status": "evaluated", "any_english_chunk": bool(index % 7 == 0)},
                        "answer_language_id": {"status": "insufficient_text", "target_language_top1": None},
                        "prompt_token_ids": [index, 99],
                        "request_seed": 5000 + index,
                    }, ensure_ascii=False))
                (candidate_dir / "generations.jsonl").write_text("\n".join(output_lines) + "\n", encoding="utf-8")
            summary = aggregate_language("fr", output_root=output_root, config_path=config_path)
            self.assertEqual(summary["status"], "PASS_AGGREGATE")
            self.assertEqual(summary["candidate_summaries"]["host_baseline"]["n_records"], 250)
            self.assertEqual(summary["paired_comparisons"]["selected_minus_host"]["accuracy_all_records"]["n_paired_records"], 250)
            self.assertEqual(summary["candidate_summaries"]["host_baseline"]["reasoning_language_id"]["denominator_all_records"], 250)


if __name__ == "__main__":
    unittest.main()
