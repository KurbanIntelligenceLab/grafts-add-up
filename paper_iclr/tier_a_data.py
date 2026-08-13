"""Build Tier-A SUTURE manifests from public/cached datasets.

The builder is deliberately explicit about the data boundary:

* ``P_util`` comes only from GSM8K *train* questions translated by the pinned
  local checkpoint; its numeric answer is inherited from the English source.
* ``P_risk`` and ``C`` use disjoint source records from the multilingual
  translation pool and contain no task labels.
* MGSM ``train`` is the available development split and MGSM ``test`` is
  written once as an untouched final manifest.
* every evaluation/probe record is SHA-256 addressed and pairwise checked.

This script may download public datasets if they are not already cached.  It
never downloads or invents a model checkpoint; the translator model must be
the contract's pinned local checkpoint.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

try:
    from paper_iclr.data_manifest import (
        ManifestError,
        assert_pairwise_disjoint,
        summarize_manifest_set,
        write_manifest,
    )
except ModuleNotFoundError:
    from data_manifest import (
        ManifestError,
        assert_pairwise_disjoint,
        summarize_manifest_set,
        write_manifest,
    )


MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MODEL_REVISION = "2e1fd397ee46e1388853d2af2c993145b0f1098a"
LANGUAGE_NAMES = {"es": "Spanish", "zh": "Chinese", "sw": "Swahili"}


def _clean_translation(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"^```(?:text|[a-zA-Z_-]+)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"^(?:translation|translated text)\s*:\s*", "", text, flags=re.I)
    if not text:
        raise ManifestError("local translation returned empty text")
    return text.strip()


class LocalTranslator:
    """Deterministic local machine translation through the frozen model."""

    def __init__(
        self,
        *,
        model_id: str = MODEL_ID,
        revision: str = MODEL_REVISION,
        device: str | None = None,
        batch_size: int = 4,
        max_new_tokens: int = 256,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=True,
            dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        ).to(self.device).eval()
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens

    def _prompt(self, text: str, language: str) -> str:
        return (
            f"Translate the following English text into {LANGUAGE_NAMES[language]}. "
            "Output only the translation, with no explanation or quotation marks.\n\n"
            f"English text:\n{text}\n\nTranslation:"
        )

    def _chat_prompt(self, text: str, language: str) -> str:
        instruction = self._prompt(text, language)
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": "You are a precise translator. Output only the translation.",
                    },
                    {"role": "user", "content": instruction},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        return instruction

    def translate(self, texts: Sequence[str], language: str) -> List[str]:
        if language not in LANGUAGE_NAMES:
            raise ManifestError(f"unsupported target language: {language}")
        outputs: List[str] = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = list(texts[start : start + self.batch_size])
            print(
                f"[translation] {start + 1}-{start + len(batch_texts)}/{len(texts)} "
                f"to {language}",
                flush=True,
            )
            prompts = [self._chat_prompt(text, language) for text in batch_texts]
            encoded = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(self.device)
            with torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            prompt_width = encoded["input_ids"].shape[1]
            for row in generated[:, prompt_width:]:
                outputs.append(_clean_translation(self.tokenizer.decode(row, skip_special_tokens=True)))
        if len(outputs) != len(texts):
            raise ManifestError(f"translator returned {len(outputs)} outputs for {len(texts)} inputs")
        return outputs


def _load_translation_pool() -> List[Dict[str, Any]]:
    dataset = load_dataset(
        "CohereLabsCommunity/multilingual-reward-bench",
        "translation",
        split="test",
        download_mode="reuse_dataset_if_exists",
    )
    rows = []
    for index, row in enumerate(dataset):
        subset = str(row["subset"])
        if not subset.startswith("translation-en-"):
            continue
        source = str(row["source"]).strip()
        if source:
            rows.append(
                {
                    "source_id": f"multilingual_reward_bench:translation:{index}",
                    "source_text": source,
                    "source_subset": subset,
                }
            )
    if len(rows) < 200:
        raise ManifestError(
            f"translation pool has only {len(rows)} English-source records; need at least 200"
        )
    return rows


def _load_native_instruction_pool(language: str) -> List[Dict[str, Any]]:
    """Load target-language general instruction records when a public split exists."""

    config = {"es": "spa_Latn", "zh": "zho_Hans"}.get(language)
    if config is None:
        return []
    dataset = load_dataset(
        "CohereLabsCommunity/multilingual-reward-bench",
        config,
        split="test",
        download_mode="reuse_dataset_if_exists",
    )
    rows = []
    for index, row in enumerate(dataset):
        prompt = str(row["prompt"]).strip()
        completion = str(row["chosen"]).strip()
        if prompt and completion:
            rows.append(
                {
                    "source_id": f"multilingual_reward_bench:{config}:test:{index}",
                    "prompt": prompt,
                    "completion": completion,
                    "language": language,
                }
            )
    if not rows:
        raise ManifestError(f"native instruction pool is empty for {language}")
    return rows


def _load_gsm8k_train(n: int, *, start: int = 0) -> List[Dict[str, Any]]:
    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split="train",
        download_mode="reuse_dataset_if_exists",
    )
    if start < 0 or start + n > len(dataset):
        raise ManifestError(
            f"requested GSM8K rows [{start}, {start + n}) but only {len(dataset)} "
            "training rows are available"
        )
    rows = []
    for index in range(start, start + n):
        row = dataset[index]
        answer = str(row["answer"])
        match = re.findall(r"####\s*(-?\d+(?:\.\d+)?)", answer)
        if not match:
            raise ManifestError(f"GSM8K row {index} has no parseable final answer")
        rows.append(
            {
                "source_id": f"gsm8k:train:{index}",
                "source_question": str(row["question"]).strip(),
                "source_text": str(row["question"]).strip(),
                "source_answer": answer,
                "answer_number": match[-1],
            }
        )
    return rows


def _load_mgsm(language: str, split: str) -> List[Dict[str, Any]]:
    dataset = load_dataset(
        "juletxara/mgsm",
        language,
        split=split,
        download_mode="reuse_dataset_if_exists",
    )
    rows = []
    for index, row in enumerate(dataset):
        question = str(row["question"]).strip()
        number = row["answer_number"]
        if not question or number is None:
            raise ManifestError(f"MGSM {language}/{split} row {index} is incomplete")
        rows.append(
            {
                "source_id": f"mgsm:{language}:{split}:{index}",
                "question": question,
                "answer_number": str(number),
                "language": language,
            }
        )
    return rows


def _make_sft_rows(targets: Sequence[str], sources: Sequence[str], language: str) -> List[Dict[str, Any]]:
    return [
        {
            "source_id": f"translation_pool:host_train:{i}",
            "prompt": f"Instruction: {target}\nRespond in {LANGUAGE_NAMES[language]}:",
            "completion": target,
            "language": language,
            "translation_source": source,
        }
        for i, (target, source) in enumerate(zip(targets, sources))
    ]


def _make_native_sft_rows(
    rows: Sequence[Mapping[str, Any]],
    language: str,
) -> List[Dict[str, Any]]:
    return [
        {
            "source_id": row["source_id"],
            "prompt": row["prompt"],
            "completion": row["completion"],
            "language": language,
            "source_dataset": "CohereLabsCommunity/multilingual-reward-bench",
        }
        for row in rows
    ]


def build(
    *,
    language: str,
    output_dir: str | Path,
    n_util: int = 200,
    n_risk: int = 200,
    n_calibration: int = 500,
    n_host_train: int = 256,
    translation_batch_size: int = 4,
    max_new_tokens: int = 96,
    device: str | None = None,
) -> Dict[str, Any]:
    if language not in LANGUAGE_NAMES:
        raise ManifestError(f"language must be one of {sorted(LANGUAGE_NAMES)}")
    pool = _load_translation_pool()
    native_pool = _load_native_instruction_pool(language)
    if native_pool:
        if n_host_train + n_calibration > len(native_pool):
            raise ManifestError(
                f"need {n_host_train + n_calibration} native instruction records but "
                f"only {len(native_pool)} are available for {language}"
            )
    if n_risk > len(pool):
        raise ManifestError(f"need {n_risk} translation records but only {len(pool)} are available")
    translator = LocalTranslator(
        batch_size=translation_batch_size,
        max_new_tokens=max_new_tokens,
        device=device,
    )

    risk_pool = pool[:n_risk]
    if native_pool:
        host_pool = list(native_pool[:n_host_train])
        calibration_pool = list(native_pool[n_host_train : n_host_train + n_calibration])
        fallback_mode = "native_target_instruction_pool"
    else:
        available_after_risk = pool[n_risk:]
        if len(available_after_risk) >= n_host_train + n_calibration:
            host_pool = list(available_after_risk[:n_host_train])
            calibration_pool = list(
                available_after_risk[n_host_train : n_host_train + n_calibration]
            )
            fallback_mode = "translated_instruction_pool"
        else:
            # The cached translation benchmark is enough for the risk probe,
            # but not for both the 500-item certificate and host training in
            # a lower-resource language.  Use held-out GSM8K train questions
            # for those two target-language text pools.  The ranges are
            # explicitly disjoint from P_util [0:n_util] and from each other.
            calibration_pool = _load_gsm8k_train(n_calibration, start=n_util)
            pool_host_count = min(n_host_train, len(available_after_risk))
            host_pool = list(available_after_risk[:pool_host_count])
            remaining_host = n_host_train - pool_host_count
            if remaining_host:
                host_pool.extend(
                    _load_gsm8k_train(
                        remaining_host,
                        start=n_util + n_calibration,
                    )
                )
            fallback_mode = "gsm8k_heldout_calibration_and_host_supplement"
    risk_translations = translator.translate(
        [row["source_text"] for row in risk_pool],
        language,
    )

    gsm_rows = _load_gsm8k_train(n_util)
    utility_translations = translator.translate(
        [row["source_question"] for row in gsm_rows],
        language,
    )
    p_util = [
        {
            "source_id": row["source_id"],
            "question": translated,
            "source_question": row["source_question"],
            "answer_number": row["answer_number"],
            "language": language,
            "translation_model": MODEL_ID,
            "translation_revision": MODEL_REVISION,
        }
        for row, translated in zip(gsm_rows, utility_translations)
    ]
    p_risk = [
        {
            "source_id": row["source_id"],
            "target_text": translated,
            "donor_text": row["source_text"],
            "language": language,
            "translation_model": MODEL_ID,
            "translation_revision": MODEL_REVISION,
        }
        for row, translated in zip(risk_pool, risk_translations)
    ]
    if native_pool:
        calibration = [
            {
                "source_id": row["source_id"],
                "target_text": row["prompt"],
                "language": language,
                "translation_model": "dataset-provided",
                "translation_revision": None,
            }
            for row in calibration_pool
        ]
        host_train = _make_native_sft_rows(host_pool, language)
    else:
        calibration_translations = translator.translate(
            [row["source_text"] for row in calibration_pool],
            language,
        )
        calibration = [
            {
                "source_id": row["source_id"],
                "target_text": translated,
                "language": language,
                "translation_model": MODEL_ID,
                "translation_revision": MODEL_REVISION,
            }
            for row, translated in zip(calibration_pool, calibration_translations)
        ]
        host_translations = translator.translate(
            [row["source_text"] for row in host_pool],
            language,
        )
        host_train = _make_sft_rows(
            host_translations,
            [row["source_text"] for row in host_pool],
            language,
        )
    development = _load_mgsm(language, "train")
    test = _load_mgsm(language, "test")

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths = {
        "P_util": target / "P_util.jsonl",
        "P_risk": target / "P_risk.jsonl",
        "C": target / "C.jsonl",
        "MGSM_dev": target / "MGSM_dev.jsonl",
        "MGSM_test": target / "MGSM_test.jsonl",
    }
    risk_source_dataset = (
        "CohereLabsCommunity/multilingual-reward-bench:"
        f"{'spa_Latn' if language == 'es' else 'zho_Hans' if language == 'zh' else 'translation'}:test"
    )
    if native_pool:
        calibration_source_dataset = risk_source_dataset
        host_source_dataset = risk_source_dataset
    elif fallback_mode == "gsm8k_heldout_calibration_and_host_supplement":
        calibration_source_dataset = "openai/gsm8k:main:train:held_out_ranges"
        host_source_dataset = (
            "mixed:CohereLabsCommunity/multilingual-reward-bench:translation:test"
            "+openai/gsm8k:main:train:held_out_range"
        )
    else:
        calibration_source_dataset = risk_source_dataset
        host_source_dataset = risk_source_dataset
    summaries = {
        "P_util": write_manifest(
            paths["P_util"],
            p_util,
            manifest_name="P_util",
            metadata={
                "language": language,
                "source_dataset": "openai/gsm8k:main:train",
                "translation_model": MODEL_ID,
                "translation_revision": MODEL_REVISION,
            },
        ),
        "P_risk": write_manifest(
            paths["P_risk"],
            p_risk,
            manifest_name="P_risk",
            metadata={
                "language": language,
                "source_dataset": risk_source_dataset,
                "translation_model": MODEL_ID,
                "translation_revision": MODEL_REVISION,
            },
        ),
        "C": write_manifest(
            paths["C"],
            calibration,
            manifest_name="C",
            metadata={
                "language": language,
                "source_dataset": calibration_source_dataset,
                "translation_model": (
                    "dataset-provided"
                    if native_pool
                    else MODEL_ID
                ),
                "translation_revision": None if native_pool else MODEL_REVISION,
            },
        ),
        "MGSM_dev": write_manifest(
            paths["MGSM_dev"],
            development,
            manifest_name="MGSM_dev",
            metadata={"language": language, "source_dataset": f"juletxara/mgsm:{language}:train"},
        ),
        "MGSM_test": write_manifest(
            paths["MGSM_test"],
            test,
            manifest_name="MGSM_test",
            metadata={"language": language, "source_dataset": f"juletxara/mgsm:{language}:test"},
        ),
    }
    loaded = {}
    for name, path in paths.items():
        with path.open("r", encoding="utf-8") as handle:
            lines = [json.loads(line) for line in handle if line.strip()][1:]
        loaded[name] = lines
    disjoint = assert_pairwise_disjoint(loaded)

    host_path = target / "host_train.jsonl"
    host_summary = write_manifest(
        host_path,
        host_train,
        manifest_name="host_train",
        metadata={
            "language": language,
            "source_dataset": host_source_dataset,
            "translation_model": (
                "dataset-provided"
                if native_pool
                else MODEL_ID
            ),
            "translation_revision": None if native_pool else MODEL_REVISION,
        },
    )
    summary = {
        "language": language,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "fallback_mode": fallback_mode,
        "manifests": summaries,
        "host_train": host_summary,
        "disjointness": disjoint,
        "counts": {name: len(rows) for name, rows in {**loaded, "host_train": host_train}.items()},
    }
    with (target / "data_build_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
    # Re-read through the public helper to ensure the files on disk, not only
    # in-memory rows, pass the integrity contract.
    disk_summary = summarize_manifest_set({name: str(path) for name, path in paths.items()})
    summary["disk_summary"] = disk_summary
    with (target / "data_build_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=sorted(LANGUAGE_NAMES), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-util", type=int, default=200)
    parser.add_argument("--n-risk", type=int, default=200)
    parser.add_argument("--n-calibration", type=int, default=500)
    parser.add_argument("--n-host-train", type=int, default=256)
    parser.add_argument("--translation-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    summary = build(
        language=args.language,
        output_dir=args.output_dir,
        n_util=args.n_util,
        n_risk=args.n_risk,
        n_calibration=args.n_calibration,
        n_host_train=args.n_host_train,
        translation_batch_size=args.translation_batch_size,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
