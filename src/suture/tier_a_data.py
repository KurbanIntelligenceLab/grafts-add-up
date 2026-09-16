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
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

try:
    from suture.data_manifest import (
        ManifestError,
        assert_pairwise_disjoint,
        read_manifest,
        summarize_manifest_set,
        write_manifest,
    )
    from suture.tier_a_config import (
        ACTIVE_CONTRACT_VERSION,
        HOST_SOURCE_CANDIDATES,
        LANGUAGE_NAMES,
        LID_MODEL_PATH,
        MODEL_ID,
        MODEL_REVISION,
        N_BELEBELE,
        N_CALIBRATION,
        N_EXPERT_VALIDATION,
        N_HOST_TRAIN,
        N_MGSM_DEV,
        N_MGSM_TEST,
        N_READINESS_DONOR,
        N_READINESS_HOST,
        N_RISK,
        N_UTIL,
        apply_qwen3_chat,
        frozen_mgsm_indices,
        refuse_v1_write,
        refuse_frozen_write,
        repo_root,
    )
    from suture.tier_a_preflight import pretrained_load_args
    from suture.tier_a_text import (
        TextFilterError,
        balanced_round_robin,
        filter_host_rows,
        instruction_pair_from_row,
        looks_like_code,
    )
except ModuleNotFoundError:
    from data_manifest import (
        ManifestError,
        assert_pairwise_disjoint,
        read_manifest,
        summarize_manifest_set,
        write_manifest,
    )
    from tier_a_config import (
        ACTIVE_CONTRACT_VERSION,
        HOST_SOURCE_CANDIDATES,
        LANGUAGE_NAMES,
        LID_MODEL_PATH,
        MODEL_ID,
        MODEL_REVISION,
        N_BELEBELE,
        N_CALIBRATION,
        N_EXPERT_VALIDATION,
        N_HOST_TRAIN,
        N_MGSM_DEV,
        N_MGSM_TEST,
        N_READINESS_DONOR,
        N_READINESS_HOST,
        N_RISK,
        N_UTIL,
        apply_qwen3_chat,
        frozen_mgsm_indices,
        refuse_v1_write,
        refuse_frozen_write,
        repo_root,
    )
    from tier_a_preflight import pretrained_load_args
    from tier_a_text import (
        TextFilterError,
        balanced_round_robin,
        filter_host_rows,
        instruction_pair_from_row,
        looks_like_code,
    )


BELEBELE_CONFIGS = {"es": "spa_Latn", "zh": "zho_Hans", "sw": "swh_Latn"}


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
        self.model_id = model_id
        self.revision = revision
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        source = pretrained_load_args(repo_root())
        self.tokenizer = AutoTokenizer.from_pretrained(
            source["pretrained_model_name_or_path"],
            local_files_only=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            source["pretrained_model_name_or_path"],
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
        return apply_qwen3_chat(
            self.tokenizer,
            instruction,
            system="You are a precise translator. Output only the translation.",
            enable_thinking=False,
        )

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
    seen: set[str] = set()
    for index, row in enumerate(dataset):
        subset = str(row["subset"])
        english = ""
        kind = ""
        if subset.startswith("translation-en-"):
            english = str(row["source"]).strip()
            kind = "source"
        elif subset.startswith("translation-") and subset.split("-")[2] == "en":
            english = str(row.get("chosen") or "").strip()
            kind = "chosen_en"
        if not english or english in seen:
            continue
        seen.add(english)
        rows.append(
            {
                "source_id": f"multilingual_reward_bench:translation:{index}:{kind}",
                "source_text": english,
                "source_subset": subset,
            }
        )
    if len(rows) < 500:
        raise ManifestError(
            f"translation pool has only {len(rows)} unique English records; need at least 500"
        )
    return rows


def _load_native_instruction_pool(language: str) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load licensed target-language instruction records. Never supplement with GSM8K."""

    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    errors: List[str] = []
    parsed_by_source: Dict[str, int] = {}
    empty_by_source: Dict[str, int] = {}
    for spec in HOST_SOURCE_CANDIDATES[language]:
        dataset_name = spec["dataset"]
        parsed_by_source.setdefault(dataset_name, 0)
        empty_by_source.setdefault(dataset_name, 0)
        kwargs: Dict[str, Any] = {
            "path": dataset_name,
            "split": spec["split"],
            "download_mode": "reuse_dataset_if_exists",
        }
        if spec.get("config"):
            kwargs["name"] = spec["config"]
        try:
            dataset = load_dataset(**kwargs)
        except Exception as exc:
            errors.append(f"{dataset_name}:{spec.get('config') or '_'}: {exc}")
            continue
        for index, row in enumerate(dataset):
            prompt, completion = instruction_pair_from_row(row)
            if not prompt or not completion:
                empty_by_source[dataset_name] += 1
                continue
            source_id = f"{dataset_name}:{spec.get('config') or '_'}:{spec['split']}:{index}"
            if source_id in seen:
                continue
            seen.add(source_id)
            parsed_by_source[dataset_name] += 1
            rows.append(
                {
                    "source_id": source_id,
                    "prompt": prompt,
                    "completion": completion,
                    "language": language,
                    "source_dataset": dataset_name,
                }
            )
    zero_parsed = [name for name, count in parsed_by_source.items() if count == 0]
    if zero_parsed:
        raise ManifestError(
            f"STOP_DATA: {language} produced 0 parseable instruction pairs from {zero_parsed}. "
            f"empty_pairs={empty_by_source}; load_errors={errors or 'none'}"
        )
    lid_model = None
    lid_path = repo_root() / LID_MODEL_PATH
    if lid_path.is_file():
        import fasttext

        lid_model = fasttext.load_model(str(lid_path))
    kept, stats = filter_host_rows(rows, language, lid_model=lid_model)
    leftover_code = sum(
        1
        for row in kept
        if looks_like_code(row["prompt"]) or looks_like_code(row["completion"])
    )
    if leftover_code:
        raise ManifestError(
            f"STOP_DATA: {leftover_code} residual code-like rows remain after host filtering"
        )
    stats["load_errors"] = errors
    stats["n_empty_pair"] = sum(empty_by_source.values())
    stats["parsed_by_source"] = parsed_by_source
    stats["empty_by_source"] = empty_by_source
    needed = N_HOST_TRAIN + N_CALIBRATION + N_EXPERT_VALIDATION + N_READINESS_HOST
    if len(kept) < needed:
        raise ManifestError(
            f"STOP_DATA: {language} has only {len(kept)} natural-language target-language "
            f"instruction/response examples after filtering code/non-target rows "
            f"(raw unique={len(rows)}, code={stats['n_code']}, "
            f"wrong_language={stats['n_wrong_language']}); need at least {needed}. "
            "GSM8K host/calibration supplement is forbidden. "
            + ("Load errors: " + "; ".join(errors) if errors else "No extra sources loaded.")
        )
    contributing = [
        source
        for source, counts in (stats.get("by_source") or {}).items()
        if int(counts.get("n_kept") or 0) > 0
    ]
    if len(HOST_SOURCE_CANDIDATES[language]) >= 2 and len(contributing) < 2:
        raise ManifestError(
            f"STOP_DATA: {language} host pool is not source-balanced after filtering; "
            f"contributing={contributing}; by_source={stats.get('by_source')}"
        )
    for row in kept:
        row["filter_stats"] = {"applied": True}
    return kept, stats


def _instruction_pair(row: Mapping[str, Any]) -> tuple[str, str]:
    return instruction_pair_from_row(row)


def _load_belebele(language: str, n: int = N_BELEBELE) -> List[Dict[str, Any]]:
    config = BELEBELE_CONFIGS[language]
    dataset = load_dataset(
        "facebook/belebele",
        config,
        split="test",
        download_mode="reuse_dataset_if_exists",
    )
    ranked = sorted(
        range(len(dataset)),
        key=lambda index: hashlib.sha256(
            f"belebele:{config}:{index}".encode("utf-8")
        ).hexdigest(),
    )
    chosen = ranked[:n]
    if len(chosen) != n:
        raise ManifestError(f"Belebele {config} has only {len(dataset)} rows; need {n}")
    rows = []
    for index in chosen:
        row = dataset[int(index)]
        question = str(row.get("question") or "").strip()
        choices = [
            str(row.get(key) or "").strip()
            for key in ("mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4")
        ]
        correct = str(row.get("correct_answer_num") or row.get("gold") or "").strip()
        if not question or not all(choices) or not correct:
            raise ManifestError(f"Belebele {config} row {index} is incomplete")
        rows.append(
            {
                "source_id": f"belebele:{config}:{index}",
                "question": question,
                "choices": choices,
                "answer_number": str(correct),
                "language": language,
                "flores_passage": str(row.get("flores_passage") or "").strip(),
            }
        )
    return rows


def _split_mgsm(language: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    test_rows = _load_mgsm(language, "test")
    if len(test_rows) != 250:
        raise ManifestError(
            f"MGSM {language} test has {len(test_rows)} rows; expected 250"
        )
    split = frozen_mgsm_indices()
    if len(split["dev"]) != N_MGSM_DEV or len(split["test"]) != N_MGSM_TEST:
        raise ManifestError("frozen MGSM split sizes are wrong")
    if set(split["dev"]) & set(split["test"]):
        raise ManifestError("MGSM 64/186 split overlaps")
    development = []
    holdout = []
    for index in split["dev"]:
        row = dict(test_rows[index])
        row["split_index"] = int(index)
        development.append(row)
    for index in split["test"]:
        row = dict(test_rows[index])
        row["split_index"] = int(index)
        holdout.append(row)
    return development, holdout


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
                "source_id": f"mgsm:item:{index}",
                "mgsm_language": language,
                "mgsm_split": split,
                "mgsm_index": index,
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
            "source_dataset": row.get("source_dataset", "licensed_target_language_pool"),
        }
        for row in rows
    ]


def build(
    *,
    language: str,
    output_dir: str | Path,
    n_util: int = N_UTIL,
    n_risk: int = N_RISK,
    n_calibration: int = N_CALIBRATION,
    n_host_train: int = N_HOST_TRAIN,
    translation_batch_size: int = 1,
    max_new_tokens: int = 96,
    device: str | None = None,
) -> Dict[str, Any]:
    if language not in LANGUAGE_NAMES:
        raise ManifestError(f"language must be one of {sorted(LANGUAGE_NAMES)}")
    target = Path(output_dir)
    refuse_frozen_write(target)
    pool = _load_translation_pool()
    native_pool, host_filter_stats = _load_native_instruction_pool(language)
    if n_risk > len(pool):
        raise ManifestError(f"need {n_risk} translation records but only {len(pool)} are available")
    if n_host_train < N_HOST_TRAIN:
        raise ManifestError(f"host train must have at least {N_HOST_TRAIN} examples")
    needed_native = n_host_train + n_calibration + N_EXPERT_VALIDATION + N_READINESS_HOST
    if len(native_pool) < needed_native:
        raise ManifestError(
            f"STOP_DATA: {language} native pool has {len(native_pool)} rows; need {needed_native}"
        )
    translator = LocalTranslator(
        batch_size=translation_batch_size,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    ordered = balanced_round_robin(
        native_pool, needed_native, salt=f"v3-host-{language}"  # frozen sampling salt; do not change
    )
    host_pool = list(ordered[:n_host_train])
    calibration_pool = list(ordered[n_host_train : n_host_train + n_calibration])
    host_validation_pool = list(
        ordered[
            n_host_train + n_calibration : n_host_train + n_calibration + N_EXPERT_VALIDATION
        ]
    )
    host_readiness_pool = list(
        ordered[
            n_host_train + n_calibration + N_EXPERT_VALIDATION :
            n_host_train + n_calibration + N_EXPERT_VALIDATION + N_READINESS_HOST
        ]
    )
    _assert_natural_host_split(host_pool, language, "host_train", require_source_balance=True)
    _assert_natural_host_split(calibration_pool, language, "C", require_source_balance=True)
    _assert_natural_host_split(host_validation_pool, language, "expert_validation_host")
    _assert_natural_host_split(host_readiness_pool, language, "readiness_host")
    fallback_mode = "native_target_instruction_pool"
    risk_pool = pool[:n_risk]
    risk_translations = translator.translate(
        [row["source_text"] for row in risk_pool],
        language,
    )
    gsm_rows = _load_gsm8k_train(n_util)
    utility_translations = translator.translate(
        [row["source_question"] for row in gsm_rows],
        language,
    )
    donor_validation = _load_gsm8k_train(N_EXPERT_VALIDATION, start=4000)
    donor_readiness = _load_gsm8k_train(N_READINESS_DONOR, start=5000)
    used_gsm = {row["source_id"] for row in gsm_rows}
    if any(row["source_id"] in used_gsm for row in donor_validation + donor_readiness):
        raise ManifestError("donor validation/readiness overlaps P_util GSM8K range")
    if {row["source_id"] for row in donor_validation} & {row["source_id"] for row in donor_readiness}:
        raise ManifestError("donor validation overlaps donor readiness")
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
    development, test = _split_mgsm(language)
    belebele = _load_belebele(language)
    expert_validation_donor = [
        {
            "source_id": row["source_id"],
            "question": row["source_question"],
            "answer_number": row["answer_number"],
            "language": "en",
            "purpose": "checkpoint_selection_only",
        }
        for row in donor_validation
    ]
    expert_validation_host = [
        {
            "source_id": row["source_id"],
            "prompt": row["prompt"],
            "completion": row["completion"],
            "language": language,
            "purpose": "checkpoint_selection_only",
        }
        for row in host_validation_pool
    ]
    readiness_donor = [
        {
            "source_id": row["source_id"],
            "question": row["source_question"],
            "answer_number": row["answer_number"],
            "language": "en",
            "purpose": "final_readiness_only",
        }
        for row in donor_readiness
    ]
    readiness_host = [
        {
            "source_id": row["source_id"],
            "prompt": row["prompt"],
            "completion": row["completion"],
            "language": language,
            "purpose": "final_readiness_only",
        }
        for row in host_readiness_pool
    ]
    target.mkdir(parents=True, exist_ok=True)
    paths = {
        "P_util": target / "P_util.jsonl",
        "P_risk": target / "P_risk.jsonl",
        "C": target / "C.jsonl",
        "MGSM_dev": target / "MGSM_dev.jsonl",
        "MGSM_test": target / "MGSM_test.jsonl",
        "host_train": target / "host_train.jsonl",
        "Belebele": target / "Belebele.jsonl",
        "expert_validation_donor": target / "expert_validation_donor.jsonl",
        "expert_validation_host": target / "expert_validation_host.jsonl",
        "readiness_donor": target / "readiness_donor.jsonl",
        "readiness_host": target / "readiness_host.jsonl",
    }
    native_source = ",".join(sorted({row["source_dataset"] for row in native_pool}))
    payloads = {
        "P_util": (
            p_util,
            {
                "language": language,
                "source_dataset": "openai/gsm8k:main:train",
                "translation_model": MODEL_ID,
                "translation_revision": MODEL_REVISION,
            },
        ),
        "P_risk": (
            p_risk,
            {
                "language": language,
                "source_dataset": "CohereLabsCommunity/multilingual-reward-bench:translation:test",
                "translation_model": MODEL_ID,
                "translation_revision": MODEL_REVISION,
            },
        ),
        "C": (
            calibration,
            {
                "language": language,
                "source_dataset": native_source,
                "translation_model": "dataset-provided",
            },
        ),
        "MGSM_dev": (
            development,
            {
                "language": language,
                "source_dataset": f"juletxara/mgsm:{language}:test",
                "n_items": N_MGSM_DEV,
                "split": "hash_frozen_64_of_250",
            },
        ),
        "MGSM_test": (
            test,
            {
                "language": language,
                "source_dataset": f"juletxara/mgsm:{language}:test",
                "n_items": N_MGSM_TEST,
                "split": "hash_frozen_186_holdout",
            },
        ),
        "host_train": (
            host_train,
            {
                "language": language,
                "source_dataset": native_source,
                "gsm8k_supplement": False,
            },
        ),
        "Belebele": (
            belebele,
            {
                "language": language,
                "source_dataset": f"facebook/belebele:{BELEBELE_CONFIGS[language]}",
                "n_items": N_BELEBELE,
            },
        ),
        "expert_validation_donor": (
            expert_validation_donor,
            {"language": "en", "purpose": "checkpoint_selection_only"},
        ),
        "expert_validation_host": (
            expert_validation_host,
            {"language": language, "purpose": "checkpoint_selection_only"},
        ),
        "readiness_donor": (
            readiness_donor,
            {"language": "en", "purpose": "final_readiness_only"},
        ),
        "readiness_host": (
            readiness_host,
            {"language": language, "purpose": "final_readiness_only"},
        ),
    }
    summaries = {
        name: write_manifest(paths[name], rows, manifest_name=name, metadata=metadata)
        for name, (rows, metadata) in payloads.items()
    }
    loaded = {name: rows for name, (rows, _) in payloads.items()}
    disjoint = assert_pairwise_disjoint(loaded)
    summary = {
        "language": language,
        "contract_version": ACTIVE_CONTRACT_VERSION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "fallback_mode": fallback_mode,
        "gsm8k_host_supplement": False,
        "host_filter": host_filter_stats,
        "checkpoint_selection_rule": "max_gain_among_passing",
        "mgsm_split": {key: list(value) for key, value in frozen_mgsm_indices().items()},
        "manifests": summaries,
        "disjointness": disjoint,
        "counts": {name: len(rows) for name, rows in loaded.items()},
    }
    disk_summary = summarize_manifest_set({name: str(path) for name, path in paths.items()})
    summary["disk_summary"] = disk_summary
    with (target / "data_build_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return summary


HOST_REBUILD_KEEP = (
    "P_util",
    "P_risk",
    "MGSM_dev",
    "MGSM_test",
    "Belebele",
    "expert_validation_donor",
    "readiness_donor",
)
HOST_REBUILD_WRITE = (
    "host_train",
    "C",
    "expert_validation_host",
    "readiness_host",
)


def _assert_natural_host_split(
    rows: Sequence[Mapping[str, Any]],
    language: str,
    label: str,
    *,
    require_source_balance: bool = False,
) -> None:
    n_code = sum(
        1
        for row in rows
        if looks_like_code(str(row.get("prompt") or row.get("target_text") or ""))
        or looks_like_code(str(row.get("completion") or ""))
    )
    if n_code:
        raise ManifestError(
            f"STOP_DATA: {label} has {n_code}/{len(rows)} residual code-like rows"
        )
    sources = {str(row.get("source_dataset") or "") for row in rows}
    sources.discard("")
    if (
        require_source_balance
        and len(HOST_SOURCE_CANDIDATES[language]) >= 2
        and len(sources) < 2
    ):
        raise ManifestError(
            f"STOP_DATA: {label} is not source-balanced; sources={sorted(sources)}"
        )


def rebuild_host_only(
    *,
    language: str,
    output_dir: str | Path,
    n_calibration: int = N_CALIBRATION,
    n_host_train: int = N_HOST_TRAIN,
) -> Dict[str, Any]:
    """Rewrite host/calibration manifests without re-running GPU translation."""

    target = Path(output_dir)
    refuse_frozen_write(target)
    native_pool, host_filter_stats = _load_native_instruction_pool(language)
    needed_native = n_host_train + n_calibration + N_EXPERT_VALIDATION + N_READINESS_HOST
    if len(native_pool) < needed_native:
        raise ManifestError(
            f"STOP_DATA: {language} native pool has {len(native_pool)} rows; need {needed_native}"
        )
    ordered = balanced_round_robin(
        native_pool, needed_native, salt=f"v3-host-{language}"  # frozen sampling salt; do not change
    )
    host_pool = list(ordered[:n_host_train])
    calibration_pool = list(ordered[n_host_train : n_host_train + n_calibration])
    host_validation_pool = list(
        ordered[
            n_host_train + n_calibration : n_host_train + n_calibration + N_EXPERT_VALIDATION
        ]
    )
    host_readiness_pool = list(
        ordered[
            n_host_train + n_calibration + N_EXPERT_VALIDATION :
            n_host_train + n_calibration + N_EXPERT_VALIDATION + N_READINESS_HOST
        ]
    )
    _assert_natural_host_split(host_pool, language, "host_train", require_source_balance=True)
    _assert_natural_host_split(calibration_pool, language, "C", require_source_balance=True)
    _assert_natural_host_split(host_validation_pool, language, "expert_validation_host")
    _assert_natural_host_split(host_readiness_pool, language, "readiness_host")
    paths = {
        "P_util": target / "P_util.jsonl",
        "P_risk": target / "P_risk.jsonl",
        "C": target / "C.jsonl",
        "MGSM_dev": target / "MGSM_dev.jsonl",
        "MGSM_test": target / "MGSM_test.jsonl",
        "host_train": target / "host_train.jsonl",
        "Belebele": target / "Belebele.jsonl",
        "expert_validation_donor": target / "expert_validation_donor.jsonl",
        "expert_validation_host": target / "expert_validation_host.jsonl",
        "readiness_donor": target / "readiness_donor.jsonl",
        "readiness_host": target / "readiness_host.jsonl",
    }
    kept_rows: Dict[str, List[Dict[str, Any]]] = {}
    for name in HOST_REBUILD_KEEP:
        path = paths[name]
        if not path.is_file():
            raise ManifestError(f"STOP_DATA: cannot rebuild host files; missing {path}")
        _header, records = read_manifest(path)
        kept_rows[name] = records
    native_source = ",".join(sorted({row["source_dataset"] for row in native_pool}))
    host_train = _make_native_sft_rows(host_pool, language)
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
    expert_validation_host = [
        {
            "source_id": row["source_id"],
            "prompt": row["prompt"],
            "completion": row["completion"],
            "language": language,
            "purpose": "checkpoint_selection_only",
        }
        for row in host_validation_pool
    ]
    readiness_host = [
        {
            "source_id": row["source_id"],
            "prompt": row["prompt"],
            "completion": row["completion"],
            "language": language,
            "purpose": "final_readiness_only",
        }
        for row in host_readiness_pool
    ]
    rewrite = {
        "C": (
            calibration,
            {
                "language": language,
                "source_dataset": native_source,
                "translation_model": "dataset-provided",
                "rebuild": "host_only",
            },
        ),
        "host_train": (
            host_train,
            {
                "language": language,
                "source_dataset": native_source,
                "gsm8k_supplement": False,
                "rebuild": "host_only",
            },
        ),
        "expert_validation_host": (
            expert_validation_host,
            {"language": language, "purpose": "checkpoint_selection_only", "rebuild": "host_only"},
        ),
        "readiness_host": (
            readiness_host,
            {"language": language, "purpose": "final_readiness_only", "rebuild": "host_only"},
        ),
    }
    summaries = {}
    loaded: Dict[str, List[Dict[str, Any]]] = dict(kept_rows)
    for name, (rows, metadata) in rewrite.items():
        summaries[name] = write_manifest(
            paths[name], rows, manifest_name=name, metadata=metadata
        )
        loaded[name] = rows
    disjoint = assert_pairwise_disjoint(loaded)
    summary_path = target / "data_build_summary.json"
    summary: Dict[str, Any] = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "language": language,
            "contract_version": ACTIVE_CONTRACT_VERSION,
            "host_filter": host_filter_stats,
            "host_rebuild": "host_only_no_translation",
            "gsm8k_host_supplement": False,
            "disjointness": disjoint,
            "counts": {name: len(rows) for name, rows in loaded.items()},
        }
    )
    existing_manifests = dict(summary.get("manifests") or {})
    existing_manifests.update(summaries)
    summary["manifests"] = existing_manifests
    disk_summary = summarize_manifest_set({name: str(path) for name, path in paths.items()})
    summary["disk_summary"] = disk_summary
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=sorted(LANGUAGE_NAMES), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-util", type=int, default=N_UTIL)
    parser.add_argument("--n-risk", type=int, default=N_RISK)
    parser.add_argument("--n-calibration", type=int, default=N_CALIBRATION)
    parser.add_argument("--n-host-train", type=int, default=N_HOST_TRAIN)
    parser.add_argument("--translation-batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--device", default=None)
    parser.add_argument("--rebuild-host-only", action="store_true")
    args = parser.parse_args()
    if args.rebuild_host_only:
        summary = rebuild_host_only(
            language=args.language,
            output_dir=args.output_dir,
            n_calibration=args.n_calibration,
            n_host_train=args.n_host_train,
        )
    else:
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
