"""Build the frozen, label-free B3 probe manifests.

The B3 model pair is evaluated on two disjoint probe sources:

* ``P_util``: the first 500 rows of the pinned GSM8K train split, translated
  into the host language while retaining only the original numeric answer;
* ``P_risk``: 500 independently selected English records from the pinned
  Multilingual RewardBench translation split, translated into the host language
  without importing a task label.

The translation model is the exact local Qwen3-1.7B snapshot already used by
the repository's controlled data pipeline.  This builder writes only the
isolated B3 result root and refuses to replace existing manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from suture.b3_lighton import (
    B3Error,
    contract_path,
    load_contract,
    repo_root,
    sha256_file,
    write_json,
    write_run_manifest,
)
from suture.data_manifest import assert_pairwise_disjoint, write_manifest
from suture.tier_a_config import apply_qwen3_chat


LANGUAGE_LABELS = {
    "fr": "français",
    "zh": "中文",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise B3Error(message)


def _clean_translation(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"^```(?:text|[a-zA-Z_-]+)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"^(?:translation|translated text)\s*:\s*", "", text, flags=re.IGNORECASE)
    _require(bool(text), "translation model returned empty text")
    return text.strip()


def _answer_number(answer: str) -> str:
    matches = re.findall(r"####\s*(-?\d+(?:\.\d+)?)", str(answer))
    _require(bool(matches), "GSM8K row has no parseable final answer")
    return matches[-1]


def _load_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise B3Error("datasets is required to build B3 probes") from exc
    return load_dataset


def load_utility_source(contract: Mapping[str, Any]) -> List[Dict[str, Any]]:
    source = contract["data"]["sources"]["utility"]
    dataset = _load_datasets()(
        source["dataset"],
        source["config"],
        split=source["split"],
        revision=source["revision"],
        download_mode="reuse_dataset_if_exists",
    )
    _require(len(dataset) >= 500, "utility source has fewer than 500 rows")
    rows = []
    for index in range(500):
        row = dataset[index]
        question = str(row.get("question") or "").strip()
        _require(bool(question), f"utility row {index} has an empty question")
        rows.append(
            {
                "source_id": f"{source['dataset']}:{source['config']}:{source['split']}:{index}",
                "source_index": index,
                "source_question": question,
                "answer_number": _answer_number(str(row.get("answer") or "")),
            }
        )
    return rows


def load_risk_source(contract: Mapping[str, Any]) -> List[Dict[str, Any]]:
    source = contract["data"]["sources"]["risk"]
    dataset = _load_datasets()(
        source["dataset"],
        source["config"],
        split=source["split"],
        revision=source["revision"],
        download_mode="reuse_dataset_if_exists",
    )
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(dataset):
        subset = str(row.get("subset") or "")
        english = ""
        kind = ""
        if subset.startswith("translation-en-"):
            english = str(row.get("source") or "").strip()
            kind = "source"
        elif subset.startswith("translation-") and len(subset.split("-")) > 2:
            parts = subset.split("-")
            if parts[2] == "en":
                english = str(row.get("chosen") or "").strip()
                kind = "chosen_en"
        if not english or english in seen:
            continue
        seen.add(english)
        source_id = f"{source['dataset']}:{source['config']}:{source['split']}:{index}:{kind}"
        rows.append(
            {
                "source_id": source_id,
                "source_index": index,
                "source_subset": subset,
                "source_text": english,
            }
        )
    _require(len(rows) >= 500, f"risk source has only {len(rows)} unique English records")
    rows.sort(key=lambda row: hashlib.sha256(
        f"b3-risk:{row['source_id']}".encode("utf-8")
    ).hexdigest())
    return rows[:500]


class B3Translator:
    """Greedy local translation using the frozen Qwen3-1.7B snapshot."""

    def __init__(
        self,
        snapshot: Path,
        *,
        device: str,
        batch_size: int,
        max_new_tokens: int,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise B3Error("torch and transformers are required for B3 translation") from exc
        self.torch = torch
        self.device = torch.device(device)
        _require(
            self.device.type == "cuda" and torch.cuda.is_available(),
            "B3 data translation requires an available CUDA device; refusing CPU fallback",
        )
        _require(snapshot.is_dir(), f"translation snapshot does not exist: {snapshot}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot),
            local_files_only=True,
            trust_remote_code=False,
        )
        _require(self.tokenizer.pad_token_id is not None, "translation tokenizer has no pad token")
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            str(snapshot),
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        ).to(self.device).eval()
        _require(
            all(parameter.dtype == torch.bfloat16 for parameter in self.model.parameters()),
            "translation model did not load in bfloat16",
        )
        self.batch_size = int(batch_size)
        self.max_new_tokens = int(max_new_tokens)
        _require(self.batch_size == 1, "B3 translation batch size is frozen at 1")

    def translate(self, texts: Sequence[str], language: str) -> List[str]:
        _require(language in LANGUAGE_LABELS, f"unsupported B3 language {language!r}")
        outputs: List[str] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            prompts = [
                apply_qwen3_chat(
                    self.tokenizer,
                    (
                        f"Translate the following English text into {LANGUAGE_LABELS[language]}. "
                        "Output only the translation, with no explanation or quotation marks.\n\n"
                        f"English text:\n{text}\n\nTranslation:"
                    ),
                    system="You are a precise translator. Output only the translation.",
                    enable_thinking=False,
                )
                for text in batch
            ]
            encoded = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(self.device)
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True,
                )
            prompt_width = encoded["input_ids"].shape[1]
            outputs.extend(
                _clean_translation(self.tokenizer.decode(row[prompt_width:], skip_special_tokens=True))
                for row in generated
            )
            print(
                f"[b3-translation] {start + 1}-{start + len(batch)}/{len(texts)} {language}",
                flush=True,
            )
        _require(len(outputs) == len(texts), "translation count does not match source count")
        return outputs


def _utility_records(
    rows: Sequence[Mapping[str, Any]],
    translations: Sequence[str],
    tokenizer: Any,
    language: str,
    translation_spec: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    records = []
    for row, translated in zip(rows, translations):
        record_id = f"b3:{language}:P_util:{row['source_id']}"
        records.append(
            {
                "id": record_id,
                "input_text": apply_qwen3_chat(
                    tokenizer,
                    f"{translated}\nAnswer:",
                    enable_thinking=False,
                ),
                "utility_texts": [str(row["answer_number"])],
                "source_id": row["source_id"],
                "source_index": row["source_index"],
                "language": language,
                "translation_model": translation_spec["id"],
                "translation_revision": translation_spec["revision"],
                "source_question_sha256": hashlib.sha256(
                    row["source_question"].encode("utf-8")
                ).hexdigest(),
            }
        )
    return records


def _risk_records(
    rows: Sequence[Mapping[str, Any]],
    translations: Sequence[str],
    tokenizer: Any,
    language: str,
    translation_spec: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    label = LANGUAGE_LABELS[language]
    records = []
    for row, translated in zip(rows, translations):
        record_id = f"b3:{language}:P_risk:{row['source_id']}"
        records.append(
            {
                "id": record_id,
                "input_text": apply_qwen3_chat(
                    tokenizer,
                    f"{translated}\nRespond in {label}:",
                    enable_thinking=False,
                ),
                "risk_target_texts": [label],
                "risk_donor_texts": ["English"],
                "source_id": row["source_id"],
                "source_index": row["source_index"],
                "source_subset": row["source_subset"],
                "language": language,
                "translation_model": translation_spec["id"],
                "translation_revision": translation_spec["revision"],
                "source_text_sha256": hashlib.sha256(
                    row["source_text"].encode("utf-8")
                ).hexdigest(),
            }
        )
    return records


def _measurement_records(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    language: str,
    source: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    records = []
    for row in rows:
        record_id = f"b3:{language}:MGSM_rev2_test:{row['source_index']}"
        records.append(
            {
                "id": record_id,
                "input_text": apply_qwen3_chat(
                    tokenizer,
                    f"{row['question']}\nAnswer:",
                    enable_thinking=False,
                ),
                "answer_number": str(row["answer_number"]),
                "source_id": row["source_id"],
                "source_index": row["source_index"],
                "language": language,
                "dataset": source["dataset"],
                "dataset_revision": source["revision"],
            }
        )
    return records


def _load_measurement_rows(
    contract: Mapping[str, Any],
    language: str,
) -> List[Dict[str, Any]]:
    source = contract["data"]["sources"]["held_out_measurement"]
    config = source["configs"][language]
    dataset = _load_datasets()(
        source["dataset"],
        config,
        split=source["split"],
        revision=source["revision"],
        download_mode="reuse_dataset_if_exists",
    )
    _require(len(dataset) == int(source["n_items"]), f"{language} MGSM-Rev2 size is not 250")
    rows = []
    for index, row in enumerate(dataset):
        question = str(row.get("question") or "").strip()
        answer = row.get("answer_number")
        _require(bool(question) and answer is not None, f"MGSM row {index} is incomplete")
        rows.append(
            {
                "source_id": f"{source['dataset']}:{config}:{source['split']}:{index}",
                "source_index": index,
                "question": question,
                "answer_number": str(answer),
            }
        )
    return rows


def build_measurement_manifest(
    *,
    language: str,
    output_root: Path,
    tokenizer_snapshot: Path,
    contract_file: Path | None = None,
) -> Dict[str, Any]:
    """Write the held-out MGSM-Rev2 manifest without loading model weights."""

    contract_file = contract_file or contract_path()
    contract = load_contract(contract_file)
    _require(language in LANGUAGE_LABELS, f"unsupported B3 language {language!r}")
    data_root = output_root / f"{language}_host__en_donor" / "data"
    target = data_root / "MGSM_rev2_test.jsonl"
    _require(not target.exists(), f"held-out manifest already exists: {target}")
    _require(tokenizer_snapshot.is_dir(), f"tokenizer snapshot does not exist: {tokenizer_snapshot}")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise B3Error("transformers is required to build the held-out manifest") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_snapshot),
        local_files_only=True,
        trust_remote_code=False,
    )
    _require(tokenizer.pad_token_id is not None, "held-out tokenizer has no pad token")
    tokenizer.padding_side = "left"
    source = contract["data"]["sources"]["held_out_measurement"]
    rows = _load_measurement_rows(contract, language)
    records = _measurement_records(rows, tokenizer, language, source)
    max_length = int(contract["data"]["held_out_measurement"]["max_length"])
    for record in records:
        token_count = len(tokenizer(record["input_text"], add_special_tokens=False)["input_ids"])
        _require(
            token_count <= max_length,
            f"{record['id']} exceeds held-out max_length {max_length}",
        )
    selection_paths = {
        "P_util": data_root / "P_util.jsonl",
        "P_risk": data_root / "P_risk.jsonl",
    }
    for name, path in selection_paths.items():
        _require(path.is_file(), f"selection manifest is missing before held-out build: {path}")
    from suture.data_manifest import read_manifest

    _, utility = read_manifest(selection_paths["P_util"])
    _, risk = read_manifest(selection_paths["P_risk"])
    assert_pairwise_disjoint({"P_util": utility, "P_risk": risk, "held_out": records})
    _require(
        {row["id"] for row in records}.isdisjoint(
            {row["id"] for row in utility} | {row["id"] for row in risk}
        ),
        "held-out IDs overlap selection IDs",
    )
    summary = write_manifest(
        target,
        records,
        manifest_name=f"b3_{language}_MGSM_rev2_test",
        metadata={
            "contract_name": contract["contract_name"],
            "contract_version": contract["contract_version"],
            "language": language,
            "source": source,
            "objective": contract["data"]["held_out_measurement"]["objective"],
            "selection_manifests": {
                name: str(path.resolve()) for name, path in selection_paths.items()
            },
        },
    )
    payload = {
        "schema_version": 1,
        "status": "PASS_HELD_OUT_DATA",
        "language": language,
        "manifest": summary,
        "tokenizer_snapshot": str(tokenizer_snapshot.resolve()),
    }
    write_json(data_root / "measurement_data_summary.json", payload)
    write_run_manifest(
        data_root / "measurement_run_manifest.json",
        stage="measurement_data",
        root=repo_root(),
        contract_file=contract_file,
        command=sys.argv,
        inputs=payload,
    )
    return payload


def build(
    *,
    language: str,
    output_root: Path,
    translation_snapshot: Path,
    device: str = "cuda:0",
    contract_file: Path | None = None,
) -> Dict[str, Any]:
    contract_file = contract_file or contract_path()
    contract = load_contract(contract_file)
    _require(language in LANGUAGE_LABELS, f"unsupported B3 language {language!r}")
    selection = contract["data"]["selection"]
    translation_spec = contract["data"]["sources"]["translation"]
    _require(
        sha256_file(translation_snapshot / "config.json") is not None,
        "translation snapshot config could not be hashed",
    )
    pair_root = output_root / f"{language}_host__en_donor"
    data_root = pair_root / "data"
    utility_path = data_root / "P_util.jsonl"
    risk_path = data_root / "P_risk.jsonl"
    _require(not utility_path.exists() and not risk_path.exists(), f"B3 data output already exists: {data_root}")

    utility_source = load_utility_source(contract)
    risk_source = load_risk_source(contract)
    translator = B3Translator(
        translation_snapshot,
        device=device,
        batch_size=int(translation_spec["batch_size"]),
        max_new_tokens=int(translation_spec["max_new_tokens"]),
    )
    utility_translations = translator.translate(
        [row["source_question"] for row in utility_source],
        language,
    )
    risk_translations = translator.translate(
        [row["source_text"] for row in risk_source],
        language,
    )
    utility_records = _utility_records(
        utility_source,
        utility_translations,
        translator.tokenizer,
        language,
        translation_spec,
    )
    risk_records = _risk_records(
        risk_source,
        risk_translations,
        translator.tokenizer,
        language,
        translation_spec,
    )
    _require(len(utility_records) == int(selection["minimum_utility_records"]) + 492, "utility count is not 500")
    _require(len(risk_records) == int(selection["minimum_risk_records"]) + 492, "risk count is not 500")
    assert_pairwise_disjoint({"P_util": utility_records, "P_risk": risk_records})
    _require(
        {row["id"] for row in utility_records}.isdisjoint({row["id"] for row in risk_records}),
        "B3 probe IDs overlap",
    )

    metadata = {
        "contract_name": contract["contract_name"],
        "contract_version": contract["contract_version"],
        "language": language,
        "translation_model": translation_spec,
        "source_provenance": contract["data"]["sources"],
        "selection_max_length": selection["max_length"],
        "readout_uses_first_token_only": True,
    }
    utility_summary = write_manifest(
        utility_path,
        utility_records,
        manifest_name=f"b3_{language}_P_util",
        metadata=metadata,
    )
    risk_summary = write_manifest(
        risk_path,
        risk_records,
        manifest_name=f"b3_{language}_P_risk",
        metadata=metadata,
    )
    summary = {
        "schema_version": 1,
        "status": "PASS_DATA",
        "contract_name": contract["contract_name"],
        "language": language,
        "utility": utility_summary,
        "risk": risk_summary,
        "disjointness": {
            "pairwise_disjoint": True,
            "id_sets_disjoint": True,
        },
        "translation_snapshot": str(translation_snapshot.resolve()),
        "translation_config_sha256": sha256_file(translation_snapshot / "config.json"),
    }
    write_json(data_root / "data_build_summary.json", summary)
    write_run_manifest(
        data_root / "run_manifest.json",
        stage="data",
        root=repo_root(),
        contract_file=contract_file,
        command=sys.argv,
        inputs=summary,
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=tuple(LANGUAGE_LABELS), required=True)
    parser.add_argument("--translation-snapshot", type=Path, required=True)
    parser.add_argument("--measurement-only", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root() / "results" / "b3" / "lighton_qwen3_8b",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--contract", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.measurement_only:
            summary = build_measurement_manifest(
                language=args.language,
                output_root=args.output_root,
                tokenizer_snapshot=args.translation_snapshot,
                contract_file=args.contract,
            )
        else:
            summary = build(
                language=args.language,
                output_root=args.output_root,
                translation_snapshot=args.translation_snapshot,
                device=args.device,
                contract_file=args.contract,
            )
    except B3Error as exc:
        print(f"STOP_DATA: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
