"""Validate that Tier-A experts are non-degenerate before scientific runs."""

from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import pyarrow  # noqa: F401  — must load before torch on Windows
import fasttext
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from suture.data_manifest import ManifestError, read_manifest
    from suture.tier_a_config import (
        EXPECTED_LID,
        MAX_BELEBELE_DROP,
        MIN_DONOR_EM,
        MIN_DONOR_GAIN,
        MIN_HOST_LOGPROB_GAIN,
        MIN_HOST_NONEMPTY,
        MIN_HOST_TARGET_LANGUAGE,
        MODEL_ID,
        MODEL_REVISION,
        assert_adapter_matches_pin,
        refuse_v1_write,
        refuse_frozen_write,
        repo_root,
    )
    from suture.tier_a_preflight import pretrained_load_args
except ModuleNotFoundError:
    from data_manifest import ManifestError, read_manifest
    from tier_a_config import (
        EXPECTED_LID,
        MAX_BELEBELE_DROP,
        MIN_DONOR_EM,
        MIN_DONOR_GAIN,
        MIN_HOST_LOGPROB_GAIN,
        MIN_HOST_NONEMPTY,
        MIN_HOST_TARGET_LANGUAGE,
        MODEL_ID,
        MODEL_REVISION,
        assert_adapter_matches_pin,
        refuse_v1_write,
        refuse_frozen_write,
        repo_root,
    )
    from tier_a_preflight import pretrained_load_args


def _lid_predict(lid_model: Any, text: str) -> tuple[List[str], List[float]]:
    predictions = lid_model.f.predict(text or " ", 1, 0.0, "strict")
    if not predictions:
        return [], []
    probabilities, labels = zip(*predictions)
    return list(labels), [float(value) for value in probabilities]


def _records(path: Path) -> List[Dict[str, Any]]:
    _, rows = read_manifest(path)
    return rows


def _adapter_path(root: Path, role: str) -> Path:
    root = root.resolve()
    direct = root / "adapter_config.json"
    nested = root / role / "adapter_config.json"
    if direct.is_file():
        return root
    if nested.is_file():
        return nested.parent
    raise ManifestError(f"adapter config not found below {root}")


def _load_model(
    *,
    device: torch.device,
    adapter_dir: Path | None = None,
    adapter_name: str = "adapter",
):
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    source = pretrained_load_args(repo_root())
    tokenizer = AutoTokenizer.from_pretrained(
        source["pretrained_model_name_or_path"],
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        source["pretrained_model_name_or_path"],
        local_files_only=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device).eval()
    if adapter_dir is not None:
        assert_adapter_matches_pin(adapter_dir)
        model = PeftModel.from_pretrained(
            model,
            str(adapter_dir),
            adapter_name=adapter_name,
            is_trainable=False,
        ).to(device).eval()
        model.set_adapter(adapter_name)
    return model, tokenizer


def _generate(
    model,
    tokenizer,
    texts: Sequence[str],
    *,
    device: torch.device,
    max_length: int,
    max_new_tokens: int,
) -> List[str]:
    encoded = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)
    prompt_width = encoded["input_ids"].shape[1]
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    return [
        tokenizer.decode(row[prompt_width:], skip_special_tokens=True).strip()
        for row in generated
    ]


def _number_matches(value: str, expected: str) -> bool:
    matches = re.findall(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
    if not matches:
        return False
    try:
        return Decimal(matches[-1]) == Decimal(str(expected))
    except InvalidOperation:
        return matches[-1] == str(expected)


def readiness_gate(
    *,
    donor_gain: float,
    host_nonempty_rate: float,
    host_target_language_rate: float,
    min_nonempty_rate: float,
    min_target_language_rate: float,
    donor_em: float | None = None,
    host_logprob_gain: float | None = None,
    belebele_drop: float | None = None,
    min_donor_gain: float = 0.0,
    min_donor_em: float | None = None,
    min_host_logprob_gain: float | None = None,
    max_belebele_drop: float | None = None,
) -> str:
    if donor_gain <= min_donor_gain:
        return "STOP_EXPERT_READINESS"
    if host_nonempty_rate < min_nonempty_rate:
        return "STOP_EXPERT_READINESS"
    if host_target_language_rate < min_target_language_rate:
        return "STOP_EXPERT_READINESS"
    if min_donor_em is not None and (donor_em is None or donor_em < min_donor_em):
        return "STOP_EXPERT_READINESS"
    if min_host_logprob_gain is not None and (
        host_logprob_gain is None or host_logprob_gain < min_host_logprob_gain
    ):
        return "STOP_EXPERT_READINESS"
    if max_belebele_drop is not None and (
        belebele_drop is None or belebele_drop > max_belebele_drop
    ):
        return "STOP_EXPERT_READINESS"
    return "PASS_READINESS"


def _donor_eval(
    model,
    tokenizer,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        prompts = [
            "Solve the following math problem in English.\n"
            f"Question:\n{row['question']}\nAnswer:"
            for row in batch
        ]
        generations = _generate(
            model,
            tokenizer,
            prompts,
            device=device,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
        )
        rows.extend(
            {
                "source_id": row["source_id"],
                "expected_answer": str(row["answer_number"]),
                "generation": generation,
                "exact_match": _number_matches(
                    generation,
                    str(row["answer_number"]),
                ),
            }
            for row, generation in zip(batch, generations)
        )
    return {
        "n": len(rows),
        "exact_match": (
            float(sum(bool(row["exact_match"]) for row in rows) / len(rows))
            if rows
            else 0.0
        ),
        "items": rows,
    }


def _host_eval(
    model,
    tokenizer,
    lid_model,
    records: Sequence[Mapping[str, Any]],
    *,
    language: str,
    device: torch.device,
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    expected = EXPECTED_LID[language]
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        prompts = [str(row["prompt"]) for row in batch]
        generations = _generate(
            model,
            tokenizer,
            prompts,
            device=device,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
        )
        for row, generation in zip(batch, generations):
            text = generation.replace("\n", " ").strip()
            labels, probabilities = _lid_predict(lid_model, text)
            label = str(labels[0]) if labels else ""
            probability = float(probabilities[0]) if probabilities else 0.0
            rows.append(
                {
                    "source_id": row["source_id"],
                    "generation": generation,
                    "nonempty": bool(text),
                    "token_count": len(text.split()),
                    "language_label": label,
                    "language_probability": probability,
                    "target_language": language,
                    "target_language_match": label == expected,
                }
            )
    return {
        "n": len(rows),
        "nonempty_rate": (
            float(sum(bool(row["nonempty"]) for row in rows) / len(rows))
            if rows
            else 0.0
        ),
        "target_language_rate": (
            float(
                sum(bool(row["target_language_match"]) for row in rows)
                / len(rows)
            )
            if rows
            else 0.0
        ),
        "items": rows,
    }


def _mean_completion_logprob(
    model,
    tokenizer,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    max_length: int,
) -> Dict[str, Any]:
    try:
        from suture.tier_a_eval import mean_completion_logprob
    except ModuleNotFoundError:
        from tier_a_eval import mean_completion_logprob
    scored = mean_completion_logprob(
        model,
        tokenizer,
        records,
        device=device,
        max_length=max_length,
        declared_n=len(records),
    )
    return {
        "n": scored["n"],
        "mean_nats_per_token": scored["mean_nats_per_token"],
        "complete_denominator": True,
        "scorer": scored["scorer"],
    }


def _belebele_eval(
    model,
    tokenizer,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    max_length: int,
) -> Dict[str, Any]:
    try:
        from suture.tier_a_eval import belebele_mc_eval
    except ModuleNotFoundError:
        from tier_a_eval import belebele_mc_eval
    return belebele_mc_eval(
        model,
        tokenizer,
        records,
        device=device,
        max_length=max_length,
        declared_n=len(records),
    )


def run(
    *,
    language: str,
    readiness_donor: Path,
    readiness_host: Path,
    host_adapter: Path,
    donor_adapter: Path,
    lid_model_path: Path,
    output: Path,
    device: str,
    batch_size: int = 1,
    max_length: int = 128,
    max_new_tokens: int = 128,
    min_target_language_rate: float = MIN_HOST_TARGET_LANGUAGE,
    min_nonempty_rate: float = MIN_HOST_NONEMPTY,
    belebele_path: Path | None = None,
    contract_version: int = 3,
) -> Dict[str, Any]:
    refuse_frozen_write(output)
    if language not in EXPECTED_LID:
        raise ManifestError(f"unsupported readiness language: {language}")
    if not lid_model_path.is_file():
        raise ManifestError(f"fastText language-ID model not found: {lid_model_path}")
    target_device = torch.device(device)
    donor_rows = _records(readiness_donor)
    host_rows = _records(readiness_host)
    belebele_rows = _records(belebele_path) if belebele_path is not None else []
    lid = fasttext.load_model(str(lid_model_path))

    base, tokenizer = _load_model(device=target_device)
    donor_base = _donor_eval(
        base,
        tokenizer,
        donor_rows,
        device=target_device,
        batch_size=batch_size,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
    )
    host_base_lp = _mean_completion_logprob(
        base, tokenizer, host_rows, device=target_device, max_length=max_length
    )
    belebele_base = (
        _belebele_eval(base, tokenizer, belebele_rows, device=target_device, max_length=max_length)
        if belebele_rows
        else None
    )
    del base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    donor_model, donor_tokenizer = _load_model(
        device=target_device,
        adapter_dir=_adapter_path(donor_adapter, "donor"),
        adapter_name="donor",
    )
    donor_expert = _donor_eval(
        donor_model,
        donor_tokenizer,
        donor_rows,
        device=target_device,
        batch_size=batch_size,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
    )
    del donor_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    host_base_model, host_base_tokenizer = _load_model(device=target_device)
    host_base = _host_eval(
        host_base_model,
        host_base_tokenizer,
        lid,
        host_rows,
        language=language,
        device=target_device,
        batch_size=batch_size,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
    )
    del host_base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    host_model, host_tokenizer = _load_model(
        device=target_device,
        adapter_dir=_adapter_path(host_adapter, "host"),
        adapter_name="host",
    )
    host_expert = _host_eval(
        host_model,
        host_tokenizer,
        lid,
        host_rows,
        language=language,
        device=target_device,
        batch_size=batch_size,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
    )
    host_expert_lp = _mean_completion_logprob(
        host_model, host_tokenizer, host_rows, device=target_device, max_length=max_length
    )
    belebele_host = (
        _belebele_eval(
            host_model, host_tokenizer, belebele_rows, device=target_device, max_length=max_length
        )
        if belebele_rows
        else None
    )
    del host_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    donor_gain = donor_expert["exact_match"] - donor_base["exact_match"]
    logprob_gain = host_expert_lp["mean_nats_per_token"] - host_base_lp["mean_nats_per_token"]
    belebele_drop = None
    if belebele_base is not None and belebele_host is not None:
        belebele_drop = float(belebele_base["accuracy"] - belebele_host["accuracy"])
    qwen3 = int(contract_version) >= 2
    report: Dict[str, Any] = {
        "language": language,
        "contract_version": int(contract_version),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "lid_model": str(lid_model_path),
        "readiness_donor": donor_base,
        "donor_expert": donor_expert,
        "host_base": host_base,
        "host_expert": host_expert,
        "host_base_logprob": host_base_lp,
        "host_expert_logprob": host_expert_lp,
        "host_logprob_gain": logprob_gain,
        "belebele_base": belebele_base,
        "belebele_host": belebele_host,
        "belebele_drop": belebele_drop,
        "thresholds": {
            "min_target_language_rate": min_target_language_rate,
            "min_nonempty_rate": min_nonempty_rate,
            "min_donor_gain": MIN_DONOR_GAIN if qwen3 else 0.0,
            "min_donor_em": MIN_DONOR_EM if qwen3 else None,
            "min_host_logprob_gain": MIN_HOST_LOGPROB_GAIN if qwen3 else None,
            "max_belebele_drop": MAX_BELEBELE_DROP if qwen3 else None,
        },
        "donor_gain": donor_gain,
    }
    report["gate"] = readiness_gate(
        donor_gain=float(donor_gain),
        host_nonempty_rate=float(host_expert["nonempty_rate"]),
        host_target_language_rate=float(host_expert["target_language_rate"]),
        min_nonempty_rate=min_nonempty_rate,
        min_target_language_rate=min_target_language_rate,
        donor_em=float(donor_expert["exact_match"]),
        host_logprob_gain=float(logprob_gain),
        belebele_drop=belebele_drop,
        min_donor_gain=MIN_DONOR_GAIN if qwen3 else 0.0,
        min_donor_em=MIN_DONOR_EM if qwen3 else None,
        min_host_logprob_gain=MIN_HOST_LOGPROB_GAIN if qwen3 else None,
        max_belebele_drop=MAX_BELEBELE_DROP if qwen3 else None,
    )
    setup_issues = []
    if int(host_base_lp.get("n") or 0) != len(host_rows) or int(
        host_expert_lp.get("n") or 0
    ) != len(host_rows):
        setup_issues.append("incomplete_logprob_denominator")
    if belebele_rows and (
        int((belebele_base or {}).get("n") or 0) != len(belebele_rows)
        or int((belebele_host or {}).get("n") or 0) != len(belebele_rows)
        or (belebele_host or {}).get("scorer") != "multiple_choice_logprob"
    ):
        setup_issues.append("belebele_scorer_or_denominator")
    report["setup_issues"] = setup_issues
    report["denominator_complete"] = not setup_issues
    if setup_issues:
        report["gate"] = "STOP_SETUP"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", required=True, choices=sorted(EXPECTED_LID))
    parser.add_argument("--readiness-donor", required=True, type=Path)
    parser.add_argument("--readiness-host", required=True, type=Path)
    parser.add_argument("--host-adapter", required=True, type=Path)
    parser.add_argument("--donor-adapter", required=True, type=Path)
    parser.add_argument("--lid-model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--min-target-language-rate", type=float, default=MIN_HOST_TARGET_LANGUAGE)
    parser.add_argument("--min-nonempty-rate", type=float, default=MIN_HOST_NONEMPTY)
    parser.add_argument("--belebele", type=Path)
    parser.add_argument("--contract-version", type=int, default=3)
    args = parser.parse_args()
    report = run(
        language=args.language,
        readiness_donor=args.readiness_donor,
        readiness_host=args.readiness_host,
        host_adapter=args.host_adapter,
        donor_adapter=args.donor_adapter,
        lid_model_path=args.lid_model,
        output=args.output,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        min_target_language_rate=args.min_target_language_rate,
        min_nonempty_rate=args.min_nonempty_rate,
        belebele_path=args.belebele,
        contract_version=args.contract_version,
    )
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
