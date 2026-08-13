"""Train matched Tier-A LoRA experts from one pinned base checkpoint.

This is intentionally a small, auditable gate recipe rather than a claim that
the resulting adapters are publication-quality.  Host and donor are trained
with the same optimizer, sequence length, batch convention, number of steps,
and seed from the same frozen base revision.  The outputs are PEFT adapters so
the two experts can later share one GPU-resident base during SUTURE scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from paper_iclr.data_manifest import ManifestError, read_manifest
except ModuleNotFoundError:
    from data_manifest import ManifestError, read_manifest


MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MODEL_REVISION = "2e1fd397ee46e1388853d2af2c993145b0f1098a"
DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_records(path: Path) -> List[Dict[str, Any]]:
    _, records = read_manifest(path)
    return records


def _make_examples(
    *,
    role: str,
    host_records: Sequence[Mapping[str, Any]],
    utility_records: Sequence[Mapping[str, Any]],
) -> List[Tuple[str, str]]:
    if role == "host":
        examples = [
            (str(row["prompt"]), str(row["completion"]))
            for row in host_records
            if row.get("prompt") and row.get("completion")
        ]
    elif role == "donor":
        examples = [
            (
                "Solve the following math problem in English.\nQuestion:\n"
                + str(row["source_question"]),
                "#### " + str(row["answer_number"]),
            )
            for row in utility_records
            if row.get("source_question") and row.get("answer_number") is not None
        ]
    else:
        raise ManifestError(f"unknown expert role: {role}")
    if not examples:
        raise ManifestError(f"no training examples available for {role}")
    return examples


def _batch_encode(
    tokenizer,
    examples: Sequence[Tuple[str, str]],
    indices: Sequence[int],
    *,
    max_length: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    texts = [examples[i][0] + "\n" + examples[i][1] for i in indices]
    prompts = [examples[i][0] for i in indices]
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    labels = encoded["input_ids"].clone()
    prompt_encoded = tokenizer(
        prompts,
        return_tensors=None,
        padding=False,
        truncation=True,
        max_length=max_length,
    )
    for row, prompt_ids in enumerate(prompt_encoded["input_ids"]):
        prompt_len = min(len(prompt_ids), labels.shape[1])
        labels[row, :prompt_len] = -100
    labels[encoded["attention_mask"] == 0] = -100
    return {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
        "labels": labels.to(device),
    }


def train_role(
    *,
    role: str,
    data_dir: Path,
    output_dir: Path,
    language: str,
    seed: int,
    steps: int,
    batch_size: int,
    grad_accumulation: int,
    max_length: int,
    learning_rate: float,
    lora_rank: int,
    device: str | None,
) -> Dict[str, Any]:
    if steps <= 0 or batch_size <= 0 or grad_accumulation <= 0:
        raise ManifestError("steps, batch_size, and grad_accumulation must be positive")
    utility_path = data_dir / "P_util.jsonl"
    host_path = data_dir / "host_train.jsonl"
    utility_records = _read_records(utility_path)
    host_records = _read_records(host_path)
    examples = _make_examples(
        role=role,
        host_records=host_records,
        utility_records=utility_records,
    )

    _seed_everything(seed)
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if target_device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(target_device)
    model.config.use_cache = False
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=2 * lora_rank,
        lora_dropout=0.05,
        target_modules=list(DEFAULT_TARGET_MODULES),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(model, lora_config, adapter_name=role)
    if hasattr(peft_model, "gradient_checkpointing_enable"):
        peft_model.gradient_checkpointing_enable()
    if hasattr(peft_model, "enable_input_require_grads"):
        peft_model.enable_input_require_grads()
    peft_model.train()
    trainable = [parameter for parameter in peft_model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ManifestError(f"{role} PEFT model has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.0)

    order = np.arange(len(examples))
    losses: List[float] = []
    optimizer.zero_grad(set_to_none=True)
    for step in range(steps):
        if step % len(order) == 0:
            if step:
                np.random.default_rng(seed + step).shuffle(order)
        start = (step * batch_size) % len(order)
        indices = [int(order[(start + offset) % len(order)]) for offset in range(batch_size)]
        batch = _batch_encode(
            tokenizer,
            examples,
            indices,
            max_length=max_length,
            device=target_device,
        )
        with torch.autocast(
            device_type=target_device.type,
            dtype=torch.bfloat16 if target_device.type == "cuda" else torch.float32,
            enabled=target_device.type == "cuda",
        ):
            loss = peft_model(**batch).loss
        if loss is None or not torch.isfinite(loss):
            raise ManifestError(f"{role} produced a non-finite loss at step {step}")
        (loss / grad_accumulation).backward()
        if (step + 1) % grad_accumulation == 0 or step + 1 == steps:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
        if (step + 1) % max(1, min(10, steps // 4 or 1)) == 0:
            print(f"[{role}:{language}] step {step + 1}/{steps} loss={losses[-1]:.5f}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    peft_model.eval()
    peft_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    metadata = {
        "role": role,
        "language": language,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "gradient_accumulation": grad_accumulation,
        "max_length": max_length,
        "learning_rate": learning_rate,
        "lora_rank": lora_rank,
        "target_modules": list(DEFAULT_TARGET_MODULES),
        "n_examples": len(examples),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_mean": float(np.mean(losses)),
        "data_files": {
            "P_util": _file_sha256(utility_path),
            "host_train": _file_sha256(host_path),
        },
        "device": str(target_device),
        "dtype": str(dtype),
        "trainable_parameters": int(sum(parameter.numel() for parameter in trainable)),
        "total_parameters": int(sum(parameter.numel() for parameter in peft_model.parameters())),
    }
    with (output_dir / "training_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    del peft_model, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["host", "donor"], required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    metadata = train_role(
        role=args.role,
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        language=args.language,
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        grad_accumulation=args.grad_accumulation,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        device=args.device,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
