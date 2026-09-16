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
    from suture.data_manifest import ManifestError, read_manifest
    from suture.tier_a_config import (
        CHECKPOINT_STEPS,
        DEFAULT_TARGET_MODULES,
        EXPERT_SEED,
        LORA_RANK,
        MODEL_ID,
        MODEL_REVISION,
        TRAIN_BATCH_SIZE,
        TRAIN_GRAD_ACCUM,
        TRAIN_LEARNING_RATE,
        TRAIN_MAX_LENGTH,
        TRAIN_STEPS,
        refuse_v1_write,
        repo_root,
    )
    from suture.tier_a_preflight import pretrained_load_args
except ModuleNotFoundError:
    from data_manifest import ManifestError, read_manifest
    from tier_a_config import (
        CHECKPOINT_STEPS,
        DEFAULT_TARGET_MODULES,
        EXPERT_SEED,
        LORA_RANK,
        MODEL_ID,
        MODEL_REVISION,
        TRAIN_BATCH_SIZE,
        TRAIN_GRAD_ACCUM,
        TRAIN_LEARNING_RATE,
        TRAIN_MAX_LENGTH,
        TRAIN_STEPS,
        refuse_v1_write,
        repo_root,
    )
    from tier_a_preflight import pretrained_load_args


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pin_adapter_config(adapter_dir: Path) -> None:
    """Rewrite PEFT configs so they name the official Qwen3 repo, not a local path."""

    for config_path in adapter_dir.rglob("adapter_config.json"):
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        payload["base_model_name_or_path"] = MODEL_ID
        payload["revision"] = MODEL_REVISION
        payload["suture_contract_version"] = 3
        config_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


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
    if max_length < 2:
        raise ManifestError("max_length must leave room for prompt and completion tokens")
    input_rows: List[List[int]] = []
    label_rows: List[List[int]] = []
    for index in indices:
        prompt_ids = tokenizer(
            examples[index][0],
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]
        completion_ids = tokenizer(
            "\n" + examples[index][1],
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        if not completion_ids:
            raise ManifestError(f"example {index} has no completion tokens")
        if len(completion_ids) >= max_length:
            prompt_ids = []
            completion_ids = completion_ids[:max_length]
        else:
            prompt_budget = max_length - len(completion_ids)
            prompt_ids = prompt_ids[-prompt_budget:]
        input_ids = list(prompt_ids) + list(completion_ids)
        input_rows.append(input_ids)
        label_rows.append([-100] * len(prompt_ids) + list(completion_ids))
    encoded = tokenizer.pad(
        {"input_ids": input_rows},
        padding=True,
        return_tensors="pt",
    )
    labels = torch.full_like(encoded["input_ids"], -100)
    for row, row_labels in enumerate(label_rows):
        labels[row, : len(row_labels)] = torch.as_tensor(
            row_labels,
            dtype=labels.dtype,
        )
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
    checkpoint_steps: Sequence[int] = (),
    max_grad_norm: float = 0.5,
) -> Dict[str, Any]:
    if steps <= 0 or batch_size <= 0 or grad_accumulation <= 0:
        raise ManifestError("steps, batch_size, and grad_accumulation must be positive")
    if max_grad_norm <= 0:
        raise ManifestError("max_grad_norm must be positive")
    checkpoints = tuple(sorted(set(int(step) for step in checkpoint_steps)))
    if any(step <= 0 or step > steps for step in checkpoints):
        raise ManifestError("checkpoint steps must be in the inclusive range [1, steps]")
    refuse_v1_write(output_dir)
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
    source = pretrained_load_args(repo_root())
    tokenizer = AutoTokenizer.from_pretrained(
        source["pretrained_model_name_or_path"],
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        source["pretrained_model_name_or_path"],
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

    output_dir.mkdir(parents=True, exist_ok=True)
    order = np.arange(len(examples))
    losses: List[float] = []
    optimizer.zero_grad(set_to_none=True)

    def save_checkpoint(step: int) -> None:
        checkpoint_dir = output_dir / f"checkpoint-{step:04d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        peft_model.save_pretrained(checkpoint_dir, safe_serialization=True)
        tokenizer.save_pretrained(checkpoint_dir)
        _pin_adapter_config(checkpoint_dir)

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
            torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
        if (step + 1) % max(1, min(10, steps // 4 or 1)) == 0:
            print(f"[{role}:{language}] step {step + 1}/{steps} loss={losses[-1]:.5f}", flush=True)
        if step + 1 in checkpoints:
            save_checkpoint(step + 1)

    peft_model.eval()
    peft_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    _pin_adapter_config(output_dir)
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
        "max_grad_norm": max_grad_norm,
        "lora_rank": lora_rank,
        "target_modules": list(DEFAULT_TARGET_MODULES),
        "n_examples": len(examples),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_mean": float(np.mean(losses)),
        "checkpoint_steps": list(checkpoints),
        "checkpoint_paths": [
            str(output_dir / f"checkpoint-{step:04d}")
            for step in checkpoints
        ],
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
    parser.add_argument("--seed", type=int, default=EXPERT_SEED)
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--grad-accumulation", type=int, default=TRAIN_GRAD_ACCUM)
    parser.add_argument("--max-length", type=int, default=TRAIN_MAX_LENGTH)
    parser.add_argument("--learning-rate", type=float, default=TRAIN_LEARNING_RATE)
    parser.add_argument("--lora-rank", type=int, default=LORA_RANK)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--checkpoint-steps",
        default=",".join(str(step) for step in CHECKPOINT_STEPS),
        help="comma-separated training-step checkpoints, e.g. 128,256,512,768,1024",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    checkpoint_steps = tuple(
        int(value.strip())
        for value in args.checkpoint_steps.split(",")
        if value.strip()
    )
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
        checkpoint_steps=checkpoint_steps,
        max_grad_norm=args.max_grad_norm,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
