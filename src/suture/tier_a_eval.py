"""Fail-closed completion log-probability and multiple-choice Belebele scoring."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch


class IncompleteDenominatorError(RuntimeError):
    """Raised when a declared evaluation row cannot contribute a finite score."""


def align_prompt_completion(
    tokenizer: Any,
    prompt: str,
    completion: str,
    *,
    max_length: int,
) -> Tuple[List[int], List[int]]:
    if not str(prompt or "").strip() or not str(completion or "").strip():
        raise IncompleteDenominatorError("empty prompt or completion")
    prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False, truncation=False)["input_ids"]
    if not completion_ids:
        raise IncompleteDenominatorError("completion produced zero tokens")
    budget = int(max_length)
    if budget < 2:
        raise IncompleteDenominatorError("max_length too small to keep a completion token")
    max_completion = min(len(completion_ids), budget - 1)
    completion_ids = list(completion_ids[:max_completion])
    max_prompt = budget - len(completion_ids)
    if max_prompt < 1:
        raise IncompleteDenominatorError("no room left for a prompt token")
    prompt_ids = list(prompt_ids[-max_prompt:])
    return prompt_ids, completion_ids


def score_completion(
    model: Any,
    tokenizer: Any,
    prompt: str,
    completion: str,
    *,
    device: torch.device,
    max_length: int,
) -> Dict[str, float]:
    prompt_ids, completion_ids = align_prompt_completion(
        tokenizer, prompt, completion, max_length=max_length
    )
    input_ids = torch.tensor([prompt_ids + completion_ids], device=device)
    labels = input_ids.clone()
    labels[:, : len(prompt_ids)] = -100
    with torch.inference_mode():
        loss = model(input_ids=input_ids, labels=labels).loss
    if loss is None or not torch.isfinite(loss):
        raise IncompleteDenominatorError("non-finite completion log-probability")
    n_tokens = len(completion_ids)
    mean_nats = -float(loss.detach().cpu())
    return {
        "n_tokens": float(n_tokens),
        "mean_nats_per_token": mean_nats,
        "sum_nats": mean_nats * n_tokens,
    }


def mean_completion_logprob(
    model: Any,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    max_length: int,
    declared_n: int | None = None,
) -> Dict[str, Any]:
    declared = int(declared_n if declared_n is not None else len(records))
    if len(records) != declared:
        raise IncompleteDenominatorError(
            f"declared n={declared} but received {len(records)} records"
        )
    rows: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        prompt = str(record.get("prompt") or record.get("question") or "")
        completion = str(record.get("completion") or "")
        scored = score_completion(
            model,
            tokenizer,
            prompt,
            completion,
            device=device,
            max_length=max_length,
        )
        rows.append({"index": index, "source_id": record.get("source_id"), **scored})
    if len(rows) != declared:
        raise IncompleteDenominatorError(
            f"scored {len(rows)} of declared {declared} rows"
        )
    mean = float(sum(row["mean_nats_per_token"] for row in rows) / declared)
    return {
        "n": declared,
        "mean_nats_per_token": mean,
        "complete_denominator": True,
        "scorer": "teacher_forced_completion_left_truncated_prompt",
        "items": rows,
    }


def multiple_choice_logprob(
    model: Any,
    tokenizer: Any,
    prompt: str,
    choices: Sequence[str],
    *,
    device: torch.device,
    max_length: int,
) -> Dict[str, Any]:
    if len(choices) < 2 or any(not str(choice).strip() for choice in choices):
        raise IncompleteDenominatorError("Belebele choices are missing")
    scores = []
    for choice in choices:
        scored = score_completion(
            model,
            tokenizer,
            prompt,
            " " + str(choice).strip(),
            device=device,
            max_length=max_length,
        )
        scores.append(scored["mean_nats_per_token"])
    predicted = int(max(range(len(scores)), key=lambda index: scores[index])) + 1
    return {"scores": scores, "predicted": predicted}


def belebele_mc_eval(
    model: Any,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    max_length: int,
    declared_n: int | None = None,
) -> Dict[str, Any]:
    declared = int(declared_n if declared_n is not None else len(records))
    if len(records) != declared:
        raise IncompleteDenominatorError(
            f"Belebele declared n={declared} but received {len(records)}"
        )
    items = []
    correct = 0
    for record in records:
        passage = str(record.get("flores_passage") or "")
        question = str(record.get("question") or "")
        choices = list(record.get("choices") or [])
        gold = str(record.get("answer_number") or "").strip()
        if not question or not gold:
            raise IncompleteDenominatorError("Belebele row missing question or answer")
        prompt = f"{passage}\n\n{question}\nAnswer:" if passage else f"{question}\nAnswer:"
        scored = multiple_choice_logprob(
            model,
            tokenizer,
            prompt,
            choices,
            device=device,
            max_length=max_length,
        )
        match = str(scored["predicted"]) == gold
        correct += int(match)
        items.append(
            {
                "source_id": record.get("source_id"),
                "predicted": scored["predicted"],
                "gold": gold,
                "exact_match": match,
                "choice_logprobs": scored["scores"],
            }
        )
    return {
        "n": declared,
        "accuracy": float(correct / declared) if declared else 0.0,
        "scorer": "multiple_choice_logprob",
        "complete_denominator": True,
        "items": items,
    }


def choose_max_gain_among_passing(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise IncompleteDenominatorError("no checkpoints to select")
    passing = [dict(row) for row in rows if row.get("passed")]
    pool = passing or [dict(row) for row in rows]
    selected = max(pool, key=lambda row: (float(row["gain"]), float(row.get("value") or 0.0)))
    selected["selection_rule"] = "max_gain_among_passing"
    if not passing:
        selected["passed"] = False
        selected["note"] = "no_checkpoint_met_thresholds"
    return selected


select_best_passing_by_gain = choose_max_gain_among_passing
IncompleteDenominatorError = IncompleteDenominatorError
mean_completion_logprob = mean_completion_logprob
belebele_mc_eval = belebele_mc_eval

