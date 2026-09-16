"""Language/code filters for host supervision.  No torch; safe to unit-test."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Mapping, Sequence, Tuple

CODE_PATTERN = re.compile(
    r"(?i)(?:```(?:python|java|javascript|typescript|cpp|c\+\+|rust|go|php|ruby|sql|bash|html)"
    r"|\bdef\s+\w+\s*\("
    r"|\bclass\s+\w+\s*[:(]"
    r"|\bimport\s+\w+"
    r"|\bfrom\s+\w+\s+import\b"
    r"|\bfunction\s+\w+\s*\("
    r"|console\.log\s*\("
    r"|public\s+static\s+"
    r"|#include\s*<"
    r"|\bfn\s+\w+\s*\("
    r"|#!/usr/bin"
    r"|package\s+main\b"
    r"|escriba una funci[oó]n"
    r"|escribe una funci[oó]n"
    r"|write a (?:[a-z+#]+ )?function"
    r"|implement(?:a|e)? una funci[oó]n"
    r"|\blet mut\b"
    r"|\bfunc\s+\w+\s*\("
    r"|HashMap|"
    r"std::"
    r"|=>\s*\{"
    r"|\bvar\s+\w+\s*=)"
)

PROGRAMMING_TASK_PATTERN = re.compile(
    r"(?i)(?:"
    r"\b(?:javascript|typescript|python|java|rust|golang|c\+\+|html|css)\b.{0,80}"
    r"(?:funci[oó]n|function|c[oó]digo|codigo|programa|script|class)"
    r"|(?:funci[oó]n|function|c[oó]digo|codigo|programa|script).{0,80}"
    r"\b(?:javascript|typescript|python|java|rust|golang|c\+\+|html|css)\b"
    r"|write (?:a |an |me )?(?:python |java |javascript |js )?(?:function|program|script|class)"
    r"|implementa (?:un |una )?(?:funci[oó]n|programa|algoritmo)"
    r"|crea (?:un |una )?(?:funci[oó]n|programa|script)"
    r"|public class\s+\w+"
    r"|SELECT\s+\S.+\s+FROM\s+"
    r")"
)

EXPECTED_FT_LABEL = {
    "es": "__label__es",
    "zh": "__label__zh",
    "sw": "__label__sw",
}


class TextFilterError(RuntimeError):
    """Raised when filtered host data cannot meet the frozen counts."""


def _plain_text(value: Any) -> str:
    if value is None or isinstance(value, (Mapping, list, tuple)):
        return ""
    return str(value).strip()


def instruction_pair_from_row(row: Mapping[str, Any]) -> tuple[str, str]:
    """Extract a prompt/completion pair from heterogeneous instruction schemas."""

    nested = row.get("inputs")
    if isinstance(nested, Mapping):
        nested_pair = instruction_pair_from_row(
            {
                "instruction": nested.get("1-instruction") or nested.get("instruction"),
                "input": nested.get("2-input") or nested.get("input"),
                "output": nested.get("3-output") or nested.get("output"),
            }
        )
        if nested_pair[0] and nested_pair[1]:
            return nested_pair
    instruction = _plain_text(row.get("instruction"))
    extra = _plain_text(row.get("input"))
    output = _plain_text(row.get("output") or row.get("response"))
    chosen = row.get("chosen")
    if not output and isinstance(chosen, Mapping):
        output = _plain_text(
            chosen.get("text") or chosen.get("content") or chosen.get("completion")
        )
    elif not output:
        output = _plain_text(chosen)
    if instruction and output:
        prompt = f"{instruction}\n{extra}".strip() if extra else instruction
        return prompt, output
    for prompt_key, completion_key in (
        ("prompt", "chosen"),
        ("prompt", "completion"),
        ("question", "answer"),
    ):
        prompt = _plain_text(row.get(prompt_key))
        completion_value = row.get(completion_key)
        if isinstance(completion_value, Mapping):
            completion = _plain_text(
                completion_value.get("text") or completion_value.get("content")
            )
        else:
            completion = _plain_text(completion_value)
        if prompt and completion:
            return prompt, completion
    return "", ""


def looks_like_code(text: str) -> bool:
    blob = str(text or "")
    if not blob.strip():
        return True
    if CODE_PATTERN.search(blob) or PROGRAMMING_TASK_PATTERN.search(blob):
        return True
    if blob.count("{") + blob.count("}") >= 4 and blob.count(";") >= 3:
        punct = sum(character in "{}();=" for character in blob)
        if punct / max(len(blob), 1) >= 0.06:
            return True
    return False


def _lid_predict(lid_model: Any, text: str) -> tuple[str, float]:
    """Avoid fastText.predict + NumPy 2 copy=False crashes on Windows."""

    blob = text or " "
    native = getattr(lid_model, "f", None)
    if native is not None and hasattr(native, "predict"):
        predictions = native.predict(blob, 1, 0.0, "strict")
        if not predictions:
            return "", 0.0
        probabilities, labels = zip(*predictions)
        return str(labels[0]), float(probabilities[0])
    labels, probabilities = lid_model.predict(blob, k=1)
    label = str(labels[0]) if labels else ""
    probability = (
        float(probabilities[0]) if probabilities is not None and len(probabilities) else 0.0
    )
    return label, probability


def is_target_language(text: str, language: str, lid_model: Any | None) -> bool:
    expected = EXPECTED_FT_LABEL[language]
    blob = str(text or "").replace("\n", " ").strip()
    if not blob:
        return False
    if lid_model is None:
        return not looks_like_code(blob)
    label, probability = _lid_predict(lid_model, blob[:4000])
    return label == expected and probability >= 0.35


def filter_host_rows(
    rows: Sequence[Mapping[str, Any]],
    language: str,
    *,
    lid_model: Any | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    n_code = 0
    n_wrong = 0
    by_source: Dict[str, Dict[str, int]] = {}
    for row in rows:
        source = str(row.get("source_dataset") or row.get("source_dataset") or "unknown")
        by_source.setdefault(source, {"n_in": 0, "n_kept": 0})
        by_source[source]["n_in"] += 1
        prompt = str(row.get("prompt") or "")
        completion = str(row.get("completion") or "")
        if looks_like_code(prompt) or looks_like_code(completion):
            n_code += 1
            continue
        if not is_target_language(prompt + "\n" + completion, language, lid_model):
            n_wrong += 1
            continue
        kept.append(dict(row))
        by_source[source]["n_kept"] += 1
    stats = {
        "n_in": len(rows),
        "n_kept": len(kept),
        "n_code": n_code,
        "n_wrong_language": n_wrong,
        "by_source": by_source,
        "filter": "natural_target_language_exclude_code",
    }
    return kept, stats


def balanced_round_robin(
    rows: Sequence[Mapping[str, Any]],
    n: int,
    *,
    salt: str,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("source_dataset") or row.get("source_dataset") or "unknown"), []).append(row)
    for source, group in grouped.items():
        grouped[source] = sorted(
            group,
            key=lambda item: hashlib.sha256(
                f"{salt}:{item.get('source_id')}".encode("utf-8")
            ).hexdigest(),
        )
    sources = sorted(grouped)
    chosen: List[Dict[str, Any]] = []
    index = {source: 0 for source in sources}
    while len(chosen) < n:
        progressed = False
        for source in sources:
            cursor = index[source]
            pool = grouped[source]
            if cursor < len(pool):
                chosen.append(dict(pool[cursor]))
                index[source] = cursor + 1
                progressed = True
                if len(chosen) >= n:
                    break
        if not progressed:
            break
    if len(chosen) < n:
        raise TextFilterError(
            f"need {n} source-balanced host rows after filtering; have {len(chosen)}"
        )
    return chosen


looks_like_code = looks_like_code
looks_like_code = looks_like_code
filter_host_rows = filter_host_rows
filter_host_rows = filter_host_rows
balanced_round_robin = balanced_round_robin
balanced_round_robin = balanced_round_robin
TextFilterError = TextFilterError
instruction_pair_from_row = instruction_pair_from_row
