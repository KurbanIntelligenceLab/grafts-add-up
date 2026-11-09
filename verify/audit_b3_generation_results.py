"""Independently audit retained B3 free-generation records and summaries.

This auditor does not load a language model or contact a remote service. It
requires the six completed candidate runs, their input receipt, and the local
fastText language-ID model. A report is written only after all six JSONL files
contain the expected 250 records; any audit failure produces a FAIL report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "b3" / "generation_followup_v1"
DEFAULT_MANIFEST_ROOT = ROOT / "results" / "b3" / "lighton_qwen3_8b"
DEFAULT_CONFIG = ROOT / "configs" / "b3_generation_followup_v1.json"
DEFAULT_SOURCE_CONTRACT = ROOT / "configs" / "b3_lighton_qwen3_8b_v1.json"
DEFAULT_RUNNER = ROOT / "src" / "suture" / "b3_generation_followup.py"
DEFAULT_LID = ROOT / "models" / "lid.176.ftz"
FROZEN_CONFIG_SHA256 = "785f600ad422910847c35dd1b9d82d5a7a2315fe798dc56a9b745c7aad9264c6"
FROZEN_RUNNER_SHA256 = "1ab53cf1ddf42012e3b41c4a380690fb4fa587762df0274015972d4c86c00d69"
LANGUAGES = ("fr", "zh")
CANDIDATES = ("host_baseline", "selected", "published_reference")
STOP_MARKERS = ("<|im_end|>", "<|endoftext|>", "</s>")
TEACHER_FORCED_RE = re.compile(
    r"\A<\|im_start\|>user\n(?P<question>.*?)\nAnswer:<\|im_end\|>\n"
    r"<\|im_start\|>assistant\n<think>\n\n</think>\n\n\Z",
    re.DOTALL,
)
LABELS = {"fr": "__label__fr", "zh": "__label__zh"}


class AuditFailure(RuntimeError):
    """A failed evidence or consistency check."""


def require(condition: Any, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditFailure(f"cannot read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_file_hashes(file_hashes: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(file_hashes):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(file_hashes[name]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def canonical_record_hash(record: Mapping[str, Any]) -> str:
    clean = {str(key): value for key, value in record.items() if key != "_hash"}
    canonical = json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def read_source_manifest(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    try:
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditFailure(f"cannot read source manifest {path}: {exc}") from exc
    require(lines and isinstance(lines[0], dict) and "_manifest" in lines[0], f"invalid manifest header: {path}")
    header = lines[0]
    require(header.get("n_records") == len(lines) - 1, f"manifest record count mismatch: {path}")
    records: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(lines[1:]):
        require(isinstance(value, dict), f"manifest row {index + 1} is not an object")
        record = dict(value)
        record_id = str(record.get("id", ""))
        require(record_id and record_id not in records, f"missing or duplicate source ID {record_id!r}")
        calculated = canonical_record_hash(record)
        require(record.get("_hash") == calculated, f"source row hash mismatch for {record_id}")
        records[record_id] = record
    return dict(header), records


def extract_question(record: Mapping[str, Any]) -> str:
    match = TEACHER_FORCED_RE.fullmatch(str(record.get("input_text", "")))
    require(match is not None, f"source record {record.get('id')} is not the frozen MGSM form")
    question = str(match.group("question")).strip()
    require(bool(question), f"source record {record.get('id')} has an empty question")
    return question


def request_seed(global_seed: int, record_id: str) -> int:
    digest = hashlib.sha256(f"{global_seed}:{record_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def parse_answer(text: str, language: str) -> dict[str, Any]:
    answer_text = text.split("</think>", 1)[1] if "</think>" in text else text
    clean = answer_text
    for marker in STOP_MARKERS:
        clean = clean.replace(marker, "")
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    final_line = lines[-1] if lines else ""
    if not final_line:
        return {"status": "missing_final_line", "prediction": None, "final_line": ""}
    normalized = re.sub(r"\*\*|__|`", "", final_line).strip()
    marker_pattern = re.compile(
        r"^(?P<label>Réponse(?:\s+finale)?|Answer|答案|最终答案)\s*[:：]\s*(?P<value>.*)$",
        flags=re.IGNORECASE,
    )
    marker = marker_pattern.fullmatch(normalized)
    if marker:
        label = str(marker.group("label"))
        value = marker.group("value").strip()
        format_status = "localized_label" if (
            (language == "fr" and label.casefold().startswith("réponse"))
            or (language == "zh" and label in {"答案", "最终答案"})
        ) else "nonlocalized_label"
    else:
        value = normalized
        label = None
        format_status = "bare_final_integer"
    value = value.strip().strip("$ ")
    boxed = re.fullmatch(r"\\boxed\s*\{\s*(-?\d+)\s*\}\s*[.!。]?", value)
    integer = boxed.group(1) if boxed else None
    if integer is None:
        direct = re.fullmatch(r"(-?\d+)\s*[.!。]?", value)
        integer = direct.group(1) if direct else None
    if integer is None:
        return {
            "status": "invalid_integer_answer" if marker else "missing_or_ambiguous_answer_marker",
            "prediction": None,
            "final_line": final_line,
            "label": label,
            "format_status": format_status,
            "strict_localized_format": False,
        }
    try:
        normalized_integer = str(int(integer))
    except ValueError:
        return {"status": "invalid_integer_answer", "prediction": None, "final_line": final_line, "label": label, "format_status": format_status}
    return {
        "status": "parsed",
        "prediction": normalized_integer,
        "raw_prediction": integer,
        "final_line": final_line,
        "label": label,
        "format_status": format_status,
        "strict_localized_format": format_status == "localized_label",
    }


def derive_reasoning(text: str) -> tuple[str, str]:
    if "</think>" in text:
        segment = text.split("</think>", 1)[0]
        if "<think>" in segment:
            segment = segment.split("<think>", 1)[1]
        return segment.strip(), "complete"
    if not text.strip():
        return "", "missing_reasoning_text"
    segment = text.split("<think>", 1)[1] if "<think>" in text else text
    return segment.strip(), "incomplete_no_think_close"


def answer_segment(text: str, finish_reason: str | None) -> tuple[str, str]:
    if "</think>" in text:
        return text.split("</think>", 1)[1], "post_think_segment"
    if finish_reason == "stop" and text.strip():
        return text, "completed_no_think_fallback"
    return "", "missing_think_close_before_completion"


def fasttext_top1(model: Any, text: str) -> tuple[str, float]:
    native = getattr(model, "f", None)
    require(native is not None and callable(getattr(native, "predict", None)), "fastText native f.predict is unavailable")
    predictions = native.predict((text or " ").rstrip() + "\n", 1, 0.0, "strict")
    if not predictions:
        return "", 0.0
    probability, label = predictions[0]
    return str(label), float(probability)


def language_id(text: str, language: str, model: Any, minimum: int) -> dict[str, Any]:
    cleaned = " ".join(text.split())
    letters = sum(character.isalpha() for character in cleaned)
    if letters < minimum:
        return {
            "status": "insufficient_text",
            "letter_characters": letters,
            "minimum_letter_characters": minimum,
            "predicted_label": None,
            "probability": None,
            "target_language_top1": None,
        }
    label, probability = fasttext_top1(model, cleaned)
    return {
        "status": "evaluated",
        "letter_characters": letters,
        "minimum_letter_characters": minimum,
        "predicted_label": label,
        "probability": probability,
        "target_language_top1": label == LABELS[language],
    }


def english_chunk_diagnostic(text: str, model: Any, spec: Mapping[str, Any]) -> dict[str, Any]:
    chunk_size = int(spec["chunk_characters"])
    min_letters = int(spec["minimum_letter_characters"])
    threshold = float(spec["minimum_probability"])
    stride = int(spec.get("stride_characters", chunk_size))
    require(stride == chunk_size, "unsupported overlapping English chunk protocol")
    chunks = []
    for start in range(0, len(text), stride):
        chunk = text[start : start + chunk_size]
        letters = sum(character.isalpha() for character in chunk)
        if letters < min_letters:
            continue
        label, probability = fasttext_top1(model, " ".join(chunk.split()))
        chunks.append({
            "start_character": start,
            "letter_characters": letters,
            "predicted_label": label,
            "probability": probability,
            "english_chunk": label == "__label__en" and probability >= threshold,
        })
    english_n = sum(bool(item["english_chunk"]) for item in chunks)
    covered = sum(min(chunk_size, max(0, len(text) - int(item["start_character"]))) for item in chunks)
    return {
        "status": "evaluated" if chunks else "no_eligible_chunks",
        "chunk_characters": chunk_size,
        "stride_characters": stride,
        "minimum_letter_characters": min_letters,
        "minimum_probability": threshold,
        "text_characters": len(text),
        "covered_characters": covered,
        "eligible_character_coverage": covered / len(text) if text else 0.0,
        "eligible_chunks": len(chunks),
        "english_chunks": english_n,
        "any_english_chunk": bool(english_n) if chunks else None,
        "chunks": chunks,
    }


def expected_lid_fields(row: Mapping[str, Any], language: str, config: Mapping[str, Any], model: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    text = str(row.get("decoded_generated_text", ""))
    finish_reason = row.get("finish_reason")
    truncated = finish_reason == "length"
    minimum = int(config["language_id"]["minimum_letter_characters"])
    reasoning, reasoning_status = derive_reasoning(text)
    require(row.get("reasoning_segment") == reasoning, f"reasoning text extraction mismatch: {row.get('record_id')}")
    require(row.get("reasoning_segment_status") == reasoning_status, f"reasoning status mismatch: {row.get('record_id')}")
    if reasoning_status in {"complete", "incomplete_no_think_close"}:
        base_lid = language_id(reasoning, language, model, minimum)
        lid = dict(base_lid)
        diagnostic_target = base_lid.get("target_language_top1")
        if truncated:
            lid["status"] = "truncated_output"
            lid["target_language_top1"] = None
        elif reasoning_status != "complete":
            lid["status"] = "incomplete_reasoning"
            lid["target_language_top1"] = None
        lid["diagnostic_target_language_top1"] = diagnostic_target
        chunk_detail = english_chunk_diagnostic(reasoning, model, config["language_id"]["english_chunk_diagnostic"])
        if truncated or reasoning_status != "complete":
            chunks = {
                "status": "truncated_reasoning" if truncated else "incomplete_reasoning",
                "eligible_chunks": 0,
                "english_chunks": 0,
                "any_english_chunk": None,
                "partial_diagnostic": chunk_detail,
            }
        else:
            chunks = chunk_detail
    else:
        lid = {
            "status": reasoning_status,
            "letter_characters": 0,
            "minimum_letter_characters": minimum,
            "predicted_label": None,
            "probability": None,
            "target_language_top1": None,
            "diagnostic_target_language_top1": None,
        }
        chunks = {"status": reasoning_status, "eligible_chunks": 0, "english_chunks": 0, "any_english_chunk": None, "chunks": []}
    answer_text, answer_status = answer_segment(text, str(finish_reason) if finish_reason is not None else None)
    require(row.get("answer_parse", {}).get("segment_status") == answer_status, f"answer segment status mismatch: {row.get('record_id')}")
    if answer_status == "post_think_segment":
        answer_content = re.sub(
            r"(?:Réponse(?:\s+finale)?|Answer|答案|最终答案)\s*[:：]",
            " ",
            answer_text,
            flags=re.IGNORECASE,
        )
        answer_lid = language_id(answer_content, language, model, minimum)
    else:
        answer_lid = {
            "status": "not_evaluated_no_think_close",
            "letter_characters": 0,
            "minimum_letter_characters": minimum,
            "predicted_label": None,
            "probability": None,
            "target_language_top1": None,
        }
    return lid, chunks, answer_lid


def assert_nested_equal(actual: Any, expected: Any, path: str, *, tolerance: float = 1e-6) -> None:
    if isinstance(expected, Mapping):
        require(isinstance(actual, Mapping), f"{path} is not an object")
        for key, value in expected.items():
            require(key in actual, f"{path}.{key} is missing")
            assert_nested_equal(actual[key], value, f"{path}.{key}", tolerance=tolerance)
        return
    if isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"{path} list length mismatch")
        for index, (left, right) in enumerate(zip(actual, expected)):
            assert_nested_equal(left, right, f"{path}[{index}]", tolerance=tolerance)
        return
    if isinstance(expected, float):
        require(isinstance(actual, (float, int)) and math.isfinite(float(actual)), f"{path} is not a finite number")
        require(abs(float(actual) - expected) <= tolerance, f"{path} mismatch: {actual} != {expected}")
        return
    require(actual == expected, f"{path} mismatch: {actual!r} != {expected!r}")


def metric_summary(rows: Mapping[str, Mapping[str, Any]], lid_recomputed: Mapping[str, dict[str, Any]], chunks_recomputed: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    values = list(rows.values())
    n = len(values)
    correct = sum(bool(row["_audit_exact_match"]) for row in values)
    parse_status = Counter(str(row.get("answer_parse", {}).get("status", "missing")) for row in values)
    format_status = Counter(str(row.get("answer_parse", {}).get("format_status", "unknown")) for row in values)
    strict_format = sum(bool(row.get("answer_parse", {}).get("strict_localized_format")) for row in values)
    truncated = sum(bool(row.get("truncated")) for row in values)
    lid_values = [lid_recomputed[key] for key in rows]
    evaluated = [item for item in lid_values if item.get("status") == "evaluated"]
    lid_target = sum(item.get("target_language_top1") is True for item in evaluated)
    chunks = [chunks_recomputed[key] for key in rows]
    chunk_evaluable = [item for item in chunks if item.get("status") == "evaluated"]
    english_any = sum(item.get("any_english_chunk") is True for item in chunk_evaluable)
    answer_lid_values = [row["_audit_answer_lid"] for row in values]
    return {
        "n_records": n,
        "accuracy_all_records": correct / n,
        "n_correct": correct,
        "n_incorrect_or_unparseable": n - correct,
        "parse_status_counts": dict(sorted(parse_status.items())),
        "answer_format_status_counts": dict(sorted(format_status.items())),
        "strict_localized_format_count": strict_format,
        "strict_localized_format_rate_all_records": strict_format / n,
        "truncated_records": truncated,
        "reasoning_language_id": {
            "target_top1_count": lid_target,
            "unknown_count": n - len(evaluated),
            "denominator_all_records": n,
            "target_top1_rate_all_records": lid_target / n,
            "evaluable_count": len(evaluated),
            "target_top1_rate_evaluable": lid_target / len(evaluated) if evaluated else None,
        },
        "reasoning_english_chunk_diagnostic": {
            "any_english_chunk_count": english_any,
            "unknown_count": n - len(chunk_evaluable),
            "denominator_all_records": n,
            "any_english_chunk_rate_all_records": english_any / n,
            "evaluable_count": len(chunk_evaluable),
        },
        "answer_language_id": {
            "evaluated_count": sum(item.get("status") == "evaluated" for item in answer_lid_values),
            "unknown_count": sum(item.get("status") != "evaluated" for item in answer_lid_values),
            "target_top1_count": sum(item.get("target_language_top1") is True for item in answer_lid_values),
            "denominator_all_records": n,
        },
    }


def quantile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def bootstrap_python(values: Sequence[float], seed: int, n_resamples: int) -> dict[str, Any]:
    rng = random.Random(seed)
    n = len(values)
    observed = sum(values) / n
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples))
    return {
        "mean_difference": observed,
        "ci95_percentile": [quantile(means, 0.025), quantile(means, 0.975)],
        "n_paired_records": n,
        "bootstrap_resamples": n_resamples,
        "bootstrap_seed": seed,
    }


def bootstrap_numpy_independent(values: Sequence[float], seed: int, n_resamples: int) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        raise AuditFailure("NumPy is required for an independent bootstrap audit") from exc
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    n = len(array)
    means = np.empty(n_resamples, dtype=np.float64)
    block = max(1, min(n_resamples, 2000))
    offset = 0
    while offset < n_resamples:
        count = min(block, n_resamples - offset)
        indices = rng.integers(0, n, size=(count, n), endpoint=False)
        means[offset : offset + count] = array[indices].mean(axis=1)
        offset += count
    ci = np.quantile(means, [0.025, 0.975], method="linear")
    return {"ci95_percentile": [float(ci[0]), float(ci[1])], "bootstrap_seed": seed, "bootstrap_resamples": n_resamples}


def equal_or_close(actual: Any, expected: Any, path: str, tolerance: float = 1e-9) -> None:
    if isinstance(expected, float):
        require(isinstance(actual, (int, float)) and abs(float(actual) - expected) <= tolerance, f"{path} mismatch: {actual!r} != {expected!r}")
    else:
        require(actual == expected, f"{path} mismatch: {actual!r} != {expected!r}")


def candidate_spec(config: Mapping[str, Any], language: str, candidate: str) -> dict[str, Any]:
    language_data = config["languages"][language]
    if candidate == "host_baseline":
        model = dict(language_data["host"])
        graft = []
    elif candidate == "selected":
        model = {"id": f"materialized:{language_data['host']['id']}+{config['donor']['id']}", "revision": None}
        graft = list(language_data["selected_window"])
    else:
        model = dict(language_data["published_reference"])
        start, end = [int(item) for item in language_data["published_window_inclusive"]]
        graft = list(range(start, end + 1))
    return {"name": candidate, "model": model, "graft_layers": graft}


def load_rows(path: Path, run: Mapping[str, Any], language: str, candidate: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                require(isinstance(item, dict), f"{path}:{line_number} is not an object")
                record_id = str(item.get("record_id", ""))
                require(record_id and record_id not in rows, f"duplicate/missing output ID in {path}:{line_number}")
                require(item.get("run_identity") == run.get("run_identity"), f"run identity mismatch in {path}:{line_number}")
                require(item.get("language") == language and item.get("candidate") == candidate, f"output key mismatch in {path}:{line_number}")
                rows[record_id] = item
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditFailure(f"cannot read generations {path}: {exc}") from exc
    return rows


def verify_run_inputs(
    run: Mapping[str, Any], language: str, candidate: str, config: Mapping[str, Any],
    source_contract_hash: str, config_hash: str, runner_hash: str, lid_hash: str,
    manifest_path: Path, manifest_header: Mapping[str, Any], manifest_hash: str,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    require(run.get("schema_version") == 1 and run.get("run_kind") == "b3_free_generation_followup", f"{language}/{candidate}: unexpected run schema")
    require(run.get("status") == "PASS_GENERATION", f"{language}/{candidate}: generation status is not PASS_GENERATION")
    require(run.get("language") == language and run.get("candidate") == candidate, f"{language}/{candidate}: run key mismatch")
    require(run.get("completed_records") == 250, f"{language}/{candidate}: completed_records is not 250")
    require(run.get("candidate_spec") == candidate_spec(config, language, candidate), f"{language}/{candidate}: candidate spec does not match frozen config")
    require(run.get("followup_config", {}).get("sha256") == config_hash, f"{language}/{candidate}: follow-up config hash mismatch")
    require(run.get("frozen_source_contract", {}).get("sha256") == source_contract_hash, f"{language}/{candidate}: source contract hash mismatch")
    require(run.get("runner_sha256") == runner_hash, f"{language}/{candidate}: runner code hash mismatch")
    require(run.get("fasttext_lid", {}).get("sha256") == lid_hash, f"{language}/{candidate}: fastText model hash mismatch")
    require(run.get("data_manifest", {}).get("sha256") == manifest_hash, f"{language}/{candidate}: source manifest hash mismatch")
    require(run.get("data_manifest", {}).get("header") == manifest_header, f"{language}/{candidate}: embedded source manifest header mismatch")
    require(run.get("data_manifest", {}).get("n_records") == 250, f"{language}/{candidate}: run source record count mismatch")
    require(run.get("followup_config", {}).get("path"), f"{language}/{candidate}: missing config path")
    require(run.get("frozen_source_contract", {}).get("path"), f"{language}/{candidate}: missing source contract path")
    require(run.get("prompt_protocol", {}).get("template") == config["chat_template"], f"{language}/{candidate}: prompt template mismatch")
    instruction = config["languages"][language]["prompt_instruction"]
    require(run.get("prompt_protocol", {}).get("instruction") == instruction, f"{language}/{candidate}: prompt instruction mismatch")
    require(run.get("prompt_protocol", {}).get("instruction_sha256") == hashlib.sha256(instruction.encode("utf-8")).hexdigest(), f"{language}/{candidate}: prompt instruction hash mismatch")
    require(run.get("generation") == config["generation"], f"{language}/{candidate}: decoding config mismatch")
    require(run.get("data_manifest", {}).get("path"), f"{language}/{candidate}: missing manifest path")

    model = run.get("model_snapshot")
    require(isinstance(model, Mapping), f"{language}/{candidate}: missing model snapshot inventory")
    file_hashes = model.get("file_hashes")
    require(isinstance(file_hashes, Mapping) and file_hashes, f"{language}/{candidate}: model snapshot has no file hashes")
    require(model.get("file_count") == len(file_hashes), f"{language}/{candidate}: model snapshot file count mismatch")
    require(model.get("snapshot_sha256") == aggregate_file_hashes(file_hashes), f"{language}/{candidate}: model snapshot aggregate hash mismatch")

    language_receipt = receipt.get("languages", {}).get(language, {})
    receipt_candidate = language_receipt.get("candidates", {}).get(candidate, {})
    require(bool(receipt_candidate), f"candidate input receipt is missing {language}/{candidate}")
    require(language_receipt.get("data_manifest_sha256") == manifest_hash, f"input receipt source manifest hash mismatch for {language}")
    provenance_hash = str(language_receipt.get("tensor_mapping_receipt_sha256", ""))
    require(re.fullmatch(r"[0-9a-f]{64}", provenance_hash) is not None, f"invalid tensor mapping receipt hash for {language}")
    expected = candidate_spec(config, language, candidate)["model"]
    require(receipt_candidate.get("model_id") == ("exact_selected_layer_swap" if candidate == "selected" else expected["id"]), f"{language}/{candidate}: receipt model ID mismatch")
    require(receipt_candidate.get("revision") == expected.get("revision"), f"{language}/{candidate}: receipt revision mismatch")
    if candidate != "selected":
        require(receipt_candidate.get("download_commit_and_blob_hashes_verified") is True, f"{language}/{candidate}: pinned download/blob hashes were not attested")
    core = receipt_candidate.get("core_file_sha256")
    require(isinstance(core, Mapping) and core, f"{language}/{candidate}: receipt contains no core file hashes")
    if candidate != "selected":
        lfs_files = receipt_candidate.get("lfs_weight_files_verified")
        safetensors_files = {str(name) for name in core if str(name).endswith(".safetensors")}
        require(isinstance(lfs_files, list) and safetensors_files.issubset(set(lfs_files)), f"{language}/{candidate}: receipt does not attest LFS hashes for every weight shard")
    for filename, digest in core.items():
        require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None, f"{language}/{candidate}: invalid core hash for {filename}")
        key = str(filename)
        if key not in file_hashes:
            matches = [name for name in file_hashes if name.endswith("/" + key)]
            require(len(matches) == 1, f"{language}/{candidate}: receipt core file {key} is absent or ambiguous in model snapshot")
            key = matches[0]
        require(file_hashes[key] == digest, f"{language}/{candidate}: model snapshot differs from input receipt for {filename}")
    if candidate == "selected":
        materialization = run.get("materialization_manifest")
        require(isinstance(materialization, Mapping), f"{language}: selected run lacks graft provenance")
        require(materialization.get("exact_mapping_verified") is True, f"{language}: selected graft does not attest exact mapping")
        require(materialization.get("graft_layers") == config["languages"][language]["selected_window"], f"{language}: selected graft layers mismatch")
        require(materialization.get("host_revision") == config["languages"][language]["host"]["revision"], f"{language}: selected host revision mismatch")
        require(materialization.get("donor_revision") == config["donor"]["revision"], f"{language}: selected donor revision mismatch")
    return {"model_snapshot_sha256": model["snapshot_sha256"], "receipt_core_files_checked": len(core), "tensor_mapping_receipt_sha256": provenance_hash}


def tensor_aggregate(hashes: Mapping[str, str]) -> str:
    payload = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def layer_of(parameter_name: str) -> int | None:
    parts = parameter_name.split(".")
    if len(parts) > 3 and parts[:2] == ["model", "layers"]:
        try:
            return int(parts[2])
        except ValueError:
            return None
    return None


def verify_tensor_mapping_receipt(
    results_root: Path,
    language: str,
    input_receipt: Mapping[str, Any],
    config: Mapping[str, Any],
    source_contract: Mapping[str, Any],
) -> dict[str, Any]:
    language_receipt = input_receipt.get("languages", {}).get(language, {})
    receipt_name = str(language_receipt.get("tensor_mapping_receipt_file", ""))
    require(receipt_name and Path(receipt_name).name == receipt_name, f"{language}: invalid tensor mapping receipt filename")
    path = results_root / receipt_name
    require(path.is_file(), f"{language}: missing tensor mapping receipt {path}")
    expected_file_hash = language_receipt.get("tensor_mapping_receipt_sha256")
    actual_file_hash = sha256_file(path)
    require(actual_file_hash == expected_file_hash, f"{language}: tensor mapping receipt file hash mismatch")
    provenance = read_json(path)
    require(provenance.get("language") == language, f"{language}: mapping receipt language mismatch")
    require(provenance.get("official_published_equals_reconstructed_window") is True, f"{language}: published swap parity is not attested")
    require(provenance.get("official_published_mismatched_tensor_names") == [], f"{language}: published swap has tensor mismatches")
    require(provenance.get("selected_saved_equals_exact_mapping") is True, f"{language}: selected checkpoint mapping is not attested")

    host_spec = next(row for row in source_contract["models"]["hosts"] if row["language"] == language)
    require(provenance.get("host") == host_spec, f"{language}: mapping receipt host spec differs from source contract")
    require(provenance.get("donor") == source_contract["models"]["donor"], f"{language}: mapping receipt donor spec differs from source contract")
    selected_window = [int(layer) for layer in config["languages"][language]["selected_window"]]
    published_bounds = [int(layer) for layer in config["languages"][language]["published_window_inclusive"]]
    published_window = list(range(published_bounds[0], published_bounds[1] + 1))
    require(provenance.get("selected_window_zero_based_inclusive") == [min(selected_window), max(selected_window)], f"{language}: selected window in mapping receipt is incorrect")
    require(provenance.get("published_window_zero_based_inclusive") == published_bounds, f"{language}: published window in mapping receipt is incorrect")

    tensors = provenance.get("tensor_hashes")
    require(isinstance(tensors, Mapping), f"{language}: mapping receipt lacks tensor hashes")
    role_names = ("host", "donor", "official_published", "selected")
    require(all(isinstance(tensors.get(role), Mapping) for role in role_names), f"{language}: mapping receipt is missing tensor roles")
    keysets = [set(tensors[role]) for role in role_names]
    require(all(keys == keysets[0] for keys in keysets[1:]), f"{language}: tensor role inventories do not match")
    require(len(keysets[0]) == 399, f"{language}: expected 399 tensor names, found {len(keysets[0])}")
    for role in role_names:
        require(all(re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in tensors[role].values()), f"{language}: malformed tensor hash in role {role}")

    expected_published = {}
    expected_selected = {}
    for name, host_hash in tensors["host"].items():
        layer = layer_of(str(name))
        expected_published[name] = tensors["donor"][name] if layer in published_window else host_hash
        expected_selected[name] = tensors["donor"][name] if layer in selected_window else host_hash
    require(dict(tensors["official_published"]) == expected_published, f"{language}: official published tensor hashes do not match its declared swap window")
    require(dict(tensors["selected"]) == expected_selected, f"{language}: selected tensor hashes do not match its declared swap window")

    aggregate_receipts = provenance.get("source_aggregate_tensor_sha256", {})
    for role in role_names:
        require(aggregate_receipts.get(role) == tensor_aggregate(tensors[role]), f"{language}: tensor aggregate hash mismatch for {role}")
    return {
        "path": str(path),
        "sha256": actual_file_hash,
        "tensor_names": len(keysets[0]),
        "verified_roles": list(role_names),
        "selected_window_zero_based_inclusive": [min(selected_window), max(selected_window)],
        "published_window_zero_based_inclusive": published_bounds,
        "published_swap_exact_mapping": True,
        "selected_swap_exact_mapping": True,
    }


def verify_summary(
    summary: Mapping[str, Any], language: str, candidates: Mapping[str, dict[str, dict[str, Any]]],
    lid_by_candidate: Mapping[str, dict[str, dict[str, Any]]], chunks_by_candidate: Mapping[str, dict[str, dict[str, Any]]],
    answer_lid_by_candidate: Mapping[str, dict[str, dict[str, Any]]], runs: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    require(summary.get("status") == "PASS_AGGREGATE" and summary.get("language") == language, f"{language}: summary is not a completed aggregate")
    expected_summaries = {}
    for candidate in CANDIDATES:
        rows = candidates[candidate]
        for record_id, row in rows.items():
            row["_audit_answer_lid"] = answer_lid_by_candidate[candidate][record_id]
        calculated = metric_summary(rows, lid_by_candidate[candidate], chunks_by_candidate[candidate])
        expected_summaries[candidate] = calculated
        actual = summary.get("candidate_summaries", {}).get(candidate)
        require(isinstance(actual, Mapping), f"{language}: summary lacks {candidate}")
        assert_nested_equal(actual, calculated, f"summary.{candidate}")
    expected_identities = {name: runs[name].get("run_identity") for name in CANDIDATES}
    require(summary.get("candidate_run_identities") == expected_identities, f"{language}: summary run identities do not match runs")

    comparisons_to_make = (
        ("selected_minus_host", "selected", "host_baseline"),
        ("published_minus_host", "published_reference", "host_baseline"),
        ("selected_minus_published", "selected", "published_reference"),
    )
    base_seed = int(config["metrics"]["bootstrap_seed"])
    n_resamples = int(config["metrics"]["bootstrap_resamples"])
    ordered_ids = sorted(candidates["host_baseline"])
    independent_results = {}
    tolerance = 0.02
    comparisons = summary.get("paired_comparisons", {})
    for name, left, right in comparisons_to_make:
        seed = base_seed + int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "big")
        metrics = {
            "accuracy_all_records": [
                float(bool(candidates[left][rid]["_audit_exact_match"])) - float(bool(candidates[right][rid]["_audit_exact_match"]))
                for rid in ordered_ids
            ],
            "reasoning_target_lid_top1_all_records": [
                float(lid_by_candidate[left][rid].get("target_language_top1") is True) - float(lid_by_candidate[right][rid].get("target_language_top1") is True)
                for rid in ordered_ids
            ],
            "reasoning_english_chunk_any_all_records": [
                float(chunks_by_candidate[left][rid].get("any_english_chunk") is True) - float(chunks_by_candidate[right][rid].get("any_english_chunk") is True)
                for rid in ordered_ids
            ],
        }
        seed_offsets = {"accuracy_all_records": 0, "reasoning_target_lid_top1_all_records": 1, "reasoning_english_chunk_any_all_records": 2}
        for metric, values in metrics.items():
            metric_seed = seed + seed_offsets[metric]
            exact = bootstrap_python(values, metric_seed, n_resamples)
            actual = comparisons.get(name, {}).get(metric)
            require(isinstance(actual, Mapping), f"{language}: missing paired comparison {name}.{metric}")
            assert_nested_equal(actual, exact, f"summary.paired_comparisons.{name}.{metric}", tolerance=1e-10)
            distinct_seed = int.from_bytes(hashlib.sha256(f"independent-audit:{language}:{name}:{metric}".encode()).digest()[:4], "big")
            require(distinct_seed != metric_seed, "independent bootstrap seed accidentally matches the runner seed")
            independent = bootstrap_numpy_independent(values, distinct_seed, n_resamples)
            require(abs(float(actual["ci95_percentile"][0]) - independent["ci95_percentile"][0]) <= tolerance, f"{language}: independent bootstrap lower bound for {name}.{metric} differs beyond {tolerance}")
            require(abs(float(actual["ci95_percentile"][1]) - independent["ci95_percentile"][1]) <= tolerance, f"{language}: independent bootstrap upper bound for {name}.{metric} differs beyond {tolerance}")
            independent_results[f"{name}.{metric}"] = {
                "runner_ci95": actual["ci95_percentile"],
                "independent_numpy_ci95": independent["ci95_percentile"],
                "independent_seed": distinct_seed,
                "absolute_endpoint_tolerance": tolerance,
            }
    return {"candidate_summaries": expected_summaries, "independent_bootstrap": independent_results}


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--receipts", type=Path, default=None, help="defaults to candidate_input_receipts.json in results root")
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-contract", type=Path, default=DEFAULT_SOURCE_CONTRACT)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--lid-model", type=Path, default=DEFAULT_LID)
    return parser


def retained_generation_counts(results_root: Path) -> dict[str, int] | None:
    counts = {}
    for language in LANGUAGES:
        for candidate in CANDIDATES:
            path = results_root / language / candidate / "generations.jsonl"
            if not path.is_file():
                return None
            seen = set()
            try:
                with path.open("r", encoding="utf-8") as stream:
                    for line_number, line in enumerate(stream, 1):
                        if not line.strip():
                            continue
                        item = json.loads(line)
                        if not isinstance(item, dict):
                            return None
                        record_id = str(item.get("record_id", ""))
                        if not record_id or record_id in seen:
                            return None
                        seen.add(record_id)
            except (OSError, json.JSONDecodeError):
                return None
            counts[f"{language}/{candidate}"] = len(seen)
            if len(seen) != 250:
                return None
    return counts


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    retained_counts = retained_generation_counts(args.results_root)
    require(retained_counts is not None, "all six 250-row generation files must be present before auditing; no report written")
    config = read_json(args.config)
    require(config.get("candidate_names") == list(CANDIDATES), "unexpected candidate list in frozen config")
    require(set(config.get("languages", {})) == set(LANGUAGES), "config must include French and Chinese")
    source_contract = read_json(args.source_contract)
    source_contract_hash = sha256_file(args.source_contract)
    require(config.get("source_contract") == str(Path(args.source_contract).relative_to(ROOT)).replace("\\", "/"), "follow-up config does not identify the supplied source contract")
    require(source_contract.get("contract_name") == "b3_lighton_qwen3_8b_v1", "source contract name mismatch")
    config_hash = sha256_file(args.config)
    runner_hash = sha256_file(args.runner)
    require(config_hash == FROZEN_CONFIG_SHA256, "follow-up config hash differs from the final production freeze")
    require(runner_hash == FROZEN_RUNNER_SHA256, "generation runner hash differs from the final production freeze")
    lid_hash = sha256_file(args.lid_model)
    receipt_path = args.receipts or (args.results_root / "candidate_input_receipts.json")
    receipt = read_json(receipt_path)
    require(receipt.get("source_contract_sha256") == source_contract_hash, "input receipt source contract hash mismatch")
    require(args.lid_model.is_file(), f"missing fastText model: {args.lid_model}")
    try:
        import fasttext
        lid_model = fasttext.load_model(str(args.lid_model))
    except Exception as exc:
        raise AuditFailure(f"cannot load local fastText model using native prediction: {exc}") from exc
    require(getattr(getattr(lid_model, "f", None), "predict", None) is not None, "fastText native f.predict is unavailable")

    manifests = {}
    source_records = {}
    manifest_hashes = {}
    for language in LANGUAGES:
        manifest_path = args.manifest_root / f"{language}_host__en_donor" / "data" / "MGSM_rev2_test.jsonl"
        header, records = read_source_manifest(manifest_path)
        expected_name = f"b3_{language}_MGSM_rev2_test"
        require(header.get("_manifest") == expected_name, f"unexpected {language} source manifest name")
        metadata = header.get("metadata", {})
        dataset = config["dataset"]
        source = metadata.get("source", {})
        require(metadata.get("language") == language, f"{language}: source language mismatch")
        require(source.get("dataset") == dataset["id"] and source.get("revision") == dataset["revision"], f"{language}: source dataset/revision mismatch")
        require(source.get("split") == dataset["split"] and len(records) == int(dataset["n_items_per_language"]), f"{language}: source split/count mismatch")
        require(all(record.get("language") == language and record.get("dataset_revision") == dataset["revision"] for record in records.values()), f"{language}: source records have inconsistent language/revision")
        manifests[language] = (manifest_path, header)
        source_records[language] = records
        manifest_hashes[language] = sha256_file(manifest_path)

    # Check completeness before creating any report, so an absent or partial
    # generation upload is reported as missing evidence rather than an audit.
    runs: dict[str, dict[str, dict[str, Any]]] = {lang: {} for lang in LANGUAGES}
    rows_by_lang: dict[str, dict[str, dict[str, dict[str, Any]]]] = {lang: {} for lang in LANGUAGES}
    source_files: dict[str, dict[str, Any]] = {}
    incomplete = []
    for language in LANGUAGES:
        for candidate in CANDIDATES:
            directory = args.results_root / language / candidate
            run_path = directory / "run.json"
            rows_path = directory / "generations.jsonl"
            if not run_path.is_file() or not rows_path.is_file():
                incomplete.append(f"{language}/{candidate}: missing {run_path if not run_path.is_file() else rows_path}")
                continue
            run = read_json(run_path)
            digest_before_load = sha256_file(rows_path)
            rows = load_rows(rows_path, run, language, candidate)
            require(sha256_file(rows_path) == digest_before_load, f"{language}/{candidate}: generations file changed while loading")
            runs[language][candidate] = run
            rows_by_lang[language][candidate] = rows
            source_files[f"{language}/{candidate}/generations.jsonl"] = {
                "sha256": digest_before_load,
                "n_records": len(rows),
            }
            if len(rows) != 250:
                incomplete.append(f"{language}/{candidate}: {len(rows)} records, expected 250")
    require(not incomplete, "generation evidence is incomplete; no audit report written:\n  " + "\n  ".join(incomplete))

    report: dict[str, Any] = {
        "schema_version": 1,
        "audit_kind": "b3_generation_followup_independent_audit",
        "status": "FAIL_AUDIT",
        "results_root": str(args.results_root.resolve()),
        "input_hashes": {
            "receipts_sha256": sha256_file(receipt_path),
            "config_sha256": config_hash,
            "source_contract_sha256": source_contract_hash,
            "runner_sha256": runner_hash,
            "fasttext_lid_sha256": lid_hash,
        },
        "languages": {},
    }
    try:
        mapping_checks = {
            language: verify_tensor_mapping_receipt(args.results_root, language, receipt, config, source_contract)
            for language in LANGUAGES
        }
        report["tensor_mapping_receipts"] = mapping_checks
        for language in LANGUAGES:
            manifest_path, manifest_header = manifests[language]
            records = source_records[language]
            runs_for_language = runs[language]
            rows_for_language = rows_by_lang[language]
            input_checks = {}
            for candidate in CANDIDATES:
                input_checks[candidate] = verify_run_inputs(
                    runs_for_language[candidate], language, candidate, config,
                    source_contract_hash, config_hash, runner_hash, lid_hash,
                    manifest_path, manifest_header, manifest_hashes[language], receipt,
                )
            base = runs_for_language["host_baseline"]
            for candidate in CANDIDATES[1:]:
                run = runs_for_language[candidate]
                for field in ("data_manifest", "followup_config", "frozen_source_contract", "prompt_protocol", "fasttext_lid", "generation"):
                    require(run.get(field) == base.get(field), f"{language}: {candidate} differs from host in common field {field}")
                require(run.get("container_image_digest") == base.get("container_image_digest"), f"{language}: candidates used different container digests")
                require(run.get("runtime", {}).get("package_versions") == base.get("runtime", {}).get("package_versions"), f"{language}: candidates used different package versions")
                for key in ("torch_cuda_version", "cuda_available", "gpu_devices"):
                    require(run.get("runtime", {}).get(key) == base.get("runtime", {}).get(key), f"{language}: candidates differ in runtime field {key}")
                require(run.get("tokenizer", {}).get("sha256") == base.get("tokenizer", {}).get("sha256"), f"{language}: candidates used different tokenizer files")

            ids = set(records)
            require(all(set(rows_for_language[candidate]) == ids for candidate in CANDIDATES), f"{language}: candidate IDs do not exactly cover source manifest")
            recomputed_lid: dict[str, dict[str, dict[str, Any]]] = {}
            recomputed_chunks: dict[str, dict[str, dict[str, Any]]] = {}
            recomputed_answer_lid: dict[str, dict[str, dict[str, Any]]] = {}
            prompt_receipts = {}
            for candidate in CANDIDATES:
                rows = rows_for_language[candidate]
                recomputed_lid[candidate] = {}
                recomputed_chunks[candidate] = {}
                recomputed_answer_lid[candidate] = {}
                for record_id in sorted(ids):
                    source = records[record_id]
                    row = rows[record_id]
                    require(row.get("record_hash") == source.get("_hash"), f"{language}/{candidate}: source hash mismatch for {record_id}")
                    gold = str(source.get("answer_number"))
                    require(re.fullmatch(r"-?\d+", gold) is not None, f"{language}: invalid gold integer for {record_id}")
                    require(str(row.get("ground_truth_answer")) == gold, f"{language}/{candidate}: gold answer mismatch for {record_id}")
                    require(row.get("question") == extract_question(source), f"{language}/{candidate}: source question mismatch for {record_id}")
                    require(row.get("request_seed") == request_seed(int(config["generation"]["seed"]), record_id), f"{language}/{candidate}: wrong per-record seed for {record_id}")
                    prompt_ids = row.get("prompt_token_ids")
                    require(isinstance(prompt_ids, list) and all(isinstance(token, int) and token >= 0 for token in prompt_ids), f"{language}/{candidate}: invalid prompt token IDs for {record_id}")
                    prompt_hash = hashlib.sha256(",".join(str(token) for token in prompt_ids).encode("ascii")).hexdigest()
                    require(row.get("prompt_sha256") == prompt_hash, f"{language}/{candidate}: prompt hash mismatch for {record_id}")
                    if candidate == "host_baseline":
                        prompt_receipts[record_id] = (prompt_ids, prompt_hash)
                    else:
                        require((prompt_ids, prompt_hash) == prompt_receipts[record_id], f"{language}/{candidate}: prompt differs from host for {record_id}")
                    generated_ids = row.get("generated_token_ids")
                    require(isinstance(generated_ids, list) and len(generated_ids) == row.get("generated_token_count"), f"{language}/{candidate}: generated token count mismatch for {record_id}")
                    require(all(isinstance(token, int) and token >= 0 for token in generated_ids), f"{language}/{candidate}: invalid generated token ID for {record_id}")
                    decoded = str(row.get("decoded_generated_text", ""))
                    finish_reason = row.get("finish_reason")
                    truncated = finish_reason == "length"
                    require(row.get("truncated") is truncated, f"{language}/{candidate}: truncation flag disagrees with finish reason for {record_id}")

                    answer_text, answer_status = answer_segment(decoded, str(finish_reason) if finish_reason is not None else None)
                    expected_answer = parse_answer(answer_text, language)
                    expected_answer["segment_status"] = answer_status
                    assert_nested_equal(row.get("answer_parse"), expected_answer, f"{language}/{candidate}/{record_id}.answer_parse")
                    prediction = expected_answer.get("prediction")
                    try:
                        exact = prediction is not None and int(prediction) == int(gold) and not truncated
                    except (TypeError, ValueError):
                        exact = False
                    require(row.get("exact_match") is exact, f"{language}/{candidate}: exact_match flag mismatch for {record_id}")
                    require(row.get("counted_correct") is exact, f"{language}/{candidate}: counted_correct flag mismatch for {record_id}")
                    row["_audit_exact_match"] = exact

                    expected_lid, expected_chunks, expected_answer_lid = expected_lid_fields(row, language, config, lid_model)
                    assert_nested_equal(row.get("reasoning_language_id"), expected_lid, f"{language}/{candidate}/{record_id}.reasoning_language_id")
                    assert_nested_equal(row.get("reasoning_english_chunk_diagnostic"), expected_chunks, f"{language}/{candidate}/{record_id}.reasoning_english_chunk_diagnostic")
                    assert_nested_equal(row.get("answer_language_id"), expected_answer_lid, f"{language}/{candidate}/{record_id}.answer_language_id")
                    recomputed_lid[candidate][record_id] = expected_lid
                    recomputed_chunks[candidate][record_id] = expected_chunks
                    recomputed_answer_lid[candidate][record_id] = expected_answer_lid

            summary_path = args.results_root / language / "summary.json"
            summary = read_json(summary_path)
            summary_checks = verify_summary(
                summary, language, rows_for_language, recomputed_lid, recomputed_chunks,
                recomputed_answer_lid, runs_for_language, config,
            )
            report["languages"][language] = {
                "status": "PASS",
                "source_manifest": {"path": str(manifest_path), "sha256": manifest_hashes[language], "records": len(records)},
                "candidate_input_checks": input_checks,
                "candidate_counts": {candidate: len(rows_for_language[candidate]) for candidate in CANDIDATES},
                "recomputed_accuracy": {candidate: summary_checks["candidate_summaries"][candidate]["accuracy_all_records"] for candidate in CANDIDATES},
                "recomputed_reasoning_target_lid_rate": {candidate: summary_checks["candidate_summaries"][candidate]["reasoning_language_id"]["target_top1_rate_all_records"] for candidate in CANDIDATES},
                "recomputed_reasoning_english_chunk_rate": {candidate: summary_checks["candidate_summaries"][candidate]["reasoning_english_chunk_diagnostic"]["any_english_chunk_rate_all_records"] for candidate in CANDIDATES},
                "independent_bootstrap": summary_checks["independent_bootstrap"],
            }
        report["status"] = "PASS"
        report["conclusion"] = "All six candidate runs, source records, prompt/seed pairing, parsed answers, recomputed LID diagnostics, model receipts, and summary statistics passed."
    except AuditFailure as exc:
        report["status"] = "FAIL"
        report["failure"] = str(exc)
    if report["status"] == "PASS":
        try:
            require(len(source_files) == 6, "source file receipt does not contain six candidate outputs")
            require(all(item.get("n_records") == 250 for item in source_files.values()), "source file receipt contains a non-250 row count")
            require(sum(int(item["n_records"]) for item in source_files.values()) == 1500, "source file receipt total row count is not 1500")
            for relative_path, receipt_item in source_files.items():
                current_hash = sha256_file(args.results_root / Path(relative_path))
                require(current_hash == receipt_item["sha256"], f"{relative_path} changed during audit; refusing PASS")
            report["total_rows"] = 1500
            report["source_files"] = source_files
        except AuditFailure as exc:
            report["status"] = "FAIL"
            report["failure"] = str(exc)
    output = args.results_root / "audit_report.json"
    write_report(output, report)
    if report["status"] != "PASS":
        raise AuditFailure(str(report.get("failure")))
    report["report_path"] = str(output)
    return report


def write_fallback_failure_report(args: argparse.Namespace, failure: str, record_counts: Mapping[str, int]) -> None:
    output = args.results_root / "audit_report.json"
    if output.is_file():
        try:
            previous = read_json(output)
            if previous.get("audit_kind") == "b3_generation_followup_independent_audit" and previous.get("status") == "FAIL" and previous.get("failure") == failure:
                return
        except AuditFailure:
            pass
    hashes = {}
    for name, path in (
        ("config", args.config),
        ("source_contract", args.source_contract),
        ("runner", args.runner),
        ("lid_model", args.lid_model),
        ("receipts", args.receipts or (args.results_root / "candidate_input_receipts.json")),
    ):
        if path.is_file():
            hashes[f"{name}_sha256"] = sha256_file(path)
    write_report(output, {
        "schema_version": 1,
        "audit_kind": "b3_generation_followup_independent_audit",
        "status": "FAIL",
        "results_root": str(args.results_root.resolve()),
        "failure": failure,
        "complete_generation_rows_present": True,
        "candidate_record_counts": dict(record_counts),
        "input_hashes": hashes,
    })


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_audit(args)
    except Exception as exc:
        counts = retained_generation_counts(args.results_root)
        if counts is not None:
            try:
                write_fallback_failure_report(args, str(exc), counts)
            except Exception as report_exc:
                print(f"AUDIT_REPORT_WRITE_FAILED: {report_exc}", file=sys.stderr)
        print(f"AUDIT_NOT_PASSED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"], "report_path": report["report_path"], "languages": report["languages"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
