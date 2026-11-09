"""Free-generation evaluation for the frozen B3 French and Chinese models.

This follow-up is intentionally separate from the frozen teacher-forced B3
measurements. It evaluates one ordinary local Hugging Face checkpoint per
process with offline vLLM, and it does not download models or call remote APIs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from suture.data_manifest import read_manifest
from suture.paths import REPO_ROOT


CONFIG_PATH = REPO_ROOT / "configs" / "b3_generation_followup_v1.json"
SOURCE_CONTRACT_PATH = REPO_ROOT / "configs" / "b3_lighton_qwen3_8b_v1.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "b3" / "generation_followup_v1"
MATERIALIZATION_MANIFEST = "suture_graft_provenance.json"
CANDIDATES = ("host_baseline", "selected", "published_reference")
LANGUAGES = ("fr", "zh")
EXPECTED_FT_LABEL = {"fr": "__label__fr", "zh": "__label__zh"}
TEACHER_FORCED_RE = re.compile(
    r"\A<\|im_start\|>user\n(?P<question>.*?)\nAnswer:<\|im_end\|>\n"
    r"<\|im_start\|>assistant\n<think>\n\n</think>\n\n\Z",
    re.DOTALL,
)
STOP_MARKERS = ("<|im_end|>", "<|endoftext|>", "</s>")


class GenerationFollowupError(RuntimeError):
    """Raised when the generation follow-up cannot satisfy its contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GenerationFollowupError(message)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GenerationFollowupError(f"could not read JSON at {path}: {exc}") from exc
    _require(isinstance(payload, dict), f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aggregate_hash(file_hashes: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(file_hashes):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hashes[name].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def hash_snapshot(path: Path) -> Dict[str, Any]:
    """Hash all regular model files, following HF cache file symlinks."""

    root = path.expanduser().resolve()
    _require(root.is_dir(), f"model directory does not exist: {root}")
    files: Dict[str, str] = {}
    sizes: Dict[str, int] = {}
    for item in sorted(root.rglob("*")):
        if not item.is_file():
            continue
        relative = item.relative_to(root)
        if ".cache" in relative.parts or item.name == MATERIALIZATION_MANIFEST:
            continue
        key = relative.as_posix()
        files[key] = sha256_file(item)
        sizes[key] = item.stat().st_size
    _require(files, f"model directory has no regular files: {root}")
    return {
        "path": str(root),
        "file_count": len(files),
        "total_bytes": sum(sizes.values()),
        "file_hashes": files,
        "snapshot_sha256": _aggregate_hash(files),
    }


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    config = _read_json(path)
    _require(config.get("schema_version") == 1, "unsupported follow-up config schema")
    _require(config.get("study_name") == "b3_generation_followup_v1", "unexpected follow-up config")
    _require(config.get("status") == "frozen_before_generation", "follow-up config is not frozen")
    _require(config.get("candidate_names") == list(CANDIDATES), "unexpected candidate list")
    _require(config.get("languages") and set(config["languages"]) == set(LANGUAGES), "languages are incomplete")
    return config


def _language_config(config: Mapping[str, Any], language: str) -> Mapping[str, Any]:
    _require(language in LANGUAGES, f"unsupported language: {language}")
    return config["languages"][language]


def candidate_spec(config: Mapping[str, Any], language: str, candidate: str) -> Dict[str, Any]:
    _require(candidate in CANDIDATES, f"unsupported candidate: {candidate}")
    language_data = _language_config(config, language)
    if candidate == "host_baseline":
        model = dict(language_data["host"])
        graft: List[int] = []
    elif candidate == "selected":
        model = {
            "id": f"materialized:{language_data['host']['id']}+{config['donor']['id']}",
            "revision": None,
        }
        graft = list(language_data["selected_window"])
    else:
        model = dict(language_data["published_reference"])
        start, end = [int(item) for item in language_data["published_window_inclusive"]]
        graft = list(range(start, end + 1))
    return {"name": candidate, "model": model, "graft_layers": graft}


def extract_question(record: Mapping[str, Any]) -> str:
    match = TEACHER_FORCED_RE.fullmatch(str(record.get("input_text", "")))
    _require(match is not None, f"{record.get('id', '<unknown>')} is not the frozen MGSM prompt form")
    question = match.group("question").strip()
    _require(bool(question), f"{record.get('id', '<unknown>')} has an empty question")
    return question


def build_user_content(language: str, question: str, config: Mapping[str, Any] | None = None) -> str:
    config = config or load_config()
    language_data = _language_config(config, language)
    instruction = str(language_data["prompt_instruction"])
    _require(question.strip() == question and bool(question), "question must be nonempty and trimmed")
    return instruction + "\n" + question


def build_prompt_token_ids(tokenizer: Any, language: str, question: str, config: Mapping[str, Any] | None = None) -> List[int]:
    content = build_user_content(language, question, config)
    template = (config or load_config())["chat_template"]
    _require(template.get("enable_thinking") is True, "native reasoning must be enabled")
    try:
        value = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except Exception as exc:
        raise GenerationFollowupError(f"failed to render the pinned Qwen3 chat template: {exc}") from exc
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        _require(len(value) == 1, "prompt tokenizer returned more than one sequence")
        value = value[0]
    _require(isinstance(value, (list, tuple)) and value, "chat template returned no token IDs")
    try:
        ids = [int(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise GenerationFollowupError("chat template returned invalid token IDs") from exc
    _require(all(item >= 0 for item in ids), "prompt contains a negative token ID")
    return ids


def load_measurement_manifest(path: Path, language: str, config: Mapping[str, Any] | None = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    config = config or load_config()
    try:
        header, records = read_manifest(path)
    except Exception as exc:
        raise GenerationFollowupError(f"invalid held-out data manifest {path}: {exc}") from exc
    expected = _language_config(config, language)
    _require(header.get("_manifest") == f"b3_{language}_MGSM_rev2_test", "unexpected MGSM manifest name")
    _require(len(records) == int(config["dataset"]["n_items_per_language"]), "MGSM manifest item count mismatch")
    metadata = header.get("metadata", {})
    _require(metadata.get("language") == language, "MGSM manifest language mismatch")
    source = metadata.get("source", {})
    _require(source.get("dataset") == config["dataset"]["id"], "MGSM source dataset mismatch")
    _require(source.get("revision") == config["dataset"]["revision"], "MGSM source revision mismatch")
    identifiers = [str(item.get("id", "")) for item in records]
    _require(len(set(identifiers)) == len(records) and all(identifiers), "MGSM IDs are missing or duplicated")
    for index, record in enumerate(records):
        _require(record.get("language") == language, f"MGSM record {index} language mismatch")
        _require(record.get("dataset_revision") == config["dataset"]["revision"], f"MGSM record {index} revision mismatch")
        _require(record.get("answer_number") is not None, f"MGSM record {index} answer is missing")
        extract_question(record)
    return header, records


def _last_nonempty_line(text: str) -> str:
    clean = text
    for marker in STOP_MARKERS:
        clean = clean.replace(marker, "")
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def parse_final_answer(text: str, language: str) -> Dict[str, Any]:
    """Parse only the localized final answer line, never numbers in reasoning."""

    _require(language in LANGUAGES, f"unsupported language: {language}")
    if "</think>" in text:
        answer_text = text.split("</think>", 1)[1]
    else:
        answer_text = text
    final_line = _last_nonempty_line(answer_text)
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


def _fasttext_predict(model: Any, text: str) -> Tuple[str, float]:
    """Use fastText's native binding to avoid its NumPy 2 copy=False path."""

    native = getattr(model, "f", None)
    if native is None or not callable(getattr(native, "predict", None)):
        raise GenerationFollowupError("fastText model does not expose native f.predict")
    predictions = native.predict((text or " ").rstrip() + "\n", 1, 0.0, "strict")
    if not predictions:
        return "", 0.0
    probability, label = predictions[0]
    return str(label), float(probability)


def _letter_count(text: str) -> int:
    return sum(character.isalpha() for character in text)


def language_id(text: str, language: str, model: Any, config: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    config = config or load_config()
    minimum = int(config["language_id"]["minimum_letter_characters"])
    cleaned = " ".join(text.split())
    n_letters = _letter_count(cleaned)
    if n_letters < minimum:
        return {
            "status": "insufficient_text",
            "letter_characters": n_letters,
            "minimum_letter_characters": minimum,
            "predicted_label": None,
            "probability": None,
            "target_language_top1": None,
        }
    label, probability = _fasttext_predict(model, cleaned)
    return {
        "status": "evaluated",
        "letter_characters": n_letters,
        "minimum_letter_characters": minimum,
        "predicted_label": label,
        "probability": probability,
        "target_language_top1": label == EXPECTED_FT_LABEL[language],
    }


def _reasoning_segment(text: str) -> Tuple[str, str]:
    """Extract reasoning when the prompt may prefill the opening think marker."""

    if "</think>" in text:
        segment = text.split("</think>", 1)[0]
        if "<think>" in segment:
            segment = segment.split("<think>", 1)[1]
        return segment.strip(), "complete"
    if not text.strip():
        return "", "missing_reasoning_text"
    segment = text.split("<think>", 1)[1] if "<think>" in text else text
    return segment.strip(), "incomplete_no_think_close"


def _answer_segment(text: str, finish_reason: str | None) -> Tuple[str, str]:
    if "</think>" in text:
        return text.split("</think>", 1)[1], "post_think_segment"
    if finish_reason == "stop" and text.strip():
        return text, "completed_no_think_fallback"
    return "", "missing_think_close_before_completion"


def _english_chunk_diagnostic(text: str, model: Any, config: Mapping[str, Any]) -> Dict[str, Any]:
    spec = config["language_id"]["english_chunk_diagnostic"]
    chunk_size = int(spec["chunk_characters"])
    min_letters = int(spec["minimum_letter_characters"])
    threshold = float(spec["minimum_probability"])
    chunks: List[Dict[str, Any]] = []
    stride = int(spec.get("stride_characters", chunk_size))
    _require(stride == chunk_size, "English chunk diagnostic currently requires non-overlapping fixed chunks")
    for start in range(0, len(text), stride):
        chunk = text[start : start + chunk_size]
        letters = _letter_count(chunk)
        if letters < min_letters:
            continue
        label, probability = _fasttext_predict(model, " ".join(chunk.split()))
        chunks.append({
            "start_character": start,
            "letter_characters": letters,
            "predicted_label": label,
            "probability": probability,
            "english_chunk": label == "__label__en" and probability >= threshold,
        })
    english_n = sum(bool(item["english_chunk"]) for item in chunks)
    return {
        "status": "evaluated" if chunks else "no_eligible_chunks",
        "chunk_characters": chunk_size,
        "stride_characters": stride,
        "minimum_letter_characters": min_letters,
        "minimum_probability": threshold,
        "text_characters": len(text),
        "covered_characters": sum(min(chunk_size, max(0, len(text) - int(item["start_character"]))) for item in chunks),
        "eligible_character_coverage": (
            sum(min(chunk_size, max(0, len(text) - int(item["start_character"]))) for item in chunks) / len(text)
            if text else 0.0
        ),
        "eligible_chunks": len(chunks),
        "english_chunks": english_n,
        "any_english_chunk": bool(english_n) if chunks else None,
        "chunks": chunks,
    }


def _run_identity(payload: Mapping[str, Any]) -> str:
    stable = {key: value for key, value in payload.items() if key not in {"created_at", "status", "completed_records"}}
    return hashlib.sha256(json.dumps(stable, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _load_existing_rows(path: Path, expected_identity: str, expected_ids: set[str]) -> Dict[str, Dict[str, Any]]:
    existing: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return existing
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GenerationFollowupError(f"invalid output JSONL line {line_number}: {exc}") from exc
            record_id = str(row.get("record_id", ""))
            _require(record_id in expected_ids, f"output contains unexpected record ID {record_id!r}")
            _require(record_id not in existing, f"output duplicates record ID {record_id!r}")
            _require(row.get("run_identity") == expected_identity, "output row belongs to another run identity")
            existing[record_id] = row
    return existing


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _environment_report(image_digest: str) -> Dict[str, Any]:
    versions: Dict[str, Any] = {}
    for package in ("torch", "transformers", "vllm", "fasttext-wheel", "fasttext", "safetensors", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    report: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": versions,
        "container_image_digest": image_digest,
        "vllm_attention_backend_env": os.environ.get("VLLM_ATTENTION_BACKEND"),
        "vllm_use_v1_env": os.environ.get("VLLM_USE_V1"),
    }
    try:
        import torch

        report["torch_cuda_version"] = torch.version.cuda
        report["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            report["gpu_devices"] = [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "memory_bytes": int(torch.cuda.get_device_properties(index).total_memory),
                }
                for index in range(torch.cuda.device_count())
            ]
    except Exception as exc:
        report["torch_hardware_error"] = str(exc)
    return report


def _validate_image_digest(value: str) -> str:
    match = re.fullmatch(r"sha256:([0-9a-f]{64})", value.strip().lower())
    _require(match is not None, "container image digest must be sha256 followed by 64 lowercase hex characters")
    return "sha256:" + match.group(1)


def _load_runtime() -> Tuple[Any, Any, Any, Any]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        import fasttext
        import torch
    except ImportError as exc:
        raise GenerationFollowupError(f"generation requires torch, transformers, vLLM, and fastText: {exc}") from exc
    version = importlib.metadata.version("vllm")
    _require(version == "0.11.0", f"vLLM version must be 0.11.0, found {version}")
    return AutoTokenizer, LLM, SamplingParams, (fasttext, torch)


def _load_tokenizer(path: Path) -> Any:
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            str(path),
            trust_remote_code=False,
            local_files_only=True,
        )
    except Exception as exc:
        raise GenerationFollowupError(f"could not load local tokenizer {path}: {exc}") from exc


def _tokenizer_inventory(path: Path, tokenizer: Any | None = None) -> Dict[str, Any]:
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "config.json",
        "generation_config.json",
    )
    files = {name: sha256_file(path / name) for name in names if (path / name).is_file()}
    _require("tokenizer.json" in files and "tokenizer_config.json" in files, f"tokenizer files are incomplete: {path}")
    values: Dict[str, Any] = {"path": str(path.resolve()), "file_hashes": files, "sha256": _aggregate_hash(files)}
    if tokenizer is not None:
        values["special_token_ids"] = {
            name: getattr(tokenizer, name, None)
            for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")
        }
        values["chat_template_sha256"] = hashlib.sha256(
            str(getattr(tokenizer, "chat_template", "")).encode("utf-8")
        ).hexdigest()
    return values


def _checkpoint_decode_behavior(model_dir: Path, tokenizer_dir: Path) -> Dict[str, Any]:
    model_config = _read_json(model_dir / "config.json")
    generation_path = model_dir / "generation_config.json"
    generation_config = _read_json(generation_path) if generation_path.is_file() else {}
    tokenizer_config_path = tokenizer_dir / "tokenizer_config.json"
    tokenizer_config = _read_json(tokenizer_config_path)
    relevant = (
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "do_sample",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "max_length",
        "max_new_tokens",
    )
    return {
        "model_config_sha256": sha256_file(model_dir / "config.json"),
        "generation_config_sha256": sha256_file(generation_path) if generation_path.is_file() else None,
        "generation_config_values": {key: generation_config.get(key) for key in relevant if key in generation_config},
        "model_eos_token_id": model_config.get("eos_token_id"),
        "model_bos_token_id": model_config.get("bos_token_id"),
        "tokenizer_config_sha256": sha256_file(tokenizer_config_path),
        "tokenizer_special_tokens": {
            key: tokenizer_config.get(key)
            for key in ("bos_token", "eos_token", "pad_token", "additional_special_tokens")
            if key in tokenizer_config
        },
        "eos_behavior": "model-config EOS ends generation; ignore_eos=false; stop and stop_token_ids unset",
        "sampling_overrides_generation_config": [
            "temperature", "top_p", "top_k", "min_p", "presence_penalty",
            "frequency_penalty", "repetition_penalty", "max_tokens", "seed",
        ],
    }


def _validate_model_dir(path: Path) -> None:
    _require(path.is_dir(), f"model directory does not exist: {path}")
    config = _read_json(path / "config.json")
    _require(config.get("model_type") == "qwen3", "model config is not Qwen3")
    _require(config.get("num_hidden_layers") == 36, "model must have 36 layers")
    _require(config.get("architectures") == ["Qwen3ForCausalLM"], "unexpected model architecture")
    _require(not config.get("quantization_config"), "quantized model directory is forbidden")


def _candidate_materialization_check(
    candidate: str,
    model_path: Path,
    language: str,
    config: Mapping[str, Any],
) -> Dict[str, Any] | None:
    if candidate != "selected":
        return None
    manifest_path = model_path / MATERIALIZATION_MANIFEST
    _require(manifest_path.is_file(), f"selected candidate is missing {MATERIALIZATION_MANIFEST}")
    manifest = _read_json(manifest_path)
    spec = candidate_spec(config, language, candidate)
    _require(manifest.get("language") == language, "selected materialization language mismatch")
    _require(manifest.get("graft_layers") == spec["graft_layers"], "selected materialization window mismatch")
    _require(manifest.get("host_revision") == config["languages"][language]["host"]["revision"], "selected host revision mismatch")
    _require(manifest.get("donor_revision") == config["donor"]["revision"], "selected donor revision mismatch")
    _require(manifest.get("exact_mapping_verified") is True, "selected materialization lacks exact tensor mapping verification")
    return manifest


def _get_finish_reason(request_output: Any) -> str | None:
    if getattr(request_output, "outputs", None):
        return getattr(request_output.outputs[0], "finish_reason", None)
    return None


def _get_generated_tokens(request_output: Any) -> List[int]:
    outputs = getattr(request_output, "outputs", None)
    if not outputs:
        return []
    return [int(token) for token in (getattr(outputs[0], "token_ids", None) or [])]


def _get_generated_text(request_output: Any, tokenizer: Any, token_ids: Sequence[int]) -> Tuple[str, str]:
    outputs = getattr(request_output, "outputs", None)
    vllm_text = str(getattr(outputs[0], "text", "")) if outputs else ""
    try:
        decoded = tokenizer.decode(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        decoded = tokenizer.decode(list(token_ids), skip_special_tokens=False)
    return vllm_text, str(decoded)


def _prepare_record(
    record: Mapping[str, Any],
    prompt_ids: Sequence[int],
    *,
    language: str,
    candidate: str,
    expected: Mapping[str, Any],
    tokenizer: Any,
    lid_model: Any,
    run_identity: str,
    config: Mapping[str, Any],
    request_seed: int,
    generation_output: Any,
) -> Dict[str, Any]:
    generated_ids = _get_generated_tokens(generation_output)
    vllm_text, decoded_text = _get_generated_text(generation_output, tokenizer, generated_ids)
    finish_reason = _get_finish_reason(generation_output)
    answer_text, answer_segment_status = _answer_segment(decoded_text, finish_reason)
    answer = parse_final_answer(answer_text, language)
    answer["segment_status"] = answer_segment_status
    truncated = finish_reason == "length"
    prediction = answer["prediction"]
    try:
        exact_match = prediction is not None and int(prediction) == int(str(record["answer_number"]))
    except (TypeError, ValueError):
        exact_match = False
    if truncated:
        exact_match = False
    reasoning, reasoning_status = _reasoning_segment(decoded_text)
    if reasoning_status in {"complete", "incomplete_no_think_close"}:
        reasoning_lid = language_id(reasoning, language, lid_model, config)
        chunk_lid = _english_chunk_diagnostic(reasoning, lid_model, config)
        reasoning_lid["diagnostic_target_language_top1"] = reasoning_lid.get("target_language_top1")
        if truncated:
            reasoning_lid["status"] = "truncated_output"
            reasoning_lid["target_language_top1"] = None
        elif reasoning_status != "complete":
            reasoning_lid["status"] = "incomplete_reasoning"
            reasoning_lid["target_language_top1"] = None
        if truncated or reasoning_status != "complete":
            chunk_lid = {
                "status": "truncated_reasoning" if truncated else "incomplete_reasoning",
                "eligible_chunks": 0,
                "english_chunks": 0,
                "any_english_chunk": None,
                "partial_diagnostic": chunk_lid,
            }
    else:
        reasoning_lid = {
            "status": reasoning_status,
            "letter_characters": 0,
            "minimum_letter_characters": int(config["language_id"]["minimum_letter_characters"]),
            "predicted_label": None,
            "probability": None,
            "target_language_top1": None,
            "diagnostic_target_language_top1": None,
        }
        chunk_lid = {
            "status": reasoning_status,
            "eligible_chunks": 0,
            "english_chunks": 0,
            "any_english_chunk": None,
            "chunks": [],
        }
    if answer_segment_status == "post_think_segment":
        answer_content = re.sub(
            r"(?:Réponse(?:\s+finale)?|Answer|答案|最终答案)\s*[:：]",
            " ",
            answer_text,
            flags=re.IGNORECASE,
        )
        answer_lid = language_id(answer_content, language, lid_model, config)
    else:
        answer_lid = {
            "status": "not_evaluated_no_think_close",
            "letter_characters": 0,
            "minimum_letter_characters": int(config["language_id"]["minimum_letter_characters"]),
            "predicted_label": None,
            "probability": None,
            "target_language_top1": None,
        }
    return {
        "schema_version": 1,
        "run_identity": run_identity,
        "record_id": str(record["id"]),
        "record_hash": str(record.get("_hash", "")),
        "request_seed": int(request_seed),
        "language": language,
        "candidate": candidate,
        "graft_layers": list(expected["graft_layers"]),
        "question": extract_question(record),
        "ground_truth_answer": str(record["answer_number"]),
        "prompt_token_ids": [int(token) for token in prompt_ids],
        "prompt_sha256": hashlib.sha256(",".join(str(int(token)) for token in prompt_ids).encode("ascii")).hexdigest(),
        "generated_token_ids": generated_ids,
        "generated_token_count": len(generated_ids),
        "vllm_generated_text": vllm_text,
        "decoded_generated_text": decoded_text,
        "finish_reason": finish_reason,
        "truncated": bool(truncated),
        "answer_parse": answer,
        "exact_match": exact_match,
        "counted_correct": exact_match,
        "reasoning_segment": reasoning,
        "reasoning_segment_status": reasoning_status,
        "reasoning_segment_status": reasoning_status,
        "reasoning_language_id": reasoning_lid,
        "reasoning_english_chunk_diagnostic": chunk_lid,
        "answer_language_id": answer_lid,
    }


def run_candidate(
    *,
    language: str,
    candidate: str,
    model_dir: Path,
    tokenizer_dir: Path,
    manifest_path: Path,
    lid_path: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    config_path: Path = CONFIG_PATH,
    source_contract_path: Path = SOURCE_CONTRACT_PATH,
    container_image_digest: str,
    resume: bool = True,
    llm_factory: Callable[..., Any] | None = None,
    sampling_params_factory: Callable[..., Any] | None = None,
    tokenizer: Any | None = None,
    lid_model: Any | None = None,
    enforce_cuda: bool = True,
) -> Dict[str, Any]:
    """Run all 250 records for one candidate in fixed-size vLLM batches."""

    started = time.perf_counter()
    image_digest = _validate_image_digest(container_image_digest)
    config = load_config(config_path)
    _require(language in LANGUAGES and candidate in CANDIDATES, "invalid language or candidate")
    _validate_model_dir(model_dir)
    _require(tokenizer_dir.is_dir(), f"tokenizer directory does not exist: {tokenizer_dir}")
    _require(lid_path.is_file(), f"fastText LID model is missing: {lid_path}")
    header, records = load_measurement_manifest(manifest_path, language, config)
    identity = candidate_spec(config, language, candidate)
    materialization = _candidate_materialization_check(candidate, model_dir, language, config)
    if tokenizer is None:
        tokenizer = _load_tokenizer(tokenizer_dir)
    tokenizer_inventory = _tokenizer_inventory(tokenizer_dir, tokenizer)
    lid_hash = sha256_file(lid_path)
    if lid_model is None:
        try:
            import fasttext

            lid_model = fasttext.load_model(str(lid_path))
        except Exception as exc:
            raise GenerationFollowupError(f"could not load local fastText model {lid_path}: {exc}") from exc
    if enforce_cuda:
        try:
            import torch
        except ImportError as exc:
            raise GenerationFollowupError("PyTorch is required for vLLM execution") from exc
        _require(bool(torch.cuda.is_available()), "CUDA is unavailable; refusing CPU model generation")
    if llm_factory is None or sampling_params_factory is None:
        _AutoTokenizer, default_llm, default_sampling, (fasttext_module, _torch) = _load_runtime()
        del _AutoTokenizer, fasttext_module, _torch
        llm_factory = llm_factory or default_llm
        sampling_params_factory = sampling_params_factory or default_sampling
    version = importlib.metadata.version("vllm") if llm_factory.__module__.startswith("vllm") else "injected-test-runtime"
    if version != "injected-test-runtime":
        _require(version == "0.11.0", f"vLLM version must be 0.11.0, found {version}")

    source_contract = _read_json(source_contract_path)
    _require(source_contract.get("contract_name") == "b3_lighton_qwen3_8b_v1", "frozen source contract mismatch")
    model_inventory = hash_snapshot(model_dir)
    data_hash = sha256_file(manifest_path)
    config_hash = sha256_file(config_path)
    source_contract_hash = sha256_file(source_contract_path)
    code_hash = sha256_file(Path(__file__))
    output_dir = output_root / language / candidate
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "generations.jsonl"
    run_path = output_dir / "run.json"
    environment = _environment_report(image_digest)
    run_payload = {
        "schema_version": 1,
        "run_kind": "b3_free_generation_followup",
        "status": "RUNNING",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "language": language,
        "candidate": candidate,
        "candidate_spec": identity,
        "model_path": str(model_dir.resolve()),
        "model_snapshot": model_inventory,
        "tokenizer": tokenizer_inventory,
        "checkpoint_decode_behavior": _checkpoint_decode_behavior(model_dir, tokenizer_dir),
        "data_manifest": {"path": str(manifest_path.resolve()), "sha256": data_hash, "header": header, "n_records": len(records)},
        "fasttext_lid": {"path": str(lid_path.resolve()), "sha256": lid_hash},
        "followup_config": {"path": str(config_path.resolve()), "sha256": config_hash},
        "frozen_source_contract": {"path": str(source_contract_path.resolve()), "sha256": source_contract_hash},
        "runner_sha256": code_hash,
        "container_image_digest": image_digest,
        "runtime": environment,
        "generation": dict(config["generation"]),
        "prompt_protocol": {
            "template": dict(config["chat_template"]),
            "instruction": config["languages"][language]["prompt_instruction"],
            "instruction_sha256": hashlib.sha256(config["languages"][language]["prompt_instruction"].encode("utf-8")).hexdigest(),
        },
        "materialization_manifest": materialization,
        "completed_records": 0,
        "output_file": str(output_path.resolve()),
    }
    run_identity = _run_identity(run_payload)
    run_payload["run_identity"] = run_identity
    if run_path.exists():
        old_run = _read_json(run_path)
        _require(old_run.get("run_identity") == run_identity, "existing run metadata differs; refusing unsafe resume")
        if not resume:
            raise GenerationFollowupError(f"output already exists and resume is disabled: {output_dir}")
    elif output_path.exists() and output_path.stat().st_size > 0:
        raise GenerationFollowupError("generation rows exist without a run manifest")
    existing = _load_existing_rows(output_path, run_identity, {str(item["id"]) for item in records})
    _require(resume or not existing, "existing generation rows require --resume")
    if len(existing) == len(records):
        run_payload.update({"status": "PASS_GENERATION", "completed_records": len(existing), "resumed_without_generation": True})
        write_json(run_path, run_payload)
        return run_payload
    write_json(run_path, run_payload)

    prompt_ids: Dict[str, List[int]] = {}
    for record in records:
        prompt_ids[str(record["id"])] = build_prompt_token_ids(
            tokenizer,
            language,
            extract_question(record),
            config,
        )
    record_by_id = {str(item["id"]): item for item in records}
    ordered_pending = [str(item["id"]) for item in records if str(item["id"]) not in existing]
    sampling = config["generation"]
    heartbeat_stop = threading.Event()

    def heartbeat() -> None:
        while not heartbeat_stop.wait(60):
            print(
                f"HEARTBEAT {language}/{candidate}: {len(existing)}/{len(records)} records, "
                f"elapsed={time.perf_counter() - started:.0f}s",
                flush=True,
            )

    heartbeat_thread = threading.Thread(target=heartbeat, name="b3-generation-heartbeat", daemon=True)
    heartbeat_thread.start()
    print(f"START {language}/{candidate}: {len(records)} records, vLLM {version}", flush=True)
    try:
        llm = llm_factory(
            model=str(model_dir.resolve()),
            tokenizer=str(tokenizer_dir.resolve()),
            tokenizer_mode="auto",
            trust_remote_code=False,
            tensor_parallel_size=1,
            dtype="bfloat16",
            quantization=None,
            seed=int(sampling["seed"]),
            gpu_memory_utilization=float(sampling["gpu_memory_utilization"]),
            max_model_len=int(sampling["max_model_len"]),
            max_num_seqs=int(sampling["max_num_seqs"]),
            max_num_batched_tokens=int(sampling["max_num_batched_tokens"]),
            enable_prefix_caching=False,
            enforce_eager=bool(sampling["enforce_eager"]),
        )
    except Exception as exc:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=2)
        raise GenerationFollowupError(f"could not initialize vLLM for {candidate}: {exc}") from exc

    batch_size = int(sampling["batch_size"])
    global_seed = int(sampling["seed"])
    try:
        for offset in range(0, len(ordered_pending), batch_size):
            current_ids = ordered_pending[offset : offset + batch_size]
            prompt_batch = [{"prompt_token_ids": prompt_ids[record_id]} for record_id in current_ids]
            params = []
            request_seeds: Dict[str, int] = {}
            for record_id in current_ids:
                digest = hashlib.sha256(f"{global_seed}:{record_id}".encode("utf-8")).digest()
                request_seed = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
                request_seeds[record_id] = request_seed
                params.append(sampling_params_factory(
                    n=int(sampling["n"]),
                    best_of=None,
                    presence_penalty=float(sampling["presence_penalty"]),
                    frequency_penalty=float(sampling["frequency_penalty"]),
                    repetition_penalty=float(sampling["repetition_penalty"]),
                    temperature=float(sampling["temperature"]),
                    top_p=float(sampling["top_p"]),
                    top_k=int(sampling["top_k"]),
                    min_p=float(sampling["min_p"]),
                    max_tokens=int(sampling["max_tokens"]),
                    min_tokens=int(sampling["min_tokens"]),
                    seed=request_seed,
                    stop=sampling["stop"],
                    stop_token_ids=sampling["stop_token_ids"],
                    ignore_eos=bool(sampling["ignore_eos"]),
                    logprobs=None,
                    prompt_logprobs=None,
                    include_stop_str_in_output=bool(sampling["include_stop_str_in_output"]),
                    skip_special_tokens=False,
                    detokenize=True,
                    spaces_between_special_tokens=bool(sampling["spaces_between_special_tokens"]),
                    truncate_prompt_tokens=None,
                    bad_words=None,
                    logit_bias=None,
                    allowed_token_ids=None,
                    extra_args=None,
                ))
            outputs = llm.generate(prompt_batch, sampling_params=params, use_tqdm=False)
            _require(len(outputs) == len(current_ids), "vLLM returned a different number of outputs than prompts")
            rows = [
                _prepare_record(
                    record_by_id[record_id],
                    prompt_ids[record_id],
                    language=language,
                    candidate=candidate,
                    expected=identity,
                    tokenizer=tokenizer,
                    lid_model=lid_model,
                    run_identity=run_identity,
                    config=config,
                    request_seed=request_seeds[record_id],
                    generation_output=output,
                )
                for record_id, output in zip(current_ids, outputs)
            ]
            _write_rows(output_path, rows)
            existing.update({str(row["record_id"]): dict(row) for row in rows})
            run_payload["completed_records"] = len(existing)
            run_payload["last_completed_batch"] = {"start": offset, "size": len(rows)}
            write_json(run_path, run_payload)
            print(
                f"PROGRESS {language}/{candidate}: {len(existing)}/{len(records)} records, "
                f"elapsed={time.perf_counter() - started:.0f}s",
                flush=True,
            )
    except Exception as exc:
        run_payload.update({"status": "INCOMPLETE", "completed_records": len(existing), "error": f"{type(exc).__name__}: {exc}"})
        write_json(run_path, run_payload)
        if isinstance(exc, GenerationFollowupError):
            raise
        raise GenerationFollowupError(f"generation failed after {len(existing)} records: {exc}") from exc
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=2)
        del llm
    _require(len(existing) == len(records), "generation did not finish all MGSM records")
    run_payload.update({
        "status": "PASS_GENERATION",
        "completed_records": len(existing),
        "resumed_without_generation": False,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "truncated_records": sum(bool(row["truncated"]) for row in existing.values()),
    })
    write_json(run_path, run_payload)
    return run_payload


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        return float("nan")
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def paired_bootstrap(values: Sequence[float], *, seed: int, n_resamples: int) -> Dict[str, Any]:
    _require(len(values) > 0, "bootstrap requires at least one paired record")
    rng = random.Random(seed)
    n = len(values)
    observed = sum(values) / n
    means = []
    for _ in range(n_resamples):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return {
        "mean_difference": observed,
        "ci95_percentile": [_quantile(means, 0.025), _quantile(means, 0.975)],
        "n_paired_records": n,
        "bootstrap_resamples": n_resamples,
        "bootstrap_seed": seed,
    }


def _read_generation_rows(path: Path, run_path: Path, language: str, candidate: str) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    run = _read_json(run_path)
    _require(run.get("status") == "PASS_GENERATION", f"run is incomplete: {run_path}")
    _require(run.get("language") == language and run.get("candidate") == candidate, "generation metadata key mismatch")
    rows: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            record_id = str(item.get("record_id", ""))
            _require(record_id and record_id not in rows, f"duplicate or missing output ID in {path}:{line_number}")
            _require(item.get("run_identity") == run.get("run_identity"), f"output run identity mismatch in {path}:{line_number}")
            rows[record_id] = item
    _require(len(rows) == 250, f"{path} must contain exactly 250 outputs")
    return run, rows


def _metric_summary(rows: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    values = list(rows.values())
    n = len(values)
    correct = sum(bool(row.get("exact_match")) for row in values)
    parse_status = Counter(str(row.get("answer_parse", {}).get("status", "missing")) for row in values)
    format_status = Counter(str(row.get("answer_parse", {}).get("format_status", "unknown")) for row in values)
    strict_format = sum(bool(row.get("answer_parse", {}).get("strict_localized_format")) for row in values)
    truncated = sum(bool(row.get("truncated")) for row in values)
    lid_values = [row.get("reasoning_language_id", {}) for row in values]
    evaluated = [item for item in lid_values if item.get("status") == "evaluated"]
    lid_target = sum(item.get("target_language_top1") is True for item in evaluated)
    lid_unknown = n - len(evaluated)
    all_target_count = lid_target
    chunks = [row.get("reasoning_english_chunk_diagnostic", {}) for row in values]
    chunk_evaluable = [item for item in chunks if item.get("status") == "evaluated"]
    english_any = sum(item.get("any_english_chunk") is True for item in chunk_evaluable)
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
            "unknown_count": lid_unknown,
            "denominator_all_records": n,
            "target_top1_rate_all_records": all_target_count / n,
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
            "evaluated_count": sum(row.get("answer_language_id", {}).get("status") == "evaluated" for row in values),
            "unknown_count": sum(row.get("answer_language_id", {}).get("status") != "evaluated" for row in values),
            "target_top1_count": sum(row.get("answer_language_id", {}).get("target_language_top1") is True for row in values),
            "denominator_all_records": n,
        },
    }


def aggregate_language(
    language: str,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    config_path: Path = CONFIG_PATH,
) -> Dict[str, Any]:
    config = load_config(config_path)
    candidate_rows: Dict[str, Dict[str, Dict[str, Any]]] = {}
    candidate_runs: Dict[str, Dict[str, Any]] = {}
    for candidate in CANDIDATES:
        directory = output_root / language / candidate
        run, rows = _read_generation_rows(directory / "generations.jsonl", directory / "run.json", language, candidate)
        candidate_runs[candidate] = run
        candidate_rows[candidate] = rows
    manifest_path = Path(candidate_runs["host_baseline"]["data_manifest"]["path"])
    _require(sha256_file(manifest_path) == candidate_runs["host_baseline"]["data_manifest"]["sha256"], "source MGSM manifest hash changed after generation")
    _header, source_records = load_measurement_manifest(manifest_path, language, config)
    source_by_id = {str(item["id"]): item for item in source_records}
    id_sets = {candidate: set(rows) for candidate, rows in candidate_rows.items()}
    _require(all(ids == id_sets[CANDIDATES[0]] for ids in list(id_sets.values())[1:]), "candidate output record IDs differ")
    _require(id_sets["host_baseline"] == set(source_by_id), "candidate outputs do not exactly cover the source MGSM record IDs")
    for candidate, rows in candidate_rows.items():
        for record_id, row in rows.items():
            source = source_by_id[record_id]
            _require(row.get("record_hash") == source.get("_hash"), f"{candidate} output source record hash mismatch for {record_id}")
            _require(str(row.get("ground_truth_answer")) == str(source.get("answer_number")), f"{candidate} output gold answer mismatch for {record_id}")
            prediction = row.get("answer_parse", {}).get("prediction")
            try:
                expected_correct = prediction is not None and int(prediction) == int(str(source["answer_number"])) and not bool(row.get("truncated"))
            except (TypeError, ValueError):
                expected_correct = False
            _require(bool(row.get("exact_match")) == bool(expected_correct), f"{candidate} exact-match flag disagrees with parsed answer for {record_id}")
    base_run = candidate_runs["host_baseline"]
    for candidate in CANDIDATES[1:]:
        run = candidate_runs[candidate]
        for key in ("data_manifest", "followup_config", "frozen_source_contract", "prompt_protocol", "fasttext_lid"):
            _require(run.get(key) == base_run.get(key), f"candidate {candidate} differs in common protocol field {key}")
        _require(run.get("container_image_digest") == base_run.get("container_image_digest"), "candidate runs use different container digests")
        _require(run.get("runtime", {}).get("package_versions") == base_run.get("runtime", {}).get("package_versions"), "candidate runs use different software versions")
        for runtime_key in ("torch_cuda_version", "cuda_available", "gpu_devices"):
            _require(run.get("runtime", {}).get(runtime_key) == base_run.get("runtime", {}).get(runtime_key), f"candidate runs differ in runtime hardware field {runtime_key}")
        _require(run.get("tokenizer", {}).get("sha256") == base_run.get("tokenizer", {}).get("sha256"), "candidate runs use different tokenizer files")
        for decode_key in ("model_eos_token_id", "model_bos_token_id", "tokenizer_special_tokens"):
            _require(
                run.get("checkpoint_decode_behavior", {}).get(decode_key)
                == base_run.get("checkpoint_decode_behavior", {}).get(decode_key),
                f"candidate runs differ in decode behavior field {decode_key}",
            )
        for record_id in sorted(id_sets["host_baseline"]):
            left = candidate_rows[candidate][record_id]
            base = candidate_rows["host_baseline"][record_id]
            _require(left.get("prompt_token_ids") == base.get("prompt_token_ids"), f"candidate prompt token IDs differ for {record_id}")
            _require(left.get("request_seed") == base.get("request_seed"), f"candidate per-record seed differs for {record_id}")
    summaries = {candidate: _metric_summary(candidate_rows[candidate]) for candidate in CANDIDATES}
    comparisons: Dict[str, Any] = {}
    comparisons_to_make = (
        ("selected_minus_host", "selected", "host_baseline"),
        ("published_minus_host", "published_reference", "host_baseline"),
        ("selected_minus_published", "selected", "published_reference"),
    )
    base_seed = int(config["metrics"]["bootstrap_seed"])
    n_resamples = int(config["metrics"]["bootstrap_resamples"])
    ordered_ids = sorted(id_sets["host_baseline"])
    for name, left, right in comparisons_to_make:
        accuracy_diff = [
            float(bool(candidate_rows[left][record_id]["exact_match"]))
            - float(bool(candidate_rows[right][record_id]["exact_match"]))
            for record_id in ordered_ids
        ]
        seed = base_seed + int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "big")
        comparisons[name] = {
            "accuracy_all_records": paired_bootstrap(accuracy_diff, seed=seed, n_resamples=n_resamples),
        }
        lid_diff = [
            float(candidate_rows[left][record_id]["reasoning_language_id"].get("target_language_top1") is True)
            - float(candidate_rows[right][record_id]["reasoning_language_id"].get("target_language_top1") is True)
            for record_id in ordered_ids
        ]
        comparisons[name]["reasoning_target_lid_top1_all_records"] = paired_bootstrap(
            lid_diff,
            seed=seed + 1,
            n_resamples=n_resamples,
        )
        english_diff = [
            float(candidate_rows[left][record_id]["reasoning_english_chunk_diagnostic"].get("any_english_chunk") is True)
            - float(candidate_rows[right][record_id]["reasoning_english_chunk_diagnostic"].get("any_english_chunk") is True)
            for record_id in ordered_ids
        ]
        comparisons[name]["reasoning_english_chunk_any_all_records"] = paired_bootstrap(
            english_diff,
            seed=seed + 2,
            n_resamples=n_resamples,
        )
    summary = {
        "schema_version": 1,
        "status": "PASS_AGGREGATE",
        "run_kind": "b3_free_generation_followup",
        "language": language,
        "candidate_summaries": summaries,
        "paired_comparisons": comparisons,
        "estimands": {
            "accuracy": "exact integer answer from the final localized answer line; all unparseable and truncated outputs count incorrect",
            "reasoning_language_id": "fastText top-1 label on complete generated reasoning after at least 20 alphabetic characters; incomplete, truncated, or too-short traces are unknown and remain in the all-record denominator",
            "english_chunk_diagnostic": "any non-overlapping 200-character reasoning chunk with at least 20 alphabetic characters classified as English at probability >= 0.5; unknown counts as no positive in all-record rate",
            "seed_scope": "one fixed per-record seed schedule; descriptive follow-up, not the paper's 10-seed estimate",
        },
        "candidate_run_identities": {key: candidate_runs[key]["run_identity"] for key in CANDIDATES},
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_root / language / "summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run one candidate in offline vLLM")
    run.add_argument("--language", choices=LANGUAGES, required=True)
    run.add_argument("--candidate", choices=CANDIDATES, required=True)
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--tokenizer-dir", type=Path, required=True, help="use the pinned host tokenizer for every candidate")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--lid-model", type=Path, default=REPO_ROOT / "models" / "lid.176.ftz")
    run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    run.add_argument("--config", type=Path, default=CONFIG_PATH)
    run.add_argument("--source-contract", type=Path, default=SOURCE_CONTRACT_PATH)
    run.add_argument("--container-image-digest", default=os.environ.get("VLLM_CONTAINER_IMAGE_DIGEST"))
    run.add_argument("--no-resume", action="store_true")

    aggregate = sub.add_parser("aggregate", help="aggregate the three completed candidate runs")
    aggregate.add_argument("--language", choices=LANGUAGES, required=True)
    aggregate.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    aggregate.add_argument("--config", type=Path, default=CONFIG_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "aggregate":
            report = aggregate_language(args.language, output_root=args.output_root, config_path=args.config)
        else:
            image = args.container_image_digest or ""
            report = run_candidate(
                language=args.language,
                candidate=args.candidate,
                model_dir=args.model_dir,
                tokenizer_dir=args.tokenizer_dir,
                manifest_path=args.manifest,
                lid_path=args.lid_model,
                output_root=args.output_root,
                config_path=args.config,
                source_contract_path=args.source_contract,
                container_image_digest=image,
                resume=not args.no_resume,
            )
    except GenerationFollowupError as exc:
        print(f"STOP_B3_GENERATION_FOLLOWUP: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
