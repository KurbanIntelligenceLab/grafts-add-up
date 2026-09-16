"""Fail-closed preparation and selection runner for the LightOn Qwen3-8B B3 study.

This module is intentionally separate from the frozen Tier-A and Qwen3-1.7B
protocols.  It can validate exact local Hugging Face snapshots without loading
weights, and it can score declared probe manifests with two independent full
models.  Selection uses only the residual adapter; no routed graft model is
constructed until a later measurement stage.

The expected cloud workflow is:

    python -m suture.b3_lighton validate-contract
    python -m suture.b3_lighton prepare --download-dir /data/models
    python -m suture.b3_lighton preflight --language fr \
        --host-path /data/models/Qwen3-8B-FR \
        --donor-path /data/models/Qwen3-8B-EN \
        --load-models
    python -m suture.b3_lighton score --language fr \
        --host-path /data/models/Qwen3-8B-FR \
        --donor-path /data/models/Qwen3-8B-EN
    python -m suture.b3_lighton measure --language fr \
        --host-path /data/models/Qwen3-8B-FR \
        --donor-path /data/models/Qwen3-8B-EN \
        --measurement-manifest results/b3/lighton_qwen3_8b/fr_host__en_donor/data/MGSM_rev2_test.jsonl

No command in this file may substitute a mutable revision, quantized model,
remote-code class, or a different contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from suture.paths import REPO_ROOT


CONTRACT_RELATIVE = Path("configs") / "b3_lighton_qwen3_8b_v1.json"
ARCHITECTURE_FIELDS = (
    "model_type",
    "class_name",
    "layer_path",
    "n_layers",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_size",
    "max_position_embeddings",
    "torch_dtype",
)


class B3Error(RuntimeError):
    """Raised when the B3 contract cannot be satisfied."""


def repo_root() -> Path:
    return REPO_ROOT


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise B3Error(message)


def _json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise B3Error(f"could not read JSON file {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def contract_path(root: Path | None = None) -> Path:
    return (root or repo_root()) / CONTRACT_RELATIVE


def _is_revision(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 40:
        return False
    return all(character in "0123456789abcdef" for character in value.lower())


def _window(value: Any, *, name: str, n_layers: int) -> Tuple[int, int]:
    _require(
        isinstance(value, list) and len(value) == 2 and all(isinstance(x, int) for x in value),
        f"{name} must be a two-element integer list",
    )
    start, end = value
    _require(0 <= start <= end < n_layers, f"{name} is outside [0, {n_layers})")
    return start, end


def validate_contract(contract: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and return a detached copy of the frozen B3 contract."""

    value = json.loads(json.dumps(dict(contract)))
    _require(value.get("contract_version") == 1, "B3 contract_version must be 1")
    _require(
        value.get("contract_name") == "b3_lighton_qwen3_8b_v1",
        "unexpected B3 contract name",
    )
    _require(
        value.get("contract_status") == "frozen_preflight_only",
        "B3 contract is not in its frozen preflight-only state",
    )

    architecture = value.get("architecture")
    _require(isinstance(architecture, dict), "contract architecture is missing")
    expected_architecture = {
        "model_type": "qwen3",
        "class_name": "Qwen3ForCausalLM",
        "layer_path": "model.layers",
        "n_layers": 36,
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "max_position_embeddings": 40960,
        "torch_dtype": "bfloat16",
    }
    for key, expected in expected_architecture.items():
        _require(
            architecture.get(key) == expected,
            f"contract architecture.{key} must be {expected!r}",
        )

    models = value.get("models")
    _require(isinstance(models, dict), "contract models are missing")
    donor = models.get("donor")
    hosts = models.get("hosts")
    _require(isinstance(donor, dict), "contract donor is missing")
    _require(isinstance(hosts, list) and hosts, "contract hosts are empty")
    _require(_is_revision(donor.get("revision")), "donor revision must be an immutable SHA")
    _require(donor.get("id") == "lightonai/Qwen3-8B-EN", "unexpected donor model")

    languages = set()
    for host in hosts:
        _require(isinstance(host, dict), "each host must be an object")
        language = host.get("language")
        _require(isinstance(language, str) and language not in languages, "host languages must be unique")
        languages.add(language)
        _require(_is_revision(host.get("revision")), f"{language} revision must be an immutable SHA")
        _require(isinstance(host.get("id"), str), f"{language} host id is missing")
        _require(host.get("id") != donor.get("id"), f"{language} host and donor must differ")
        published = host.get("published_swap")
        _require(isinstance(published, dict), f"{language} published swap is missing")
        _require(
            _is_revision(published.get("revision")),
            f"{language} published swap revision must be an immutable SHA",
        )
        _window(
            published.get("window_inclusive"),
            name=f"{language}.published_swap.window_inclusive",
            n_layers=architecture["n_layers"],
        )

    loading = value.get("loading")
    _require(isinstance(loading, dict), "contract loading section is missing")
    for key, expected in (
        ("local_files_only", True),
        ("trust_remote_code", False),
        ("dtype", "bfloat16"),
        ("attn_implementation", "eager"),
        ("quantization", None),
        ("selection_use_cache", False),
        ("measurement_use_cache", False),
    ):
        _require(loading.get(key) == expected, f"contract loading.{key} must be {expected!r}")

    tokenizer = value.get("tokenizer")
    _require(isinstance(tokenizer, dict), "contract tokenizer section is missing")
    required_tokenizer_files = tokenizer.get("required_files")
    _require(
        isinstance(required_tokenizer_files, list)
        and required_tokenizer_files
        and all(isinstance(item, str) for item in required_tokenizer_files),
        "contract tokenizer.required_files is invalid",
    )
    _require(tokenizer.get("must_match_across_pair") is True, "pair tokenizer equality must be required")
    _require(
        value.get("data", {})
        .get("probe_manifest_format", {})
        .get("readout_uses_first_token_only")
        is True,
        "B3 readout policy must use the first token only",
    )

    selection = value.get("data", {}).get("selection")
    _require(isinstance(selection, dict), "contract data.selection section is missing")
    _require(selection.get("shape") == "interval", "B3 selection shape must be interval")
    _require(selection.get("tau") == 0.05, "B3 selection tau must be 0.05")
    _require(selection.get("batch_size") == 1, "B3 selection batch size must be 1")
    _require(isinstance(selection.get("max_length"), int) and selection["max_length"] > 0, "invalid B3 max_length")
    windows = value.get("data", {}).get("windows", {})
    _require(windows.get("shape") == "contiguous", "B3 window shape must be contiguous")
    _require(windows.get("expected_count") == 36 * 37 // 2, "B3 window count is inconsistent")
    measurement = value.get("data", {}).get("held_out_measurement")
    _require(isinstance(measurement, dict), "contract held-out measurement section is missing")
    _require(
        measurement.get("objective") == "mean teacher-forced answer log-probability per target token",
        "B3 measurement objective is not frozen",
    )
    _require(
        measurement.get("candidates") == ["host_baseline", "selected", "published_reference"],
        "B3 measurement candidate set is not frozen",
    )
    _require(measurement.get("cache") is False, "B3 measurement cache policy must be false")
    _require(measurement.get("batch_size") == 1, "B3 measurement batch size must be 1")
    _require(measurement.get("max_length") == 256, "B3 measurement max_length must be 256")
    _require(measurement.get("truncation") == "forbid", "B3 measurement truncation policy must be forbid")

    results = value.get("results")
    _require(isinstance(results, dict), "contract results section is missing")
    _require(results.get("root") == "results/b3/lighton_qwen3_8b", "B3 result root is not isolated")
    for key in ("held_out_measurement_manifest", "measurement", "measurements"):
        _require(isinstance(results.get(key), str) and results[key], f"B3 results.{key} is missing")
    return value


def load_contract(path: Path | None = None) -> Dict[str, Any]:
    path = path or contract_path()
    return validate_contract(_json(path))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise B3Error(f"could not hash {path}: {exc}") from exc
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and ".cache" not in path.relative_to(root).parts:
            yield path


def _aggregate_hash(file_hashes: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(file_hashes):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hashes[name].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _architecture_summary(config: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "model_type": config.get("model_type"),
        "class_name": (config.get("architectures") or [None])[0],
        "layer_path": "model.layers",
        "n_layers": config.get("num_hidden_layers"),
        "hidden_size": config.get("hidden_size"),
        "intermediate_size": config.get("intermediate_size"),
        "num_attention_heads": config.get("num_attention_heads"),
        "num_key_value_heads": config.get("num_key_value_heads"),
        "head_dim": config.get("head_dim"),
        "vocab_size": config.get("vocab_size"),
        "max_position_embeddings": config.get("max_position_embeddings"),
        "torch_dtype": config.get("torch_dtype"),
    }


def _required_weight_files(snapshot: Path) -> List[str]:
    index_path = snapshot / "model.safetensors.index.json"
    _require(index_path.is_file(), f"missing safetensors index: {index_path}")
    index = _json(index_path)
    weight_map = index.get("weight_map")
    _require(isinstance(weight_map, dict) and weight_map, "safetensors index has no weight_map")
    names = sorted(set(weight_map.values()))
    _require(all(isinstance(name, str) for name in names), "safetensors weight_map contains invalid names")
    missing = [name for name in names if not (snapshot / name).is_file()]
    _require(not missing, f"snapshot is missing weight shards: {missing}")
    return names


def inspect_snapshot(
    snapshot: Path,
    model_spec: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    hash_files: bool = True,
) -> Dict[str, Any]:
    """Validate one local, immutable HF snapshot and return its inventory."""

    snapshot = snapshot.expanduser().resolve()
    _require(snapshot.is_dir(), f"snapshot directory does not exist: {snapshot}")
    config_path = snapshot / "config.json"
    config = _json(config_path)
    expected_architecture = contract["architecture"]
    observed = _architecture_summary(config)
    for key in ARCHITECTURE_FIELDS:
        _require(
            observed.get(key) == expected_architecture.get(key),
            f"{model_spec.get('id')} config mismatch for {key}: "
            f"{observed.get(key)!r} != {expected_architecture.get(key)!r}",
        )
    _require(
        config.get("architectures") == [expected_architecture["class_name"]],
        f"{model_spec.get('id')} has an unexpected architectures field",
    )

    required_tokenizer_files = contract["tokenizer"]["required_files"]
    missing_tokenizer = [
        name for name in required_tokenizer_files if not (snapshot / name).is_file()
    ]
    _require(not missing_tokenizer, f"snapshot is missing tokenizer files: {missing_tokenizer}")
    tokenizer_config = _json(snapshot / "tokenizer_config.json")
    _require(
        tokenizer_config.get("pad_token") is not None,
        f"{model_spec.get('id')} tokenizer has no declared pad_token",
    )
    _require(
        tokenizer_config.get("chat_template"),
        f"{model_spec.get('id')} tokenizer has no chat_template",
    )
    weight_files = _required_weight_files(snapshot)

    file_hashes: Dict[str, str] = {}
    if hash_files:
        file_hashes = {
            _relative(path, snapshot): sha256_file(path) for path in _iter_files(snapshot)
        }
    return {
        "id": model_spec["id"],
        "revision": model_spec["revision"],
        "path": str(snapshot),
        "architecture": observed,
        "tokenizer": {
            "required_files": required_tokenizer_files,
            "pad_token": tokenizer_config.get("pad_token"),
            "padding_side": tokenizer_config.get("padding_side"),
            "chat_template_sha256": hashlib.sha256(
                str(tokenizer_config["chat_template"]).encode("utf-8")
            ).hexdigest(),
        },
        "weight_files": weight_files,
        "file_count": len(file_hashes) if hash_files else None,
        "file_hashes": file_hashes,
        "snapshot_sha256": _aggregate_hash(file_hashes) if hash_files else None,
    }


def validate_pair_snapshots(
    host_path: Path,
    donor_path: Path,
    host_spec: Mapping[str, Any],
    donor_spec: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    hash_files: bool = True,
) -> Dict[str, Any]:
    host = inspect_snapshot(host_path, host_spec, contract, hash_files=hash_files)
    donor = inspect_snapshot(donor_path, donor_spec, contract, hash_files=hash_files)
    for key in ARCHITECTURE_FIELDS:
        _require(
            host["architecture"][key] == donor["architecture"][key],
            f"host/donor architecture mismatch for {key}",
        )

    for filename in contract["tokenizer"]["required_files"]:
        host_file = host_path / filename
        donor_file = donor_path / filename
        _require(
            host_file.read_bytes() == donor_file.read_bytes(),
            f"host/donor tokenizer file differs: {filename}",
        )
    return {"host": host, "donor": donor, "pair_compatible": True}


def _host_spec(contract: Mapping[str, Any], language: str) -> Dict[str, Any]:
    for host in contract["models"]["hosts"]:
        if host["language"] == language:
            return dict(host)
    raise B3Error(f"unsupported B3 language {language!r}")


def _git_state(root: Path) -> Dict[str, Any]:
    def run(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            return ""
        return (result.stdout or "").strip()

    status = run("status", "--short", "--untracked-files=all")
    diff = run("diff", "--no-ext-diff", "--binary")
    return {
        "commit": run("rev-parse", "HEAD") or None,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
    }


def _environment() -> Dict[str, Any]:
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    for module_name in ("torch", "transformers", "accelerate", "numpy"):
        try:
            module = __import__(module_name)
            payload[module_name] = getattr(module, "__version__", "unknown")
        except Exception:
            payload[module_name] = None
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_run_manifest(
    path: Path,
    *,
    stage: str,
    root: Path,
    contract_file: Path,
    command: Sequence[str],
    inputs: Mapping[str, Any],
) -> Dict[str, Any]:
    payload = {
        "schema_version": 1,
        "run_kind": f"b3_{stage}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "contract": {
            "path": str(contract_file.resolve()),
            "sha256": sha256_file(contract_file),
        },
        "command": list(command),
        "git": _git_state(root),
        "environment": _environment(),
        "inputs": dict(inputs),
    }
    write_json(path, payload)
    return payload


def _load_tokenizer(path: Path, contract: Mapping[str, Any]):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise B3Error("transformers is required to load the B3 tokenizer") from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(path),
            local_files_only=contract["loading"]["local_files_only"],
            trust_remote_code=contract["loading"]["trust_remote_code"],
        )
    except Exception as exc:
        raise B3Error(f"failed to load the exact local tokenizer at {path}: {exc}") from exc
    _require(tokenizer.pad_token_id is not None, "B3 tokenizer has no pad_token_id")
    tokenizer.padding_side = contract["tokenizer"]["padding_side"]
    _require(tokenizer.padding_side == "left", "B3 tokenizer padding policy was not applied")
    return tokenizer


def _load_model(path: Path, contract: Mapping[str, Any]):
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise B3Error("torch and transformers are required to load B3 models") from exc
    if not torch.cuda.is_available():
        raise B3Error("B3 full-model loading requires CUDA; refusing CPU fallback")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(path),
            local_files_only=contract["loading"]["local_files_only"],
            trust_remote_code=contract["loading"]["trust_remote_code"],
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation=contract["loading"]["attn_implementation"],
        )
    except Exception as exc:
        raise B3Error(f"failed to load exact B3 model at {path}: {exc}") from exc
    _require(
        all(parameter.dtype == torch.bfloat16 for parameter in model.parameters()),
        "B3 model loaded with a dtype other than bfloat16",
    )
    _require(type(model).__name__ == contract["architecture"]["class_name"], "unexpected B3 model class")
    return model


def _hardware_report(device: str = "cuda:0", contract: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise B3Error("torch is required for the B3 hardware check") from exc
    _require(torch.cuda.is_available(), "CUDA is unavailable")
    props = torch.cuda.get_device_properties(device)
    total_gib = float(props.total_memory / (1024**3))
    minimum = float((contract or {}).get("compute", {}).get("minimum_gpu_memory_gib", 0))
    _require(
        total_gib >= minimum,
        f"GPU has {total_gib:.2f} GiB, below the B3 minimum of {minimum:.2f} GiB",
    )
    return {
        "device": device,
        "name": props.name,
        "total_memory_gib": total_gib,
        "minimum_memory_gib": minimum,
        "cuda_version": torch.version.cuda,
    }


def _token_ids(tokenizer: Any, text: str) -> List[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded.get("input_ids")
    if not isinstance(ids, list) or not ids:
        raise B3Error(f"readout text produced no tokens: {text!r}")
    return [int(ids[0])]


def _text_list(value: Any, name: str) -> List[str]:
    if isinstance(value, str):
        return [value]
    _require(
        isinstance(value, list) and value and all(isinstance(item, str) for item in value),
        f"{name} must be a nonempty string list",
    )
    return list(value)


def _prompt_from_record(record: Mapping[str, Any], kind: str, tokenizer: Any, max_length: int):
    try:
        from suture.suture_torch import TorchPrompt
    except ImportError as exc:
        raise B3Error("torch and the SUTURE adapter are required for B3 prompts") from exc
    _require(isinstance(record.get("id"), str) and record["id"], "probe record id is missing")
    input_text = record.get("input_text")
    _require(isinstance(input_text, str) and input_text, f"{record.get('id')} input_text is missing")
    encoded = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    kwargs: Dict[str, Any] = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "metadata": {"id": record["id"], "kind": kind},
    }
    if kind == "utility":
        kwargs["utility_token_ids"] = [
            token
            for text in _text_list(record.get("utility_texts"), f"{record['id']}.utility_texts")
            for token in _token_ids(tokenizer, text)
        ]
    elif kind == "risk":
        kwargs["risk_target_token_ids"] = [
            token
            for text in _text_list(record.get("risk_target_texts"), f"{record['id']}.risk_target_texts")
            for token in _token_ids(tokenizer, text)
        ]
        kwargs["risk_donor_token_ids"] = [
            token
            for text in _text_list(record.get("risk_donor_texts"), f"{record['id']}.risk_donor_texts")
            for token in _token_ids(tokenizer, text)
        ]
    else:
        raise B3Error(f"unknown B3 probe kind {kind!r}")
    return TorchPrompt(**kwargs)


def load_probe(path: Path, kind: str, tokenizer: Any, contract: Mapping[str, Any]):
    try:
        from suture.data_manifest import read_manifest
    except ImportError as exc:
        raise B3Error("the SUTURE data manifest module is required") from exc
    try:
        _, records = read_manifest(path)
    except Exception as exc:
        raise B3Error(f"could not load B3 {kind} probe manifest {path}: {exc}") from exc
    minimum = int(contract["data"]["selection"][f"minimum_{kind}_records"])
    _require(len(records) >= minimum, f"{kind} probe has {len(records)} records; minimum is {minimum}")
    ids = [record.get("id") for record in records]
    _require(all(isinstance(item, str) for item in ids), f"{kind} probe contains a missing id")
    _require(len(set(ids)) == len(ids), f"{kind} probe contains duplicate ids")
    return records, [
        _prompt_from_record(record, kind, tokenizer, int(contract["data"]["selection"]["max_length"]))
        for record in records
    ]


def _assert_probe_ids_disjoint(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> None:
    overlap = sorted({str(item["id"]) for item in left} & {str(item["id"]) for item in right})
    _require(not overlap, f"P_util/P_risk overlap by record id: {overlap[:5]}")


def _score_payload(scores: Any, windows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "contract_name": contract["contract_name"],
        "contract_version": contract["contract_version"],
        "n_layers": int(scores.n_layers),
        "utility": scores.utility.tolist(),
        "risk": scores.risk.tolist(),
        "injection_norm": scores.injection_norm.tolist(),
        "utility_sem": None if scores.utility_sem is None else scores.utility_sem.tolist(),
        "risk_sem": None if scores.risk_sem is None else scores.risk_sem.tolist(),
        "n_probe_utility": scores.n_probe_utility,
        "n_probe_risk": scores.n_probe_risk,
        "meta": dict(scores.meta),
        "windows": list(windows),
    }


def _windows(scores: Any) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for start in range(scores.n_layers):
        for end in range(start, scores.n_layers):
            selected = list(range(start, end + 1))
            utility, risk = scores.predict(selected)
            output.append(
                {
                    "start": start,
                    "end": end,
                    "length": end - start + 1,
                    "predicted_utility": utility,
                    "predicted_risk": risk,
                    "epsilon_S": scores.epsilon(selected),
                }
            )
    return output


def _pair_paths(
    language: str,
    host_path: Path,
    donor_path: Path,
    contract: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    host = _host_spec(contract, language)
    donor = dict(contract["models"]["donor"])
    pair = validate_pair_snapshots(
        host_path,
        donor_path,
        host,
        donor,
        contract,
        hash_files=True,
    )
    return pair["host"], pair["donor"]


def run_preflight(
    *,
    language: str,
    host_path: Path,
    donor_path: Path,
    output_root: Path,
    contract_file: Path | None = None,
    load_models: bool = False,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    root = repo_root()
    contract_file = contract_file or contract_path(root)
    contract = load_contract(contract_file)
    host_spec = _host_spec(contract, language)
    donor_spec = dict(contract["models"]["donor"])
    pair_slug = f"{language}_host__en_donor"
    stage_root = output_root / pair_slug / "preflight"
    report: Dict[str, Any] = {
        "schema_version": 1,
        "contract_name": contract["contract_name"],
        "language": language,
        "status": "RUNNING",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        pair = validate_pair_snapshots(
            host_path,
            donor_path,
            host_spec,
            donor_spec,
            contract,
            hash_files=True,
        )
        report["pair"] = pair
        if load_models:
            report["hardware"] = _hardware_report(device, contract)
            host_model = _load_model(host_path, contract)
            donor_model = _load_model(donor_path, contract)
            try:
                from suture.suture_torch import HFResidualAdapter

                adapter = HFResidualAdapter(host_model, donor_model, device=device)
                _require(adapter.n_layers == contract["architecture"]["n_layers"], "adapter layer count mismatch")
                report["adapter_architecture"] = adapter.architecture
                report["selection_merged_model_builds"] = adapter.merged_model_builds
            finally:
                del host_model, donor_model
        report["status"] = "PASS_PREFLIGHT"
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(stage_root / "preflight_report.json", report)
        write_run_manifest(
            stage_root / "run_manifest.json",
            stage="preflight",
            root=root,
            contract_file=contract_file,
            command=sys.argv,
            inputs={"pair": pair, "load_models": load_models, "device": device},
        )
        return report
    except Exception as exc:
        report["status"] = "STOP_SETUP"
        report["error"] = str(exc)
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(stage_root / "preflight_report.json", report)
        raise B3Error(str(exc)) from exc


def run_score(
    *,
    language: str,
    host_path: Path,
    donor_path: Path,
    output_root: Path,
    utility_path: Path,
    risk_path: Path,
    contract_file: Path | None = None,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    root = repo_root()
    contract_file = contract_file or contract_path(root)
    contract = load_contract(contract_file)
    host_spec = _host_spec(contract, language)
    donor_spec = dict(contract["models"]["donor"])
    pair = validate_pair_snapshots(
        host_path,
        donor_path,
        host_spec,
        donor_spec,
        contract,
        hash_files=True,
    )
    hardware = _hardware_report(device, contract)
    tokenizer = _load_tokenizer(host_path, contract)
    utility_records, utility_probe = load_probe(utility_path, "utility", tokenizer, contract)
    risk_records, risk_probe = load_probe(risk_path, "risk", tokenizer, contract)
    _assert_probe_ids_disjoint(utility_records, risk_records)
    host_model = _load_model(host_path, contract)
    donor_model = _load_model(donor_path, contract)
    try:
        from suture.suture_torch import HFResidualAdapter
        from suture.suture_metrics import select_graft

        adapter = HFResidualAdapter(host_model, donor_model, device=device)
        scores = adapter.score_selection(utility_probe, risk_probe)
        windows = _windows(scores)
        _require(
            len(windows) == contract["data"]["windows"]["expected_count"],
            "B3 interval enumeration count mismatch",
        )
        selection = select_graft(
            scores,
            tau=float(contract["data"]["selection"]["tau"]),
            shape=str(contract["data"]["selection"]["shape"]),
        )
        _require(adapter.merged_model_builds == 0, "selection built a grafted model")
    finally:
        del host_model, donor_model

    pair_slug = f"{language}_host__en_donor"
    stage_root = output_root / pair_slug / "score"
    score_payload = _score_payload(scores, windows, contract)
    score_payload["hardware"] = hardware
    score_payload["pair"] = pair
    score_payload["probe_paths"] = {
        "utility": str(utility_path.resolve()),
        "risk": str(risk_path.resolve()),
    }
    selection_payload = {
        "schema_version": 1,
        "contract_name": contract["contract_name"],
        "language": language,
        "selection": selection,
        "published_reference_window": _host_spec(contract, language)["published_swap"]["window_inclusive"],
        "merged_model_builds_during_selection": 0,
        "status": "SCORES_ONLY",
    }
    write_json(stage_root / "selection_scores.json", score_payload)
    write_json(stage_root / "selection.json", selection_payload)
    write_run_manifest(
        stage_root / "run_manifest.json",
        stage="score",
        root=root,
        contract_file=contract_file,
        command=sys.argv,
        inputs={
            "pair": pair,
            "utility_path": str(utility_path.resolve()),
            "risk_path": str(risk_path.resolve()),
            "hardware": hardware,
            "selection": selection,
        },
    )
    return selection_payload


def _load_measurement_records(
    path: Path,
    tokenizer: Any,
    contract: Mapping[str, Any],
    language: str,
) -> List[Dict[str, Any]]:
    try:
        from suture.data_manifest import read_manifest
    except ImportError as exc:
        raise B3Error("the SUTURE data manifest module is required") from exc
    try:
        header, records = read_manifest(path)
    except Exception as exc:
        raise B3Error(f"could not load held-out B3 manifest {path}: {exc}") from exc
    source = contract["data"]["sources"]["held_out_measurement"]
    _require(header.get("_manifest") == f"b3_{language}_MGSM_rev2_test", "held-out manifest name mismatch")
    metadata = header.get("metadata")
    _require(isinstance(metadata, Mapping), "held-out manifest metadata is missing")
    _require(metadata.get("contract_name") == contract["contract_name"], "held-out manifest contract mismatch")
    _require(metadata.get("contract_version") == contract["contract_version"], "held-out manifest version mismatch")
    _require(metadata.get("language") == language, "held-out manifest language mismatch")
    _require(
        metadata.get("objective") == contract["data"]["held_out_measurement"]["objective"],
        "held-out manifest objective mismatch",
    )
    _require(metadata.get("source") == source, "held-out manifest source pin mismatch")
    expected = int(contract["data"]["sources"]["held_out_measurement"]["n_items"])
    _require(len(records) == expected, f"held-out manifest has {len(records)} records; expected {expected}")
    output = []
    max_length = int(contract["data"]["held_out_measurement"]["max_length"])
    for record in records:
        _require(isinstance(record.get("id"), str) and record["id"], "held-out record id is missing")
        _require(
            record["id"].startswith(f"b3:{language}:MGSM_rev2_test:"),
            f"{record['id']} does not belong to the declared held-out language",
        )
        _require(isinstance(record.get("input_text"), str) and record["input_text"], "held-out input is missing")
        _require(record.get("answer_number") is not None, f"{record['id']} answer_number is missing")
        _require(
            str(record.get("dataset")) == source["dataset"]
            and str(record.get("dataset_revision")) == source["revision"]
            and str(record.get("language")) == language,
            f"{record['id']} provenance does not match the held-out contract",
        )
        _require(
            re.fullmatch(r"-?\d+", str(record["answer_number"])) is not None,
            f"{record['id']} answer_number is not an integer",
        )
        encoded = tokenizer(
            record["input_text"],
            return_tensors="pt",
            truncation=False,
            padding=False,
        )
        _require(
            encoded["input_ids"].shape[1] <= max_length,
            f"{record['id']} exceeds held-out max_length {max_length}",
        )
        target = tokenizer(
            str(record["answer_number"]),
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"]
        _require(target.shape[1] > 0, f"{record['id']} answer tokenization is empty")
        output.append(
            {
                "record": record,
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "target_input_ids": target,
            }
        )
    return output


def _candidate_windows(
    contract: Mapping[str, Any],
    selection_payload: Mapping[str, Any],
    language: str,
) -> List[Dict[str, Any]]:
    selection = selection_payload.get("selection")
    _require(isinstance(selection, Mapping), "selection artifact is missing its selection object")
    selected = tuple(sorted(set(int(index) for index in selection.get("graft", []))))
    _require(
        all(0 <= index < int(contract["architecture"]["n_layers"]) for index in selected),
        "selection contains an out-of-range layer",
    )
    if selected:
        _require(
            selected == tuple(range(selected[0], selected[-1] + 1)),
            "selection is not a contiguous interval",
        )
    published_start, published_end = _host_spec(
        contract,
        language,
    )["published_swap"]["window_inclusive"]
    published = tuple(range(int(published_start), int(published_end) + 1))
    return [
        {"name": "host_baseline", "graft": []},
        {"name": "selected", "graft": list(selected)},
        {"name": "published_reference", "graft": list(published)},
    ]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    import hashlib

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_measure(
    *,
    language: str,
    host_path: Path,
    donor_path: Path,
    output_root: Path,
    measurement_path: Path,
    selection_path: Path | None = None,
    contract_file: Path | None = None,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    root = repo_root()
    contract_file = contract_file or contract_path(root)
    contract = load_contract(contract_file)
    host_spec = _host_spec(contract, language)
    donor_spec = dict(contract["models"]["donor"])
    pair = validate_pair_snapshots(
        host_path,
        donor_path,
        host_spec,
        donor_spec,
        contract,
        hash_files=True,
    )
    hardware = _hardware_report(device, contract)
    pair_slug = f"{language}_host__en_donor"
    stage_root = output_root / pair_slug / "measure"
    _require(not stage_root.exists(), f"measurement output already exists: {stage_root}")
    if selection_path is None:
        selection_path = output_root / pair_slug / "score" / "selection.json"
    _require(selection_path.is_file(), f"selection artifact is missing: {selection_path}")
    selection_payload = _json(selection_path)
    _require(
        selection_payload.get("merged_model_builds_during_selection") == 0,
        "selection artifact does not prove zero builds during selection",
    )
    selection_manifest_paths = {
        "P_util": output_root / pair_slug / "data" / "P_util.jsonl",
        "P_risk": output_root / pair_slug / "data" / "P_risk.jsonl",
    }
    try:
        from suture.data_manifest import read_manifest

        _, utility_records = read_manifest(selection_manifest_paths["P_util"])
        _, risk_records = read_manifest(selection_manifest_paths["P_risk"])
    except Exception as exc:
        raise B3Error(f"selection manifests could not be read before measurement: {exc}") from exc
    measurement_records = _load_measurement_records(
        measurement_path,
        _load_tokenizer(host_path, contract),
        contract,
        language,
    )
    from suture.data_manifest import assert_pairwise_disjoint

    assert_pairwise_disjoint(
        {
            "P_util": utility_records,
            "P_risk": risk_records,
            "held_out": [item["record"] for item in measurement_records],
        }
    )
    selection_ids = {str(row["id"]) for row in utility_records + risk_records}
    _require(
        selection_ids.isdisjoint({str(item["record"]["id"]) for item in measurement_records}),
        "held-out records overlap selection records",
    )
    candidates = _candidate_windows(contract, selection_payload, language)

    host_model = _load_model(host_path, contract)
    donor_model = _load_model(donor_path, contract)
    aggregate: List[Dict[str, Any]] = []
    measured_rows: List[Dict[str, Any]] = []
    try:
        from suture.suture_torch import HFResidualAdapter

        adapter = HFResidualAdapter(host_model, donor_model, device=device)
        for candidate in candidates:
            graft = tuple(int(index) for index in candidate["graft"])
            routed = adapter.build_grafted_model(graft)
            normalized_scores = []
            raw_scores = []
            token_counts = []
            for item in measurement_records:
                score = routed.sequence_logprob(
                    item["input_ids"],
                    item["target_input_ids"],
                    item["attention_mask"],
                )
                raw = float(score.detach().cpu().reshape(-1)[0])
                count = int(item["target_input_ids"].shape[1])
                raw_scores.append(raw)
                token_counts.append(count)
                normalized = raw / count
                normalized_scores.append(normalized)
                measured_rows.append(
                    {
                        "candidate": candidate["name"],
                        "graft": list(graft),
                        "record_id": item["record"]["id"],
                        "answer_token_count": count,
                        "answer_logprob": raw,
                        "answer_logprob_per_token": normalized,
                    }
                )
            aggregate.append(
                {
                    "candidate": candidate["name"],
                    "graft": list(graft),
                    "n_records": len(normalized_scores),
                    "mean_answer_logprob": sum(raw_scores) / len(raw_scores),
                    "mean_answer_logprob_per_token": sum(normalized_scores) / len(normalized_scores),
                    "mean_answer_token_count": sum(token_counts) / len(token_counts),
                }
            )
            del routed
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        _require(
            adapter.merged_model_builds == len(candidates),
            "measurement build accounting does not match candidate count",
        )
    except Exception as exc:
        if isinstance(exc, B3Error):
            raise
        raise B3Error(f"held-out measurement failed: {exc}") from exc
    finally:
        del host_model, donor_model

    measurements_path = stage_root / "measurements.jsonl"
    measurements_sha256 = _write_jsonl(measurements_path, measured_rows)
    payload = {
        "schema_version": 1,
        "status": "PASS_MEASURE",
        "contract_name": contract["contract_name"],
        "language": language,
        "objective": contract["data"]["held_out_measurement"]["objective"],
        "pair": pair,
        "hardware": hardware,
        "measurement_manifest": str(measurement_path.resolve()),
        "measurement_manifest_sha256": sha256_file(measurement_path),
        "selection_artifact": str(selection_path.resolve()),
        "selection_artifact_sha256": sha256_file(selection_path),
        "candidates": aggregate,
        "measurements_path": str(measurements_path.resolve()),
        "measurements_sha256": measurements_sha256,
        "selection_merged_model_builds": 0,
        "measurement_merged_model_builds": len(candidates),
        "interpretation": "held-out candidate comparison; not a full 666-window ground-truth sweep",
    }
    write_json(stage_root / "measure.json", payload)
    write_run_manifest(
        stage_root / "run_manifest.json",
        stage="measure",
        root=root,
        contract_file=contract_file,
        command=sys.argv,
        inputs=payload,
    )
    return payload


def prepare_snapshots(
    download_dir: Path,
    contract_file: Path | None = None,
    language: str | None = None,
) -> Dict[str, Any]:
    contract_file = contract_file or contract_path()
    contract = load_contract(contract_file)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise B3Error("huggingface_hub is required for B3 preparation") from exc
    specs = [dict(contract["models"]["donor"])]
    hosts = contract["models"]["hosts"]
    if language is not None:
        hosts = [_host_spec(contract, language)]
    specs.extend(dict(host) for host in hosts)
    paths = {}
    for spec in specs:
        target = download_dir / spec["id"].split("/", 1)[1]
        paths[spec["id"]] = str(
            snapshot_download(
                repo_id=spec["id"],
                revision=spec["revision"],
                local_dir=str(target),
                local_files_only=False,
                repo_type="model",
            )
        )
    return {"contract_name": contract["contract_name"], "snapshots": paths}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("validate-contract")

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--download-dir", type=Path, required=True)
    prepare.add_argument("--language", choices=("fr", "zh"), default=None)

    for command in ("preflight", "score", "measure"):
        item = subparsers.add_parser(command)
        item.add_argument("--language", choices=("fr", "zh"), required=True)
        item.add_argument("--host-path", type=Path, required=True)
        item.add_argument("--donor-path", type=Path, required=True)
        item.add_argument("--output-root", type=Path, default=repo_root() / "results" / "b3" / "lighton_qwen3_8b")
        item.add_argument("--device", default="cuda:0")
    preflight = subparsers.choices["preflight"]
    preflight.add_argument("--load-models", action="store_true")
    score = subparsers.choices["score"]
    score.add_argument("--utility-manifest", type=Path, required=True)
    score.add_argument("--risk-manifest", type=Path, required=True)
    measure = subparsers.choices["measure"]
    measure.add_argument("--measurement-manifest", type=Path, required=True)
    measure.add_argument("--selection-path", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate-contract":
            contract = load_contract(args.contract)
            print(json.dumps({
                "status": "PASS_CONTRACT",
                "contract_name": contract["contract_name"],
                "contract_sha256": sha256_file(args.contract or contract_path()),
            }, sort_keys=True))
            return 0
        if args.command == "prepare":
            print(
                json.dumps(
                    prepare_snapshots(args.download_dir, args.contract, args.language),
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "preflight":
            report = run_preflight(
                language=args.language,
                host_path=args.host_path,
                donor_path=args.donor_path,
                output_root=args.output_root,
                contract_file=args.contract,
                load_models=args.load_models,
                device=args.device,
            )
            print(json.dumps({"status": report["status"], "language": args.language}, sort_keys=True))
            return 0
        if args.command == "score":
            result = run_score(
                language=args.language,
                host_path=args.host_path,
                donor_path=args.donor_path,
                output_root=args.output_root,
                utility_path=args.utility_manifest,
                risk_path=args.risk_manifest,
                contract_file=args.contract,
                device=args.device,
            )
            print(json.dumps(result, sort_keys=True))
            return 0
        if args.command == "measure":
            result = run_measure(
                language=args.language,
                host_path=args.host_path,
                donor_path=args.donor_path,
                output_root=args.output_root,
                measurement_path=args.measurement_manifest,
                selection_path=args.selection_path,
                contract_file=args.contract,
                device=args.device,
            )
            print(json.dumps(result, sort_keys=True))
            return 0
    except B3Error as exc:
        print(f"STOP_SETUP: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unhandled command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
