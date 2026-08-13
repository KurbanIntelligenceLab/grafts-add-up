"""Run the Tier-A P1--P3 plumbing gates and the E1 window sweep.

The CLI is intentionally fail-closed.  It loads a single base model with two
PEFT adapters, uses the shared-base adapter for selection, and only constructs
routed graft evaluations after scores have been written.  E1 evaluates every
contiguous window; it does not silently sample a partial sweep.

The available MGSM development split in ``juletxara/mgsm`` is the eight-example
``train`` split.  It is used as ``MGSM_dev`` by the data builder and the 250
example test split is never read by this script.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2Config,
    Qwen2ForCausalLM,
)

try:
    from paper_iclr.data_manifest import ManifestError, read_manifest
    from paper_iclr.suture_metrics import (
        GraftScores,
        linearisation_diagnostic,
        select_graft,
        spearman,
    )
    from paper_iclr.suture_torch import AdapterError, HFResidualAdapter, TorchPrompt
except ModuleNotFoundError:
    from data_manifest import ManifestError, read_manifest
    from suture_metrics import GraftScores, linearisation_diagnostic, select_graft, spearman
    from suture_torch import AdapterError, HFResidualAdapter, TorchPrompt


MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MODEL_REVISION = "2e1fd397ee46e1388853d2af2c993145b0f1098a"
FAKE_DONOR_NOISE_SCALES = {
    "es": 1e-3,
    "zh": 2e-3,
    "sw": 3e-3,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_records(path: Path) -> List[Dict[str, Any]]:
    _, records = read_manifest(path)
    return records


def _load_scores(path: Path) -> GraftScores:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return GraftScores(
        utility=np.asarray(payload["utility"], dtype=float),
        risk=np.asarray(payload["risk"], dtype=float),
        injection_norm=np.asarray(payload["injection_norm"], dtype=float),
        n_probe_utility=int(payload["n_probe_utility"]),
        n_probe_risk=int(payload["n_probe_risk"]),
        meta=dict(payload.get("meta", {})),
    )


def _adapter_path(root: Path, name: str) -> Path:
    root = root.resolve()
    direct = root / "adapter_config.json"
    nested = root / name / "adapter_config.json"
    if direct.is_file():
        return root
    if nested.is_file():
        return nested.parent
    raise ManifestError(f"PEFT adapter config not found under {root}")


def _token_ids(tokenizer, text: str) -> List[int]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ManifestError(f"tokenizer produced no tokens for readout text {text!r}")
    return [int(token) for token in ids]


def _prompt_from_text(
    tokenizer,
    text: str,
    *,
    utility_tokens: Sequence[int] | int | None = None,
    risk_target_tokens: Sequence[int] | int | None = None,
    risk_donor_tokens: Sequence[int] | int | None = None,
    max_length: int = 512,
) -> TorchPrompt:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    return TorchPrompt(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        utility_token_ids=utility_tokens,
        risk_target_token_ids=risk_target_tokens,
        risk_donor_token_ids=risk_donor_tokens,
        metadata={"text": text},
    )


def _batched_prompt(
    tokenizer,
    texts: Sequence[str],
    *,
    utility_tokens: Sequence[Sequence[int] | int] | Sequence[int] | int | None = None,
    risk_target_tokens: Sequence[Sequence[int] | int] | Sequence[int] | int | None = None,
    risk_donor_tokens: Sequence[Sequence[int] | int] | Sequence[int] | int | None = None,
    max_length: int = 512,
) -> TorchPrompt:
    encoded = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return TorchPrompt(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        utility_token_ids=utility_tokens,
        risk_target_token_ids=risk_target_tokens,
        risk_donor_token_ids=risk_donor_tokens,
        metadata={"n_items": len(texts)},
    )


def _utility_prompts(
    tokenizer,
    records: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    max_length: int,
) -> List[TorchPrompt]:
    prompts = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        texts = [str(row["question"]) + "\nAnswer:" for row in batch]
        tokens = [_token_ids(tokenizer, str(row["answer_number"]))[:1] for row in batch]
        prompts.append(
            _batched_prompt(
                tokenizer,
                texts,
                utility_tokens=tokens,
                max_length=max_length,
            )
        )
    return prompts


def _risk_prompts(
    tokenizer,
    records: Sequence[Mapping[str, Any]],
    *,
    language: str,
    batch_size: int,
    max_length: int,
) -> List[TorchPrompt]:
    language_word = {"es": "español", "zh": "中文", "sw": "Kiswahili"}[language]
    target_tokens = _token_ids(tokenizer, language_word)
    donor_tokens = _token_ids(tokenizer, "English")
    prompts = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        texts = [
            str(row["target_text"]) + f"\nRespond in {language_word}:" for row in batch
        ]
        prompts.append(
            _batched_prompt(
                tokenizer,
                texts,
                risk_target_tokens=target_tokens,
                risk_donor_tokens=donor_tokens,
                max_length=max_length,
            )
        )
    return prompts


def _dev_prompt(tokenizer, records: Sequence[Mapping[str, Any]], max_length: int) -> TorchPrompt:
    texts = [str(row["question"]) + "\nAnswer:" for row in records]
    targets = [_token_ids(tokenizer, str(row["answer_number"]))[:1] for row in records]
    return _batched_prompt(
        tokenizer,
        texts,
        utility_tokens=targets,
        max_length=max_length,
    )


def _load_shared_pair(host_dir: Path, donor_dir: Path, device: str) -> Tuple[Any, Any]:
    host_dir = host_dir.resolve()
    donor_dir = donor_dir.resolve()
    host_adapter_dir = _adapter_path(host_dir, "host")
    donor_adapter_dir = _adapter_path(donor_dir, "donor")
    target_device = torch.device(device)
    dtype = torch.bfloat16 if target_device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(target_device).eval()
    base.config.use_cache = False
    shared = PeftModel.from_pretrained(
        base,
        str(host_adapter_dir),
        adapter_name="host",
        is_trainable=False,
    )
    shared.load_adapter(str(donor_adapter_dir), adapter_name="donor", is_trainable=False)
    shared.set_adapter("host")
    return shared, tokenizer


def _load_fake_pair(
    host_dir: Path,
    device: str,
    noise_scale: float = 1e-3,
) -> Tuple[Any, Any]:
    host_dir = host_dir.resolve()
    host_adapter_dir = _adapter_path(host_dir, "host")
    target_device = torch.device(device)
    dtype = torch.bfloat16 if target_device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(target_device).eval()
    base.config.use_cache = False
    shared = PeftModel.from_pretrained(
        base,
        str(host_adapter_dir),
        adapter_name="host",
        is_trainable=False,
    )
    shared.load_adapter(str(host_adapter_dir), adapter_name="fake", is_trainable=False)
    generator = torch.Generator(device=target_device)
    generator.manual_seed(991)
    with torch.no_grad():
        for name, parameter in shared.named_parameters():
            if "fake" in name:
                parameter.add_(
                    noise_scale
                    * torch.randn(
                        parameter.shape,
                        generator=generator,
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                )
    shared.set_adapter("host")
    return shared, tokenizer


def _random_sets(n_layers: int, count: int, seed: int) -> List[Tuple[int, ...]]:
    generator = np.random.default_rng(seed)
    sets = []
    for _ in range(count):
        size = int(generator.integers(1, n_layers + 1))
        sets.append(tuple(sorted(generator.choice(n_layers, size=size, replace=False).tolist())))
    return sets


def _measure_sets(
    adapter: HFResidualAdapter,
    prompts: Sequence[TorchPrompt],
    grafts: Sequence[Tuple[int, ...]],
) -> Tuple[np.ndarray, np.ndarray]:
    host = adapter.build_grafted_model(())
    host_value = np.mean([host.readout(prompt, "utility") for prompt in prompts])
    measured = []
    for graft in grafts:
        model = adapter.build_grafted_model(graft)
        value = np.mean([model.readout(prompt, "utility") for prompt in prompts])
        measured.append(value - host_value)
    predicted = np.asarray([sum(0.0 for _ in graft) for graft in grafts], dtype=float)
    return predicted, np.asarray(measured, dtype=float)


def _predicted_for_sets(scores: GraftScores, grafts: Sequence[Tuple[int, ...]]) -> np.ndarray:
    return np.asarray([scores.predict(graft)[0] for graft in grafts], dtype=float)


def _p3_float64_fixture() -> Dict[str, Any]:
    """Run P3 on a compact Qwen-shaped float64 plumbing fixture.

    The deployed Tier-A checkpoint is bfloat16, for which finite differences
    are quantised at the hidden-state scale.  P3 validates the adapter's
    trajectory/adjoint implementation independently on the same Qwen2 block
    API in float64; the real checkpoint remains the scientific P2/E1 fixture.
    """

    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=16,
        bos_token_id=1,
        eos_token_id=2,
    )
    torch.manual_seed(104)
    host = Qwen2ForCausalLM(config).double()
    donor = copy.deepcopy(host)
    prompt = TorchPrompt(
        input_ids=torch.tensor([[1, 5, 7, 9]], dtype=torch.long),
        utility_token_ids=torch.tensor([10]),
        risk_target_token_ids=torch.tensor([11]),
        risk_donor_token_ids=torch.tensor([12]),
    )
    fixture = HFResidualAdapter(host, donor, device="cpu")
    result = fixture.adjoint_check(
        prompt,
        layer=1,
        readout="utility",
        delta_scale=1e-3,
        seed=104,
    )
    result.update(
        {
            "fixture": "tiny_qwen2_float64",
            "source_checkpoint": "synthetic_plumbing_fixture",
            "real_checkpoint_bf16_finite_difference": "not_used_for_gate",
        }
    )
    return result


def run_pilots(
    *,
    data_dir: Path,
    host_dir: Path,
    donor_dir: Path,
    output_dir: Path,
    language: str,
    device: str,
    prompt_limit: int,
    prompt_batch_size: int,
    max_length: int,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    p_util = _load_records(data_dir / "P_util.jsonl")[:prompt_limit]
    p_risk = _load_records(data_dir / "P_risk.jsonl")[:prompt_limit]
    fake_noise_scale = FAKE_DONOR_NOISE_SCALES[language]
    fake_shared, tokenizer = _load_fake_pair(
        host_dir,
        device,
        noise_scale=fake_noise_scale,
    )
    fake_adapter = HFResidualAdapter(
        fake_shared,
        fake_shared,
        device=device,
        host_adapter_name="host",
        donor_adapter_name="fake",
    )
    utility_prompts = _utility_prompts(
        tokenizer,
        p_util,
        batch_size=prompt_batch_size,
        max_length=max_length,
    )
    risk_prompts = _risk_prompts(
        tokenizer,
        p_risk,
        language=language,
        batch_size=prompt_batch_size,
        max_length=max_length,
    )
    all_results: Dict[str, Any] = {
        "language": language,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "n_utility_records": len(p_util),
        "n_risk_records": len(p_risk),
        "fake_donor_noise_scale": fake_noise_scale,
        "adapter": fake_adapter.architecture,
    }

    def one_pilot(label: str, model_adapter: HFResidualAdapter, grafts: Sequence[Tuple[int, ...]]) -> Dict[str, Any]:
        start = time.perf_counter()
        scores = model_adapter.score_selection(utility_prompts, risk_prompts)
        predicted = _predicted_for_sets(scores, grafts)
        _, measured = _measure_sets(model_adapter, utility_prompts, grafts)
        diagnostics = linearisation_diagnostic(predicted, measured)
        diagnostics.update(
            {
                "label": label,
                "grafts": [list(graft) for graft in grafts],
                "injection_norm_sum_mean": float(np.mean([scores.epsilon(graft) for graft in grafts])),
                "merged_models_built_after_selection": model_adapter.merged_model_builds,
                "wall_clock_seconds": time.perf_counter() - start,
            }
        )
        scores.to_json(str(output_dir / f"{label}_scores.json"))
        return diagnostics

    fake_grafts = _random_sets(fake_adapter.n_layers, 10, seed=101)
    all_results["P1"] = one_pilot("P1_fake_donor", fake_adapter, fake_grafts)
    del fake_adapter, fake_shared
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    shared, tokenizer = _load_shared_pair(host_dir, donor_dir, device)
    adapter = HFResidualAdapter(
        shared,
        shared,
        device=device,
        host_adapter_name="host",
        donor_adapter_name="donor",
    )
    utility_prompts = _utility_prompts(
        tokenizer,
        p_util,
        batch_size=prompt_batch_size,
        max_length=max_length,
    )
    risk_prompts = _risk_prompts(
        tokenizer,
        p_risk,
        language=language,
        batch_size=prompt_batch_size,
        max_length=max_length,
    )
    all_results["adapter"] = adapter.architecture
    real_grafts = _random_sets(adapter.n_layers, 10, seed=102)
    p2 = one_pilot("P2_real_donor", adapter, real_grafts)
    p2["jacobian_gap_proxy"] = adapter.jacobian_gap_proxy(
        utility_prompts[0],
        layer=adapter.n_layers // 2,
        seed=103,
    )
    all_results["P2"] = p2
    all_results["P3"] = _p3_float64_fixture()
    with (output_dir / "pilots.json").open("w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2, sort_keys=True)
    return all_results


def _contiguous_windows(n_layers: int) -> List[Tuple[int, int]]:
    return [(start, end) for start in range(n_layers) for end in range(start, n_layers)]


def run_e1(
    *,
    data_dir: Path,
    host_dir: Path,
    donor_dir: Path,
    output_dir: Path,
    language: str,
    device: str,
    score_batch_size: int,
    max_length: int,
    tau: float,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    utility_records = _load_records(data_dir / "P_util.jsonl")
    risk_records = _load_records(data_dir / "P_risk.jsonl")
    dev_records = _load_records(data_dir / "MGSM_dev.jsonl")
    shared, tokenizer = _load_shared_pair(host_dir, donor_dir, device)
    adapter = HFResidualAdapter(
        shared,
        shared,
        device=device,
        host_adapter_name="host",
        donor_adapter_name="donor",
    )
    utility_prompts = _utility_prompts(
        tokenizer,
        utility_records,
        batch_size=score_batch_size,
        max_length=max_length,
    )
    risk_prompts = _risk_prompts(
        tokenizer,
        risk_records,
        language=language,
        batch_size=score_batch_size,
        max_length=max_length,
    )
    dev_prompt = _dev_prompt(tokenizer, dev_records, max_length=max_length)
    dev_utility_prompts = _utility_prompts(
        tokenizer,
        dev_records,
        batch_size=score_batch_size,
        max_length=max_length,
    )
    scores_path = output_dir / "e1_scores.json"
    if scores_path.is_file():
        scores = _load_scores(scores_path)
        selection_score_seconds = 0.0
        selection_resumed = True
    else:
        score_start = time.perf_counter()
        scores = adapter.score_selection(utility_prompts, risk_prompts)
        selection_score_seconds = time.perf_counter() - score_start
        scores.to_json(str(scores_path))
        selection_resumed = False
    selected = select_graft(scores, tau=tau, shape="interval")
    print(
        f"[E1:{language}] selection scores ready "
        f"({'resumed' if selection_resumed else 'computed'})",
        flush=True,
    )

    aligned_scores_path = output_dir / "e1_dev_aligned_scores.json"
    if aligned_scores_path.is_file():
        dev_scores = _load_scores(aligned_scores_path)
        alignment_score_seconds = 0.0
        alignment_resumed = True
    else:
        alignment_start = time.perf_counter()
        dev_scores = adapter.score_selection(dev_utility_prompts, risk_prompts)
        alignment_score_seconds = time.perf_counter() - alignment_start
        dev_scores.to_json(str(aligned_scores_path))
        alignment_resumed = False
    print(
        f"[E1:{language}] dev-aligned scores ready "
        f"({'resumed' if alignment_resumed else 'computed'})",
        flush=True,
    )

    windows = _contiguous_windows(adapter.n_layers)
    predicted = _predicted_for_sets(
        dev_scores,
        [tuple(range(i, j + 1)) for i, j in windows],
    )
    baseline_path = output_dir / "e1_baseline.json"
    if baseline_path.is_file():
        with baseline_path.open("r", encoding="utf-8") as handle:
            baseline_payload = json.load(handle)
        baseline_value = float(baseline_payload["utility"])
        baseline_accuracy = float(baseline_payload["teacher_forced_accuracy"])
        baseline_resumed = True
    else:
        baseline_model = adapter.build_grafted_model(())
        baseline_value = baseline_model.readout(dev_prompt, "utility")
        baseline_accuracy = baseline_model.next_token_accuracy(dev_prompt)
        with baseline_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "utility": float(baseline_value),
                    "teacher_forced_accuracy": float(baseline_accuracy),
                    "n_dev_items": len(dev_records),
                },
                handle,
                indent=2,
                sort_keys=True,
            )
        baseline_resumed = False
    rows_by_key: Dict[Tuple[int, int], Dict[str, Any]] = {}
    sweep_path = output_dir / "e1_window_sweep.jsonl"
    if sweep_path.is_file():
        with sweep_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    rows_by_key[(int(row["start"]), int(row["end"]))] = row
    resumed_windows = len(rows_by_key)
    sweep_start = time.perf_counter()
    with sweep_path.open("a", encoding="utf-8") as handle:
        for (start, end), prediction in zip(windows, predicted):
            if (start, end) in rows_by_key:
                continue
            graft = tuple(range(start, end + 1))
            model = adapter.build_grafted_model(graft)
            measured_value = model.readout(dev_prompt, "utility") - baseline_value
            accuracy = model.next_token_accuracy(dev_prompt)
            row = {
                "start": start,
                "end": end,
                "graft": list(graft),
                "predicted_utility": float(prediction),
                "measured_utility": float(measured_value),
                "teacher_forced_accuracy": float(accuracy),
                "n_dev_items": len(dev_records),
            }
            rows_by_key[(start, end)] = row
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            completed = len(rows_by_key)
            if completed == 1 or completed % 25 == 0 or completed == len(windows):
                print(
                    f"[E1:{language}] measured windows {completed}/{len(windows)}",
                    flush=True,
                )
    rows = [rows_by_key[(start, end)] for start, end in windows]
    sweep_seconds = time.perf_counter() - sweep_start
    measured = np.asarray([row["measured_utility"] for row in rows], dtype=float)
    predicted_arr = np.asarray([row["predicted_utility"] for row in rows], dtype=float)
    rank = spearman(predicted_arr, measured)
    best_pred_index = int(np.argmax(predicted_arr))
    best_true_index = int(np.argmax(measured))
    selected_window = tuple(selected["graft"])
    selected_index = next(
        index
        for index, (start, end) in enumerate(windows)
        if tuple(range(start, end + 1)) == selected_window
    )
    achievable_gain = max(float(measured.max()), 0.0)
    selected_gain = float(measured[selected_index])
    regret = float(measured[best_true_index] - selected_gain)
    top_k = 5
    top_pred = set(np.argsort(-predicted_arr)[:top_k].tolist())
    top_true = set(np.argsort(-measured)[:top_k].tolist())
    report = {
        "language": language,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "n_layers": adapter.n_layers,
        "n_windows": len(windows),
        "n_utility_records": len(utility_records),
        "n_risk_records": len(risk_records),
        "n_dev_records": len(dev_records),
        "n_dev_scoring_records": len(dev_records),
        "tau": tau,
        "suture_selected": selected,
        "sweep_best_window": list(windows[best_true_index]),
        "predicted_best_window": list(windows[best_pred_index]),
        "spearman": float(rank),
        "pearson": float(np.corrcoef(predicted_arr, measured)[0, 1]),
        "selected_teacher_forced_accuracy": float(rows[selected_index]["teacher_forced_accuracy"]),
        "baseline_teacher_forced_accuracy": float(baseline_accuracy),
        "accuracy_regret_not_estimated": True,
        "utility_regret": regret,
        "utility_regret_fraction_of_achievable_gain": (
            float(regret / achievable_gain) if achievable_gain > 0 else None
        ),
        "top5_overlap": len(top_pred & top_true),
        "score_seconds": selection_score_seconds + alignment_score_seconds,
        "selection_score_seconds": selection_score_seconds,
        "dev_alignment_score_seconds": alignment_score_seconds,
        "sweep_seconds": sweep_seconds,
        "merged_models_built": 1 + len(rows),
        "selection_merged_models_built": 0,
        "sweep_is_exhaustive": True,
        "selection_scores_resumed": selection_resumed,
        "dev_alignment_scores_resumed": alignment_resumed,
        "baseline_resumed": baseline_resumed,
        "sweep_windows_resumed": resumed_windows,
        "sweep_windows_completed": len(rows),
        "selection_score_source": "P_util_and_P_risk",
        "ranking_prediction_source": "MGSM_dev_utility_postselection_alignment",
        "measured_objective": "teacher-forced utility log-probability change on MGSM_dev",
        "note": (
            "Selection is frozen from P_util/P_risk. Correlation and regret are "
            "computed from a post-selection utility score on the same MGSM_dev "
            "items as the exhaustive measured sweep. Teacher-forced next-token "
            "accuracy is diagnostic; free-generation exact-match is not substituted."
        ),
        "gate": (
            "PASS_E1"
            if rank >= 0.5 and (achievable_gain <= 0 or regret / max(achievable_gain, 1e-12) < 0.5)
            else "STOP_E2_REGIME_INVESTIGATION"
        ),
    }
    with (output_dir / "e1_gate.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    pilots = sub.add_parser("pilots")
    e1 = sub.add_parser("e1")
    for command in (pilots, e1):
        command.add_argument("--data-dir", required=True)
        command.add_argument("--host-dir", required=True)
        command.add_argument("--donor-dir", required=True)
        command.add_argument("--output-dir", required=True)
        command.add_argument("--language", required=True)
        command.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        command.add_argument("--max-length", type=int, default=512)
    pilots.add_argument("--prompt-limit", type=int, default=8)
    pilots.add_argument("--prompt-batch-size", type=int, default=1)
    e1.add_argument("--score-batch-size", type=int, default=1)
    e1.add_argument("--tau", type=float, default=0.05)
    args = parser.parse_args()
    if args.command == "pilots":
        result = run_pilots(
            data_dir=Path(args.data_dir),
            host_dir=Path(args.host_dir),
            donor_dir=Path(args.donor_dir),
            output_dir=Path(args.output_dir),
            language=args.language,
            device=args.device,
            prompt_limit=args.prompt_limit,
            prompt_batch_size=args.prompt_batch_size,
            max_length=args.max_length,
        )
    else:
        result = run_e1(
            data_dir=Path(args.data_dir),
            host_dir=Path(args.host_dir),
            donor_dir=Path(args.donor_dir),
            output_dir=Path(args.output_dir),
            language=args.language,
            device=args.device,
            score_batch_size=args.score_batch_size,
            max_length=args.max_length,
            tau=args.tau,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
