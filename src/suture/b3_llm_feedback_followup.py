"""Isolated 8B follow-up for the ICLR early-feedback requests.

This module reads the frozen B3 score and measurement artifacts, then writes
only to a separate ``llm_feedback_followup`` result root.  It measures the
ten-window language calibration, compares parameter movement and first-order
weight attribution, and evaluates any newly selected windows on the existing
250-record answer and 76-record language holdouts.

The follow-up is teacher-forced only.  It does not run free-generation
language identification, does not construct a conformal certificate, and
does not modify any canonical B3 artifact.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from suture.b3_language_fidelity import _load_fidelity_prompts
from suture.b3_lighton import (
    B3Error,
    _hardware_report,
    _host_spec,
    _load_measurement_records,
    _load_model,
    _load_tokenizer,
    load_probe,
    _require,
    contract_path,
    load_contract,
    repo_root,
    sha256_file,
    validate_pair_snapshots,
    write_json,
)
from suture.data_manifest import assert_pairwise_disjoint, read_manifest
from suture.suture_metrics import GraftScores, best_interval


CALIBRATION_SEED = 20260919
BOOTSTRAP_SEED = 20260919
BOOTSTRAP_RESAMPLES = 20_000
DEFAULT_OUTPUT_ROOT = repo_root() / "results" / "b3" / "lighton_qwen3_8b" / "llm_feedback_followup"


def _json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise B3Error(f"could not read JSON file {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def _graft_key(graft: Sequence[int]) -> Tuple[int, ...]:
    return tuple(sorted(set(int(index) for index in graft)))


def _graft_list(graft: Sequence[int]) -> List[int]:
    return list(_graft_key(graft))


def _all_windows(n_layers: int) -> List[Tuple[int, ...]]:
    return [
        tuple(range(start, end + 1))
        for start in range(n_layers)
        for end in range(start, n_layers)
    ]


def _calibration_windows(
    n_layers: int,
    *,
    n_windows: int,
    seed: int,
) -> List[Tuple[int, ...]]:
    windows = _all_windows(n_layers)
    _require(0 < n_windows <= len(windows), "invalid calibration window count")
    rng = np.random.default_rng(seed)
    indices = sorted(int(index) for index in rng.choice(len(windows), size=n_windows, replace=False))
    return [windows[index] for index in indices]


def _load_scores(
    score_path: Path,
    *,
    expected_layers: int,
) -> GraftScores:
    payload = _json(score_path)
    utility = np.asarray(payload.get("utility"), dtype=float)
    risk = np.asarray(payload.get("risk"), dtype=float)
    injection_norm = np.asarray(payload.get("injection_norm"), dtype=float)
    _require(
        utility.shape == (expected_layers,)
        and risk.shape == (expected_layers,)
        and injection_norm.shape == (expected_layers,),
        "frozen score artifact has an unexpected layer shape",
    )
    return GraftScores(
        utility=utility,
        risk=risk,
        injection_norm=injection_norm,
        n_probe_utility=int(payload["n_probe_utility"]),
        n_probe_risk=int(payload["n_probe_risk"]),
        utility_sem=None,
        risk_sem=None,
        meta={"source": "frozen_b3_selection_scores"},
    )


def _selection_graft(selection_path: Path) -> Tuple[int, ...]:
    payload = _json(selection_path)
    selection = payload.get("selection")
    _require(isinstance(selection, Mapping), "frozen selection artifact lacks selection")
    graft = _graft_key(selection.get("graft", []))
    if graft:
        _require(graft == tuple(range(graft[0], graft[-1] + 1)), "frozen selection is not contiguous")
    return graft


def _published_graft(contract: Mapping[str, Any], language: str) -> Tuple[int, ...]:
    start, end = _host_spec(contract, language)["published_swap"]["window_inclusive"]
    return tuple(range(int(start), int(end) + 1))


def _fit_origin(predicted: Sequence[float], measured: Sequence[float]) -> Dict[str, float]:
    x = np.asarray(predicted, dtype=float)
    y = np.asarray(measured, dtype=float)
    _require(x.shape == y.shape and x.size > 0, "calibration arrays are empty or mismatched")
    denominator = float(np.dot(x, x))
    _require(denominator > 1e-18, "language calibration predictions have zero norm")
    slope = float(np.dot(x, y) / denominator)
    residual = y - slope * x
    return {
        "slope_b": slope,
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "mae": float(np.mean(np.abs(residual))),
        "n_windows": int(x.size),
    }


def _bootstrap_difference(
    difference: Sequence[float],
    *,
    seed: int,
) -> Dict[str, Any]:
    values = np.asarray(list(difference), dtype=float)
    _require(values.size > 1, "paired bootstrap needs at least two records")
    rng = np.random.default_rng(seed)
    bootstrap = rng.choice(
        values,
        size=(BOOTSTRAP_RESAMPLES, values.size),
        replace=True,
    ).mean(axis=1)
    return {
        "n_records": int(values.size),
        "mean": float(values.mean()),
        "ci95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "fraction_positive": float(np.mean(values > 0)),
    }


def _environment() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    for module_name in ("torch", "transformers", "numpy"):
        try:
            module = __import__(module_name)
            result[module_name] = getattr(module, "__version__", "unknown")
        except Exception:
            result[module_name] = None
    return result


def _empty_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _readout_values(
    adapter: Any,
    graft: Sequence[int],
    prompts: Sequence[Any],
    readout: str,
) -> np.ndarray:
    routed = adapter.build_grafted_model(graft)
    try:
        return np.asarray(
            [routed.readout(prompt, readout) for prompt in prompts],
            dtype=float,
        )
    finally:
        del routed
        _empty_cache()


def _calibrate(
    adapter: Any,
    scores: GraftScores,
    utility_prompts: Sequence[Any],
    risk_prompts: Sequence[Any],
    windows: Sequence[Tuple[int, ...]],
    *,
    private_path: Path,
) -> Dict[str, Any]:
    """Measure calibration windows and fit the language slope through zero."""

    private_path.parent.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    host = adapter.build_grafted_model(())
    try:
        host_utility = np.asarray(
            [host.readout(prompt, "utility") for prompt in utility_prompts],
            dtype=float,
        )
        host_risk = np.asarray(
            [host.readout(prompt, "risk") for prompt in risk_prompts],
            dtype=float,
        )
    finally:
        del host
        _empty_cache()

    predicted_utility: List[float] = []
    predicted_risk: List[float] = []
    measured_utility: List[float] = []
    measured_risk: List[float] = []
    for index, graft in enumerate(windows):
        routed = adapter.build_grafted_model(graft)
        try:
            utility_values = np.asarray(
                [routed.readout(prompt, "utility") for prompt in utility_prompts],
                dtype=float,
            )
            risk_values = np.asarray(
                [routed.readout(prompt, "risk") for prompt in risk_prompts],
                dtype=float,
            )
        finally:
            del routed
            _empty_cache()
        predicted_u, predicted_r = scores.predict(graft)
        measured_u = float(np.mean(utility_values - host_utility))
        measured_r = float(np.mean(risk_values - host_risk))
        predicted_utility.append(predicted_u)
        predicted_risk.append(predicted_r)
        measured_utility.append(measured_u)
        measured_risk.append(measured_r)
        records.append(
            {
                "index": index,
                "graft": _graft_list(graft),
                "predicted_utility": predicted_u,
                "predicted_risk": predicted_r,
                "measured_utility": measured_u,
                "measured_risk": measured_r,
            }
        )
        print(
            f"calibration window {index + 1}/{len(windows)} "
            f"{list(graft)} risk={measured_r:+.5f}",
            flush=True,
        )

    with private_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in records:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    fit = _fit_origin(predicted_risk, measured_risk)
    return {
        "seed": CALIBRATION_SEED,
        "n_windows": len(windows),
        "windows": records,
        "fit": fit,
        "measurement": {
            "utility_probe_count": len(utility_prompts),
            "risk_probe_count": len(risk_prompts),
            "private_per_record": True,
        },
    }


def _parameter_movement(adapter: Any) -> np.ndarray:
    movement = np.zeros(adapter.n_layers, dtype=float)
    for layer_index, (host_layer, donor_layer) in enumerate(
        zip(adapter.host_layers, adapter.donor_layers)
    ):
        donor_parameters = dict(donor_layer.named_parameters())
        squared = 0.0
        for name, host_parameter in host_layer.named_parameters():
            donor_parameter = donor_parameters.get(name)
            _require(donor_parameter is not None, f"donor layer is missing parameter {name}")
            difference = (
                donor_parameter.detach().float() - host_parameter.detach().float()
            )
            squared += float(torch_sum_square(difference))
        movement[layer_index] = float(np.sqrt(squared))
    return movement


def torch_sum_square(value: Any) -> float:
    """Return a scalar without retaining a large temporary tensor."""

    return float(value.square().sum().detach().cpu())


def _weight_attribution(
    adapter: Any,
    prompts: Sequence[Any],
    *,
    readout: str,
) -> np.ndarray:
    """Compute signed per-layer grad(parameter) dot (donor - host)."""

    import torch

    layer_parameters: List[List[Any]] = [
        list(layer.parameters()) for layer in adapter.host_layers
    ]
    flat_parameters = [parameter for layer in layer_parameters for parameter in layer]
    _require(flat_parameters, "host layers contain no parameters for attribution")
    previous_flags = [parameter.requires_grad for parameter in flat_parameters]
    for parameter in flat_parameters:
        parameter.requires_grad_(True)
    donor_parameters = [
        dict(layer.named_parameters()) for layer in adapter.donor_layers
    ]
    host_parameters = [
        dict(layer.named_parameters()) for layer in adapter.host_layers
    ]
    totals = np.zeros(adapter.n_layers, dtype=float)
    try:
        for prompt_index, prompt in enumerate(prompts):
            with adapter._active_adapter(adapter.host_model, adapter.host_adapter_name):
                with torch.enable_grad():
                    outputs = adapter.host_model(**prompt.model_inputs(adapter.device))
                    logits = getattr(outputs, "logits", None)
                    _require(logits is not None, "host attribution forward returned no logits")
                    value = adapter._readout(logits, prompt, readout)
                    gradients = torch.autograd.grad(
                        value,
                        flat_parameters,
                        retain_graph=False,
                        allow_unused=True,
                    )
            offset = 0
            for layer_index in range(adapter.n_layers):
                layer_total = 0.0
                for parameter in layer_parameters[layer_index]:
                    gradient = gradients[offset]
                    offset += 1
                    if gradient is None:
                        continue
                    name = next(
                        name
                        for name, candidate in host_parameters[layer_index].items()
                        if candidate is parameter
                    )
                    donor_parameter = donor_parameters[layer_index][name]
                    delta = donor_parameter.detach().float() - parameter.detach().float()
                    layer_total += float((gradient.detach().float() * delta).sum().cpu())
                totals[layer_index] += layer_total
            del outputs, logits, value, gradients
            if (prompt_index + 1) % 25 == 0 or prompt_index + 1 == len(prompts):
                print(
                    f"weight attribution {readout} "
                    f"{prompt_index + 1}/{len(prompts)}",
                    flush=True,
                )
                _empty_cache()
    finally:
        for parameter, flag in zip(flat_parameters, previous_flags):
            parameter.requires_grad_(flag)
        _empty_cache()
    return totals / float(len(prompts))


def _best_window(values: Sequence[float]) -> Tuple[int, ...] | None:
    array = np.asarray(values, dtype=float)
    _require(array.ndim == 1 and array.size > 0, "cannot select from empty layer values")
    _, argument = best_interval(array, np.zeros_like(array), 0.0)
    if argument is None:
        return None
    return tuple(range(argument[0], argument[1] + 1))


def _best_constrained(
    values: Sequence[float],
    risk: Sequence[float],
    tau: float,
) -> Tuple[int, ...] | None:
    _, argument = best_interval(
        np.asarray(values, dtype=float),
        np.asarray(risk, dtype=float),
        tau,
    )
    if argument is None:
        return None
    return tuple(range(argument[0], argument[1] + 1))


def _selection_record(
    graft: Tuple[int, ...] | None,
    *,
    basis: str,
    utility: Sequence[float],
    risk: Sequence[float],
    tau: float,
) -> Dict[str, Any]:
    if graft is None:
        return {
            "graft": [],
            "basis": basis,
            "predicted_utility": None,
            "predicted_risk": None,
            "feasible": False,
        }
    predicted_utility = float(np.asarray(utility)[list(graft)].sum())
    predicted_risk = float(np.asarray(risk)[list(graft)].sum())
    return {
        "graft": _graft_list(graft),
        "basis": basis,
        "predicted_utility": predicted_utility,
        "predicted_risk": predicted_risk,
        "feasible": bool(predicted_risk >= -tau),
    }


def _load_frozen_measurement(
    path: Path,
) -> Dict[Tuple[int, ...], Dict[str, Any]]:
    payload = _json(path)
    result: Dict[Tuple[int, ...], Dict[str, Any]] = {}
    for row in payload.get("candidates", []):
        graft = _graft_key(row.get("graft", []))
        result[graft] = {
            "answer_mean": float(row["mean_answer_logprob_per_token"]),
            "answer_n": int(row["n_records"]),
            "source": "frozen_b3_measurement",
        }
    return result


def _frozen_language_means(
    path: Path,
    language: str,
    selected_graft: Tuple[int, ...],
    published_graft: Tuple[int, ...],
) -> Dict[Tuple[int, ...], Dict[str, Any]]:
    payload = _json(path)
    values = payload["languages"][language]["means"]
    n_records = int(payload["analysis"]["n_records_per_language"])
    return {
        (): {
            "language_mean": float(values["host"]),
            "language_n": n_records,
            "source": "frozen_b3_language_measurement",
        },
        selected_graft: {
            "language_mean": float(values["selected"]),
            "language_n": n_records,
            "source": "frozen_b3_language_measurement",
        },
        published_graft: {
            "language_mean": float(values["published"]),
            "language_n": n_records,
            "source": "frozen_b3_language_measurement",
        },
    }


def _measure_new_candidates(
    adapter: Any,
    candidates: Sequence[Tuple[int, ...]],
    answer_records: Sequence[Mapping[str, Any]],
    language_records: Sequence[Mapping[str, Any]],
    language_prompts: Sequence[Any],
    *,
    private_path: Path,
) -> Dict[Tuple[int, ...], Dict[str, Any]]:
    """Measure only grafts absent from the frozen host/selected/published set."""

    import torch

    private_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    aggregates: Dict[Tuple[int, ...], Dict[str, Any]] = {}
    for candidate_index, graft in enumerate(candidates):
        routed = adapter.build_grafted_model(graft)
        answer_values: List[float] = []
        language_values: List[float] = []
        try:
            for item in answer_records:
                raw = float(
                    routed.sequence_logprob(
                        item["input_ids"],
                        item["target_input_ids"],
                        item["attention_mask"],
                    )
                    .detach()
                    .cpu()
                    .reshape(-1)[0]
                )
                token_count = int(item["target_input_ids"].shape[1])
                answer_values.append(raw / token_count)
                rows.append(
                    {
                        "candidate_index": candidate_index,
                        "graft": _graft_list(graft),
                        "record_id": item["record"]["id"],
                        "metric": "answer_logprob_per_token",
                        "value": raw / token_count,
                    }
                )
            for record, prompt in zip(language_records, language_prompts):
                value = float(routed.readout(prompt, "risk"))
                language_values.append(value)
                rows.append(
                    {
                        "candidate_index": candidate_index,
                        "graft": _graft_list(graft),
                        "record_id": record["id"],
                        "metric": "language_readout",
                        "value": value,
                    }
                )
        finally:
            del routed
            _empty_cache()
        aggregates[graft] = {
            "answer_mean": float(np.mean(answer_values)),
            "answer_values": np.asarray(answer_values, dtype=float),
            "answer_n": len(answer_values),
            "language_mean": float(np.mean(language_values)),
            "language_values": np.asarray(language_values, dtype=float),
            "language_n": len(language_values),
            "source": "new_followup_measurement",
        }
        print(
            f"held-out candidate {candidate_index + 1}/{len(candidates)} "
            f"{list(graft)}",
            flush=True,
        )
    with private_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    del torch
    return aggregates


def _public_metric(
    graft: Tuple[int, ...],
    measured: Mapping[Tuple[int, ...], Mapping[str, Any]],
    host: Mapping[str, Any],
    *,
    answer: bool,
) -> Dict[str, Any]:
    values = measured[graft]
    metric = "answer" if answer else "language"
    mean_key = f"{metric}_mean"
    n_key = f"{metric}_n"
    output = {
        "mean": float(values[mean_key]),
        "n_records": int(values[n_key]),
        "source": values.get("source"),
    }
    if answer:
        output["unit"] = "nats per target token"
    else:
        output["unit"] = "target-language-minus-English first-token log-probability"
    if "answer_values" in values and answer:
        output["delta_vs_host"] = _bootstrap_difference(
            values["answer_values"] - host["answer_values"],
            seed=BOOTSTRAP_SEED,
        )
    elif "language_values" in values and not answer:
        output["delta_vs_host"] = _bootstrap_difference(
            values["language_values"] - host["language_values"],
            seed=BOOTSTRAP_SEED + 1,
        )
    return output


def _run_language(
    *,
    language: str,
    host_path: Path,
    donor_path: Path,
    output_root: Path,
    contract_file: Path,
    calibration_count: int,
    calibration_limit: int | None,
    attribution_limit: int | None,
    utility_path: Path | None,
    risk_path: Path | None,
    measurement_path: Path | None,
    language_path: Path | None,
    selection_path: Path | None,
    score_path: Path | None,
    device: str,
) -> Dict[str, Any]:
    contract = load_contract(contract_file)
    pair_slug = f"{language}_host__en_donor"
    canonical_root = repo_root() / contract["results"]["root"]
    output_pair = output_root / pair_slug
    _require(not output_pair.exists(), f"follow-up output already exists: {output_pair}")
    utility_path = utility_path or canonical_root / pair_slug / "data" / "P_util.jsonl"
    risk_path = risk_path or canonical_root / pair_slug / "data" / "P_risk.jsonl"
    measurement_path = measurement_path or canonical_root / pair_slug / "data" / "MGSM_rev2_test.jsonl"
    language_path = language_path or canonical_root / pair_slug / "fidelity" / "data" / "language_holdout.jsonl"
    selection_path = selection_path or canonical_root / pair_slug / "score" / "selection.json"
    score_path = score_path or canonical_root / pair_slug / "score" / "selection_scores.json"
    frozen_measurement_path = canonical_root / pair_slug / "measure" / "measure.json"
    frozen_language_path = canonical_root / "posthoc_language_fidelity.json"

    pair = validate_pair_snapshots(
        host_path,
        donor_path,
        _host_spec(contract, language),
        dict(contract["models"]["donor"]),
        contract,
        hash_files=True,
    )
    tokenizer = _load_tokenizer(host_path, contract)
    utility_records, utility_prompts = load_probe(
        utility_path,
        "utility",
        tokenizer,
        contract,
    )
    risk_records, risk_prompts = load_probe(
        risk_path,
        "risk",
        tokenizer,
        contract,
    )
    _require(
        len(utility_records) == int(_load_scores(score_path, expected_layers=36).n_probe_utility),
        "utility manifest count differs from frozen score provenance",
    )
    _require(
        len(risk_records) == int(_load_scores(score_path, expected_layers=36).n_probe_risk),
        "risk manifest count differs from frozen score provenance",
    )
    answer_records = _load_measurement_records(
        measurement_path,
        tokenizer,
        contract,
        language,
    )
    language_header, language_records, language_prompts = _load_fidelity_prompts(
        language_path,
        tokenizer,
        contract,
    )
    assert_pairwise_disjoint(
        {
            "P_util": utility_records,
            "P_risk": risk_records,
            "MGSM_rev2_test": [item["record"] for item in answer_records],
            "language_holdout": language_records,
        }
    )
    scores = _load_scores(score_path, expected_layers=36)
    original_graft = _selection_graft(selection_path)
    published_graft = _published_graft(contract, language)
    tau = float(contract["data"]["selection"]["tau"])
    calibration_windows = _calibration_windows(
        scores.n_layers,
        n_windows=calibration_count,
        seed=CALIBRATION_SEED,
    )

    host_model = _load_model(host_path, contract)
    donor_model = _load_model(donor_path, contract)
    adapter = None
    try:
        from suture.suture_torch import HFResidualAdapter

        adapter = HFResidualAdapter(host_model, donor_model, device=device)
        calibration = _calibrate(
            adapter,
            scores,
            utility_prompts[:calibration_limit] if calibration_limit else utility_prompts,
            risk_prompts[:calibration_limit] if calibration_limit else risk_prompts,
            calibration_windows,
            private_path=output_pair / "private" / "calibration.jsonl",
        )
        slope = float(calibration["fit"]["slope_b"])
        calibrated_risk = scores.risk * slope
        calibrated_graft = _best_constrained(scores.utility, calibrated_risk, tau)
        movement = _parameter_movement(adapter)
        movement_unconstrained = _best_window(movement)
        movement_constrained = _best_constrained(movement, calibrated_risk, tau)
        attribution_prompts = (
            utility_prompts[:attribution_limit]
            if attribution_limit
            else utility_prompts
        )
        weight_utility = _weight_attribution(
            adapter,
            attribution_prompts,
            readout="utility",
        )
        weight_risk = _weight_attribution(
            adapter,
            risk_prompts[:attribution_limit] if attribution_limit else risk_prompts,
            readout="risk",
        )
        weight_unconstrained = _best_window(weight_utility)
        weight_constrained = _best_constrained(weight_utility, weight_risk * slope, tau)

        roles: Dict[str, Dict[str, Any]] = {
            "suture_uncalibrated": _selection_record(
                original_graft,
                basis="frozen uncalibrated B3 selection",
                utility=scores.utility,
                risk=scores.risk,
                tau=tau,
            ),
            "suture_calibrated": _selection_record(
                calibrated_graft,
                basis="frozen SUTURE utility with fitted language slope",
                utility=scores.utility,
                risk=calibrated_risk,
                tau=tau,
            ),
            "movement_unconstrained": _selection_record(
                movement_unconstrained,
                basis="sum of per-layer host-donor L2 movement",
                utility=movement,
                risk=calibrated_risk,
                tau=tau,
            ),
            "movement_calibrated_feasible": _selection_record(
                movement_constrained,
                basis="movement utility with fitted SUTURE language feasibility",
                utility=movement,
                risk=calibrated_risk,
                tau=tau,
            ),
            "weight_attribution_unconstrained": _selection_record(
                weight_unconstrained,
                basis="first-order host weight attribution on utility probes",
                utility=weight_utility,
                risk=calibrated_risk,
                tau=tau,
            ),
            "weight_attribution_calibrated_feasible": _selection_record(
                weight_constrained,
                basis="weight attribution utility with first-order language attribution",
                utility=weight_utility,
                risk=weight_risk * slope,
                tau=tau,
            ),
        }

        frozen_measurement = _load_frozen_measurement(frozen_measurement_path)
        frozen_language = _frozen_language_means(
            frozen_language_path,
            language,
            original_graft,
            published_graft,
        )
        frozen_metrics: Dict[Tuple[int, ...], Dict[str, Any]] = {}
        for graft, values in frozen_measurement.items():
            frozen_metrics.setdefault(graft, {}).update(values)
        for graft, values in frozen_language.items():
            frozen_metrics.setdefault(graft, {}).update(values)

        role_grafts = {
            _graft_key(value["graft"])
            for value in roles.values()
            if value.get("graft") is not None
        }
        new_grafts = sorted(
            graft
            for graft in role_grafts
            if graft not in ((), original_graft, published_graft)
        )
        measured_new = _measure_new_candidates(
            adapter,
            [()] + new_grafts if new_grafts else [],
            answer_records,
            language_records,
            language_prompts,
            private_path=output_pair / "private" / "heldout_measurements.jsonl",
        )
        measured = dict(frozen_metrics)
        host_metrics = measured_new.get(())
        if host_metrics is not None:
            measured[()] = host_metrics
        measured.update({graft: values for graft, values in measured_new.items() if graft})
        _require(() in measured, "host metrics are missing")

        public_roles: Dict[str, Any] = {}
        for name, role in roles.items():
            graft = _graft_key(role.get("graft", []))
            _require(graft in measured, f"metrics missing for role {name}: {graft}")
            public_roles[name] = {
                **role,
                "graft": _graft_list(graft),
                "held_out_answer": _public_metric(
                    graft,
                    measured,
                    measured[()],
                    answer=True,
                ),
                "held_out_language": _public_metric(
                    graft,
                    measured,
                    measured[()],
                    answer=False,
                ),
            }

        public_pair = {
            "schema_version": 1,
            "status": "PASS_LLM_FEEDBACK_FOLLOWUP",
            "contract_name": contract["contract_name"],
            "contract_version": contract["contract_version"],
            "language": language,
            "calibration": {
                "seed": calibration["seed"],
                "n_windows": calibration["n_windows"],
                "fit": calibration["fit"],
                "windows": [
                    {
                        key: value
                        for key, value in row.items()
                        if key != "index"
                    }
                    for row in calibration["windows"]
                ],
                "probe_counts": calibration["measurement"],
            },
            "scores": {
                "score_artifact_sha256": sha256_file(score_path),
                "n_layers": scores.n_layers,
                "n_probe_utility": scores.n_probe_utility,
                "n_probe_risk": scores.n_probe_risk,
                "fitted_risk_slope_applied_to": "frozen per-layer risk scores",
            },
            "baselines": {
                "parameter_movement": movement.tolist(),
                "weight_attribution_utility": weight_utility.tolist(),
                "weight_attribution_risk": weight_risk.tolist(),
                "attribution_probe_counts": {
                    "utility": len(attribution_prompts),
                    "risk": len(risk_prompts[:attribution_limit] if attribution_limit else risk_prompts),
                },
            },
            "roles": public_roles,
            "new_grafts_measured": [_graft_list(graft) for graft in new_grafts],
            "frozen_grafts_not_remeasured": {
                "original_selected": _graft_list(original_graft),
                "published_reference": _graft_list(published_graft),
            },
            "generation_lid": {
                "status": "NOT_RUN",
                "reason": "BF16 cached and uncached greedy decoding remains outside this follow-up.",
            },
            "private_records": {
                "status": "RETAINED_AUTHOR_ONLY",
                "calibration": "private/calibration.jsonl",
                "held_out": "private/heldout_measurements.jsonl",
            },
            "hardware": _hardware_report(device, contract),
            "language_manifest": {
                "sha256": sha256_file(language_path),
                "n_records": len(language_records),
                "header_schema_version": language_header.get("schema_version"),
            },
            "pair": {
                "host": {
                    "id": pair["host"]["id"],
                    "revision": pair["host"]["revision"],
                    "snapshot_sha256": pair["host"]["snapshot_sha256"],
                },
                "donor": {
                    "id": pair["donor"]["id"],
                    "revision": pair["donor"]["revision"],
                    "snapshot_sha256": pair["donor"]["snapshot_sha256"],
                },
            },
            "canonical_artifacts_modified": False,
        }
        output_pair.mkdir(parents=True, exist_ok=True)
        write_json(output_pair / "summary.json", public_pair)
        write_json(
            output_pair / "run_manifest.json",
            {
                "schema_version": 1,
                "run_kind": "b3_llm_feedback_followup",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "contract_sha256": sha256_file(contract_file),
                "score_sha256": sha256_file(score_path),
                "selection_sha256": sha256_file(selection_path),
                "utility_manifest_sha256": sha256_file(utility_path),
                "risk_manifest_sha256": sha256_file(risk_path),
                "measurement_manifest_sha256": sha256_file(measurement_path),
                "language_manifest_sha256": sha256_file(language_path),
                "environment": _environment(),
                "hardware": public_pair["hardware"],
                "language": language,
                "calibration_seed": CALIBRATION_SEED,
                "canonical_artifacts_modified": False,
            },
        )
        return public_pair
    finally:
        if adapter is not None:
            del adapter
        del host_model, donor_model
        gc.collect()
        _empty_cache()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=("fr", "zh"), required=True)
    parser.add_argument("--host-path", type=Path, required=True)
    parser.add_argument("--donor-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--contract", type=Path, default=None)
    parser.add_argument("--utility-manifest", type=Path, default=None)
    parser.add_argument("--risk-manifest", type=Path, default=None)
    parser.add_argument("--measurement-manifest", type=Path, default=None)
    parser.add_argument("--language-manifest", type=Path, default=None)
    parser.add_argument("--selection-path", type=Path, default=None)
    parser.add_argument("--score-path", type=Path, default=None)
    parser.add_argument("--calibration-count", type=int, default=10)
    parser.add_argument(
        "--calibration-limit",
        type=int,
        default=None,
        help="optional prefix of P_risk/P_util for a documented smoke run",
    )
    parser.add_argument(
        "--attribution-limit",
        type=int,
        default=None,
        help="optional prefix of P_util/P_risk for a documented smoke run",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    contract_file = args.contract or contract_path()
    try:
        report = _run_language(
            language=args.language,
            host_path=args.host_path,
            donor_path=args.donor_path,
            output_root=args.output_root,
            contract_file=contract_file,
            calibration_count=args.calibration_count,
            calibration_limit=args.calibration_limit,
            attribution_limit=args.attribution_limit,
            utility_path=args.utility_manifest,
            risk_path=args.risk_manifest,
            measurement_path=args.measurement_manifest,
            language_path=args.language_manifest,
            selection_path=args.selection_path,
            score_path=args.score_path,
            device=args.device,
        )
    except B3Error as exc:
        print(f"STOP_LLM_FEEDBACK_FOLLOWUP: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
