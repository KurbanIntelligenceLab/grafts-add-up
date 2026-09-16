"""Contract-v2 exhaustive E1 with a compute-bounded decode policy.

Ranking uses one decode seed on all contiguous windows.  After selection,
five decode seeds are used only for host, selected, dev-oracle, and the
top-five predicted windows.  Belebele is scored on host/selected/oracle only.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import pyarrow  # noqa: F401  — must load before torch on Windows
import fasttext
import numpy as np
import torch

try:
    from suture.data_manifest import ManifestError
    from suture.suture_metrics import select_graft, spearman
    from suture.suture_torch import AdapterError, HFResidualAdapter
    from suture.tier_a_config import (
        DECODE_TEMPERATURE,
        DECODE_TOP_P,
        GENERATION_BATCH_SIZE,
        MAX_NEW_TOKENS,
        MIN_SPEARMAN,
        MODEL_ID,
        MODEL_REVISION,
        RANKING_DECODE_SEEDS,
        SCORE_BATCH_SIZE,
        STABILITY_DECODE_SEEDS,
        TAU,
        refuse_v1_write,
        refuse_frozen_write,
    )
    from suture.tier_a_gate import (
        EXPECTED_LID_LABEL,
        _contiguous_windows,
        _evaluate_generation,
        _full_sequence_score,
        _load_records,
        _load_scores,
        _load_shared_pair,
        _predicted_for_sets,
        _risk_prompts,
        _sha256,
        _utility_prompts,
        _write_immutable_json,
    )
    from suture.tier_a_readiness import _belebele_eval
except ModuleNotFoundError:
    from data_manifest import ManifestError
    from suture_metrics import select_graft, spearman
    from suture_torch import AdapterError, HFResidualAdapter
    from tier_a_config import (
        DECODE_TEMPERATURE,
        DECODE_TOP_P,
        GENERATION_BATCH_SIZE,
        MAX_NEW_TOKENS,
        MIN_SPEARMAN,
        MODEL_ID,
        MODEL_REVISION,
        RANKING_DECODE_SEEDS,
        SCORE_BATCH_SIZE,
        STABILITY_DECODE_SEEDS,
        TAU,
        refuse_v1_write,
        refuse_frozen_write,
    )
    from tier_a_gate import (
        EXPECTED_LID_LABEL,
        _contiguous_windows,
        _evaluate_generation,
        _full_sequence_score,
        _load_records,
        _load_scores,
        _load_shared_pair,
        _predicted_for_sets,
        _risk_prompts,
        _sha256,
        _utility_prompts,
        _write_immutable_json,
    )
    from tier_a_readiness import _belebele_eval


def load_v2_e1_sweep(path: Path) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Load Qwen3 E1 sweep rows.

    Row field protocol_version 3 is a schema tag so these rows cannot mix with
    Coder E1 rows (protocol_version 2). It is not a third contract.
    """
    rows_by_key: Dict[Tuple[int, int], Dict[str, Any]] = {}
    if not path.is_file():
        return rows_by_key
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("protocol_version", -1)) != 3:
                raise ManifestError("contract-v2 E1 sweep contains a row from another protocol")
            key = (int(row["start"]), int(row["end"]))
            if key in rows_by_key:
                raise ManifestError(f"duplicate sweep window {key} in {path}")
            rows_by_key[key] = row
    return rows_by_key


def _evaluate_dev(
    model: Any,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    language: str,
    lid_model: Any,
    device: str,
    decode_seeds: Sequence[int],
    max_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    batch_size: int,
) -> Dict[str, Any]:
    generation = _evaluate_generation(
        model,
        tokenizer,
        records,
        language=language,
        lid_model=lid_model,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        decode_seeds=decode_seeds,
        sample=True,
        temperature=temperature,
        top_p=top_p,
    )
    sequence = _full_sequence_score(
        model,
        tokenizer,
        records,
        device=device,
        max_length=max_length,
    )
    return {"generation": generation, "teacher_forced_full_sequence": sequence}


def run_e1_v2(
    *,
    data_dir: Path,
    host_dir: Path,
    donor_dir: Path,
    output_dir: Path,
    language: str,
    device: str,
    lid_model_path: Path,
    tau: float = TAU,
    window_limit: int | None = None,
    probe_limit: int | None = None,
) -> Dict[str, Any]:
    refuse_v1_write(output_dir)
    refuse_frozen_write(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "e1_gate.json"
    if report_path.is_file():
        return json.loads(report_path.read_text(encoding="utf-8"))
    if language not in EXPECTED_LID_LABEL:
        raise ManifestError(f"unsupported E1 language: {language}")
    if not lid_model_path.is_file():
        raise ManifestError(f"fastText language-ID model not found: {lid_model_path}")

    paths = {
        "P_util": data_dir / "P_util.jsonl",
        "P_risk": data_dir / "P_risk.jsonl",
        "MGSM_dev": data_dir / "MGSM_dev.jsonl",
        "C": data_dir / "C.jsonl",
        "Belebele": data_dir / "Belebele.jsonl",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise ManifestError(f"missing contract-v2 E1 manifest {name}: {path}")
    config = {
        "protocol_version": 3,
        "language": language,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "ranking_decode_seeds": list(RANKING_DECODE_SEEDS),
        "stability_decode_seeds": list(STABILITY_DECODE_SEEDS),
        "decoding": {
            "enable_thinking": False,
            "temperature": DECODE_TEMPERATURE,
            "top_p": DECODE_TOP_P,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
        "input_hashes": {name: _sha256(path) for name, path in paths.items()},
    }
    _write_immutable_json(output_dir / "e1_config.json", config)

    lid_model = fasttext.load_model(str(lid_model_path))
    utility_records = _load_records(paths["P_util"])
    risk_records = _load_records(paths["P_risk"])
    dev_records = _load_records(paths["MGSM_dev"])
    certificate_records = _load_records(paths["C"])
    belebele_records = _load_records(paths["Belebele"])
    if probe_limit:
        utility_records = utility_records[:probe_limit]
        risk_records = risk_records[:probe_limit]

    shared, tokenizer = _load_shared_pair(host_dir, donor_dir, device)
    adapter = HFResidualAdapter(
        shared,
        shared,
        device=device,
        host_adapter_name="host",
        donor_adapter_name="donor",
    )
    utility_prompts = _utility_prompts(
        tokenizer, utility_records, batch_size=SCORE_BATCH_SIZE, max_length=128
    )
    risk_prompts = _risk_prompts(
        tokenizer,
        risk_records,
        language=language,
        batch_size=SCORE_BATCH_SIZE,
        max_length=128,
    )
    scores_path = output_dir / "e1_selection_scores.json"
    if scores_path.is_file():
        scores = _load_scores(scores_path)
        selection_resumed = True
        selection_score_seconds = 0.0
    else:
        start = time.perf_counter()
        scores = adapter.score_selection(utility_prompts, risk_prompts)
        selection_score_seconds = time.perf_counter() - start
        scores.to_json(str(scores_path))
        selection_resumed = False
    if adapter.merged_model_builds != 0:
        raise AdapterError("contract-v2 selection built merged models")
    selected = select_graft(scores, tau=tau, shape="interval")

    all_windows = _contiguous_windows(adapter.n_layers)
    windows = list(all_windows[:window_limit] if window_limit else all_windows)
    predicted = _predicted_for_sets(
        scores, [tuple(range(start, end + 1)) for start, end in windows]
    )
    sweep_path = output_dir / "e1_window_sweep.jsonl"
    rows_by_key: Dict[Tuple[int, int], Dict[str, Any]] = {}
    if sweep_path.is_file():
        rows_by_key = load_v2_e1_sweep(sweep_path)
    sweep_start = time.perf_counter()
    with sweep_path.open("a", encoding="utf-8") as handle:
        for (start, end), prediction in zip(windows, predicted):
            if (start, end) in rows_by_key:
                continue
            graft = tuple(range(start, end + 1))
            model = adapter.build_grafted_model(graft)
            evaluated = _evaluate_dev(
                model,
                tokenizer,
                dev_records,
                language=language,
                lid_model=lid_model,
                device=device,
                decode_seeds=RANKING_DECODE_SEEDS,
                max_length=128,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=DECODE_TEMPERATURE,
                top_p=DECODE_TOP_P,
                batch_size=GENERATION_BATCH_SIZE,
            )
            row = {
                "protocol_version": 3,
                "start": start,
                "end": end,
                "graft": list(graft),
                "predicted_utility": float(prediction),
                "measured_exact_match": float(evaluated["generation"]["exact_match"]),
                "measured_drift_rate": float(evaluated["generation"]["drift_rate"]),
                "measured_teacher_forced_sequence_logprob": float(
                    evaluated["teacher_forced_full_sequence"]["mean_logprob"]
                ),
                "generation": evaluated["generation"],
                "teacher_forced_full_sequence": evaluated["teacher_forced_full_sequence"],
                "ranking_decode_seeds": list(RANKING_DECODE_SEEDS),
                "n_dev_items": len(dev_records),
            }
            rows_by_key[(start, end)] = row
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
    rows = [rows_by_key[key] for key in windows]
    measured = np.asarray([row["measured_exact_match"] for row in rows], dtype=float)
    predicted_arr = np.asarray([row["predicted_utility"] for row in rows], dtype=float)
    rank = float(spearman(predicted_arr, measured)) if len(rows) > 1 else 0.0
    pearson = (
        float(np.corrcoef(predicted_arr, measured)[0, 1])
        if np.std(predicted_arr) > 0 and np.std(measured) > 0
        else 0.0
    )
    selected_graft = tuple(selected["graft"])
    selected_index = next(
        i for i, (start, end) in enumerate(windows) if tuple(range(start, end + 1)) == selected_graft
    )
    best_true_index = int(np.argmax(measured))
    top_pred = list(np.argsort(-predicted_arr)[:5])
    extra = top_pred.tolist() if hasattr(top_pred, "tolist") else list(top_pred)
    stability_indices = sorted({selected_index, best_true_index, *extra})
    # host baseline is the empty graft; measure with stability seeds.
    baseline_model = adapter.build_grafted_model(())
    baseline = _evaluate_dev(
        baseline_model,
        tokenizer,
        dev_records,
        language=language,
        lid_model=lid_model,
        device=device,
        decode_seeds=STABILITY_DECODE_SEEDS,
        max_length=128,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=DECODE_TEMPERATURE,
        top_p=DECODE_TOP_P,
        batch_size=GENERATION_BATCH_SIZE,
    )
    baseline["belebele"] = _belebele_eval(
        baseline_model, tokenizer, belebele_records, device=torch.device(device), max_length=128
    )
    certificate_inputs = [
        {"source_id": str(row["source_id"]), "question": str(row["target_text"]), "answer_number": "0"}
        for row in certificate_records
    ]
    baseline["certificate"] = _evaluate_generation(
        baseline_model,
        tokenizer,
        certificate_inputs,
        language=language,
        lid_model=lid_model,
        device=device,
        batch_size=GENERATION_BATCH_SIZE,
        max_length=128,
        max_new_tokens=MAX_NEW_TOKENS,
        decode_seeds=STABILITY_DECODE_SEEDS,
        sample=True,
        temperature=DECODE_TEMPERATURE,
        top_p=DECODE_TOP_P,
    )
    _write_immutable_json(output_dir / "e1_baseline.json", {"protocol_version": 3, "graft": [], **baseline})

    def _stability_for(index: int) -> Dict[str, Any]:
        start, end = windows[index]
        graft = tuple(range(start, end + 1))
        model = adapter.build_grafted_model(graft)
        evaluated = _evaluate_dev(
            model,
            tokenizer,
            dev_records,
            language=language,
            lid_model=lid_model,
            device=device,
            decode_seeds=STABILITY_DECODE_SEEDS,
            max_length=128,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=DECODE_TEMPERATURE,
            top_p=DECODE_TOP_P,
            batch_size=GENERATION_BATCH_SIZE,
        )
        belebele = _belebele_eval(
            model, tokenizer, belebele_records, device=torch.device(device), max_length=128
        )
        return {
            "start": start,
            "end": end,
            "graft": list(graft),
            **evaluated,
            "belebele": belebele,
        }

    selected_stability = _stability_for(selected_index)
    oracle_stability = _stability_for(best_true_index)
    top5_stability = [_stability_for(index) for index in top_pred]
    selected_model = adapter.build_grafted_model(selected_graft)
    selected_certificate = _evaluate_generation(
        selected_model,
        tokenizer,
        certificate_inputs,
        language=language,
        lid_model=lid_model,
        device=device,
        batch_size=GENERATION_BATCH_SIZE,
        max_length=128,
        max_new_tokens=MAX_NEW_TOKENS,
        decode_seeds=STABILITY_DECODE_SEEDS,
        sample=True,
        temperature=DECODE_TEMPERATURE,
        top_p=DECODE_TOP_P,
    )
    host_em = float(baseline["generation"]["exact_match"])
    selected_em = float(selected_stability["generation"]["exact_match"])
    best_em = float(oracle_stability["generation"]["exact_match"])
    regret = float(best_em - selected_em)
    achievable = max(best_em - host_em, 0.0)
    selected_drift = float(selected_stability["generation"]["drift_rate"])
    report = {
        "protocol_version": 3,
        "language": language,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "n_layers": adapter.n_layers,
        "n_windows": len(windows),
        "n_dev_records": len(dev_records),
        "tau": tau,
        "suture_selected": selected,
        "sweep_best_window": list(windows[best_true_index]),
        "spearman": rank,
        "pearson": pearson,
        "selected_exact_match": selected_em,
        "baseline_exact_match": host_em,
        "sweep_best_exact_match": best_em,
        "accuracy_regret": regret,
        "accuracy_regret_fraction_of_achievable_gain": (
            float(regret / achievable) if achievable > 0 else None
        ),
        "top5_overlap": len(set(top_pred) & set(np.argsort(-measured)[:5].tolist())),
        "selected_drift_rate": selected_drift,
        "baseline_drift_rate": float(baseline["generation"]["drift_rate"]),
        "selected_belebele": selected_stability["belebele"],
        "baseline_belebele": baseline["belebele"],
        "oracle_belebele": oracle_stability["belebele"],
        "selected_certificate": selected_certificate,
        "baseline_certificate": baseline["certificate"],
        "stability": {
            "selected": selected_stability,
            "dev_oracle": oracle_stability,
            "top5_predicted": top5_stability,
        },
        "selection_score_seconds": selection_score_seconds,
        "sweep_seconds": time.perf_counter() - sweep_start,
        "merged_models_built": adapter.merged_model_builds,
        "selection_merged_models_built": 0,
        "sweep_is_exhaustive": window_limit is None,
        "selection_scores_resumed": selection_resumed,
        "selection_score_source": "P_util_and_P_risk",
        "measured_objective": "free_generation_exact_match_on_MGSM_dev",
        "gate": (
            "SMOKE_NOT_CANONICAL"
            if window_limit is not None or probe_limit is not None
            else (
                "PASS_E1"
                if rank >= MIN_SPEARMAN
                and (achievable <= 0 or regret / max(achievable, 1e-12) < 0.5)
                and selected_drift <= 0.5
                else "STOP_E2_REGIME_INVESTIGATION"
            )
        ),
    }
    _write_immutable_json(report_path, report)
    return report


def run_holdout_v2(
    *,
    data_dir: Path,
    host_dir: Path,
    donor_dir: Path,
    e1_dir: Path,
    output_dir: Path,
    language: str,
    device: str,
    lid_model_path: Path,
) -> Dict[str, Any]:
    refuse_v1_write(output_dir)
    refuse_frozen_write(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "mgsm_test_once.json"
    if out.is_file():
        return json.loads(out.read_text(encoding="utf-8"))
    report = json.loads((e1_dir / "e1_gate.json").read_text(encoding="utf-8"))
    test_records = _load_records(data_dir / "MGSM_test.jsonl")
    if len(test_records) != 186:
        raise ManifestError(f"MGSM_test must have 186 items, found {len(test_records)}")
    lid_model = fasttext.load_model(str(lid_model_path))
    shared, tokenizer = _load_shared_pair(host_dir, donor_dir, device)
    adapter = HFResidualAdapter(
        shared, shared, device=device, host_adapter_name="host", donor_adapter_name="donor"
    )
    selected = tuple(report["suture_selected"]["graft"])
    oracle = tuple(range(int(report["sweep_best_window"][0]), int(report["sweep_best_window"][1]) + 1))
    payload = {"protocol_version": 3, "n_test": len(test_records), "windows": {}}
    for name, graft in (("host", ()), ("selected", selected), ("dev_oracle", oracle)):
        model = adapter.build_grafted_model(graft)
        payload["windows"][name] = _evaluate_dev(
            model,
            tokenizer,
            test_records,
            language=language,
            lid_model=lid_model,
            device=device,
            decode_seeds=RANKING_DECODE_SEEDS,
            max_length=128,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=DECODE_TEMPERATURE,
            top_p=DECODE_TOP_P,
            batch_size=GENERATION_BATCH_SIZE,
        )
        payload["windows"][name]["graft"] = list(graft)
    _write_immutable_json(out, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", choices=("e1", "holdout"), default="e1")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--host-dir", required=True, type=Path)
    parser.add_argument("--donor-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--language", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lid-model", type=Path, default=Path("models/lid.176.ftz"))
    parser.add_argument("--e1-dir", type=Path)
    args = parser.parse_args()
    if args.command == "holdout":
        payload = run_holdout_v2(
            data_dir=args.data_dir,
            host_dir=args.host_dir,
            donor_dir=args.donor_dir,
            e1_dir=args.e1_dir or args.data_dir.parent / "e1",
            output_dir=args.output_dir,
            language=args.language,
            device=args.device,
            lid_model_path=args.lid_model,
        )
        print(json.dumps({"n_test": payload.get("n_test"), "windows": list((payload.get("windows") or {}))}, indent=2))
        return 0
    report = run_e1_v2(
        data_dir=args.data_dir,
        host_dir=args.host_dir,
        donor_dir=args.donor_dir,
        output_dir=args.output_dir,
        language=args.language,
        device=args.device,
        lid_model_path=args.lid_model,
    )
    print(json.dumps({"gate": report.get("gate"), "spearman": report.get("spearman")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
