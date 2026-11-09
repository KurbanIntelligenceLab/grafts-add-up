"""Recompute B3 language-slope selection from frozen JSON artifacts.

This is a deterministic, CPU-only post-hoc diagnostic. It does not load models,
make API calls, modify frozen artifacts, or constitute a new experiment.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TAU = 0.05
LANGUAGES = ("fr", "zh")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def windows(utility: list[float], risk: list[float]) -> list[dict[str, Any]]:
    if len(utility) != 36 or len(risk) != 36:
        raise ValueError("expected 36 per-layer utility and risk scores")
    result = []
    for start in range(36):
        task = language = 0.0
        for end in range(start, 36):
            task += float(utility[end])
            language += float(risk[end])
            result.append(
                {
                    "start": start,
                    "end": end,
                    "predicted_utility": task,
                    "predicted_risk": language,
                }
            )
    if len(result) != 666:
        raise AssertionError(f"expected 666 windows, got {len(result)}")
    return result


def choose(candidates: list[dict[str, Any]], slope: float | None) -> dict[str, Any]:
    eligible = candidates
    if slope is not None:
        eligible = [row for row in candidates if slope * row["predicted_risk"] >= -TAU]
    host = {"start": None, "end": None, "predicted_utility": 0.0, "predicted_risk": 0.0}
    if not eligible:
        return {"status": "host_no_feasible_window", **host}
    best = max(eligible, key=lambda row: row["predicted_utility"])
    if best["predicted_utility"] <= 0:
        return {"status": "host_nonpositive_utility", **host}
    return {"status": "selected", **best}


def calibration_summary(
    rows: list[dict[str, Any]], slope: float
) -> dict[str, Any]:
    enriched = []
    for row in rows:
        predicted = float(row["predicted_risk"])
        measured = float(row["measured_risk"])
        original_feasible = predicted >= -TAU
        calibrated_feasible = slope * predicted >= -TAU
        measured_feasible = measured >= -TAU
        enriched.append(
            {
                "start": int(row["graft"][0]),
                "end": int(row["graft"][-1]),
                "predicted_risk_b1": predicted,
                "predicted_risk_fitted": slope * predicted,
                "measured_risk": measured,
                "predicted_sign_b1": _sign(predicted),
                "predicted_sign_fitted": _sign(slope * predicted),
                "measured_sign": _sign(measured),
                "measured_feasible": measured_feasible,
                "predicted_feasible_b1": original_feasible,
                "predicted_feasible_fitted": calibrated_feasible,
                "false_feasible_b1": original_feasible and not measured_feasible,
                "false_feasible_fitted": calibrated_feasible and not measured_feasible,
            }
        )
    return {
        "n_windows": len(rows),
        "fit_and_evaluation_rows_identical": True,
        "interpretation": "in-sample diagnostic only; these ten windows fit b and do not independently validate it",
        "measured_violations": sum(not row["measured_feasible"] for row in enriched),
        "false_feasible_b1": sum(row["false_feasible_b1"] for row in enriched),
        "false_feasible_fitted": sum(row["false_feasible_fitted"] for row in enriched),
        "sign_agreement_b1_vs_measured": sum(
            row["predicted_sign_b1"] == row["measured_sign"] for row in enriched
        ),
        "sign_agreement_fitted_vs_measured": sum(
            row["predicted_sign_fitted"] == row["measured_sign"] for row in enriched
        ),
        "rows": enriched,
    }


def _sign(value: float) -> str:
    return "positive" if value > 0 else "negative" if value < 0 else "zero"


def run() -> dict[str, Any]:
    output: dict[str, Any] = {
        "analysis": "post-hoc offline diagnostic of frozen B3 artifacts",
        "new_model_experiment": False,
        "computation": "Python standard library only; CPU; no model/API/GPU calls",
        "feasibility_rule": "b * A_risk(S) >= -tau",
        "tau": TAU,
        "window_count_expected": 666,
        "languages": {},
    }
    for language in LANGUAGES:
        pair = f"{language}_host__en_donor"
        score_path = ROOT / "results" / "b3" / "lighton_qwen3_8b" / pair / "score" / "selection_scores.json"
        selection_path = ROOT / "results" / "b3" / "lighton_qwen3_8b" / pair / "score" / "selection.json"
        summary_path = ROOT / "results" / "b3" / "lighton_qwen3_8b" / "llm_feedback_followup" / pair / "summary.json"
        scores = read_json(score_path)
        selection = read_json(selection_path)
        summary = read_json(summary_path)
        fitted_b = float(summary["calibration"]["fit"]["slope_b"])
        calibration_rows = summary["calibration"]["windows"]
        x = [float(row["predicted_risk"]) for row in calibration_rows]
        y = [float(row["measured_risk"]) for row in calibration_rows]
        denom = sum(value * value for value in x)
        refit_b = sum(a * b for a, b in zip(x, y)) / denom
        residuals = [b - refit_b * a for a, b in zip(x, y)]
        rmse = (sum(value * value for value in residuals) / len(residuals)) ** 0.5
        mae = sum(abs(value) for value in residuals) / len(residuals)
        if abs(refit_b - fitted_b) > 1e-12:
            raise AssertionError(f"{language}: refitted b {refit_b} != saved {fitted_b}")
        candidate_windows = windows(scores["utility"], scores["risk"])
        saved_windows = {
            (int(row["start"]), int(row["end"])): row for row in scores["windows"]
        }
        if len(saved_windows) != 666:
            raise AssertionError(f"{language}: frozen score artifact does not contain 666 unique windows")
        for row in candidate_windows:
            saved = saved_windows[(row["start"], row["end"])]
            for field in ("predicted_utility", "predicted_risk"):
                if abs(row[field] - float(saved[field])) > 1e-9:
                    raise AssertionError(
                        f"{language}: recomputed {field} differs for window "
                        f"[{row['start']},{row['end']}]"
                    )
        b1 = choose(candidate_windows, 1.0)
        calibrated = choose(candidate_windows, refit_b)
        unconstrained = choose(candidate_windows, None)
        stored = selection["selection"]
        expected_graft = tuple(range(int(stored["graft"][0]), int(stored["graft"][-1]) + 1))
        if expected_graft != tuple(int(index) for index in stored["graft"]):
            raise AssertionError(f"{language}: frozen selection is not a contiguous window")
        if (b1["start"], b1["end"]) != (stored["graft"][0], stored["graft"][-1]):
            raise AssertionError(f"{language}: recomputed b=1 winner differs from frozen selection")
        scaled_selected_risk = refit_b * float(stored["predicted_risk"])
        output["languages"][language] = {
            "sources": {
                "selection_scores_sha256": sha256(score_path),
                "selection_sha256": sha256(selection_path),
                "followup_summary_sha256": sha256(summary_path),
            },
            "calibration": {
                "n_windows": len(calibration_rows),
                "saved_b": fitted_b,
                "refit_b_through_origin": refit_b,
                "rmse_recomputed": rmse,
                "mae_recomputed": mae,
                "saved_rmse": float(summary["calibration"]["fit"]["rmse"]),
                "saved_mae": float(summary["calibration"]["fit"]["mae"]),
                "shrink_factor_relative_to_b1": 1.0 / refit_b,
                "risk_threshold_after_scaling": -TAU / refit_b,
                "b1_selection": b1,
                "fitted_b_selection": calibrated,
                "unconstrained_selection": unconstrained,
                "frozen_b1_selection": {
                    "start": stored["graft"][0],
                    "end": stored["graft"][-1],
                    "predicted_utility": float(stored["predicted_utility"]),
                    "predicted_risk_b1": float(stored["predicted_risk"]),
                    "predicted_risk_fitted_b": scaled_selected_risk,
                },
                "candidate_windows": len(candidate_windows),
                "all_666_saved_window_scores_reproduced": True,
                "predicted_feasible_windows_b1": sum(
                    row["predicted_risk"] >= -TAU for row in candidate_windows
                ),
                "predicted_feasible_windows_fitted_b": sum(
                    refit_b * row["predicted_risk"] >= -TAU for row in candidate_windows
                ),
                "calibration_window_diagnostics_in_sample": calibration_summary(
                    calibration_rows, refit_b
                ),
            },
        }
    return output


if __name__ == "__main__":
    result = run()
    destination = ROOT / "results" / "b3" / "generation_followup_v1" / "calibration_check.json"
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(destination)
