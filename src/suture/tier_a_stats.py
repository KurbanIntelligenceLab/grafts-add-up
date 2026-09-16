"""Compute item-aligned Tier-A statistics without reading MGSM_test.

The Spanish canonical seed-0 path is a single run.  This module therefore does
not average p-values across missing expert seeds.  It also separates C-set
empirical drift from a Clopper-Pearson certificate, and it does not treat the
non-empty target-language check as retained general capability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from suture.paths import PAPER_DIR

try:
    from suture.suture_metrics import certify_drift, spearman
except ModuleNotFoundError:
    from suture_metrics import certify_drift, spearman


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _item_means(row: Mapping[str, Any]) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = {}
    for item in row["generation"]["items"]:
        grouped.setdefault(str(item["source_id"]), []).append(
            float(bool(item["exact_match"]))
        )
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def paired_bootstrap(
    selected: Mapping[str, float],
    best: Mapping[str, float],
    *,
    repetitions: int = 10_000,
    seed: int = 0,
) -> Dict[str, Any]:
    ids = sorted(set(selected) & set(best))
    if not ids:
        raise ValueError("no overlapping item IDs for paired bootstrap")
    differences = np.asarray(
        [best[item_id] - selected[item_id] for item_id in ids],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    samples = rng.choice(differences, size=(repetitions, len(differences)), replace=True)
    means = samples.mean(axis=1)
    observed = float(differences.mean())
    return {
        "n_items": len(ids),
        "observed_regret": observed,
        "mean": float(means.mean()),
        "ci95": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
        "p_regret_le_zero": float((np.sum(means <= 0.0) + 1) / (repetitions + 1)),
        "item_ids": ids,
    }


def holm_adjust(p_values: Mapping[str, float]) -> Dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: Dict[str, float] = {}
    running = 0.0
    n = len(ordered)
    for index, (name, value) in enumerate(ordered):
        adjusted_value = min(1.0, (n - index) * float(value))
        running = max(running, adjusted_value)
        adjusted[name] = running
    return adjusted


def _drift_flags(items: Sequence[Mapping[str, Any]]) -> List[int]:
    return [0 if bool(item.get("target_language_match")) else 1 for item in items]


def _timing(report: Mapping[str, Any], manifest: Mapping[str, Any]) -> Dict[str, Any]:
    selection_resumed = bool(report.get("selection_scores_resumed"))
    sweep_resumed = int(report.get("sweep_windows_resumed") or 0)
    manifest_duration = manifest.get("duration_seconds")
    return {
        "selection_score_seconds_recorded": report.get("score_seconds"),
        "selection_timing_status": (
            "unknown_after_resume" if selection_resumed else "measured"
        ),
        "sweep_seconds_recorded": report.get("sweep_seconds"),
        "sweep_windows_resumed": sweep_resumed,
        "sweep_timing_status": (
            "partial_after_resume" if sweep_resumed else "measured"
        ),
        "run_manifest_duration_seconds": manifest_duration,
        "run_manifest_duration_status": (
            "unknown_after_resume"
            if manifest_duration in (None, 0, 0.0)
            else "measured"
        ),
    }


def analyze_canonical_run(
    path: Path,
    *,
    repetitions: int = 10_000,
    seed: int = 0,
) -> Dict[str, Any]:
    report = _load_json(path / "e1_gate.json")
    rows = _load_rows(path / "e1_window_sweep.jsonl")
    manifest = (
        _load_json(path / "run_manifest.json")
        if (path / "run_manifest.json").is_file()
        else {}
    )
    selected = tuple(report["suture_selected"]["graft"])
    best = tuple(
        range(int(report["sweep_best_window"][0]), int(report["sweep_best_window"][1]) + 1)
    )
    by_graft = {tuple(row["graft"]): row for row in rows}
    if selected not in by_graft or best not in by_graft:
        raise ValueError(f"selected or best window missing from {path}")
    bootstrap = paired_bootstrap(
        _item_means(by_graft[selected]),
        _item_means(by_graft[best]),
        repetitions=repetitions,
        seed=seed,
    )
    predicted = np.asarray([row["predicted_utility"] for row in rows], dtype=float)
    teacher_forced = np.asarray(
        [row["measured_teacher_forced_sequence_logprob"] for row in rows],
        dtype=float,
    )
    exact_match = np.asarray([row["measured_exact_match"] for row in rows], dtype=float)
    selected_certificate_items = report["selected_certificate"]["items"]
    baseline_certificate_items = report["baseline_certificate"]["items"]
    selected_c_flags = _drift_flags(selected_certificate_items)
    baseline_c_flags = _drift_flags(baseline_certificate_items)
    selected_certificate = certify_drift(selected_c_flags, delta=0.05)
    baseline_certificate = certify_drift(baseline_c_flags, delta=0.05)
    selected_retention = report.get(
        "selected_target_language_nonempty",
        report["selected_retention"],
    )
    return {
        "path": str(path).replace("\\", "/"),
        "language": report["language"],
        "expert_seeds_measured": [0],
        "p_values_averaged_across_seeds": False,
        "gate": report["gate"],
        "spearman_cross_set_exact_match": float(report["spearman"]),
        "pearson_cross_set_exact_match": float(report["pearson"]),
        "spearman_cross_set_teacher_forced": (
            float(spearman(predicted, teacher_forced))
            if np.std(predicted) > 0 and np.std(teacher_forced) > 0
            else 0.0
        ),
        "n_windows": len(rows),
        "n_dev_items": int(report["n_dev_records"]),
        "decode_seeds": list(report.get("selected_certificate", {}).get("decode_seeds", [])),
        "selected_exact_match": float(report["selected_exact_match"]),
        "sweep_best_exact_match": float(report["sweep_best_exact_match"]),
        "baseline_exact_match": float(report["baseline_exact_match"]),
        "accuracy_regret": float(report["accuracy_regret"]),
        "accuracy_regret_fraction_of_achievable_gain": report[
            "accuracy_regret_fraction_of_achievable_gain"
        ],
        "bootstrap": bootstrap,
        "task_set_drift": {
            "source": "MGSM_dev_free_generation",
            "selected_rate": float(report["selected_drift_rate"]),
            "baseline_rate": float(report["baseline_drift_rate"]),
            "n": int(np.size(exact_match)),
            "is_certificate": False,
        },
        "c_set": {
            "source": "C_held_out_prompts",
            "status": "empirical_plus_clopper_pearson",
            "selected_empirical_rate": selected_certificate["empirical_rate"],
            "baseline_empirical_rate": baseline_certificate["empirical_rate"],
            "n_selected": selected_certificate["n_calibration"],
            "certificate": selected_certificate,
            "baseline_certificate": baseline_certificate,
            "is_certificate_until_bound_applied": False,
        },
        "target_language_nonempty_check": {
            "measured_name": "target_language_nonempty_check",
            "target_language_rate": float(report["selected_retention_rate"]),
            "nonempty_rate": float(selected_retention["nonempty_rate"]),
            "n": int(selected_retention["n"]),
            "belebele_or_knowledge": "unmeasured",
            "not_general_capability_retention": True,
        },
        "retention": {
            "measured_name": "target_language_nonempty_check",
            "alias_of": "target_language_nonempty_check",
            "target_language_rate": float(report["selected_retention_rate"]),
            "nonempty_rate": float(selected_retention["nonempty_rate"]),
            "n": int(selected_retention["n"]),
            "belebele_or_knowledge": "unmeasured",
            "not_general_capability_retention": True,
        },
        "timing": _timing(report, manifest),
        "same_probe_versus_cross_set": {
            "headline_ranking": "cross_set_P_util_P_risk_to_MGSM_dev",
            "same_probe_linearisation": "P2_and_E2_on_selection_probes",
            "cross_set_failure_is_bookkeeping": False,
            "teacher_forced_on_MGSM_dev_is_still_cross_set": True,
        },
        "mgsm_test_read": False,
    }


V2_ANALYSIS_SCHEMA = {
    "schema_version": 1,
    "contract_version": 2,
    "required_language_fields": [
        "spearman_cross_set_exact_match",
        "pearson_cross_set_exact_match",
        "accuracy_regret",
        "accuracy_regret_fraction_of_achievable_gain",
        "top5_overlap",
        "selected_drift_rate",
        "belebele_retention",
        "readiness_gains",
        "bootstrap",
        "e2_regime",
        "compute",
    ],
    "classifications": [
        "broad_improvement",
        "language_dependent_applicability",
        "persistent_cross_set_failure",
        "expert_or_setup_failure",
    ],
}


def analyze_v2_language(path: Path) -> Dict[str, Any]:
    report = _load_json(path / "e1_gate.json")
    rows = _load_rows(path / "e1_window_sweep.jsonl")
    selected = tuple(report["suture_selected"]["graft"])
    best = tuple(
        range(int(report["sweep_best_window"][0]), int(report["sweep_best_window"][1]) + 1)
    )
    by_graft = {tuple(row["graft"]): row for row in rows}
    bootstrap = paired_bootstrap(
        _item_means(by_graft[selected]),
        _item_means(by_graft[best]),
        repetitions=10_000,
        seed=0,
    )
    predicted = np.asarray([row["predicted_utility"] for row in rows], dtype=float)
    teacher_forced = np.asarray(
        [row["measured_teacher_forced_sequence_logprob"] for row in rows],
        dtype=float,
    )
    return {
        "path": str(path).replace("\\", "/"),
        "language": report["language"],
        "contract_version": 2,
        "gate": report["gate"],
        "spearman_cross_set_exact_match": float(report["spearman"]),
        "pearson_cross_set_exact_match": float(report["pearson"]),
        "spearman_cross_set_teacher_forced": (
            float(spearman(predicted, teacher_forced))
            if np.std(predicted) > 0 and np.std(teacher_forced) > 0
            else 0.0
        ),
        "n_windows": len(rows),
        "n_dev_items": int(report["n_dev_records"]),
        "selected_exact_match": float(report["selected_exact_match"]),
        "sweep_best_exact_match": float(report["sweep_best_exact_match"]),
        "baseline_exact_match": float(report["baseline_exact_match"]),
        "accuracy_regret": float(report["accuracy_regret"]),
        "accuracy_regret_fraction_of_achievable_gain": report[
            "accuracy_regret_fraction_of_achievable_gain"
        ],
        "top5_overlap": report.get("top5_overlap"),
        "selected_drift_rate": report.get("selected_drift_rate"),
        "bootstrap": bootstrap,
        "belebele_retention": {
            "selected": report.get("selected_belebele"),
            "baseline": report.get("baseline_belebele"),
            "oracle": report.get("oracle_belebele"),
            "not_lid_proxy": True,
        },
        "c_set": {
            "selected_empirical_rate": report.get("selected_certificate", {}).get("drift_rate"),
            "baseline_empirical_rate": report.get("baseline_certificate", {}).get("drift_rate"),
        },
        "compute": {
            "selection_score_seconds": report.get("selection_score_seconds"),
            "sweep_seconds": report.get("sweep_seconds"),
            "merged_models_built": report.get("merged_models_built"),
            "selection_merged_models_built": report.get("selection_merged_models_built"),
        },
        "e2_regime": "recorded_separately_explanatory_only",
        "readiness_gains": "recorded_separately",
        "mgsm_test_read": False,
        "same_probe_versus_cross_set": {
            "headline_ranking": "cross_set_P_util_P_risk_to_MGSM_dev",
            "teacher_forced_on_MGSM_dev_is_still_cross_set": True,
        },
    }


def classify_v2_panel(results: Mapping[str, Dict[str, Any]], blocked: Sequence[str]) -> str:
    if not results:
        return "expert_or_setup_failure"
    passed = [row["gate"] == "PASS_E1" for row in results.values()]
    if all(passed) and len(passed) == 3 and not blocked:
        return "broad_improvement"
    if any(passed) and (any(not value for value in passed) or blocked):
        return "language_dependent_applicability"
    if blocked:
        return "language_dependent_applicability" if results else "expert_or_setup_failure"
    return "persistent_cross_set_failure"


def analyze_v2_panel(root: Path) -> Dict[str, Any]:
    """Lock-step v2 analysis.  Must be importable before holdout is read."""

    results: Dict[str, Dict[str, Any]] = {}
    blocked: List[str] = []
    blocked_details: Dict[str, Dict[str, Any]] = {}
    for language in ("es", "zh", "sw"):
        e1 = root / language / "e1"
        readiness = root / language / "readiness" / "readiness.json"
        if (e1 / "e1_gate.json").is_file():
            row = analyze_v2_language(e1)
            if readiness.is_file():
                ready = _load_json(readiness)
                row["readiness_gains"] = {
                    "donor_gain": ready.get("donor_gain"),
                    "donor_em": (ready.get("donor_expert") or {}).get("exact_match"),
                    "host_logprob_gain": ready.get("host_logprob_gain"),
                    "host_target_language_rate": (ready.get("host_expert") or {}).get(
                        "target_language_rate"
                    ),
                    "belebele_drop": ready.get("belebele_drop"),
                    "gate": ready.get("gate"),
                }
            e2_report = root / language / "e2" / "e2_report.json"
            if e2_report.is_file():
                row["e2_regime"] = _load_json(e2_report)
            holdout_path = root / language / "holdout" / "mgsm_test_once.json"
            if holdout_path.is_file():
                holdout = _load_json(holdout_path)
                windows = holdout.get("windows") or {}
                row["mgsm_test_read"] = True
                row["holdout"] = {
                    "n": holdout.get("n_test"),
                    "baseline_exact_match": (windows.get("host") or {}).get("generation", {}).get("exact_match"),
                    "selected_exact_match": (windows.get("selected") or {}).get("generation", {}).get("exact_match"),
                    "oracle_exact_match": (windows.get("dev_oracle") or {}).get("generation", {}).get("exact_match"),
                }
            results[language] = row
        elif readiness.is_file():
            ready = _load_json(readiness)
            blocked.append(language)
            blocked_details[language] = {
                "gate": ready.get("gate"),
                "donor_gain": ready.get("donor_gain"),
                "donor_em": (ready.get("donor_expert") or {}).get("exact_match"),
                "host_logprob_gain": ready.get("host_logprob_gain"),
                "host_target_language_rate": (ready.get("host_expert") or {}).get(
                    "target_language_rate"
                ),
                "host_nonempty_rate": (ready.get("host_expert") or {}).get("nonempty_rate"),
                "belebele_drop": ready.get("belebele_drop"),
            }
        else:
            blocked.append(language)
            blocked_details[language] = {"gate": "missing_readiness"}
    p_values = {
        language: float(row["bootstrap"]["p_regret_le_zero"])
        for language, row in results.items()
    }
    summary = {
        "schema": V2_ANALYSIS_SCHEMA,
        "contract_version": 2,
        "comparison_to_v1": "new_study_applicability_comparison_not_an_erased_retry",
        "v1_spanish_canonical": "results/tier_a/es/e1_canonical_seed0",
        "expert_seeds_measured": [0],
        "languages": list(results),
        "blocked_languages": blocked,
        "blocked_details": blocked_details,
        "results": results,
        "holm_adjusted_p_values": holm_adjust(p_values) if p_values else {},
        "classification": classify_v2_panel(results, blocked),
        "mgsm_test_read": any(row.get("mgsm_test_read") for row in results.values()),
        "p_values_averaged_across_seeds": False,
    }
    output = root / "v2_panel_statistics.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        update_paper_v2(summary)
    except OSError:
        pass
    return summary


V2_PAPER_BEGIN = "% BEGIN_CONTRACT_V2"
V2_PAPER_END = "% END_CONTRACT_V2"


def _tex_ident(value: Any) -> str:
    return str(value or "").replace("_", r"\_")


def _tex_num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def render_v2_paper_block(summary: Mapping[str, Any]) -> str:
    classification = str(summary.get("classification") or "pending")
    languages = summary.get("results") or {}
    blocked = summary.get("blocked_languages") or []
    lines = [
        V2_PAPER_BEGIN,
        r"\paragraph{Contract-v2 follow-up (authorized, separate study).}",
        r"After the Spanish v1 stop, a new hashed contract authorized a fail-closed rerun on exactly \texttt{Qwen/Qwen3-1.7B} revision \texttt{70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}. Frozen v1 artifacts, including \texttt{results/tier\_a/es/e1\_canonical\_seed0/}, were not rewritten. New artifacts live under \texttt{results/v2/qwen3\_1\_7b/}. The v2 development split is a hash-frozen 64/186 MGSM partition with identical item IDs across Spanish, Chinese, and Swahili. One expert-training seed was used, so v2 tests cross-language behavior rather than training-seed robustness. Decoding was frozen in non-thinking mode ($T{=}0.7$, top-$p{=}0.8$, top-$k{=}20$, 128 new tokens). Comparison to v1 is a new-study applicability comparison, not an erased retry. The original Spanish v1 negative result (free-generation Spearman $-0.018$, regret $0.05$) is retained.",
    ]
    if languages or blocked:
        lines.append(
            rf"Panel classification: \texttt{{{_tex_ident(classification)}}}. "
            rf"Languages that reached E1: {', '.join(languages) if languages else 'none'}. "
            rf"Blocked before E1: {', '.join(blocked) if blocked else 'none'}."
        )
        details = summary.get("blocked_details") or {}
        for language in blocked:
            row = details.get(language) or {}
            lines.append(
                rf"{language} readiness: gate \texttt{{{_tex_ident(row.get('gate'))}}}, "
                rf"donor EM ${_tex_num(row.get('donor_em'))}$ "
                rf"(gain ${_tex_num(row.get('donor_gain'))}$), "
                rf"host LID ${_tex_num(row.get('host_target_language_rate'))}$, "
                rf"host logprob gain ${_tex_num(row.get('host_logprob_gain'))}$, "
                rf"Belebele drop ${_tex_num(row.get('belebele_drop'))}$. "
                r"No E1."
            )
        if not languages and blocked:
            lines.append(
                r"P1--P3, E2, exhaustive E1, and MGSM-test were not run because no language pair passed expert readiness. Thresholds were not weakened after seeing these outcomes."
            )
        for language, row in languages.items():
            hold = row.get("holdout") or {}
            ready = row.get("readiness_gains") or {}
            if row.get("mgsm_test_read"):
                holdout_text = (
                    f"Holdout $n={hold.get('n')}$: selected ${_tex_num(hold.get('selected_exact_match'))}$, "
                    f"host ${_tex_num(hold.get('baseline_exact_match'))}$, "
                    f"dev-oracle ${_tex_num(hold.get('oracle_exact_match'))}."
                )
            else:
                holdout_text = "MGSM-test unread for this language."
            lines.append(
                rf"{language}: Spearman ${_tex_num(row.get('spearman_cross_set_exact_match'))}$, "
                rf"Pearson ${_tex_num(row.get('pearson_cross_set_exact_match'))}$, "
                rf"regret ${_tex_num(row.get('accuracy_regret'))}$, "
                rf"selected EM ${_tex_num(row.get('selected_exact_match'))}$ vs "
                rf"host ${_tex_num(row.get('baseline_exact_match'))}$, "
                rf"drift ${_tex_num(row.get('selected_drift_rate'))}$, "
                rf"gate \texttt{{{_tex_ident(row.get('gate'))}}}. "
                rf"Readiness donor EM ${_tex_num(ready.get('donor_em'))}$ "
                rf"(gain ${_tex_num(ready.get('donor_gain'))}$). "
                + holdout_text
            )
        if summary.get("holm_adjusted_p_values"):
            holm = ", ".join(
                f"{name} {_tex_num(value)}"
                for name, value in (summary.get("holm_adjusted_p_values") or {}).items()
            )
            lines.append(rf"Holm-adjusted regret $p$-values: {holm}.")
    else:
        lines.append(
            r"v2 E1 results are not yet written into this paragraph; the protocol is frozen and the original v1 Spanish stop remains the reported contract-v1 outcome."
        )
    lines.append(V2_PAPER_END)
    return "\n".join(lines) + "\n"


def update_paper_v2(summary: Mapping[str, Any], paper_path: Path | None = None) -> Path:
    path = paper_path or PAPER_DIR / "main.tex"
    text = path.read_text(encoding="utf-8")
    block = render_v2_paper_block(summary)
    if V2_PAPER_BEGIN in text and V2_PAPER_END in text:
        start = text.index(V2_PAPER_BEGIN)
        end = text.index(V2_PAPER_END) + len(V2_PAPER_END)
        text = text[:start] + block.rstrip() + text[end:]
    else:
        needle = "\\end{document}"
        if needle not in text:
            raise OSError(f"cannot find \\end{{document}} in {path}")
        text = text.replace(needle, block + needle, 1)
    path.write_text(text, encoding="utf-8")
    return path


def _seed_result(path: Path) -> Dict[str, Any]:
    return analyze_canonical_run(path)


def _canonical_run_path(root: Path, language: str, seed: int) -> Path:
    candidates = (
        root / language / f"e1_canonical_seed{seed}",
        root / language / f"seed{seed}" / "e1",
    )
    for path in candidates:
        if (path / "e1_gate.json").is_file():
            return path
    return candidates[0]


def run(
    *,
    root: Path,
    languages: Sequence[str] = ("es",),
    expert_seeds: Sequence[int] = (0,),
) -> Dict[str, Any]:
    results: Dict[str, List[Dict[str, Any]]] = {}
    missing: List[str] = []
    for language in languages:
        results[language] = []
        for seed in expert_seeds:
            path = _canonical_run_path(root, language, int(seed))
            if not (path / "e1_gate.json").is_file():
                missing.append(str(path).replace("\\", "/"))
                continue
            results[language].append(analyze_canonical_run(path))
    p_values = {}
    for language, rows in results.items():
        if len(rows) == 1:
            p_values[language] = float(rows[0]["bootstrap"]["p_regret_le_zero"])
        elif len(rows) > 1:
            raise ValueError(
                f"{language} has {len(rows)} expert seeds; do not average p-values. "
                "Report each seed separately."
            )
    summary = {
        "expert_seeds_requested": list(expert_seeds),
        "languages": list(languages),
        "results": results,
        "missing_canonical_runs": missing,
        "holm_adjusted_p_values": holm_adjust(p_values) if p_values else {},
        "p_values_averaged_across_seeds": False,
        "mgsm_test_read": False,
        "blocked_languages": [
            language
            for language in ("zh", "sw")
            if language not in results or not results[language]
        ],
    }
    output = root / "tier_a_replication_statistics.json"
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    spanish = root / "es" / "e1_canonical_seed0_posthoc_statistics.json"
    if results.get("es"):
        with spanish.open("w", encoding="utf-8") as handle:
            json.dump(results["es"][0], handle, indent=2, sort_keys=True)
            handle.write("\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--languages", default="es")
    parser.add_argument("--expert-seeds", default="0")
    parser.add_argument("--v2", action="store_true")
    args = parser.parse_args()
    if args.v2:
        summary = analyze_v2_panel(args.root)
        print(json.dumps(summary, indent=2, sort_keys=True)[:4000])
        return 0
    summary = run(
        root=args.root,
        languages=tuple(value for value in args.languages.split(",") if value),
        expert_seeds=tuple(
            int(value) for value in args.expert_seeds.split(",") if value
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not summary["missing_canonical_runs"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

