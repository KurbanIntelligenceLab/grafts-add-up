"""Audit controlled-stack summaries and plots against released per-stack rows.

The JSON files retain per-stack outcomes, but not every candidate-window score
or the original full-run generator. This script audits reported aggregates.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "paper" / "figs"
RESULTS = ROOT / "results" / "controlled_stacks"
CI_ERRORS: list[float] = []


def load(name: str, n: int) -> tuple[dict, list[dict]]:
    data = json.loads((RESULTS / name).read_text(encoding="utf-8"))
    assert len(data["stacks"]) == data["summary"]["n_stacks"] == n
    if all("seed" in row for row in data["stacks"]):
        assert len({row["seed"] for row in data["stacks"]}) == n
    return data["summary"], data["stacks"]


def check_stats(values: list[float], published: dict) -> None:
    array = np.asarray(values, dtype=float)
    assert np.isclose(array.mean(), published["mean"], atol=1e-11)
    assert np.isclose(array.std(ddof=1), published["sd"], atol=1e-11)
    for key, observed in (("min", array.min()), ("max", array.max())):
        if key in published:
            assert np.isclose(observed, published[key], atol=1e-11)


def check_ci(difference: np.ndarray, published: dict) -> None:
    """Independent percentile bootstrap; bounds may vary with the RNG seed."""
    rng = np.random.default_rng(20260923)
    draws = difference[rng.integers(0, len(difference), (20000, len(difference)))].mean(axis=1)
    bounds = np.quantile(draws, (0.025, 0.975))
    error = float(np.max(np.abs(bounds - published["ci95"])))
    CI_ERRORS.append(error)
    assert error < 0.01, (bounds, published["ci95"], error)


def check_comparison(stacks: list[dict], section: str, summary: dict) -> int:
    checks = 0
    baseline = np.array([row[section]["suture"]["regret"] for row in stacks])
    for method, published in summary["table"].items():
        for metric in ("spearman", "regret"):
            values = [row[section][method][metric] for row in stacks]
            stats = ({"mean": published[f"{metric}_mean"], "sd": published[f"{metric}_sd"]}
                     if f"{metric}_mean" in published else published[metric])
            check_stats(values, stats)
            checks += 1
    for method, published in summary["tests"].items():
        difference = np.array([row[section][method]["regret"] for row in stacks]) - baseline
        assert np.isclose(difference.mean(), published["mean_diff"], atol=1e-11)
        observed_p = 1.0 if np.allclose(difference, 0, atol=1e-14) else wilcoxon(difference).pvalue
        assert np.isclose(observed_p, published["wilcoxon_p"], atol=1e-11), (section, method, observed_p, published["wilcoxon_p"])
        check_ci(difference, published)
        checks += 3
    return checks


def close_table(observed: np.ndarray, actual: np.ndarray, tolerance: float) -> None:
    assert observed.shape == actual.shape
    assert np.allclose(observed, actual, atol=tolerance, rtol=0), np.max(abs(observed - actual))


def main() -> None:
    checks = 0
    controlled = {}
    for filename, n in (("table1_12stacks.json", 12), ("robust48.json", 48),
                        ("trained_12.json", 12), ("trained_48.json", 48)):
        summary, stacks = load(filename, n)
        controlled[filename] = (summary, stacks)
        for section in ("all78", "w6", "w3"):
            if section in summary:
                checks += check_comparison(stacks, section, summary[section])
        if "superposition_pearson" in summary:
            check_stats([row["superposition"]["pearson"] for row in stacks],
                        summary["superposition_pearson"])
            checks += 1
        if "certify" in summary:
            for key, published in summary["certify"].items():
                if isinstance(published, dict):
                    check_stats([row["certify"][key] for row in stacks], published)
                    checks += 1
        if "builds" in summary:
            baseline = np.array([row["builds"]["suture_0"] for row in stacks])
            for method, published in summary["builds"]["table"].items():
                check_stats([row["builds"][method] for row in stacks], published)
                checks += 1
            for method, published in summary["builds"]["tests"].items():
                difference = np.array([row["builds"][method] for row in stacks]) - baseline
                assert np.isclose(difference.mean(), published["mean_diff"], atol=1e-11)
                observed_p = 1.0 if np.allclose(difference, 0, atol=1e-14) else wilcoxon(difference).pvalue
                assert np.isclose(observed_p, published["wilcoxon_p"], atol=1e-11), (filename, method, observed_p, published["wilcoxon_p"])
                check_ci(difference, published)
                checks += 3

    _, robust = controlled["robust48.json"]
    superplot = np.genfromtxt(FIGS / "fig_super48.dat", names=True)
    close_table(superplot["pearson"],
                np.sort([row["superposition"]["pearson"] for row in robust]), 0.000051)
    certplot = np.genfromtxt(FIGS / "fig_cert48.dat", names=True)
    sorted_cert = sorted(robust, key=lambda row: row["certify"]["deploy_certified_mean"])
    for column, field, tolerance in (("certified", "deploy_certified_mean", 0.0051),
                                      ("false", "deploy_false_mean", 0.00051),
                                      ("beats", "competitors_that_beat_selection", 0)):
        close_table(certplot[column], np.array([row["certify"][field] for row in sorted_cert]), tolerance)
    assert round(float(np.median(superplot["pearson"])), 3) == 0.953
    assert round(float(certplot["certified"].mean()), 1) == 60.5
    assert round(float(certplot["false"].mean()), 2) == 0.09
    checks += 6

    _, trained = controlled["trained_48.json"]
    regretplot = np.genfromtxt(FIGS / "fig_regret_ecdf.dat", names=True)
    for column, key in (("suture", "suture_0"), ("sweep", "sweep_78"),
                        ("random11", "random_11"), ("local11", "local_11")):
        close_table(regretplot[column], np.sort([row["builds"][key] for row in trained]), 0.000051)
        checks += 1

    ablation = np.genfromtxt(FIGS / "fig_ablation2.dat", names=True)
    pert_summary, _ = controlled["table1_12stacks.json"]
    train_summary, _ = controlled["trained_48.json"]
    methods = ("suture", "drop_adjoint", "drop_injection", "inter_expert", "random")
    train_methods = ("suture", "drop_adjoint", "drop_injection", "movement", "random")
    for i, (pert, train) in enumerate(zip(methods, train_methods)):
        p = pert_summary["w6"]["table"][pert]
        t = train_summary["w6"]["table"][train]["regret"]
        for column, value in (("perturbed", p["regret_mean"]),
                              ("perturbed_sd", p["regret_sd"]),
                              ("trained", t["mean"]), ("trained_sd", t["sd"])):
            assert abs(float(ablation[column][i]) - value) < 0.000051
            checks += 1

    scale = json.loads((RESULTS / "scale_sweep_12.json").read_text(encoding="utf-8"))
    regime = np.genfromtxt(FIGS / "fig_regime_multi.dat", names=True)
    for i, (scale_key, published) in enumerate(scale["summary"]["by_scale"].items()):
        stack_rows = [row for row in scale["stacks"] if row["scale"] == float(scale_key)]
        assert len(stack_rows) == 12
        for metric in ("suture_spearman", "suture_regret", "superposition", "cert_frac"):
            check_stats([row[metric] for row in stack_rows], published[metric])
            checks += 1
        for col, metric, stat in (("rho", "suture_spearman", "mean"),
                                  ("rho_sd", "suture_spearman", "sd"),
                                  ("cert", "cert_frac", "mean"),
                                  ("cert_sd", "cert_frac", "sd")):
            assert abs(float(regime[col][i]) - published[metric][stat]) < 0.000051
            checks += 1

    print(f"PASS {checks} controlled-stack aggregate, paired-test, CI, and figure-data checks")
    print(f"Largest independent bootstrap bound difference: {max(CI_ERRORS):.4f}")
    print("LIMIT: original candidate-window scores and full-run generator are still absent")


if __name__ == "__main__":
    main()
