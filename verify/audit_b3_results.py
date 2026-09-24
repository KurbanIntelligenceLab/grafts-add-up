"""Check published 8B answer and language results against released records.

Run from the repository root: python verify/audit_b3_results.py
This does not rerun models or alter frozen evidence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "results" / "b3" / "lighton_qwen3_8b"
SEED = 20260913
RESAMPLES = 20_000
FIDELITY_SEED = 20260918
FIDELITY_SHA256 = {
    "fr": "cdf59f7812c9a844d64ef32122f9b30d54b052060d5ba13291b2df22f08d4fa0",
    "zh": "c93c10854673d773caa616476cdd0c5e365ed5ddd5c331cf5d1be0ce4bfacc26",
}


def records(language: str) -> dict[str, dict[str, float]]:
    path = ROOT / f"{language}_host__en_donor" / "measure" / "measurements.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    by_candidate: dict[str, dict[str, float]] = {}
    for row in rows:
        by_candidate.setdefault(row["candidate"], {})[row["record_id"]] = row[
            "answer_logprob_per_token"
        ]
    expected = {"host_baseline", "selected", "published_reference"}
    assert set(by_candidate) == expected
    ids = list(by_candidate["host_baseline"])
    assert len(ids) == 250
    assert all(set(values) == set(ids) for values in by_candidate.values())
    return by_candidate


def difference(
    data: dict[str, dict[str, float]], left: str, right: str
) -> np.ndarray:
    # Preserve JSONL insertion order: it is the order used by the frozen analysis.
    ids = data["host_baseline"]
    return np.array([data[left][key] - data[right][key] for key in ids])


def check() -> None:
    published = json.loads((ROOT / "posthoc_statistics.json").read_text(encoding="utf-8"))
    comparison = json.loads(
        (ROOT / "posthoc_selected_vs_published.json").read_text(encoding="utf-8")
    )
    assert published["analysis"]["seed"] == SEED
    assert published["analysis"]["bootstrap_resamples"] == RESAMPLES
    rng = np.random.default_rng(SEED)

    for language in ("fr", "zh"):
        path = ROOT / f"{language}_host__en_donor" / "measure" / "measurements.jsonl"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == published["source_sha256"][f"{language}_measurements"]
        data = records(language)

        for label, left, right in (
            ("selected_minus_host", "selected", "host_baseline"),
            ("published_reference_minus_host", "published_reference", "host_baseline"),
        ):
            delta = difference(data, left, right)
            reported = published["comparisons"][language][label]
            interval = np.quantile(
                rng.choice(delta, (RESAMPLES, len(delta)), replace=True).mean(axis=1),
                [0.025, 0.975],
            )
            assert abs(delta.mean() - reported["mean"]) < 1e-6
            assert np.allclose(interval, reported["ci95"], atol=1e-6)
            assert abs(np.mean(delta > 0) - reported["fraction_positive"]) < 1e-6
            print(f"PASS {language} {label}: {delta.mean():+.6f}, CI {interval}")

        delta = difference(data, "selected", "published_reference")
        reported = comparison["comparisons"][language]["selected_minus_published"]
        assert abs(delta.mean() - reported["mean"]) < 1e-12
        assert abs(np.mean(delta > 0) - reported["fraction_positive"]) < 1e-12
        # The released JSON does not specify the draw ordering for this separate
        # bootstrap. Its stored interval is within sampling error of a fresh
        # 20,000-resample interval from the same released paired observations.
        interval = np.quantile(
            np.random.default_rng(SEED).choice(
                delta, (RESAMPLES, len(delta)), replace=True
            ).mean(axis=1),
            [0.025, 0.975],
        )
        assert np.allclose(interval, reported["ci95"], atol=0.003)
        print(f"PASS {language} selected_minus_published: {delta.mean():+.6f}")


def check_language_fidelity() -> None:
    summary = json.loads(
        (ROOT / "posthoc_language_fidelity.json").read_text(encoding="utf-8")
    )
    assert summary["analysis"]["n_records_per_language"] == 76
    for language in ("fr", "zh"):
        probe_path = ROOT / "language_fidelity_records" / f"{language}_probes.jsonl"
        probe_text = probe_path.read_text(encoding="utf-8")
        assert "F:\\" not in probe_text and "/workspace/" not in probe_text
        probe_lines = [json.loads(line) for line in probe_text.splitlines()]
        probe_header, probes = probe_lines[0], probe_lines[1:]
        assert probe_header["n_records"] == len(probes) == 76
        for probe in probes:
            canonical = json.dumps(
                {key: value for key, value in probe.items() if key != "_hash"},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == probe["_hash"]

        path = ROOT / "language_fidelity_records" / f"{language}_measurements.jsonl"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == FIDELITY_SHA256[language]
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        by_id: dict[str, dict[str, float]] = {}
        for row in rows:
            by_id.setdefault(row["record_id"], {})[row["candidate"]] = row[
                "language_readout"
            ]
        assert len(by_id) == 76 and len(rows) == 228
        assert set(by_id) == {probe["id"] for probe in probes}
        assert all(
            set(values) == {"host_baseline", "selected", "published_reference"}
            for values in by_id.values()
        )
        language_summary = summary["languages"][language]
        for label, candidate in (
            ("host", "host_baseline"),
            ("selected", "selected"),
            ("published", "published_reference"),
        ):
            mean = np.mean([by_id[key][candidate] for key in by_id])
            assert round(float(mean), 3) == language_summary["means"][label]

        for label, left, right in (
            ("selected_minus_host", "selected", "host_baseline"),
            ("selected_minus_published", "selected", "published_reference"),
        ):
            delta = np.asarray(
                [by_id[key][left] - by_id[key][right] for key in sorted(by_id)]
            )
            interval = np.quantile(
                np.random.default_rng(FIDELITY_SEED)
                .choice(delta, (RESAMPLES, len(delta)), replace=True)
                .mean(axis=1),
                [0.025, 0.975],
            )
            reported = language_summary[label]
            assert abs(delta.mean() - reported["mean"]) < 1e-9
            assert np.allclose(interval, reported["ci95"], atol=1e-9)
            print(f"PASS {language} fidelity {label}: {delta.mean():+.6f}, CI {interval}")


if __name__ == "__main__":
    check()
    check_language_fidelity()
