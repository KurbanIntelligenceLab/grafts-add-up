"""Retain conservative numeral-consistency flags for all frozen audit pairs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from suture.translation_number_checks import audit_numbers


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "results/b3/translation_audit_v1")
    args = parser.parse_args()
    root = args.root
    pairs = root / "paired_texts.jsonl"
    module = ROOT / "src/suture/translation_number_checks.py"
    for name in ("number_freeze.json", "number_records.jsonl", "number_summary.json"):
        if (root / name).exists():
            raise RuntimeError(f"Refusing to overwrite {name}")
    freeze = {"input_sha256": sha(pairs), "checker_sha256": sha(module),
              "runner_sha256": sha(__file__), "scope": "EN to FR/ZH explicit numeral multisets",
              "classification": "match, mismatch, review-needed under the declared conservative grammar",
              "limitations": ["Not a confirmed translation error rate or semantic/units validation.",
                              "Ambiguous and unsupported constructions remain review-needed.",
                              "No probe filtering, replacement, or selection rerun."],
              "timing": "post-submission, post-hoc audit; not externally preregistered"}
    write(root / "number_freeze.json", freeze)
    rows = [json.loads(x) for x in pairs.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2000 and len({r["id"] for r in rows}) == 2000
    checks = []
    for r in rows:
        checks.append({"id": r["id"], "language": r["language"], "probe": r["probe"],
                       "check": audit_numbers(r["source_text"], r["target_text"], r["language"])})
    assert sha(module) == freeze["checker_sha256"]
    with (root / "number_records.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in checks:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    grouped = defaultdict(list)
    for row in checks:
        grouped[f"{row['language']}/{row['probe']}"].append(row)
    summary = {"protocol": freeze, "groups": {}, "total_pairs": len(checks)}
    for group, rs in sorted(grouped.items()):
        numeric = [r for r in rs if r["check"]["source_has_numeric_signal"]]
        no_signal = [r for r in rs if not r["check"]["source_has_numeric_signal"]
                     and not r["check"]["target_has_numeric_signal"]]
        summary["groups"][group] = {
            "n": len(rs), "source_numeric_signal_n": len(numeric),
            "no_numeric_signal_on_either_side_n": len(no_signal),
            "supported_status_counts": dict(Counter(r["check"]["status"] for r in rs)),
            "source_numeric_supported_status_counts": dict(Counter(r["check"]["status"] for r in numeric)),
            "strict_digit_status_counts": dict(Counter(r["check"]["strict_digit_comparison"]["status"] for r in rs)),
            "strict_digit_multiset_difference_n": sum(bool(r["check"]["strict_digit_comparison"]["missing_from_target"]
                 or r["check"]["strict_digit_comparison"]["extra_in_target"]) for r in rs),
            "supported_multiset_difference_n": sum(bool(r["check"]["supported_number_comparison"]["missing_from_target"]
                 or r["check"]["supported_number_comparison"]["extra_in_target"]) for r in rs),
            "ambiguous_span_pair_n": sum(bool(r["check"]["ambiguous_source_spans"] or r["check"]["ambiguous_target_spans"]) for r in rs),
            "unhandled_span_pair_n": sum(bool(r["check"]["unhandled_source_spans"] or r["check"]["unhandled_target_spans"]) for r in rs),
        }
    summary["artifacts"] = {p.name: sha(p) for p in (pairs, root / "number_freeze.json", root / "number_records.jsonl")}
    write(root / "number_summary.json", summary)
    print(json.dumps(summary["groups"], indent=2))


if __name__ == "__main__":
    main()
