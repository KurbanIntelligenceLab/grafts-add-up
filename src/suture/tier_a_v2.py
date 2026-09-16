"""Contract-v2 Windows-safe orchestrator.

Stages run in isolated subprocesses.  Writes go only to results/v2/qwen3_1_7b.
Frozen v1 trees are refused. Leftover results/v4 paths are refused.  A non-empty setup-issue list aborts
before pilots/E2/E1.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from suture.paths import REPO_ROOT


ROOT = REPO_ROOT
PYTHON = sys.executable
LANGUAGES = ("es", "zh", "sw")
STAGES = (
    "preflight",
    "data",
    "train",
    "checkpoint",
    "readiness",
    "pilots",
    "e2",
    "e1",
    "analyze",
    "holdout",
)


def _panel() -> Path:
    from suture.tier_a_config import results_root

    return results_root(ROOT)


def _lid() -> Path:
    from suture.tier_a_config import LID_MODEL_PATH, repo_root

    return repo_root() / LID_MODEL_PATH


def _run(args: List[str]) -> None:
    print("RUN", args, flush=True)
    completed = subprocess.run([PYTHON, "-u", *args], cwd=str(ROOT))
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _selected(path: Path) -> Path:
    return Path(_json(path)["selected"]["path"])


def _write_panel(payload: Dict[str, Any]) -> None:
    from suture.tier_a_config import refuse_frozen_write

    path = _panel() / "panel_status.json"
    refuse_frozen_write(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_manifest(run_dir: Path) -> None:
    """Fail closed if a stage directory has artifacts but no run manifest."""

    marker = run_dir / "run_manifest.json"
    if marker.is_file():
        return
    has_artifacts = run_dir.is_dir() and any(run_dir.iterdir())
    if has_artifacts:
        raise SystemExit(f"STOP_SETUP: artifacts at {run_dir} have no run_manifest.json")


def _manifest(run_dir: Path, data_dir: Path, kind: str, language: str, command: str) -> None:
    from suture.run_manifest import write_manifest

    manifest = run_dir / "run_manifest.json"
    if manifest.is_file():
        return
    now = datetime.now(timezone.utc).isoformat()
    write_manifest(
        root=ROOT,
        run_dir=run_dir,
        data_dir=data_dir,
        run_kind=kind,
        language=language,
        command=command,
        seed=0,
        started_at=now,
        ended_at=now,
    )


def collect_data_setup_issues(lang: Path) -> List[str]:
    issues: List[str] = []
    summary = lang / "data" / "data_build_summary.json"
    if summary.is_file():
        filt = _json(summary).get("host_filter") or {}
        by_source = filt.get("by_source") or {}
        contributing = [
            source
            for source, counts in by_source.items()
            if int((counts or {}).get("n_kept") or 0) > 0
        ]
        if int(filt.get("n_kept") or 0) == 0:
            issues.append("host_filter_kept_zero")
        if len(contributing) < 2:
            issues.append(f"host_filter_single_source={contributing}")
        empty_pair = int(filt.get("n_empty_pair") or 0)
        parsed = filt.get("parsed_by_source") or {}
        if parsed and any(int(count or 0) == 0 for count in parsed.values()):
            issues.append(f"host_source_unparsed={parsed}")
        if empty_pair and not parsed:
            issues.append(f"host_empty_pairs={empty_pair}")
    host_train = lang / "data" / "host_train.jsonl"
    if host_train.is_file():
        from suture.tier_a_text import looks_like_code

        n_rows = 0
        n_code = 0
        sources = set()
        with host_train.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "_manifest" in row:
                    continue
                n_rows += 1
                sources.add(str(row.get("source_dataset") or ""))
                prompt = str(row.get("prompt") or "")
                completion = str(row.get("completion") or "")
                if looks_like_code(prompt) or looks_like_code(completion):
                    n_code += 1
        sources.discard("")
        if n_code:
            issues.append(f"host_train_residual_code={n_code}/{n_rows}")
        if n_rows and len(sources) < 2:
            issues.append(f"host_train_single_source={sorted(sources)}")
    return issues


collect_data_setup_issues = collect_data_setup_issues


def collect_setup_issues(lang: Path) -> List[str]:
    issues: List[str] = []
    ready = lang / "readiness" / "readiness.json"
    if ready.is_file():
        payload = _json(ready)
        issues.extend(list(payload.get("setup_issues") or []))
        if payload.get("gate") == "STOP_SETUP":
            issues.append("readiness_stop_setup")
        host_n = int((payload.get("host_expert_logprob") or {}).get("n") or 0)
        if host_n and host_n < 128:
            issues.append(f"incomplete_logprob_n={host_n}")
        scorer = (payload.get("belebele_host") or {}).get("scorer")
        if scorer and scorer != "multiple_choice_logprob":
            issues.append(f"belebele_scorer={scorer}")
    host_ck = lang / "checkpoint" / "host.json"
    if host_ck.is_file():
        selected = _json(host_ck).get("selected") or {}
        if selected.get("selection_rule") not in {None, "max_gain_among_passing"}:
            issues.append("unexpected_checkpoint_rule")
        metric_n = int((selected.get("metric") or {}).get("n") or 0)
        if metric_n and metric_n < 64:
            issues.append(f"checkpoint_incomplete_n={metric_n}")
    summary = lang / "data" / "data_build_summary.json"
    if summary.is_file():
        filt = _json(summary).get("host_filter") or {}
        if int(filt.get("n_code") or 0) and int(filt.get("n_kept") or 0) == 0:
            issues.append("host_filter_kept_zero")
    return issues


def _setup_issues(language: str) -> List[str]:
    return collect_setup_issues(_panel() / language)


collect_setup_issues = collect_setup_issues
require_manifest = require_manifest


def main() -> int:
    from suture.tier_a_config import (
        ACTIVE_CONTRACT_VERSION,
        MODEL_ID,
        MODEL_REVISION,
        load_v2_contract,
        refuse_frozen_write,
        sha256_file,
        v2_contract_path,
    )

    load_v2_contract(ROOT)
    refuse_frozen_write(_panel() / ".keep")
    panel: Dict[str, Any] = {
        "contract_version": ACTIVE_CONTRACT_VERSION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "contract_sha256": sha256_file(v2_contract_path(ROOT)),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "RUNNING",
        "languages": {},
        "setup_issues": {},
    }
    _write_panel(panel)
    preflight = _panel() / "preflight" / "preflight_report.json"
    if not preflight.is_file():
        _run(["-m", "suture.tier_a_preflight", "--load-weights"])
    for language in LANGUAGES:
        lang = _panel() / language
        data_summary = lang / "data" / "data_build_summary.json"
        if not data_summary.is_file():
            _run(
                [
                    "-m",
                    "suture.tier_a_data",
                    "--language",
                    language,
                    "--output-dir",
                    str(lang / "data"),
                    "--translation-batch-size",
                    "3",
                    "--device",
                    "cuda",
                ]
            )
            _manifest(lang / "data", lang / "data", "data", language, "tier_a_data")
        else:
            require_manifest(lang / "data")
        data_issues = collect_data_setup_issues(lang)
        if data_issues:
            panel["status"] = "STOP_SETUP"
            panel["setup_issues"][language] = data_issues
            _write_panel(panel)
            raise SystemExit(f"STOP_SETUP {language} data: {data_issues}")
        donor_meta = _panel() / "shared" / "donor" / "training_metadata.json"
        if not donor_meta.is_file():
            _run(
                [
                    "-m",
                    "suture.tier_a_train",
                    "--role",
                    "donor",
                    "--language",
                    "en",
                    "--data-dir",
                    str(lang / "data"),
                    "--output-dir",
                    str(_panel() / "shared" / "donor"),
                    "--seed",
                    "0",
                ]
            )
            _manifest(_panel() / "shared" / "donor", lang / "data", "training", "en", "tier_a_train")
        else:
            require_manifest(_panel() / "shared" / "donor")
        host_meta = lang / "adapters" / "host" / "training_metadata.json"
        if not host_meta.is_file():
            _run(
                [
                    "-m",
                    "suture.tier_a_train",
                    "--role",
                    "host",
                    "--language",
                    language,
                    "--data-dir",
                    str(lang / "data"),
                    "--output-dir",
                    str(lang / "adapters" / "host"),
                    "--seed",
                    "0",
                ]
            )
            _manifest(lang / "adapters" / "host", lang / "data", "training", language, "tier_a_train")
        else:
            require_manifest(lang / "adapters" / "host")
        donor_ck = lang / "checkpoint" / "donor.json"
        host_ck = lang / "checkpoint" / "host.json"
        if not donor_ck.is_file():
            _run(
                [
                    "-m",
                    "suture.tier_a_checkpoint",
                    "--role",
                    "donor",
                    "--adapter-root",
                    str(_panel() / "shared" / "donor"),
                    "--validation",
                    str(lang / "data" / "expert_validation_donor.jsonl"),
                    "--output",
                    str(donor_ck),
                ]
            )
        if not host_ck.is_file():
            _run(
                [
                    "-m",
                    "suture.tier_a_checkpoint",
                    "--role",
                    "host",
                    "--adapter-root",
                    str(lang / "adapters" / "host"),
                    "--validation",
                    str(lang / "data" / "expert_validation_host.jsonl"),
                    "--output",
                    str(host_ck),
                ]
            )
            _manifest(lang / "checkpoint", lang / "data", "checkpoint", language, "tier_a_checkpoint")
        else:
            require_manifest(lang / "checkpoint")
        ready = lang / "readiness" / "readiness.json"
        if not ready.is_file():
            _run(
                [
                    "-m",
                    "suture.tier_a_readiness",
                    "--language",
                    language,
                    "--readiness-donor",
                    str(lang / "data" / "readiness_donor.jsonl"),
                    "--readiness-host",
                    str(lang / "data" / "readiness_host.jsonl"),
                    "--host-adapter",
                    str(_selected(host_ck)),
                    "--donor-adapter",
                    str(_selected(donor_ck)),
                    "--lid-model",
                    str(_lid()),
                    "--output",
                    str(ready),
                    "--belebele",
                    str(lang / "data" / "Belebele.jsonl"),
                    "--contract-version",
                    "3",
                ]
            )
            _manifest(lang / "readiness", lang / "data", "readiness", language, "tier_a_readiness")
        else:
            require_manifest(lang / "readiness")
        issues = _setup_issues(language)
        panel["setup_issues"][language] = issues
        panel["languages"][language] = {"readiness": _json(ready).get("gate"), "setup_issues": issues}
        _write_panel(panel)
        if issues:
            panel["status"] = "STOP_SETUP"
            panel["ended_at"] = datetime.now(timezone.utc).isoformat()
            _write_panel(panel)
            print(f"{language}: STOP_SETUP {issues}", flush=True)
            return 2
        if _json(ready).get("gate") != "PASS_READINESS":
            print(f"{language}: STOP_EXPERT_READINESS", flush=True)
            continue
        pilots = lang / "pilots" / "pilots.json"
        if not pilots.is_file():
            _run(
                [
                    "-m",
                    "suture.tier_a_gate",
                    "pilots",
                    "--data-dir",
                    str(lang / "data"),
                    "--host-dir",
                    str(_selected(host_ck)),
                    "--donor-dir",
                    str(_selected(donor_ck)),
                    "--output-dir",
                    str(lang / "pilots"),
                    "--language",
                    language,
                    "--device",
                    "cuda",
                    "--prompt-limit",
                    "8",
                    "--prompt-batch-size",
                    "1",
                    "--max-length",
                    "128",
                ]
            )
        if float((_json(pilots).get("P1") or {}).get("spearman") or 0.0) < 0.9:
            print(f"{language}: STOP_PLUMBING_P1", flush=True)
            continue
        if float((_json(pilots).get("P3") or {}).get("relative_error") or 1.0) >= 1e-4:
            print(f"{language}: STOP_PLUMBING_P3", flush=True)
            continue
        if not (lang / "e2" / "e2_report.json").is_file():
            _run(
                [
                    "-m",
                    "suture.tier_a_e2",
                    "--language",
                    language,
                    "--data-dir",
                    str(lang / "data"),
                    "--host-root",
                    str(lang / "adapters" / "host"),
                    "--donor-root",
                    str(_panel() / "shared" / "donor"),
                    "--output-dir",
                    str(lang / "e2"),
                    "--device",
                    "cuda",
                    "--max-length",
                    "128",
                ]
            )
        e1_gate = lang / "e1" / "e1_gate.json"
        if not e1_gate.is_file():
            _run(
                [
                    "-m",
                    "suture.tier_a_e1_v2",
                    "--data-dir",
                    str(lang / "data"),
                    "--host-dir",
                    str(_selected(host_ck)),
                    "--donor-dir",
                    str(_selected(donor_ck)),
                    "--output-dir",
                    str(lang / "e1"),
                    "--language",
                    language,
                    "--device",
                    "cuda",
                    "--lid-model",
                    str(_lid()),
                ]
            )
        holdout = lang / "holdout" / "mgsm_test_once.json"
        if e1_gate.is_file() and not holdout.is_file():
            _run(
                [
                    "-m",
                    "suture.tier_a_e1_v2",
                    "holdout",
                    "--data-dir",
                    str(lang / "data"),
                    "--host-dir",
                    str(_selected(host_ck)),
                    "--donor-dir",
                    str(_selected(donor_ck)),
                    "--e1-dir",
                    str(lang / "e1"),
                    "--output-dir",
                    str(lang / "holdout"),
                    "--language",
                    language,
                    "--device",
                    "cuda",
                    "--lid-model",
                    str(_lid()),
                ]
            )
    _run(["-m", "suture.tier_a_stats", "--v2", "--root", str(_panel())])
    panel["status"] = "COMPLETED"
    panel["ended_at"] = datetime.now(timezone.utc).isoformat()
    _write_panel(panel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

