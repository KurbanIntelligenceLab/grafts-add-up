"""Isolated powered Spanish free-generation ranking.

Reuses frozen Coder adapters and canonical SUTURE scores. Writes only under
results/reviewer_followup/powered_spanish_ranking/. Canonical E1 is not
rewritten. The eight-item MGSM_dev split is not used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import fasttext
import numpy as np

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from suture.data_manifest import ManifestError, read_manifest
from suture.suture_metrics import GraftScores
from suture.suture_torch import HFResidualAdapter
from suture.tier_a_config import (
    V1_MODEL_ID,
    V1_MODEL_REVISION,
    refuse_legacy_write,
    refuse_v1_write,
    repo_root,
)
from suture.tier_a_gate import (
    _adapter_path,
    _contiguous_windows,
    _evaluate_generation,
    _predicted_for_sets,
    _sha256,
)


CONTRACT = Path("configs/powered_spanish_ranking_v1.json")
N_ITEMS = 64
FORBIDDEN_PREFIXES = (
    Path("results/tier_a"),
    Path("results/v2"),
    Path("results/b3"),
)


def _load_contract() -> Dict[str, Any]:
    return json.loads((repo_root() / CONTRACT).read_text(encoding="utf-8"))


def _frozen_subset(records: Sequence[Mapping[str, Any]], n: int) -> List[Dict[str, Any]]:
    ordered = sorted(
        records,
        key=lambda row: hashlib.sha256(str(row["source_id"]).encode("utf-8")).hexdigest(),
    )
    if n > len(ordered):
        raise ManifestError(f"need {n} MGSM_test records, found {len(ordered)}")
    return [dict(row) for row in ordered[:n]]


def _load_scores(path: Path) -> GraftScores:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return GraftScores(
        utility=np.asarray(payload["utility"], dtype=float),
        risk=np.asarray(payload["risk"], dtype=float),
        injection_norm=np.asarray(payload["injection_norm"], dtype=float),
        n_probe_utility=int(payload["n_probe_utility"]),
        n_probe_risk=int(payload["n_probe_risk"]),
        meta=dict(payload.get("meta", {})),
    )


def _refuse_result_root(path: Path) -> None:
    refuse_v1_write(path)
    refuse_legacy_write(path)
    resolved = path.resolve()
    root = repo_root()
    for rel in FORBIDDEN_PREFIXES:
        frozen = (root / rel).resolve()
        if resolved == frozen or frozen in resolved.parents:
            raise ManifestError(f"refusing write into frozen result tree {frozen}")


def _load_coder_pair(host_dir: Path, donor_dir: Path, device: str):
    host_adapter = _adapter_path(host_dir, "host")
    donor_adapter = _adapter_path(donor_dir, "donor")
    host_cfg = json.loads((host_adapter / "adapter_config.json").read_text(encoding="utf-8"))
    donor_cfg = json.loads((donor_adapter / "adapter_config.json").read_text(encoding="utf-8"))
    if host_cfg.get("base_model_name_or_path") != V1_MODEL_ID:
        raise ManifestError(f"host adapter is not the frozen Coder pin: {host_cfg}")
    if donor_cfg.get("base_model_name_or_path") != V1_MODEL_ID:
        raise ManifestError(f"donor adapter is not the frozen Coder pin: {donor_cfg}")
    target = torch.device(device)
    dtype = torch.bfloat16 if target.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        V1_MODEL_ID,
        revision=V1_MODEL_REVISION,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        V1_MODEL_ID,
        revision=V1_MODEL_REVISION,
        local_files_only=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(target).eval()
    base.config.use_cache = False
    shared = PeftModel.from_pretrained(
        base,
        str(host_adapter),
        adapter_name="host",
        is_trainable=False,
    )
    shared.load_adapter(str(donor_adapter), adapter_name="donor", is_trainable=False)
    shared.set_adapter("host")
    return shared, tokenizer


def _load_resume(path: Path) -> Dict[Tuple[int, int], Dict[str, Any]]:
    rows: Dict[Tuple[int, int], Dict[str, Any]] = {}
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (int(row["start"]), int(row["end"]))
            if key in rows:
                raise ManifestError(f"duplicate window {key} in {path}")
            rows[key] = row
    return rows


def run(*, window_limit: int | None, item_limit: int | None, device: str) -> Dict[str, Any]:
    contract = _load_contract()
    root = repo_root()
    output_dir = root / contract["result_root"]
    _refuse_result_root(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = contract["source_artifacts"]
    selection_path = root / sources["selection_scores"]
    test_path = root / sources["mgsm_test"]
    host_dir = root / sources["host_adapter"]
    donor_dir = root / sources["donor_adapter"]
    lid_path = root / "models" / "lid.176.ftz"
    for path in (selection_path, test_path, host_dir, donor_dir, lid_path):
        if not path.exists():
            raise ManifestError(f"STOP_SETUP missing {path}")

    _, test_records = read_manifest(test_path)
    items = _frozen_subset(test_records, N_ITEMS)
    if item_limit is not None:
        items = items[: int(item_limit)]
    scores = _load_scores(selection_path)
    n_layers = int(len(scores.utility))
    windows = _contiguous_windows(n_layers)
    if window_limit is not None:
        windows = windows[: int(window_limit)]

    config = {
        "contract": contract["contract_name"],
        "n_items": len(items),
        "item_ids": [row["source_id"] for row in items],
        "n_windows": len(windows),
        "selection_scores_sha256": _sha256(selection_path),
        "mgsm_test_sha256": _sha256(test_path),
        "decoding": contract["measurement"]["decoding"],
        "diagnostic": window_limit is not None or item_limit is not None,
    }
    config_path = output_dir / "ranking_config.json"
    if config_path.is_file():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous.get("item_ids") != config["item_ids"] and item_limit is None:
            raise ManifestError("frozen item subset changed; refusing to continue")
    else:
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    shared, tokenizer = _load_coder_pair(host_dir, donor_dir, device)
    adapter = HFResidualAdapter(
        shared,
        shared,
        device=device,
        host_adapter_name="host",
        donor_adapter_name="donor",
    )
    lid_model = fasttext.load_model(str(lid_path))
    predicted = _predicted_for_sets(
        scores,
        [tuple(range(start, end + 1)) for start, end in windows],
    )
    sweep_path = output_dir / "window_sweep.jsonl"
    rows = _load_resume(sweep_path)
    started = time.perf_counter()
    with sweep_path.open("a", encoding="utf-8") as handle:
        for (start, end), prediction in zip(windows, predicted):
            if (start, end) in rows:
                continue
            graft = tuple(range(start, end + 1))
            model = adapter.build_grafted_model(graft)
            generation = _evaluate_generation(
                model,
                tokenizer,
                items,
                language="es",
                lid_model=lid_model,
                device=device,
                batch_size=8,
                max_length=128,
                max_new_tokens=64,
                decode_seeds=[0],
                sample=False,
                temperature=0.0,
                top_p=1.0,
            )
            row = {
                "start": start,
                "end": end,
                "graft": list(graft),
                "predicted_utility": float(prediction),
                "measured_exact_match": float(generation["exact_match"]),
                "n_items": len(items),
                "generation": generation,
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            rows[(start, end)] = row
            done = len(rows)
            if done == 1 or done % 10 == 0 or done == len(windows):
                elapsed = time.perf_counter() - started
                print(f"[powered-ranking] {done}/{len(windows)} windows in {elapsed:.1f}s")

    exact = np.array([rows[key]["measured_exact_match"] for key in windows], dtype=np.float64)
    pred = np.array([rows[key]["predicted_utility"] for key in windows], dtype=np.float64)
    from scipy.stats import spearmanr

    summary = {
        "n_windows": len(windows),
        "n_items": len(items),
        "spearman": float(spearmanr(pred, exact).statistic),
        "selected_window": list(max(windows, key=lambda key: rows[key]["predicted_utility"])),
        "selected_exact_match": float(
            rows[max(windows, key=lambda key: rows[key]["predicted_utility"])][
                "measured_exact_match"
            ]
        ),
        "best_exact_match": float(exact.max()),
        "host_window_not_in_sweep": "host is the empty graft; this sweep is windows only",
        "complete": window_limit is None and item_limit is None and len(rows) == 406,
    }
    (output_dir / "ranking_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window-limit", type=int)
    parser.add_argument("--item-limit", type=int)
    args = parser.parse_args()
    summary = run(
        window_limit=args.window_limit,
        item_limit=args.item_limit,
        device=args.device,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
