"""Run the reviewer-response Qwen3 plumbing diagnostic.

This is deliberately separate from the frozen v1/v2 experiment entry points.
It tests the implementation at singleton and small graft sizes before any
ranking experiment is considered.  Its output is immutable and lives under
``results/reviewer_followup/qwen3_1_7b``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from suture.suture_metrics import GraftScores, spearman
from suture.suture_torch import HFResidualAdapter
from suture.tier_a_config import (
    MODEL_ID,
    MODEL_REVISION,
    repo_root,
    refuse_frozen_write,
    sha256_file,
)
from suture.tier_a_gate import (
    _load_fake_pair,
    _load_records,
    _load_shared_pair,
    _risk_prompts,
    _utility_prompts,
)


CONTRACT_PATH = Path("configs/reviewer_followup_contract_v1.json")
DEFAULT_OUTPUT = Path("results/reviewer_followup/qwen3_1_7b")


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _graft_probe(n_layers: int) -> List[Tuple[int, ...]]:
    """Fixed, small perturbations that cannot hide a layer-specific bug."""

    representatives = sorted(
        {
            index
            for index in (0, 1, 7, 13, 14, 20, 26, 27)
            if index < n_layers
        }
    )
    singletons = [(index,) for index in representatives]
    adjacent_pairs = [
        (index, index + 1)
        for index in (0, 7, 13, 20, 26)
        if index + 1 < n_layers
    ]
    small_sets = [
        tuple(range(0, min(4, n_layers))),
        tuple(range(max(0, n_layers // 2 - 2), min(n_layers, n_layers // 2 + 2))),
        tuple(range(max(0, n_layers - 4), n_layers)),
    ]
    return [()] + singletons + adjacent_pairs + small_sets


def _mean_readout(
    adapter: HFResidualAdapter,
    prompts: Sequence[Any],
    graft: Tuple[int, ...],
) -> float:
    model = adapter.build_grafted_model(graft)
    values = [model.readout(prompt, "utility") for prompt in prompts]
    return float(np.mean(values))


def _diagnostic(
    adapter: HFResidualAdapter,
    utility_prompts: Sequence[Any],
    risk_prompts: Sequence[Any],
    grafts: Sequence[Tuple[int, ...]],
    *,
    label: str,
    noise_scale: float | None,
) -> Dict[str, Any]:
    started = time.perf_counter()
    scores: GraftScores = adapter.score_selection(utility_prompts, risk_prompts)
    predicted = np.asarray([scores.predict(graft)[0] for graft in grafts], dtype=float)
    host_value = _mean_readout(adapter, utility_prompts, ())
    measured = np.asarray(
        [
            _mean_readout(adapter, utility_prompts, graft) - host_value
            for graft in grafts
        ],
        dtype=float,
    )
    if np.std(predicted) == 0 and np.std(measured) == 0:
        rank = 1.0
    elif np.std(predicted) == 0 or np.std(measured) == 0:
        rank = 0.0
    else:
        rank = float(spearman(predicted, measured))
    max_abs_prediction = float(np.max(np.abs(predicted))) if len(predicted) else 0.0
    max_abs_measured = float(np.max(np.abs(measured))) if len(measured) else 0.0
    zero_ok = bool(
        max_abs_prediction <= 5e-5
        and max_abs_measured <= 5e-5
        and float(np.max(scores.injection_norm)) <= 5e-5
    )
    return {
        "label": label,
        "noise_scale": noise_scale,
        "n_grafts": len(grafts),
        "graft_sizes": sorted({len(graft) for graft in grafts}),
        "spearman": rank,
        "pearson": (
            float(np.corrcoef(predicted, measured)[0, 1])
            if np.std(predicted) > 0 and np.std(measured) > 0
            else (1.0 if np.std(predicted) == 0 and np.std(measured) == 0 else 0.0)
        ),
        "max_abs_predicted": max_abs_prediction,
        "max_abs_measured": max_abs_measured,
        "mean_injection_norm": float(np.mean(scores.injection_norm)),
        "max_injection_norm": float(np.max(scores.injection_norm)),
        "mean_epsilon": float(np.mean([scores.epsilon(graft) for graft in grafts])),
        "zero_donor_invariant": zero_ok,
        "merged_models_built_after_selection": adapter.merged_model_builds,
        "wall_clock_seconds": time.perf_counter() - started,
    }


def _write_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"immutable reviewer-followup artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _release(*objects: Any) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(
    *,
    data_dir: Path,
    host_dir: Path,
    donor_dir: Path,
    output_dir: Path,
    device: str,
    prompt_limit: int = 8,
    max_length: int = 128,
) -> Dict[str, Any]:
    root = repo_root()
    contract_path = root / CONTRACT_PATH
    if not contract_path.is_file():
        raise RuntimeError(f"missing reviewer follow-up contract: {contract_path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract["base_model"]["id"] != MODEL_ID or contract["base_model"]["revision"] != MODEL_REVISION:
        raise RuntimeError("reviewer follow-up contract is not pinned to the v2 model")
    output_dir = output_dir.resolve()
    refuse_frozen_write(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    utility_records = _load_records(data_dir / "P_util.jsonl")[:prompt_limit]
    risk_records = _load_records(data_dir / "P_risk.jsonl")[:prompt_limit]
    if len(utility_records) != prompt_limit or len(risk_records) != prompt_limit:
        raise RuntimeError("reviewer diagnostic did not obtain the requested prompt denominator")

    shared, tokenizer = _load_shared_pair(host_dir, donor_dir, device)
    utility_prompts = _utility_prompts(
        tokenizer, utility_records, batch_size=prompt_limit, max_length=max_length
    )
    risk_prompts = _risk_prompts(
        tokenizer, risk_records, language="es", batch_size=prompt_limit, max_length=max_length
    )

    # The zero-donor check uses the host adapter on both sides.  It should be
    # exact up to bfloat16/log-softmax roundoff and catches routing errors.
    zero_adapter = HFResidualAdapter(
        shared,
        shared,
        device=device,
        host_adapter_name="host",
        donor_adapter_name="host",
    )
    grafts = _graft_probe(zero_adapter.n_layers)
    results: List[Dict[str, Any]] = []
    results.append(
        _diagnostic(
            zero_adapter,
            utility_prompts,
            risk_prompts,
            grafts,
            label="zero_donor",
            noise_scale=0.0,
        )
    )
    _release(zero_adapter, shared, tokenizer)

    for noise_scale in (1e-6, 1e-5, 1e-4):
        fake_shared, fake_tokenizer = _load_fake_pair(
            host_dir, device, noise_scale=noise_scale
        )
        fake_adapter = HFResidualAdapter(
            fake_shared,
            fake_shared,
            device=device,
            host_adapter_name="host",
            donor_adapter_name="fake",
        )
        fake_prompts = _utility_prompts(
            fake_tokenizer, utility_records, batch_size=prompt_limit, max_length=max_length
        )
        fake_risk_prompts = _risk_prompts(
            fake_tokenizer,
            risk_records,
            language="es",
            batch_size=prompt_limit,
            max_length=max_length,
        )
        results.append(
            _diagnostic(
                fake_adapter,
                fake_prompts,
                fake_risk_prompts,
                grafts,
                label="fake_donor",
                noise_scale=noise_scale,
            )
        )
        _release(fake_adapter, fake_shared, fake_tokenizer)

    real_shared, real_tokenizer = _load_shared_pair(host_dir, donor_dir, device)
    real_prompts = _utility_prompts(
        real_tokenizer, utility_records, batch_size=prompt_limit, max_length=max_length
    )
    real_risk_prompts = _risk_prompts(
        real_tokenizer,
        risk_records,
        language="es",
        batch_size=prompt_limit,
        max_length=max_length,
    )
    real_adapter = HFResidualAdapter(
        real_shared,
        real_shared,
        device=device,
        host_adapter_name="host",
        donor_adapter_name="donor",
    )
    results.append(
        _diagnostic(
            real_adapter,
            real_prompts,
            real_risk_prompts,
            grafts,
            label="real_donor",
            noise_scale=None,
        )
    )
    _release(real_adapter, real_shared, real_tokenizer)

    zero = next(row for row in results if row["label"] == "zero_donor")
    fake = [row for row in results if row["label"] == "fake_donor"]
    payload: Dict[str, Any] = {
        "contract_version": contract["contract_version"],
        "contract_sha256": _sha256(contract_path),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "language": "es",
        "prompt_limit": prompt_limit,
        "max_length": max_length,
        "device": device,
        "host_dir": str(host_dir),
        "donor_dir": str(donor_dir),
        "grafts": [list(graft) for graft in grafts],
        "results": results,
        "gate": (
            "STOP_P1_ZERO"
            if not zero["zero_donor_invariant"]
            else (
                "P1_PASS_DIAGNOSTIC_ONLY"
                if any(row["spearman"] >= 0.9 for row in fake)
                else "STOP_P1_FAKE"
            )
        ),
        "ranking_authorized": False,
        "interpretation": (
            "This diagnostic does not authorize E1, MGSM-test, or a new claim "
            "about free-generation ranking."
        ),
    }
    _write_immutable(output_dir / "plumbing_diagnostic.json", payload)
    _write_immutable(
        output_dir / "run_manifest.json",
        {
            "contract": str(CONTRACT_PATH),
            "contract_sha256": _sha256(contract_path),
            "command": "python -m suture.reviewer_followup",
            "data_dir": str(data_dir),
            "host_dir": str(host_dir),
            "donor_dir": str(donor_dir),
            "output_dir": str(output_dir),
            "prompt_limit": prompt_limit,
            "max_length": max_length,
            "canonical_artifacts_modified": False,
        },
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--host-dir", type=Path, required=True)
    parser.add_argument("--donor-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prompt-limit", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()
    report = run(
        data_dir=args.data_dir,
        host_dir=args.host_dir,
        donor_dir=args.donor_dir,
        output_dir=args.output_dir,
        device=args.device,
        prompt_limit=args.prompt_limit,
        max_length=args.max_length,
    )
    print(json.dumps({"gate": report["gate"], "results": report["results"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
