"""Isolated held-out language-readout study for the B3 Qwen3-8B pairs.

This module is deliberately separate from :mod:`suture.b3_lighton`.  It does
not modify the frozen B3 contract, selection manifests, or answer-logprob
measurements.  It adds a held-out language-probe manifest and scores only the
three already-declared candidate grafts: host, selected, and published.

The study is teacher-forced only.  It does not generate text and therefore
does not report a language-identifier rate.  All model loading remains
fail-closed: exact local snapshots, BF16, CUDA, and the frozen 8B contract
are required.

Examples::

    python -m suture.b3_language_fidelity build \
        --language fr \
        --translation-snapshot models/Qwen3-1.7B-70d244cc

    python -m suture.b3_language_fidelity measure \
        --language fr \
        --host-path models/Qwen3-8B-FR \
        --donor-path models/Qwen3-8B-EN
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from suture.b3_lighton import (
    B3Error,
    _host_spec,
    _hardware_report,
    _load_model,
    _load_tokenizer,
    _prompt_from_record,
    _require,
    contract_path,
    load_contract,
    repo_root,
    sha256_file,
    validate_pair_snapshots,
    write_json,
    write_run_manifest,
)
from suture.b3_data import B3Translator, LANGUAGE_LABELS, _load_datasets
from suture.data_manifest import assert_pairwise_disjoint, read_manifest, write_manifest
from suture.tier_a_config import apply_qwen3_chat


HOLDOUT_OFFSET = 500
DEFAULT_N_RECORDS = 250
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 20260918
LANGUAGE_MANIFEST_NAME = "b3_{language}_language_fidelity"


def _risk_rows(contract: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Load and order every unique English risk source record."""

    source = contract["data"]["sources"]["risk"]
    dataset = _load_datasets()(
        source["dataset"],
        source["config"],
        split=source["split"],
        revision=source["revision"],
        download_mode="reuse_dataset_if_exists",
    )
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(dataset):
        subset = str(row.get("subset") or "")
        english = ""
        kind = ""
        if subset.startswith("translation-en-"):
            english = str(row.get("source") or "").strip()
            kind = "source"
        elif subset.startswith("translation-") and len(subset.split("-")) > 2:
            parts = subset.split("-")
            if parts[2] == "en":
                english = str(row.get("chosen") or "").strip()
                kind = "chosen_en"
        if not english or english in seen:
            continue
        seen.add(english)
        rows.append(
            {
                "source_id": f"{source['dataset']}:{source['config']}:{source['split']}:{index}:{kind}",
                "source_index": index,
                "source_subset": subset,
                "source_text": english,
            }
        )
    rows.sort(
        key=lambda row: hashlib.sha256(
            f"b3-risk:{row['source_id']}".encode("utf-8")
        ).hexdigest()
    )
    return rows


def _holdout_rows(
    contract: Mapping[str, Any],
    *,
    offset: int = HOLDOUT_OFFSET,
    n_records: int = DEFAULT_N_RECORDS,
) -> List[Dict[str, Any]]:
    _require(offset >= HOLDOUT_OFFSET, "language holdout cannot precede the frozen 500 risk records")
    _require(n_records > 0, "language holdout must request at least one record")
    rows = _risk_rows(contract)
    _require(
        len(rows) > offset,
        f"risk source has no records after frozen offset {offset}",
    )
    selected = rows[offset : offset + n_records]
    _require(selected, "language holdout selection is empty")
    return selected


def _language_records(
    rows: Sequence[Mapping[str, Any]],
    translations: Sequence[str],
    tokenizer: Any,
    language: str,
    translation_spec: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    label = LANGUAGE_LABELS[language]
    _require(len(rows) == len(translations), "translation count does not match holdout source")
    records: List[Dict[str, Any]] = []
    for row, translated in zip(rows, translations):
        source_id = str(row["source_id"])
        records.append(
            {
                "id": f"b3:{language}:language_fidelity:{source_id}",
                "input_text": apply_qwen3_chat(
                    tokenizer,
                    f"{translated}\nRespond in {label}:",
                    enable_thinking=False,
                ),
                "risk_target_texts": [label],
                "risk_donor_texts": ["English"],
                "language": language,
                "source_id": source_id,
                "source_index": int(row["source_index"]),
                "source_subset": str(row["source_subset"]),
                "source_text_sha256": hashlib.sha256(
                    str(row["source_text"]).encode("utf-8")
                ).hexdigest(),
                "translation_model": translation_spec["id"],
                "translation_revision": translation_spec["revision"],
            }
        )
    return records


def _source_ids(records: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(record.get("source_id")) for record in records}


def _check_source_disjointness(
    language_records: Sequence[Mapping[str, Any]],
    utility_records: Sequence[Mapping[str, Any]],
    risk_records: Sequence[Mapping[str, Any]],
    measurement_records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    holdout_sources = _source_ids(language_records)
    frozen_sources = {
        "P_util": _source_ids(utility_records),
        "P_risk": _source_ids(risk_records),
        "MGSM_rev2_test": _source_ids(measurement_records),
    }
    overlaps = {
        name: sorted(holdout_sources & values)
        for name, values in frozen_sources.items()
        if holdout_sources & values
    }
    _require(not overlaps, f"language holdout overlaps frozen source records: {overlaps}")
    return {
        "holdout": len(holdout_sources),
        **{name: len(values) for name, values in frozen_sources.items()},
        "pairwise_source_disjoint": True,
    }


def _load_measurement_source_records(
    path: Path,
) -> List[Dict[str, Any]]:
    _, records = read_manifest(path)
    return records


def build(
    *,
    language: str,
    output_root: Path,
    translation_snapshot: Path,
    contract_file: Path | None = None,
    device: str = "cuda:0",
    n_records: int = DEFAULT_N_RECORDS,
) -> Dict[str, Any]:
    """Build the disjoint translated language-fidelity manifest."""

    contract_file = contract_file or contract_path()
    contract = load_contract(contract_file)
    _require(language in LANGUAGE_LABELS, f"unsupported B3 language {language!r}")
    translation_spec = contract["data"]["sources"]["translation"]
    pair_root = output_root / f"{language}_host__en_donor"
    fidelity_root = pair_root / "fidelity"
    data_root = fidelity_root / "data"
    target = data_root / "language_holdout.jsonl"
    _require(not fidelity_root.exists(), f"fidelity output already exists: {fidelity_root}")
    _require(translation_snapshot.is_dir(), f"translation snapshot does not exist: {translation_snapshot}")

    frozen_data_root = pair_root / "data"
    utility_records = _load_measurement_source_records(frozen_data_root / "P_util.jsonl")
    risk_records = _load_measurement_source_records(frozen_data_root / "P_risk.jsonl")
    measurement_records = _load_measurement_source_records(
        frozen_data_root / "MGSM_rev2_test.jsonl"
    )
    rows = _holdout_rows(contract, n_records=n_records)
    translator = B3Translator(
        translation_snapshot,
        device=device,
        batch_size=int(translation_spec["batch_size"]),
        max_new_tokens=int(translation_spec["max_new_tokens"]),
    )
    translations = translator.translate(
        [str(row["source_text"]) for row in rows],
        language,
    )
    records = _language_records(
        rows,
        translations,
        translator.tokenizer,
        language,
        translation_spec,
    )
    disjointness = _check_source_disjointness(
        records,
        utility_records,
        risk_records,
        measurement_records,
    )
    assert_pairwise_disjoint({"language_fidelity": records})
    selection_max_length = int(contract["data"]["selection"]["max_length"])
    for record in records:
        token_count = len(
            translator.tokenizer(record["input_text"], add_special_tokens=False)["input_ids"]
        )
        _require(
            token_count <= selection_max_length,
            f"{record['id']} exceeds selection max_length {selection_max_length}",
        )
    metadata = {
        "contract_name": contract["contract_name"],
        "contract_version": contract["contract_version"],
        "language": language,
        "purpose": "held_out_teacher_forced_language_readout",
        "selection_offset": HOLDOUT_OFFSET,
        "selection_max_length": selection_max_length,
        "readout": "log p(target-language first token) - log p(English first token)",
        "source_provenance": contract["data"]["sources"]["risk"],
        "translation_model": translation_spec,
        "frozen_selection_manifests": {
            "P_util": str((frozen_data_root / "P_util.jsonl").resolve()),
            "P_risk": str((frozen_data_root / "P_risk.jsonl").resolve()),
            "MGSM_rev2_test": str((frozen_data_root / "MGSM_rev2_test.jsonl").resolve()),
        },
    }
    summary = write_manifest(
        target,
        records,
        manifest_name=LANGUAGE_MANIFEST_NAME.format(language=language),
        metadata=metadata,
    )
    payload = {
        "schema_version": 1,
        "status": "PASS_LANGUAGE_FIDELITY_DATA",
        "contract_name": contract["contract_name"],
        "language": language,
        "manifest": summary,
        "disjointness": disjointness,
        "translation_snapshot": str(translation_snapshot.resolve()),
        "translation_config_sha256": sha256_file(translation_snapshot / "config.json"),
        "generation_lid": {
            "status": "NOT_RUN",
            "reason": "This isolated study is teacher-forced only.",
        },
    }
    write_json(fidelity_root / "data_build_summary.json", payload)
    write_run_manifest(
        fidelity_root / "data_run_manifest.json",
        stage="language_fidelity_data",
        root=repo_root(),
        contract_file=contract_file,
        command=sys.argv,
        inputs=payload,
    )
    return payload


def _load_fidelity_prompts(
    path: Path,
    tokenizer: Any,
    contract: Mapping[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Any]]:
    header, records = read_manifest(path)
    _require(
        header.get("_manifest")
        == LANGUAGE_MANIFEST_NAME.format(language=header.get("metadata", {}).get("language")),
        "language fidelity manifest name does not match its language",
    )
    metadata = header.get("metadata")
    _require(isinstance(metadata, Mapping), "language fidelity metadata is missing")
    _require(
        metadata.get("contract_name") == contract["contract_name"],
        "language fidelity contract mismatch",
    )
    _require(
        metadata.get("purpose") == "held_out_teacher_forced_language_readout",
        "unexpected language fidelity manifest purpose",
    )
    max_length = int(contract["data"]["selection"]["max_length"])
    prompts = [
        _prompt_from_record(record, "risk", tokenizer, max_length)
        for record in records
    ]
    return header, records, prompts


def _bootstrap_difference(
    difference: Sequence[float],
    *,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    values = np.asarray(list(difference), dtype=float)
    _require(values.size > 1, "paired bootstrap needs at least two probes")
    rng = np.random.default_rng(seed)
    bootstrap = rng.choice(
        values,
        size=(BOOTSTRAP_RESAMPLES, values.size),
        replace=True,
    ).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "fraction_positive": float((values > 0).mean()),
    }


def _candidate_windows(
    contract: Mapping[str, Any],
    selection_path: Path,
    language: str,
) -> List[Dict[str, Any]]:
    selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
    selection = selection_payload.get("selection")
    _require(isinstance(selection, Mapping), "selection artifact lacks selection")
    selected = sorted({int(index) for index in selection.get("graft", [])})
    if selected:
        _require(
            selected == list(range(selected[0], selected[-1] + 1)),
            "selected graft is not a contiguous interval",
        )
    start, end = _host_spec(contract, language)["published_swap"]["window_inclusive"]
    return [
        {"name": "host_baseline", "graft": []},
        {"name": "selected", "graft": selected},
        {"name": "published_reference", "graft": list(range(int(start), int(end) + 1))},
    ]


def _comparison(
    rows: Sequence[Mapping[str, Any]],
    left: str,
    right: str,
) -> Dict[str, Any]:
    by_id: Dict[str, Dict[str, float]] = {}
    for row in rows:
        by_id.setdefault(str(row["record_id"]), {})[str(row["candidate"])] = float(
            row["language_readout"]
        )
    _require(
        all(set(values) == {"host_baseline", "selected", "published_reference"} for values in by_id.values()),
        "fidelity measurements are not complete candidate triples",
    )
    values = np.asarray(
        [by_id[record_id][left] - by_id[record_id][right] for record_id in sorted(by_id)],
        dtype=float,
    )
    result = _bootstrap_difference(values)
    result["n_records"] = int(values.size)
    return result


def measure(
    *,
    language: str,
    host_path: Path,
    donor_path: Path,
    output_root: Path,
    contract_file: Path | None = None,
    probe_path: Path | None = None,
    selection_path: Path | None = None,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    """Score host, selected, and published candidates on the holdout."""

    contract_file = contract_file or contract_path()
    contract = load_contract(contract_file)
    _require(language in LANGUAGE_LABELS, f"unsupported B3 language {language!r}")
    pair_root = output_root / f"{language}_host__en_donor"
    fidelity_root = pair_root / "fidelity"
    measure_root = fidelity_root / "measure"
    _require(not measure_root.exists(), f"fidelity measurement output exists: {measure_root}")
    probe_path = probe_path or fidelity_root / "data" / "language_holdout.jsonl"
    selection_path = selection_path or pair_root / "score" / "selection.json"
    _require(probe_path.is_file(), f"language fidelity manifest is missing: {probe_path}")
    _require(selection_path.is_file(), f"selection artifact is missing: {selection_path}")
    host_spec = _host_spec(contract, language)
    donor_spec = dict(contract["models"]["donor"])
    pair = validate_pair_snapshots(
        host_path,
        donor_path,
        host_spec,
        donor_spec,
        contract,
        hash_files=True,
    )
    hardware = _hardware_report(device, contract)
    tokenizer = _load_tokenizer(host_path, contract)
    header, records, prompts = _load_fidelity_prompts(probe_path, tokenizer, contract)
    candidates = _candidate_windows(contract, selection_path, language)

    host_model = _load_model(host_path, contract)
    donor_model = _load_model(donor_path, contract)
    measured: List[Dict[str, Any]] = []
    try:
        from suture.suture_torch import HFResidualAdapter

        adapter = HFResidualAdapter(host_model, donor_model, device=device)
        for candidate in candidates:
            graft = tuple(int(index) for index in candidate["graft"])
            routed = adapter.build_grafted_model(graft)
            for record, prompt in zip(records, prompts):
                measured.append(
                    {
                        "candidate": candidate["name"],
                        "graft": list(graft),
                        "record_id": record["id"],
                        "language_readout": routed.readout(prompt, "risk"),
                    }
                )
            del routed
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        _require(adapter.merged_model_builds == len(candidates), "candidate build count mismatch")
    except Exception as exc:
        if isinstance(exc, B3Error):
            raise
        raise B3Error(f"held-out language fidelity measurement failed: {exc}") from exc
    finally:
        del host_model, donor_model

    measurements_path = measure_root / "measurements.jsonl"
    measurements_path.parent.mkdir(parents=True, exist_ok=True)
    with measurements_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in measured:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    payload = {
        "schema_version": 1,
        "status": "PASS_LANGUAGE_FIDELITY",
        "contract_name": contract["contract_name"],
        "language": language,
        "objective": "mean teacher-forced language readout",
        "readout": "log p(target-language first token) - log p(English first token)",
        "probe_manifest": str(probe_path.resolve()),
        "probe_manifest_sha256": sha256_file(probe_path),
        "probe_header": header,
        "hardware": hardware,
        "pair": pair,
        "candidates": candidates,
        "measurements_path": str(measurements_path.resolve()),
        "measurements_sha256": sha256_file(measurements_path),
        "measurement_merged_model_builds": len(candidates),
        "comparisons": {
            "selected_minus_host": _comparison(measured, "selected", "host_baseline"),
            "selected_minus_published": _comparison(
                measured,
                "selected",
                "published_reference",
            ),
        },
        "generation_lid": {
            "status": "NOT_RUN",
            "reason": "BF16 cached and uncached greedy decoding was not part of this study.",
        },
    }
    write_json(measure_root / "measure.json", payload)
    write_run_manifest(
        measure_root / "run_manifest.json",
        stage="language_fidelity_measure",
        root=repo_root(),
        contract_file=contract_file,
        command=sys.argv,
        inputs=payload,
    )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "measure"):
        item = subparsers.add_parser(command)
        item.add_argument("--language", choices=tuple(LANGUAGE_LABELS), required=True)
        item.add_argument(
            "--output-root",
            type=Path,
            default=repo_root() / "results" / "b3" / "lighton_qwen3_8b",
        )
        item.add_argument("--contract", type=Path, default=None)
        item.add_argument("--device", default="cuda:0")
    build_parser = subparsers.choices["build"]
    build_parser.add_argument("--translation-snapshot", type=Path, required=True)
    build_parser.add_argument("--n-records", type=int, default=DEFAULT_N_RECORDS)
    measure_parser = subparsers.choices["measure"]
    measure_parser.add_argument("--host-path", type=Path, required=True)
    measure_parser.add_argument("--donor-path", type=Path, required=True)
    measure_parser.add_argument("--probe-path", type=Path, default=None)
    measure_parser.add_argument("--selection-path", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build":
            payload = build(
                language=args.language,
                output_root=args.output_root,
                translation_snapshot=args.translation_snapshot,
                contract_file=args.contract,
                device=args.device,
                n_records=args.n_records,
            )
        else:
            payload = measure(
                language=args.language,
                host_path=args.host_path,
                donor_path=args.donor_path,
                output_root=args.output_root,
                contract_file=args.contract,
                probe_path=args.probe_path,
                selection_path=args.selection_path,
                device=args.device,
            )
    except B3Error as exc:
        print(f"STOP_LANGUAGE_FIDELITY: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
