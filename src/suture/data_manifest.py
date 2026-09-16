"""Hash-addressed data manifests for the SUTURE Tier-A protocol.

The experiment contract separates selection probes, certificate calibration,
development evaluation, and untouched test data.  This module makes that
separation executable: every canonical record receives a SHA-256 digest, every
manifest carries source metadata, and pairwise disjointness fails closed when a
hash is missing or repeated.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


class ManifestError(RuntimeError):
    """Raised for malformed or overlapping experimental data manifests."""


def canonical_record(record: Mapping[str, Any]) -> str:
    """Return the stable UTF-8 representation used for hashing."""

    if not isinstance(record, Mapping):
        raise ManifestError(f"manifest record must be a mapping, got {type(record)!r}")
    clean = {str(key): value for key, value in record.items() if key != "_hash"}
    try:
        return json.dumps(
            clean,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"record is not JSON-canonicalizable: {clean!r}") from exc


def record_hash(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_record(record).encode("utf-8")).hexdigest()


def _validate_records(records: Sequence[Mapping[str, Any]], name: str) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    hashes = set()
    for index, record in enumerate(records):
        item = dict(record)
        digest = record_hash(item)
        if item.get("_hash") not in (None, digest):
            raise ManifestError(f"{name}[{index}] contains an incorrect _hash")
        if digest in hashes:
            raise ManifestError(f"{name} contains duplicate canonical records at index {index}")
        item["_hash"] = digest
        hashes.add(digest)
        output.append(item)
    if not output:
        raise ManifestError(f"{name} is empty")
    return output


def write_manifest(
    path: str | Path,
    records: Sequence[Mapping[str, Any]],
    *,
    manifest_name: str,
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Write a UTF-8 JSONL manifest with a self-describing header."""

    validated = _validate_records(records, manifest_name)
    header = {
        "_manifest": manifest_name,
        "schema_version": 1,
        "n_records": len(validated),
        "metadata": dict(metadata or {}),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(header, ensure_ascii=False, sort_keys=True) + "\n")
        for item in validated:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "path": str(target),
        "manifest_name": manifest_name,
        "n_records": len(validated),
        "record_hashes": [item["_hash"] for item in validated],
        "file_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "metadata": dict(metadata or {}),
    }


def read_manifest(path: str | Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    target = Path(path)
    if not target.exists():
        raise ManifestError(f"manifest not found: {target}")
    with target.open("r", encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle if line.strip()]
    if not lines or "_manifest" not in lines[0]:
        raise ManifestError(f"{target} is missing its manifest header")
    header = dict(lines[0])
    records = _validate_records(lines[1:], str(target))
    if header.get("n_records") != len(records):
        raise ManifestError(
            f"{target} header says {header.get('n_records')} records but contains {len(records)}"
        )
    return header, records


def assert_pairwise_disjoint(manifests: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    """Require nonempty, hashed, pairwise-disjoint canonical records."""

    hashes: Dict[str, set[str]] = {}
    for name, records in manifests.items():
        validated = _validate_records(records, name)
        hashes[name] = {item["_hash"] for item in validated}
    collisions = []
    names = list(hashes)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = sorted(hashes[left] & hashes[right])
            if overlap:
                collisions.append({"left": left, "right": right, "hashes": overlap})
    if collisions:
        raise ManifestError("manifest hash overlap: " + json.dumps(collisions, sort_keys=True))
    return {
        "manifests": {name: len(values) for name, values in hashes.items()},
        "pairwise_disjoint": True,
        "hash_algorithm": "sha256",
    }


def summarize_manifest_set(paths: Mapping[str, str | Path]) -> Dict[str, Any]:
    loaded = {}
    file_summaries = {}
    for name, path in paths.items():
        header, records = read_manifest(path)
        loaded[name] = records
        file_summaries[name] = {
            "path": str(path),
            "header": header,
            "file_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        }
    disjoint = assert_pairwise_disjoint(loaded)
    return {"files": file_summaries, "disjointness": disjoint}


__all__ = [
    "ManifestError",
    "assert_pairwise_disjoint",
    "canonical_record",
    "read_manifest",
    "record_hash",
    "summarize_manifest_set",
    "write_manifest",
]
