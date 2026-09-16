"""Write a hash-addressed provenance manifest beside a Tier-A run.

The gate scripts intentionally keep experiment logic separate from provenance
bookkeeping.  This small CLI is run after a pilot or sweep and records the
exact input/output artifacts, source hashes, dependency pins, and command
line, while preserving an explicit null revision if a caller runs outside git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

try:
    from suture.tier_a_config import (
        ACTIVE_CONTRACT_VERSION,
        MODEL_ID,
        MODEL_REVISION,
        refuse_v1_write,
        refuse_frozen_write,
        repo_root,
        sha256_file,
        v2_contract_path,
    )
except ModuleNotFoundError:
    from tier_a_config import (
        ACTIVE_CONTRACT_VERSION,
        MODEL_ID,
        MODEL_REVISION,
        refuse_v1_write,
        refuse_frozen_write,
        repo_root,
        sha256_file,
        v2_contract_path,
    )


RUN_KINDS = (
    "preflight",
    "data",
    "training",
    "checkpoint",
    "readiness",
    "pilots",
    "e2",
    "e1",
    "holdout",
    "analysis",
    "verification",
    "inventory",
    "attestation",
)
REQUIRED_TIMESTAMP_KINDS = {
    "e2",
    "readiness",
    "verification",
    "inventory",
    "attestation",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return f"<external>/{path.name}"


def _portable_command(command: str, root: Path) -> str:
    """Remove checkout-specific prefixes from a recorded command."""

    value = str(command)
    for prefix in (str(root), root.as_posix()):
        value = value.replace(prefix, "<repo>")
    return value.replace("\\", "/")


def _hash_files(paths: Iterable[Path], root: Path) -> Dict[str, str]:
    return {
        _relative(path, root): _sha256(path)
        for path in sorted({path.resolve() for path in paths if path.is_file()})
    }


def _git_revision(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def _git_output(root: Path, *arguments: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=False,
            capture_output=True,
        )
    except OSError:
        return b""
    return result.stdout + result.stderr


def _git_state(root: Path) -> Dict[str, Any]:
    status = _git_output(
        root,
        "status",
        "--short",
        "--untracked-files=all",
    )
    diff = _git_output(root, "diff", "--no-ext-diff", "--binary")
    return {
        "commit": _git_revision(root),
        "dirty": bool(status.strip()),
        "status_sha256": _sha256_bytes(status),
        "diff_sha256": _sha256_bytes(diff),
    }


def _gpu_snapshot() -> list[str]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return []
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _timestamp(value: str | None, *, default: datetime) -> datetime:
    if value is None:
        return default
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def write_manifest(
    *,
    root: Path,
    run_dir: Path,
    data_dir: Path,
    run_kind: str,
    language: str,
    command: str,
    seed: int | None = None,
    status: str = "completed",
    exit_status: int = 0,
    started_at: str | None = None,
    ended_at: str | None = None,
    run_id: str | None = None,
) -> Path:
    if status not in {"completed", "failed", "partial"}:
        raise ValueError(f"unsupported run status: {status}")
    if run_kind not in RUN_KINDS:
        raise ValueError(f"unsupported run kind: {run_kind}")
    if run_kind in REQUIRED_TIMESTAMP_KINDS and (started_at is None or ended_at is None):
        raise ValueError(
            f"{run_kind} manifests require explicit started_at and ended_at; "
            "do not invent timestamps. Use write_retrospective_audit() when they are unknown."
        )
    output = run_dir / "run_manifest.json"
    if output.exists():
        raise FileExistsError(
            f"immutable run manifest already exists: {output}"
        )
    created = datetime.now(timezone.utc)
    started = _timestamp(started_at, default=created)
    ended = _timestamp(ended_at, default=created)
    if ended < started:
        raise ValueError("ended_at must not precede started_at")
    refuse_frozen_write(output, root=root)

    script_paths = [
        root / "src" / "suture" / "tier_a_config.py",
        root / "src" / "suture" / "tier_a_preflight.py",
        root / "src" / "suture" / "run_manifest.py",
        root / "src" / "suture" / "tier_a_data.py",
        root / "src" / "suture" / "tier_a_train.py",
        root / "src" / "suture" / "tier_a_gate.py",
        root / "src" / "suture" / "tier_a_e2.py",
        root / "src" / "suture" / "tier_a_stats.py",
        root / "src" / "suture" / "tier_a_readiness.py",
        root / "src" / "suture" / "tier_a_readiness_data.py",
        root / "src" / "suture" / "suture_torch.py",
        root / "src" / "suture" / "suture_metrics.py",
        root / "src" / "suture" / "data_manifest.py",
        root / "configs" / "experimental_contract_v1.json",
        root / "configs" / "experimental_contract_v2.json",
        root / "requirements-gpu.txt",
        root / "models" / "lid.176.ftz",
    ]
    input_paths = [
        path
        for path in list(data_dir.iterdir())
        + list((data_dir / "readiness").rglob("*"))
        if (
            path.is_file()
            and path.suffix in {".jsonl", ".json"}
            and not path.name.startswith("run_manifest")
        )
    ]
    output_paths = [
        path
        for path in run_dir.rglob("*")
        if path.is_file() and not path.name.startswith("run_manifest")
    ]

    pilots_path = run_dir / "pilots.json"
    e1_path = run_dir / "e1_gate.json"
    pilots = _load_json(pilots_path) if pilots_path.is_file() else {}
    e1 = _load_json(e1_path) if e1_path.is_file() else {}
    git = _git_state(root)
    payload: Dict[str, Any] = {
        "schema_version": 2,
        "run_id": run_id or f"{run_kind}-{language}-{started:%Y%m%dT%H%M%SZ}",
        "run_kind": run_kind,
        "stage": run_kind,
        "language": language,
        "status": status,
        "exit_status": int(exit_status),
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "duration_seconds": (ended - started).total_seconds(),
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
        },
        "contract_version": ACTIVE_CONTRACT_VERSION,
        "contract_sha256": (
            sha256_file(v2_contract_path(root))
            if v2_contract_path(root).is_file()
            else None
        ),
        "seed": seed,
        "command": _portable_command(command, root),
        "python": sys.version,
        "platform": platform.platform(),
        "hardware": {
            "gpu": _gpu_snapshot(),
        },
        "git": git,
        # Keep this flat field for consumers of the original schema.
        "git_commit": git["commit"],
        "selection_contract": {
            "merged_models_built_during_selection": int(
                e1.get("selection_merged_models_built", 0)
            ),
            "e1_sweep_is_exhaustive": e1.get("sweep_is_exhaustive"),
            "e1_merged_models_built": e1.get("merged_models_built"),
            "p1_p2_results": {
                name: pilots.get(name, {}).get("merged_models_built_after_selection")
                for name in ("P1", "P2")
                if name in pilots
            },
        },
        "files": {
            "scripts_and_environment": _hash_files(script_paths, root),
            "inputs": _hash_files(input_paths, root),
            "outputs": _hash_files(output_paths, root),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    # The explicit existence check above makes accidental replacement fail.
    os.replace(temporary, output)
    return output


def write_retrospective_audit(
    *,
    root: Path,
    run_dir: Path,
    data_dir: Path,
    run_kind: str,
    language: str,
    command: str,
    reason: str,
) -> Path:
    """Record hashes for a completed stage whose original timestamps are unknown.

    This never writes ``run_manifest.json``, so it cannot impersonate a timed run.
    """

    if run_kind not in RUN_KINDS:
        raise ValueError(f"unsupported run kind: {run_kind}")
    output = run_dir / "retrospective_provenance.json"
    if output.exists():
        raise FileExistsError(f"immutable retrospective audit already exists: {output}")
    created = datetime.now(timezone.utc)
    payload = {
        "schema_version": 1,
        "kind": "retrospective_provenance",
        "run_kind": run_kind,
        "stage": run_kind,
        "language": language,
        "command": _portable_command(command, root),
        "reason": reason,
        "started_at": None,
        "ended_at": None,
        "duration_seconds": None,
        "timestamp_status": "unknown",
        "recorded_at": created.isoformat(),
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "git": _git_state(root),
        "files": {
            "scripts_and_environment": _hash_files(
                [
                    root / "src" / "suture" / "run_manifest.py",
                    root / "src" / "suture" / "tier_a_e2.py",
                    root / "src" / "suture" / "tier_a_readiness.py",
                    root / "src" / "suture" / "tier_a_train.py",
                    root / "configs" / "experimental_contract_v1.json",
                    root / "requirements-gpu.txt",
                ],
                root,
            ),
            "inputs": _hash_files(
                [
                    path
                    for path in list(data_dir.iterdir())
                    if path.is_file() and path.suffix in {".jsonl", ".json"}
                ],
                root,
            ),
            "outputs": _hash_files(
                [
                    path
                    for path in run_dir.rglob("*")
                    if path.is_file()
                    and path.name not in {"run_manifest.json", "retrospective_provenance.json"}
                    and path.suffix.lower() in {".json", ".jsonl", ".md"}
                ],
                root,
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, output)
    return output


def assert_resume_compatible(existing: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Refuse to resume a run whose pin, contract, or code hashes drifted."""

    checks = (
        ("model.id", existing.get("model", {}).get("id"), expected.get("model_id")),
        ("model.revision", existing.get("model", {}).get("revision"), expected.get("model_revision")),
        ("contract_sha256", existing.get("contract_sha256"), expected.get("contract_sha256")),
        ("git_commit", existing.get("git_commit"), expected.get("git_commit")),
    )
    for name, left, right in checks:
        if right is None:
            continue
        if left != right:
            raise ValueError(f"refuse resume: {name} {left!r} != {right!r}")


assert_resume_compatible = assert_resume_compatible


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--run-kind", choices=RUN_KINDS, required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--status",
        choices=("completed", "failed", "partial"),
        default="completed",
    )
    parser.add_argument("--exit-status", type=int, default=0)
    parser.add_argument("--started-at")
    parser.add_argument("--ended-at")
    parser.add_argument("--run-id")
    parser.add_argument("--retrospective", action="store_true")
    parser.add_argument("--reason", default="")
    args = parser.parse_args()
    if args.retrospective:
        if not args.reason:
            raise SystemExit("retrospective audits require --reason")
        output = write_retrospective_audit(
            root=Path(args.root).resolve(),
            run_dir=Path(args.run_dir).resolve(),
            data_dir=Path(args.data_dir).resolve(),
            run_kind=args.run_kind,
            language=args.language,
            command=args.command,
            reason=args.reason,
        )
    else:
        output = write_manifest(
            root=Path(args.root).resolve(),
            run_dir=Path(args.run_dir).resolve(),
            data_dir=Path(args.data_dir).resolve(),
            run_kind=args.run_kind,
            language=args.language,
            command=args.command,
            seed=args.seed,
            status=args.status,
            exit_status=args.exit_status,
            started_at=args.started_at,
            ended_at=args.ended_at,
            run_id=args.run_id,
        )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
