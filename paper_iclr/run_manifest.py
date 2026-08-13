"""Write a hash-addressed provenance manifest beside a Tier-A run.

The gate scripts intentionally keep experiment logic separate from provenance
bookkeeping.  This small CLI is run after a pilot or sweep and records the
exact input/output artifacts, source hashes, environment lock, and command
line without inventing a git revision when the local repository has no commit.
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
from typing import Any, Dict, Iterable


MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MODEL_REVISION = "2e1fd397ee46e1388853d2af2c993145b0f1098a"


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
        return str(path.resolve()).replace("\\", "/")


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
    status_text = status.decode("utf-8", errors="replace")
    untracked = [
        line[3:]
        for line in status_text.splitlines()
        if line.startswith("?? ")
    ]
    return {
        "commit": _git_revision(root),
        "dirty": bool(status.strip()),
        "status_sha256": _sha256_bytes(status),
        "diff_sha256": _sha256_bytes(diff),
        "untracked_paths": untracked,
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

    script_paths = [
        root / "paper_iclr" / "run_manifest.py",
        root / "paper_iclr" / "tier_a_data.py",
        root / "paper_iclr" / "tier_a_train.py",
        root / "paper_iclr" / "tier_a_gate.py",
        root / "paper_iclr" / "suture_torch.py",
        root / "paper_iclr" / "suture_metrics.py",
        root / "paper_iclr" / "data_manifest.py",
        root / "env.lock",
    ]
    input_paths = [
        path
        for path in data_dir.iterdir()
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

    pilots = _load_json(run_dir / "pilots.json")
    e1 = _load_json(run_dir / "e1_gate.json")
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
        "seed": seed,
        "command": command,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--run-kind", choices=("pilots", "e1", "data", "training"), required=True)
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
    args = parser.parse_args()
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
