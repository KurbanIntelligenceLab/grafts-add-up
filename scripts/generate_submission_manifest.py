"""Generate the machine-readable public-file inventory.

The manifest is intentionally derived from the release tree rather than from
Git's index.  This lets reviewers inspect the exact file-level rationale before
the first commit is created and keeps ignored local weights out of the list.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "submission_manifest.json"
SKIP_DIRECTORIES = {
    ".git",
    ".cursor",
    ".pytest_cache",
    "__pycache__",
    "archive",
    "build",
    "models",
}
SKIP_NAMES = {".env", "submission_manifest.json"}
SKIP_SUFFIXES = {
    ".aux",
    ".bbl",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
    ".pdf",
    ".synctex.gz",
    ".toc",
    ".safetensors",
    ".bin",
    ".pt",
    ".ckpt",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _included(path: Path) -> bool:
    relative_parts = path.relative_to(ROOT).parts
    if any(part in SKIP_DIRECTORIES for part in relative_parts[:-1]):
        return False
    if path.name in SKIP_NAMES or path.name == "tokenizer.json":
        return False
    if any(path.name.endswith(suffix) for suffix in SKIP_SUFFIXES):
        # Small verification and run logs under results/ are part of the
        # provenance record; LaTeX/build logs elsewhere are generated output.
        if relative_parts[:1] != ("results",):
            return False
    if relative_parts[:1] == ("paper",):
        return path.name in {
            "main.tex",
            "suture.bib",
            "build.sh",
            "iclr2027_conference.sty",
            "iclr2027_conference.bst",
        } or len(relative_parts) >= 3 and relative_parts[1] == "figs" and path.suffix in {
            ".dat",
            ".tex",
        }
    return True


def _rationale(relative: str) -> str:
    if relative == "README.md":
        return "Reviewer-facing overview, setup, reproduction workflow, and claim boundaries."
    if relative == "LICENSE":
        return "Software license for the released code and documentation."
    if relative == "CITATION.cff":
        return "Machine-readable citation metadata for the reproducibility package."
    if relative in {"requirements.txt", "requirements-gpu.txt", "pyproject.toml", ".env.example", ".gitignore"}:
        return "Root project metadata or dependency configuration required for a clean checkout."
    if relative.startswith("paper/"):
        return "Manuscript source or compile input consumed by the ICLR paper build."
    if relative.startswith("src/suture/"):
        return "Reusable experiment implementation, scoring adapter, provenance, or contract enforcement code."
    if relative.startswith("verify/"):
        return "Theory, figure-data, power, clean-room, or negative-control verification harness cited by the paper."
    if relative.startswith("tests/"):
        return "Offline regression test for a released data, scoring, provenance, or fail-closed guard."
    if relative.startswith("scripts/"):
        return "Reviewer-facing command-line entry point for the reproducibility workflow."
    if relative.startswith("configs/"):
        return "Frozen experiment contract or protocol pin required to interpret and rerun the corresponding study lane."
    if relative.startswith("results/b3/"):
        return "Canonical B3 LightOn/Qwen3-8B selection and held-out measurement artifact reported in the paper."
    if relative.startswith("results/tier_a/"):
        return "Canonical Tier-A evidence, source data, audit, attestation, or verification output used by the paper."
    if relative.startswith("results/v2/"):
        return "Frozen Qwen3-1.7B readiness, data, or stopped-panel artifact documenting the scoped follow-up."
    if relative.startswith("results/reviewer_followup/"):
        return "Frozen reviewer-response diagnostic or powered Spanish ranking artifact referenced by the paper."
    if relative.startswith("docs/"):
        return "Public protocol, artifact inventory, project explanation, reproducibility, or paper-build documentation."
    if relative.startswith("manifests/"):
        return "Machine-readable release inventory; this file records the rationale for every other included file."
    return "Public repository metadata required to reproduce or audit the submission."


def _iter_files() -> Iterable[Path]:
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and _included(path):
            yield path


def build_manifest() -> dict:
    files = []
    for path in _iter_files():
        relative = path.relative_to(ROOT).as_posix()
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "rationale": _rationale(relative),
            }
        )
    files.append(
        {
            "path": "manifests/submission_manifest.json",
            "bytes": None,
            "sha256": None,
            "rationale": _rationale("manifests/submission_manifest.json"),
            "note": "Self-entry is intentionally unhashed because the manifest contains its own inventory.",
        }
    )
    files.sort(key=lambda item: item["path"])
    return {
        "schema_version": 1,
        "kind": "public_submission_manifest",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": "repository checkout",
        "file_count": len(files),
        "weights_included": False,
        "files": files,
        "excluded_classes": {
            "local_checkpoints": "Model weights, adapter binaries, tokenizer copies, and local Hugging Face snapshots.",
            "generated_outputs": "LaTeX intermediates, PDFs, build logs, caches, and bytecode.",
            "internal_process": "Author-only review, advisor, novelty, and submission-gate notes.",
            "superseded_runs": "Smoke, preliminary, abandoned, and superseded result directories.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=MANIFEST)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_manifest()
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output} ({payload['file_count']} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
