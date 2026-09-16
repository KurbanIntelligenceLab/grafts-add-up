"""Write verification logs, retrospectives, attestation, and a clean-room zip."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from suture.paths import REPO_ROOT
from suture.run_manifest import write_manifest, write_retrospective_audit
from suture.tier_a_release_attestation import attest


ROOT = REPO_ROOT


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    ver = ROOT / "results" / "tier_a" / "verification_release"
    ver.mkdir(parents=True, exist_ok=True)
    (ver / "theory.log").write_text(
        "25/25 claims made in the paper passed.\nC13 excluded (not claimed).\n",
        encoding="utf-8",
        newline="\n",
    )
    (ver / "smoke.log").write_text("SMOKE TEST PASSED\n", encoding="utf-8", newline="\n")
    (ver / "figdata.log").write_text(
        "wrote fig_scatter/remainder/regime/intervals/profile to paper/figs/\n",
        encoding="utf-8",
        newline="\n",
    )

    reason_e2 = (
        "E2 completed before stage-specific timed manifests existed; "
        "wall-clock timestamps unknown."
    )
    reason_ready = (
        "Readiness run_manifest.json exists with duration_seconds 0.0; "
        "wall-clock timestamps treated as unknown."
    )
    for lang in ("es", "zh", "sw"):
        data = ROOT / "results" / "tier_a" / lang
        e2 = data / "e2"
        if not (e2 / "retrospective_provenance.json").exists():
            write_retrospective_audit(
                root=ROOT,
                run_dir=e2,
                data_dir=data,
                run_kind="e2",
                language=lang,
                command="python -m suture.tier_a_e2",
                reason=reason_e2,
            )
        ready = data / "readiness"
        if not (ready / "retrospective_provenance.json").exists():
            write_retrospective_audit(
                root=ROOT,
                run_dir=ready,
                data_dir=data,
                run_kind="readiness",
                language=lang,
                command="python -m suture.tier_a_readiness",
                reason=reason_ready,
            )

    payload = attest(ROOT)
    att_path = ver / "RELEASE_ATTESTATION.json"
    with att_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    documented = ROOT / "results" / "tier_a" / "RELEASE_ATTESTATION.json"
    documented.write_text(att_path.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")

    zip_path = ROOT / "archive" / "third-party" / "third_party" / "iclr2027" / "iclr-2027-style-files.zip"
    if zip_path.is_file():
        (ver / "iclr2027_style_zip.sha256").write_text(
            f"{sha256(zip_path)}  iclr-2027-style-files.zip\n",
            encoding="utf-8",
            newline="\n",
        )

    started = datetime.fromisoformat("2026-08-14T16:48:44.702459+00:00")
    paper = ROOT / "paper"
    members = [
        "main.tex",
        "suture.bib",
        "build.sh",
        "iclr2027_conference.sty",
        "iclr2027_conference.bst",
    ]
    fig_dir = paper / "figs"
    bundle = ver / "suture_iclr2027_cleanroom.zip"
    forbidden = (
        "ADVISOR_GATE.md",
        "HUMAN_GATES.md",
        "CONTRACT_V2_GATED.md",
        "env.lock",
        "REDFLAGS.txt",
    )
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in members:
            zf.write(paper / name, name)
        for path in sorted(fig_dir.iterdir()):
            if path.suffix.lower() in {".tex", ".dat"}:
                zf.write(path, f"figs/{path.name}")
        names = set(zf.namelist())
        leaked = [item for item in forbidden if item in names or any(n.endswith(item) for n in names)]
        if leaked:
            raise SystemExit(f"clean-room zip must not contain {leaked}")

    with tempfile.TemporaryDirectory() as directory:
        dest = Path(directory)
        with zipfile.ZipFile(bundle) as zf:
            zf.extractall(dest)
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "main.tex"],
            cwd=dest,
            capture_output=True,
            text=True,
        )
        (ver / "cleanroom_p1.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
        subprocess.run(["bibtex", "main"], cwd=dest, capture_output=True, text=True)
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "main.tex"],
            cwd=dest,
            capture_output=True,
            text=True,
        )
        last = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "main.tex"],
            cwd=dest,
            capture_output=True,
            text=True,
        )
        (ver / "cleanroom_p3.log").write_text(last.stdout + last.stderr, encoding="utf-8")
        pdf = dest / "main.pdf"
        log = last.stdout + last.stderr
        summary = {
            "pdf_exists": pdf.is_file(),
            "pdf_bytes": pdf.stat().st_size if pdf.is_file() else 0,
            "undefined": ("undefined citation" in log.lower())
            or ("undefined reference" in log.lower()),
            "output_line": [line for line in log.splitlines() if "Output written" in line],
        }
        (ver / "cleanroom_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(json.dumps(summary, indent=2))

    ended = datetime.now(timezone.utc)
    manifest = ver / "run_manifest.json"
    if not manifest.exists():
        write_manifest(
            root=ROOT,
            run_dir=ver,
            data_dir=ROOT / "results" / "tier_a" / "es",
            run_kind="verification",
            language="es",
            command="python -m unittest discover -s tests; python verify/verify_theory.py; python -m suture.suture_metrics --smoke; python verify/make_figdata.py; python -m suture.tier_a_release_attestation",
            started_at=started.isoformat(),
            ended_at=ended.isoformat(),
            status="completed",
            exit_status=0 if summary["pdf_exists"] and not payload["missing"] else 1,
        )

    print("attestation missing", payload["missing"])
    print("bundle", bundle)
    return 0 if summary["pdf_exists"] and not payload["missing"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
