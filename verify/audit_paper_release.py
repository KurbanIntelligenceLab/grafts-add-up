"""Check that manuscript file references belong to the public release.

The release manifest records staged Git content. This audit checks local TeX
inputs, plot data, bibliography files, and repository artifacts named in the
manuscript against that inventory. External model and dataset identifiers are
outside its scope.
"""

from __future__ import annotations

import fnmatch
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = {
    entry["path"]
    for entry in json.loads((ROOT / "manifests" / "submission_manifest.json").read_text(encoding="utf-8"))["files"]
}
REPO_PREFIXES = ("configs/", "paper/", "results/", "scripts/", "src/", "verify/")
FILE_SUFFIXES = (".py", ".json", ".jsonl", ".dat", ".tex", ".bib", ".bst", ".sty")


def check_public(relative: str) -> None:
    assert relative in PUBLIC, f"Missing from public release manifest: {relative}"
    assert (ROOT / relative).is_file(), f"Missing from checkout: {relative}"


def tex_sources() -> dict[str, str]:
    pending = ["paper/main.tex"]
    sources: dict[str, str] = {}
    while pending:
        relative = pending.pop()
        if relative in sources:
            continue
        check_public(relative)
        content = (ROOT / relative).read_text(encoding="utf-8")
        sources[relative] = content
        for name in re.findall(r"\\(?:input|include)\{([^{}]+)\}", content):
            if "..." in name:  # explanatory comment, not a TeX dependency
                continue
            target = f"paper/{name}{'' if name.endswith('.tex') else '.tex'}"
            pending.append(target)
    return sources


def artifact_matches(value: str) -> list[str]:
    value = value.replace(r"\_", "_")
    if value.startswith(REPO_PREFIXES):
        if "*" in value:
            return sorted(path for path in PUBLIC if fnmatch.fnmatchcase(path, value))
        if value.endswith("/"):
            return sorted(path for path in PUBLIC if path.startswith(value))
        return [value] if value in PUBLIC else []
    if value.endswith(FILE_SUFFIXES):
        return sorted(path for path in PUBLIC if Path(path).name == value)
    return []  # model IDs, dataset splits, revision hashes, or code variables


def main() -> None:
    sources = tex_sources()
    checks: set[str] = set(sources)
    checks.update(("paper/iclr2027_conference.sty", "paper/iclr2027_conference.bst",
                   "paper/suture.bib"))
    for content in sources.values():
        checks.update(f"paper/{name}" for name in re.findall(r"figs/[A-Za-z0-9_.-]+\.dat", content))
        checks.update(f"paper/{name}" for name in re.findall(r"figs/[A-Za-z0-9_.-]+\.tex", content))
    for relative in sorted(checks):
        check_public(relative)

    manuscript = sources["paper/main.tex"]
    artifacts = re.findall(r"\\artifact\{([^{}]+)\}", manuscript)
    checked_artifacts = 0
    for value in artifacts:
        normalized = value.replace(r"\_", "_")
        if normalized.startswith(REPO_PREFIXES) or normalized.endswith(FILE_SUFFIXES):
            matches = artifact_matches(value)
            assert matches, f"Manuscript artifact has no public match: {normalized}"
            for relative in matches:
                check_public(relative)
            checked_artifacts += 1
    print(f"PASS {len(checks)} TeX inputs and plot files; {checked_artifacts} repository artifact references")


if __name__ == "__main__":
    main()
