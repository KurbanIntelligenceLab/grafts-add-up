"""Repository-relative paths shared by the SUTURE pipeline.

The package is installed from ``src/`` but all experiment artifacts live at
the repository root.  Keeping this resolution in one module prevents the
pipeline from depending on the caller's current working directory and avoids
machine-specific absolute paths.
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
PAPER_DIR = REPO_ROOT / "paper"
RESULTS_DIR = REPO_ROOT / "results"
MODELS_DIR = REPO_ROOT / "models"


def repo_path(*parts: str) -> Path:
    """Return a path rooted at the repository checkout."""

    return REPO_ROOT.joinpath(*parts)
