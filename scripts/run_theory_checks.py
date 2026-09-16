"""Run the offline checks used by the paper's theory and validation sections."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip unittest discovery when only the numerical checks are needed.",
    )
    args = parser.parse_args()

    commands = []
    if not args.skip_tests:
        commands.append([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
    commands.extend(
        [
            [sys.executable, "verify/verify_theory.py"],
            [sys.executable, "verify/power_analysis.py"],
            [sys.executable, "verify/independent_check.py"],
            [sys.executable, "verify/corrections_check.py"],
            [sys.executable, "-m", "suture.suture_metrics", "--smoke"],
            [sys.executable, "-m", "suture.b3_lighton", "validate-contract"],
        ]
    )
    for command in commands:
        print("+", " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
