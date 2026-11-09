"""Public entry point for v2 B3 generation rescoring."""

from __future__ import annotations

import sys

from suture.b3_generation_scoring_v2 import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
