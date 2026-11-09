"""Public entry point for frozen-parser B3 generation rescoring."""

from __future__ import annotations

import sys

from suture.b3_generation_scoring import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
