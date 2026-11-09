"""Public entry point for the isolated B3 free-generation follow-up."""

from __future__ import annotations

import sys

from suture.b3_generation_followup import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
