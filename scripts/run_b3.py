"""Public entry point for the frozen B3 LightOn/Qwen3-8B protocol."""

from __future__ import annotations

import sys

from suture.b3_lighton import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
