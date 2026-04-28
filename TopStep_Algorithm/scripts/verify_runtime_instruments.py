#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.instruments import known_instruments  # noqa: E402


REQUIRED_INSTRUMENTS = ("6B", "6E", "NQ", "ES", "MNQ", "MES")
PAPER_ONLY_INSTRUMENTS = ("6B", "6E")


def main() -> int:
    instruments = known_instruments()
    defined = sorted(instruments)
    missing_required = [symbol for symbol in REQUIRED_INSTRUMENTS if symbol not in instruments]

    print(f"runtime_root={PROJECT_ROOT}")
    print(f"defined_instruments={','.join(defined)}")
    print(f"required_instruments={','.join(REQUIRED_INSTRUMENTS)}")

    if missing_required:
        print(f"ERROR: missing required runtime instruments: {','.join(missing_required)}", file=sys.stderr)
        return 1

    print("status=ok_runtime_registry")
    print(f"paper_only_instruments={','.join(PAPER_ONLY_INSTRUMENTS)}")
    print("6B_6E_status=paper_enabled_live_blocked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
