#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.instruments import known_instruments  # noqa: E402


CURRENTLY_SUPPORTED = ("ES", "MES", "MNQ", "NQ")
INTENDED_NOT_IMPLEMENTED = ("6B", "6E")


def main() -> int:
    instruments = known_instruments()
    defined = sorted(instruments)
    missing_supported = [symbol for symbol in CURRENTLY_SUPPORTED if symbol not in instruments]
    unexpectedly_supported = [symbol for symbol in INTENDED_NOT_IMPLEMENTED if symbol in instruments]

    print(f"runtime_root={PROJECT_ROOT}")
    print(f"defined_instruments={','.join(defined)}")
    print(f"required_current_instruments={','.join(CURRENTLY_SUPPORTED)}")

    if missing_supported:
        print(f"ERROR: missing current runtime instruments: {','.join(missing_supported)}", file=sys.stderr)
        return 1

    if unexpectedly_supported:
        print(
            "ERROR: 6B/6E appeared in the registry, but FX execution/backtest support "
            "has not been verified by this script.",
            file=sys.stderr,
        )
        print(f"unexpected_instruments={','.join(unexpectedly_supported)}", file=sys.stderr)
        return 1

    print("status=ok_current_runtime_registry")
    print("6B_6E_status=not_supported_in_active_runtime")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
