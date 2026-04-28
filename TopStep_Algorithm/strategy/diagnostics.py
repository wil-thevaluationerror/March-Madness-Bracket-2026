from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


class StrategyDiagnosticsLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, payload: dict[str, Any]) -> None:
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "event": "strategy_bar_diagnostic",
            **payload,
        }
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        except Exception as exc:
            _log.warning("strategy_diagnostics_write_failed path=%s reason=%s", self.path, exc)
