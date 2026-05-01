from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _serialize(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return {key: _serialize(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _serialize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value


class EventLogger:
    def __init__(self, base_dir: str) -> None:
        self.base_path = Path(base_dir)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.events_path = self.base_path / "events.jsonl"
        self.ledger_path = self.base_path / "trade_ledger.jsonl"

    def log_event(self, event_type: str, **payload: Any) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": event_type,
            "payload": _serialize(payload),
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def log_trade(self, **payload: Any) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "payload": _serialize(payload),
        }
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
