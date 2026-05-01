from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class Bar:
    timestamp: datetime  # UTC, timezone-aware
    open: float
    high: float
    low: float
    close: float
    volume: int
