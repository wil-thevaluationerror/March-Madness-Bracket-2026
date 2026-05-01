from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

from api.market_data import Bar

UTC = ZoneInfo("UTC")
MIN_ASIAN_RANGE_BARS = 72


@dataclass(slots=True)
class AsianRange:
    high: float
    low: float

    @property
    def midpoint(self) -> float:
        return (self.high + self.low) / 2


def _atr14(bars: list[Bar]) -> float:
    """Simple ATR-14 over the provided bars."""
    if len(bars) < 2:
        return 0.0
    trs: list[float] = []
    for i in range(1, len(bars)):
        tr = max(
            bars[i].high - bars[i].low,
            abs(bars[i].high - bars[i - 1].close),
            abs(bars[i].low - bars[i - 1].close),
        )
        trs.append(tr)
    window = trs[-14:] if len(trs) >= 14 else trs
    return sum(window) / len(window)


def compute_asian_range(bars: list[Bar], session_date: date) -> AsianRange | None:
    """Build the Asian-session high/low for a given London-open session date.

    Window: 01:00–07:00 UTC on *session_date* (equivalent to 20:00–02:00 ET
    the prior ET evening → morning of the London open).

    Returns None if:
    - Fewer than 72 5-minute bars fall in the window (insufficient data).
    - The resulting range is narrower than 0.5 × ATR-14 of all provided bars
      (range too compressed to be a meaningful liquidity pool).
    """
    start = datetime(session_date.year, session_date.month, session_date.day, 1, 0, tzinfo=UTC)
    end = datetime(session_date.year, session_date.month, session_date.day, 7, 0, tzinfo=UTC)

    window_bars = [b for b in bars if start <= b.timestamp < end]
    if len(window_bars) < MIN_ASIAN_RANGE_BARS:
        return None

    high = max(b.high for b in window_bars)
    low = min(b.low for b in window_bars)

    atr = _atr14(bars)
    if atr > 0 and (high - low) < 0.5 * atr:
        return None

    return AsianRange(high=high, low=low)
