from __future__ import annotations

from dataclasses import dataclass

from api.market_data import Bar
from strategy.asian_range import AsianRange


@dataclass
class SweepSignal:
    direction: str        # "BUY" (swept Asian low) or "SELL" (swept Asian high)
    level: float          # the Asian high or low that was swept
    candles_remaining: int  # expires after this many bars without confluence


class SweepDetector:
    """Detects Asian-range liquidity sweeps on 5-minute bars.

    A sweep is confirmed when a bar's wick pierces the Asian range boundary
    by at least 0.4×ATR14 but the candle *closes back inside* the range —
    signalling that resting stop orders above/below the range were grabbed
    and price reversed.

    Only one active sweep is tracked at a time.  Call ``update()`` on each
    bar to detect a new sweep, then ``tick()`` to decrement the expiry counter.
    If ``candles_remaining`` reaches zero the signal is cleared.
    """

    def __init__(self) -> None:
        self._active: SweepSignal | None = None

    @property
    def active(self) -> SweepSignal | None:
        return self._active

    def update(self, bar: Bar, asian_range: AsianRange, atr14: float) -> SweepSignal | None:
        """Check *bar* for a liquidity sweep and return a new SweepSignal if found.

        A new signal replaces any existing one on the same bar.
        """
        threshold = 0.4 * atr14

        # Bullish sweep: wick below Asian low, close back inside → expect price to rise.
        bullish = (
            bar.low < asian_range.low - threshold
            and bar.close >= asian_range.low
        )

        # Bearish sweep: wick above Asian high, close back inside → expect price to fall.
        bearish = (
            bar.high > asian_range.high + threshold
            and bar.close <= asian_range.high
        )

        if bullish:
            self._active = SweepSignal(
                direction="BUY",
                level=asian_range.low,
                candles_remaining=3,
            )
        elif bearish:
            self._active = SweepSignal(
                direction="SELL",
                level=asian_range.high,
                candles_remaining=3,
            )

        return self._active if (bullish or bearish) else None

    def tick(self) -> None:
        """Decrement expiry counter; clear the signal if it has expired."""
        if self._active is None:
            return
        self._active.candles_remaining -= 1
        if self._active.candles_remaining <= 0:
            self._active = None

    def clear(self) -> None:
        """Consume (clear) the active sweep — call after a signal is emitted."""
        self._active = None
