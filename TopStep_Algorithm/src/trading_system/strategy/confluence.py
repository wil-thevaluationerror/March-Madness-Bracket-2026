from __future__ import annotations

from dataclasses import dataclass

from trading_system.api.market_data import Bar


@dataclass
class ConfluenceResult:
    confluence_type: str  # "OB", "FVG", "OB+FVG", or "NONE"
    description: str      # human-readable detail for logging / diagnostics


def _find_order_block(
    bars: list[Bar],
    direction: str,
    current_price: float,
    atr14: float,
) -> bool:
    """Return True if a valid order block exists near current_price.

    Scans the last 20 bars (most-recent-first) looking for the last
    counter-trend candle before a sequence of impulse candles.

    For BUY:  last bearish candle (close < open) followed by bullish impulse,
              OB body [open, close] must be below current price AND within
              1×ATR14 of it (OB zone overhead, not below entry).
    For SELL: last bullish candle (close > open) followed by bearish impulse,
              OB body [ob_open, ob_close] must be above current price AND
              within 1×ATR14 of it.

    Directional check intentionally does NOT use abs() — the OB must be on
    the correct side of price.
    """
    lookback = bars[-20:] if len(bars) >= 20 else bars
    n = len(lookback)

    for i in range(n - 2, 0, -1):
        bar = lookback[i]
        if direction == "BUY":
            # Bearish OB candle: close < open
            if bar.close >= bar.open:
                continue
            # Followed by at least one bullish candle
            if lookback[i + 1].close <= lookback[i + 1].open:
                continue
            ob_high = bar.open
            ob_low = bar.close
            # OB must be below current price (directional — no abs)
            if ob_high >= current_price:
                continue
            # Within 1×ATR of current price
            if current_price - ob_low > atr14:
                continue
            return True
        else:  # SELL
            # Bullish OB candle: close > open
            if bar.close <= bar.open:
                continue
            # Followed by at least one bearish candle
            if lookback[i + 1].close >= lookback[i + 1].open:
                continue
            ob_low = bar.open
            ob_high = bar.close
            # OB must be above current price (directional — no abs)
            if ob_low <= current_price:
                continue
            # Within 1×ATR of current price
            if ob_high - current_price > atr14:
                continue
            return True

    return False


def _find_fvg(bars: list[Bar], direction: str) -> bool:
    """Return True if a Fair Value Gap exists in the last 20 bars.

    Three-bar pattern, scanning most-recent-first:
    Bullish FVG: bars[i].low > bars[i-2].high  (gap up — price should fill from above)
    Bearish FVG: bars[i].high < bars[i-2].low  (gap down — price should fill from below)
    """
    lookback = bars[-20:] if len(bars) >= 20 else bars
    n = len(lookback)

    for i in range(n - 1, 1, -1):
        if direction == "BUY":
            if lookback[i].low > lookback[i - 2].high:
                return True
        else:  # SELL
            if lookback[i].high < lookback[i - 2].low:
                return True

    return False


def find_confluence(
    bars: list[Bar],
    direction: str,
    current_price: float,
    atr14: float,
) -> ConfluenceResult:
    """Detect Order Block and/or Fair Value Gap confluence near current_price.

    Parameters
    ----------
    bars:
        Recent 5-minute bars (caller should pass the last 20).
    direction:
        "BUY" or "SELL" — gates which OB/FVG patterns are relevant.
    current_price:
        The close of the sweep confirmation bar (entry reference).
    atr14:
        14-bar ATR used for proximity thresholds.

    Returns
    -------
    ConfluenceResult with confluence_type of "OB+FVG", "OB", "FVG", or "NONE".
    """
    has_ob = _find_order_block(bars, direction, current_price, atr14)
    has_fvg = _find_fvg(bars, direction)

    if has_ob and has_fvg:
        return ConfluenceResult(
            confluence_type="OB+FVG",
            description=f"{direction} Order Block + Fair Value Gap near {current_price:.5f}",
        )
    if has_ob:
        return ConfluenceResult(
            confluence_type="OB",
            description=f"{direction} Order Block near {current_price:.5f}",
        )
    if has_fvg:
        return ConfluenceResult(
            confluence_type="FVG",
            description=f"{direction} Fair Value Gap near {current_price:.5f}",
        )
    return ConfluenceResult(confluence_type="NONE", description="No confluence found")
