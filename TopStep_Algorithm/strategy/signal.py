from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from api.market_data import Bar
from backtest.raw_setup_ledger import RawSetupEvent
from config import INSTRUMENTS
from strategy.asian_range import AsianRange, compute_asian_range
from strategy.confluence import ConfluenceResult, find_confluence
from strategy.sweep_detector import SweepDetector


@dataclass
class SignalConfig:
    """Strategy parameters sourced from the active trading profile.

    All fields mirror the corresponding ``StrategyConfig`` attributes in
    ``backtest/config.py``.  Defaults reproduce the previous hard-coded
    behaviour so existing callers that omit the config are unaffected.
    """

    # TP1 = entry ± target_atr_multiple × ATR14
    # TP2 = entry ± (target_atr_multiple + 1) × ATR14
    target_atr_multiple: float = 2.0

    # Move SL to break-even after this many ATRs of profit (0.0 = disabled).
    breakeven_trigger_atr: float = 0.0

    # Minimum ADX-14 value required for entry (0.0 = disabled).
    adx_min_threshold: float = 0.0

    # Current ATR must be ≥ this fraction of the 40-bar rolling mean ATR
    # (0.0 = disabled; 0.85 = skip low-volatility bars).
    atr_min_pct: float = 0.0

    # Require this many consecutive aligned EMA slopes before entry
    # (0 = single-bar check only).
    ema_trend_persistence_bars: int = 0

    # Permitted confluence types.  Signals whose confluence_type is not in this set
    # are discarded before building the bracket.  Default allows all types.
    # Set to frozenset({"FVG", "OB+FVG"}) to exclude the OB-only cluster, which
    # shows 17.6% WR in OOS data (-$2,206 total).
    allowed_confluence_types: frozenset[str] = frozenset({"OB", "FVG", "OB+FVG"})


@dataclass(slots=True)
class TradeSignal:
    symbol: str
    direction: str       # "BUY" or "SELL"
    entry_price: float
    stop_price: float
    tp1_price: float
    tp2_price: float
    confluence: ConfluenceResult
    atr14: float
    ema_slope: float
    setup_type: str = "asian_range_sweep"
    adx_at_entry: float = 0.0
    vwap_distance_at_entry: float = 0.0
    range_width_atr: float = 0.0
    sweep_depth_atr: float = 0.0
    distance_to_asian_mid: float = 0.0
    time_since_sweep: int = 0
    raw_setup_event_id: str = ""


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------

def _ema(bars: list[Bar], period: int) -> list[float]:
    """Compute EMA over bar closes.

    Seeds with SMA of the first *period* closes, then applies the standard
    EMA multiplier.  Returns an empty list if fewer than *period* bars are
    provided.

    The result has length ``max(0, len(bars) - period + 1)``:
    - result[0]  = SMA(first period closes)
    - result[1]  = EMA of bar[period]
    - ...
    """
    if len(bars) < period:
        return []
    closes = [b.close for b in bars]
    k = 2.0 / (period + 1)
    sma = sum(closes[:period]) / period
    result: list[float] = [sma]
    for close in closes[period:]:
        result.append(close * k + result[-1] * (1.0 - k))
    return result


def _round_to_tick(price: float, tick_size: float) -> float:
    """Round *price* to the nearest *tick_size*."""
    return round(round(price / tick_size) * tick_size, 10)


def _compute_atr14(bars: list[Bar]) -> float:
    """Simple 14-bar ATR (simple average of true ranges)."""
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


def _compute_adx14(bars: list[Bar], period: int = 14) -> float:
    """Compute ADX-14 using Wilder smoothing.

    Requires at least ``2 * period + 1`` bars for a meaningful result;
    returns 0.0 if there is insufficient data.
    """
    if len(bars) < period * 2 + 1:
        return 0.0

    trs: list[float] = []
    plus_dms: list[float] = []
    minus_dms: list[float] = []

    for i in range(1, len(bars)):
        up_move = bars[i].high - bars[i - 1].high
        down_move = bars[i - 1].low - bars[i].low
        plus_dms.append(up_move if up_move > down_move and up_move > 0 else 0.0)
        minus_dms.append(down_move if down_move > up_move and down_move > 0 else 0.0)
        trs.append(max(
            bars[i].high - bars[i].low,
            abs(bars[i].high - bars[i - 1].close),
            abs(bars[i].low - bars[i - 1].close),
        ))

    def _wilder(values: list[float], p: int) -> list[float]:
        if len(values) < p:
            return []
        smoothed = [sum(values[:p])]
        for v in values[p:]:
            smoothed.append(smoothed[-1] - smoothed[-1] / p + v)
        return smoothed

    atr_s = _wilder(trs, period)
    plus_s = _wilder(plus_dms, period)
    minus_s = _wilder(minus_dms, period)

    n = min(len(atr_s), len(plus_s), len(minus_s))
    dx_values: list[float] = []
    for k in range(n):
        a = atr_s[k]
        if a == 0.0:
            dx_values.append(0.0)
            continue
        plus_di = 100.0 * plus_s[k] / a
        minus_di = 100.0 * minus_s[k] / a
        di_sum = plus_di + minus_di
        dx_values.append(abs(plus_di - minus_di) / di_sum * 100.0 if di_sum > 0 else 0.0)

    adx_s = _wilder(dx_values, period)
    return adx_s[-1] / period if adx_s else 0.0


def _rolling_mean_atr(bars: list[Bar], window: int = 40) -> float:
    """Mean true range over a longer lookback window.

    Used as the reference baseline for the ``atr_min_pct`` volatility floor.
    Returns 0.0 if there is insufficient data.
    """
    subset = bars[-window:] if len(bars) >= window else bars
    if len(subset) < 2:
        return 0.0
    trs = [
        max(
            subset[i].high - subset[i].low,
            abs(subset[i].high - subset[i - 1].close),
            abs(subset[i].low - subset[i - 1].close),
        )
        for i in range(1, len(subset))
    ]
    return sum(trs) / len(trs)


def _session_vwap(bars: list[Bar], current_date: date) -> float | None:
    """Return session VWAP using bars available at decision time."""
    session_bars = [b for b in bars if b.timestamp.date() == current_date and b.volume > 0]
    total_volume = sum(b.volume for b in session_bars)
    if total_volume <= 0:
        return None
    return sum(((b.high + b.low + b.close) / 3.0) * b.volume for b in session_bars) / total_volume


def _session_vwap_distance(bars: list[Bar], current_date: date, close: float) -> float:
    """Return close minus session VWAP using bars available at decision time."""
    vwap = _session_vwap(bars, current_date)
    if vwap is None:
        return 0.0
    return close - vwap


def _ema_slope_aligned(ema_values: list[float], direction: str, n_bars: int) -> bool:
    """Return True if the last *n_bars* EMA slopes are all aligned with *direction*.

    Requires at least ``n_bars + 1`` EMA values; returns False otherwise.
    """
    if n_bars <= 0:
        # Single-bar check: caller already verified the last slope
        return True
    if len(ema_values) < n_bars + 1:
        return False
    for i in range(len(ema_values) - n_bars, len(ema_values)):
        slope = ema_values[i] - ema_values[i - 1]
        if direction == "BUY" and slope <= 0:
            return False
        if direction == "SELL" and slope >= 0:
            return False
    return True


# ---------------------------------------------------------------------------
# Signal engine
# ---------------------------------------------------------------------------

class SignalEngine:
    """Assembles the full London-sweep signal chain for one symbol.

    Maintains stateful sweep detection across bars within a session.
    Create a **fresh** instance per session — never reuse across sessions.

    Parameters
    ----------
    symbol:
        Instrument symbol (must be present in ``config.INSTRUMENTS``).
    config:
        Profile-driven strategy parameters.  Defaults replicate the
        previous hard-coded values so existing callers are unaffected.
    """

    def __init__(self, symbol: str, config: SignalConfig | None = None) -> None:
        self._symbol = symbol
        self._instrument = INSTRUMENTS[symbol]
        self._detector = SweepDetector()
        self._cfg = config or SignalConfig()
        self._raw_setup_events: list[RawSetupEvent] = []
        self._raw_event_seq = 0

    def pop_raw_setup_events(self) -> list[RawSetupEvent]:
        """Return and clear raw setup events emitted since the previous call."""
        events = self._raw_setup_events
        self._raw_setup_events = []
        return events

    def _next_event_id(self, bar: Bar) -> str:
        self._raw_event_seq += 1
        ts = bar.timestamp.strftime("%Y%m%d%H%M%S")
        return f"{self._symbol}-{ts}-{self._raw_event_seq}"

    def _emit_raw_event(
        self,
        *,
        bar: Bar,
        setup_stage: str,
        direction_candidate: str = "",
        passed: bool = False,
        rejection_stage: str = "",
        rejection_reason: str = "",
        rejection_category: str = "",
        rule_name: str = "",
        threshold_value: float | None = None,
        observed_value: float | None = None,
        asian_range: AsianRange | None = None,
        sweep_side: str = "",
        sweep_depth_points: float | None = None,
        time_since_sweep: int | None = None,
        atr14: float | None = None,
        adx: float | None = None,
        ema_fast: float | None = None,
        ema_slow: float | None = None,
        ema_slope: float | None = None,
        vwap: float | None = None,
        confluence_type: str = "",
        accepted_into_final_candidate: bool = False,
    ) -> str:
        current_date = bar.timestamp.date()
        asian_high = asian_range.high if asian_range is not None else None
        asian_low = asian_range.low if asian_range is not None else None
        asian_mid = asian_range.midpoint if asian_range is not None else None
        range_width_points = (
            asian_range.high - asian_range.low if asian_range is not None else None
        )
        range_width_atr = (
            range_width_points / atr14
            if range_width_points is not None and atr14 and atr14 > 0
            else None
        )
        event_id = self._next_event_id(bar)
        event = RawSetupEvent(
            event_id=event_id,
            timestamp=bar.timestamp,
            symbol=self._symbol,
            timeframe="5m",
            session=current_date,
            direction_candidate=direction_candidate,
            setup_stage=setup_stage,
            setup_type="asian_range_sweep",
            confluence_type=confluence_type,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
            asian_high=asian_high,
            asian_low=asian_low,
            asian_mid=asian_mid,
            distance_to_asian_high=bar.close - asian_high if asian_high is not None else None,
            distance_to_asian_low=bar.close - asian_low if asian_low is not None else None,
            distance_to_asian_mid=bar.close - asian_mid if asian_mid is not None else None,
            range_width_points=range_width_points,
            range_width_atr=range_width_atr,
            sweep_side=sweep_side,
            sweep_depth_points=sweep_depth_points,
            sweep_depth_atr=(
                sweep_depth_points / atr14
                if sweep_depth_points is not None and atr14 and atr14 > 0
                else None
            ),
            time_since_sweep=time_since_sweep,
            atr_at_decision=atr14,
            adx_at_decision=adx,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            ema_slope=ema_slope,
            vwap=vwap,
            vwap_distance=bar.close - vwap if vwap is not None else None,
            hour_utc=bar.timestamp.hour,
            weekday=bar.timestamp.strftime("%A"),
            month=bar.timestamp.month,
            passed=passed,
            rejected=not passed,
            rejection_stage=rejection_stage,
            rejection_reason=rejection_reason,
            rejection_category=rejection_category,
            rule_name=rule_name,
            threshold_value=threshold_value,
            observed_value=observed_value,
            accepted_into_final_candidate=accepted_into_final_candidate,
            accepted_into_trade=False,
        )
        self._raw_setup_events.append(event)
        return event_id

    def process_bar(
        self,
        bars_5m: list[Bar],
        bars_1h: list[Bar],
    ) -> TradeSignal | None:
        """Evaluate the current bar (last element of *bars_5m*) for a signal.

        Parameters
        ----------
        bars_5m:
            All 5-minute bars up to and including the current bar (no lookahead).
        bars_1h:
            All 1-hour bars up to (but NOT including) the current bar's
            timestamp (caller must apply bisect slice — see simulator).

        Returns
        -------
        TradeSignal if all conditions are met, otherwise None.
        """
        if len(bars_5m) < 15:
            return None

        current_bar = bars_5m[-1]
        current_date: date = current_bar.timestamp.date()
        cfg = self._cfg

        # 1. ATR-14
        atr14 = _compute_atr14(bars_5m)
        if atr14 == 0.0:
            self._emit_raw_event(
                bar=current_bar,
                setup_stage="atr_check",
                rejection_stage="atr_check",
                rejection_reason="missing_atr",
                rejection_category="data_quality_rejection",
                rule_name="atr14_available",
                observed_value=atr14,
            )
            return None

        # 2. ATR volatility floor — skip low-ATR bars (ranging conditions)
        if cfg.atr_min_pct > 0.0:
            ref_atr = _rolling_mean_atr(bars_5m)
            if ref_atr > 0.0 and atr14 / ref_atr < cfg.atr_min_pct:
                self._emit_raw_event(
                    bar=current_bar,
                    setup_stage="atr_threshold",
                    rejection_stage="atr_threshold",
                    rejection_reason="atr_below_threshold",
                    rejection_category="market_structure_rejection",
                    rule_name="atr_min_pct",
                    threshold_value=cfg.atr_min_pct,
                    observed_value=atr14 / ref_atr,
                    atr14=atr14,
                )
                return None

        # 3. Asian range
        asian_range: AsianRange | None = compute_asian_range(bars_5m, current_date)
        if asian_range is None:
            self._emit_raw_event(
                bar=current_bar,
                setup_stage="asian_range",
                rejection_stage="asian_range",
                rejection_reason="missing_asian_range",
                rejection_category="data_quality_rejection",
                rule_name="asian_range_available",
                atr14=atr14,
            )
            return None

        # 4. Sweep detection
        self._detector.update(current_bar, asian_range, atr14)
        sweep = self._detector.active
        if sweep is None:
            self._emit_raw_event(
                bar=current_bar,
                setup_stage="sweep_detection",
                rejection_stage="sweep_detection",
                rejection_reason="no_sweep_detected",
                rejection_category="market_structure_rejection",
                rule_name="asian_range_sweep",
                asian_range=asian_range,
                atr14=atr14,
            )
            self._detector.tick()
            return None

        direction = sweep.direction
        if direction not in {"BUY", "SELL"}:
            self._emit_raw_event(
                bar=current_bar,
                setup_stage="sweep_direction",
                direction_candidate=direction,
                rejection_stage="sweep_direction",
                rejection_reason="invalid_sweep_direction",
                rejection_category="market_structure_rejection",
                rule_name="sweep_direction",
                asian_range=asian_range,
                atr14=atr14,
            )
            self._detector.tick()
            return None
        time_since_sweep = 3 - sweep.candles_remaining
        if direction == "BUY":
            sweep_depth_points = max(asian_range.low - current_bar.low, 0.0)
            sweep_side = "low"
        else:
            sweep_depth_points = max(current_bar.high - asian_range.high, 0.0)
            sweep_side = "high"

        # 5. ADX momentum gate — filter out choppy/range-bound regimes
        adx = _compute_adx14(bars_5m)
        if cfg.adx_min_threshold > 0.0 and len(bars_5m) < 29:
            self._emit_raw_event(
                bar=current_bar,
                setup_stage="adx_check",
                direction_candidate=direction,
                rejection_stage="adx_check",
                rejection_reason="missing_adx",
                rejection_category="data_quality_rejection",
                rule_name="adx14_available",
                asian_range=asian_range,
                sweep_side=sweep_side,
                sweep_depth_points=sweep_depth_points,
                time_since_sweep=time_since_sweep,
                atr14=atr14,
                observed_value=adx,
            )
            self._detector.tick()
            return None
        if cfg.adx_min_threshold > 0.0:
            if adx < cfg.adx_min_threshold:
                self._emit_raw_event(
                    bar=current_bar,
                    setup_stage="adx_threshold",
                    direction_candidate=direction,
                    rejection_stage="adx_threshold",
                    rejection_reason="adx_below_threshold",
                    rejection_category="market_structure_rejection",
                    rule_name="adx_min_threshold",
                    threshold_value=cfg.adx_min_threshold,
                    observed_value=adx,
                    asian_range=asian_range,
                    sweep_side=sweep_side,
                    sweep_depth_points=sweep_depth_points,
                    time_since_sweep=time_since_sweep,
                    atr14=atr14,
                    adx=adx,
                )
                self._detector.tick()
                return None

        # 6. EMA-50 trend filter on 1h bars (with optional persistence check)
        if len(bars_1h) < 50:
            self._emit_raw_event(
                bar=current_bar,
                setup_stage="ema_trend",
                direction_candidate=direction,
                rejection_stage="ema_trend",
                rejection_reason="ema_trend_filter_failed",
                rejection_category="market_structure_rejection",
                rule_name="ema50_history_available",
                threshold_value=50,
                observed_value=len(bars_1h),
                asian_range=asian_range,
                sweep_side=sweep_side,
                sweep_depth_points=sweep_depth_points,
                time_since_sweep=time_since_sweep,
                atr14=atr14,
                adx=adx,
            )
            self._detector.tick()
            return None

        ema_values = _ema(bars_1h, 50)
        if len(ema_values) < 2:
            self._emit_raw_event(
                bar=current_bar,
                setup_stage="ema_trend",
                direction_candidate=direction,
                rejection_stage="ema_trend",
                rejection_reason="ema_trend_filter_failed",
                rejection_category="market_structure_rejection",
                rule_name="ema50_values_available",
                observed_value=len(ema_values),
                asian_range=asian_range,
                sweep_side=sweep_side,
                sweep_depth_points=sweep_depth_points,
                time_since_sweep=time_since_sweep,
                atr14=atr14,
                adx=adx,
            )
            self._detector.tick()
            return None

        ema_slope = ema_values[-1] - ema_values[-2]
        ema_fast = ema_values[-1]
        ema_slow = ema_values[-2]
        trend_aligned = (
            (direction == "BUY" and ema_slope > 0)
            or (direction == "SELL" and ema_slope < 0)
        )
        if not trend_aligned:
            self._emit_raw_event(
                bar=current_bar,
                setup_stage="ema_trend",
                direction_candidate=direction,
                rejection_stage="ema_trend",
                rejection_reason="ema_trend_filter_failed",
                rejection_category="market_structure_rejection",
                rule_name="ema_slope_aligned",
                observed_value=ema_slope,
                asian_range=asian_range,
                sweep_side=sweep_side,
                sweep_depth_points=sweep_depth_points,
                time_since_sweep=time_since_sweep,
                atr14=atr14,
                adx=adx,
                ema_fast=ema_fast,
                ema_slow=ema_slow,
                ema_slope=ema_slope,
            )
            self._detector.tick()
            return None

        # EMA persistence: require N consecutive aligned slopes (whipsaw filter)
        if cfg.ema_trend_persistence_bars > 0:
            if not _ema_slope_aligned(ema_values, direction, cfg.ema_trend_persistence_bars):
                self._emit_raw_event(
                    bar=current_bar,
                    setup_stage="ema_trend",
                    direction_candidate=direction,
                    rejection_stage="ema_trend",
                    rejection_reason="ema_trend_filter_failed",
                    rejection_category="market_structure_rejection",
                    rule_name="ema_trend_persistence_bars",
                    threshold_value=cfg.ema_trend_persistence_bars,
                    observed_value=ema_slope,
                    asian_range=asian_range,
                    sweep_side=sweep_side,
                    sweep_depth_points=sweep_depth_points,
                    time_since_sweep=time_since_sweep,
                    atr14=atr14,
                    adx=adx,
                    ema_fast=ema_fast,
                    ema_slow=ema_slow,
                    ema_slope=ema_slope,
                )
                self._detector.tick()
                return None

        # 7. Confluence — OB or FVG required
        confluence = find_confluence(bars_5m[-20:], direction, current_bar.close, atr14)
        if confluence.confluence_type == "NONE":
            self._emit_raw_event(
                bar=current_bar,
                setup_stage="confluence",
                direction_candidate=direction,
                rejection_stage="confluence",
                rejection_reason="confluence_failed",
                rejection_category="market_structure_rejection",
                rule_name="ob_or_fvg_required",
                asian_range=asian_range,
                sweep_side=sweep_side,
                sweep_depth_points=sweep_depth_points,
                time_since_sweep=time_since_sweep,
                atr14=atr14,
                adx=adx,
                ema_fast=ema_fast,
                ema_slow=ema_slow,
                ema_slope=ema_slope,
                vwap=_session_vwap(bars_5m, current_date),
                confluence_type="NONE",
            )
            self._detector.tick()
            return None

        # 7b. Confluence type allowlist filter
        if confluence.confluence_type not in cfg.allowed_confluence_types:
            self._emit_raw_event(
                bar=current_bar,
                setup_stage="confluence_allowlist",
                direction_candidate=direction,
                rejection_stage="confluence_allowlist",
                rejection_reason="confluence_failed",
                rejection_category="market_structure_rejection",
                rule_name="allowed_confluence_types",
                asian_range=asian_range,
                sweep_side=sweep_side,
                sweep_depth_points=sweep_depth_points,
                time_since_sweep=time_since_sweep,
                atr14=atr14,
                adx=adx,
                ema_fast=ema_fast,
                ema_slow=ema_slow,
                ema_slope=ema_slope,
                vwap=_session_vwap(bars_5m, current_date),
                confluence_type=confluence.confluence_type,
            )
            self._detector.tick()
            return None

        # 8. Build bracket prices using profile target multiple
        tick = self._instrument.tick_size
        entry = _round_to_tick(current_bar.close, tick)
        mult = cfg.target_atr_multiple
        range_width_atr = (asian_range.high - asian_range.low) / atr14 if atr14 > 0 else 0.0
        sweep_depth_atr = sweep_depth_points / atr14 if atr14 > 0 else 0.0
        distance_to_asian_mid = (current_bar.close - asian_range.midpoint) / atr14 if atr14 > 0 else 0.0
        vwap = _session_vwap(bars_5m, current_date)
        vwap_distance = current_bar.close - vwap if vwap is not None else 0.0

        if direction == "BUY":
            stop = _round_to_tick(entry - 1.0 * atr14, tick)
            tp1  = _round_to_tick(entry + mult * atr14, tick)
            tp2  = _round_to_tick(entry + (mult + 1.0) * atr14, tick)
        else:
            stop = _round_to_tick(entry + 1.0 * atr14, tick)
            tp1  = _round_to_tick(entry - mult * atr14, tick)
            tp2  = _round_to_tick(entry - (mult + 1.0) * atr14, tick)

        if not math.isfinite(entry):
            self._emit_raw_event(
                bar=current_bar,
                setup_stage="entry_trigger",
                direction_candidate=direction,
                rejection_stage="entry_trigger",
                rejection_reason="invalid_entry_price",
                rejection_category="data_quality_rejection",
                rule_name="entry_price_finite",
                asian_range=asian_range,
                sweep_side=sweep_side,
                sweep_depth_points=sweep_depth_points,
                time_since_sweep=time_since_sweep,
                atr14=atr14,
                adx=adx,
                ema_fast=ema_fast,
                ema_slow=ema_slow,
                ema_slope=ema_slope,
                vwap=vwap,
                confluence_type=confluence.confluence_type,
            )
            self._detector.tick()
            return None
        stop_valid = stop < entry if direction == "BUY" else stop > entry
        if not math.isfinite(stop) or not stop_valid:
            self._emit_raw_event(
                bar=current_bar,
                setup_stage="stop_placement",
                direction_candidate=direction,
                rejection_stage="stop_placement",
                rejection_reason="invalid_stop_price",
                rejection_category="risk_rejection",
                rule_name="stop_directional_valid",
                observed_value=stop,
                asian_range=asian_range,
                sweep_side=sweep_side,
                sweep_depth_points=sweep_depth_points,
                time_since_sweep=time_since_sweep,
                atr14=atr14,
                adx=adx,
                ema_fast=ema_fast,
                ema_slow=ema_slow,
                ema_slope=ema_slope,
                vwap=vwap,
                confluence_type=confluence.confluence_type,
            )
            self._detector.tick()
            return None
        target_valid = tp1 > entry and tp2 > tp1 if direction == "BUY" else tp1 < entry and tp2 < tp1
        if not math.isfinite(tp1) or not math.isfinite(tp2) or not target_valid:
            self._emit_raw_event(
                bar=current_bar,
                setup_stage="target_placement",
                direction_candidate=direction,
                rejection_stage="target_placement",
                rejection_reason="invalid_target_price",
                rejection_category="risk_rejection",
                rule_name="target_directional_valid",
                observed_value=tp1,
                asian_range=asian_range,
                sweep_side=sweep_side,
                sweep_depth_points=sweep_depth_points,
                time_since_sweep=time_since_sweep,
                atr14=atr14,
                adx=adx,
                ema_fast=ema_fast,
                ema_slow=ema_slow,
                ema_slope=ema_slope,
                vwap=vwap,
                confluence_type=confluence.confluence_type,
            )
            self._detector.tick()
            return None

        event_id = self._emit_raw_event(
            bar=current_bar,
            setup_stage="final_candidate",
            direction_candidate=direction,
            passed=True,
            asian_range=asian_range,
            sweep_side=sweep_side,
            sweep_depth_points=sweep_depth_points,
            time_since_sweep=time_since_sweep,
            atr14=atr14,
            adx=adx,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            ema_slope=ema_slope,
            vwap=vwap,
            confluence_type=confluence.confluence_type,
            accepted_into_final_candidate=True,
        )

        # Consume the sweep — signal emitted, detector cleared.
        self._detector.clear()
        self._detector.tick()

        return TradeSignal(
            symbol=self._symbol,
            direction=direction,
            entry_price=entry,
            stop_price=stop,
            tp1_price=tp1,
            tp2_price=tp2,
            confluence=confluence,
            atr14=atr14,
            ema_slope=ema_slope,
            adx_at_entry=adx,
            vwap_distance_at_entry=vwap_distance,
            range_width_atr=range_width_atr,
            sweep_depth_atr=sweep_depth_atr,
            distance_to_asian_mid=distance_to_asian_mid,
            time_since_sweep=time_since_sweep,
            raw_setup_event_id=event_id,
        )
