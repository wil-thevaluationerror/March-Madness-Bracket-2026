from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable
from uuid import uuid4

import pandas as pd

from config import StrategyConfig
from models.instruments import resolve_instrument
from models.orders import OrderIntent, OrderType, Regime, Side, TimeInForce


@dataclass(slots=True)
class SignalInput:
    symbol: str
    timestamp: datetime
    regime: Regime
    long_signal: bool
    short_signal: bool
    entry_price: float
    stop_price: float
    target_price: float | None
    signal_score: float
    qty: int = 1


def build_order_intent(signal: SignalInput, config: StrategyConfig) -> OrderIntent | None:
    if signal.regime == Regime.CHOP_MEAN_REVERT:
        return None
    if not signal.long_signal and not signal.short_signal:
        return None
    side = Side.BUY if signal.long_signal else Side.SELL
    return OrderIntent(
        intent_id=f"{config.strategy_id}-{uuid4().hex[:12]}",
        symbol=signal.symbol,
        side=side,
        qty=signal.qty,
        entry_type=OrderType[config.default_entry_order_type],
        entry_price=signal.entry_price,
        stop_price=signal.stop_price,
        target_price=signal.target_price,
        time_in_force=TimeInForce.DAY,
        reason="regime_filtered_breakout",
        signal_ts=signal.timestamp,
        signal_score=signal.signal_score,
        regime=signal.regime,
        strategy_id=config.strategy_id,
        metadata={},
    )


def generate_intents(
    df: pd.DataFrame,
    config: StrategyConfig | None = None,
    *,
    diagnostics_callback: Callable[[dict[str, Any]], None] | None = None,
    diagnostic_since: datetime | pd.Timestamp | None = None,
    diagnostic_context: dict[str, Any] | None = None,
) -> list[OrderIntent]:
    strategy_config = config or StrategyConfig()
    intents: list[OrderIntent] = []
    if df.empty:
        return intents
    diagnostic_context = diagnostic_context or {}
    diagnostic_since_ts = pd.Timestamp(diagnostic_since) if diagnostic_since is not None else None

    frame = df.copy().sort_values(["symbol", "ts_event"]).reset_index(drop=True)
    if "atr" not in frame.columns:
        frame["prev_close"] = frame.groupby("symbol")["close"].shift(1)
        true_range = pd.concat(
            [
                frame["high"] - frame["low"],
                (frame["high"] - frame["prev_close"]).abs(),
                (frame["low"] - frame["prev_close"]).abs(),
            ],
            axis=1,
        ).max(axis=1)
        frame["true_range"] = true_range
        frame["atr"] = frame.groupby("symbol", group_keys=False)["true_range"].transform(
            lambda series: series.rolling(14, min_periods=1).mean()
        )
        frame["atr_median"] = frame.groupby("symbol", group_keys=False)["atr"].transform(
            lambda series: series.rolling(50, min_periods=5).median()
        )
    frame["atr"] = frame["atr"].replace(0.0, pd.NA).bfill().fillna(1.0)
    frame["rolling_median_volume"] = frame.groupby("symbol", group_keys=False)["volume"].transform(
        lambda series: series.rolling(20, min_periods=1).median()
    ).replace(0.0, pd.NA).bfill().fillna(1.0)
    frame["breakout_level"] = frame.groupby("symbol", group_keys=False)["high"].transform(
        lambda series: series.shift(1).rolling(strategy_config.breakout_lookback_bars, min_periods=1).max()
    )
    frame["breakout_level"] = frame["breakout_level"].fillna(frame["close"])
    frame["breakdown_level"] = frame.groupby("symbol", group_keys=False)["low"].transform(
        lambda series: series.shift(1).rolling(strategy_config.breakout_lookback_bars, min_periods=1).min()
    )
    frame["breakdown_level"] = frame["breakdown_level"].fillna(frame["close"])

    last_signal_index_by_key: dict[tuple[str, Side], int] = {}
    last_trigger_level_by_key: dict[tuple[str, Side], float] = {}

    # EMA persistence tracking: count consecutive bars where EMA_fast > EMA_slow (long)
    # or EMA_fast < EMA_slow (short) per symbol.  Reset to 0 on any bar that flips.
    ema_persist_long: dict[str, int] = {}   # symbol → consecutive bars EMA bullish
    ema_persist_short: dict[str, int] = {}  # symbol → consecutive bars EMA bearish
    ema_persistence_required = int(getattr(strategy_config, "ema_trend_persistence_bars", 0))

    def emit_diagnostic(
        *,
        row: object,
        symbol: str,
        close_price: float,
        atr: float | None,
        adx: float | None,
        ema_trend_state: str,
        vwap_condition: str,
        breakout_condition: str,
        signal_score: float | None,
        decision: str,
        no_trade_reason: str | None,
    ) -> None:
        if diagnostics_callback is None:
            return
        ts_event = getattr(row, "ts_event", None)
        if ts_event is None:
            return
        bar_ts = pd.Timestamp(ts_event)
        if diagnostic_since_ts is not None and bar_ts <= diagnostic_since_ts:
            return
        try:
            diagnostics_callback(
                {
                    "symbol": symbol,
                    "bar_timestamp": bar_ts.isoformat(),
                    "close": close_price,
                    "session_allowed": diagnostic_context.get("session_allowed", True),
                    "bars_loaded": len(frame),
                    "atr": atr,
                    "adx": adx,
                    "ema_trend_state": ema_trend_state,
                    "vwap_condition": vwap_condition,
                    "breakout_condition": breakout_condition,
                    "signal_score": signal_score,
                    "decision": decision,
                    "no_trade_reason": no_trade_reason,
                    "risk_allowed": diagnostic_context.get("risk_allowed"),
                }
            )
        except Exception:
            # Diagnostics must never alter signal generation.
            return

    for index, row in enumerate(frame.itertuples(index=False)):
        symbol = str(getattr(row, "symbol", strategy_config.default_symbol))
        instrument = resolve_instrument(symbol)
        close_price = float(getattr(row, "close"))
        ema_fast = float(getattr(row, "ema_fast", close_price))
        ema_slow = float(getattr(row, "ema_slow", close_price))
        vwap = float(getattr(row, "vwap", close_price))
        atr = max(float(getattr(row, "atr", 1.0) or 1.0), instrument.tick_size)
        atr_median = float(getattr(row, "atr_median", atr) or atr)
        atr_pct = (atr / atr_median) if atr_median > 0 else 1.0
        breakout_level = float(getattr(row, "breakout_level", close_price) or close_price)
        breakdown_level = float(getattr(row, "breakdown_level", close_price) or close_price)
        adx_val = float(getattr(row, "adx", 0.0) or 0.0)

        # Update EMA persistence counters for this bar (regardless of signal direction).
        if ema_fast > ema_slow:
            ema_persist_long[symbol] = ema_persist_long.get(symbol, 0) + 1
            ema_persist_short[symbol] = 0
            ema_trend_state = "bullish"
        elif ema_fast < ema_slow:
            ema_persist_short[symbol] = ema_persist_short.get(symbol, 0) + 1
            ema_persist_long[symbol] = 0
            ema_trend_state = "bearish"
        else:
            ema_persist_long[symbol] = 0
            ema_persist_short[symbol] = 0
            ema_trend_state = "neutral"

        is_long_setup = close_price > vwap and ema_fast > ema_slow and close_price > breakout_level
        is_short_setup = close_price < vwap and ema_fast < ema_slow and close_price < breakdown_level
        vwap_condition = "above_vwap" if close_price > vwap else "below_vwap" if close_price < vwap else "unknown"
        breakout_condition = "long_breakout" if close_price > breakout_level else "short_breakout" if close_price < breakdown_level else "none"
        if not is_long_setup and not is_short_setup:
            emit_diagnostic(
                row=row,
                symbol=symbol,
                close_price=close_price,
                atr=atr,
                adx=adx_val,
                ema_trend_state=ema_trend_state,
                vwap_condition=vwap_condition,
                breakout_condition=breakout_condition,
                signal_score=None,
                decision="no_trade",
                no_trade_reason="no_breakout",
            )
            continue

        if is_long_setup:
            side = Side.BUY
            trigger_level = breakout_level
            trend_distance = close_price - trigger_level
            trend_state = "ema_above_vwap_above"
            long_signal = True
            short_signal = False
        else:
            side = Side.SELL
            trigger_level = breakdown_level
            trend_distance = trigger_level - close_price
            trend_state = "ema_below_vwap_below"
            long_signal = False
            short_signal = True

        # EMA trend persistence filter: skip entry if EMA alignment hasn't held
        # for the required number of consecutive bars.
        if ema_persistence_required > 0:
            persist_count = ema_persist_long.get(symbol, 0) if is_long_setup else ema_persist_short.get(symbol, 0)
            if persist_count < ema_persistence_required:
                emit_diagnostic(
                    row=row,
                    symbol=symbol,
                    close_price=close_price,
                    atr=atr,
                    adx=adx_val,
                    ema_trend_state=ema_trend_state,
                    vwap_condition=vwap_condition,
                    breakout_condition=breakout_condition,
                    signal_score=None,
                    decision="no_trade",
                    no_trade_reason="ema_persistence",
                )
                continue

        extension = abs(close_price - trigger_level)
        if extension > float(strategy_config.max_entry_extension_atr) * atr:
            emit_diagnostic(
                row=row,
                symbol=symbol,
                close_price=close_price,
                atr=atr,
                adx=adx_val,
                ema_trend_state=ema_trend_state,
                vwap_condition=vwap_condition,
                breakout_condition=breakout_condition,
                signal_score=None,
                decision="no_trade",
                no_trade_reason="extension_too_high",
            )
            continue

        signal_key = (symbol, side)
        last_index = last_signal_index_by_key.get(signal_key)
        last_trigger_level = last_trigger_level_by_key.get(signal_key)

        # ADX regime filter: skip choppy bars when threshold is set
        if strategy_config.adx_min_threshold > 0.0 and adx_val < strategy_config.adx_min_threshold:
            emit_diagnostic(
                row=row,
                symbol=symbol,
                close_price=close_price,
                atr=atr,
                adx=adx_val,
                ema_trend_state=ema_trend_state,
                vwap_condition=vwap_condition,
                breakout_condition=breakout_condition,
                signal_score=None,
                decision="no_trade",
                no_trade_reason="adx_below_threshold",
            )
            continue

        # ATR volatility filter: skip anomalously low-ATR bars (ranging/thin market).
        # atr_pct = atr / atr_median; values < atr_min_pct indicate compressed volatility
        # where the 2× stop target is unlikely to be reached before reversal or session end.
        atr_min_pct = float(getattr(strategy_config, "atr_min_pct", 0.0))
        if atr_min_pct > 0.0 and atr_pct < atr_min_pct:
            emit_diagnostic(
                row=row,
                symbol=symbol,
                close_price=close_price,
                atr=atr,
                adx=adx_val,
                ema_trend_state=ema_trend_state,
                vwap_condition=vwap_condition,
                breakout_condition=breakout_condition,
                signal_score=None,
                decision="no_trade",
                no_trade_reason="atr_below_threshold",
            )
            continue

        # Use 5-min ATR for stop sizing when configured — reduces stop-outs from 1-min noise
        if strategy_config.use_5min_atr_for_stops:
            atr_for_stop = max(float(getattr(row, "atr_5min", atr) or atr), instrument.tick_size)
        else:
            atr_for_stop = atr

        target_mult = max(float(getattr(strategy_config, "target_atr_multiple", 2.0)), 1.0)
        if is_long_setup:
            stop_price = close_price - atr_for_stop
            target_price = close_price + (target_mult * atr_for_stop)
        else:
            stop_price = close_price + atr_for_stop
            target_price = close_price - (target_mult * atr_for_stop)
        if (
            last_index is not None
            and index - last_index <= strategy_config.reentry_cooldown_bars
            and last_trigger_level is not None
            and (
                (side == Side.BUY and trigger_level <= last_trigger_level)
                or (side == Side.SELL and trigger_level >= last_trigger_level)
            )
        ):
            emit_diagnostic(
                row=row,
                symbol=symbol,
                close_price=close_price,
                atr=atr,
                adx=adx_val,
                ema_trend_state=ema_trend_state,
                vwap_condition=vwap_condition,
                breakout_condition=breakout_condition,
                signal_score=None,
                decision="no_trade",
                no_trade_reason="reentry_cooldown",
            )
            continue

        trend_strength = max(abs(ema_fast - ema_slow) / atr, 0.0)
        breakout_strength = max(trend_distance / atr, 0.0)
        rolling_median_volume = max(float(getattr(row, "rolling_median_volume", 1.0) or 1.0), 1.0)
        volume_strength = max(float(getattr(row, "volume", 0.0)) / rolling_median_volume, 0.0)
        vwap_alignment = max(0.0, 1.0 - abs(close_price - vwap) / atr)

        # Volume confirmation gate: skip low-volume signals when filter is active.
        # volume_strength < filter means volume is below the required fraction of median.
        if strategy_config.volume_entry_filter > 0.0 and volume_strength < strategy_config.volume_entry_filter:
            emit_diagnostic(
                row=row,
                symbol=symbol,
                close_price=close_price,
                atr=atr,
                adx=adx_val,
                ema_trend_state=ema_trend_state,
                vwap_condition=vwap_condition,
                breakout_condition=breakout_condition,
                signal_score=None,
                decision="no_trade",
                no_trade_reason="volume_filter",
            )
            continue

        # Normalize all components to [0, 1] so the weighted score is bounded and
        # thresholds (e.g. reentry_signal_score_min) remain interpretable.
        # trend_strength: cap at 2 ATRs of EMA separation
        # breakout_freshness: INVERTED extension fraction — rewards entries CLOSE to the
        #   breakout level (less momentum used up = better R:R), penalises overextended entries.
        #   0.0 = price exactly at max extension; 1.0 = price exactly at breakout level.
        # volume_strength: cap at 2x median (extreme volume is noise, not signal)
        max_extension = max(float(strategy_config.max_entry_extension_atr), 1e-9)
        trend_strength_norm = min(trend_strength / 2.0, 1.0)
        breakout_freshness_norm = 1.0 - min(breakout_strength / max_extension, 1.0)
        volume_strength_norm = min(volume_strength / 2.0, 1.0)
        vwap_alignment_norm = vwap_alignment  # already in [0, 1]
        regime_weight = 1.0

        signal_score = (
            0.35 * trend_strength_norm
            + 0.30 * breakout_freshness_norm
            + 0.20 * volume_strength_norm
            + 0.10 * vwap_alignment_norm
            + 0.05 * regime_weight
        )

        # Initial entry quality gate: skip weak setups before submitting an intent.
        if strategy_config.min_entry_signal_score > 0.0 and signal_score < strategy_config.min_entry_signal_score:
            emit_diagnostic(
                row=row,
                symbol=symbol,
                close_price=close_price,
                atr=atr,
                adx=adx_val,
                ema_trend_state=ema_trend_state,
                vwap_condition=vwap_condition,
                breakout_condition=breakout_condition,
                signal_score=signal_score,
                decision="no_trade",
                no_trade_reason="signal_score_below_threshold",
            )
            continue
        ts_event = getattr(row, "ts_event")
        signal = SignalInput(
            symbol=symbol,
            timestamp=ts_event if isinstance(ts_event, datetime) else pd.Timestamp(ts_event).to_pydatetime(),
            regime=Regime.TREND_EXPANSION,
            long_signal=long_signal,
            short_signal=short_signal,
            entry_price=close_price,
            stop_price=stop_price,
            target_price=target_price,
            signal_score=signal_score,
            qty=strategy_config.base_qty,
        )
        intent = build_order_intent(signal, strategy_config)
        if intent is not None:
            intent.metadata.update(
                {
                    "atr": atr,
                    "atr_median": atr_median,
                    "atr_pct": atr_pct,
                    "atr_extreme": bool(atr_median > 0 and atr > (3.0 * atr_median)),
                    "ema_fast": ema_fast,
                    "ema_slow": ema_slow,
                    "vwap": vwap,
                    "trend_state": trend_state,
                    # Raw (unbounded) component values for diagnostics
                    "trend_strength": trend_strength,
                    "breakout_strength": breakout_strength,
                    "volume_strength": volume_strength,
                    "vwap_alignment": vwap_alignment,
                    # Normalized [0,1] components that compose the reported signal_score
                    "trend_strength_norm": trend_strength_norm,
                    # breakout_freshness_norm is INVERTED — high value = close to breakout level
                    "breakout_freshness_norm": breakout_freshness_norm,
                    "volume_strength_norm": volume_strength_norm,
                    # breakout_level is the directional trigger reference used by re-entry logic.
                    "breakout_level": trigger_level,
                }
            )
            intents.append(intent)
            emit_diagnostic(
                row=row,
                symbol=symbol,
                close_price=close_price,
                atr=atr,
                adx=adx_val,
                ema_trend_state=ema_trend_state,
                vwap_condition=vwap_condition,
                breakout_condition=breakout_condition,
                signal_score=signal_score,
                decision="trade",
                no_trade_reason=None,
            )
            last_signal_index_by_key[signal_key] = index
            last_trigger_level_by_key[signal_key] = trigger_level
    return intents
