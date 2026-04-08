from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
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


def generate_intents(df: pd.DataFrame, config: StrategyConfig | None = None) -> list[OrderIntent]:
    strategy_config = config or StrategyConfig()
    intents: list[OrderIntent] = []
    if df.empty:
        return intents

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

        is_long_setup = close_price > vwap and ema_fast > ema_slow and close_price > breakout_level
        is_short_setup = close_price < vwap and ema_fast < ema_slow and close_price < breakdown_level
        if not is_long_setup and not is_short_setup:
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

        extension = abs(close_price - trigger_level)
        if extension > float(strategy_config.max_entry_extension_atr) * atr:
            continue

        signal_key = (symbol, side)
        last_index = last_signal_index_by_key.get(signal_key)
        last_trigger_level = last_trigger_level_by_key.get(signal_key)

        if is_long_setup:
            stop_price = close_price - atr
            target_price = close_price + (2.0 * atr)
        else:
            stop_price = close_price + atr
            target_price = close_price - (2.0 * atr)
        if (
            last_index is not None
            and index - last_index <= strategy_config.reentry_cooldown_bars
            and last_trigger_level is not None
            and (
                (side == Side.BUY and trigger_level <= last_trigger_level)
                or (side == Side.SELL and trigger_level >= last_trigger_level)
            )
        ):
            continue

        trend_strength = max(abs(ema_fast - ema_slow) / atr, 0.0)
        breakout_strength = max(trend_distance / atr, 0.0)
        rolling_median_volume = max(float(getattr(row, "rolling_median_volume", 1.0) or 1.0), 1.0)
        volume_strength = max(float(getattr(row, "volume", 0.0)) / rolling_median_volume, 0.0)
        vwap_alignment = max(0.0, 1.0 - abs(close_price - vwap) / atr)
        regime_weight = 1.0
        signal_score = (
            0.35 * trend_strength
            + 0.30 * breakout_strength
            + 0.20 * volume_strength
            + 0.10 * vwap_alignment
            + 0.05 * regime_weight
        )
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
                    "trend_strength": trend_strength,
                    "breakout_strength": breakout_strength,
                    "volume_strength": volume_strength,
                    "vwap_alignment": vwap_alignment,
                    # breakout_level is the directional trigger reference used by re-entry logic.
                    "breakout_level": trigger_level,
                }
            )
            intents.append(intent)
            last_signal_index_by_key[signal_key] = index
            last_trigger_level_by_key[signal_key] = trigger_level
    return intents
