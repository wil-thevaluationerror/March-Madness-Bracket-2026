from __future__ import annotations

import pandas as pd


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame = frame.sort_values(["symbol", "ts_event"]).reset_index(drop=True)
    frame["session_date"] = frame["ts_event"].dt.date
    typical_price = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    cumulative_value = (typical_price * frame["volume"]).groupby([frame["symbol"], frame["session_date"]]).cumsum()
    cumulative_volume = frame["volume"].groupby([frame["symbol"], frame["session_date"]]).cumsum().replace(0, pd.NA)
    frame["vwap"] = cumulative_value / cumulative_volume
    return frame


def add_ema(df: pd.DataFrame, fast_span: int = 20, slow_span: int = 50) -> pd.DataFrame:
    frame = df.copy()
    frame = frame.sort_values(["symbol", "ts_event"]).reset_index(drop=True)
    frame["ema_fast"] = frame.groupby("symbol", group_keys=False)["close"].transform(
        lambda series: series.ewm(span=fast_span, adjust=False).mean()
    )
    frame["ema_slow"] = frame.groupby("symbol", group_keys=False)["close"].transform(
        lambda series: series.ewm(span=slow_span, adjust=False).mean()
    )
    return frame


def _add_basic_atr_columns(frame: pd.DataFrame, period: int, median_window: int, atr_col: str, median_col: str) -> pd.DataFrame:
    working = frame.copy()
    working["prev_close"] = working.groupby("symbol")["close"].shift(1)
    true_range = pd.concat(
        [
            working["high"] - working["low"],
            (working["high"] - working["prev_close"]).abs(),
            (working["low"] - working["prev_close"]).abs(),
        ],
        axis=1,
    ).max(axis=1)
    working["true_range"] = true_range
    working[atr_col] = working.groupby("symbol", group_keys=False)["true_range"].transform(
        lambda series: series.rolling(period, min_periods=1).mean()
    )
    working[median_col] = working.groupby("symbol", group_keys=False)[atr_col].transform(
        lambda series: series.rolling(median_window, min_periods=5).median()
    )
    return working


def add_atr(df: pd.DataFrame, period: int = 14, median_window: int = 50) -> pd.DataFrame:
    frame = df.copy()
    frame = frame.sort_values(["symbol", "ts_event"]).reset_index(drop=True)
    frame = _add_basic_atr_columns(frame, period, median_window, "atr", "atr_median")

    bars_5min = (
        frame.set_index("ts_event")
        .groupby("symbol", group_keys=False)
        .resample("5min", label="right", closed="right")
        .agg(
            {
                "symbol": "first",
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["symbol", "open", "high", "low", "close"])
        .reset_index()
    )
    bars_5min = _add_basic_atr_columns(bars_5min, period, median_window, "atr_5min", "atr_5min_median")
    frame = pd.merge_asof(
        frame.sort_values(["symbol", "ts_event"]),
        bars_5min[["symbol", "ts_event", "atr_5min", "atr_5min_median"]].sort_values(["symbol", "ts_event"]),
        on="ts_event",
        by="symbol",
        direction="backward",
    )
    frame["atr_5min"] = frame["atr_5min"].fillna(frame["atr"])
    frame["atr_5min_median"] = frame["atr_5min_median"].fillna(frame["atr_median"])
    return frame


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add ADX (Average Directional Index) using Wilder's smoothing.

    Adds columns: ``adx``, ``plus_di``, ``minus_di``.
    ADX < 20 ≈ choppy/ranging; ADX > 25 ≈ trending regime.
    Wilder's smoothing: EMA with alpha = 1/period (equivalent to span=2*period-1
    in pandas ewm, but we compute it manually to match the standard definition).
    """
    frame = df.copy().sort_values(["symbol", "ts_event"]).reset_index(drop=True)

    def _wilder_smooth(series: pd.Series, n: int) -> pd.Series:
        """Wilder's moving average: seed on first n non-NaN values, then alpha=1/n."""
        result = pd.Series(float("nan"), index=series.index, dtype=float)
        values = series.values
        # Collect the indices of the first n non-NaN observations for seeding.
        non_nan_idx = [i for i, v in enumerate(values) if not (v != v or v is None)]  # NaN check
        if len(non_nan_idx) < n:
            return result
        seed_indices = non_nan_idx[:n]
        seed = float(sum(values[i] for i in seed_indices) / n)
        seed_pos = seed_indices[-1]
        result.iloc[seed_pos] = seed
        alpha = 1.0 / n
        prev = seed
        for i in range(seed_pos + 1, len(values)):
            v = values[i]
            if v != v:  # NaN
                result.iloc[i] = prev
            else:
                prev = prev * (1.0 - alpha) + float(v) * alpha
                result.iloc[i] = prev
        return result

    def _adx_for_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy().reset_index(drop=True)
        high = g["high"]
        low = g["low"]
        prev_close = g["close"].shift(1)
        prev_high = high.shift(1)
        prev_low = low.shift(1)

        # True range
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)

        # Directional movement
        up_move = high - prev_high
        dn_move = prev_low - low
        plus_dm = up_move.where((up_move > dn_move) & (up_move > 0), 0.0)
        minus_dm = dn_move.where((dn_move > up_move) & (dn_move > 0), 0.0)

        # Wilder smooth TR, +DM, -DM
        atr_w = _wilder_smooth(tr, period)
        plus_dm_w = _wilder_smooth(plus_dm, period)
        minus_dm_w = _wilder_smooth(minus_dm, period)

        # DI lines
        plus_di = (plus_dm_w / atr_w.replace(0, float("nan"))) * 100
        minus_di = (minus_dm_w / atr_w.replace(0, float("nan"))) * 100

        # DX
        di_sum = (plus_di + minus_di).replace(0, float("nan"))
        dx = ((plus_di - minus_di).abs() / di_sum) * 100

        adx = _wilder_smooth(dx, period)

        g["adx"] = adx
        g["plus_di"] = plus_di
        g["minus_di"] = minus_di
        return g

    groups = []
    for sym, grp in frame.groupby("symbol", sort=False):
        groups.append(_adx_for_group(grp))
    result = pd.concat(groups).sort_values(["symbol", "ts_event"]).reset_index(drop=True)
    # Fill any leading NaN with a neutral value so downstream code doesn't break
    result["adx"] = result["adx"].fillna(0.0)
    result["plus_di"] = result["plus_di"].fillna(0.0)
    result["minus_di"] = result["minus_di"].fillna(0.0)
    return result
