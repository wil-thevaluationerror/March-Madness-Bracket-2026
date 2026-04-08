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
