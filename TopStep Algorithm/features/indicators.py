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


def add_atr(df: pd.DataFrame, period: int = 14, median_window: int = 50) -> pd.DataFrame:
    frame = df.copy()
    frame = frame.sort_values(["symbol", "ts_event"]).reset_index(drop=True)
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
        lambda series: series.rolling(period, min_periods=1).mean()
    )
    frame["atr_median"] = frame.groupby("symbol", group_keys=False)["atr"].transform(
        lambda series: series.rolling(median_window, min_periods=5).median()
    )
    return frame
