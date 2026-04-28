from __future__ import annotations

from typing import Any

import pandas as pd

from config import SessionConfig
from models.instruments import infer_symbol_root


def _time_in_window(value, start, end) -> bool:
    if start <= end:
        return start <= value < end
    return value >= start or value < end


def preprocess(df: pd.DataFrame, session_config: SessionConfig | None = None) -> pd.DataFrame:
    frame = df.copy()
    frame["ts_event"] = frame["ts_event"].dt.tz_convert("America/Chicago")

    config = session_config or SessionConfig()
    windows = tuple(config.session_windows)
    if windows:
        frame = frame[
            frame["ts_event"].dt.time.apply(
                lambda value: any(_time_in_window(value, window.market_open, window.exchange_close) for window in windows)
            )
        ]
    frame = frame.sort_values(["ts_event", "symbol"]).reset_index(drop=True)
    return frame


def select_primary_symbol(
    df: pd.DataFrame,
    preferred_symbol: str | None = None,
    *,
    jump_threshold_pct: float = 2.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if df.empty:
        return df.copy(), {"selected_symbol": None, "available_symbols": {}, "price_jump_flags": []}

    frame = df.copy()
    symbol_text = frame["symbol"].astype(str)
    outright = frame[~symbol_text.str.contains("-", regex=False)].copy()
    if outright.empty:
        raise ValueError("No outright futures symbols available after removing spread symbols.")

    available_symbols = outright["symbol"].value_counts().to_dict()
    if preferred_symbol is not None:
        preferred_root = infer_symbol_root(str(preferred_symbol))
        preferred_matches = {
            str(symbol): int(count)
            for symbol, count in available_symbols.items()
            if infer_symbol_root(str(symbol)) == preferred_root
        }
        if preferred_matches:
            selected_symbol = str(max(preferred_matches.items(), key=lambda item: item[1])[0])
        elif preferred_symbol in available_symbols:
            selected_symbol = str(preferred_symbol)
        else:
            raise ValueError(
                f"Preferred symbol {preferred_symbol!r} was not found in the available outright symbols."
            )
    else:
        selected_symbol = str(max(available_symbols.items(), key=lambda item: item[1])[0])

    filtered = outright[outright["symbol"] == selected_symbol].copy()
    filtered = filtered.sort_values("ts_event").reset_index(drop=True)
    filtered["prev_close"] = filtered["close"].shift(1)
    filtered["price_jump_pct"] = (
        ((filtered["close"] - filtered["prev_close"]).abs() / filtered["prev_close"].abs()) * 100.0
    )
    jump_flags = filtered[filtered["price_jump_pct"] > jump_threshold_pct][
        ["ts_event", "symbol", "close", "prev_close", "price_jump_pct"]
    ].copy()

    diagnostics = {
        "selected_symbol": selected_symbol,
        "selected_root_symbol": infer_symbol_root(selected_symbol),
        "available_symbols": available_symbols,
        "rows_before_filter": int(len(frame)),
        "rows_after_outright_filter": int(len(outright)),
        "rows_after_symbol_filter": int(len(filtered)),
        "non_outright_rows_dropped": int(len(frame) - len(outright)),
        "other_outright_rows_dropped": int(len(outright) - len(filtered)),
        "price_jump_threshold_pct": float(jump_threshold_pct),
        "price_jump_flag_count": int(len(jump_flags)),
        "price_jump_flags": [
            {
                "ts_event": row.ts_event.isoformat(),
                "symbol": str(row.symbol),
                "close": float(row.close),
                "prev_close": float(row.prev_close),
                "price_jump_pct": float(row.price_jump_pct),
            }
            for row in jump_flags.head(10).itertuples(index=False)
        ],
    }
    filtered.attrs["data_diagnostics"] = diagnostics
    return filtered.drop(columns=["prev_close", "price_jump_pct"]), diagnostics
