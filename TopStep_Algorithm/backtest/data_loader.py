from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from api.market_data import Bar

log = logging.getLogger(__name__)
UTC = ZoneInfo("UTC")

# London session bounds in UTC
_LONDON_OPEN_UTC = (8, 30)    # 03:30 ET
_LONDON_CLOSE_UTC = (13, 30)  # 08:30 ET

# Asian range bounds in UTC  (20:00–02:00 ET = 01:00–07:00 UTC on same calendar day)
_ASIAN_START_UTC = (1, 0)
_ASIAN_END_UTC = (7, 0)

# Minimum 5-minute bars required in the full 08:30-13:30 UTC London window.
_MIN_LONDON_BARS = 60

# 1h bars: look back 72 h before London open; require at least 52 (EMA-50 warmup + slope)
_LOOKBACK_HOURS = 72
_MIN_1H_BARS = 52


@dataclass
class SessionDay:
    symbol: str
    date: date
    bars_5m: list[Bar]          # full day including pre-London, oldest → newest
    bars_1h: list[Bar]          # 72h lookback from session start, oldest → newest
    london_start_idx: int       # index into bars_5m where 08:30 UTC begins
    london_end_idx: int         # index into bars_5m where 13:30 UTC ends (exclusive)


def _df_to_bars(df: pd.DataFrame, ts_col: str) -> list[Bar]:
    """Convert a OHLCV DataFrame slice to a sorted list of Bar objects."""
    bars: list[Bar] = []
    for row in df.itertuples(index=False):
        ts = getattr(row, ts_col)
        if not isinstance(ts, datetime):
            ts = pd.Timestamp(ts).to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        bars.append(
            Bar(
                timestamp=ts,
                open=float(getattr(row, "open")),
                high=float(getattr(row, "high")),
                low=float(getattr(row, "low")),
                close=float(getattr(row, "close")),
                volume=int(getattr(row, "volume")),
            )
        )
    bars.sort(key=lambda b: b.timestamp)
    return bars


def _parent_symbol(symbol: str) -> str:
    """Return Databento parent-symbol form for a configured symbol."""
    return f"{symbol.upper()}.FUT"


def _metadata_symbols(store: object) -> list[str]:
    metadata = getattr(store, "metadata", None)
    raw_symbols = getattr(metadata, "symbols", None)
    return [str(s).upper() for s in raw_symbols] if raw_symbols else []


def _filter_requested_symbol(
    df: pd.DataFrame,
    *,
    path: str,
    requested_symbol: str,
    available_symbols: list[str],
) -> pd.DataFrame:
    """Validate and filter rows to the requested outright futures parent."""
    expected_parent = _parent_symbol(requested_symbol)
    if available_symbols and expected_parent not in available_symbols:
        raise ValueError(
            f"{path} does not contain {expected_parent}; "
            f"available parent symbols: {available_symbols}"
        )

    if "symbol" not in df.columns:
        if available_symbols == [expected_parent]:
            return df
        raise ValueError(
            f"Cannot validate rows for {requested_symbol} in {path}: "
            "Databento output has no symbol column."
        )

    raw_symbol = df["symbol"].astype(str).str.upper()
    prefix = requested_symbol.upper()
    # Parent queries can include calendar spreads. The strategy trades outright
    # futures only, so remove spread rows before front-month selection.
    mask = raw_symbol.str.startswith(prefix) & ~raw_symbol.str.contains("-", regex=False)
    filtered = df.loc[mask].copy()
    if filtered.empty:
        raise ValueError(
            f"{path} contains {expected_parent} metadata but no outright "
            f"{requested_symbol} rows after symbol filtering."
        )
    return filtered


def _load_raw(path: str, symbol: str) -> pd.DataFrame:
    """Load a .dbn.zst file and return a front-month selected, UTC DataFrame.

    Front-month selection: for each timestamp group, keep the row with the
    highest volume after filtering to the requested outright futures parent.

    Price scaling: databento returns float prices when price_type='float'
    (the default).  A sanity check detects fixed-point encoding (prices > 1e6
    for FX) and scales by 1e-9 if needed.
    """
    try:
        import databento as db
    except ImportError as exc:
        raise ImportError(
            "databento is required for data loading. Install with: pip install databento"
        ) from exc

    store = db.DBNStore.from_file(path)
    available_symbols = _metadata_symbols(store)
    df = store.to_df()  # price_type='float', pretty_ts=True, map_symbols=True by default

    # Normalise index → columns
    if df.index.name in ("ts_recv", "ts_event"):
        df = df.reset_index()

    df = _filter_requested_symbol(
        df,
        path=path,
        requested_symbol=symbol,
        available_symbols=available_symbols,
    )

    # Determine timestamp column
    ts_col = "ts_event" if "ts_event" in df.columns else df.columns[0]
    if ts_col not in df.columns:
        raise ValueError(f"Cannot find ts_event column in {path}. Columns: {list(df.columns)}")

    # Ensure UTC-aware timestamps
    if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    elif df[ts_col].dt.tz is None:
        df[ts_col] = df[ts_col].dt.tz_localize("UTC")
    else:
        df[ts_col] = df[ts_col].dt.tz_convert("UTC")

    # Price scaling guard: if prices look like fixed-point integers (> 1e6 for FX)
    for col in ("open", "high", "low", "close"):
        if col in df.columns and df[col].median() > 1_000_000:
            log.debug("Detected fixed-point prices in %s; dividing by 1e9", path)
            for c in ("open", "high", "low", "close"):
                df[c] = df[c] / 1e9
            break

    # Front-month selection: keep row with highest volume per timestamp
    df = df.loc[df.groupby(ts_col)["volume"].idxmax()]
    df = df.sort_values(ts_col).reset_index(drop=True)

    return df, ts_col


def _resample_to_5m(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """Resample 1-minute OHLCV bars to 5-minute bars."""
    df_indexed = df.set_index(ts_col)[["open", "high", "low", "close", "volume"]]
    df_5m = df_indexed.resample("5min").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["close"])
    return df_5m.reset_index().rename(columns={ts_col: "ts_event", df_5m.index.name: "ts_event"})


def _resample_to_1h(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """Resample 1-minute OHLCV bars to 1-hour bars."""
    df_indexed = df.set_index(ts_col)[["open", "high", "low", "close", "volume"]]
    df_1h = df_indexed.resample("1h").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["close"])
    return df_1h.reset_index().rename(columns={df_1h.index.name: "ts_event"})


class DataLoader:
    """Loads Databento .dbn.zst bar files and builds SessionDay objects.

    Parameters
    ----------
    path_1m:
        Path to the 1-minute OHLCV .dbn.zst file for *symbol*.
    symbol:
        Instrument symbol (must be in config.INSTRUMENTS).
    path_1h:
        Optional path to a pre-built 1-hour OHLCV .dbn.zst file.
        If not provided, 1h bars are resampled from the 1m file.
    """

    def __init__(self, path_1m: str, symbol: str, path_1h: str | None = None) -> None:
        self._path_1m = path_1m
        self._path_1h = path_1h
        self._symbol = symbol
        self._df_5m: pd.DataFrame | None = None
        self._df_1h: pd.DataFrame | None = None

    def load(self) -> None:
        """Read and resample bar data from disk.  Call before build_sessions()."""
        log.info("Loading 1m data for %s from %s", self._symbol, self._path_1m)
        df_1m, ts_col = _load_raw(self._path_1m, self._symbol)

        self._df_5m = _resample_to_5m(df_1m, ts_col)
        log.info("Resampled to 5m: %d bars", len(self._df_5m))

        if self._path_1h is not None:
            log.info("Loading 1h data for %s from %s", self._symbol, self._path_1h)
            df_1h_raw, ts_col_1h = _load_raw(self._path_1h, self._symbol)
            self._df_1h = df_1h_raw.rename(columns={ts_col_1h: "ts_event"})[
                ["ts_event", "open", "high", "low", "close", "volume"]
            ]
        else:
            self._df_1h = _resample_to_1h(df_1m, ts_col)
        log.info("1h bars available: %d", len(self._df_1h))

    def build_sessions(self) -> dict[date, SessionDay]:
        """Build one SessionDay per trading date that has a valid London window.

        A session is included only if:
        - At least ``_MIN_LONDON_BARS`` 5m bars fall in 08:30–13:30 UTC.
        - At least ``_MIN_1H_BARS`` 1h bars exist in the 72h lookback window.

        Returns a mapping from session date → SessionDay, sorted by date.
        """
        if self._df_5m is None or self._df_1h is None:
            raise RuntimeError("Call load() before build_sessions()")

        all_5m = _df_to_bars(self._df_5m, "ts_event")
        all_1h = _df_to_bars(self._df_1h, "ts_event")

        # Identify all candidate London session dates
        london_dates: set[date] = set()
        for bar in all_5m:
            h, m = bar.timestamp.hour, bar.timestamp.minute
            if (h, m) >= _LONDON_OPEN_UTC and (h, m) < _LONDON_CLOSE_UTC:
                london_dates.add(bar.timestamp.date())

        sessions: dict[date, SessionDay] = {}

        for session_date in sorted(london_dates):
            london_open_dt = datetime(
                session_date.year, session_date.month, session_date.day,
                _LONDON_OPEN_UTC[0], _LONDON_OPEN_UTC[1], tzinfo=UTC,
            )
            london_close_dt = datetime(
                session_date.year, session_date.month, session_date.day,
                _LONDON_CLOSE_UTC[0], _LONDON_CLOSE_UTC[1], tzinfo=UTC,
            )

            # bars_5m: previous-day 16:00 UTC through London close + 1h buffer
            # Ensures the Asian-range window (01:00–07:00 UTC on session_date) is included.
            prev_day = session_date - timedelta(days=1)
            bars_5m_start = datetime(
                prev_day.year, prev_day.month, prev_day.day, 16, 0, tzinfo=UTC
            )
            bars_5m_end = london_close_dt + timedelta(hours=1)

            bars_5m = [
                b for b in all_5m
                if bars_5m_start <= b.timestamp < bars_5m_end
            ]

            # Find London window indices
            london_start_idx = next(
                (i for i, b in enumerate(bars_5m) if b.timestamp >= london_open_dt), None
            )
            london_end_idx = next(
                (i for i, b in enumerate(bars_5m) if b.timestamp >= london_close_dt),
                len(bars_5m),
            )

            if london_start_idx is None:
                continue

            london_bars_count = london_end_idx - london_start_idx
            if london_bars_count < _MIN_LONDON_BARS:
                log.debug("Skipping %s: only %d London bars", session_date, london_bars_count)
                continue

            # bars_1h: 72h before London open up to (not including) London close timestamp
            lookback_start = london_open_dt - timedelta(hours=_LOOKBACK_HOURS)
            bars_1h = [
                b for b in all_1h
                if lookback_start <= b.timestamp < london_close_dt
            ]
            bars_1h.sort(key=lambda b: b.timestamp)

            if len(bars_1h) < _MIN_1H_BARS:
                log.debug(
                    "Skipping %s: only %d 1h bars (need %d for EMA-50 warmup)",
                    session_date, len(bars_1h), _MIN_1H_BARS,
                )
                continue

            sessions[session_date] = SessionDay(
                symbol=self._symbol,
                date=session_date,
                bars_5m=bars_5m,
                bars_1h=bars_1h,
                london_start_idx=london_start_idx,
                london_end_idx=london_end_idx,
            )

        log.info("Built %d sessions for %s", len(sessions), self._symbol)
        return sessions
