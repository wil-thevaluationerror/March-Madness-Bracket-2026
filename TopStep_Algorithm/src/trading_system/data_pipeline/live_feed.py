"""
TopstepX live 1-minute bar feed.

Pulls OHLCV bars from /api/History/retrieveBars, maintains a rolling
indicator-enriched DataFrame, and generates OrderIntents using the
existing strategy rules on each tick.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Callable

import pandas as pd

from trading_system.config import StrategyConfig, TopstepConnectionConfig
from trading_system.execution.topstep_live_adapter import UrlLibTopstepTransport
from trading_system.features.indicators import add_adx, add_atr, add_ema, add_vwap
from trading_system.core.domain import OrderIntent
from trading_system.strategy.rules import generate_intents

_log = logging.getLogger(__name__)

# ProjectX bar unit codes
_UNIT_MINUTE = 2


class TopstepLiveFeed:
    """
    Maintains a rolling window of 1-minute MES bars fetched from the
    TopstepX history API.  On each call to ``tick()`` it:

    1. Fetches any bars completed since the last poll.
    2. Appends them to the rolling DataFrame and recomputes all indicators.
    3. Runs ``generate_intents()`` and returns only intents whose
       ``signal_ts`` is newer than the previous tick — preventing
       duplicate submissions across heartbeat intervals.

    Parameters
    ----------
    config:
        TopstepConnectionConfig with api_base_url and credentials.
    token_provider:
        Zero-argument callable that returns the current valid JWT string.
        The live adapter already manages token refresh; pass
        ``lambda: adapter.access_token`` to share the same token.
    contract_id:
        ProjectX contract ID string, e.g. ``"CON.F.US.MES.M26"``.
    symbol:
        Internal symbol label used in the DataFrame, e.g. ``"MES"``.
    warmup_bars:
        Number of bars to fetch on initialisation to seed indicators.
        EMA(50) needs ~100+ bars to stabilise; default 150 is safe.
    rolling_window:
        Maximum number of bars kept in memory.  300 = 5 hours of 1-min
        bars, enough for a full session plus overnight context.
    """

    def __init__(
        self,
        config: TopstepConnectionConfig,
        token_provider: Callable[[], str | None],
        contract_id: str,
        symbol: str = "MES",
        *,
        warmup_bars: int = 150,
        rolling_window: int = 300,
        diagnostics_callback: Callable[[dict[str, Any]], None] | None = None,
        risk_allowed_provider: Callable[[], bool | None] | None = None,
    ) -> None:
        self.config = config
        self.token_provider = token_provider
        self.contract_id = contract_id
        self.symbol = symbol
        self.warmup_bars = warmup_bars
        self.rolling_window = rolling_window
        self.diagnostics_callback = diagnostics_callback
        self.risk_allowed_provider = risk_allowed_provider
        self.transport = UrlLibTopstepTransport()
        self._df: pd.DataFrame | None = None
        self._last_bar_ts: datetime | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """
        Fetch warmup bars and seed the indicator DataFrame.

        Returns True on success, False if no bars were returned
        (e.g. outside market hours).
        """
        _log.info("live_feed_init contract=%s warmup=%d", self.contract_id, self.warmup_bars)
        now = datetime.now(UTC)
        start = now - timedelta(minutes=self.warmup_bars + 20)
        bars = self._request_bars(start, now, limit=self.warmup_bars)
        if not bars:
            _log.warning("live_feed_init no bars returned — market may be closed")
            return False
        self._df = self._build_df(bars)
        self._df = self._enrich(self._df)
        self._last_bar_ts = self._df["ts_event"].max()
        _log.info(
            "live_feed_ready bars=%d last_bar=%s",
            len(self._df),
            self._last_bar_ts.isoformat(),
        )
        return True

    def tick(self, strategy_config: StrategyConfig) -> list[OrderIntent]:
        """
        Fetch completed bars since the last poll, update indicators,
        and return any new OrderIntents.

        Safe to call even if ``initialize()`` returned False — will
        attempt a lazy warmup on the first successful bar fetch.
        """
        if self._last_bar_ts is None:
            # Lazy init: try to seed on first successful tick
            if not self.initialize():
                return []

        assert self._last_bar_ts is not None
        fetch_start = self._last_bar_ts + timedelta(seconds=1)
        new_bars = self._request_bars(fetch_start, datetime.now(UTC), limit=10)
        if not new_bars:
            return []

        new_df = self._build_df(new_bars)
        combined = pd.concat([self._df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["ts_event", "symbol"])
        combined = combined.sort_values("ts_event").reset_index(drop=True)
        combined = combined.tail(self.rolling_window).reset_index(drop=True)
        combined = self._enrich(combined)

        self._df = combined
        prev_ts = self._last_bar_ts
        self._last_bar_ts = combined["ts_event"].max()

        # Run strategy on full enriched window; keep only intents from new bars
        risk_allowed = self.risk_allowed_provider() if self.risk_allowed_provider is not None else None
        all_intents = generate_intents(
            combined,
            strategy_config,
            diagnostics_callback=self.diagnostics_callback,
            diagnostic_since=prev_ts,
            diagnostic_context={
                "session_allowed": True,
                "risk_allowed": risk_allowed,
            },
        )
        fresh = [i for i in all_intents if i.signal_ts > prev_ts]
        if fresh:
            _log.info("live_feed_signals count=%d new_bar_ts=%s", len(fresh), self._last_bar_ts)
        return fresh

    @property
    def last_bar_ts(self) -> datetime | None:
        return self._last_bar_ts

    @property
    def is_ready(self) -> bool:
        return self._df is not None and self._last_bar_ts is not None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _request_bars(
        self,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> list[dict[str, Any]]:
        token = self.token_provider()
        if not token:
            _log.warning("live_feed_no_token skipping bar fetch")
            return []
        payload = {
            "contractId": self.contract_id,
            "live": False,
            "startTime": start.isoformat(),
            "endTime": end.isoformat(),
            "unit": _UNIT_MINUTE,
            "unitNumber": 1,
            "limit": limit,
            "includePartialBar": False,
        }
        try:
            resp = self.transport.post(
                self.config.api_base_url,
                "/api/History/retrieveBars",
                payload,
                bearer_token=token,
                timeout_seconds=self.config.request_timeout_seconds,
            )
        except RuntimeError as exc:
            _log.warning("live_feed_fetch_error %s", exc)
            return []
        bars = list(resp.get("bars") or [])
        _log.debug("live_feed_fetched bars=%d start=%s end=%s", len(bars), start.isoformat(), end.isoformat())
        return bars

    def _build_df(self, bars: list[dict[str, Any]]) -> pd.DataFrame:
        rows = []
        for bar in bars:
            # API returns UTC ISO timestamps; convert to Chicago for session logic
            ts = pd.Timestamp(bar["t"]).tz_convert("America/Chicago")
            rows.append(
                {
                    "ts_event": ts,
                    "symbol": self.symbol,
                    "open": float(bar["o"]),
                    "high": float(bar["h"]),
                    "low": float(bar["l"]),
                    "close": float(bar["c"]),
                    "volume": int(bar["v"]),
                }
            )
        df = pd.DataFrame(rows)
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=False)
        return df

    _COMPUTED_COLS = {
        "vwap", "ema_fast", "ema_slow",
        "atr", "atr_median", "atr_5min", "atr_5min_median",
        "adx", "plus_di", "minus_di",
        "session_date", "prev_close", "true_range",
        "rolling_median_volume",
    }

    @classmethod
    def _enrich(cls, df: pd.DataFrame) -> pd.DataFrame:
        # Drop any previously computed columns so indicator functions
        # start from raw OHLCV and don't produce duplicate suffix columns.
        drop_cols = [c for c in df.columns if c in cls._COMPUTED_COLS]
        df = df.drop(columns=drop_cols)
        df = add_vwap(df)   # session-anchored VWAP (resets per calendar date)
        df = add_ema(df, fast_span=20, slow_span=50)
        df = add_atr(df, period=14, median_window=50)
        df = add_adx(df, period=14)
        return df
