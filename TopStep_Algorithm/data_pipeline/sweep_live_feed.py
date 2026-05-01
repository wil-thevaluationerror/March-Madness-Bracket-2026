"""London-sweep live bar feed for the SignalEngine.

Fetches 5-minute and 1-hour bars from the TopstepX history API,
maintains rolling windows, and drives ``SignalEngine.process_bar()``
on every new 5-minute bar during the London session (08:30–13:30 UTC).

When a ``TradeSignal`` is produced it is immediately converted to an
``OrderIntent`` via ``trade_signal_to_intent()`` and returned to the
caller (``run_trader.run_loop``) for submission to the execution engine.

Design notes
------------
* A **fresh** ``SignalEngine`` is created for each calendar date —
  this matches the backtest simulator's per-session behaviour and ensures
  the sweep-detection state machine never leaks across sessions.
* 1h bar lookback: 80 bars (enough for EMA-50 + slope warmup + bisect
  no-lookahead buffer).  The feed fetches 80 h = ~3.3 days of 1h bars
  on init, then appends completed bars on each tick.
* The feed is **session-scoped**: ``tick()`` is a no-op outside the
  London window (08:30–13:30 UTC) so it is safe to call every minute.
"""
from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta
from typing import Any, Callable

from api.market_data import Bar
from backtest.config import StrategyConfig, TopstepConnectionConfig
from execution.topstep_live_adapter import UrlLibTopstepTransport
from models.orders import OrderIntent
from strategy.intent_bridge import trade_signal_to_intent_pair
from strategy.signal import SignalConfig, SignalEngine

_log = logging.getLogger(__name__)

# ProjectX bar unit code: 2 = minute
_UNIT_MINUTE = 2

# London session bounds (UTC)
_LONDON_OPEN_H = 8
_LONDON_OPEN_M = 30
_LONDON_CLOSE_H = 13
_LONDON_CLOSE_M = 30

# How many 1h bars to fetch on initialisation (EMA-50 warmup + slope buffer)
_WARMUP_1H_BARS = 80

# Maximum 5m bars kept in memory per session (~6 h of 5-min bars)
_ROLLING_5M = 72

# Maximum 1h bars kept in memory (well above EMA-50 warmup)
_ROLLING_1H = 100


def _api_bar_to_bar(api_bar: dict[str, Any]) -> Bar:
    """Convert a raw ProjectX bar dict to the internal Bar namedtuple."""
    ts = datetime.fromisoformat(api_bar["t"])
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    else:
        ts = ts.astimezone(UTC)
    return Bar(
        timestamp=ts,
        open=float(api_bar["o"]),
        high=float(api_bar["h"]),
        low=float(api_bar["l"]),
        close=float(api_bar["c"]),
        volume=int(api_bar["v"]),
    )


def _is_london_session(now_utc: datetime) -> bool:
    h, m = now_utc.hour, now_utc.minute
    after_open = (h, m) >= (_LONDON_OPEN_H, _LONDON_OPEN_M)
    before_close = (h, m) < (_LONDON_CLOSE_H, _LONDON_CLOSE_M)
    return after_open and before_close


class SweepLiveFeed:
    """Live 5m/1h bar feed that drives the London-sweep ``SignalEngine``.

    Parameters
    ----------
    config:
        ``TopstepConnectionConfig`` (api_base_url + credentials).
    token_provider:
        Zero-argument callable returning the current valid JWT string.
        Share the adapter's token: ``lambda: adapter.access_token``.
    contract_id:
        ProjectX contract ID, e.g. ``"CON.F.US.6B.M26"``.
    symbol:
        Internal symbol label (must be in ``config.INSTRUMENTS``).
    strategy_config:
        ``StrategyConfig`` from the active profile.  Used to build the
        ``SignalConfig`` that tunes ADX, ATR floor, target multiple, etc.
    base_qty:
        Number of contracts per signal (from ``config.strategy.base_qty``).
        The risk engine can reduce this further via drawdown tiers.
    """

    def __init__(
        self,
        config: TopstepConnectionConfig,
        token_provider: Callable[[], str | None],
        contract_id: str,
        symbol: str,
        strategy_config: StrategyConfig,
        base_qty: int = 1,
        position_active_provider: Callable[[], bool] | None = None,
    ) -> None:
        self._conn = config
        self._token_provider = token_provider
        self._contract_id = contract_id
        self._symbol = symbol
        self._qty = max(1, base_qty)
        self._transport = UrlLibTopstepTransport()

        self._signal_cfg = SignalConfig(
            target_atr_multiple=strategy_config.target_atr_multiple,
            breakeven_trigger_atr=strategy_config.breakeven_trigger_atr,
            adx_min_threshold=strategy_config.adx_min_threshold,
            atr_min_pct=strategy_config.atr_min_pct,
            ema_trend_persistence_bars=strategy_config.ema_trend_persistence_bars,
        )

        self._position_active_provider = position_active_provider

        self._bars_5m: list[Bar] = []
        self._bars_1h: list[Bar] = []
        self._last_5m_ts: datetime | None = None
        self._last_1h_ts: datetime | None = None

        # Per-session state: engine resets each calendar date
        self._session_date: date | None = None
        self._engine: SignalEngine | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """Fetch warmup 1h bars and the last hour of 5m bars.

        Returns True if at least some bars were retrieved.
        """
        now = datetime.now(UTC)
        _log.info("sweep_feed_init contract=%s symbol=%s", self._contract_id, self._symbol)

        # 1h warmup
        h1_start = now - timedelta(hours=_WARMUP_1H_BARS + 2)
        raw_1h = self._request_bars(h1_start, now, unit_number=60)
        self._bars_1h = [_api_bar_to_bar(b) for b in raw_1h]
        self._bars_1h.sort(key=lambda b: b.timestamp)
        if self._bars_1h:
            self._last_1h_ts = self._bars_1h[-1].timestamp

        # 5m warmup (last 2 h to seed Asian-range detection)
        m5_start = now - timedelta(hours=2)
        raw_5m = self._request_bars(m5_start, now, unit_number=5)
        self._bars_5m = [_api_bar_to_bar(b) for b in raw_5m]
        self._bars_5m.sort(key=lambda b: b.timestamp)
        if self._bars_5m:
            self._last_5m_ts = self._bars_5m[-1].timestamp

        _log.info(
            "sweep_feed_ready 5m_bars=%d 1h_bars=%d",
            len(self._bars_5m),
            len(self._bars_1h),
        )
        return bool(self._bars_5m or self._bars_1h)

    def tick(self) -> list[OrderIntent]:
        """Fetch any new completed bars, run SignalEngine, return intents.

        This method is safe to call every minute; it is a no-op outside
        the London session window (08:30–13:30 UTC).
        """
        now = datetime.now(UTC)
        if not _is_london_session(now):
            return []

        if self._last_5m_ts is None:
            if not self.initialize():
                return []

        # Fetch new 5m bars
        fetch_start_5m = (self._last_5m_ts + timedelta(seconds=1)) if self._last_5m_ts else (now - timedelta(hours=1))
        raw_5m = self._request_bars(fetch_start_5m, now, unit_number=5)
        new_5m = sorted([_api_bar_to_bar(b) for b in raw_5m], key=lambda b: b.timestamp)

        # Fetch new 1h bars
        fetch_start_1h = (self._last_1h_ts + timedelta(seconds=1)) if self._last_1h_ts else (now - timedelta(hours=_WARMUP_1H_BARS + 2))
        raw_1h = self._request_bars(fetch_start_1h, now, unit_number=60)
        new_1h = sorted([_api_bar_to_bar(b) for b in raw_1h], key=lambda b: b.timestamp)

        # Append and trim rolling windows
        self._bars_5m.extend(new_5m)
        self._bars_5m = sorted(
            {b.timestamp: b for b in self._bars_5m}.values(),
            key=lambda b: b.timestamp,
        )[-_ROLLING_5M:]

        self._bars_1h.extend(new_1h)
        self._bars_1h = sorted(
            {b.timestamp: b for b in self._bars_1h}.values(),
            key=lambda b: b.timestamp,
        )[-_ROLLING_1H:]

        if self._bars_5m:
            self._last_5m_ts = self._bars_5m[-1].timestamp
        if self._bars_1h:
            self._last_1h_ts = self._bars_1h[-1].timestamp

        if not new_5m:
            return []

        # Open position guard — skip signal generation while a position is active
        if self._position_active_provider is not None and self._position_active_provider():
            return []

        intents: list[OrderIntent] = []
        for bar in new_5m:
            # Roll SignalEngine on the incoming bar's session date, not wall-clock
            # now. This keeps replay/test feeds and delayed history fetches aligned
            # with the data actually being evaluated.
            bar_date = bar.timestamp.date()
            if self._session_date != bar_date:
                _log.info("sweep_feed_new_session date=%s symbol=%s", bar_date, self._symbol)
                self._session_date = bar_date
                self._engine = SignalEngine(self._symbol, self._signal_cfg)

            assert self._engine is not None

            # Build the 5m slice up to and including this bar
            bars_5m_slice = [b for b in self._bars_5m if b.timestamp <= bar.timestamp]

            # No-lookahead: exclude any 1h bar whose open timestamp >= current 5m bar timestamp
            bars_1h_slice = [b for b in self._bars_1h if b.timestamp < bar.timestamp]

            signal = self._engine.process_bar(bars_5m_slice, bars_1h_slice)
            if signal is not None:
                new_intents = trade_signal_to_intent_pair(signal, self._qty, now=bar.timestamp)
                intents.extend(new_intents)
                _log.info(
                    "sweep_feed_signal symbol=%s dir=%s entry=%.5f sl=%.5f tp1=%.5f",
                    signal.symbol,
                    signal.direction,
                    signal.entry_price,
                    signal.stop_price,
                    signal.tp1_price,
                )

        return intents

    @property
    def is_ready(self) -> bool:
        return self._last_5m_ts is not None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _request_bars(
        self,
        start: datetime,
        end: datetime,
        unit_number: int,
    ) -> list[dict[str, Any]]:
        token = self._token_provider()
        if not token:
            _log.warning("sweep_feed_no_token skipping bar fetch unit_number=%d", unit_number)
            return []
        payload = {
            "contractId": self._contract_id,
            "live": False,
            "startTime": start.isoformat(),
            "endTime": end.isoformat(),
            "unit": _UNIT_MINUTE,
            "unitNumber": unit_number,
            "limit": 200,
            "includePartialBar": False,
        }
        try:
            resp = self._transport.post(
                self._conn.api_base_url,
                "/api/History/retrieveBars",
                payload,
                bearer_token=token,
                timeout_seconds=self._conn.request_timeout_seconds,
            )
        except RuntimeError as exc:
            _log.warning("sweep_feed_fetch_error unit_number=%d err=%s", unit_number, exc)
            return []
        bars = list(resp.get("bars") or [])
        _log.debug(
            "sweep_feed_fetched unit_number=%d bars=%d start=%s",
            unit_number, len(bars), start.isoformat(),
        )
        return bars
