"""TDD tests for live-readiness gaps 1-4.

Gap 1 — TP2 leg execution:        trade_signal_to_intent_pair()
Gap 2 — OB confluence filter:     SignalConfig.allowed_confluence_types gate
Gap 3 — Contract resolution:      _build_sweep_feed handles cache miss
Gap 4 — Open position guard:      SweepLiveFeed.tick() skips signals when position active
"""
from __future__ import annotations

import sys
import types
from datetime import UTC, date, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from strategy.confluence import ConfluenceResult
from strategy.intent_bridge import trade_signal_to_intent, trade_signal_to_intent_pair
from strategy.signal import SignalConfig, TradeSignal


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_signal(
    *,
    direction: str = "BUY",
    symbol: str = "6B",
    entry: float = 1.2500,
    stop: float = 1.2480,
    tp1: float = 1.2540,
    tp2: float = 1.2560,
    confluence_type: str = "FVG",
    atr14: float = 0.0010,
    ema_slope: float = 0.0002,
) -> TradeSignal:
    return TradeSignal(
        symbol=symbol,
        direction=direction,
        entry_price=entry,
        stop_price=stop,
        tp1_price=tp1,
        tp2_price=tp2,
        confluence=ConfluenceResult(
            confluence_type=confluence_type,
            description=f"{confluence_type}@{entry:.5f}",
        ),
        atr14=atr14,
        ema_slope=ema_slope,
    )


# ===========================================================================
# GAP 1 — trade_signal_to_intent_pair()
# ===========================================================================

class TestTP2Split:
    """trade_signal_to_intent_pair must split qty and route to correct targets."""

    def test_qty_1_returns_only_tp1_intent(self) -> None:
        """Cannot split a single contract — only TP1 intent should be returned."""
        intents = trade_signal_to_intent_pair(_make_signal(), qty=1)
        assert len(intents) == 1
        assert intents[0].target_price == pytest.approx(1.2540)

    def test_qty_2_splits_evenly(self) -> None:
        intents = trade_signal_to_intent_pair(_make_signal(), qty=2)
        assert len(intents) == 2
        tp1, tp2 = intents
        assert tp1.qty == 1
        assert tp2.qty == 1

    def test_qty_5_floor_split(self) -> None:
        """floor(5/2)=2 for TP1, remainder 3 for TP2."""
        intents = trade_signal_to_intent_pair(_make_signal(), qty=5)
        assert len(intents) == 2
        tp1, tp2 = intents
        assert tp1.qty == 2
        assert tp2.qty == 3
        assert tp1.qty + tp2.qty == 5

    def test_tp1_intent_targets_tp1_price(self) -> None:
        intents = trade_signal_to_intent_pair(_make_signal(tp1=1.2540, tp2=1.2560), qty=4)
        assert intents[0].target_price == pytest.approx(1.2540)

    def test_tp2_intent_targets_tp2_price(self) -> None:
        intents = trade_signal_to_intent_pair(_make_signal(tp1=1.2540, tp2=1.2560), qty=4)
        assert intents[1].target_price == pytest.approx(1.2560)

    def test_both_intents_share_same_stop(self) -> None:
        intents = trade_signal_to_intent_pair(_make_signal(stop=1.2480), qty=4)
        assert all(i.stop_price == pytest.approx(1.2480) for i in intents)

    def test_both_intents_share_same_entry(self) -> None:
        intents = trade_signal_to_intent_pair(_make_signal(entry=1.2500), qty=4)
        assert all(i.entry_price == pytest.approx(1.2500) for i in intents)

    def test_intent_ids_are_unique_with_shared_pair_prefix(self) -> None:
        intents = trade_signal_to_intent_pair(_make_signal(), qty=4)
        ids = [i.intent_id for i in intents]
        # Must be unique
        assert len(set(ids)) == 2
        # TP1 and TP2 share a pair root but are distinguished
        assert "tp1" in ids[0]
        assert "tp2" in ids[1]

    def test_tp2_metadata_marks_leg(self) -> None:
        intents = trade_signal_to_intent_pair(_make_signal(), qty=4)
        assert intents[1].metadata.get("leg") == "tp2"

    def test_qty_0_floored_to_1_gives_single_intent(self) -> None:
        intents = trade_signal_to_intent_pair(_make_signal(), qty=0)
        assert len(intents) == 1
        assert intents[0].qty == 1

    def test_now_override_propagates_to_all_intents(self) -> None:
        fixed = datetime(2026, 1, 2, 9, 0, tzinfo=UTC)
        intents = trade_signal_to_intent_pair(_make_signal(), qty=4, now=fixed)
        assert all(i.signal_ts == fixed for i in intents)


# ===========================================================================
# GAP 2 — OB-only confluence filter via SignalConfig.allowed_confluence_types
# ===========================================================================

class TestConfluenceFilter:
    """SignalEngine must block signals whose confluence type is not in the allowed set."""

    def test_signal_config_default_allows_all_confluence_types(self) -> None:
        cfg = SignalConfig()
        # Default: all types allowed (no restriction)
        assert "OB" in cfg.allowed_confluence_types
        assert "FVG" in cfg.allowed_confluence_types
        assert "OB+FVG" in cfg.allowed_confluence_types

    def test_signal_config_can_restrict_to_fvg_only(self) -> None:
        cfg = SignalConfig(allowed_confluence_types=frozenset({"FVG", "OB+FVG"}))
        assert "OB" not in cfg.allowed_confluence_types

    def _run_engine_with_confluence(
        self, confluence_type: str, allowed: frozenset[str]
    ) -> object:
        """Drive SignalEngine.process_bar with a patched confluence result."""
        from strategy.asian_range import AsianRange
        from strategy.signal import SignalEngine
        from strategy.sweep_detector import SweepSignal
        from api.market_data import Bar

        cfg = SignalConfig(allowed_confluence_types=allowed)
        engine = SignalEngine("6B", cfg)

        confluence_result = ConfluenceResult(
            confluence_type=confluence_type, description="test"
        )
        sweep = SweepSignal(direction="BUY", level=1.249, candles_remaining=3)

        bar = Bar(datetime(2026, 1, 2, 9, 0, tzinfo=UTC), 1.25, 1.251, 1.249, 1.25, 100)
        bars_5m = [bar] * 20
        bars_1h = [bar] * 52

        with patch("strategy.signal.find_confluence", return_value=confluence_result), \
             patch("strategy.signal._compute_atr14", return_value=0.001), \
             patch("strategy.signal._rolling_mean_atr", return_value=0.001), \
             patch("strategy.signal.compute_asian_range", return_value=AsianRange(high=1.252, low=1.248)), \
             patch("strategy.signal._compute_adx14", return_value=30.0), \
             patch("strategy.signal._ema", return_value=[1.24 + i * 0.0001 for i in range(52)]), \
             patch("strategy.signal._ema_slope_aligned", return_value=True), \
             patch.object(engine._detector, "update"), \
             patch.object(engine._detector, "tick"), \
             patch.object(engine._detector, "clear"), \
             patch.object(type(engine._detector), "active",
                          new_callable=lambda: property(lambda self: sweep)):  # type: ignore[return-value]
            return engine.process_bar(bars_5m, bars_1h)

    def test_signal_engine_skips_ob_when_restricted(self) -> None:
        """When allowed_confluence_types excludes OB, process_bar returns None for OB signals."""
        result = self._run_engine_with_confluence(
            "OB", allowed=frozenset({"FVG", "OB+FVG"})
        )
        assert result is None, "OB signal should be blocked when allowed_confluence_types excludes OB"

    def test_signal_engine_allows_fvg_when_restricted_to_fvg(self) -> None:
        """FVG signals pass through when allowed_confluence_types = {FVG, OB+FVG}."""
        result = self._run_engine_with_confluence(
            "FVG", allowed=frozenset({"FVG", "OB+FVG"})
        )
        assert result is not None, "FVG signal should pass through when FVG is allowed"


# ===========================================================================
# GAP 3 — Contract resolution in _build_sweep_feed
# ===========================================================================

class TestSweepFeedContractResolution:
    """_build_sweep_feed returns None gracefully when contract resolution fails."""

    def _make_engine_with_inner(self, contract_in_cache: bool) -> MagicMock:
        """Build a minimal mock ExecutionEngine with an inner LiveTopstepAdapter."""
        from execution.topstep_live_adapter import LiveTopstepAdapter

        inner = MagicMock(spec=LiveTopstepAdapter)
        inner.access_token = "tok-abc"
        inner.config = MagicMock()
        inner.config.request_timeout_seconds = 5

        if contract_in_cache:
            inner.contract_cache_by_symbol = {"6B": {"id": "CON.F.US.6B.H26"}}
        else:
            inner.contract_cache_by_symbol = {}
            inner._resolve_contract.side_effect = RuntimeError("not found")

        from execution.topstepx_adapter import TopstepXAdapter
        adapter = MagicMock(spec=TopstepXAdapter)
        adapter._impl = inner

        engine = MagicMock()
        engine.adapter = adapter
        engine.config = MagicMock()
        engine.config.strategy.preferred_symbol = "6B"
        engine.config.strategy.default_symbol = "6B"
        engine.config.strategy.base_qty = 1
        engine.config.strategy.target_atr_multiple = 2.0
        engine.config.strategy.breakeven_trigger_atr = 0.0
        engine.config.strategy.adx_min_threshold = 25.0
        engine.config.strategy.atr_min_pct = 0.85
        engine.config.strategy.ema_trend_persistence_bars = 3
        return engine

    def test_returns_feed_when_contract_in_cache(self) -> None:
        import scripts.run_trader as rt
        engine = self._make_engine_with_inner(contract_in_cache=True)
        with patch("data_pipeline.sweep_live_feed.SweepLiveFeed.initialize", return_value=True):
            feed = rt._build_sweep_feed(engine)
        assert feed is not None

    def test_returns_none_when_contract_resolution_fails(self) -> None:
        import scripts.run_trader as rt
        engine = self._make_engine_with_inner(contract_in_cache=False)
        feed = rt._build_sweep_feed(engine)
        assert feed is None

    def test_returns_none_when_adapter_is_not_live(self) -> None:
        """If adapter is not LiveTopstepAdapter (e.g. MOCK mode), feed should be None."""
        import scripts.run_trader as rt
        engine = MagicMock()
        # _impl is not a LiveTopstepAdapter
        engine.adapter._impl = object()
        feed = rt._build_sweep_feed(engine)
        assert feed is None


# ===========================================================================
# GAP 4 — Open position guard in SweepLiveFeed.tick()
# ===========================================================================

class TestOpenPositionGuard:
    """SweepLiveFeed must not emit new intents when a position is already active."""

    def _make_feed(self, position_active: bool) -> object:
        from backtest.config import TopstepConnectionConfig, StrategyConfig
        from data_pipeline.sweep_live_feed import SweepLiveFeed

        cfg = TopstepConnectionConfig()
        strat = StrategyConfig()
        feed = SweepLiveFeed(
            config=cfg,
            token_provider=lambda: "fake-token",
            contract_id="CON.F.US.6B.H26",
            symbol="6B",
            strategy_config=strat,
            base_qty=2,
            position_active_provider=lambda: position_active,
        )
        return feed

    def test_tick_returns_empty_when_position_active(self) -> None:
        feed = self._make_feed(position_active=True)
        # Seed minimal state so it passes the initialization check
        from api.market_data import Bar
        bar = Bar(datetime(2026, 1, 2, 9, 0, tzinfo=UTC), 1.25, 1.251, 1.249, 1.25, 100)
        feed._bars_5m = [bar]
        feed._bars_1h = [bar] * 52
        feed._last_5m_ts = bar.timestamp
        feed._session_date = date(2026, 1, 2)
        feed._engine = MagicMock()
        feed._engine.process_bar.return_value = _make_signal()  # Would fire a signal

        # Patch _request_bars so no API calls are made, but return a new bar
        new_bar = Bar(datetime(2026, 1, 2, 9, 5, tzinfo=UTC), 1.25, 1.251, 1.249, 1.251, 100)
        feed._request_bars = MagicMock(return_value=[
            {"t": new_bar.timestamp.isoformat(), "o": 1.25, "h": 1.251, "l": 1.249, "c": 1.251, "v": 100}
        ])

        # london session UTC — patch _is_london_session
        with patch("data_pipeline.sweep_live_feed._is_london_session", return_value=True):
            intents = feed.tick()

        assert intents == [], "Should yield no intents when position is already active"

    def test_tick_emits_intent_when_no_active_position(self) -> None:
        feed = self._make_feed(position_active=False)
        from api.market_data import Bar
        bar = Bar(datetime(2026, 1, 2, 9, 0, tzinfo=UTC), 1.25, 1.251, 1.249, 1.25, 100)
        feed._bars_5m = [bar]
        feed._bars_1h = [bar] * 52
        feed._last_5m_ts = bar.timestamp
        feed._session_date = date(2026, 1, 2)
        feed._engine = MagicMock()
        feed._engine.process_bar.return_value = _make_signal()

        new_bar = Bar(datetime(2026, 1, 2, 9, 5, tzinfo=UTC), 1.25, 1.251, 1.249, 1.251, 100)
        feed._request_bars = MagicMock(return_value=[
            {"t": new_bar.timestamp.isoformat(), "o": 1.25, "h": 1.251, "l": 1.249, "c": 1.251, "v": 100}
        ])

        with patch("data_pipeline.sweep_live_feed._is_london_session", return_value=True):
            intents = feed.tick()

        assert len(intents) > 0, "Should emit intent(s) when position is not active"

    def test_feed_without_provider_still_emits(self) -> None:
        """position_active_provider=None means no guard — always generate signals."""
        from backtest.config import TopstepConnectionConfig, StrategyConfig
        from data_pipeline.sweep_live_feed import SweepLiveFeed

        feed = SweepLiveFeed(
            config=TopstepConnectionConfig(),
            token_provider=lambda: "fake-token",
            contract_id="CON.F.US.6B.H26",
            symbol="6B",
            strategy_config=StrategyConfig(),
            base_qty=2,
            position_active_provider=None,  # no guard
        )
        from api.market_data import Bar
        bar = Bar(datetime(2026, 1, 2, 9, 0, tzinfo=UTC), 1.25, 1.251, 1.249, 1.25, 100)
        feed._bars_5m = [bar]
        feed._bars_1h = [bar] * 52
        feed._last_5m_ts = bar.timestamp
        feed._session_date = date(2026, 1, 2)
        feed._engine = MagicMock()
        feed._engine.process_bar.return_value = _make_signal()

        new_bar = Bar(datetime(2026, 1, 2, 9, 5, tzinfo=UTC), 1.25, 1.251, 1.249, 1.251, 100)
        feed._request_bars = MagicMock(return_value=[
            {"t": new_bar.timestamp.isoformat(), "o": 1.25, "h": 1.251, "l": 1.249, "c": 1.251, "v": 100}
        ])

        with patch("data_pipeline.sweep_live_feed._is_london_session", return_value=True):
            intents = feed.tick()

        assert len(intents) > 0
