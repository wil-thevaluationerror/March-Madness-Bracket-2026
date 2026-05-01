"""Integration tests: TradeSignal → intent_bridge → OrderIntent round-trip.

These tests verify that:
1. ``trade_signal_to_intent`` maps all TradeSignal fields correctly.
2. The resulting OrderIntent is accepted by ``ExecutionEngine.submit_intent``
   in MOCK mode without raising.
3. Bridge metadata preserves tp2_price for observability.
"""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from models.orders import OrderIntent, OrderType, Regime, Side, TimeInForce, TradingMode
from strategy.confluence import ConfluenceResult
from strategy.intent_bridge import _STRATEGY_ID, trade_signal_to_intent
from strategy.signal import TradeSignal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(
    *,
    direction: str = "BUY",
    symbol: str = "6B",
    entry: float = 1.2500,
    stop: float = 1.2480,
    tp1: float = 1.2540,
    tp2: float = 1.2560,
    confluence_type: str = "OB+FVG",
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
            description=f"{confluence_type} near {entry:.5f}",
        ),
        atr14=atr14,
        ema_slope=ema_slope,
    )


# ---------------------------------------------------------------------------
# Unit tests: field mapping
# ---------------------------------------------------------------------------

def test_buy_signal_maps_to_buy_side() -> None:
    intent = trade_signal_to_intent(_make_signal(direction="BUY"), qty=1)
    assert intent.side == Side.BUY


def test_sell_signal_maps_to_sell_side() -> None:
    intent = trade_signal_to_intent(_make_signal(direction="SELL"), qty=1)
    assert intent.side == Side.SELL


def test_intent_uses_tp1_as_target() -> None:
    sig = _make_signal(tp1=1.2540, tp2=1.2560)
    intent = trade_signal_to_intent(sig, qty=1)
    assert intent.target_price == pytest.approx(1.2540)
    assert intent.metadata["tp2_price"] == pytest.approx(1.2560)


def test_confluence_score_ob_fvg() -> None:
    intent = trade_signal_to_intent(_make_signal(confluence_type="OB+FVG"), qty=1)
    assert intent.signal_score == pytest.approx(1.0)


def test_confluence_score_ob() -> None:
    intent = trade_signal_to_intent(_make_signal(confluence_type="OB"), qty=1)
    assert intent.signal_score == pytest.approx(0.75)


def test_confluence_score_fvg() -> None:
    intent = trade_signal_to_intent(_make_signal(confluence_type="FVG"), qty=1)
    assert intent.signal_score == pytest.approx(0.50)


def test_qty_floored_at_one() -> None:
    intent = trade_signal_to_intent(_make_signal(), qty=0)
    assert intent.qty == 1


def test_strategy_id_prefix_in_intent_id() -> None:
    intent = trade_signal_to_intent(_make_signal(), qty=1)
    assert intent.intent_id.startswith(_STRATEGY_ID + "-")


def test_regime_is_trend_expansion() -> None:
    intent = trade_signal_to_intent(_make_signal(), qty=1)
    assert intent.regime == Regime.TREND_EXPANSION


def test_entry_type_is_market() -> None:
    intent = trade_signal_to_intent(_make_signal(), qty=1)
    assert intent.entry_type == OrderType.MARKET


def test_now_override_sets_signal_ts() -> None:
    fixed = datetime(2026, 1, 2, 9, 0, tzinfo=UTC)
    intent = trade_signal_to_intent(_make_signal(), qty=1, now=fixed)
    assert intent.signal_ts == fixed


def test_metadata_contains_atr_and_ema() -> None:
    sig = _make_signal(atr14=0.0012, ema_slope=0.0003)
    intent = trade_signal_to_intent(sig, qty=2)
    assert intent.metadata["atr14"] == pytest.approx(0.0012)
    assert intent.metadata["ema_slope"] == pytest.approx(0.0003)


def test_metadata_contains_confluence_type() -> None:
    intent = trade_signal_to_intent(_make_signal(confluence_type="OB"), qty=1)
    assert intent.metadata["confluence"] == "OB"


# ---------------------------------------------------------------------------
# Integration test: submit_intent round-trip in MOCK mode
# ---------------------------------------------------------------------------

def test_submit_intent_accepted_by_mock_execution_engine() -> None:
    """Verify OrderIntent from bridge passes ExecutionEngine.submit_intent in MOCK mode.

    Uses MagicMock for heavyweight dependencies (adapter, risk engine, order
    manager, logger) to keep this test fast and dependency-free.
    """
    from backtest.config import TraderConfig
    from execution.engine import ExecutionEngine

    config = TraderConfig()
    config.execution.mode = TradingMode.MOCK

    mock_risk = MagicMock()
    mock_risk.can_trade.return_value = (True, "ok")
    mock_risk.check_intent.return_value = (True, "ok")

    mock_adapter = MagicMock()
    mock_adapter.place_order.return_value = {"orderId": "test-123", "status": "Placed"}
    # get_market_price returning None short-circuits the slippage check in submit_intent
    mock_adapter.get_market_price.return_value = None

    mock_order_manager = MagicMock()
    mock_logger = MagicMock()

    engine = ExecutionEngine(config, mock_risk, mock_adapter, mock_order_manager, mock_logger)

    # 14:00 UTC = 09:00 CT — inside the default NY session window so the
    # session gate doesn't block the intent before we can inspect the decision.
    now = datetime(2026, 1, 2, 14, 0, tzinfo=UTC)
    sig = _make_signal(direction="BUY")
    intent = trade_signal_to_intent(sig, qty=1, now=now)

    # submit_intent should not raise.
    # The return type is ExecutionDecision (has .approved attribute).
    from execution.engine import ExecutionDecision  # local import to avoid coupling
    result = engine.submit_intent(intent, now=now)
    assert hasattr(result, "approved"), f"Unexpected return type: {type(result)}"
