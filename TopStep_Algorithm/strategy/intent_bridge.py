"""Bridge: TradeSignal (backtest / SignalEngine) → OrderIntent (execution engine).

This module is the connection between the London-sweep signal engine and the
live execution engine.  It converts a ``TradeSignal`` into the ``OrderIntent``
format expected by ``ExecutionEngine.submit_intent()``.

Live bracket note
-----------------
``SignalEngine`` produces two profit targets (TP1, TP2) for a 50/50 contract
split.  The current ``ExecutionEngine``/``OrderPlan`` supports a single target
order.  Accordingly this bridge maps **TP1 only** to ``target_price``.  TP2 is
stored in ``metadata["tp2_price"]`` for observability.  A future iteration can
implement TP2 by splitting qty and sequencing a second intent after TP1 fills.
"""
from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from models.orders import OrderIntent, OrderType, Regime, Side, TimeInForce
from strategy.signal import TradeSignal

# Confluence type → signal quality score [0, 1]
_CONFLUENCE_SCORE: dict[str, float] = {
    "OB+FVG": 1.0,
    "OB": 0.75,
    "FVG": 0.50,
    "NONE": 0.0,
}

_STRATEGY_ID = "london_sweep"


def trade_signal_to_intent_pair(
    signal: TradeSignal,
    qty: int,
    *,
    now: datetime | None = None,
) -> list[OrderIntent]:
    """Convert a ``TradeSignal`` to a 50/50 two-leg bracket as a list of intents.

    Parameters
    ----------
    signal:
        Signal produced by ``SignalEngine.process_bar``.
    qty:
        Total number of contracts.  When qty < 2, only the TP1 leg is returned
        (a single contract cannot be split).
    now:
        Signal timestamp override.  Defaults to UTC now.

    Returns
    -------
    A list of one or two ``OrderIntent`` objects:
    - Always: TP1 leg (floor(qty / 2) contracts, target = tp1_price).
    - When qty ≥ 2: TP2 leg (remaining contracts, target = tp2_price).

    Both legs share the same entry, stop, side, strategy_id, and ``signal_ts``.
    They are distinguished by unique ``intent_id`` values and ``metadata["leg"]``
    values ("tp1" / "tp2").
    """
    qty = max(1, qty)
    tp1_qty = max(1, qty // 2)
    tp2_qty = qty - tp1_qty

    side = Side.BUY if signal.direction == "BUY" else Side.SELL
    score = _CONFLUENCE_SCORE.get(signal.confluence.confluence_type, 0.5)
    ts = now or datetime.now(UTC)
    pair_id = uuid4().hex[:12]

    shared: dict = dict(
        symbol=signal.symbol,
        side=side,
        entry_type=OrderType.MARKET,
        entry_price=signal.entry_price,
        stop_price=signal.stop_price,
        time_in_force=TimeInForce.DAY,
        reason="london_liquidity_sweep",
        signal_ts=ts,
        signal_score=score,
        regime=Regime.TREND_EXPANSION,
        strategy_id=_STRATEGY_ID,
        allow_scale_out=False,
    )

    tp1_intent = OrderIntent(
        intent_id=f"{_STRATEGY_ID}-tp1-{pair_id}",
        qty=tp1_qty,
        target_price=signal.tp1_price,
        metadata={
            "atr14": signal.atr14,
            "ema_slope": signal.ema_slope,
            "confluence": signal.confluence.confluence_type,
            "leg": "tp1",
            "tp2_price": signal.tp2_price,
            "pair_id": pair_id,
        },
        **shared,
    )

    if tp2_qty == 0:
        return [tp1_intent]

    tp2_intent = OrderIntent(
        intent_id=f"{_STRATEGY_ID}-tp2-{pair_id}",
        qty=tp2_qty,
        target_price=signal.tp2_price,
        metadata={
            "atr14": signal.atr14,
            "ema_slope": signal.ema_slope,
            "confluence": signal.confluence.confluence_type,
            "leg": "tp2",
            "pair_id": pair_id,
        },
        **shared,
    )

    return [tp1_intent, tp2_intent]


def trade_signal_to_intent(
    signal: TradeSignal,
    qty: int,
    *,
    now: datetime | None = None,
) -> OrderIntent:
    """Convert a ``TradeSignal`` to a live-trading ``OrderIntent``.

    Parameters
    ----------
    signal:
        Signal produced by ``SignalEngine.process_bar``.
    qty:
        Number of contracts (from profile ``base_qty``; the risk engine will
        normalize further via drawdown tiers).
    now:
        Signal timestamp override.  Defaults to UTC now.

    Returns
    -------
    ``OrderIntent`` ready for ``ExecutionEngine.submit_intent()``.
    """
    side = Side.BUY if signal.direction == "BUY" else Side.SELL
    score = _CONFLUENCE_SCORE.get(signal.confluence.confluence_type, 0.5)
    ts = now or datetime.now(UTC)

    return OrderIntent(
        intent_id=f"{_STRATEGY_ID}-{uuid4().hex[:12]}",
        symbol=signal.symbol,
        side=side,
        qty=max(1, qty),
        entry_type=OrderType.MARKET,
        entry_price=signal.entry_price,
        stop_price=signal.stop_price,
        # Live uses TP1 as the single profit target.
        target_price=signal.tp1_price,
        time_in_force=TimeInForce.DAY,
        reason="london_liquidity_sweep",
        signal_ts=ts,
        signal_score=score,
        # The London sweep is a momentum trade: price sweeps liquidity and
        # reverses with directional follow-through — TREND_EXPANSION regime.
        regime=Regime.TREND_EXPANSION,
        strategy_id=_STRATEGY_ID,
        allow_scale_out=False,
        metadata={
            "atr14": signal.atr14,
            "ema_slope": signal.ema_slope,
            "confluence": signal.confluence.confluence_type,
            # TP2 stored for observability / future multi-leg support
            "tp2_price": signal.tp2_price,
        },
    )
