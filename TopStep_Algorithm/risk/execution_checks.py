from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from math import isfinite
from math import exp
from zoneinfo import ZoneInfo

from config import RiskLimits, SessionConfig
from execution.scheduler import SessionScheduler
from models.instruments import resolve_instrument
from models.orders import ExecutionDecision, OrderIntent, OrderPlan, Regime, Side
from risk.engine import RiskEngine


@dataclass(slots=True)
class AccountState:
    cash_balance: float = 0.0
    net_liquidation: float = 0.0
    cushion_to_max_loss_limit: float = 0.0


@dataclass(slots=True)
class ActiveExposure:
    intent_id: str
    symbol: str
    side: Side
    qty: int
    regime: Regime | str
    entry_price: float
    stop_price: float
    signal_score: float = 0.0
    entry_ts: datetime | None = None
    trend_state: str | None = None
    is_stacked: bool = False


CENTRAL_TZ = ZoneInfo("America/Chicago")


def _time_bucket_central(value: datetime) -> str:
    ts = value if value.tzinfo is not None else value.replace(tzinfo=CENTRAL_TZ)
    ts = ts.astimezone(CENTRAL_TZ)
    bucket_time = ts.time()
    if time(8, 30) <= bucket_time < time(10, 30):
        return "OPEN"
    if time(11, 0) <= bucket_time < time(12, 30):
        return "MIDDAY"
    if time(12, 30) <= bucket_time < time(14, 45):
        return "AFTERNOON"
    return "OTHER"


def _decayed_risk(entry_price: float, stop_price: float, qty: int, entry_ts: datetime | None, now: datetime, decay_lambda: float) -> float:
    base_risk = abs(entry_price - stop_price) * qty
    if entry_ts is None or decay_lambda <= 0.0:
        return base_risk
    elapsed_minutes = max((now - entry_ts).total_seconds() / 60.0, 0.0)
    return base_risk * exp(-decay_lambda * elapsed_minutes)


def validate_intent(
    intent: OrderIntent,
    account_state: AccountState,
    risk_engine: RiskEngine,
    risk_limits: RiskLimits,
    session_config: SessionConfig,
    now: datetime,
    plan: OrderPlan | None = None,
    open_position_count: int = 0,
    active_exposures: list[ActiveExposure] | None = None,
) -> ExecutionDecision:
    scheduler = SessionScheduler(session_config)
    exposures = active_exposures or []
    can_trade, reason = risk_engine.can_trade(now)
    if not can_trade:
        return ExecutionDecision(False, reason, 0, None)

    if not scheduler.is_trading_session(now):
        return ExecutionDecision(False, "session_closed", 0, None)

    if scheduler.is_past_new_trade_cutoff(now):
        return ExecutionDecision(False, "past_new_trade_cutoff", 0, None)

    if intent.regime == Regime.CHOP_MEAN_REVERT:
        return ExecutionDecision(False, "regime_disabled", 0, None)

    if open_position_count >= risk_limits.max_concurrent_positions:
        return ExecutionDecision(False, "max_positions_reached", 0, None)

    intent_regime = str(intent.regime.value if hasattr(intent.regime, "value") else intent.regime)
    intent_side = intent.side.value if hasattr(intent.side, "value") else str(intent.side)
    intent_trend_state = str(intent.metadata.get("trend_state") or "unknown")
    intent_trend_key = f"{intent.symbol}|{intent_side}|{intent_regime}|{intent_trend_state}"
    is_reentry = (
        open_position_count == 0
        and risk_engine.state.last_exit_at is not None
        and risk_engine.state.current_trend_key == intent_trend_key
        and risk_engine.state.entries_in_current_trend > 0
    )

    if is_reentry:
        # Hard cap on sequential reentries in the same trend.  The stacking depth
        # check (max_same_direction_entries_per_trend) only fires for simultaneous
        # open positions, so it never applies to the sequential reentry path.  WFO
        # analysis showed entries #3+ have ~14% WR and are a consistent money-loser.
        projected_entry_num = risk_engine.projected_entry_number(intent)
        if projected_entry_num > int(risk_limits.max_same_direction_entries_per_trend):
            return ExecutionDecision(False, "max_reentries_reached", 0, None)
        if now - risk_engine.state.last_exit_at < timedelta(minutes=risk_limits.min_reentry_spacing_bars):
            return ExecutionDecision(False, "reentry_spacing", 0, None)
        breakout_level = float(intent.metadata.get("breakout_level", intent.entry_price or 0.0) or 0.0)
        breakout_delta = breakout_level - float(risk_engine.state.last_exit_breakout_level)
        if breakout_delta <= float(risk_limits.reentry_breakout_delta_min):
            return ExecutionDecision(False, "reentry_breakout_delta", 0, None)
        atr_pct = float(intent.metadata.get("atr_pct", 1.0) or 1.0)
        if atr_pct > float(risk_limits.reentry_atr_pct_max):
            return ExecutionDecision(False, "reentry_atr_pct", 0, None)
        if float(intent.signal_score) < float(risk_limits.reentry_signal_score_min):
            return ExecutionDecision(False, "reentry_signal_score", 0, None)

    drawdown_multiplier = float(risk_engine.current_risk_multiplier())
    scaling_index = min(open_position_count, max(len(risk_limits.convex_position_scaling) - 1, 0))
    position_scale = float(risk_limits.convex_position_scaling[scaling_index])
    effective_size_scale = max(drawdown_multiplier * position_scale, 0.0)

    normalized_qty = max(1, round(intent.qty * effective_size_scale))

    stop_distance = abs((intent.entry_price or 0.0) - intent.stop_price)
    if intent.entry_price is None or stop_distance <= 0.0:
        return ExecutionDecision(False, "invalid_stop_distance", 0, None)
    atr_value = float(intent.metadata.get("atr", stop_distance) or stop_distance or 0.0)
    if not isfinite(atr_value) or atr_value <= 0.0:
        return ExecutionDecision(False, "invalid_atr", 0, None)
    stop_distance_pct = stop_distance / abs(float(intent.entry_price))
    if stop_distance_pct > float(risk_limits.max_stop_distance_pct):
        return ExecutionDecision(False, "stop_distance_too_wide", 0, None)
    instrument = resolve_instrument(intent.symbol)
    tick_size = instrument.tick_size
    stop_distance_ticks = stop_distance / tick_size
    if stop_distance_ticks > float(risk_limits.max_stop_distance_ticks):
        return ExecutionDecision(False, "stop_distance_too_wide", 0, None)

    same_direction_exposures = [exposure for exposure in exposures if exposure.side == intent.side]
    if same_direction_exposures and not risk_limits.enable_stacking:
        return ExecutionDecision(False, "stacking_disabled", 0, None)

    cluster_risk_budget = float(risk_limits.risk_budget_threshold) * float(risk_limits.max_cluster_risk_fraction)
    regime_key = intent_regime
    decayed_cluster_risk = sum(
        _decayed_risk(
            exposure.entry_price,
            exposure.stop_price,
            exposure.qty,
            exposure.entry_ts,
            now,
            float(risk_limits.heat_decay_lambda),
        )
        for exposure in exposures
        if exposure.side == intent.side
        and str(exposure.regime.value if hasattr(exposure.regime, "value") else exposure.regime) == regime_key
    )
    projected_cluster_risk = decayed_cluster_risk + (stop_distance * normalized_qty)
    if projected_cluster_risk >= cluster_risk_budget:
        return ExecutionDecision(False, "cluster_risk_exceeded", 0, None)

    if same_direction_exposures:
        allowed_regimes = {
            str(regime.value if hasattr(regime, "value") else regime)
            for regime in risk_limits.stacking_allowed_regimes
        }
        intent_regime = regime_key
        if intent_regime not in allowed_regimes:
            return ExecutionDecision(False, "stacking_blocked_regime", 0, None)
        time_bucket = _time_bucket_central(now)
        if risk_limits.disable_stacking_midday and time_bucket == "MIDDAY":
            return ExecutionDecision(False, "midday_stacking_blocked", 0, None)
        stacking_disabled, stacking_disabled_reason = risk_engine.stacking_is_disabled(now)
        if stacking_disabled:
            return ExecutionDecision(False, stacking_disabled_reason, 0, None)
        latest_same_direction = max(
            same_direction_exposures,
            key=lambda exposure: exposure.entry_ts or datetime.min.replace(tzinfo=now.tzinfo),
        )
        trend_state = str(intent.metadata.get("trend_state") or "unknown")
        same_trend_count = sum(1 for exposure in same_direction_exposures if str(exposure.trend_state or "unknown") == trend_state)
        if same_trend_count >= risk_limits.max_same_direction_entries_per_trend:
            return ExecutionDecision(False, "stack_depth_reached", 0, None)
        atr = abs((intent.entry_price or 0.0) - intent.stop_price)
        atr = max(float(intent.metadata.get("atr", atr) or atr or 0.0), instrument.tick_size)
        atr_pct = float(intent.metadata.get("atr_pct", 1.0) or 1.0)
        if atr_pct > float(risk_limits.stacking_volatility_upper_atr_pct):
            return ExecutionDecision(False, "stacking_volatility_blocked", 0, None)
        spacing_ok = (
            latest_same_direction.entry_ts is not None
            and now - latest_same_direction.entry_ts >= timedelta(minutes=risk_limits.same_direction_spacing_bars)
        )
        price_ok = (
            intent.entry_price is not None
            and abs(float(intent.entry_price) - float(latest_same_direction.entry_price))
            >= float(risk_limits.stacking_price_atr_threshold) * atr
        )
        effective_margin = float(risk_limits.stacking_score_margin)
        effective_margin += risk_engine.crossed_drawdown_tiers() * float(risk_limits.stacking_drawdown_margin_step)
        cluster_usage = projected_cluster_risk / cluster_risk_budget if cluster_risk_budget > 0 else 0.0
        if cluster_usage >= 0.5:
            effective_margin += float(risk_limits.stacking_cluster_margin_step)
        if not risk_limits.disable_stacking_midday and time_bucket == "MIDDAY":
            effective_margin += float(risk_limits.midday_stacking_score_penalty)
        stacking_score_threshold = float(risk_limits.stacking_base_score_threshold + effective_margin)
        if float(intent.signal_score) <= stacking_score_threshold:
            return ExecutionDecision(False, "stacking_signal_threshold", 0, None)
        if price_ok:
            stacking_reason = "stacking_allowed_price"
        elif spacing_ok:
            stacking_reason = "stacking_allowed_spacing"
        else:
            return ExecutionDecision(False, "same_direction_blocked", 0, None)
    else:
        stacking_reason = "approved"

    same_regime_count = sum(
        1
        for exposure in exposures
        if str(exposure.regime.value if hasattr(exposure.regime, "value") else exposure.regime)
        == str(intent.regime.value if hasattr(intent.regime, "value") else intent.regime)
        and exposure.side != intent.side
    )
    if same_regime_count >= risk_limits.max_positions_per_regime:
        return ExecutionDecision(False, "regime_limit", 0, None)

    if (
        not risk_engine.is_backtest_mode
        and risk_engine.open_intents_for(intent.symbol) >= risk_limits.max_active_intents_per_symbol
    ):
        return ExecutionDecision(False, "active_intent_limit", 0, None)

    position = risk_engine.position_for(intent.symbol)
    projected_position = abs(position.qty) + normalized_qty
    if not risk_engine.is_backtest_mode and projected_position > risk_limits.max_position_size:
        return ExecutionDecision(False, "max_position_size", 0, None)

    estimated_risk = stop_distance * normalized_qty * instrument.point_value
    cushion = account_state.cushion_to_max_loss_limit
    # Only enforce broker cushion check when the API returns a positive value.
    # A value of 0.0 indicates the broker did not populate this field; the
    # internal_daily_loss_budget check below provides the actual risk gate.
    if not risk_engine.is_backtest_mode and cushion > 0.0 and estimated_risk > cushion:
        return ExecutionDecision(False, "max_loss_buffer_exceeded", 0, None)

    remaining_daily_budget = abs(risk_limits.internal_daily_loss_limit + min(risk_engine.state.daily_realized_pnl, 0.0))
    if not risk_engine.is_backtest_mode and estimated_risk > remaining_daily_budget:
        return ExecutionDecision(False, "internal_daily_loss_budget_exceeded", 0, None)

    open_risk = sum(abs(exposure.entry_price - exposure.stop_price) * exposure.qty for exposure in exposures)
    projected_open_risk = open_risk + (stop_distance * normalized_qty)
    if projected_open_risk > risk_limits.risk_budget_threshold:
        return ExecutionDecision(False, "risk_budget_exceeded", 0, None)

    if same_direction_exposures:
        stacked_open_risk = sum(
            _decayed_risk(
                exposure.entry_price,
                exposure.stop_price,
                exposure.qty,
                exposure.entry_ts,
                now,
                float(risk_limits.heat_decay_lambda),
            )
            for exposure in exposures
            if exposure.is_stacked
        )
        stacked_risk_budget = float(risk_limits.risk_budget_threshold) * float(risk_limits.stacked_risk_budget_fraction)
        projected_stacked_risk = stacked_open_risk + (stop_distance * normalized_qty)
        if projected_stacked_risk >= stacked_risk_budget:
            return ExecutionDecision(False, "stacked_risk_budget_exceeded", 0, None)

    return ExecutionDecision(True, stacking_reason, normalized_qty, plan)
