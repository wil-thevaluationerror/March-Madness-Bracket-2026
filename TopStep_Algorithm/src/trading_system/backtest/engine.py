from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from trading_system.execution.engine import ExecutionEngine
from trading_system.core.instruments import resolve_instrument
from trading_system.core.domain import BrokerOrder, ExecutionReport, OrderIntent, OrderState, OrderType, Side, TimeInForce

CENTRAL_TZ = ZoneInfo("America/Chicago")


@dataclass(slots=True)
class SimulatedPosition:
    intent_id: str
    symbol: str
    side: Side
    entry_price: float
    qty: int
    stop_price: float
    target_price: float | None
    entry_ts: datetime
    regime: str
    stop_order_id: str | None
    target_order_id: str | None
    entry_order_id: str
    signal_score: float = 0.0
    is_stacked: bool = False
    stacking_reason: str | None = None
    risk_multiplier: float = 1.0
    position_scale: float = 1.0
    cluster_risk_usage_pct: float = 0.0
    mfe: float = 0.0
    mae: float = 0.0
    is_reentry: bool = False
    entry_number_in_trend: int = 1
    trend_key: str = ""
    atr_at_entry: float = 0.0
    stop_distance: float = 0.0
    stop_distance_pct: float = 0.0
    stop_distance_ticks: float = 0.0
    abnormal_stop: bool = False
    # Set True once the stop has been slid to entry price via the breakeven trigger.
    breakeven_stop_active: bool = False


@dataclass(slots=True)
class SimulatedTrade:
    intent_id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    qty: int
    pnl: float
    entry_ts: datetime
    exit_ts: datetime
    exit_reason: str
    regime: str
    time_bucket: str
    is_stacked: bool = False
    stacking_reason: str | None = None
    risk_multiplier: float = 1.0
    position_scale: float = 1.0
    cluster_risk_usage_pct: float = 0.0
    mae: float = 0.0
    mfe: float = 0.0
    is_reentry: bool = False
    entry_number_in_trend: int = 1
    trend_key: str = ""
    atr_at_entry: float = 0.0
    stop_distance: float = 0.0
    stop_distance_pct: float = 0.0
    stop_distance_ticks: float = 0.0
    abnormal_stop: bool = False


@dataclass(slots=True)
class BacktestResult:
    fills: list[ExecutionReport] = field(default_factory=list)
    denied_intents: list[str] = field(default_factory=list)
    approved_intents: list[str] = field(default_factory=list)
    denial_counts: dict[str, int] = field(default_factory=dict)
    denial_examples: dict[str, list[dict[str, object]]] = field(default_factory=dict)
    trades: list[SimulatedTrade] = field(default_factory=list)
    position_open_blocked_signals: int = 0
    percentage_blocked: float = 0.0
    concurrency_analysis: dict[int, dict[str, float | int]] = field(default_factory=dict)
    max_concurrent_positions: int = 2
    blocked_by_max_positions: int = 0
    percentage_blocked_by_max_positions: float = 0.0
    avg_concurrent_positions: float = 0.0
    max_observed_concurrent_positions: int = 0
    avg_stacked_heat_usage_pct: float = 0.0
    max_stacked_heat_usage_pct: float = 0.0
    stacking_disabled_bar_percentage: float = 0.0
    stacking_disabled_events: int = 0
    avg_risk_multiplier: float = 1.0
    avg_position_scale: float = 1.0
    max_cluster_risk_usage_pct: float = 0.0
    cluster_risk_usage_curve: list[dict[str, float | str]] = field(default_factory=list)
    risk_multiplier_curve: list[dict[str, float | str]] = field(default_factory=list)
    reentry_count: int = 0
    reentry_pnl: float = 0.0
    reentry_win_rate: float = 0.0
    reentry_drawdown: float = 0.0
    initial_entry_pnl: float = 0.0
    avg_minutes_between_entries: float = 0.0
    trades_per_trend: dict[str, int] = field(default_factory=dict)
    stop_diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ConcurrencySimulationState:
    level: int
    positions: list[SimulatedPosition] = field(default_factory=list)
    trades: list[SimulatedTrade] = field(default_factory=list)
    blocked_by_position_limit: int = 0
    concurrent_counts: list[int] = field(default_factory=list)


def _coerce_time(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=CENTRAL_TZ)
    return value.astimezone(CENTRAL_TZ)


def _time_bucket_central(value: datetime) -> str:
    ts = _coerce_time(value)
    bucket_time = ts.time()
    if time(8, 30) <= bucket_time < time(10, 30):
        return "OPEN"
    if time(10, 30) <= bucket_time < time(12, 30):
        return "MIDDAY"
    if time(12, 30) <= bucket_time < time(14, 45):
        return "AFTERNOON"
    if time(14, 45) <= bucket_time < time(15, 0):
        return "LATE"
    return "OUTSIDE_SESSION"


def _regime_value(intent: OrderIntent) -> str:
    return intent.regime.value if hasattr(intent.regime, "value") else str(intent.regime)


def _side_value(side: Side) -> str:
    return side.value if hasattr(side, "value") else str(side)


def _trade_pnl(symbol: str, side: Side, entry_price: float, exit_price: float, qty: int) -> float:
    point_value = resolve_instrument(symbol).point_value
    if side == Side.BUY:
        return (exit_price - entry_price) * qty * point_value
    return (entry_price - exit_price) * qty * point_value


class SimulatedBacktestEngine:
    def __init__(
        self,
        execution_engine: ExecutionEngine,
        *,
        max_concurrent_positions: int = 2,
        concurrency_levels: tuple[int, ...] = (1, 2),
    ) -> None:
        self.execution_engine = execution_engine
        self.max_concurrent_positions = max(1, max_concurrent_positions)
        self.concurrency_levels = tuple(sorted({level for level in concurrency_levels if level > 0})) or (1,)

    def run(self, bars: pd.DataFrame, intents: list[OrderIntent], now: datetime | None = None) -> BacktestResult:
        result = BacktestResult()
        if bars.empty:
            return result

        commission_per_lot = float(self.execution_engine.config.execution.commission_per_lot)
        sorted_bars = bars.sort_values(["ts_event", "symbol"]).reset_index(drop=True)
        sorted_intents = sorted(intents, key=lambda intent: (_coerce_time(intent.signal_ts), intent.symbol, intent.intent_id))
        intents_by_time: dict[datetime, list[OrderIntent]] = defaultdict(list)
        for intent in sorted_intents:
            intents_by_time[_coerce_time(intent.signal_ts)].append(intent)

        bars_by_time: dict[datetime, list[dict[str, Any]]] = defaultdict(list)
        for row in sorted_bars.itertuples(index=False):
            bar = self._bar_from_row(row)
            bars_by_time[bar["ts_event"]].append(bar)

        custom_denial_counts: Counter[str] = Counter()
        custom_denial_examples: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
        open_positions: list[SimulatedPosition] = []
        concurrency_states = {
            level: ConcurrencySimulationState(level=level)
            for level in self.concurrency_levels
        }
        primary_concurrent_counts: list[int] = []
        last_bar_for_symbol: dict[str, dict[str, Any]] = {}
        result.max_concurrent_positions = self.max_concurrent_positions
        stacked_heat_usage_samples: list[float] = []
        stacking_disabled_bars = 0
        risk_multiplier_samples: list[float] = []
        position_scale_samples: list[float] = []
        cluster_risk_usage_samples: list[float] = []

        for timestamp in sorted(bars_by_time):
            bars_for_time = bars_by_time[timestamp]
            for bar in bars_for_time:
                last_bar_for_symbol[bar["symbol"]] = bar
                open_positions = self._process_primary_exits(open_positions, bar, result)
                for simulation_state in concurrency_states.values():
                    simulation_state.positions = self._process_analysis_exits(simulation_state.positions, bar, simulation_state)

            intents_for_time = sorted(
                intents_by_time.get(timestamp, []),
                key=lambda intent: (intent.signal_score, intent.intent_id),
                reverse=True,
            )
            for intent in intents_for_time:
                signal_bar = last_bar_for_symbol.get(intent.symbol)
                if signal_bar is None:
                    continue
                self._apply_concurrency_analysis(concurrency_states, intent, signal_bar, last_bar_for_symbol)

                if len(open_positions) >= self.max_concurrent_positions:
                    weakest_position = min(open_positions, key=lambda position: (position.signal_score, position.entry_ts))
                    if self.max_concurrent_positions > 1 and intent.signal_score > weakest_position.signal_score:
                        weakest_bar = last_bar_for_symbol.get(weakest_position.symbol)
                        if weakest_bar is not None:
                            replacement_fill = self._emit_flatten_fill(
                                weakest_position,
                                timestamp,
                                weakest_bar["close"],
                                "replaced_by_higher_score",
                            )
                            self.execution_engine.drain_adapter_events()
                            result.fills.append(replacement_fill)
                            result.trades.append(
                                self._close_trade(
                                    weakest_position,
                                    weakest_bar["close"],
                                    timestamp,
                                    "replaced_by_higher_score",
                                    commission_per_lot,
                                )
                            )
                            open_positions = [
                                position for position in open_positions if position.intent_id != weakest_position.intent_id
                            ]
                    else:
                        self._record_custom_denial(
                            custom_denial_counts,
                            custom_denial_examples,
                            "max_positions_reached",
                            intent,
                            timestamp,
                        )
                        result.denied_intents.append(intent.intent_id)
                        continue

                decision = self.execution_engine.submit_intent(intent, now=timestamp)
                if not decision.approved:
                    result.denied_intents.append(intent.intent_id)
                    continue

                result.approved_intents.append(intent.intent_id)
                parent_id = decision.order_plan.entry.order_id if decision.order_plan else None
                if parent_id is None:
                    continue
                approved_qty = max(decision.normalized_qty, 1)
                risk_multiplier = float(self.execution_engine.risk_engine.current_risk_multiplier())
                scaling_index = min(len(open_positions), max(len(self.execution_engine.config.risk.convex_position_scaling) - 1, 0))
                position_scale = float(self.execution_engine.config.risk.convex_position_scaling[scaling_index])
                cluster_usage_pct = self._cluster_risk_usage_pct(
                    open_positions,
                    intent.side,
                    _regime_value(intent),
                    timestamp,
                )
                entry_fill = self.execution_engine.adapter.emit_fill(
                    parent_id,
                    approved_qty,
                    intent.entry_price or signal_bar["close"],
                    timestamp=timestamp,
                )
                self.execution_engine.drain_adapter_events()
                result.fills.append(entry_fill)
                working_intent = self.execution_engine.intent_registry.get(intent.intent_id, intent)
                stop_distance = abs((intent.entry_price or signal_bar["close"]) - intent.stop_price)
                stop_distance_pct = (
                    stop_distance / abs(float(intent.entry_price or signal_bar["close"]))
                    if float(intent.entry_price or signal_bar["close"]) != 0.0
                    else 0.0
                )
                stop_distance_ticks = stop_distance / resolve_instrument(intent.symbol).tick_size
                open_positions.append(
                    SimulatedPosition(
                        intent_id=intent.intent_id,
                        symbol=intent.symbol,
                        side=intent.side,
                        entry_price=intent.entry_price or signal_bar["close"],
                        qty=approved_qty,
                        stop_price=intent.stop_price,
                        target_price=intent.target_price,
                        entry_ts=timestamp,
                        regime=_regime_value(intent),
                        stop_order_id=decision.order_plan.stop.order_id if decision.order_plan and decision.order_plan.stop else None,
                        target_order_id=decision.order_plan.target.order_id if decision.order_plan and decision.order_plan.target else None,
                        entry_order_id=parent_id,
                        signal_score=intent.signal_score,
                        is_stacked=decision.reason.startswith("stacking_allowed_"),
                        stacking_reason=decision.reason.removeprefix("stacking_allowed_")
                        if decision.reason.startswith("stacking_allowed_")
                        else None,
                        risk_multiplier=risk_multiplier,
                        position_scale=position_scale,
                        cluster_risk_usage_pct=cluster_usage_pct,
                        is_reentry=bool(working_intent.metadata.get("is_reentry")),
                        entry_number_in_trend=int(working_intent.metadata.get("entry_number_in_trend", 1) or 1),
                        trend_key=str(working_intent.metadata.get("trend_key") or ""),
                        atr_at_entry=float(working_intent.metadata.get("atr", 0.0) or 0.0),
                        stop_distance=stop_distance,
                        stop_distance_pct=stop_distance_pct,
                        stop_distance_ticks=stop_distance_ticks,
                        abnormal_stop=bool(
                            stop_distance_pct > float(self.execution_engine.config.risk.max_stop_distance_pct)
                            or stop_distance_ticks > float(self.execution_engine.config.risk.max_stop_distance_ticks)
                        ),
                    )
                )
                risk_multiplier_samples.append(risk_multiplier)
                position_scale_samples.append(position_scale)

            stacked_risk_budget = max(
                float(self.execution_engine.config.risk.risk_budget_threshold)
                * float(self.execution_engine.config.risk.stacked_risk_budget_fraction),
                1e-9,
            )
            stacked_open_risk = sum(
                abs(position.entry_price - position.stop_price) * position.qty
                for position in open_positions
                if position.is_stacked
            )
            stacked_heat_usage_samples.append((stacked_open_risk / stacked_risk_budget) * 100.0)
            cluster_usage_pct = self._max_cluster_risk_usage_pct(open_positions, timestamp)
            cluster_risk_usage_samples.append(cluster_usage_pct)
            result.cluster_risk_usage_curve.append(
                {"timestamp": timestamp.isoformat(), "cluster_risk_usage_pct": cluster_usage_pct}
            )
            risk_multiplier_value = float(self.execution_engine.risk_engine.current_risk_multiplier())
            result.risk_multiplier_curve.append(
                {"timestamp": timestamp.isoformat(), "risk_multiplier": risk_multiplier_value}
            )
            if (
                self.execution_engine.risk_engine.state.stacking_disabled_until is not None
                and timestamp < self.execution_engine.risk_engine.state.stacking_disabled_until
            ):
                stacking_disabled_bars += 1
            for simulation_state in concurrency_states.values():
                simulation_state.concurrent_counts.append(len(simulation_state.positions))
            primary_concurrent_counts.append(len(open_positions))

        for position in open_positions:
            last_bar = last_bar_for_symbol.get(position.symbol)
            if last_bar is not None:
                exit_fill = self._emit_flatten_fill(position, last_bar["ts_event"], last_bar["close"], "end_of_data")
                self.execution_engine.drain_adapter_events()
                result.fills.append(exit_fill)
                result.trades.append(self._close_trade(position, last_bar["close"], last_bar["ts_event"], "end_of_data", commission_per_lot))
        for simulation_state in concurrency_states.values():
            self._close_analysis_positions_at_end(simulation_state, last_bar_for_symbol)

        engine_analytics = self.execution_engine.denial_analytics()
        denial_counts = Counter(dict(engine_analytics.get("counts", {})))
        denial_counts.update(custom_denial_counts)

        denial_examples: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
        for reason, examples in dict(engine_analytics.get("examples", {})).items():
            denial_examples[reason].extend(list(examples))
        for reason, examples in custom_denial_examples.items():
            denial_examples[reason].extend(example for example in examples if len(denial_examples[reason]) < 5)

        result.denial_counts = dict(denial_counts)
        result.denial_examples = {reason: examples[:5] for reason, examples in denial_examples.items()}
        result.blocked_by_max_positions = int(denial_counts.get("max_positions_reached", 0))
        result.percentage_blocked_by_max_positions = (
            (result.blocked_by_max_positions / len(sorted_intents)) * 100.0 if sorted_intents else 0.0
        )
        result.position_open_blocked_signals = result.blocked_by_max_positions
        result.percentage_blocked = result.percentage_blocked_by_max_positions
        result.avg_concurrent_positions = (
            sum(primary_concurrent_counts) / len(primary_concurrent_counts) if primary_concurrent_counts else 0.0
        )
        result.max_observed_concurrent_positions = max(primary_concurrent_counts, default=0)
        result.avg_stacked_heat_usage_pct = (
            sum(stacked_heat_usage_samples) / len(stacked_heat_usage_samples) if stacked_heat_usage_samples else 0.0
        )
        result.max_stacked_heat_usage_pct = max(stacked_heat_usage_samples, default=0.0)
        result.stacking_disabled_bar_percentage = (
            (stacking_disabled_bars / len(bars_by_time)) * 100.0 if bars_by_time else 0.0
        )
        result.stacking_disabled_events = int(self.execution_engine.risk_engine.state.stacking_disabled_events)
        result.avg_risk_multiplier = sum(risk_multiplier_samples) / len(risk_multiplier_samples) if risk_multiplier_samples else 1.0
        result.avg_position_scale = sum(position_scale_samples) / len(position_scale_samples) if position_scale_samples else 1.0
        result.max_cluster_risk_usage_pct = max(cluster_risk_usage_samples, default=0.0)
        reentry_trades = [trade for trade in result.trades if trade.is_reentry]
        initial_trades = [trade for trade in result.trades if not trade.is_reentry]
        result.reentry_count = len(reentry_trades)
        result.reentry_pnl = sum(trade.pnl for trade in reentry_trades)
        result.initial_entry_pnl = sum(trade.pnl for trade in initial_trades)
        result.reentry_win_rate = (
            sum(1 for trade in reentry_trades if trade.pnl > 0) / len(reentry_trades) if reentry_trades else 0.0
        )
        result.reentry_drawdown = min(
            (point["drawdown"] for point in self._build_trade_equity_curve(reentry_trades)),
            default=0.0,
        )
        result.trades_per_trend = dict(Counter(trade.trend_key for trade in result.trades if trade.trend_key))
        sorted_entries = sorted(result.trades, key=lambda trade: trade.entry_ts)
        entry_gaps = [
            (current.entry_ts - previous.entry_ts).total_seconds() / 60.0
            for previous, current in zip(sorted_entries, sorted_entries[1:])
        ]
        result.avg_minutes_between_entries = sum(entry_gaps) / len(entry_gaps) if entry_gaps else 0.0
        stop_distances = [trade.stop_distance for trade in result.trades]
        stop_distance_pcts = [trade.stop_distance_pct for trade in result.trades]
        atr_values = [trade.atr_at_entry for trade in result.trades]
        abnormal_trades = [trade for trade in result.trades if trade.abnormal_stop]
        sorted_by_atr = sorted(result.trades, key=lambda trade: trade.atr_at_entry, reverse=True)[:10]
        result.stop_diagnostics = {
            "avg_atr": (sum(atr_values) / len(atr_values)) if atr_values else 0.0,
            "max_atr": max(atr_values, default=0.0),
            "avg_stop_distance": (sum(stop_distances) / len(stop_distances)) if stop_distances else 0.0,
            "median_stop_distance": sorted(stop_distances)[len(stop_distances) // 2] if stop_distances else 0.0,
            "max_stop_distance": max(stop_distances, default=0.0),
            "avg_stop_distance_pct": (sum(stop_distance_pcts) / len(stop_distance_pcts)) if stop_distance_pcts else 0.0,
            "max_stop_distance_pct": max(stop_distance_pcts, default=0.0),
            "abnormal_stop_count": len(abnormal_trades),
            "abnormal_stop_pct": (len(abnormal_trades) / len(result.trades)) * 100.0 if result.trades else 0.0,
            "largest_atr_trades": [
                {
                    "intent_id": trade.intent_id,
                    "symbol": trade.symbol,
                    "entry_ts": trade.entry_ts.isoformat(),
                    "atr": trade.atr_at_entry,
                    "stop_distance": trade.stop_distance,
                    "stop_distance_pct": trade.stop_distance_pct,
                }
                for trade in sorted_by_atr
            ],
            "flagged_trades": [
                {
                    "intent_id": trade.intent_id,
                    "symbol": trade.symbol,
                    "entry_ts": trade.entry_ts.isoformat(),
                    "entry_price": trade.entry_price,
                    "stop_price": trade.entry_price - trade.stop_distance if trade.side == "BUY" else trade.entry_price + trade.stop_distance,
                    "stop_distance": trade.stop_distance,
                    "stop_distance_pct": trade.stop_distance_pct,
                    "stop_distance_ticks": trade.stop_distance_ticks,
                    "atr": trade.atr_at_entry,
                }
                for trade in abnormal_trades[:10]
            ],
        }
        result.concurrency_analysis = {
            level: self._build_concurrency_metrics(state, len(sorted_intents))
            for level, state in concurrency_states.items()
        }
        return result

    @staticmethod
    def _bar_from_row(row: Any) -> dict[str, Any]:
        return {
            "ts_event": _coerce_time(getattr(row, "ts_event")),
            "symbol": str(getattr(row, "symbol")),
            "open": float(getattr(row, "open")),
            "high": float(getattr(row, "high")),
            "low": float(getattr(row, "low")),
            "close": float(getattr(row, "close")),
            "volume": float(getattr(row, "volume", 0.0)),
        }

    @staticmethod
    def _update_excursions(position: SimulatedPosition, high: float, low: float) -> None:
        point_value = resolve_instrument(position.symbol).point_value
        if position.side == Side.BUY:
            favorable = (high - position.entry_price) * position.qty * point_value
            adverse = (low - position.entry_price) * position.qty * point_value
        else:
            favorable = (position.entry_price - low) * position.qty * point_value
            adverse = (position.entry_price - high) * position.qty * point_value
        position.mfe = max(position.mfe, favorable)
        position.mae = min(position.mae, adverse)

    def _apply_breakeven_stop(self, position: SimulatedPosition, bar: dict[str, Any]) -> None:
        """Slide the stop to entry price once price has moved breakeven_trigger_atr × stop_distance in our favour.

        Using stop_distance (not atr_at_entry) ensures the trigger scales correctly when
        the stop was sized from 5-min ATR instead of 1-min ATR.  For example, if the stop
        is 4 points away (1× 5-min ATR) and breakeven_trigger_atr=0.75, the breakeven stop
        activates after a 3-point favourable move — 75% of the risk taken, not a tiny 1-min
        ATR fraction that fires within the first bar.
        """
        trigger = float(self.execution_engine.config.strategy.breakeven_trigger_atr)
        if trigger <= 0.0 or position.stop_distance <= 0.0 or position.breakeven_stop_active:
            return
        threshold = trigger * position.stop_distance
        if position.side == Side.BUY and bar["high"] >= position.entry_price + threshold:
            position.stop_price = max(position.stop_price, position.entry_price)
            position.breakeven_stop_active = True
        elif position.side == Side.SELL and bar["low"] <= position.entry_price - threshold:
            position.stop_price = min(position.stop_price, position.entry_price)
            position.breakeven_stop_active = True

    def _process_primary_exits(
        self,
        positions: list[SimulatedPosition],
        bar: dict[str, Any],
        result: BacktestResult,
    ) -> list[SimulatedPosition]:
        commission_per_lot = float(self.execution_engine.config.execution.commission_per_lot)
        remaining_positions: list[SimulatedPosition] = []
        for position in positions:
            if position.symbol != bar["symbol"]:
                remaining_positions.append(position)
                continue
            self._update_excursions(position, bar["high"], bar["low"])
            self._apply_breakeven_stop(position, bar)
            exit_fill, exit_price, exit_reason = self._check_exit(position, bar)
            if exit_fill is not None and exit_price is not None and exit_reason is not None:
                self.execution_engine.drain_adapter_events()
                result.fills.append(exit_fill)
                result.trades.append(self._close_trade(position, exit_price, bar["ts_event"], exit_reason, commission_per_lot))
                continue
            remaining_positions.append(position)
        return remaining_positions

    def _process_analysis_exits(
        self,
        positions: list[SimulatedPosition],
        bar: dict[str, Any],
        simulation_state: ConcurrencySimulationState,
    ) -> list[SimulatedPosition]:
        remaining_positions: list[SimulatedPosition] = []
        for position in positions:
            if position.symbol != bar["symbol"]:
                remaining_positions.append(position)
                continue
            self._update_excursions(position, bar["high"], bar["low"])
            self._apply_breakeven_stop(position, bar)
            _, exit_price, exit_reason = self._check_simulated_exit(position, bar)
            if exit_price is not None and exit_reason is not None:
                simulation_state.trades.append(self._close_trade(position, exit_price, bar["ts_event"], exit_reason))
                continue
            remaining_positions.append(position)
        return remaining_positions

    def _apply_concurrency_analysis(
        self,
        concurrency_states: dict[int, ConcurrencySimulationState],
        intent: OrderIntent,
        bar: dict[str, Any],
        last_bar_for_symbol: dict[str, dict[str, Any]],
    ) -> None:
        if not self._analysis_intent_is_tradeable(intent, bar["ts_event"]):
            return
        for level, simulation_state in concurrency_states.items():
            if len(simulation_state.positions) >= level:
                weakest_position = min(simulation_state.positions, key=lambda position: (position.signal_score, position.entry_ts))
                if intent.signal_score > weakest_position.signal_score:
                    weakest_bar = last_bar_for_symbol.get(weakest_position.symbol, bar)
                    simulation_state.trades.append(
                        self._close_trade(
                            weakest_position,
                            weakest_bar["close"],
                            bar["ts_event"],
                            "replaced_by_higher_score",
                        )
                    )
                    simulation_state.positions = [
                        position for position in simulation_state.positions if position.intent_id != weakest_position.intent_id
                    ]
                else:
                    simulation_state.blocked_by_position_limit += 1
                    continue
            simulation_state.positions.append(self._analysis_position_from_intent(intent, bar))

    def _analysis_intent_is_tradeable(self, intent: OrderIntent, now: datetime) -> bool:
        if intent.qty <= 0:
            return False
        if intent.entry_price is None:
            return False
        if intent.stop_price <= 0:
            return False
        if not self.execution_engine.scheduler.is_trading_session(now):
            return False
        if self.execution_engine.scheduler.is_past_new_trade_cutoff(now):
            return False
        return True

    @staticmethod
    def _analysis_position_from_intent(intent: OrderIntent, bar: dict[str, Any]) -> SimulatedPosition:
        return SimulatedPosition(
            intent_id=intent.intent_id,
            symbol=intent.symbol,
            side=intent.side,
            entry_price=intent.entry_price or bar["close"],
            qty=intent.qty,
            stop_price=intent.stop_price,
            target_price=intent.target_price,
            entry_ts=bar["ts_event"],
            regime=_regime_value(intent),
            stop_order_id=None,
            target_order_id=None,
            entry_order_id=f"analysis-{intent.intent_id}",
            signal_score=intent.signal_score,
            is_reentry=bool(intent.metadata.get("is_reentry")),
            entry_number_in_trend=int(intent.metadata.get("entry_number_in_trend", 1) or 1),
            trend_key=str(intent.metadata.get("trend_key") or ""),
            atr_at_entry=float(intent.metadata.get("atr", 0.0) or 0.0),
            stop_distance=abs((intent.entry_price or bar["close"]) - intent.stop_price),
            stop_distance_pct=(
                abs((intent.entry_price or bar["close"]) - intent.stop_price) / abs(float(intent.entry_price or bar["close"]))
                if float(intent.entry_price or bar["close"]) != 0.0
                else 0.0
            ),
            stop_distance_ticks=(
                abs((intent.entry_price or bar["close"]) - intent.stop_price)
                / resolve_instrument(intent.symbol).tick_size
            ),
        )

    def _decayed_position_risk(self, position: SimulatedPosition, now: datetime) -> float:
        base_risk = abs(position.entry_price - position.stop_price) * position.qty
        elapsed_minutes = max((now - position.entry_ts).total_seconds() / 60.0, 0.0)
        return base_risk * (2.718281828459045 ** (-float(self.execution_engine.config.risk.heat_decay_lambda) * elapsed_minutes))

    def _cluster_risk_usage_pct(
        self,
        positions: list[SimulatedPosition],
        side: Side,
        regime: str,
        now: datetime,
    ) -> float:
        budget = float(self.execution_engine.config.risk.risk_budget_threshold) * float(
            self.execution_engine.config.risk.max_cluster_risk_fraction
        )
        if budget <= 0:
            return 0.0
        risk = sum(
            self._decayed_position_risk(position, now)
            for position in positions
            if position.side == side and position.regime == regime
        )
        return (risk / budget) * 100.0

    def _max_cluster_risk_usage_pct(self, positions: list[SimulatedPosition], now: datetime) -> float:
        if not positions:
            return 0.0
        groups = {(position.side, position.regime) for position in positions}
        return max(
            self._cluster_risk_usage_pct(positions, side, regime, now)
            for side, regime in groups
        )

    def _check_simulated_exit(
        self,
        position: SimulatedPosition,
        bar: dict[str, Any],
    ) -> tuple[None, float | None, str | None]:
        stop_hit = False
        target_hit = False
        if position.side == Side.BUY:
            stop_hit = bar["low"] <= position.stop_price
            target_hit = position.target_price is not None and bar["high"] >= position.target_price
        else:
            stop_hit = bar["high"] >= position.stop_price
            target_hit = position.target_price is not None and bar["low"] <= position.target_price

        if stop_hit:
            stop_reason = "breakeven_stop" if position.breakeven_stop_active else "protective_stop"
            return None, position.stop_price, stop_reason
        if target_hit and position.target_price is not None:
            return None, position.target_price, "profit_target"
        if bar["ts_event"].time() >= time(15, 0):
            return None, bar["close"], "end_of_day"
        return None, None, None

    def _close_analysis_positions_at_end(
        self,
        simulation_state: ConcurrencySimulationState,
        last_bar_for_symbol: dict[str, dict[str, Any]],
    ) -> None:
        for position in simulation_state.positions:
            last_bar = last_bar_for_symbol.get(position.symbol)
            if last_bar is None:
                continue
            simulation_state.trades.append(
                self._close_trade(position, last_bar["close"], last_bar["ts_event"], "end_of_data")
            )
        simulation_state.positions = []

    @staticmethod
    def _build_concurrency_metrics(
        simulation_state: ConcurrencySimulationState,
        total_intents: int,
    ) -> dict[str, float | int]:
        sorted_trades = sorted(simulation_state.trades, key=lambda item: item.exit_ts)
        equity_curve = SimulatedBacktestEngine._build_trade_equity_curve(sorted_trades)
        total_pnl = sum(trade.pnl for trade in sorted_trades)
        total_trades = len(sorted_trades)
        wins = sum(1 for trade in sorted_trades if trade.pnl > 0)
        max_drawdown = min((point["drawdown"] for point in equity_curve), default=0.0)
        overlap_metrics = SimulatedBacktestEngine._build_overlap_metrics(sorted_trades, simulation_state.concurrent_counts)
        time_bucket_metrics = SimulatedBacktestEngine._build_time_bucket_metrics(sorted_trades)
        total_trades = len(simulation_state.trades)
        return {
            "trades": total_trades,
            "pnl": total_pnl,
            "expectancy": total_pnl / total_trades if total_trades else 0.0,
            "win_rate": wins / total_trades if total_trades else 0.0,
            "max_drawdown": max_drawdown,
            "blocked_signals": simulation_state.blocked_by_position_limit,
            "blocked_percentage": (simulation_state.blocked_by_position_limit / total_intents) * 100.0 if total_intents else 0.0,
            "avg_concurrent_positions": overlap_metrics["avg_concurrent_positions"],
            "max_concurrent_positions": overlap_metrics["max_concurrent_positions"],
            "overlap_distribution": overlap_metrics["overlap_distribution"],
            "overlap_pair_count": overlap_metrics["overlap_pair_count"],
            "avg_overlap_duration_minutes": overlap_metrics["avg_overlap_duration_minutes"],
            "overlap_pnl_correlation": overlap_metrics["overlap_pnl_correlation"],
            "worst_clustered_loss": overlap_metrics["worst_clustered_loss"],
            "worst_clustered_loss_trade_count": overlap_metrics["worst_clustered_loss_trade_count"],
            "max_simultaneous_drawdown": overlap_metrics["max_simultaneous_drawdown"],
            "equity_curve": equity_curve,
            "drawdown_curve": [
                {"exit_ts": point["exit_ts"], "drawdown": point["drawdown"]}
                for point in equity_curve
            ],
            "time_of_day": time_bucket_metrics,
        }

    @staticmethod
    def _build_trade_equity_curve(trades: list[SimulatedTrade]) -> list[dict[str, float | str]]:
        curve: list[dict[str, float | str]] = []
        equity = 0.0
        peak = 0.0
        for trade in trades:
            equity += trade.pnl
            peak = max(peak, equity)
            curve.append(
                {
                    "exit_ts": trade.exit_ts.isoformat(),
                    "cumulative_pnl": equity,
                    "drawdown": equity - peak,
                    "trade_pnl": trade.pnl,
                }
            )
        return curve

    @staticmethod
    def _build_time_bucket_metrics(trades: list[SimulatedTrade]) -> dict[str, dict[str, float | int]]:
        metrics: dict[str, dict[str, float | int]] = {}
        for trade in trades:
            bucket = trade.time_bucket
            row = metrics.setdefault(
                bucket,
                {"trades": 0, "pnl": 0.0, "wins": 0, "max_drawdown": 0.0},
            )
            row["trades"] += 1
            row["pnl"] += trade.pnl
            if trade.pnl > 0:
                row["wins"] += 1
        for bucket, row in metrics.items():
            trades_count = int(row["trades"])
            pnl_total = float(row["pnl"])
            row["expectancy"] = pnl_total / trades_count if trades_count else 0.0
            row["win_rate"] = int(row["wins"]) / trades_count if trades_count else 0.0
        return metrics

    @staticmethod
    def _build_overlap_metrics(
        trades: list[SimulatedTrade],
        concurrent_counts: list[int],
    ) -> dict[str, float | int | list[dict[str, float | int]]]:
        avg_concurrent = sum(concurrent_counts) / len(concurrent_counts) if concurrent_counts else 0.0
        max_concurrent = max(concurrent_counts, default=0)
        distribution_counts = Counter(concurrent_counts)
        overlap_distribution = [
            {
                "concurrent_positions": count,
                "bars": bars_count,
                "percentage": (bars_count / len(concurrent_counts)) * 100.0 if concurrent_counts else 0.0,
            }
            for count, bars_count in sorted(distribution_counts.items())
        ]

        overlap_durations: list[float] = []
        pnl_left: list[float] = []
        pnl_right: list[float] = []
        sorted_by_entry = sorted(trades, key=lambda trade: trade.entry_ts)
        for index, left_trade in enumerate(sorted_by_entry):
            for right_trade in sorted_by_entry[index + 1 :]:
                if right_trade.entry_ts >= left_trade.exit_ts:
                    break
                overlap_start = max(left_trade.entry_ts, right_trade.entry_ts)
                overlap_end = min(left_trade.exit_ts, right_trade.exit_ts)
                if overlap_end <= overlap_start:
                    continue
                overlap_durations.append((overlap_end - overlap_start).total_seconds() / 60.0)
                pnl_left.append(left_trade.pnl)
                pnl_right.append(right_trade.pnl)

        overlap_pair_count = len(overlap_durations)
        avg_overlap_duration = sum(overlap_durations) / overlap_pair_count if overlap_pair_count else 0.0
        overlap_corr = SimulatedBacktestEngine._pearson_correlation(pnl_left, pnl_right)
        worst_clustered_loss, worst_clustered_loss_trade_count = SimulatedBacktestEngine._worst_clustered_loss(sorted_by_entry)
        max_simul_dd = SimulatedBacktestEngine._max_simultaneous_drawdown(sorted_by_entry)

        return {
            "avg_concurrent_positions": avg_concurrent,
            "max_concurrent_positions": max_concurrent,
            "overlap_distribution": overlap_distribution,
            "overlap_pair_count": overlap_pair_count,
            "avg_overlap_duration_minutes": avg_overlap_duration,
            "overlap_pnl_correlation": overlap_corr,
            "worst_clustered_loss": worst_clustered_loss,
            "worst_clustered_loss_trade_count": worst_clustered_loss_trade_count,
            "max_simultaneous_drawdown": max_simul_dd,
        }

    @staticmethod
    def _worst_clustered_loss(trades: list[SimulatedTrade]) -> tuple[float, int]:
        if not trades:
            return 0.0, 0
        worst_loss = 0.0
        worst_count = 0
        cluster_end = trades[0].exit_ts
        cluster_loss = min(trades[0].pnl, 0.0)
        cluster_count = 1 if trades[0].pnl < 0 else 0
        for trade in trades[1:]:
            if trade.entry_ts < cluster_end:
                cluster_end = max(cluster_end, trade.exit_ts)
                if trade.pnl < 0:
                    cluster_loss += trade.pnl
                    cluster_count += 1
            else:
                if cluster_loss < worst_loss:
                    worst_loss = cluster_loss
                    worst_count = cluster_count
                cluster_end = trade.exit_ts
                cluster_loss = min(trade.pnl, 0.0)
                cluster_count = 1 if trade.pnl < 0 else 0
        if cluster_loss < worst_loss:
            worst_loss = cluster_loss
            worst_count = cluster_count
        return worst_loss, worst_count

    @staticmethod
    def _max_simultaneous_drawdown(trades: list[SimulatedTrade]) -> float:
        events: list[tuple[datetime, int, float]] = []
        for trade in trades:
            loss_contrib = min(trade.pnl, 0.0)
            if loss_contrib == 0.0:
                continue
            events.append((trade.entry_ts, 1, loss_contrib))
            events.append((trade.exit_ts, -1, loss_contrib))
        active_loss = 0.0
        worst = 0.0
        for _, event_type, loss_contrib in sorted(events, key=lambda item: (item[0], item[1])):
            if event_type == 1:
                active_loss += loss_contrib
                worst = min(worst, active_loss)
            else:
                active_loss -= loss_contrib
        return worst

    @staticmethod
    def _pearson_correlation(left: list[float], right: list[float]) -> float:
        if len(left) < 2 or len(right) < 2 or len(left) != len(right):
            return 0.0
        mean_left = sum(left) / len(left)
        mean_right = sum(right) / len(right)
        cov = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
        var_left = sum((a - mean_left) ** 2 for a in left)
        var_right = sum((b - mean_right) ** 2 for b in right)
        if var_left <= 0.0 or var_right <= 0.0:
            return 0.0
        return cov / ((var_left ** 0.5) * (var_right ** 0.5))

    def _check_exit(
        self,
        position: SimulatedPosition,
        bar: dict[str, Any],
    ) -> tuple[ExecutionReport | None, float | None, str | None]:
        stop_hit = False
        target_hit = False
        if position.side == Side.BUY:
            stop_hit = bar["low"] <= position.stop_price
            target_hit = position.target_price is not None and bar["high"] >= position.target_price
        else:
            stop_hit = bar["high"] >= position.stop_price
            target_hit = position.target_price is not None and bar["low"] <= position.target_price

        if stop_hit and position.stop_order_id is not None:
            stop_reason = "breakeven_stop" if position.breakeven_stop_active else "protective_stop"
            report = self.execution_engine.adapter.emit_fill(
                position.stop_order_id,
                position.qty,
                position.stop_price,
                timestamp=bar["ts_event"],
            )
            return report, position.stop_price, stop_reason
        if target_hit and position.target_order_id is not None and position.target_price is not None:
            report = self.execution_engine.adapter.emit_fill(
                position.target_order_id,
                position.qty,
                position.target_price,
                timestamp=bar["ts_event"],
            )
            return report, position.target_price, "profit_target"
        if bar["ts_event"].time() >= time(15, 0):
            report = self._emit_flatten_fill(position, bar["ts_event"], bar["close"], "end_of_day")
            return report, bar["close"], "end_of_day"
        return None, None, None

    def _emit_flatten_fill(
        self,
        position: SimulatedPosition,
        now: datetime,
        exit_price: float,
        reason: str,
    ) -> ExecutionReport:
        chain = self.execution_engine.order_manager.chain_for_intent(position.intent_id)
        parent_order_id = chain.parent_order_id if chain else None
        if chain:
            for child_order_id in (chain.stop_order_id, chain.target_order_id):
                if not child_order_id:
                    continue
                child_order = self.execution_engine.order_manager.orders.get(child_order_id)
                if child_order and child_order.state not in {
                    OrderState.CANCELED,
                    OrderState.REJECTED,
                    OrderState.FILLED,
                }:
                    self.execution_engine._cancel_order(child_order_id)

        exit_side = Side.SELL if position.side == Side.BUY else Side.BUY
        order_id = f"sim-exit-{position.intent_id}-{reason}"
        synthetic_exit = BrokerOrder(
            order_id=order_id,
            symbol=position.symbol,
            side=exit_side,
            qty=position.qty,
            order_type=OrderType.MARKET,
            tif=TimeInForce.DAY,
            state=OrderState.ACKNOWLEDGED,
            parent_order_id=parent_order_id,
            role="flatten",
            intent_id=position.intent_id,
            reason=reason,
            broker_order_id=f"broker-{order_id}",
            submitted_at=now,
            updated_at=now,
        )
        self.execution_engine.adapter.orders[order_id] = synthetic_exit
        self.execution_engine.order_manager.orders[order_id] = synthetic_exit
        return self.execution_engine.adapter.emit_fill(order_id, position.qty, exit_price, timestamp=now)

    @staticmethod
    def _record_custom_denial(
        counts: Counter[str],
        examples: defaultdict[str, list[dict[str, object]]],
        reason: str,
        intent: OrderIntent,
        now: datetime,
    ) -> None:
        counts[reason] += 1
        if len(examples[reason]) >= 5:
            return
        examples[reason].append(
            {
                "intent_id": intent.intent_id,
                "symbol": intent.symbol,
                "signal_ts": _coerce_time(intent.signal_ts).isoformat(),
                "decision_ts": now.isoformat(),
                "signal_score": intent.signal_score,
                "entry_price": intent.entry_price,
                "stop_price": intent.stop_price,
                "target_price": intent.target_price,
                "regime": _regime_value(intent),
                "side": _side_value(intent.side),
            }
        )

    @staticmethod
    def _close_trade(
        position: SimulatedPosition,
        exit_price: float,
        exit_ts: datetime,
        exit_reason: str,
        commission_per_lot: float = 0.0,
    ) -> SimulatedTrade:
        raw_pnl = _trade_pnl(position.symbol, position.side, position.entry_price, exit_price, position.qty)
        # Round-trip commission: entry side + exit side, scaled by quantity.
        round_trip_commission = commission_per_lot * position.qty * 2
        return SimulatedTrade(
            intent_id=position.intent_id,
            symbol=position.symbol,
            side=_side_value(position.side),
            entry_price=position.entry_price,
            exit_price=exit_price,
            qty=position.qty,
            pnl=raw_pnl - round_trip_commission,
            entry_ts=position.entry_ts,
            exit_ts=exit_ts,
            exit_reason=exit_reason,
            regime=position.regime,
            time_bucket=_time_bucket_central(position.entry_ts),
            is_stacked=position.is_stacked,
            stacking_reason=position.stacking_reason,
            risk_multiplier=position.risk_multiplier,
            position_scale=position.position_scale,
            cluster_risk_usage_pct=position.cluster_risk_usage_pct,
            mae=position.mae,
            mfe=position.mfe,
            is_reentry=position.is_reentry,
            entry_number_in_trend=position.entry_number_in_trend,
            trend_key=position.trend_key,
            atr_at_entry=position.atr_at_entry,
            stop_distance=position.stop_distance,
            stop_distance_pct=position.stop_distance_pct,
            stop_distance_ticks=position.stop_distance_ticks,
            abnormal_stop=position.abnormal_stop,
        )
