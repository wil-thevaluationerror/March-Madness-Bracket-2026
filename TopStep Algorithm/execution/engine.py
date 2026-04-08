from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from config import TraderConfig
from execution.broker import BrokerAdapter
from execution.logging import EventLogger
from execution.order_manager import ManagedOrderChain, OrderManager
from execution.reconciler import Reconciler
from execution.scheduler import SessionScheduler
from execution.state_store import StateStore
from models.instruments import resolve_instrument
from models.orders import (
    BrokerOrder,
    ExecutionDecision,
    ExecutionReport,
    OrderIntent,
    OrderPlan,
    OrderState,
    OrderType,
    Regime,
    Side,
    TimeInForce,
)
from risk.engine import RiskEngine
from risk.execution_checks import AccountState, ActiveExposure, validate_intent


class ExecutionEngine:
    def __init__(
        self,
        config: TraderConfig,
        risk_engine: RiskEngine,
        adapter: BrokerAdapter,
        order_manager: OrderManager,
        logger: EventLogger,
        mode: str = "live",
    ) -> None:
        self.config = config
        self.risk_engine = risk_engine
        self.adapter = adapter
        self.order_manager = order_manager
        self.logger = logger
        self.mode = mode
        self.scheduler = SessionScheduler(config.session)
        self.reconciler = Reconciler(adapter, order_manager, risk_engine)
        self.state_store = StateStore(config.execution.trade_log_dir)
        self.processed_intents: dict[str, datetime] = {}
        self.intent_registry: dict[str, OrderIntent] = {}
        self.intent_parent_map: dict[str, str] = {}
        self.active_position_intent: dict[str, str] = {}
        self.counted_entry_orders: set[str] = set()
        self.order_submission_times: deque[datetime] = deque()
        self.denial_counts: defaultdict[str, int] = defaultdict(int)
        self.denial_examples: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
        self.disconnect_started_at: datetime | None = None
        self.flatten_in_progress = False
        self.trading_paused = False

    @property
    def is_backtest_mode(self) -> bool:
        return self.mode == "backtest" or self.risk_engine.is_backtest_mode

    @staticmethod
    def _now() -> datetime:
        return datetime.now(UTC)

    @staticmethod
    def _coerce_time(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def startup(self, *, now: datetime | None = None) -> bool:
        now = self._coerce_time(now or self._now())
        if self.mode != "backtest":
            self._restore_state()
            self._cleanup_processed_intents(now, persist=False)
        self.adapter.connect()
        if not self._ensure_connection(now):
            self.logger.log_event("startup_failed", reason="adapter_unavailable")
            if self.mode != "backtest":
                self._persist_state()
            return False
        reconcile_result = self.reconciler.reconcile_all(now)
        self.logger.log_event("startup_reconciliation", result=reconcile_result)
        if not (reconcile_result.positions_match and reconcile_result.orders_match):
            self.logger.log_event("startup_failed", reason="reconciliation_mismatch", result=reconcile_result)
            if self.mode != "backtest":
                self._persist_state()
            return False
        for symbol, position in list(self.risk_engine.state.positions.items()):
            if not position.is_flat:
                self._verify_stop_coverage(symbol, now, flatten_on_missing=False)
        self.logger.log_event("startup_complete", mode=self.adapter.mode, restored_state=bool(self.processed_intents))
        if self.mode != "backtest":
            self._persist_state()
        return not self.risk_engine.state.kill_switch.armed

    def heartbeat(self, *, now: datetime | None = None) -> None:
        now = self._coerce_time(now or self._now())
        self._cleanup_processed_intents(now)
        if not self._ensure_connection(now):
            self._persist_state()
            return
        self.drain_adapter_events()
        reconcile_result = self.reconciler.reconcile_all(now)
        self.logger.log_event("reconciliation_result", result=reconcile_result)
        for order in self.order_manager.cancel_stale_orders(now, self.config.execution.stale_order_seconds):
            if order.role == "entry":
                self._cancel_order(order.order_id)
                self.logger.log_event("stale_order_canceled", order=order)
        if self.scheduler.should_force_flatten(now):
            self.flatten_all("end_of_day", now=now)
            self.drain_adapter_events()
        if self.flatten_in_progress and self._flatten_complete():
            self.flatten_in_progress = False
            self.logger.log_event("flatten_complete")
        self._persist_state()

    def safe_shutdown(self, reason: str = "operator_shutdown", *, now: datetime | None = None) -> None:
        now = self._coerce_time(now or self._now())
        self.flatten_all(reason, now=now)
        self.drain_adapter_events()
        self.logger.log_event("shutdown_complete", reason=reason)
        self._persist_state()

    def submit_intent(self, intent: OrderIntent, *, now: datetime | None = None) -> ExecutionDecision:
        now = self._coerce_time(now or self._now())
        self._cleanup_processed_intents(now)
        self.logger.log_event("signal_emitted", intent=intent)

        if self.flatten_in_progress:
            decision = ExecutionDecision(False, "flatten_in_progress", 0, None)
            self._record_denial(decision.reason, intent, now)
            self.logger.log_event("intent_denied", intent=intent, decision=decision)
            return decision
        if self.trading_paused or not self.adapter.is_connected():
            decision = ExecutionDecision(False, "adapter_unavailable", 0, None)
            self._record_denial(decision.reason, intent, now)
            self.logger.log_event("intent_denied", intent=intent, decision=decision)
            return decision
        if intent.intent_id in self.processed_intents:
            decision = ExecutionDecision(False, "duplicate_intent", 0, None)
            self._record_denial(decision.reason, intent, now)
            self.logger.log_event("intent_denied", intent=intent, decision=decision)
            return decision
        if self._slippage_exceeded(intent):
            decision = ExecutionDecision(False, "entry_slippage_exceeded", 0, None)
            self._record_denial(decision.reason, intent, now)
            self.logger.log_event("intent_denied", intent=intent, decision=decision)
            return decision

        account_state = AccountState(**self.adapter.get_account())
        decision = validate_intent(
            intent=intent,
            account_state=account_state,
            risk_engine=self.risk_engine,
            risk_limits=self.config.risk,
            session_config=self.config.session,
            now=now,
            open_position_count=self.order_manager.active_position_count(),
            active_exposures=self._active_exposures(),
        )
        if not decision.approved:
            self._record_denial(decision.reason, intent, now)
            self.logger.log_event("intent_denied", intent=intent, decision=decision)
            return decision

        working_metadata = dict(intent.metadata)
        working_metadata["trend_key"] = self.risk_engine._trend_key_for_intent(intent)
        if self.order_manager.active_position_count() == 0 and self.risk_engine.is_reentry_for_intent(intent):
            working_metadata["is_reentry"] = True
            working_metadata["entry_number_in_trend"] = self.risk_engine.projected_entry_number(intent)
        else:
            working_metadata["is_reentry"] = False
            working_metadata["entry_number_in_trend"] = 1
        if decision.reason.startswith("stacking_allowed_"):
            working_metadata["is_stacked"] = True
            working_metadata["stacking_reason"] = decision.reason.removeprefix("stacking_allowed_")
        working_intent = replace(intent, qty=decision.normalized_qty, metadata=working_metadata)
        plan = self._build_order_plan(working_intent)
        decision = ExecutionDecision(True, decision.reason, working_intent.qty, plan)

        self.risk_engine.register_open_intent(working_intent.symbol)
        self.processed_intents[working_intent.intent_id] = now
        self.intent_registry[working_intent.intent_id] = working_intent
        self.order_manager.register_parent_child(intent.intent_id, plan.entry, plan.stop, plan.target, now)
        self.intent_parent_map[intent.intent_id] = plan.entry.order_id
        self.logger.log_event("intent_approved", intent=working_intent, decision=decision)
        try:
            submitted_plan = self._submit_plan(working_intent, plan, now)
        except RuntimeError as exc:
            self.risk_engine.resolve_open_intent(working_intent.symbol)
            if not self._is_local_submission_error(exc):
                self.risk_engine.record_api_error(now)
            decision = ExecutionDecision(False, str(exc), 0, None)
            self.logger.log_event("intent_failed", intent=working_intent, error=str(exc))
            self._persist_state()
            return decision

        self._persist_state()
        return ExecutionDecision(True, decision.reason, working_intent.qty, submitted_plan)

    def on_fill(self, report: ExecutionReport) -> None:
        self.logger.log_event("fill", report=report)
        order = self.adapter.orders[report.order_id]
        prior_position = self.risk_engine.position_for(order.symbol)
        prior_avg_price = prior_position.avg_price
        sign = 1 if order.side == Side.BUY else -1

        if order.role == "entry":
            self.risk_engine.record_fill(order.symbol, sign * report.fill_qty, report.fill_price or 0.0, report.timestamp)
            self.active_position_intent[order.symbol] = order.intent_id or order.order_id
            if order.order_id not in self.counted_entry_orders:
                self.risk_engine.record_entry(intent=self.intent_registry.get(order.intent_id or ""), now=report.timestamp)
                self.counted_entry_orders.add(order.order_id)
            self.order_manager.on_order_update(order, report.timestamp)
            self._submit_or_update_children(order, report.timestamp)
        else:
            pnl = self._calculate_exit_pnl(order.symbol, order.side, report.fill_qty, report.fill_price, prior_avg_price)
            self.risk_engine.record_fill(order.symbol, sign * report.fill_qty, report.fill_price or 0.0, report.timestamp)
            self.order_manager.on_order_update(order, report.timestamp)
            self._record_exit_pnl(order, report, pnl)

        self._verify_stop_coverage(order.symbol, report.timestamp)
        if self.flatten_in_progress and self._flatten_complete():
            self.flatten_in_progress = False
            self.logger.log_event("flatten_complete")
        self._persist_state()

    def handle_execution_report(self, report: ExecutionReport) -> None:
        order = self.adapter.orders.get(report.order_id)
        if order is None:
            return
        self.order_manager.on_order_update(order, report.timestamp)
        if report.status in {OrderState.PARTIAL, OrderState.FILLED} and report.fill_qty:
            self.on_fill(report)
        elif report.status == OrderState.REJECTED:
            self.logger.log_event("order_rejected", report=report)
            self._handle_rejection(order, report.timestamp)
            self._persist_state()
        elif report.status == OrderState.CANCELED:
            self.logger.log_event("order_canceled", report=report)
            self._persist_state()

    def handle_exit_signal(self, symbol: str, reason: str, now: datetime | None = None) -> None:
        now = self._coerce_time(now or self._now())
        position = self.risk_engine.position_for(symbol)
        if position.is_flat or not self.adapter.is_connected():
            return
        for order in list(self.order_manager.active_orders()):
            if order.symbol == symbol and order.role in {"stop", "target", "entry"}:
                self._cancel_order(order.order_id)
        if self._has_active_flatten_order(symbol):
            return
        side = Side.SELL if position.qty > 0 else Side.BUY
        intent_id = self.active_position_intent.get(symbol)
        flatten_order = BrokerOrder(
            order_id=f"flatten-{uuid4().hex[:10]}",
            symbol=symbol,
            side=side,
            qty=abs(position.qty),
            order_type=OrderType.MARKET,
            tif=self._default_tif(),
            state=OrderState.PENDING,
            parent_order_id=self.intent_parent_map.get(intent_id) if intent_id else None,
            role="flatten",
            intent_id=intent_id,
            reason=reason,
        )
        submitted = self._place_order(flatten_order, now, bypass_throttle=True)
        self.order_manager.on_order_update(submitted, now)
        self.logger.log_event("flatten_submitted", symbol=symbol, reason=reason, order=submitted)

    def flatten_all(self, reason: str, now: datetime | None = None) -> None:
        now = self._coerce_time(now or self._now())
        if self.flatten_in_progress and not self._flatten_complete():
            self.logger.log_event("flatten_skipped", reason=reason, detail="flatten_in_progress")
            return
        self.flatten_in_progress = True
        self.logger.log_event("flatten_attempt", reason=reason)
        if not self._ensure_connection(now):
            self.risk_engine.arm_kill_switch("flatten_without_connection", now)
            self._persist_state()
            return
        for order in list(self.order_manager.active_orders()):
            if order.role != "flatten":
                self._cancel_order(order.order_id)
        for symbol, position in list(self.risk_engine.state.positions.items()):
            if not position.is_flat:
                self.handle_exit_signal(symbol, reason=reason, now=now)
        self.logger.log_event("flatten_requested", reason=reason)
        self._persist_state()

    def drain_adapter_events(self) -> None:
        for report in self.adapter.poll_events():
            self.handle_execution_report(report)

    def _build_order_plan(self, intent: OrderIntent) -> OrderPlan:
        entry = BrokerOrder(
            order_id=f"entry-{uuid4().hex[:10]}",
            symbol=intent.symbol,
            side=intent.side,
            qty=intent.qty,
            order_type=intent.entry_type,
            tif=intent.time_in_force,
            state=OrderState.PENDING,
            price=intent.entry_price,
            role="entry",
            intent_id=intent.intent_id,
            reason=intent.reason,
        )
        exit_side = Side.SELL if intent.side == Side.BUY else Side.BUY
        stop = BrokerOrder(
            order_id=f"stop-{uuid4().hex[:10]}",
            symbol=intent.symbol,
            side=exit_side,
            qty=0,
            order_type=OrderType.STOP_MARKET,
            tif=intent.time_in_force,
            state=OrderState.PENDING,
            stop_price=intent.stop_price,
            parent_order_id=entry.order_id,
            role="stop",
            intent_id=intent.intent_id,
            reason="protective_stop",
        )
        target = None
        if intent.target_price is not None:
            target = BrokerOrder(
                order_id=f"target-{uuid4().hex[:10]}",
                symbol=intent.symbol,
                side=exit_side,
                qty=0,
                order_type=OrderType.LIMIT,
                tif=intent.time_in_force,
                state=OrderState.PENDING,
                price=intent.target_price,
                parent_order_id=entry.order_id,
                role="target",
                intent_id=intent.intent_id,
                reason="profit_target",
            )
        return OrderPlan(entry=entry, stop=stop, target=target)

    def _submit_plan(self, intent: OrderIntent, plan: OrderPlan, now: datetime) -> OrderPlan:
        submitted_entry = self._place_order(plan.entry, now)
        if submitted_entry.state == OrderState.REJECTED:
            raise RuntimeError("entry_rejected")
        self.order_manager.on_order_update(submitted_entry, now)
        self.logger.log_event("entry_submitted", intent=intent, entry=submitted_entry)
        return OrderPlan(entry=submitted_entry, stop=plan.stop, target=plan.target)

    def _submit_or_update_children(self, entry_order: BrokerOrder, now: datetime) -> None:
        chain = self.order_manager.chains_by_parent.get(entry_order.order_id)
        if chain is None:
            return
        desired_qty = entry_order.filled_qty
        if desired_qty <= 0:
            return
        stop_order = self.order_manager.orders.get(chain.stop_order_id) if chain.stop_order_id else None
        if stop_order is None:
            self.risk_engine.arm_kill_switch("missing_stop_template", now)
            self.flatten_all("missing_stop_template", now=now)
            return
        updated_stop = self._submit_or_resize_child(stop_order, desired_qty, now)
        self.order_manager.on_order_update(updated_stop, now)

        if updated_stop.state == OrderState.REJECTED:
            self.risk_engine.arm_kill_switch("protective_order_rejected", now)
            self.flatten_all("protective_order_rejected", now=now)
            return

        if chain.target_order_id:
            target_order = self.order_manager.orders.get(chain.target_order_id)
            if target_order is not None:
                updated_target = self._submit_or_resize_child(target_order, desired_qty, now)
                self.order_manager.on_order_update(updated_target, now)
                if updated_target.state == OrderState.REJECTED:
                    self.logger.log_event("target_rejected", order=updated_target)

    def _submit_or_resize_child(self, order: BrokerOrder, desired_qty: int, now: datetime) -> BrokerOrder:
        if order.submitted_at is None:
            candidate = BrokerOrder(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                qty=desired_qty,
                order_type=order.order_type,
                tif=order.tif,
                state=OrderState.PENDING,
                price=order.price,
                stop_price=order.stop_price,
                parent_order_id=order.parent_order_id,
                role=order.role,
                intent_id=order.intent_id,
                reason=order.reason,
            )
            submitted = self._place_order(candidate, now, bypass_throttle=True)
            self.logger.log_event("child_order_submitted", order=submitted)
            return submitted
        if order.qty == desired_qty:
            return order
        if order.state in {OrderState.CANCELED, OrderState.REJECTED, OrderState.FILLED}:
            return order
        updated = self.adapter.replace_order(order.order_id, qty=desired_qty)
        self.logger.log_event("child_order_resized", order=updated)
        return updated

    def _verify_stop_coverage(self, symbol: str, now: datetime, *, flatten_on_missing: bool = True) -> None:
        position = self.risk_engine.position_for(symbol)
        required_qty = abs(position.qty)
        covered = self.order_manager.ensure_stop_coverage(symbol, required_qty)
        self.risk_engine.record_stop_coverage(symbol, required_qty if covered else 0, now)
        self.logger.log_event("stop_coverage", symbol=symbol, required_qty=required_qty, covered=covered)
        if required_qty > 0 and not covered:
            if not self.risk_engine.state.kill_switch.armed:
                self.risk_engine.arm_kill_switch("missing_protective_stop", now)
            if flatten_on_missing and self.adapter.is_connected():
                self.flatten_all("missing_protective_stop", now=now)

    def _record_exit_pnl(self, order: BrokerOrder, report: ExecutionReport, pnl: float) -> None:
        symbol = order.symbol
        position = self.risk_engine.position_for(symbol)
        total_trade_pnl = self.risk_engine.update_open_trade_pnl(symbol, pnl)
        intent = self.intent_registry.get(order.intent_id or "")
        if position.is_flat:
            if intent and bool(intent.metadata.get("is_stacked")):
                self.risk_engine.record_stacked_trade_close(total_trade_pnl, report.timestamp)
            self.risk_engine.record_exit_context(intent, total_trade_pnl, report.timestamp)
            self.risk_engine.record_trade_close(total_trade_pnl, report.timestamp)
            self.risk_engine.clear_open_trade_pnl(symbol)
            self.risk_engine.resolve_open_intent(symbol)
            self.active_position_intent.pop(symbol, None)
            self._cancel_sibling_exit_orders(order)
        self.logger.log_trade(
            symbol=symbol,
            pnl=pnl,
            total_trade_pnl=total_trade_pnl,
            report=report,
            reason=order.reason,
            intent_id=order.intent_id,
            regime=intent.regime if intent else None,
            signal_ts=intent.signal_ts if intent else None,
            strategy_id=intent.strategy_id if intent else None,
        )

    def _handle_rejection(self, order: BrokerOrder, now: datetime) -> None:
        if order.role == "entry":
            self.risk_engine.resolve_open_intent(order.symbol)
        elif order.role == "stop":
            self.risk_engine.arm_kill_switch("protective_order_rejected", now)
            self.flatten_all("protective_order_rejected", now=now)

    def _ensure_connection(self, now: datetime) -> bool:
        if self.adapter.is_connected():
            self.disconnect_started_at = None
            self.trading_paused = False
            return True
        if self.disconnect_started_at is None:
            self.disconnect_started_at = now
            self.logger.log_event("adapter_disconnected")
        self.trading_paused = True
        if self.adapter.reconnect():
            self.disconnect_started_at = None
            self.trading_paused = False
            self.risk_engine.reset_api_errors()
            self.logger.log_event("adapter_reconnected")
            return True
        if self._has_open_risk() and self.disconnect_started_at and now - self.disconnect_started_at >= timedelta(
            seconds=self.config.execution.reconnect_timeout_seconds
        ):
            self.risk_engine.arm_kill_switch("reconnect_timeout", now)
            self.logger.log_event("adapter_reconnect_timeout")
        return False

    def _slippage_exceeded(self, intent: OrderIntent) -> bool:
        if intent.entry_price is None:
            return False
        market_price = self.adapter.get_market_price(intent.symbol)
        if market_price is None:
            return False
        tick_delta = abs(market_price - intent.entry_price) / self._tick_size(intent.symbol)
        return tick_delta > self.config.risk.max_slippage_ticks

    def _place_order(self, order: BrokerOrder, now: datetime, *, bypass_throttle: bool = False) -> BrokerOrder:
        now = self._coerce_time(now)
        if not bypass_throttle and not self.is_backtest_mode:
            self._throttle_order_submission(now)
        submitted = self.adapter.place_order(order)
        if not bypass_throttle and not self.is_backtest_mode:
            self.order_submission_times.append(now)
        return submitted

    def _cancel_order(self, order_id: str) -> None:
        try:
            self.adapter.cancel_order(order_id)
        except RuntimeError as exc:
            self.risk_engine.record_api_error(self._now())
            self.logger.log_event("order_cancel_failed", order_id=order_id, error=str(exc))

    def _throttle_order_submission(self, now: datetime) -> None:
        window_start = now - timedelta(seconds=1)
        while self.order_submission_times and self.order_submission_times[0] < window_start:
            self.order_submission_times.popleft()
        if len(self.order_submission_times) >= self.config.execution.max_orders_per_second:
            raise RuntimeError("order_throttled")
        if self.order_submission_times:
            last_order_time = self.order_submission_times[-1]
            min_gap = 1.0 / max(self.config.execution.max_orders_per_second, 1)
            if (now - last_order_time).total_seconds() < min_gap:
                raise RuntimeError("order_submission_cooldown")

    @staticmethod
    def _is_local_submission_error(exc: RuntimeError) -> bool:
        return str(exc) in {"order_throttled", "order_submission_cooldown"}

    def _cleanup_processed_intents(self, now: datetime, *, persist: bool = True) -> None:
        expiry = timedelta(seconds=self.config.execution.intent_expiry_seconds)
        expired = [intent_id for intent_id, ts in self.processed_intents.items() if now - ts > expiry]
        for intent_id in expired:
            self.processed_intents.pop(intent_id, None)
        if expired and persist:
            self.logger.log_event("processed_intents_pruned", count=len(expired))
            self._persist_state()

    def _record_denial(self, reason: str, intent: OrderIntent, now: datetime) -> None:
        normalized_reason = str(reason or "unknown")
        self.denial_counts[normalized_reason] += 1
        examples = self.denial_examples[normalized_reason]
        if len(examples) >= 5:
            return
        examples.append(
            {
                "intent_id": intent.intent_id,
                "symbol": intent.symbol,
                "signal_ts": self._coerce_time(intent.signal_ts).isoformat(),
                "decision_ts": now.isoformat(),
                "signal_score": intent.signal_score,
                "entry_price": intent.entry_price,
                "stop_price": intent.stop_price,
                "target_price": intent.target_price,
                "regime": intent.regime.value if hasattr(intent.regime, "value") else str(intent.regime),
                "side": intent.side.value if hasattr(intent.side, "value") else str(intent.side),
            }
        )

    def _active_exposures(self) -> list[ActiveExposure]:
        exposures: list[ActiveExposure] = []
        for chain in self.order_manager.active_chains():
            parent_order = self.order_manager.orders.get(chain.parent_order_id)
            if parent_order is None:
                continue
            intent = self.intent_registry.get(chain.intent_id)
            stop_order = self.order_manager.orders.get(chain.stop_order_id) if chain.stop_order_id else None
            stop_price = stop_order.stop_price if stop_order and stop_order.stop_price is not None else parent_order.price
            entry_price = parent_order.avg_fill_price if parent_order.avg_fill_price is not None else parent_order.price
            if entry_price is None or stop_price is None:
                continue
            exposures.append(
                ActiveExposure(
                    intent_id=chain.intent_id,
                    symbol=parent_order.symbol,
                    side=parent_order.side,
                    qty=max(chain.filled_qty - chain.closed_qty, 0),
                    regime=intent.regime if intent else Regime.UNKNOWN,
                    entry_price=entry_price,
                    stop_price=stop_price,
                    signal_score=float(intent.signal_score) if intent else 0.0,
                    entry_ts=parent_order.updated_at or parent_order.submitted_at,
                    trend_state=str(intent.metadata.get("trend_state") or "unknown") if intent else "unknown",
                    is_stacked=bool(intent.metadata.get("is_stacked")) if intent else False,
                )
            )
        return exposures

    def denial_analytics(self) -> dict[str, object]:
        return {
            "counts": dict(self.denial_counts),
            "examples": {reason: list(examples) for reason, examples in self.denial_examples.items()},
        }

    def _persist_state(self) -> None:
        if self.mode == "backtest":
            return
        snapshot = {
            "processed_intents": {intent_id: timestamp.isoformat() for intent_id, timestamp in self.processed_intents.items()},
            "intent_parent_map": dict(self.intent_parent_map),
            "active_position_intent": dict(self.active_position_intent),
            "counted_entry_orders": sorted(self.counted_entry_orders),
            "flatten_in_progress": self.flatten_in_progress,
            "disconnect_started_at": self.disconnect_started_at.isoformat() if self.disconnect_started_at else None,
            "risk_state": self.risk_engine.snapshot_state(),
            "order_manager": self.order_manager.snapshot_state(),
        }
        self.state_store.save(snapshot)

    def _restore_state(self) -> None:
        if self.mode == "backtest":
            return
        snapshot = self.state_store.load()
        if not snapshot:
            return
        self.processed_intents = {
            intent_id: datetime.fromisoformat(timestamp)
            for intent_id, timestamp in dict(snapshot.get("processed_intents", {})).items()
        }
        self.intent_parent_map = {
            str(intent_id): str(parent_id)
            for intent_id, parent_id in dict(snapshot.get("intent_parent_map", {})).items()
        }
        self.active_position_intent = {
            str(symbol): str(intent_id)
            for symbol, intent_id in dict(snapshot.get("active_position_intent", {})).items()
        }
        self.counted_entry_orders = set(snapshot.get("counted_entry_orders", []))
        disconnect_started_at = snapshot.get("disconnect_started_at")
        self.disconnect_started_at = datetime.fromisoformat(disconnect_started_at) if disconnect_started_at else None
        self.flatten_in_progress = bool(snapshot.get("flatten_in_progress", False))
        self.risk_engine.restore_state(dict(snapshot.get("risk_state", {})))
        self.order_manager.restore_state(dict(snapshot.get("order_manager", {})))

    def _flatten_complete(self) -> bool:
        any_open_positions = any(not position.is_flat for position in self.risk_engine.state.positions.values())
        any_flatten_orders = any(order.role == "flatten" for order in self.order_manager.active_orders())
        return not any_open_positions and not any_flatten_orders

    def _has_open_risk(self) -> bool:
        return any(not position.is_flat for position in self.risk_engine.state.positions.values()) or bool(
            self.order_manager.active_orders()
        )

    def _has_active_flatten_order(self, symbol: str) -> bool:
        return any(
            order.symbol == symbol and order.role == "flatten" and order.state in {OrderState.PENDING, OrderState.ACKNOWLEDGED}
            for order in self.order_manager.active_orders()
        )

    def _default_tif(self) -> TimeInForce:
        return TimeInForce.DAY

    @staticmethod
    def _tick_size(symbol: str) -> float:
        return resolve_instrument(symbol).tick_size

    @staticmethod
    def _calculate_exit_pnl(
        symbol: str,
        exit_side: Side,
        fill_qty: int,
        fill_price: float | None,
        prior_avg_price: float | None,
    ) -> float:
        if fill_price is None or prior_avg_price is None:
            return 0.0
        signed = 1 if exit_side == Side.SELL else -1
        return (fill_price - prior_avg_price) * fill_qty * signed * resolve_instrument(symbol).point_value

    def _cancel_sibling_exit_orders(self, filled_exit_order: BrokerOrder) -> None:
        parent_id = filled_exit_order.parent_order_id
        if not parent_id:
            return
        chain = self.order_manager.chains_by_parent.get(parent_id)
        if chain is None:
            return
        sibling_ids = [chain.stop_order_id, chain.target_order_id]
        for sibling_id in sibling_ids:
            if not sibling_id or sibling_id == filled_exit_order.order_id:
                continue
            sibling = self.adapter.orders.get(sibling_id) or self.order_manager.orders.get(sibling_id)
            if sibling and sibling.state in {OrderState.PENDING, OrderState.ACKNOWLEDGED, OrderState.PARTIAL}:
                self._cancel_order(sibling_id)
