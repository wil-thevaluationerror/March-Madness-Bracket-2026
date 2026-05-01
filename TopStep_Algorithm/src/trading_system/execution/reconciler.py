from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from trading_system.execution.broker import BrokerAdapter
from trading_system.execution.order_manager import OrderManager
from trading_system.risk.engine import RiskEngine


@dataclass(slots=True)
class ReconciliationResult:
    positions_match: bool
    orders_match: bool
    details: str


_ACCOUNT_REFRESH_INTERVAL = 20  # reconciliation cycles (~60s at default 3s poll)


class Reconciler:
    def __init__(self, adapter: BrokerAdapter, order_manager: OrderManager, risk_engine: RiskEngine) -> None:
        self.adapter = adapter
        self.order_manager = order_manager
        self.risk_engine = risk_engine
        self._reconcile_cycle: int = 0

    def reconcile_positions(self) -> ReconciliationResult:
        broker_positions = self.adapter.get_positions()
        mismatches: list[str] = []
        for symbol in sorted(set(self.risk_engine.state.positions) | set(broker_positions)):
            internal_position = self.risk_engine.state.positions.get(symbol)
            internal_qty = internal_position.qty if internal_position else 0
            broker_qty = broker_positions.get(symbol).qty if symbol in broker_positions else 0
            if internal_qty != broker_qty:
                mismatches.append(f"{symbol}:internal={internal_qty}:broker={broker_qty}")
        return ReconciliationResult(not mismatches, True, ",".join(mismatches) or "positions_match")

    def reconcile_orders(self) -> ReconciliationResult:
        broker_open_ids = {order.order_id for order in self.adapter.get_open_orders()}
        internal_open_ids = {order.order_id for order in self.order_manager.active_orders()}
        missing_in_broker = internal_open_ids - broker_open_ids
        missing_internal = broker_open_ids - internal_open_ids
        mismatches: list[str] = []
        if missing_in_broker:
            mismatches.append(f"missing_in_broker={sorted(missing_in_broker)}")
        if missing_internal:
            mismatches.append(f"missing_internal={sorted(missing_internal)}")
        return ReconciliationResult(not mismatches, not mismatches, ",".join(mismatches) or "orders_match")

    def reconcile_all(self, now: datetime) -> ReconciliationResult:
        self._reconcile_cycle += 1
        if self._reconcile_cycle % _ACCOUNT_REFRESH_INTERVAL == 0:
            refresh = getattr(self.adapter, "refresh_account_state", None)
            if callable(refresh):
                refresh()

        positions = self.reconcile_positions()
        orders = self.reconcile_orders()
        ok = positions.positions_match and orders.orders_match
        if not ok and self.risk_engine.limits.kill_switch_on_reconcile_mismatch:
            self.risk_engine.arm_kill_switch("reconciliation_mismatch", now)
        details = ";".join([positions.details, orders.details])
        return ReconciliationResult(positions.positions_match, orders.orders_match, details)
