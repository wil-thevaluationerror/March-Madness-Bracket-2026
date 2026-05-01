from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

from trading_system.core.domain import (
    BrokerOrder,
    ExecutionReport,
    OrderState,
    OrderType,
    PositionSnapshot,
    Side,
    TimeInForce,
    TradingMode,
)


class ProjectXAdapter:
    def __init__(self, mode: TradingMode = TradingMode.MOCK) -> None:
        self.mode = mode
        self.connected = False
        self.orders: dict[str, BrokerOrder] = {}
        self.positions: dict[str, PositionSnapshot] = {}
        self.account_state = {
            "cash_balance": 0.0,
            "net_liquidation": 0.0,
            "cushion_to_max_loss_limit": 1000.0,
        }
        self.event_queue: deque[ExecutionReport] = deque()
        self.callbacks: defaultdict[str, list] = defaultdict(list)
        self.reject_next_order = False
        self.reject_roles: set[str] = set()
        self.market_prices: dict[str, float] = {}
        self.reconnect_should_fail = False

    def connect(self) -> None:
        self.connected = True

    def is_connected(self) -> bool:
        return self.connected

    def reconnect(self) -> bool:
        if self.reconnect_should_fail:
            return False
        self.connected = True
        return True

    def disconnect(self) -> None:
        self.connected = False

    @staticmethod
    def _now() -> datetime:
        return datetime.now(UTC)

    def get_account(self) -> dict[str, float]:
        return dict(self.account_state)

    def get_market_price(self, symbol: str) -> float | None:
        return self.market_prices.get(symbol)

    def set_market_price(self, symbol: str, price: float) -> None:
        self.market_prices[symbol] = price

    def get_positions(self) -> dict[str, PositionSnapshot]:
        return {symbol: replace(position) for symbol, position in self.positions.items()}

    def get_open_orders(self) -> list[BrokerOrder]:
        return [
            replace(order)
            for order in self.orders.values()
            if order.state in {OrderState.PENDING, OrderState.ACKNOWLEDGED, OrderState.PARTIAL}
        ]

    def place_order(self, order_req: BrokerOrder) -> BrokerOrder:
        if not self.connected:
            raise RuntimeError("adapter_not_connected")

        order = replace(
            order_req,
            broker_order_id=f"broker-{uuid4().hex[:10]}",
            submitted_at=self._now(),
            updated_at=self._now(),
        )
        if self.reject_next_order or order.role in self.reject_roles:
            self.reject_next_order = False
            order.state = OrderState.REJECTED
            order.reason = "mock_rejection"
            self.orders[order.order_id] = order
            self._publish(
                ExecutionReport(
                    order_id=order.order_id,
                    broker_order_id=order.broker_order_id,
                    symbol=order.symbol,
                    status=OrderState.REJECTED,
                    fill_qty=0,
                    fill_price=None,
                    remaining_qty=order.qty,
                    side=order.side,
                    timestamp=self._now(),
                    message="mock_rejection",
                )
            )
            return order

        order.state = OrderState.ACKNOWLEDGED
        self.orders[order.order_id] = order
        self._publish(
            ExecutionReport(
                order_id=order.order_id,
                broker_order_id=order.broker_order_id,
                symbol=order.symbol,
                status=OrderState.ACKNOWLEDGED,
                fill_qty=0,
                fill_price=None,
                remaining_qty=order.qty,
                side=order.side,
                timestamp=self._now(),
            )
        )
        return order

    def cancel_order(self, order_id: str) -> BrokerOrder:
        if not self.connected:
            raise RuntimeError("adapter_not_connected")
        order = self.orders[order_id]
        order = replace(order, state=OrderState.CANCELED, updated_at=self._now())
        self.orders[order_id] = order
        self._publish(
            ExecutionReport(
                order_id=order.order_id,
                broker_order_id=order.broker_order_id,
                symbol=order.symbol,
                status=OrderState.CANCELED,
                fill_qty=0,
                fill_price=None,
                remaining_qty=max(order.qty - order.filled_qty, 0),
                side=order.side,
                timestamp=self._now(),
            )
        )
        return order

    def replace_order(self, order_id: str, **changes: object) -> BrokerOrder:
        if not self.connected:
            raise RuntimeError("adapter_not_connected")
        order = self.orders[order_id]
        order = replace(order, updated_at=self._now(), **changes)
        self.orders[order_id] = order
        return order

    def subscribe_events(self, event_type: str, callback) -> None:
        self.callbacks[event_type].append(callback)

    def emit_fill(
        self,
        order_id: str,
        fill_qty: int,
        fill_price: float,
        *,
        slippage: float = 0.0,
        fees: float = 0.0,
        timestamp: datetime | None = None,
    ) -> ExecutionReport:
        order = self.orders[order_id]
        event_time = timestamp or self._now()
        new_filled_qty = order.filled_qty + fill_qty
        remaining_qty = max(order.qty - new_filled_qty, 0)
        status = OrderState.FILLED if remaining_qty == 0 else OrderState.PARTIAL
        order = replace(
            order,
            filled_qty=new_filled_qty,
            avg_fill_price=fill_price,
            state=status,
            updated_at=event_time,
        )
        self.orders[order_id] = order
        self._update_position(order.symbol, order.side, fill_qty, fill_price)
        report = ExecutionReport(
            order_id=order.order_id,
            broker_order_id=order.broker_order_id,
            symbol=order.symbol,
            status=status,
            fill_qty=fill_qty,
            fill_price=fill_price,
            remaining_qty=remaining_qty,
            side=order.side,
            timestamp=event_time,
            slippage=slippage,
            fees=fees,
        )
        self._publish(report)
        return report

    def _update_position(self, symbol: str, side: Side, fill_qty: int, fill_price: float) -> None:
        position = self.positions.setdefault(symbol, PositionSnapshot(symbol=symbol))
        signed_qty = fill_qty if side == Side.BUY else -fill_qty
        if position.qty == 0:
            position.avg_price = fill_price
        position.qty += signed_qty
        if position.qty == 0:
            position.avg_price = None
        position.last_updated = self._now()

    def poll_events(self) -> list[ExecutionReport]:
        events = list(self.event_queue)
        self.event_queue.clear()
        return events

    def _publish(self, report: ExecutionReport) -> None:
        self.event_queue.append(report)
        for callback in self.callbacks["execution_report"]:
            callback(report)
