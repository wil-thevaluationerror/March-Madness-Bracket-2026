from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from trading_system.core.domain import BrokerOrder, OrderState


@dataclass(slots=True)
class ManagedOrderChain:
    intent_id: str
    parent_order_id: str
    stop_order_id: str | None = None
    target_order_id: str | None = None
    symbol: str = ""
    created_at: datetime | None = None
    last_update: datetime | None = None
    filled_qty: int = 0
    closed_qty: int = 0


class OrderManager:
    def __init__(self) -> None:
        self.orders: dict[str, BrokerOrder] = {}
        self.chains_by_parent: dict[str, ManagedOrderChain] = {}
        self.intent_to_parent: dict[str, str] = {}

    def register_parent_child(
        self,
        intent_id: str,
        parent: BrokerOrder,
        stop: BrokerOrder | None,
        target: BrokerOrder | None,
        created_at: datetime,
    ) -> ManagedOrderChain:
        self.orders[parent.order_id] = parent
        if stop:
            self.orders[stop.order_id] = stop
        if target:
            self.orders[target.order_id] = target
        chain = ManagedOrderChain(
            intent_id=intent_id,
            parent_order_id=parent.order_id,
            stop_order_id=stop.order_id if stop else None,
            target_order_id=target.order_id if target else None,
            symbol=parent.symbol,
            created_at=created_at,
            last_update=created_at,
        )
        self.chains_by_parent[parent.order_id] = chain
        self.intent_to_parent[intent_id] = parent.order_id
        return chain

    def on_order_update(self, order: BrokerOrder, updated_at: datetime) -> ManagedOrderChain | None:
        previous = self.orders.get(order.order_id)
        self.orders[order.order_id] = order
        parent_id = order.parent_order_id or order.order_id
        chain = self.chains_by_parent.get(parent_id)
        if chain is None and order.role == "entry":
            chain = ManagedOrderChain(
                intent_id=order.intent_id or order.order_id,
                parent_order_id=order.order_id,
                symbol=order.symbol,
                created_at=updated_at,
                last_update=updated_at,
            )
            self.chains_by_parent[order.order_id] = chain
        # Flatten orders may lose their parent_order_id linkage when the broker
        # adapter reconstructs the order object on fill (returning a fresh
        # BrokerOrder without preserving parent_order_id). Fall back to finding
        # the open chain for this symbol so closed_qty is always updated.
        if chain is None and order.role == "flatten" and order.symbol:
            for c in self.chains_by_parent.values():
                if c.symbol == order.symbol and c.filled_qty > c.closed_qty:
                    chain = c
                    break
        if chain is not None:
            chain.last_update = updated_at
            if order.role == "entry":
                chain.filled_qty = order.filled_qty
            elif order.role in {"stop", "target", "exit", "flatten"}:
                previous_qty = previous.filled_qty if previous else 0
                chain.closed_qty += max(order.filled_qty - previous_qty, 0)
        return chain

    def chain_for_intent(self, intent_id: str) -> ManagedOrderChain | None:
        parent_id = self.intent_to_parent.get(intent_id)
        if parent_id is None:
            return None
        return self.chains_by_parent.get(parent_id)

    def ensure_stop_coverage(self, symbol: str, required_qty: int) -> bool:
        covered_qty = 0
        for chain in self.chains_by_parent.values():
            if chain.symbol != symbol:
                continue
            stop_id = chain.stop_order_id
            if not stop_id:
                continue
            stop_order = self.orders.get(stop_id)
            if (
                stop_order
                and stop_order.submitted_at is not None
                and stop_order.state in {OrderState.ACKNOWLEDGED, OrderState.PARTIAL, OrderState.FILLED}
            ):
                covered_qty += stop_order.qty
                if covered_qty >= required_qty:
                    return True
        return required_qty == 0 or covered_qty >= required_qty

    def cancel_stale_orders(self, now: datetime, stale_seconds: int) -> list[BrokerOrder]:
        stale_orders: list[BrokerOrder] = []
        threshold = timedelta(seconds=stale_seconds)
        for order in self.orders.values():
            if order.submitted_at is None:
                continue
            if order.state not in {OrderState.PENDING, OrderState.ACKNOWLEDGED, OrderState.PARTIAL}:
                continue
            updated_at = order.updated_at or order.submitted_at
            if updated_at and now - updated_at > threshold:
                stale_orders.append(order)
        return stale_orders

    def active_orders(self) -> list[BrokerOrder]:
        return [
            order
            for order in self.orders.values()
            if order.submitted_at is not None and order.state in {OrderState.PENDING, OrderState.ACKNOWLEDGED, OrderState.PARTIAL}
        ]

    def active_position_count(self) -> int:
        count = 0
        for chain in self.chains_by_parent.values():
            if chain.filled_qty > chain.closed_qty:
                count += 1
        return count

    def active_chains(self) -> list[ManagedOrderChain]:
        return [
            chain
            for chain in self.chains_by_parent.values()
            if chain.filled_qty > chain.closed_qty
        ]

    def snapshot_state(self) -> dict[str, object]:
        def serialize_order(order: BrokerOrder) -> dict[str, object]:
            return {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side.value,
                "qty": order.qty,
                "order_type": order.order_type.value,
                "tif": order.tif.value,
                "state": order.state.value,
                "price": order.price,
                "stop_price": order.stop_price,
                "filled_qty": order.filled_qty,
                "avg_fill_price": order.avg_fill_price,
                "parent_order_id": order.parent_order_id,
                "role": order.role,
                "broker_order_id": order.broker_order_id,
                "submitted_at": order.submitted_at.isoformat() if order.submitted_at else None,
                "updated_at": order.updated_at.isoformat() if order.updated_at else None,
                "intent_id": order.intent_id,
                "reason": order.reason,
            }

        return {
            "orders": {order_id: serialize_order(order) for order_id, order in self.orders.items()},
            "chains_by_parent": {
                parent_id: {
                    "intent_id": chain.intent_id,
                    "parent_order_id": chain.parent_order_id,
                    "stop_order_id": chain.stop_order_id,
                    "target_order_id": chain.target_order_id,
                    "symbol": chain.symbol,
                    "created_at": chain.created_at.isoformat() if chain.created_at else None,
                    "last_update": chain.last_update.isoformat() if chain.last_update else None,
                    "filled_qty": chain.filled_qty,
                    "closed_qty": chain.closed_qty,
                }
                for parent_id, chain in self.chains_by_parent.items()
            },
            "intent_to_parent": dict(self.intent_to_parent),
        }

    def restore_state(self, snapshot: dict[str, object]) -> None:
        from datetime import datetime

        from trading_system.core.domain import OrderType, Side, TimeInForce

        def parse_dt(value: object) -> datetime | None:
            if isinstance(value, str) and value:
                return datetime.fromisoformat(value)
            return None

        self.orders = {}
        for order_id, raw in dict(snapshot.get("orders", {})).items():
            data = dict(raw)
            self.orders[str(order_id)] = BrokerOrder(
                order_id=str(order_id),
                symbol=str(data["symbol"]),
                side=Side(data["side"]),
                qty=int(data["qty"]),
                order_type=OrderType(data["order_type"]),
                tif=TimeInForce(data["tif"]),
                state=OrderState(data["state"]),
                price=data.get("price"),
                stop_price=data.get("stop_price"),
                filled_qty=int(data.get("filled_qty", 0)),
                avg_fill_price=data.get("avg_fill_price"),
                parent_order_id=data.get("parent_order_id"),
                role=str(data.get("role", "entry")),
                broker_order_id=data.get("broker_order_id"),
                submitted_at=parse_dt(data.get("submitted_at")),
                updated_at=parse_dt(data.get("updated_at")),
                intent_id=data.get("intent_id"),
                reason=data.get("reason"),
            )
        self.chains_by_parent = {}
        for parent_id, raw in dict(snapshot.get("chains_by_parent", {})).items():
            data = dict(raw)
            self.chains_by_parent[str(parent_id)] = ManagedOrderChain(
                intent_id=str(data.get("intent_id", parent_id)),
                parent_order_id=str(data.get("parent_order_id", parent_id)),
                stop_order_id=data.get("stop_order_id"),
                target_order_id=data.get("target_order_id"),
                symbol=str(data.get("symbol", "")),
                created_at=parse_dt(data.get("created_at")),
                last_update=parse_dt(data.get("last_update")),
                filled_qty=int(data.get("filled_qty", 0)),
                closed_qty=int(data.get("closed_qty", 0)),
            )
        self.intent_to_parent = {
            str(intent_id): str(parent_id)
            for intent_id, parent_id in dict(snapshot.get("intent_to_parent", {})).items()
        }
