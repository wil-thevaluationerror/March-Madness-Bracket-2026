from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from trading_system.core.domain import BrokerOrder, ExecutionReport, PositionSnapshot


@runtime_checkable
class BrokerAdapter(Protocol):
    mode: object
    connected: bool
    orders: dict[str, BrokerOrder]

    def connect(self) -> None: ...

    def is_connected(self) -> bool: ...

    def reconnect(self) -> bool: ...

    def disconnect(self) -> None: ...

    def get_account(self) -> dict[str, float]: ...

    def get_market_price(self, symbol: str) -> float | None: ...

    def set_market_price(self, symbol: str, price: float) -> None: ...

    def get_positions(self) -> dict[str, PositionSnapshot]: ...

    def get_open_orders(self) -> list[BrokerOrder]: ...

    def place_order(self, order_req: BrokerOrder) -> BrokerOrder: ...

    def cancel_order(self, order_id: str) -> BrokerOrder: ...

    def replace_order(self, order_id: str, **changes: object) -> BrokerOrder: ...

    def subscribe_events(self, event_type: str, callback) -> None: ...

    def emit_fill(
        self,
        order_id: str,
        fill_qty: int,
        fill_price: float,
        *,
        slippage: float = 0.0,
        fees: float = 0.0,
        timestamp: datetime | None = None,
    ) -> ExecutionReport: ...

    def poll_events(self) -> list[ExecutionReport]: ...
