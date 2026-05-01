from __future__ import annotations

from trading_system.config import TopstepConnectionConfig
from trading_system.execution.projectx_adapter import ProjectXAdapter
from trading_system.execution.topstep_live_adapter import LiveTopstepAdapter, TopstepHttpTransport, TopstepRealtimeClient
from trading_system.core.domain import TradingMode


class TopstepXAdapter:
    """
    TopstepX-facing adapter.

    MOCK mode delegates to the in-memory simulator used by the test suite.
    PAPER/LIVE modes delegate to a real-adapter skeleton that defines the
    Topstep integration surface but does not implement the HTTP/stream
    transport yet.
    """

    def __init__(
        self,
        mode: TradingMode = TradingMode.MOCK,
        config: TopstepConnectionConfig | None = None,
        transport: TopstepHttpTransport | None = None,
        realtime_client: TopstepRealtimeClient | None = None,
    ) -> None:
        if mode == TradingMode.MOCK:
            self._impl = ProjectXAdapter(mode=mode)
        else:
            self._impl = LiveTopstepAdapter(
                config or TopstepConnectionConfig(),
                mode=mode,
                transport=transport,
                realtime_client=realtime_client,
            )

    @property
    def mode(self) -> TradingMode:
        return self._impl.mode

    @property
    def connected(self) -> bool:
        return self._impl.connected

    @property
    def orders(self):
        return self._impl.orders

    def connect(self) -> None:
        self._impl.connect()

    def is_connected(self) -> bool:
        return self._impl.is_connected()

    def reconnect(self) -> bool:
        return self._impl.reconnect()

    def disconnect(self) -> None:
        self._impl.disconnect()

    def get_account(self) -> dict[str, float]:
        return self._impl.get_account()

    def get_market_price(self, symbol: str) -> float | None:
        return self._impl.get_market_price(symbol)

    def set_market_price(self, symbol: str, price: float) -> None:
        self._impl.set_market_price(symbol, price)

    def get_positions(self):
        return self._impl.get_positions()

    def get_open_orders(self):
        return self._impl.get_open_orders()

    def place_order(self, order_req):
        return self._impl.place_order(order_req)

    def cancel_order(self, order_id: str):
        return self._impl.cancel_order(order_id)

    def replace_order(self, order_id: str, **changes: object):
        return self._impl.replace_order(order_id, **changes)

    def subscribe_events(self, event_type: str, callback) -> None:
        self._impl.subscribe_events(event_type, callback)

    def emit_fill(
        self,
        order_id: str,
        fill_qty: int,
        fill_price: float,
        *,
        slippage: float = 0.0,
        fees: float = 0.0,
        timestamp=None,
    ):
        return self._impl.emit_fill(
            order_id,
            fill_qty,
            fill_price,
            slippage=slippage,
            fees=fees,
            timestamp=timestamp,
        )

    def poll_events(self):
        return self._impl.poll_events()

    def __getattr__(self, name: str):
        return getattr(self._impl, name)
