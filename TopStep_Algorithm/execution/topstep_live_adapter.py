from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections import defaultdict, deque
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol
from urllib import error, request

import aiohttp

_log = logging.getLogger(__name__)

from backtest.config import TopstepConnectionConfig
from models.instruments import infer_symbol_root
from models.orders import (
    BrokerOrder,
    ExecutionReport,
    OrderState,
    OrderType,
    PositionSnapshot,
    Side,
    TimeInForce,
    TradingMode,
)

_STATUS_MAP = {
    0: OrderState.PENDING,
    1: OrderState.ACKNOWLEDGED,
    2: OrderState.FILLED,
    3: OrderState.CANCELED,
    4: OrderState.CANCELED,
    5: OrderState.REJECTED,
    6: OrderState.PENDING,
}

_TYPE_MAP = {
    0: OrderType.MARKET,
    1: OrderType.LIMIT,
    2: OrderType.MARKET,
    3: OrderType.STOP,
    4: OrderType.STOP_MARKET,
    5: OrderType.STOP_MARKET,
    6: OrderType.LIMIT,
    7: OrderType.LIMIT,
}

_ORDER_TYPE_TO_REMOTE = {
    OrderType.LIMIT: 1,
    OrderType.MARKET: 2,
    OrderType.STOP: 3,
    OrderType.STOP_MARKET: 4,
}

_SIDE_MAP = {
    0: Side.BUY,
    1: Side.SELL,
}

_SIDE_TO_REMOTE = {
    Side.BUY: 0,
    Side.SELL: 1,
}

_POSITION_SIGN = {
    1: 1,
    2: -1,
}

_SYMBOL_ALIASES = {
    "EP": "ES",
    "ENQ": "NQ",
    "MES": "MES",
    "MNQ": "MNQ",
    "6B": "6B",
    "6E": "6E",
}
_LIVE_BLOCKED_SYMBOLS = frozenset({"6B", "6E"})


class TopstepHttpTransport(Protocol):
    def post(
        self,
        base_url: str,
        path: str,
        payload: dict[str, Any],
        *,
        bearer_token: str | None = None,
        timeout_seconds: int = 10,
    ) -> dict[str, Any]: ...


class TopstepRealtimeClient(Protocol):
    def start(self, websocket_url: str, access_token: str, account_id: int, event_callback) -> None: ...

    def stop(self) -> None: ...

    def is_running(self) -> bool: ...


class UrlLibTopstepTransport:
    def post(
        self,
        base_url: str,
        path: str,
        payload: dict[str, Any],
        *,
        bearer_token: str | None = None,
        timeout_seconds: int = 10,
    ) -> dict[str, Any]:
        raw = json.dumps(payload).encode("utf-8")
        endpoint = f"{base_url.rstrip('/')}{path}"
        headers = {
            "accept": "text/plain",
            "content-type": "application/json",
        }
        if bearer_token:
            headers["authorization"] = f"Bearer {bearer_token}"
        # Log request (redact token from output)
        _log.debug("topstep_request path=%s payload=%s", path, json.dumps(payload))
        req = request.Request(endpoint, data=raw, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            _log.warning("topstep_http_error path=%s status=%s body=%s", path, exc.code, detail)
            raise RuntimeError(f"http_error:{exc.code}:{detail}") from exc
        except error.URLError as exc:
            _log.warning("topstep_transport_error path=%s reason=%s", path, exc.reason)
            raise RuntimeError(f"transport_error:{exc.reason}") from exc
        if not body.strip():
            return {}
        parsed = json.loads(body)
        _log.debug("topstep_response path=%s body=%s", path, body[:500])
        return parsed


class AiohttpSignalRRealtimeClient:
    _RS = "\x1e"

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._running = False
        self._websocket_url = ""
        self._access_token = ""
        self._account_id = 0
        self._event_callback = None

    def start(self, websocket_url: str, access_token: str, account_id: int, event_callback) -> None:
        self.stop()
        self._websocket_url = websocket_url
        self._access_token = access_token
        self._account_id = account_id
        self._event_callback = event_callback
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="topstep-signalr", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        self._running = False

    def is_running(self) -> bool:
        return self._running and not self._stop_event.is_set()

    def _run(self) -> None:
        asyncio.run(self._runner())

    async def _runner(self) -> None:
        _BACKOFF = (1.0, 2.0, 4.0, 8.0, 30.0)
        attempt = 0
        while not self._stop_event.is_set():
            try:
                await self._connect_once()
                attempt = 0  # reset on clean exit
            except Exception as exc:
                self._running = False
                _log.warning("topstep_ws_disconnected attempt=%d reason=%s", attempt, exc)
            if not self._stop_event.is_set():
                delay = _BACKOFF[min(attempt, len(_BACKOFF) - 1)]
                _log.info("topstep_ws_reconnect_in seconds=%.1f", delay)
                await asyncio.sleep(delay)
                attempt += 1

    async def _connect_once(self) -> None:
        headers = {"Authorization": f"Bearer {self._access_token}"}
        ws_url = f"{self._websocket_url}?access_token={self._access_token}"
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=None)
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.ws_connect(ws_url, heartbeat=20) as ws:
                self._running = True
                await ws.send_str(self._frame({"protocol": "json", "version": 1}))
                await ws.receive()
                await self._subscribe(ws)
                async for msg in ws:
                    if self._stop_event.is_set():
                        break
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        for payload in self._parse_frames(msg.data):
                            await self._handle_message(ws, payload)
                    elif msg.type in {aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR}:
                        break
        self._running = False

    async def _subscribe(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        await ws.send_str(self._frame({"type": 1, "target": "SubscribeAccounts", "arguments": []}))
        await ws.send_str(self._frame({"type": 1, "target": "SubscribeOrders", "arguments": [self._account_id]}))
        await ws.send_str(self._frame({"type": 1, "target": "SubscribePositions", "arguments": [self._account_id]}))
        await ws.send_str(self._frame({"type": 1, "target": "SubscribeTrades", "arguments": [self._account_id]}))

    async def _handle_message(self, ws: aiohttp.ClientWebSocketResponse, payload: dict[str, Any]) -> None:
        message_type = int(payload.get("type", 0) or 0)
        if message_type == 1:
            target = str(payload.get("target") or "")
            arguments = list(payload.get("arguments") or [])
            event_payload = arguments[0] if len(arguments) == 1 else arguments
            if self._event_callback is not None:
                try:
                    self._event_callback(target, event_payload)
                except Exception as exc:
                    _log.warning("topstep_ws_event_ignored target=%s reason=%s payload=%s", target, exc, event_payload)
        elif message_type == 6:
            await ws.send_str(self._frame({"type": 6}))
        elif message_type == 7:
            self._running = False

    @classmethod
    def _frame(cls, payload: dict[str, Any]) -> str:
        return json.dumps(payload, separators=(",", ":")) + cls._RS

    @classmethod
    def _parse_frames(cls, data: str) -> list[dict[str, Any]]:
        frames: list[dict[str, Any]] = []
        for frame in data.split(cls._RS):
            if not frame.strip():
                continue
            frames.append(json.loads(frame))
        return frames


class LiveTopstepAdapter:
    """
    REST-backed ProjectX/Topstep adapter.

    The documented realtime path uses SignalR, but this initial implementation
    refreshes broker state via REST polling so the execution engine can connect,
    reconcile, place orders, and observe state transitions without requiring an
    additional websocket dependency.
    """

    def __init__(
        self,
        config: TopstepConnectionConfig,
        mode: TradingMode,
        transport: TopstepHttpTransport | None = None,
        realtime_client: TopstepRealtimeClient | None = None,
    ) -> None:
        if mode == TradingMode.MOCK:
            raise ValueError("LiveTopstepAdapter requires paper or live mode.")
        self.config = config
        self.mode = mode
        self.connected = False
        self.orders: dict[str, BrokerOrder] = {}
        self.positions: dict[str, PositionSnapshot] = {}
        self.account_state = {
            "cash_balance": 0.0,
            "net_liquidation": 0.0,
            "cushion_to_max_loss_limit": 0.0,
        }
        self.event_queue: deque[ExecutionReport] = deque()
        self.callbacks: defaultdict[str, list] = defaultdict(list)
        self.market_prices: dict[str, float] = {}
        self.transport = transport or UrlLibTopstepTransport()
        self.realtime_client = realtime_client or AiohttpSignalRRealtimeClient()
        self.access_token: str | None = None
        self.token_expires_at: datetime | None = None
        self.account_numeric_id: int | None = None
        self.account_simulated: bool = True  # set from API response in _load_account_snapshot
        self.contract_cache_by_symbol: dict[str, dict[str, Any]] = {}
        self.contract_cache_by_id: dict[str, dict[str, Any]] = {}
        self.last_order_sync_at: datetime | None = None

    def connect(self) -> None:
        missing = self.config.missing_required_fields()
        if missing:
            raise RuntimeError(f"topstep_config_incomplete:{','.join(missing)}")
        self._authenticate()
        self._load_account_snapshot()
        self._start_event_stream()
        self.connected = True

    def is_connected(self) -> bool:
        return self.connected

    def reconnect(self) -> bool:
        try:
            self.disconnect()
            self.connect()
        except RuntimeError:
            self.connected = False
            return False
        return True

    def disconnect(self) -> None:
        self.realtime_client.stop()
        self.connected = False

    def get_account(self) -> dict[str, float]:
        return dict(self.account_state)

    def refresh_account_state(self) -> None:
        """Re-fetch account balance and drawdown cushion from the broker API."""
        try:
            accounts_response = self._authorized_post("/api/Account/search", {"onlyActiveAccounts": True})
            accounts = list(accounts_response.get("accounts") or [])
            account = self._select_account(accounts)
            if account is None:
                return
            balance = float(account.get("balance") or 0.0)
            cushion = float(account.get("maximumDrawdown") or 0.0)
            self.account_state = {
                "cash_balance": balance,
                "net_liquidation": balance,
                "cushion_to_max_loss_limit": cushion,
            }
            _log.info(
                "account_state refreshed balance=%.2f cushion_to_max_loss_limit=%.2f",
                balance,
                cushion,
            )
        except Exception as exc:
            _log.warning("account_state_refresh_failed reason=%s", exc)

    def get_market_price(self, symbol: str) -> float | None:
        return self.market_prices.get(symbol)

    def set_market_price(self, symbol: str, price: float) -> None:
        self.market_prices[symbol] = price

    def get_positions(self) -> dict[str, PositionSnapshot]:
        if self.connected and not self.realtime_client.is_running():
            self._refresh_positions()
        return {symbol: replace(position) for symbol, position in self.positions.items()}

    def get_open_orders(self) -> list[BrokerOrder]:
        if self.connected and not self.realtime_client.is_running():
            self._refresh_open_orders()
        return [
            replace(order)
            for order in self.orders.values()
            if order.state in {OrderState.PENDING, OrderState.ACKNOWLEDGED, OrderState.PARTIAL}
        ]

    def place_order(self, order_req: BrokerOrder) -> BrokerOrder:
        self._require_connected()
        symbol_root = infer_symbol_root(order_req.symbol)
        if self.mode == TradingMode.LIVE and symbol_root in _LIVE_BLOCKED_SYMBOLS:
            raise RuntimeError("6B/6E live execution is not verified; use paper/mock only.")
        contract = self._resolve_contract(order_req.symbol)
        payload = {
            "accountId": self._require_account_id(),
            "contractId": contract["id"],
            "type": _ORDER_TYPE_TO_REMOTE[order_req.order_type],
            "side": _SIDE_TO_REMOTE[order_req.side],
            "size": order_req.qty,
            "customTag": order_req.order_id,
        }
        if order_req.order_type == OrderType.LIMIT and order_req.price is not None:
            payload["limitPrice"] = order_req.price
        if order_req.order_type in {OrderType.STOP, OrderType.STOP_MARKET} and order_req.stop_price is not None:
            payload["stopPrice"] = order_req.stop_price
        response = self._authorized_post("/api/Order/place", payload)
        remote_order_id = self._extract_order_id(response)
        submitted = replace(
            order_req,
            broker_order_id=str(remote_order_id),
            submitted_at=self._now(),
            updated_at=self._now(),
            state=OrderState.PENDING,
        )
        self.orders[submitted.order_id] = submitted
        self._refresh_orders(force_full=True)
        return self.orders.get(submitted.order_id, submitted)

    def cancel_order(self, order_id: str) -> BrokerOrder:
        self._require_connected()
        order = self.orders[order_id]
        broker_order_id = self._require_broker_order_id(order)
        self._authorized_post(
            "/api/Order/cancel",
            {
                "accountId": self._require_account_id(),
                "orderId": broker_order_id,
            },
        )
        self._refresh_orders(force_full=True)
        return self.orders.get(order_id, replace(order, state=OrderState.CANCELED, updated_at=self._now()))

    def replace_order(self, order_id: str, **changes: object) -> BrokerOrder:
        self._require_connected()
        existing = self.orders[order_id]
        updated = replace(existing, updated_at=self._now(), **changes)
        payload = {
            "accountId": self._require_account_id(),
            "id": self._require_broker_order_id(existing),
            "size": updated.qty,
            "customTag": updated.order_id,
        }
        if updated.price is not None:
            payload["limitPrice"] = updated.price
        if updated.stop_price is not None:
            payload["stopPrice"] = updated.stop_price
        self._authorized_post("/api/Order/modify", payload)
        self.orders[order_id] = updated
        self._refresh_orders(force_full=True)
        return self.orders.get(order_id, updated)

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
        raise NotImplementedError("emit_fill is mock-only; live fills must arrive from broker state updates.")

    def poll_events(self) -> list[ExecutionReport]:
        if self.connected and not self.realtime_client.is_running():
            self._refresh_orders()
            self._refresh_positions()
        events = list(self.event_queue)
        self.event_queue.clear()
        return events

    def _publish(self, report: ExecutionReport) -> None:
        self.event_queue.append(report)
        for callback in self.callbacks["execution_report"]:
            callback(report)

    def _require_connected(self) -> None:
        if not self.connected:
            raise RuntimeError("adapter_not_connected")

    @staticmethod
    def _now() -> datetime:
        return datetime.now(UTC)

    def _authenticate(self) -> None:
        response = self.transport.post(
            self.config.api_base_url,
            "/api/Auth/loginKey",
            {
                "userName": self.config.username,
                "apiKey": self.config.api_key,
            },
            timeout_seconds=self.config.request_timeout_seconds,
        )
        token = response.get("token") or response.get("jwtToken")
        if not token:
            raise RuntimeError(f"topstep_auth_failed:{response.get('errorMessage') or 'missing_token'}")
        self.access_token = str(token)
        self.token_expires_at = self._now() + timedelta(hours=24)

    def _load_account_snapshot(self) -> None:
        accounts_response = self._authorized_post("/api/Account/search", {"onlyActiveAccounts": True})
        accounts = list(accounts_response.get("accounts") or [])
        account = self._select_account(accounts)
        if account is None:
            raise RuntimeError(f"topstep_account_not_found:{self.config.account_id}")
        self.account_numeric_id = int(account["id"])
        self.account_simulated = bool(account.get("simulated", True))
        balance = float(account.get("balance") or 0.0)
        self.account_state = {
            "cash_balance": balance,
            "net_liquidation": balance,
            "cushion_to_max_loss_limit": float(account.get("maximumDrawdown") or 0.0),
        }
        _log.info(
            "account_snapshot loaded account_id=%s balance=%.2f cushion_to_max_loss_limit=%.2f simulated=%s",
            self.account_numeric_id,
            balance,
            self.account_state["cushion_to_max_loss_limit"],
            self.account_simulated,
        )
        if account.get("canTrade") is False:
            raise RuntimeError(f"topstep_account_cannot_trade:{self.account_numeric_id}")
        self._refresh_positions()
        self._refresh_open_orders()
        self.last_order_sync_at = self._now()

    def _start_event_stream(self) -> None:
        self._refresh_orders(force_full=True)
        self.realtime_client.start(
            self.config.websocket_url,
            self.access_token or "",
            self._require_account_id(),
            self._handle_realtime_event,
        )

    def _require_account_id(self) -> int:
        if self.account_numeric_id is None:
            raise RuntimeError("topstep_account_uninitialized")
        return self.account_numeric_id

    def _authorized_post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self._refresh_token_if_needed()
        try:
            response = self.transport.post(
                self.config.api_base_url,
                path,
                payload,
                bearer_token=self.access_token,
                timeout_seconds=self.config.request_timeout_seconds,
            )
        except RuntimeError as exc:
            if "http_error:401" not in str(exc):
                raise
            self._authenticate()
            response = self.transport.post(
                self.config.api_base_url,
                path,
                payload,
                bearer_token=self.access_token,
                timeout_seconds=self.config.request_timeout_seconds,
            )
        if response.get("success") is False:
            raise RuntimeError(
                f"topstep_api_error:{path}:{response.get('errorCode')}:{response.get('errorMessage') or 'unknown'}"
            )
        return response

    def _refresh_token_if_needed(self) -> None:
        if self.access_token is None or self.token_expires_at is None:
            self._authenticate()
            return
        refresh_at = self.token_expires_at - timedelta(seconds=self.config.token_refresh_margin_seconds)
        if self._now() >= refresh_at:
            self._authenticate()
            if self.connected and self.realtime_client.is_running():
                self.realtime_client.start(
                    self.config.websocket_url,
                    self.access_token or "",
                    self._require_account_id(),
                    self._handle_realtime_event,
                )

    def _select_account(self, accounts: list[dict[str, Any]]) -> dict[str, Any] | None:
        desired = str(self.config.account_id).strip()
        for account in accounts:
            if str(account.get("id")) == desired:
                return account
        for account in accounts:
            if str(account.get("name") or "").strip() == desired:
                return account
        return None

    def _resolve_contract(self, symbol: str) -> dict[str, Any]:
        symbol_root = infer_symbol_root(symbol)
        if symbol_root in self.contract_cache_by_symbol:
            return self.contract_cache_by_symbol[symbol_root]
        response = self._authorized_post(
            "/api/Contract/search",
            {
                "live": not self.account_simulated,
                "searchText": symbol_root,
            },
        )
        contracts = list(response.get("contracts") or [])
        for contract in contracts:
            if not contract.get("activeContract", True):
                continue
            normalized = self._normalize_symbol_root(contract)
            if normalized == symbol_root:
                self.contract_cache_by_symbol[symbol_root] = contract
                self.contract_cache_by_id[str(contract["id"])] = contract
                return contract
        if contracts:
            contract = contracts[0]
            self.contract_cache_by_symbol[symbol_root] = contract
            self.contract_cache_by_id[str(contract["id"])] = contract
            return contract
        raise RuntimeError(f"topstep_contract_not_found:{symbol_root}")

    def _refresh_positions(self) -> None:
        response = self._authorized_post(
            "/api/Position/searchOpen",
            {"accountId": self._require_account_id()},
        )
        remote_positions: dict[str, PositionSnapshot] = {}
        for item in list(response.get("positions") or []):
            contract = self._resolve_contract_by_id(str(item["contractId"]))
            symbol = self._normalize_symbol_root(contract)
            sign = _POSITION_SIGN.get(int(item.get("type") or 0), 0)
            qty = int(item.get("size") or 0) * sign
            remote_positions[symbol] = PositionSnapshot(
                symbol=symbol,
                qty=qty,
                avg_price=float(item.get("averagePrice")) if item.get("averagePrice") is not None else None,
                last_updated=self._parse_timestamp(item.get("creationTimestamp")) or self._now(),
            )
        self.positions = remote_positions

    def _refresh_open_orders(self) -> None:
        response = self._authorized_post(
            "/api/Order/searchOpen",
            {"accountId": self._require_account_id()},
        )
        for item in list(response.get("orders") or []):
            mapped = self._map_remote_order(item)
            self.orders[mapped.order_id] = mapped

    def _refresh_orders(self, *, force_full: bool = False) -> None:
        window_end = self._now()
        if force_full or self.last_order_sync_at is None:
            window_start = window_end - timedelta(minutes=30)
        else:
            window_start = self.last_order_sync_at - timedelta(seconds=5)
        response = self._authorized_post(
            "/api/Order/search",
            {
                "accountId": self._require_account_id(),
                "startTimestamp": window_start.isoformat(),
                "endTimestamp": window_end.isoformat(),
            },
        )
        changed_orders = list(response.get("orders") or [])
        open_response = self._authorized_post(
            "/api/Order/searchOpen",
            {"accountId": self._require_account_id()},
        )
        open_by_broker_id = {
            str(item["id"]): item
            for item in list(open_response.get("orders") or [])
        }
        for item in changed_orders:
            merged = dict(item)
            merged.update(open_by_broker_id.get(str(item["id"]), {}))
            mapped = self._map_remote_order(merged)
            previous = self.orders.get(mapped.order_id)
            self.orders[mapped.order_id] = mapped
            if previous is None:
                continue
            report = self._build_execution_report(previous, mapped)
            if report is not None:
                self._publish(report)
        self.last_order_sync_at = window_end

    def _handle_realtime_event(self, target: str, payload: Any) -> None:
        payload = self._unwrap_realtime_payload(payload)
        if target == "GatewayUserOrder" and isinstance(payload, dict):
            self._apply_realtime_order(payload)
            return
        if target == "GatewayUserPosition" and isinstance(payload, dict):
            self._apply_realtime_position(payload)
            return
        if target == "GatewayUserAccount" and isinstance(payload, dict):
            self._apply_realtime_account(payload)
            return
        if target == "GatewayUserTrade" and isinstance(payload, dict):
            self._apply_realtime_trade(payload)

    @staticmethod
    def _unwrap_realtime_payload(payload: Any) -> Any:
        if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
            return payload["data"]
        return payload

    def _apply_realtime_order(self, payload: dict[str, Any]) -> None:
        mapped = self._map_remote_order(payload)
        previous = self.orders.get(mapped.order_id)
        self.orders[mapped.order_id] = mapped
        if previous is None:
            _log.info(
                "topstep_realtime_order_learned order_id=%s broker_order_id=%s status=%s",
                mapped.order_id,
                mapped.broker_order_id,
                mapped.state.value,
            )
            return
        report = self._build_execution_report(previous, mapped)
        if report is not None:
            self._publish(report)

    def _apply_realtime_position(self, payload: dict[str, Any]) -> None:
        contract_id = payload.get("contractId")
        if contract_id is None:
            _log.warning("topstep_position_event_ignored reason=missing_contractId payload=%s", payload)
            return
        contract = self._resolve_contract_by_id(str(contract_id))
        symbol = self._normalize_symbol_root(contract)
        sign = _POSITION_SIGN.get(int(payload.get("type") or 0), 0)
        qty = int(payload.get("size") or 0) * sign
        if qty == 0:
            self.positions.pop(symbol, None)
            return
        self.positions[symbol] = PositionSnapshot(
            symbol=symbol,
            qty=qty,
            avg_price=float(payload.get("averagePrice")) if payload.get("averagePrice") is not None else None,
            last_updated=self._parse_timestamp(payload.get("creationTimestamp")) or self._now(),
        )

    def _apply_realtime_account(self, payload: dict[str, Any]) -> None:
        balance = float(payload.get("balance") or self.account_state["cash_balance"])
        max_drawdown = payload.get("maximumDrawdown")
        cushion = (
            float(max_drawdown)
            if max_drawdown is not None
            else self.account_state.get("cushion_to_max_loss_limit", 0.0)
        )
        self.account_state = {
            "cash_balance": balance,
            "net_liquidation": balance,
            "cushion_to_max_loss_limit": cushion,
        }

    def _apply_realtime_trade(self, payload: dict[str, Any]) -> None:
        symbol = "UNKNOWN"
        contract_id = payload.get("contractId")
        if contract_id is not None:
            contract = self._resolve_contract_by_id(str(contract_id))
            symbol = self._normalize_symbol_root(contract)
        self.market_prices[symbol] = float(payload.get("price") or self.market_prices.get(symbol, 0.0))

    def _build_execution_report(
        self,
        previous: BrokerOrder | None,
        current: BrokerOrder,
    ) -> ExecutionReport | None:
        if previous is None:
            status_changed = True
            filled_delta = current.filled_qty
        else:
            status_changed = previous.state != current.state
            filled_delta = max(current.filled_qty - previous.filled_qty, 0)
        if not status_changed and filled_delta == 0:
            return None
        return ExecutionReport(
            order_id=current.order_id,
            broker_order_id=current.broker_order_id,
            symbol=current.symbol,
            status=current.state,
            fill_qty=filled_delta,
            fill_price=current.avg_fill_price,
            remaining_qty=max(current.qty - current.filled_qty, 0),
            side=current.side,
            timestamp=current.updated_at or self._now(),
            message=current.reason or "",
        )

    def _map_remote_order(self, item: dict[str, Any]) -> BrokerOrder:
        contract = self._resolve_contract_by_id(str(item["contractId"]))
        symbol = self._normalize_symbol_root(contract)
        custom_tag = item.get("customTag")
        remote_order_id = str(item["id"])
        order_id = str(custom_tag or f"broker-{remote_order_id}")
        previous = self.orders.get(order_id)
        qty = int(item.get("size") or previous.qty if previous else item.get("size") or 0)
        filled_qty = int(item.get("fillVolume") or previous.filled_qty if previous else item.get("fillVolume") or 0)
        return BrokerOrder(
            order_id=order_id,
            symbol=symbol,
            side=_SIDE_MAP.get(int(item.get("side") or 0), Side.BUY),
            qty=qty,
            order_type=_TYPE_MAP.get(int(item.get("type") or 0), OrderType.MARKET),
            tif=previous.tif if previous else TimeInForce.DAY,
            state=_STATUS_MAP.get(int(item.get("status") or 0), OrderState.PENDING),
            price=float(item["limitPrice"]) if item.get("limitPrice") is not None else None,
            stop_price=float(item["stopPrice"]) if item.get("stopPrice") is not None else None,
            filled_qty=filled_qty,
            avg_fill_price=float(item["filledPrice"]) if item.get("filledPrice") is not None else None,
            parent_order_id=previous.parent_order_id if previous else None,
            role=previous.role if previous else "entry",
            broker_order_id=remote_order_id,
            submitted_at=self._parse_timestamp(item.get("creationTimestamp")),
            updated_at=self._parse_timestamp(item.get("updateTimestamp")) or self._now(),
            intent_id=previous.intent_id if previous else None,
            reason=str(item.get("errorMessage") or previous.reason if previous else ""),
        )

    def _resolve_contract_by_id(self, contract_id: str) -> dict[str, Any]:
        if contract_id in self.contract_cache_by_id:
            return self.contract_cache_by_id[contract_id]
        response = self._authorized_post("/api/Contract/searchById", {"contractId": contract_id})
        contracts = list(response.get("contracts") or [])
        if not contracts:
            contract_parts = contract_id.split(".")
            symbol_root = contract_parts[3] if len(contract_parts) >= 4 else infer_symbol_root(contract_id)
            try:
                contract = self._resolve_contract(symbol_root)
            except RuntimeError:
                contract = {"id": contract_id, "name": symbol_root, "activeContract": True}
            self.contract_cache_by_id[contract_id] = contract
            return contract
        contract = contracts[0]
        self.contract_cache_by_id[contract_id] = contract
        return contract

    def _normalize_symbol_root(self, contract: dict[str, Any]) -> str:
        name_root = infer_symbol_root(str(contract.get("name") or ""))
        if name_root in _SYMBOL_ALIASES:
            return _SYMBOL_ALIASES[name_root]
        contract_id = str(contract.get("id") or "")
        contract_parts = contract_id.split(".")
        if len(contract_parts) >= 4:
            symbol_part = contract_parts[3]
            symbol_root = infer_symbol_root(symbol_part)
            if symbol_root in _SYMBOL_ALIASES:
                return _SYMBOL_ALIASES[symbol_root]
            return symbol_root
        return name_root or "UNKNOWN"

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime | None:
        if not value:
            return None
        if isinstance(value, datetime):
            return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        if isinstance(value, str):
            normalized = value.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(normalized)
            return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
        return None

    @staticmethod
    def _extract_order_id(response: dict[str, Any]) -> int:
        for key in ("orderId", "id"):
            if key in response and response[key] is not None:
                return int(response[key])
        order = response.get("order")
        if isinstance(order, dict) and order.get("id") is not None:
            return int(order["id"])
        raise RuntimeError("topstep_order_place_missing_id")

    @staticmethod
    def _require_broker_order_id(order: BrokerOrder) -> int:
        if order.broker_order_id is None:
            raise RuntimeError(f"missing_broker_order_id:{order.order_id}")
        return int(order.broker_order_id)
