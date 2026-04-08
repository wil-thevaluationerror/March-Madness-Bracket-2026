from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_MARKET = "STOP_MARKET"


class TimeInForce(str, Enum):
    DAY = "DAY"
    GTC = "GTC"
    IOC = "IOC"


class OrderState(str, Enum):
    PENDING = "PENDING"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"


class IntentStatus(str, Enum):
    CREATED = "CREATED"
    APPROVED = "APPROVED"
    DENIED = "DENIED"
    SUBMITTED = "SUBMITTED"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class TradingMode(str, Enum):
    MOCK = "mock"
    PAPER = "paper"
    LIVE = "live"


class Regime(str, Enum):
    TREND_EXPANSION = "TREND_EXPANSION"
    HIGH_VOL_BREAKOUT = "HIGH_VOL_BREAKOUT"
    LOW_VOL_COMPRESSION = "LOW_VOL_COMPRESSION"
    CHOP_MEAN_REVERT = "CHOP_MEAN_REVERT"
    UNKNOWN = "UNKNOWN"


@dataclass(slots=True)
class OrderIntent:
    intent_id: str
    symbol: str
    side: Side
    qty: int
    entry_type: OrderType
    entry_price: float | None
    stop_price: float
    target_price: float | None
    time_in_force: TimeInForce
    reason: str
    signal_ts: datetime
    signal_score: float
    regime: Regime
    strategy_id: str
    allow_scale_out: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BrokerOrder:
    order_id: str
    symbol: str
    side: Side
    qty: int
    order_type: OrderType
    tif: TimeInForce
    state: OrderState
    price: float | None = None
    stop_price: float | None = None
    filled_qty: int = 0
    avg_fill_price: float | None = None
    parent_order_id: str | None = None
    role: str = "entry"
    broker_order_id: str | None = None
    submitted_at: datetime | None = None
    updated_at: datetime | None = None
    intent_id: str | None = None
    reason: str | None = None


@dataclass(slots=True)
class ExecutionReport:
    order_id: str
    broker_order_id: str | None
    symbol: str
    status: OrderState
    fill_qty: int
    fill_price: float | None
    remaining_qty: int
    side: Side
    timestamp: datetime
    fees: float = 0.0
    slippage: float = 0.0
    message: str = ""


@dataclass(slots=True)
class PositionSnapshot:
    symbol: str
    qty: int = 0
    avg_price: float | None = None
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    stop_covered_qty: int = 0
    last_updated: datetime | None = None

    @property
    def is_flat(self) -> bool:
        return self.qty == 0


@dataclass(slots=True)
class KillSwitchState:
    armed: bool = False
    reason: str = ""
    activated_at: datetime | None = None
    requires_manual_reset: bool = False


@dataclass(slots=True)
class OrderPlan:
    entry: BrokerOrder
    stop: BrokerOrder | None
    target: BrokerOrder | None


@dataclass(slots=True)
class ExecutionDecision:
    approved: bool
    reason: str
    normalized_qty: int
    order_plan: OrderPlan | None = None
