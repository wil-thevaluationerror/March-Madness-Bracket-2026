from __future__ import annotations

from dataclasses import dataclass
from datetime import time
import re


_CONTRACT_MONTH_CODES = frozenset("FGHJKMNQUVXZ")
_ROOT_WITH_CONTRACT_RE = re.compile(r"^(?P<root>[A-Z]+?)(?P<month>[FGHJKMNQUVXZ])(?P<year>\d{1,2})$")


@dataclass(frozen=True, slots=True)
class InstrumentSpec:
    root_symbol: str
    description: str
    tick_size: float
    tick_value: float
    session_timezone: str = "America/Chicago"
    market_open: time = time(hour=8, minute=30)
    no_new_trades_after: time = time(hour=14, minute=45)
    force_flatten_at: time = time(hour=15, minute=8)
    exchange_close: time = time(hour=15, minute=10)

    @property
    def point_value(self) -> float:
        return self.tick_value / self.tick_size

    def price_to_ticks(self, price_distance: float) -> float:
        return abs(price_distance) / self.tick_size

    def price_to_pnl(self, price_distance: float, qty: int) -> float:
        return price_distance * qty * self.point_value


_INSTRUMENT_SPECS: dict[str, InstrumentSpec] = {
    "MES": InstrumentSpec(
        root_symbol="MES",
        description="Micro E-mini S&P 500",
        tick_size=0.25,
        tick_value=1.25,
    ),
    "ES": InstrumentSpec(
        root_symbol="ES",
        description="E-mini S&P 500",
        tick_size=0.25,
        tick_value=12.5,
    ),
    "MNQ": InstrumentSpec(
        root_symbol="MNQ",
        description="Micro E-mini Nasdaq-100",
        tick_size=0.25,
        tick_value=0.5,
    ),
    "NQ": InstrumentSpec(
        root_symbol="NQ",
        description="E-mini Nasdaq-100",
        tick_size=0.25,
        tick_value=5.0,
    ),
}

_GENERIC_FUTURES_SPEC = InstrumentSpec(
    root_symbol="GENERIC",
    description="Generic futures contract",
    tick_size=0.25,
    tick_value=1.0,
)


def infer_symbol_root(symbol: str) -> str:
    cleaned = str(symbol).upper()
    cleaned = cleaned.split("-", 1)[0]
    cleaned = cleaned.split(".", 1)[0]
    match = _ROOT_WITH_CONTRACT_RE.match(cleaned)
    if match:
        return match.group("root")
    return cleaned


def resolve_instrument(symbol: str) -> InstrumentSpec:
    return _INSTRUMENT_SPECS.get(infer_symbol_root(symbol), _GENERIC_FUTURES_SPEC)


def known_instruments() -> dict[str, InstrumentSpec]:
    return dict(_INSTRUMENT_SPECS)
