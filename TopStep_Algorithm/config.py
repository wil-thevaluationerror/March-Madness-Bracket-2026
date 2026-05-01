from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Liquidity-sweep backtest instrument and risk configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str
    tick_size: float
    tick_value: float  # USD value of one tick, one contract
    name: str


@dataclass(frozen=True)
class RiskConfig:
    account_risk_pct: float = 0.005        # 0.5% per trade
    min_contracts: int = 1
    max_contracts: int = 10
    daily_loss_limit_usd: float = 1_000.0  # Topstep 50K Daily Loss Limit
    max_loss_limit_usd: float = 2_000.0    # Topstep 50K Express Max Loss Limit
    max_trades_per_session: int = 2


INSTRUMENTS: dict[str, InstrumentSpec] = {
    "6B": InstrumentSpec(
        symbol="6B",
        tick_size=0.0001,
        tick_value=6.25,
        name="British Pound Futures",
    ),
    "6E": InstrumentSpec(
        symbol="6E",
        tick_size=0.00005,
        tick_value=6.25,
        name="Euro FX Futures",
    ),
    "MES": InstrumentSpec(
        symbol="MES",
        tick_size=0.25,
        tick_value=1.25,
        name="Micro E-mini S&P 500",
    ),
    "ES": InstrumentSpec(
        symbol="ES",
        tick_size=0.25,
        tick_value=12.50,
        name="E-mini S&P 500",
    ),
}

RISK = RiskConfig(
    account_risk_pct=0.005,
    min_contracts=1,
    max_contracts=10,
    daily_loss_limit_usd=1_000.0,
    max_loss_limit_usd=2_000.0,
    max_trades_per_session=2,
)


# ---------------------------------------------------------------------------
# Live runtime re-exports (unchanged)
# ---------------------------------------------------------------------------

from backtest.config import (
    ExecutionConfig,
    RiskLimits,
    SessionConfig,
    SessionWindow,
    StrategyConfig,
    TopstepConnectionConfig,
    TraderConfig,
)
from profiles import (
    PROFILE_TOPSTEP_50K_EXPRESS,
    PROFILE_TOPSTEP_50K_EXPRESS_LONDON,
    PROFILE_TOPSTEP_50K_EXPRESS_LONDON_6B_PAPER,
    PROFILE_TOPSTEP_50K_EXPRESS_LONDON_6E_PAPER,
    apply_profile,
    available_profiles,
    build_config,
)

__all__ = [
    "ExecutionConfig",
    "RiskLimits",
    "SessionConfig",
    "SessionWindow",
    "StrategyConfig",
    "TopstepConnectionConfig",
    "TraderConfig",
    "PROFILE_TOPSTEP_50K_EXPRESS",
    "PROFILE_TOPSTEP_50K_EXPRESS_LONDON",
    "PROFILE_TOPSTEP_50K_EXPRESS_LONDON_6B_PAPER",
    "PROFILE_TOPSTEP_50K_EXPRESS_LONDON_6E_PAPER",
    "apply_profile",
    "available_profiles",
    "build_config",
]
