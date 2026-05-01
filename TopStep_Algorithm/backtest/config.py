"""Backward-compatibility shim. Use trading_system.config directly."""
from trading_system.config import *
from trading_system.config import (
    SessionWindow, default_session_windows, SessionConfig, RiskLimits,
    TopstepConnectionConfig, ExecutionConfig, StrategyConfig, TraderConfig,
    InstrumentSpec, RiskConfig, INSTRUMENTS, RISK,
)
