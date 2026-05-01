"""Backward-compatibility shim. Import from trading_system.config directly."""
from __future__ import annotations
from trading_system.config import (
    ExecutionConfig,
    InstrumentSpec,
    INSTRUMENTS,
    RiskConfig,
    RISK,
    RiskLimits,
    SessionConfig,
    SessionWindow,
    StrategyConfig,
    TopstepConnectionConfig,
    TraderConfig,
    default_session_windows,
)
from trading_system.profiles import (
    PROFILE_TOPSTEP_50K_EXPRESS,
    PROFILE_TOPSTEP_50K_EXPRESS_LONDON,
    PROFILE_TOPSTEP_50K_EXPRESS_LONDON_6B_PAPER,
    PROFILE_TOPSTEP_50K_EXPRESS_LONDON_6E_PAPER,
    apply_profile,
    available_profiles,
    build_config,
)
__all__ = [
    "ExecutionConfig", "InstrumentSpec", "INSTRUMENTS", "RiskConfig", "RISK",
    "RiskLimits", "SessionConfig", "SessionWindow",
    "StrategyConfig", "TopstepConnectionConfig", "TraderConfig",
    "default_session_windows",
    "PROFILE_TOPSTEP_50K_EXPRESS", "PROFILE_TOPSTEP_50K_EXPRESS_LONDON",
    "PROFILE_TOPSTEP_50K_EXPRESS_LONDON_6B_PAPER",
    "PROFILE_TOPSTEP_50K_EXPRESS_LONDON_6E_PAPER",
    "apply_profile", "available_profiles", "build_config",
]
