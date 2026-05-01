"""Backward-compatibility shim. Import from trading_system.profiles directly."""
from __future__ import annotations
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
    "PROFILE_TOPSTEP_50K_EXPRESS", "PROFILE_TOPSTEP_50K_EXPRESS_LONDON",
    "PROFILE_TOPSTEP_50K_EXPRESS_LONDON_6B_PAPER",
    "PROFILE_TOPSTEP_50K_EXPRESS_LONDON_6E_PAPER",
    "apply_profile", "available_profiles", "build_config",
]
