"""Backward-compatibility shim."""
import sys
import trading_system.strategy.asian_range as _mod
sys.modules[__name__] = _mod
