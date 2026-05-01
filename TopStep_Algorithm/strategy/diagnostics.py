"""Backward-compatibility shim."""
import sys
import trading_system.strategy.diagnostics as _mod
sys.modules[__name__] = _mod
