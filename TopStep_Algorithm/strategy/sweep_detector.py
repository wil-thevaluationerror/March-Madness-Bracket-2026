"""Backward-compatibility shim."""
import sys
import trading_system.strategy.sweep_detector as _mod
sys.modules[__name__] = _mod
