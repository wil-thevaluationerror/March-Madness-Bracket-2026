"""Backward-compatibility shim."""
import sys
import trading_system.strategy.confluence as _mod
sys.modules[__name__] = _mod
