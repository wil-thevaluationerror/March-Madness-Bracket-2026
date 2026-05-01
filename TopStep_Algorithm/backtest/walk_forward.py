"""Backward-compatibility shim."""
import sys
import trading_system.backtest.walk_forward as _mod
sys.modules[__name__] = _mod
