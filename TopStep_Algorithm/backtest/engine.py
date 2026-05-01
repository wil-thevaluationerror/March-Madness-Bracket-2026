"""Backward-compatibility shim."""
import sys
import trading_system.backtest.engine as _mod
sys.modules[__name__] = _mod
