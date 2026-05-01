"""Backward-compatibility shim."""
import sys
import trading_system.backtest.holidays as _mod
sys.modules[__name__] = _mod
