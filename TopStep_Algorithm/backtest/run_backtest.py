"""Backward-compatibility shim."""
import sys
import trading_system.backtest.run_backtest as _mod
sys.modules[__name__] = _mod
