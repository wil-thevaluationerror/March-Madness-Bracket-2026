"""Backward-compatibility shim."""
import sys
import trading_system.backtest.metrics as _mod
sys.modules[__name__] = _mod
