"""Backward-compatibility shim."""
import sys
import trading_system.backtest.eval_simulator as _mod
sys.modules[__name__] = _mod
