"""Backward-compatibility shim."""
import sys
import trading_system.backtest.data_loader as _mod
sys.modules[__name__] = _mod
