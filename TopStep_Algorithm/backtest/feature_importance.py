"""Backward-compatibility shim."""
import sys
import trading_system.backtest.feature_importance as _mod
sys.modules[__name__] = _mod
