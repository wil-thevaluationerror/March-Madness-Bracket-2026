"""Backward-compatibility shim."""
import sys
import trading_system.backtest.raw_setup_ledger as _mod
sys.modules[__name__] = _mod
