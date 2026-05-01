"""Backward-compatibility shim."""
import sys
import trading_system.api.market_data as _mod
sys.modules[__name__] = _mod
