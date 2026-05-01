"""Backward-compatibility shim."""
import sys
import trading_system.core.instruments as _mod
sys.modules[__name__] = _mod
