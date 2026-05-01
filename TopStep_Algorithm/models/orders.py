"""Backward-compatibility shim."""
import sys
import trading_system.core.domain as _mod
sys.modules[__name__] = _mod
