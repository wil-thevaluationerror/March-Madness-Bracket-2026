"""Backward-compatibility shim."""
import sys
import trading_system.execution.order_manager as _mod
sys.modules[__name__] = _mod
