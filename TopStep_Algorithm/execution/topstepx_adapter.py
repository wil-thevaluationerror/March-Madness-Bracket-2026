"""Backward-compatibility shim."""
import sys
import trading_system.execution.topstepx_adapter as _mod
sys.modules[__name__] = _mod
