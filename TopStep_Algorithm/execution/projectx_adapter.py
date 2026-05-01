"""Backward-compatibility shim."""
import sys
import trading_system.execution.projectx_adapter as _mod
sys.modules[__name__] = _mod
