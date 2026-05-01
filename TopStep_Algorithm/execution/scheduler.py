"""Backward-compatibility shim."""
import sys
import trading_system.execution.scheduler as _mod
sys.modules[__name__] = _mod
