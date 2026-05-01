"""Backward-compatibility shim."""
import sys
import trading_system.execution.state_store as _mod
sys.modules[__name__] = _mod
