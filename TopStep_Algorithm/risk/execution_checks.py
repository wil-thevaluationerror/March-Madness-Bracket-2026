"""Backward-compatibility shim."""
import sys
import trading_system.risk.execution_checks as _mod
sys.modules[__name__] = _mod
