"""Backward-compatibility shim."""
import sys
import trading_system.strategy.intent_bridge as _mod
sys.modules[__name__] = _mod
