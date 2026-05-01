"""Backward-compatibility shim."""
import sys
import trading_system.data_pipeline.sweep_live_feed as _mod
sys.modules[__name__] = _mod
