"""Backward-compatibility shim."""
import sys
import trading_system.data_pipeline.preprocess as _mod
sys.modules[__name__] = _mod
