"""Smoke tests that all key import paths resolve after Phase 1 restructure."""
from __future__ import annotations


def test_core_domain_imports():
    from trading_system.core.domain import (
        OrderIntent, BrokerOrder, Side, Regime, TradingMode,
        OrderState, ExecutionReport, PositionSnapshot, KillSwitchState,
    )
    assert OrderIntent is not None


def test_core_instruments_imports():
    from trading_system.core.instruments import resolve_instrument, infer_symbol_root
    assert resolve_instrument is not None


def test_config_imports():
    from trading_system.config import (
        TraderConfig, SessionConfig, RiskLimits, ExecutionConfig,
        StrategyConfig, TopstepConnectionConfig, SessionWindow,
    )
    assert TraderConfig is not None


def test_profiles_imports():
    from trading_system.profiles import (
        build_config, available_profiles,
        PROFILE_TOPSTEP_50K_EXPRESS,
        PROFILE_TOPSTEP_50K_EXPRESS_LONDON_6B_PAPER,
    )
    assert build_config is not None


def test_execution_imports():
    from trading_system.execution.engine import ExecutionEngine
    from trading_system.execution.order_manager import OrderManager
    from trading_system.execution.broker import BrokerAdapter
    assert ExecutionEngine is not None


def test_risk_imports():
    from trading_system.risk.engine import RiskEngine
    from trading_system.risk.execution_checks import validate_intent
    assert RiskEngine is not None


def test_strategy_imports():
    from trading_system.strategy.rules import generate_intents, build_order_intent
    from trading_system.strategy.signal import SignalEngine
    assert generate_intents is not None


def test_backtest_imports():
    from trading_system.backtest.simulator import SessionSimulator
    from trading_system.backtest.metrics import compute_metrics
    assert SessionSimulator is not None


def test_data_pipeline_imports():
    from trading_system.data_pipeline.live_feed import TopstepLiveFeed
    assert TopstepLiveFeed is not None


def test_root_shim_config():
    """Root config.py shim must still export everything tests depend on."""
    from config import (
        TraderConfig, build_config, PROFILE_TOPSTEP_50K_EXPRESS,
        PROFILE_TOPSTEP_50K_EXPRESS_LONDON_6B_PAPER,
    )
    assert TraderConfig is not None


def test_root_shim_profiles():
    """Root profiles.py shim must still export profile constants."""
    from profiles import PROFILE_TOPSTEP_50K_EXPRESS, build_config
    assert build_config is not None
