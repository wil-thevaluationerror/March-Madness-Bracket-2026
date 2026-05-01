from __future__ import annotations

from datetime import time

from trading_system.config import SessionWindow, TraderConfig
from trading_system.backtest.holidays import us_futures_skip_dates

PROFILE_TOPSTEP_50K_EXPRESS = "topstep-50k-express"
PROFILE_TOPSTEP_50K_EXPRESS_LONDON = "topstep-50k-express-london"
PROFILE_TOPSTEP_50K_EXPRESS_LONDON_6B_PAPER = "topstep-50k-express-london-6b-paper"
PROFILE_TOPSTEP_50K_EXPRESS_LONDON_6E_PAPER = "topstep-50k-express-london-6e-paper"


def available_profiles() -> tuple[str, ...]:
    return (
        PROFILE_TOPSTEP_50K_EXPRESS,
        PROFILE_TOPSTEP_50K_EXPRESS_LONDON,
        PROFILE_TOPSTEP_50K_EXPRESS_LONDON_6B_PAPER,
        PROFILE_TOPSTEP_50K_EXPRESS_LONDON_6E_PAPER,
    )


def _apply_topstep_50k_express_base(config: TraderConfig) -> TraderConfig:
    """Shared Topstep 50K Express tuning used across session variants."""
    config.strategy.base_qty = 8
    config.strategy.preferred_symbol = "MES"

    config.strategy.min_entry_signal_score = 0.45
    config.strategy.volume_entry_filter = 0.9
    config.strategy.use_5min_atr_for_stops = False

    config.strategy.breakeven_trigger_atr = 0.0

    config.strategy.target_atr_multiple = 2.0

    config.strategy.adx_min_threshold = 25.0

    config.strategy.atr_min_pct = 0.85

    config.strategy.ema_trend_persistence_bars = 3

    config.session.skip_dates = us_futures_skip_dates(2024, 2027)

    config.execution.commission_per_lot = 0.59

    config.risk.trailing_drawdown_kill_switch = -1800.0

    config.risk.max_position_size = 20
    config.risk.max_concurrent_positions = 1
    config.risk.enable_stacking = False
    config.risk.max_trades_per_day = 4
    config.risk.max_consecutive_losses = 3
    config.risk.internal_daily_loss_limit = 600.0
    config.risk.risk_budget_threshold = 125.0
    config.risk.reentry_breakout_delta_min = 0.0
    config.risk.reentry_signal_score_min = 0.65
    config.risk.drawdown_risk_tiers = (
        (0.0, 1.0),
        (-400.0, 0.7),
        (-900.0, 0.5),
        (-1400.0, 0.35),
        (-1750.0, 0.25),
    )
    return config


def apply_profile(config: TraderConfig, profile: str | None) -> TraderConfig:
    if profile is None:
        return config
    if profile not in available_profiles():
        raise ValueError(f"Unsupported profile: {profile}")
    _apply_topstep_50k_express_base(config)

    if profile == PROFILE_TOPSTEP_50K_EXPRESS:
        config.session.session_windows = (
            SessionWindow(
                label="new_york",
                market_open=time(hour=8, minute=30),
                no_new_trades_after=time(hour=12, minute=30),
                force_flatten_at=time(hour=15, minute=8),
                exchange_close=time(hour=15, minute=10),
            ),
        )
        return config

    config.strategy.allowed_confluence_types = frozenset({"FVG", "OB+FVG"})

    config.session.timezone = "UTC"
    config.session.session_windows = (
        SessionWindow(
            label="london",
            market_open=time(hour=8, minute=30),
            no_new_trades_after=time(hour=13, minute=0),
            force_flatten_at=time(hour=13, minute=25),
            exchange_close=time(hour=13, minute=30),
        ),
    )
    if profile == PROFILE_TOPSTEP_50K_EXPRESS_LONDON_6B_PAPER:
        config.strategy.preferred_symbol = "6B"
        config.strategy.instrument_root_symbol = "6B"
        config.strategy.base_qty = 1
        config.risk.max_position_size = 1
        config.execution.commission_per_lot = 0.0
    elif profile == PROFILE_TOPSTEP_50K_EXPRESS_LONDON_6E_PAPER:
        config.strategy.preferred_symbol = "6E"
        config.strategy.instrument_root_symbol = "6E"
        config.strategy.base_qty = 1
        config.risk.max_position_size = 1
        config.execution.commission_per_lot = 0.0
    return config


def build_config(profile: str | None = None) -> TraderConfig:
    return apply_profile(TraderConfig(), profile)
