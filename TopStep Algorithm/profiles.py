from __future__ import annotations

from backtest.config import TraderConfig

PROFILE_TOPSTEP_50K_EXPRESS = "topstep-50k-express"


def available_profiles() -> tuple[str, ...]:
    return (PROFILE_TOPSTEP_50K_EXPRESS,)


def apply_profile(config: TraderConfig, profile: str | None) -> TraderConfig:
    if profile is None:
        return config
    if profile != PROFILE_TOPSTEP_50K_EXPRESS:
        raise ValueError(f"Unsupported profile: {profile}")

    # Topstep's 50K Express Funded account uses a $2,000 maximum loss limit and starts
    # with a 2-lot scaling cap. We keep the strategy materially below that cap by default,
    # but let the runtime know the true account envelope.
    config.strategy.base_qty = 2
    config.strategy.preferred_symbol = "MES"

    config.risk.max_position_size = 20
    config.risk.max_concurrent_positions = 1
    config.risk.enable_stacking = False
    config.risk.max_trades_per_day = 8
    config.risk.max_consecutive_losses = 3
    config.risk.internal_daily_loss_limit = 500.0
    config.risk.risk_budget_threshold = 125.0
    config.risk.reentry_breakout_delta_min = -1_000_000_000.0
    config.risk.drawdown_risk_tiers = (
        (0.0, 1.0),
        (-400.0, 0.7),
        (-900.0, 0.5),
        (-1400.0, 0.35),
        (-1750.0, 0.25),
    )
    return config


def build_config(profile: str | None = None) -> TraderConfig:
    return apply_profile(TraderConfig(), profile)
