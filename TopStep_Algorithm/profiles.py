from __future__ import annotations

from datetime import time

from backtest.config import SessionWindow, TraderConfig
from backtest.holidays import us_futures_skip_dates

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
    # Topstep 50K Express: $2,000 trailing max drawdown, $3,000 profit target.
    # v10: scaled to 8 MES lots to minimise expected subscription cost before funding.
    # At 2 lots the median pass time was ~50 months (~$5,500+ in subscriptions at $110/mo).
    # At 8 lots the median drops to ~9 months (~$1,726 expected subscription cost).
    # P(pass) per attempt: 57% vs 95% at 2 lots — acceptable because faster median
    # more than compensates for extra retry cost.
    # The drawdown tiers (absolute dollar thresholds) automatically reduce lot size
    # as equity drawsdown, so bad-window losses are partially capped.  The trailing-DD
    # kill switch hard-caps cumulative session risk at -$1,800.
    config.strategy.base_qty = 8
    config.strategy.preferred_symbol = "MES"

    # Entry quality gates (v6 calibration).
    # min_entry_signal_score raised to 0.45 (was 0.35) to admit only top-quality setups.
    # volume_entry_filter at 0.9 requires near-median volume to confirm breakout intent.
    # use_5min_atr_for_stops disabled: 5-min ATR stops are 2-3× wider than 1-min ATR,
    #   pushing targets to 3–6× the typical session range and near-eliminating winners.
    config.strategy.min_entry_signal_score = 0.45
    config.strategy.volume_entry_filter = 0.9
    config.strategy.use_5min_atr_for_stops = False

    # Breakeven stop disabled: proven harmful in WFO v2/v3 (win rate collapsed to 16.5%).
    config.strategy.breakeven_trigger_atr = 0.0

    # 2:1 profit target (reverted from 3:1 used in v4/v5).
    # WFO v4/v5 showed 3:1 targets require ~$115 favorable moves; in choppy windows the
    # average MFE was only ~$36, so the 3:1 target was almost never reached.
    # 2:1 targets need only ~$77 and are achievable; commission-adjusted breakeven WR = ~36.5%.
    # v1 baseline was 34.6% WR without any filtering — ADX + EMA persistence should push
    # cleanly past 36.5%.
    config.strategy.target_atr_multiple = 2.0

    # ADX ≥ 25 regime filter: only enter on bars with confirmed directional momentum.
    # ~39% of 1-min bars qualify (vs. 60% at ADX≥20).  Raises bar for entry quality
    # without over-restricting trade count to the point of statistical irrelevance.
    config.strategy.adx_min_threshold = 25.0

    # ATR volatility floor: skip entries when current ATR < 85% of rolling median ATR.
    # Raised from 0.7 (v8) to 0.85 (v11) based on WFO trade analysis at 8 lots:
    #   low-ATR trades (bottom third, ATR < 2.21 pts): 90 trades, 26% WR, -$2,105 total
    #   mid-ATR trades (middle third, 2.21-3.29 pts): 92 trades, 37% WR, +$155 total
    #   high-ATR trades (top third, ATR > 3.29 pts): 93 trades, 44% WR, +$3,799 total
    # The bottom third is a large drag — low ATR means the 2× target is rarely reachable
    # before reversal.  0.85 cuts most of the sub-2.2pt entries while leaving mid/high.
    config.strategy.atr_min_pct = 0.85

    # EMA trend persistence: require EMA_fast > EMA_slow for 3 consecutive bars before entry.
    # Filters out EMA crossings that immediately reverse (whipsaw entries in choppy regimes).
    config.strategy.ema_trend_persistence_bars = 3

    # Holiday calendar: skip CME full-closure days and structurally thin-market
    # sessions (Black Friday, Christmas Eve, Veterans Day) where MES NY-session
    # volume is severely impaired.  W9 (Nov 30–Dec 22 2025) is the worst OOS
    # window (-$749 at 8 lots) partly because of the Thanksgiving/Christmas thin-market
    # calendar effect; this filter directly addresses those sessions.
    config.session.skip_dates = us_futures_skip_dates(2024, 2027)

    # Commission: MES round-trip at NinjaTrader/Rithmic rates ($0.59/side).
    # This is applied in the backtest P&L so results reflect real trading economics.
    config.execution.commission_per_lot = 0.59

    # Trailing drawdown kill switch: halt all trading when cumulative drawdown from the
    # session equity peak reaches -$1,800 — a $200 buffer before the $2,000 TopStep limit.
    # This runs in backtest mode too (no is_backtest_mode guard) so WFO results reflect
    # the same hard stop that will exist in production.
    config.risk.trailing_drawdown_kill_switch = -1800.0

    config.risk.max_position_size = 20
    config.risk.max_concurrent_positions = 1
    config.risk.enable_stacking = False
    # Reduced from 8 to 5 at 6 lots: each trade now risks ~$112 (3.5 pts × 6 × $5 + $7
    # commission) vs ~$37 at 2 lots.  5 trades × $112 = $560 max daily exposure before
    # the drawdown tiers step in — keeps worst-case single-day loss within 28% of the
    # $2,000 trailing drawdown limit.
    config.risk.max_trades_per_day = 4
    config.risk.max_consecutive_losses = 3
    # Internal daily loss limit scaled for 8-lot sizing.
    # At 8 lots, each trade risks ~$147 (3.5 pts × 8 × $5 + $9.44 commission).
    # 4 trades × $147 = $588 max daily exposure before tiers step in.
    # $600 cap halts after ~4 losses, within 30% of the $2,000 trailing DD limit.
    config.risk.internal_daily_loss_limit = 600.0
    # risk_budget_threshold is in price-point × contract units (not dollars).
    # At 6 lots, cluster risk per trade ≈ 3.5 pts × 6 = 21 pt-contracts.
    # Budget = 125 × 0.6 = 75 pt-contracts → one trade uses 28% of budget.
    # With max_concurrent_positions = 1, the cap is never binding; leave unchanged.
    config.risk.risk_budget_threshold = 125.0
    # Reentry breakout delta: require the breakout level to have advanced at least 0.25 ATR
    # before allowing a reentry in the same direction.  This prevents churning into the same
    # stall point repeatedly.  Use a sentinel like -1e9 to disable.
    config.risk.reentry_breakout_delta_min = 0.0
    # Reentry signal score recalibrated for the normalized [0, 1] score space.
    # 0.65 requires a strong-quality setup; equivalent to the old ~0.9 in un-normalized space.
    config.risk.reentry_signal_score_min = 0.65
    # Drawdown tiers protect the $2,000 trailing limit by scaling size down as equity erodes.
    # Tiers activate at $400, $900, $1,400, $1,750 of daily drawdown from peak.
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
        # Restrict trading to the NY regular session only (08:30–15:10 CT).
        # New entries cut off at 12:30 CT: the 12:30–14:45 AFTERNOON bucket shows
        # -$2.1/trade expectancy in OOS testing, and entry volume thins materially after midday.
        # Positions opened before 12:30 remain open until stop/target or the 15:08 flatten.
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

    # Asia/pre-London variant: use the same Topstep risk model but trade only the
    # evening-to-pre-London window from the default session schedule.
    config.session.session_windows = (
        SessionWindow(
            label="asia_pre_london",
            market_open=time(hour=17, minute=0),
            no_new_trades_after=time(hour=1, minute=35),
            force_flatten_at=time(hour=1, minute=58),
            exchange_close=time(hour=2, minute=0),
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
