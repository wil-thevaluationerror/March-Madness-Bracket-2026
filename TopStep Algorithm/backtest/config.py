from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time

from models.orders import TradingMode


@dataclass(frozen=True, slots=True)
class SessionWindow:
    label: str
    market_open: time
    no_new_trades_after: time
    force_flatten_at: time
    exchange_close: time


def default_session_windows() -> tuple[SessionWindow, ...]:
    return (
        SessionWindow(
            label="asia_pre_london",
            market_open=time(hour=17, minute=0),
            no_new_trades_after=time(hour=1, minute=35),
            force_flatten_at=time(hour=1, minute=58),
            exchange_close=time(hour=2, minute=0),
        ),
        SessionWindow(
            label="new_york",
            market_open=time(hour=8, minute=30),
            no_new_trades_after=time(hour=14, minute=45),
            force_flatten_at=time(hour=15, minute=8),
            exchange_close=time(hour=15, minute=10),
        ),
    )


@dataclass(slots=True)
class SessionConfig:
    timezone: str = "America/Chicago"
    market_open: time = time(hour=8, minute=30)
    no_new_trades_after: time = time(hour=14, minute=45)
    force_flatten_at: time = time(hour=15, minute=8)
    exchange_close: time = time(hour=15, minute=10)
    session_windows: tuple[SessionWindow, ...] = field(default_factory=default_session_windows)


@dataclass(slots=True)
class RiskLimits:
    max_position_size: int = 1
    max_concurrent_positions: int = 1
    enable_stacking: bool = False
    max_positions_per_regime: int = 1
    max_same_direction_entries_per_trend: int = 2
    max_active_intents_per_symbol: int = 1
    max_trades_per_day: int = 3
    max_consecutive_losses: int = 2
    internal_daily_loss_limit: float = 300.0
    risk_budget_threshold: float = 150.0
    stacked_risk_budget_fraction: float = 0.4
    max_cluster_risk_fraction: float = 0.6
    convex_position_scaling: tuple[float, ...] = (1.0, 0.45)
    drawdown_risk_tiers: tuple[tuple[float, float], ...] = (
        (0.0, 1.0),
        (-1000.0, 0.75),
        (-2000.0, 0.6),
        (-3000.0, 0.4),
    )
    same_direction_spacing_bars: int = 5
    stacking_allowed_regimes: tuple[str, ...] = ("TREND_EXPANSION", "HIGH_VOL_BREAKOUT")
    stacking_price_atr_threshold: float = 0.2
    stacking_base_score_threshold: float = 0.6
    stacking_score_margin: float = 0.15
    stacking_drawdown_margin_step: float = 0.05
    stacking_cluster_margin_step: float = 0.05
    stacking_volatility_upper_atr_pct: float = 1.6
    stacking_recent_pnl_lookback: int = 3
    stacking_loss_cooldown_bars: int = 5
    stacking_consecutive_loss_limit: int = 2
    disable_stacking_midday: bool = True
    midday_stacking_score_penalty: float = 0.2
    heat_decay_lambda: float = 0.15
    min_reentry_spacing_bars: int = 2
    reentry_breakout_delta_min: float = 0.0
    reentry_atr_pct_max: float = 1.2
    reentry_signal_score_min: float = 0.9
    reentry_score_margin: float = 0.05
    max_stop_distance_pct: float = 0.02
    max_stop_distance_ticks: int = 40
    max_api_errors: int = 3
    max_slippage_ticks: int = 4
    cooldown_minutes: int = 30
    kill_switch_on_reconcile_mismatch: bool = True


@dataclass(slots=True)
class ExecutionConfig:
    mode: TradingMode = TradingMode.MOCK
    use_native_brackets: bool = False
    reconcile_interval_seconds: int = 3
    stale_order_seconds: int = 20
    retry_transient_order_errors: int = 1
    intent_expiry_seconds: int = 6 * 60 * 60
    max_orders_per_second: int = 2
    reconnect_timeout_seconds: int = 15
    trade_log_dir: str = "runtime_logs"


@dataclass(slots=True)
class StrategyConfig:
    strategy_id: str = "ema_vwap_breakout"
    default_symbol: str = "MES"
    instrument_root_symbol: str = "MES"
    base_qty: int = 1
    default_entry_order_type: str = "MARKET"
    allow_lunch_trading: bool = False
    reentry_cooldown_bars: int = 5
    breakout_lookback_bars: int = 20
    preferred_symbol: str | None = None


@dataclass(slots=True)
class TraderConfig:
    session: SessionConfig = field(default_factory=SessionConfig)
    risk: RiskLimits = field(default_factory=RiskLimits)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
