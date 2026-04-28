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
    # Dates to skip entirely (no new entries, no position management beyond existing holds).
    # Use for CME holiday closures and thin-market sessions (e.g. Dec 24, Black Friday).
    # Expressed as local date values — compared against Chicago-local session date.
    skip_dates: tuple = field(default_factory=tuple)  # tuple[date, ...]


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
    # Trailing drawdown kill switch: arm and halt trading when drawdown from equity peak
    # reaches this level.  0.0 = disabled.  Set to -1800.0 for TopStep 50K Express
    # ($200 buffer before the $2,000 platform trailing drawdown limit).
    trailing_drawdown_kill_switch: float = 0.0
    reentry_score_margin: float = 0.05
    max_stop_distance_pct: float = 0.02
    max_stop_distance_ticks: int = 80
    max_api_errors: int = 3
    max_slippage_ticks: int = 4
    cooldown_minutes: int = 30
    kill_switch_on_reconcile_mismatch: bool = True


@dataclass(slots=True)
class TopstepConnectionConfig:
    environment: str = "paper"
    api_base_url: str = "https://api.topstepx.com"
    websocket_url: str = "https://rtc.topstepx.com/hubs/user"
    username: str = ""
    api_key: str = ""
    password: str = ""
    account_id: str = ""
    request_timeout_seconds: int = 10
    token_refresh_margin_seconds: int = 60

    def missing_required_fields(self) -> tuple[str, ...]:
        required = {
            "api_base_url": self.api_base_url,
            "username": self.username,
            "api_key": self.api_key,
            "account_id": self.account_id,
        }
        return tuple(name for name, value in required.items() if not str(value).strip())


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
    # Round-trip commission per lot (one side). MES at NinjaTrader/Rithmic rates = $0.59/side.
    # Set to 0.0 to disable commission modeling (default keeps old behavior).
    commission_per_lot: float = 0.0
    # When True, backtest mode enforces the same daily loss limit, trade count cap,
    # cooldown, and lock checks that live mode uses.  Default False keeps old behavior
    # (backtest skips these guards to avoid artificially restricting OOS windows).
    # Enable with --enforce-live-rules to get a simulation closer to production conditions.
    enforce_live_risk_rules: bool = False
    topstep: TopstepConnectionConfig = field(default_factory=TopstepConnectionConfig)


@dataclass(slots=True)
class StrategyConfig:
    strategy_id: str = "ema_vwap_breakout"
    default_symbol: str = "MES"
    instrument_root_symbol: str = "MES"
    base_qty: int = 1
    max_entry_extension_atr: float = 0.75
    default_entry_order_type: str = "MARKET"
    allow_lunch_trading: bool = False
    reentry_cooldown_bars: int = 5
    breakout_lookback_bars: int = 20
    preferred_symbol: str | None = None
    # Anti-overfitting quality gates
    min_entry_signal_score: float = 0.0  # 0.0 = disabled; set to 0.3-0.5 to filter weak initial entries
    volume_entry_filter: float = 0.0  # 0.0 = disabled; set to 0.8-1.0 to require near/above-median volume
    use_5min_atr_for_stops: bool = False  # True = use 5-min ATR for stop/target sizing (less noise-prone)
    # Breakeven stop: after price moves this many ATRs in our favour, slide the stop to entry.
    # 0.0 = disabled. 0.75 means slide stop to entry after 75% of a full ATR of profit.
    breakeven_trigger_atr: float = 0.0
    # R:R multiple for profit target.  Default 2.0 = 2× stop distance.
    # Set to 3.0 to reduce breakeven win-rate requirement from 33.3% → 25%.
    target_atr_multiple: float = 2.0
    # Minimum ADX value required to allow a trade entry.  0.0 = disabled.
    # Set to 20.0+ to filter out choppy/range-bound regimes.
    adx_min_threshold: float = 0.0
    # EMA trend persistence: require EMA_fast > EMA_slow (long) or < (short) for this
    # many consecutive bars before allowing entry.  0 = disabled (single-bar check only).
    # A value of 3 filters out EMA crossings that immediately reverse (whipsaw signals).
    ema_trend_persistence_bars: int = 0
    # Minimum ATR as a fraction of the rolling median ATR.  0.0 = disabled.
    # Set to 0.7 to skip entries on anomalously low-volatility bars (ranging conditions
    # where 2× ATR targets are too far to reach before a reversal or session end).
    atr_min_pct: float = 0.0


@dataclass(slots=True)
class TraderConfig:
    session: SessionConfig = field(default_factory=SessionConfig)
    risk: RiskLimits = field(default_factory=RiskLimits)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
