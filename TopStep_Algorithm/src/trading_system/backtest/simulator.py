from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING

from trading_system.api.market_data import Bar
from trading_system.backtest.data_loader import SessionDay
from trading_system.backtest.raw_setup_ledger import RawSetupEvent, with_trade_outcome
from trading_system.config import INSTRUMENTS, RISK
from trading_system.strategy.signal import SignalConfig, SignalEngine, TradeSignal

if TYPE_CHECKING:
    from trading_system.backtest.config import TraderConfig

log = logging.getLogger(__name__)


@dataclass
class SimulatorConfig:
    """Backtest simulation parameters.

    The ``from_trader_config`` factory builds this from a full
    ``TraderConfig`` / profile so that profile-calibrated values are
    automatically wired in without manual duplication.
    """

    account_balance: float = 50_000.0
    max_trades_per_session: int = 2
    daily_loss_limit_usd: float = 1_000.0

    # --- profile-driven strategy parameters ---
    # TP1 = entry ± target_atr_multiple × ATR14
    # TP2 = entry ± (target_atr_multiple + 1) × ATR14
    target_atr_multiple: float = 2.0

    # 0.0 = disabled; > 0 moves SL to entry after TP1.
    # Profile sets to 0.0 (disabled) — enabling it collapsed WR to 16.5% in WFO v2/v3.
    breakeven_trigger_atr: float = 0.0

    # Minimum ADX-14 for entry (0.0 = disabled; 25.0 = profile default).
    adx_min_threshold: float = 0.0

    # Current ATR / 40-bar mean ATR must exceed this (0.0 = disabled; 0.85 = profile default).
    atr_min_pct: float = 0.0

    # Consecutive aligned EMA slopes required (0 = single-bar; 3 = profile default).
    ema_trend_persistence_bars: int = 0

    # TopStep trailing max-loss limit (absolute USD; positive value → converted to negative threshold).
    # TopStep 50K Express: $2,000 trailing drawdown from the session equity peak.
    max_loss_limit_usd: float = 2_000.0

    # Confluence type allowlist — mirrors SignalConfig.allowed_confluence_types.
    # Default keeps all types to preserve existing backtest behaviour.
    allowed_confluence_types: frozenset[str] = frozenset({"OB", "FVG", "OB+FVG"})

    @classmethod
    def from_trader_config(
        cls,
        trader_config: "TraderConfig",
        account_balance: float = 50_000.0,
    ) -> "SimulatorConfig":
        """Build a SimulatorConfig from a full profile TraderConfig."""
        s = trader_config.strategy
        r = trader_config.risk
        return cls(
            account_balance=account_balance,
            max_trades_per_session=r.max_trades_per_day,
            daily_loss_limit_usd=r.internal_daily_loss_limit,
            max_loss_limit_usd=getattr(r, "max_loss_limit_usd", 2_000.0),
            target_atr_multiple=s.target_atr_multiple,
            breakeven_trigger_atr=s.breakeven_trigger_atr,
            adx_min_threshold=s.adx_min_threshold,
            atr_min_pct=s.atr_min_pct,
            ema_trend_persistence_bars=s.ema_trend_persistence_bars,
            allowed_confluence_types=getattr(
                s, "allowed_confluence_types", frozenset({"OB", "FVG", "OB+FVG"})
            ),
        )

    def to_signal_config(self) -> SignalConfig:
        """Export the strategy-parameter subset as a ``SignalConfig``."""
        return SignalConfig(
            target_atr_multiple=self.target_atr_multiple,
            breakeven_trigger_atr=self.breakeven_trigger_atr,
            adx_min_threshold=self.adx_min_threshold,
            atr_min_pct=self.atr_min_pct,
            ema_trend_persistence_bars=self.ema_trend_persistence_bars,
            allowed_confluence_types=self.allowed_confluence_types,
        )


@dataclass(slots=True)
class TradeResult:
    symbol: str
    session_date: date
    direction: str
    entry_price: float
    entry_time: datetime
    stop_price: float
    tp1_price: float
    tp2_price: float
    tp1_filled: bool
    tp2_filled: bool
    sl_filled: bool
    exit_price: float
    exit_time: datetime
    contracts: int
    pnl_usd: float
    r_multiple: float
    confluence_type: str
    atr14: float
    ema_slope: float
    setup_type: str = "asian_range_sweep"
    adx_at_entry: float = 0.0
    vwap_distance_at_entry: float = 0.0
    range_width_atr: float = 0.0
    sweep_depth_atr: float = 0.0
    distance_to_asian_mid: float = 0.0
    time_since_sweep: int = 0
    risk_points: float = 0.0
    risk_usd: float = 0.0
    mfe_points: float = 0.0
    mae_points: float = 0.0
    mfe_r: float = 0.0
    mae_r: float = 0.0
    holding_minutes: float = 0.0
    exit_reason: str = ""
    bars_held: int = 0
    raw_setup_event_id: str = ""


@dataclass
class _OpenTrade:
    signal: TradeSignal
    entry_time: datetime
    contracts: int
    # After TP1 fills, track partial state
    tp1_filled: bool = False
    # Contracts still open after TP1 (TP2 leg)
    contracts_tp2: int = 0
    # SL price — starts at signal.stop_price; may move to entry (BE) after TP1
    sl_price: float = 0.0
    mfe_points: float = 0.0
    mae_points: float = 0.0
    bars_held: int = 0

    def __post_init__(self) -> None:
        self.sl_price = self.signal.stop_price
        c1 = max(1, self.contracts // 2)
        self.contracts_tp2 = self.contracts - c1


def _size_position(
    signal: TradeSignal,
    account_balance: float,
    min_contracts: int,
    max_contracts: int,
) -> int:
    """Risk-based position sizing.

    risk_amount = account_balance × 0.5%
    stop_distance / tick_size = ticks at risk
    dollar_risk_per_contract = ticks_at_risk × tick_value
    contracts = risk_amount / dollar_risk_per_contract
    """
    instrument = INSTRUMENTS[signal.symbol]
    stop_distance = abs(signal.entry_price - signal.stop_price)
    if stop_distance < instrument.tick_size:
        return min_contracts
    ticks_at_risk = stop_distance / instrument.tick_size
    dollar_risk_per_contract = ticks_at_risk * instrument.tick_value
    if dollar_risk_per_contract <= 0:
        return min_contracts
    raw = int(account_balance * RISK.account_risk_pct / dollar_risk_per_contract)
    return max(min_contracts, min(max_contracts, raw))


def _pnl_for_leg(
    entry: float,
    exit_price: float,
    qty: int,
    direction: str,
    symbol: str,
) -> float:
    instrument = INSTRUMENTS[symbol]
    signed_ticks = (exit_price - entry) / instrument.tick_size
    if direction == "SELL":
        signed_ticks = -signed_ticks
    return signed_ticks * instrument.tick_value * qty


def _initial_risk_points(signal: TradeSignal) -> float:
    return abs(signal.entry_price - signal.stop_price)


def _initial_risk_usd(signal: TradeSignal, contracts: int) -> float:
    instrument = INSTRUMENTS[signal.symbol]
    risk_points = _initial_risk_points(signal)
    return (risk_points / instrument.tick_size) * instrument.tick_value * contracts


def _update_excursion(trade: _OpenTrade, bar: Bar) -> None:
    signal = trade.signal
    if signal.direction == "BUY":
        favorable = max(bar.high - signal.entry_price, 0.0)
        adverse = max(signal.entry_price - bar.low, 0.0)
    else:
        favorable = max(signal.entry_price - bar.low, 0.0)
        adverse = max(bar.high - signal.entry_price, 0.0)
    trade.mfe_points = max(trade.mfe_points, favorable)
    trade.mae_points = max(trade.mae_points, adverse)
    trade.bars_held += 1


def _build_result(
    trade: _OpenTrade,
    exit_price: float,
    exit_time: datetime,
    tp1_filled: bool,
    tp2_filled: bool,
    sl_filled: bool,
    session_date: date,
    exit_reason: str,
) -> TradeResult:
    signal = trade.signal
    instrument = INSTRUMENTS[signal.symbol]
    contracts = trade.contracts
    c_tp1 = max(1, contracts // 2)
    c_tp2 = contracts - c_tp1

    # Determine P&L based on fill outcome
    if tp2_filled:
        # Full run: TP1 at tp1_price + TP2 at tp2_price
        pnl = _pnl_for_leg(signal.entry_price, signal.tp1_price, c_tp1, signal.direction, signal.symbol)
        pnl += _pnl_for_leg(signal.entry_price, signal.tp2_price, c_tp2, signal.direction, signal.symbol)
    elif tp1_filled and sl_filled:
        # TP1 leg won; TP2 leg exits at exit_price.
        # When breakeven is enabled, exit_price = entry (SL moved to BE → TP2 P&L ≈ 0).
        # When breakeven is disabled, exit_price = original stop (TP2 leg takes a real loss).
        pnl = _pnl_for_leg(signal.entry_price, signal.tp1_price, c_tp1, signal.direction, signal.symbol)
        pnl += _pnl_for_leg(signal.entry_price, exit_price, c_tp2, signal.direction, signal.symbol)
    elif tp1_filled:
        # Session closed after TP1 — TP2 leg exits at session close
        pnl = _pnl_for_leg(signal.entry_price, signal.tp1_price, c_tp1, signal.direction, signal.symbol)
        pnl += _pnl_for_leg(signal.entry_price, exit_price, c_tp2, signal.direction, signal.symbol)
    else:
        # Full stop-out or session close before TP1
        pnl = _pnl_for_leg(signal.entry_price, exit_price, contracts, signal.direction, signal.symbol)

    # R-multiple relative to initial 1R risk
    stop_dist = abs(signal.entry_price - signal.stop_price)
    dollar_1r = (stop_dist / instrument.tick_size) * instrument.tick_value * contracts
    r_multiple = pnl / dollar_1r if dollar_1r > 0 else 0.0
    holding_minutes = (exit_time - trade.entry_time).total_seconds() / 60.0

    return TradeResult(
        symbol=signal.symbol,
        session_date=session_date,
        direction=signal.direction,
        entry_price=signal.entry_price,
        entry_time=trade.entry_time,
        stop_price=signal.stop_price,
        tp1_price=signal.tp1_price,
        tp2_price=signal.tp2_price,
        tp1_filled=tp1_filled,
        tp2_filled=tp2_filled,
        sl_filled=sl_filled,
        exit_price=exit_price,
        exit_time=exit_time,
        contracts=contracts,
        pnl_usd=pnl,
        r_multiple=r_multiple,
        confluence_type=signal.confluence.confluence_type,
        atr14=signal.atr14,
        ema_slope=signal.ema_slope,
        setup_type=signal.setup_type,
        adx_at_entry=signal.adx_at_entry,
        vwap_distance_at_entry=signal.vwap_distance_at_entry,
        range_width_atr=signal.range_width_atr,
        sweep_depth_atr=signal.sweep_depth_atr,
        distance_to_asian_mid=signal.distance_to_asian_mid,
        time_since_sweep=signal.time_since_sweep,
        risk_points=stop_dist,
        risk_usd=_initial_risk_usd(signal, contracts),
        mfe_points=trade.mfe_points,
        mae_points=trade.mae_points,
        mfe_r=trade.mfe_points / stop_dist if stop_dist > 0 else 0.0,
        mae_r=trade.mae_points / stop_dist if stop_dist > 0 else 0.0,
        holding_minutes=holding_minutes,
        exit_reason=exit_reason,
        bars_held=trade.bars_held,
        raw_setup_event_id=signal.raw_setup_event_id,
    )


def _trade_id(result: TradeResult) -> str:
    ts = result.entry_time.strftime("%Y%m%d%H%M%S")
    return f"{result.symbol}-{ts}-{result.direction}"


def _link_raw_outcome(events: list[RawSetupEvent], result: TradeResult) -> None:
    if not result.raw_setup_event_id:
        return
    for idx, event in enumerate(events):
        if event.event_id == result.raw_setup_event_id:
            events[idx] = with_trade_outcome(
                event,
                linked_trade_id=_trade_id(result),
                pnl_usd=result.pnl_usd,
                r_multiple=result.r_multiple,
                exit_reason=result.exit_reason,
                mfe_r=result.mfe_r,
                mae_r=result.mae_r,
                holding_minutes=result.holding_minutes,
            )
            return


def _pop_raw_setup_events(engine: object) -> list[RawSetupEvent]:
    popper = getattr(engine, "pop_raw_setup_events", None)
    if popper is None:
        return []
    return list(popper())


class SessionSimulator:
    """Bar-level bracket order simulator for one SessionDay.

    Fill logic (conservative):
    - Entry fills at the signal bar's close price.
    - SL is checked before TP on the same bar (worst-case fill).
    - After TP1, the remaining leg's SL moves to entry (break-even) **only
      when** ``SimulatorConfig.breakeven_trigger_atr > 0`` — matching the
      profile setting.  The profile disables this (= 0.0) because enabling it
      collapsed WR to 16.5% in WFO v2/v3.
    - At session end, any open trade is force-closed at the last bar's close.
    """

    def __init__(self, config: SimulatorConfig | None = None) -> None:
        self._cfg = config or SimulatorConfig()
        self.last_raw_setup_events: list[RawSetupEvent] = []

    def run(self, session: SessionDay) -> list[TradeResult]:
        results: list[TradeResult] = []
        raw_events: list[RawSetupEvent] = []
        trades_taken = 0
        signal_cfg = self._cfg.to_signal_config()
        engine = SignalEngine(session.symbol, signal_cfg)

        bars_5m = session.bars_5m
        bars_1h = session.bars_1h

        # Pre-compute 1h timestamps for bisect slicing (no-lookahead guarantee)
        _1h_timestamps = [b.timestamp for b in bars_1h]

        open_trade: _OpenTrade | None = None
        tp1_filled = False
        sl_filled = False
        tp2_filled = False

        for i in range(session.london_start_idx, session.london_end_idx):
            bar = bars_5m[i]

            # ── Check fills on open trade first ──────────────────────────────
            if open_trade is not None:
                signal = open_trade.signal
                is_buy = signal.direction == "BUY"
                _update_excursion(open_trade, bar)

                # Effective SL (original stop, or entry if BE was triggered)
                current_sl = open_trade.sl_price

                # SL check (conservative: SL beats TP on same bar)
                sl_hit = (
                    bar.low <= current_sl if is_buy else bar.high >= current_sl
                )
                if sl_hit:
                    exit_reason = (
                        "breakeven_stop"
                        if tp1_filled and current_sl == signal.entry_price
                        else "stop_loss"
                    )
                    result = _build_result(
                        open_trade, current_sl, bar.timestamp,
                        tp1_filled, False, True, session.date, exit_reason,
                    )
                    results.append(result)
                    _link_raw_outcome(raw_events, result)
                    open_trade = None
                    tp1_filled = sl_filled = tp2_filled = False
                    continue

                # TP1 check (if not yet filled)
                if not tp1_filled:
                    tp1_hit = (
                        bar.high >= signal.tp1_price if is_buy
                        else bar.low <= signal.tp1_price
                    )
                    if tp1_hit:
                        tp1_filled = True
                        # Move SL to break-even only when the profile enables it
                        if self._cfg.breakeven_trigger_atr > 0:
                            open_trade.sl_price = signal.entry_price
                            log.debug(
                                "TP1 filled %s %s @ %.5f — SL moved to BE",
                                signal.symbol, signal.direction, signal.tp1_price,
                            )
                        else:
                            log.debug(
                                "TP1 filled %s %s @ %.5f — SL stays at %.5f (BE disabled)",
                                signal.symbol, signal.direction,
                                signal.tp1_price, open_trade.sl_price,
                            )

                # TP2 check (only after TP1)
                if tp1_filled:
                    tp2_hit = (
                        bar.high >= signal.tp2_price if is_buy
                        else bar.low <= signal.tp2_price
                    )
                    if tp2_hit:
                        result = _build_result(
                            open_trade, signal.tp2_price, bar.timestamp,
                            True, True, False, session.date, "take_profit_2",
                        )
                        results.append(result)
                        _link_raw_outcome(raw_events, result)
                        open_trade = None
                        tp1_filled = sl_filled = tp2_filled = False

            # ── Signal generation (only if no open trade and under cap) ──────
            if open_trade is None and trades_taken < self._cfg.max_trades_per_session:
                current_5m = bars_5m[: i + 1]
                # No-lookahead bisect slice: exclude any 1h bar whose open timestamp
                # equals the current 5m bar's timestamp (that 1h bar has not yet closed).
                cutoff_idx = bisect.bisect_left(_1h_timestamps, bar.timestamp)
                current_1h = bars_1h[:cutoff_idx]

                trade_signal = engine.process_bar(current_5m, current_1h)
                raw_events.extend(_pop_raw_setup_events(engine))

                if trade_signal is not None:
                    contracts = _size_position(
                        trade_signal,
                        self._cfg.account_balance,
                        RISK.min_contracts,
                        RISK.max_contracts,
                    )
                    for idx, event in enumerate(raw_events):
                        if event.event_id == trade_signal.raw_setup_event_id:
                            raw_events[idx].accepted_into_trade = True
                            break
                    open_trade = _OpenTrade(
                        signal=trade_signal,
                        entry_time=bar.timestamp,
                        contracts=contracts,
                    )
                    trades_taken += 1
                    tp1_filled = sl_filled = tp2_filled = False
                    log.debug(
                        "Trade opened %s %s @ %.5f  SL=%.5f  TP1=%.5f  TP2=%.5f  contracts=%d",
                        trade_signal.symbol, trade_signal.direction,
                        trade_signal.entry_price, trade_signal.stop_price,
                        trade_signal.tp1_price, trade_signal.tp2_price, contracts,
                    )

        # ── Force-close any open trade at session end ─────────────────────
        if open_trade is not None:
            last_bar = bars_5m[session.london_end_idx - 1]
            result = _build_result(
                open_trade, last_bar.close, last_bar.timestamp,
                tp1_filled, False, False, session.date, "session_close",
            )
            results.append(result)
            _link_raw_outcome(raw_events, result)

        self.last_raw_setup_events = raw_events
        return results
