from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from config import RiskLimits
from models.orders import KillSwitchState, OrderIntent, PositionSnapshot


@dataclass(slots=True)
class RiskState:
    daily_realized_pnl: float = 0.0
    equity_peak: float = 0.0
    trade_count: int = 0
    completed_trade_count: int = 0
    consecutive_losses: int = 0
    cooldown_until: datetime | None = None
    last_entry_at: datetime | None = None
    last_exit_at: datetime | None = None
    last_stacked_loss_at: datetime | None = None
    last_exit_regime: str = ""
    last_exit_trend_state: str = ""
    last_exit_signal_score: float = 0.0
    last_exit_breakout_level: float = 0.0
    last_exit_volume_strength: float = 0.0
    last_exit_side: str = ""
    last_trade_was_loss: bool = False
    current_trend_key: str = ""
    entries_in_current_trend: int = 0
    stacking_disabled_until: datetime | None = None
    stacking_disabled_reason: str = ""
    recent_stacked_trade_pnls: list[float] = field(default_factory=list)
    consecutive_stacked_losses: int = 0
    stacking_disabled_events: int = 0
    api_error_count: int = 0
    open_intents_by_symbol: dict[str, int] = field(default_factory=dict)
    positions: dict[str, PositionSnapshot] = field(default_factory=dict)
    open_trade_pnl_by_symbol: dict[str, float] = field(default_factory=dict)
    locked: bool = False
    kill_switch: KillSwitchState = field(default_factory=KillSwitchState)


class RiskEngine:
    def __init__(self, limits: RiskLimits, mode: str = "live", enforce_live_risk_rules: bool = False) -> None:
        self.limits = limits
        self.mode = mode
        # When True (--enforce-live-rules flag), backtest mode applies the same daily loss
        # limit, trade count cap, and cooldown/lock guards that live mode uses.
        self.enforce_live_risk_rules = enforce_live_risk_rules
        self.state = RiskState()

    @property
    def is_backtest_mode(self) -> bool:
        return self.mode in {"backtest", "research"}

    @property
    def _apply_live_guards(self) -> bool:
        """True when live-mode risk guards should be applied (either live mode or enforce flag set)."""
        return not self.is_backtest_mode or self.enforce_live_risk_rules

    def position_for(self, symbol: str) -> PositionSnapshot:
        return self.state.positions.setdefault(symbol, PositionSnapshot(symbol=symbol))

    def update_open_trade_pnl(self, symbol: str, pnl: float) -> float:
        total = self.state.open_trade_pnl_by_symbol.get(symbol, 0.0) + pnl
        self.state.open_trade_pnl_by_symbol[symbol] = total
        return total

    def clear_open_trade_pnl(self, symbol: str) -> float:
        return self.state.open_trade_pnl_by_symbol.pop(symbol, 0.0)

    def can_trade(self, now: datetime) -> tuple[bool, str]:
        if self.state.kill_switch.armed:
            return False, f"kill_switch:{self.state.kill_switch.reason}"
        # Trailing drawdown kill switch: active in all modes when configured.
        # Arm and halt when drawdown from equity peak reaches the configured limit.
        drawdown_kill = float(getattr(self.limits, "trailing_drawdown_kill_switch", 0.0))
        if drawdown_kill < 0.0 and self.current_drawdown() <= drawdown_kill:
            self.arm_kill_switch("trailing_drawdown_limit", now)
            return False, "trailing_drawdown_limit"
        if self._apply_live_guards and self.state.locked:
            return False, "daily_lockout"
        if self._apply_live_guards and self.state.cooldown_until and now < self.state.cooldown_until:
            return False, "cooldown_active"
        if self._apply_live_guards and self.state.trade_count >= self.limits.max_trades_per_day:
            return False, "max_trades_reached"
        if self._apply_live_guards and self.state.daily_realized_pnl <= -abs(self.limits.internal_daily_loss_limit):
            return False, "daily_loss_breached"
        if self.state.api_error_count >= self.limits.max_api_errors:
            self.arm_kill_switch("api_error_threshold", now)
            return False, "api_error_threshold"
        return True, "ok"

    def current_equity(self) -> float:
        return float(self.state.daily_realized_pnl)

    def current_drawdown(self) -> float:
        return self.current_equity() - float(self.state.equity_peak)

    def current_risk_multiplier(self) -> float:
        drawdown = self.current_drawdown()
        ordered_tiers = sorted(
            ((float(threshold), float(multiplier)) for threshold, multiplier in self.limits.drawdown_risk_tiers),
            key=lambda item: item[0],
        )
        for threshold, multiplier in ordered_tiers:
            if drawdown <= threshold:
                return multiplier
        return 1.0

    def crossed_drawdown_tiers(self) -> int:
        drawdown = self.current_drawdown()
        negative_thresholds = sorted(
            float(threshold)
            for threshold, _ in self.limits.drawdown_risk_tiers
            if float(threshold) < 0.0
        )
        return sum(1 for threshold in negative_thresholds if drawdown <= threshold)

    def register_open_intent(self, symbol: str) -> None:
        self.state.open_intents_by_symbol[symbol] = self.state.open_intents_by_symbol.get(symbol, 0) + 1

    def resolve_open_intent(self, symbol: str) -> None:
        current = self.state.open_intents_by_symbol.get(symbol, 0)
        if current <= 1:
            self.state.open_intents_by_symbol.pop(symbol, None)
        else:
            self.state.open_intents_by_symbol[symbol] = current - 1

    def open_intents_for(self, symbol: str) -> int:
        return self.state.open_intents_by_symbol.get(symbol, 0)

    def record_fill(self, symbol: str, qty_delta: int, fill_price: float, now: datetime) -> None:
        position = self.position_for(symbol)
        if position.qty == 0:
            position.avg_price = fill_price
        elif (position.qty > 0 and qty_delta > 0) or (position.qty < 0 and qty_delta < 0):
            total_qty = abs(position.qty) + abs(qty_delta)
            assert position.avg_price is not None
            position.avg_price = ((position.avg_price * abs(position.qty)) + (fill_price * abs(qty_delta))) / total_qty
        position.qty += qty_delta
        if position.qty == 0:
            position.avg_price = None
            position.stop_covered_qty = 0
        position.last_updated = now

    def record_stop_coverage(self, symbol: str, covered_qty: int, now: datetime) -> None:
        position = self.position_for(symbol)
        position.stop_covered_qty = covered_qty
        position.last_updated = now

    def record_trade_close(self, pnl: float, now: datetime) -> None:
        self.state.completed_trade_count += 1
        self.state.daily_realized_pnl += pnl
        self.state.equity_peak = max(self.state.equity_peak, self.state.daily_realized_pnl)
        self.state.last_exit_at = now
        self.state.last_trade_was_loss = pnl < 0
        if pnl < 0:
            self.state.consecutive_losses += 1
            if self._apply_live_guards and self.state.consecutive_losses >= self.limits.max_consecutive_losses:
                self.state.cooldown_until = now + timedelta(minutes=self.limits.cooldown_minutes)
        else:
            self.state.consecutive_losses = 0
        if self._apply_live_guards and self.state.daily_realized_pnl <= -abs(self.limits.internal_daily_loss_limit):
            self.state.locked = True

    def _trend_key_for_intent(self, intent: OrderIntent) -> str:
        regime = intent.regime.value if hasattr(intent.regime, "value") else str(intent.regime)
        side = intent.side.value if hasattr(intent.side, "value") else str(intent.side)
        trend_state = str(intent.metadata.get("trend_state") or "unknown")
        return f"{intent.symbol}|{side}|{regime}|{trend_state}"

    def is_reentry_for_intent(self, intent: OrderIntent) -> bool:
        return (
            bool(self.state.current_trend_key)
            and self.state.last_exit_at is not None
            and self._trend_key_for_intent(intent) == self.state.current_trend_key
            and self.state.entries_in_current_trend > 0
        )

    def projected_entry_number(self, intent: OrderIntent) -> int:
        return self.state.entries_in_current_trend + 1 if self.is_reentry_for_intent(intent) else 1

    def record_entry(self, intent: OrderIntent | None = None, now: datetime | None = None) -> None:
        self.state.trade_count += 1
        if now is not None:
            self.state.last_entry_at = now
        if intent is not None:
            trend_key = self._trend_key_for_intent(intent)
            if trend_key != self.state.current_trend_key:
                self.state.current_trend_key = trend_key
                self.state.entries_in_current_trend = 0
            self.state.entries_in_current_trend += 1

    def record_exit_context(self, intent: OrderIntent | None, pnl: float, now: datetime) -> None:
        self.state.last_exit_at = now
        self.state.last_trade_was_loss = pnl < 0
        if intent is None:
            return
        self.state.last_exit_regime = intent.regime.value if hasattr(intent.regime, "value") else str(intent.regime)
        self.state.last_exit_trend_state = str(intent.metadata.get("trend_state") or "unknown")
        self.state.last_exit_signal_score = float(intent.signal_score or 0.0)
        self.state.last_exit_breakout_level = float(intent.metadata.get("breakout_level", 0.0) or 0.0)
        self.state.last_exit_volume_strength = float(intent.metadata.get("volume_strength", 0.0) or 0.0)
        self.state.last_exit_side = intent.side.value if hasattr(intent.side, "value") else str(intent.side)

    def record_stacked_loss(self, now: datetime) -> None:
        self.state.last_stacked_loss_at = now

    def stacking_is_disabled(self, now: datetime) -> tuple[bool, str]:
        if self.state.stacking_disabled_until and now < self.state.stacking_disabled_until:
            return True, self.state.stacking_disabled_reason or "stacking_disabled"
        return False, ""

    def record_stacked_trade_close(self, pnl: float, now: datetime) -> None:
        lookback = max(int(self.limits.stacking_recent_pnl_lookback), 1)
        recent = list(self.state.recent_stacked_trade_pnls)
        recent.append(float(pnl))
        self.state.recent_stacked_trade_pnls = recent[-lookback:]

        if pnl < 0:
            self.state.last_stacked_loss_at = now
            self.state.consecutive_stacked_losses += 1
        else:
            self.state.consecutive_stacked_losses = 0

        cooldown = timedelta(minutes=self.limits.stacking_loss_cooldown_bars)
        disable_reason = ""
        if self.state.consecutive_stacked_losses >= self.limits.stacking_consecutive_loss_limit:
            disable_reason = "stacking_disabled_loss_cluster"
        elif sum(self.state.recent_stacked_trade_pnls) < 0:
            disable_reason = "stacking_disabled_recent_pnl"

        if disable_reason:
            self.state.stacking_disabled_until = now + cooldown
            self.state.stacking_disabled_reason = disable_reason
            self.state.stacking_disabled_events += 1
        elif self.state.stacking_disabled_until and now >= self.state.stacking_disabled_until:
            self.state.stacking_disabled_until = None
            self.state.stacking_disabled_reason = ""

    def record_api_error(self, now: datetime) -> None:
        self.state.api_error_count += 1
        if self.state.api_error_count >= self.limits.max_api_errors:
            self.arm_kill_switch("api_error_threshold", now)

    def reset_api_errors(self) -> None:
        self.state.api_error_count = 0

    def arm_kill_switch(self, reason: str, now: datetime, requires_manual_reset: bool = True) -> None:
        self.state.kill_switch = KillSwitchState(
            armed=True,
            reason=reason,
            activated_at=now,
            requires_manual_reset=requires_manual_reset,
        )
        self.state.locked = True

    def clear_kill_switch(self) -> None:
        self.state.kill_switch = KillSwitchState()
        self.state.locked = False

    def snapshot_state(self) -> dict[str, object]:
        return {
            "daily_realized_pnl": self.state.daily_realized_pnl,
            "equity_peak": self.state.equity_peak,
            "trade_count": self.state.trade_count,
            "completed_trade_count": self.state.completed_trade_count,
            "consecutive_losses": self.state.consecutive_losses,
            "cooldown_until": self.state.cooldown_until.isoformat() if self.state.cooldown_until else None,
            "last_entry_at": self.state.last_entry_at.isoformat() if self.state.last_entry_at else None,
            "last_exit_at": self.state.last_exit_at.isoformat() if self.state.last_exit_at else None,
            "last_stacked_loss_at": self.state.last_stacked_loss_at.isoformat() if self.state.last_stacked_loss_at else None,
            "last_exit_regime": self.state.last_exit_regime,
            "last_exit_trend_state": self.state.last_exit_trend_state,
            "last_exit_signal_score": self.state.last_exit_signal_score,
            "last_exit_breakout_level": self.state.last_exit_breakout_level,
            "last_exit_volume_strength": self.state.last_exit_volume_strength,
            "last_exit_side": self.state.last_exit_side,
            "last_trade_was_loss": self.state.last_trade_was_loss,
            "current_trend_key": self.state.current_trend_key,
            "entries_in_current_trend": self.state.entries_in_current_trend,
            "stacking_disabled_until": (
                self.state.stacking_disabled_until.isoformat() if self.state.stacking_disabled_until else None
            ),
            "stacking_disabled_reason": self.state.stacking_disabled_reason,
            "recent_stacked_trade_pnls": list(self.state.recent_stacked_trade_pnls),
            "consecutive_stacked_losses": self.state.consecutive_stacked_losses,
            "stacking_disabled_events": self.state.stacking_disabled_events,
            "api_error_count": self.state.api_error_count,
            "open_intents_by_symbol": dict(self.state.open_intents_by_symbol),
            "positions": {
                symbol: {
                    "symbol": position.symbol,
                    "qty": position.qty,
                    "avg_price": position.avg_price,
                    "realized_pnl": position.realized_pnl,
                    "unrealized_pnl": position.unrealized_pnl,
                    "stop_covered_qty": position.stop_covered_qty,
                    "last_updated": position.last_updated.isoformat() if position.last_updated else None,
                }
                for symbol, position in self.state.positions.items()
            },
            "open_trade_pnl_by_symbol": dict(self.state.open_trade_pnl_by_symbol),
            "locked": self.state.locked,
            "kill_switch": {
                "armed": self.state.kill_switch.armed,
                "reason": self.state.kill_switch.reason,
                "activated_at": self.state.kill_switch.activated_at.isoformat()
                if self.state.kill_switch.activated_at
                else None,
                "requires_manual_reset": self.state.kill_switch.requires_manual_reset,
            },
        }

    def restore_state(self, snapshot: dict[str, object], *, now: datetime | None = None) -> None:
        def parse_dt(value: object) -> datetime | None:
            if isinstance(value, str) and value:
                return datetime.fromisoformat(value)
            return None

        # Detect a new trading day: if the last recorded exit is from a prior
        # calendar date, the daily loss counter belongs to a previous session and
        # must not carry over.  last_exit_at is the most reliable timestamp because
        # it is written on every trade close; fall back to last_entry_at if absent.
        reference_ts_raw = snapshot.get("last_exit_at") or snapshot.get("last_entry_at")
        reference_ts = parse_dt(reference_ts_raw)
        today = (now or datetime.now()).date()
        is_new_day = reference_ts is not None and reference_ts.date() < today

        self.state.daily_realized_pnl = 0.0 if is_new_day else float(snapshot.get("daily_realized_pnl", 0.0))
        self.state.equity_peak = 0.0 if is_new_day else float(snapshot.get("equity_peak", max(self.state.daily_realized_pnl, 0.0)))
        self.state.trade_count = int(snapshot.get("trade_count", 0))
        self.state.completed_trade_count = int(snapshot.get("completed_trade_count", 0))
        self.state.consecutive_losses = int(snapshot.get("consecutive_losses", 0))
        self.state.cooldown_until = parse_dt(snapshot.get("cooldown_until"))
        self.state.last_entry_at = parse_dt(snapshot.get("last_entry_at"))
        self.state.last_exit_at = parse_dt(snapshot.get("last_exit_at"))
        self.state.last_stacked_loss_at = parse_dt(snapshot.get("last_stacked_loss_at"))
        self.state.last_exit_regime = str(snapshot.get("last_exit_regime", ""))
        self.state.last_exit_trend_state = str(snapshot.get("last_exit_trend_state", ""))
        self.state.last_exit_signal_score = float(snapshot.get("last_exit_signal_score", 0.0))
        self.state.last_exit_breakout_level = float(snapshot.get("last_exit_breakout_level", 0.0))
        self.state.last_exit_volume_strength = float(snapshot.get("last_exit_volume_strength", 0.0))
        self.state.last_exit_side = str(snapshot.get("last_exit_side", ""))
        self.state.last_trade_was_loss = bool(snapshot.get("last_trade_was_loss", False))
        self.state.current_trend_key = str(snapshot.get("current_trend_key", ""))
        self.state.entries_in_current_trend = int(snapshot.get("entries_in_current_trend", 0))
        self.state.stacking_disabled_until = parse_dt(snapshot.get("stacking_disabled_until"))
        self.state.stacking_disabled_reason = str(snapshot.get("stacking_disabled_reason", ""))
        self.state.recent_stacked_trade_pnls = [float(value) for value in list(snapshot.get("recent_stacked_trade_pnls", []))]
        self.state.consecutive_stacked_losses = int(snapshot.get("consecutive_stacked_losses", 0))
        self.state.stacking_disabled_events = int(snapshot.get("stacking_disabled_events", 0))
        self.state.api_error_count = int(snapshot.get("api_error_count", 0))
        self.state.open_intents_by_symbol = {
            str(symbol): int(count)
            for symbol, count in dict(snapshot.get("open_intents_by_symbol", {})).items()
        }
        self.state.positions = {}
        for symbol, position_data in dict(snapshot.get("positions", {})).items():
            data = dict(position_data)
            self.state.positions[str(symbol)] = PositionSnapshot(
                symbol=str(symbol),
                qty=int(data.get("qty", 0)),
                avg_price=data.get("avg_price"),
                realized_pnl=float(data.get("realized_pnl", 0.0)),
                unrealized_pnl=float(data.get("unrealized_pnl", 0.0)),
                stop_covered_qty=int(data.get("stop_covered_qty", 0)),
                last_updated=parse_dt(data.get("last_updated")),
            )
        self.state.open_trade_pnl_by_symbol = {
            str(symbol): float(value)
            for symbol, value in dict(snapshot.get("open_trade_pnl_by_symbol", {})).items()
        }
        self.state.locked = bool(snapshot.get("locked", False))
        kill_switch = dict(snapshot.get("kill_switch", {}))
        self.state.kill_switch = KillSwitchState(
            armed=bool(kill_switch.get("armed", False)),
            reason=str(kill_switch.get("reason", "")),
            activated_at=parse_dt(kill_switch.get("activated_at")),
            requires_manual_reset=bool(kill_switch.get("requires_manual_reset", False)),
        )
