from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

from dateutil.relativedelta import relativedelta

from trading_system.backtest.data_loader import SessionDay
from trading_system.backtest.metrics import FoldMetrics, compute_metrics
from trading_system.backtest.raw_setup_ledger import RawSetupEvent, with_account_rejection
from trading_system.backtest.signal_ledger import CandidateSignalRecord, record_from_trade
from trading_system.backtest.simulator import SessionSimulator, SimulatorConfig, TradeResult
from trading_system.config import INSTRUMENTS

log = logging.getLogger(__name__)


@dataclass
class Fold:
    fold_idx: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    train_metrics: FoldMetrics
    test_metrics: FoldMetrics
    test_trades: list[TradeResult] = field(default_factory=list)
    max_loss_breached: bool = False
    final_account_pnl: float = 0.0
    max_loss_threshold: float = -2_000.0
    signal_ledger: list[CandidateSignalRecord] = field(default_factory=list)
    raw_setup_ledger: list[RawSetupEvent] = field(default_factory=list)


@dataclass
class WalkForwardResult:
    folds: list[Fold]
    all_oos_trades: list[TradeResult]
    combined_oos_metrics: FoldMetrics
    signal_ledger: list[CandidateSignalRecord] = field(default_factory=list)
    raw_setup_ledger: list[RawSetupEvent] = field(default_factory=list)


@dataclass
class _TopstepAccountState:
    cumulative_pnl: float = 0.0
    highest_end_of_day_pnl: float = 0.0
    max_loss_breached: bool = False

    def max_loss_threshold(self, max_loss_limit_usd: float) -> float:
        return min(0.0, self.highest_end_of_day_pnl - abs(max_loss_limit_usd))

    def record_trade(self, trade: TradeResult, max_loss_limit_usd: float) -> bool:
        self.cumulative_pnl += trade.pnl_usd
        if self.cumulative_pnl <= self.max_loss_threshold(max_loss_limit_usd):
            self.max_loss_breached = True
            return False
        return True

    def mark_end_of_day(self) -> None:
        self.highest_end_of_day_pnl = max(
            self.highest_end_of_day_pnl,
            self.cumulative_pnl,
        )


def _initial_risk_usd(trade: TradeResult) -> float:
    instrument = INSTRUMENTS[trade.symbol]
    stop_distance = abs(trade.entry_price - trade.stop_price)
    ticks_at_risk = stop_distance / instrument.tick_size
    return ticks_at_risk * instrument.tick_value * trade.contracts


def _mark_raw_account_rejection(
    events: list[RawSetupEvent],
    trade: TradeResult,
    rejection_reason: str,
) -> None:
    if not trade.raw_setup_event_id:
        return
    for idx, event in enumerate(events):
        if event.event_id == trade.raw_setup_event_id:
            events[idx] = with_account_rejection(event, rejection_reason=rejection_reason)
            return


class WalkForward:
    """Rolling walk-forward orchestrator.

    Sessions are supplied as ``{symbol: {date: SessionDay}}``.
    Each fold trains on ``train_months`` of data and tests on the next
    ``test_months``.  The window slides by ``test_months`` each iteration.

    Portfolio daily loss gate
    -------------------------
    Within each test fold, sessions are processed date-by-date across all
    symbols (sorted alphabetically for reproducibility).  Once the combined
    P&L for a given day reaches or exceeds the daily loss limit, remaining
    symbol sessions for that date are skipped — mirroring the TopStep combine
    intraday loss rule.
    """

    def __init__(
        self,
        sessions: dict[str, dict[date, SessionDay]],
        sim_config: SimulatorConfig | None = None,
        train_months: int = 6,
        test_months: int = 1,
    ) -> None:
        self._sessions = sessions
        self._sim_config = sim_config or SimulatorConfig()
        self._sim = SessionSimulator(self._sim_config)
        self._train_months = train_months
        self._test_months = test_months
        self._daily_loss_limit = self._sim_config.daily_loss_limit_usd

    def _all_dates(self) -> list[date]:
        dates: set[date] = set()
        for sym_sessions in self._sessions.values():
            dates.update(sym_sessions.keys())
        return sorted(dates)

    def _sessions_in_range(
        self, start: date, end: date
    ) -> dict[str, dict[date, SessionDay]]:
        return {
            sym: {d: s for d, s in sym_sessions.items() if start <= d < end}
            for sym, sym_sessions in self._sessions.items()
        }

    def _run_fold(
        self,
        train_sessions: dict[str, dict[date, SessionDay]],
        test_sessions: dict[str, dict[date, SessionDay]],
        test_start: date,
        test_end: date,
        fold_idx: int,
        account: _TopstepAccountState | None = None,
    ) -> Fold:
        # Train trades (no daily gate — used only for diagnostics)
        train_trades: list[TradeResult] = []
        for sym_sessions in train_sessions.values():
            for session in sym_sessions.values():
                train_trades.extend(self._sim.run(session))

        # Test trades — with Topstep 50K account rules.
        test_trades: list[TradeResult] = []
        all_test_dates = sorted(
            set(d for sym_sessions in test_sessions.values() for d in sym_sessions)
        )
        blocked_sessions = 0
        blocked_trades = 0
        account = account or _TopstepAccountState()
        signal_ledger: list[CandidateSignalRecord] = []
        raw_setup_ledger: list[RawSetupEvent] = []

        for d in all_test_dates:
            if account.max_loss_breached:
                break
            daily_pnl = 0.0
            candidate_trades: list[TradeResult] = []
            for sym in sorted(test_sessions.keys()):
                if d not in test_sessions[sym]:
                    continue
                if daily_pnl <= -self._daily_loss_limit:
                    log.debug(
                        "daily_loss_gate blocked sym=%s date=%s daily_pnl=%.2f",
                        sym, d, daily_pnl,
                    )
                    blocked_sessions += 1
                    continue
                day_trades = self._sim.run(test_sessions[sym][d])
                raw_setup_ledger.extend(getattr(self._sim, "last_raw_setup_events", []))
                candidate_trades.extend(day_trades)

            for trade in sorted(candidate_trades, key=lambda t: (t.entry_time, t.exit_time, t.symbol)):
                risk_usd = _initial_risk_usd(trade)
                if daily_pnl <= -self._daily_loss_limit:
                    blocked_trades += 1
                    signal_ledger.append(
                        record_from_trade(
                            trade,
                            accepted=False,
                            rejection_reason="daily_loss_limit_reached",
                            daily_loss_gate_passed=False,
                        )
                    )
                    _mark_raw_account_rejection(
                        raw_setup_ledger, trade, "daily_loss_limit_reached"
                    )
                    continue
                if daily_pnl - risk_usd <= -self._daily_loss_limit:
                    blocked_trades += 1
                    signal_ledger.append(
                        record_from_trade(
                            trade,
                            accepted=False,
                            rejection_reason="daily_loss_risk_exceeded",
                            daily_loss_gate_passed=False,
                        )
                    )
                    _mark_raw_account_rejection(
                        raw_setup_ledger, trade, "daily_loss_risk_exceeded"
                    )
                    continue
                if (
                    account.cumulative_pnl - risk_usd
                    <= account.max_loss_threshold(self._sim_config.max_loss_limit_usd)
                ):
                    blocked_trades += 1
                    signal_ledger.append(
                        record_from_trade(
                            trade,
                            accepted=False,
                            rejection_reason="max_loss_risk_exceeded",
                            max_loss_gate_passed=False,
                        )
                    )
                    _mark_raw_account_rejection(
                        raw_setup_ledger, trade, "max_loss_risk_exceeded"
                    )
                    continue
                signal_ledger.append(record_from_trade(trade, accepted=True))
                test_trades.append(trade)
                daily_pnl += trade.pnl_usd
                if not account.record_trade(trade, self._sim_config.max_loss_limit_usd):
                    log.info(
                        "Fold %d: Topstep max loss breached date=%s pnl=%.2f threshold=%.2f",
                        fold_idx,
                        d,
                        account.cumulative_pnl,
                        account.max_loss_threshold(self._sim_config.max_loss_limit_usd),
                    )
                    break

            account.mark_end_of_day()

        if blocked_sessions:
            log.info("Fold %d: %d sessions blocked by daily loss gate", fold_idx, blocked_sessions)
        if blocked_trades:
            log.info("Fold %d: %d trades blocked by Topstep rules", fold_idx, blocked_trades)

        train_metrics = compute_metrics(
            train_trades,
            account_balance=self._sim_config.account_balance,
        )
        test_metrics = compute_metrics(
            test_trades,
            test_start=test_start,
            test_end=test_end,
            account_balance=self._sim_config.account_balance,
        )

        return Fold(
            fold_idx=fold_idx,
            train_start=min(
                (d for sym in train_sessions.values() for d in sym), default=test_start
            ),
            train_end=test_start,
            test_start=test_start,
            test_end=test_end,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            test_trades=test_trades,
            max_loss_breached=account.max_loss_breached,
            final_account_pnl=account.cumulative_pnl,
            max_loss_threshold=account.max_loss_threshold(self._sim_config.max_loss_limit_usd),
            signal_ledger=signal_ledger,
            raw_setup_ledger=raw_setup_ledger,
        )

    def run(self) -> WalkForwardResult:
        all_dates = self._all_dates()
        if not all_dates:
            raise ValueError("No sessions available for walk-forward.")

        first_date = all_dates[0]
        last_date = all_dates[-1]

        folds: list[Fold] = []
        fold_idx = 0
        train_start = first_date
        oos_account = _TopstepAccountState()

        while True:
            train_end = train_start + relativedelta(months=self._train_months)
            test_start = train_end
            test_end = test_start + relativedelta(months=self._test_months)

            if test_end > last_date:
                break

            log.info(
                "Fold %d  train=[%s, %s)  test=[%s, %s)",
                fold_idx, train_start, train_end, test_start, test_end,
            )

            train_sessions = self._sessions_in_range(train_start, train_end)
            test_sessions = self._sessions_in_range(test_start, test_end)

            fold = self._run_fold(
                train_sessions,
                test_sessions,
                test_start,
                test_end,
                fold_idx,
                oos_account,
            )
            folds.append(fold)

            fold_idx += 1
            train_start += relativedelta(months=self._test_months)
            if oos_account.max_loss_breached:
                break

        if not folds:
            raise ValueError(
                f"No complete folds generated. Need at least "
                f"{self._train_months + self._test_months} months of data."
            )

        all_oos_trades = [t for fold in folds for t in fold.test_trades]
        combined_metrics = compute_metrics(
            all_oos_trades,
            test_start=folds[0].test_start,
            test_end=folds[-1].test_end,
            account_balance=self._sim_config.account_balance,
        )

        return WalkForwardResult(
            folds=folds,
            all_oos_trades=all_oos_trades,
            combined_oos_metrics=combined_metrics,
            signal_ledger=[r for fold in folds for r in fold.signal_ledger],
            raw_setup_ledger=[r for fold in folds for r in fold.raw_setup_ledger],
        )
