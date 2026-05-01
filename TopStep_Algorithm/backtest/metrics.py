from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

from backtest.simulator import TradeResult


@dataclass
class FoldMetrics:
    total_trades: int
    winners: int
    losers: int
    win_rate: float
    total_pnl_usd: float
    profit_factor: float
    avg_winner_usd: float
    avg_loser_usd: float
    avg_r: float
    expectancy_r: float
    sharpe_ratio: float
    calmar_ratio: float
    max_drawdown_usd: float
    max_drawdown_pct: float
    tp2_count: int
    tp1_only_count: int
    sl_count: int


def _all_trading_days(start: date, end: date) -> list[date]:
    """Return all weekdays (Mon–Fri) from *start* inclusive to *end* exclusive."""
    days: list[date] = []
    cur = start
    while cur < end:
        if cur.weekday() < 5:  # 0=Mon … 4=Fri
            days.append(cur)
        cur += timedelta(days=1)
    return days


def compute_metrics(
    trades: list[TradeResult],
    test_start: date | None = None,
    test_end: date | None = None,
    account_balance: float = 50_000.0,
) -> FoldMetrics:
    """Compute performance metrics over a list of TradeResult objects.

    Parameters
    ----------
    trades:
        Completed trades to evaluate.
    test_start / test_end:
        Optional date range used for the Sharpe denominator.  When provided,
        ALL weekdays in [test_start, test_end) are counted — including zero-P&L
        days — preventing Sharpe inflation from active-only day counting.
        If omitted, only trading days with at least one trade are counted.
    account_balance:
        Starting balance used to express max drawdown as a percentage.
    """
    if not trades:
        return FoldMetrics(
            total_trades=0, winners=0, losers=0, win_rate=0.0,
            total_pnl_usd=0.0, profit_factor=0.0,
            avg_winner_usd=0.0, avg_loser_usd=0.0,
            avg_r=0.0, expectancy_r=0.0,
            sharpe_ratio=0.0, calmar_ratio=0.0,
            max_drawdown_usd=0.0, max_drawdown_pct=0.0,
            tp2_count=0, tp1_only_count=0, sl_count=0,
        )

    winners = [t for t in trades if t.pnl_usd > 0]
    losers = [t for t in trades if t.pnl_usd <= 0]
    win_rate = len(winners) / len(trades)

    gross_profit = sum(t.pnl_usd for t in winners)
    gross_loss = sum(t.pnl_usd for t in losers)  # negative
    profit_factor = (
        gross_profit / abs(gross_loss) if gross_loss != 0 else float("inf")
    )

    avg_winner = gross_profit / len(winners) if winners else 0.0
    avg_loser = gross_loss / len(losers) if losers else 0.0

    avg_r = sum(t.r_multiple for t in trades) / len(trades)
    expectancy_r = (
        win_rate * (sum(t.r_multiple for t in winners) / len(winners) if winners else 0.0)
        - (1 - win_rate) * (abs(sum(t.r_multiple for t in losers) / len(losers)) if losers else 0.0)
    )

    # Daily P&L map
    daily_pnl: dict[date, float] = {}
    for t in trades:
        daily_pnl[t.session_date] = daily_pnl.get(t.session_date, 0.0) + t.pnl_usd

    # Sharpe: use all calendar trading days in the fold window as denominator
    if test_start is not None and test_end is not None:
        all_days = _all_trading_days(test_start, test_end)
        daily_returns = [daily_pnl.get(d, 0.0) for d in all_days]
    else:
        daily_returns = list(daily_pnl.values())

    sharpe = 0.0
    if len(daily_returns) >= 2:
        mean_r = sum(daily_returns) / len(daily_returns)
        variance = sum((r - mean_r) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
        std_r = variance ** 0.5
        if std_r > 0:
            sharpe = (mean_r / std_r) * math.sqrt(252)

    # Max drawdown on cumulative P&L curve (trade-by-trade)
    cum_pnl = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in sorted(trades, key=lambda x: x.exit_time):
        cum_pnl += t.pnl_usd
        if cum_pnl > peak:
            peak = cum_pnl
        dd = cum_pnl - peak
        if dd < max_dd:
            max_dd = dd

    total_pnl = sum(t.pnl_usd for t in trades)
    max_dd_pct = max_dd / account_balance if account_balance > 0 else 0.0
    calmar = total_pnl / abs(max_dd) if max_dd != 0 else 0.0

    tp2_count = sum(1 for t in trades if t.tp2_filled)
    tp1_only_count = sum(1 for t in trades if t.tp1_filled and not t.tp2_filled)
    sl_count = sum(1 for t in trades if t.sl_filled)

    return FoldMetrics(
        total_trades=len(trades),
        winners=len(winners),
        losers=len(losers),
        win_rate=win_rate,
        total_pnl_usd=total_pnl,
        profit_factor=profit_factor,
        avg_winner_usd=avg_winner,
        avg_loser_usd=avg_loser,
        avg_r=avg_r,
        expectancy_r=expectancy_r,
        sharpe_ratio=sharpe,
        calmar_ratio=calmar,
        max_drawdown_usd=max_dd,
        max_drawdown_pct=max_dd_pct,
        tp2_count=tp2_count,
        tp1_only_count=tp1_only_count,
        sl_count=sl_count,
    )
