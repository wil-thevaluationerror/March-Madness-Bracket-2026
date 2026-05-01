from __future__ import annotations

import csv
import os
from dataclasses import asdict
from datetime import date

from trading_system.backtest.metrics import FoldMetrics
from trading_system.backtest.raw_setup_ledger import write_raw_setup_csv, write_rejection_summary
from trading_system.backtest.signal_ledger import write_candidate_csv
from trading_system.backtest.simulator import TradeResult
from trading_system.backtest.walk_forward import Fold, WalkForwardResult


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _fmt(v: float, decimals: int = 2) -> str:
    return f"{v:,.{decimals}f}"


def print_results(result: WalkForwardResult, daily_loss_limit_usd: float = 750.0) -> None:
    """Print fold-by-fold and combined OOS stats to stdout."""
    sep = "─" * 110

    print()
    print("=" * 110)
    print("  LONDON SWEEP BACKTEST  —  Walk-Forward Results")
    print("=" * 110)

    # Per-fold table
    header = (
        f"{'Fold':>4}  {'Test Window':>23}  {'Trades':>6}  {'Win%':>6}  "
        f"{'P&L':>10}  {'PF':>6}  {'Sharpe':>7}  {'MaxDD':>10}"
    )
    print(header)
    print(sep)

    for fold in result.folds:
        m = fold.test_metrics
        window = f"{fold.test_start} → {fold.test_end}"
        pf = f"{m.profit_factor:.2f}" if m.profit_factor != float("inf") else "∞"
        print(
            f"{fold.fold_idx:>4}  {window:>23}  {m.total_trades:>6}  "
            f"{_pct(m.win_rate):>6}  {_fmt(m.total_pnl_usd):>10}  {pf:>6}  "
            f"{_fmt(m.sharpe_ratio):>7}  {_fmt(m.max_drawdown_usd):>10}"
        )

    print(sep)

    # Combined OOS block
    cm = result.combined_oos_metrics
    pf = f"{cm.profit_factor:.2f}" if cm.profit_factor != float("inf") else "∞"
    print()
    print("COMBINED OUT-OF-SAMPLE STATS")
    print(f"  Total trades     : {cm.total_trades}")
    print(f"  Winners / Losers : {cm.winners} / {cm.losers}")
    print(f"  Win rate         : {_pct(cm.win_rate)}")
    print(f"  Total P&L        : ${_fmt(cm.total_pnl_usd)}")
    print(f"  Profit factor    : {pf}")
    print(f"  Avg winner       : ${_fmt(cm.avg_winner_usd)}")
    print(f"  Avg loser        : ${_fmt(cm.avg_loser_usd)}")
    print(f"  Avg R            : {_fmt(cm.avg_r, 3)}")
    print(f"  Expectancy R     : {_fmt(cm.expectancy_r, 3)}")
    print(f"  Sharpe (ann.)    : {_fmt(cm.sharpe_ratio)}")
    print(f"  Calmar           : {_fmt(cm.calmar_ratio)}")
    print(f"  Trade-equity DD  : ${_fmt(cm.max_drawdown_usd)}  ({_pct(cm.max_drawdown_pct)})")
    print(f"  TP2 fills        : {cm.tp2_count}")
    print(f"  TP1-only         : {cm.tp1_only_count}")
    print(f"  SL hits          : {cm.sl_count}")

    final_fold = result.folds[-1] if result.folds else None
    if final_fold is not None:
        cushion = final_fold.final_account_pnl - final_fold.max_loss_threshold
        breached = "YES" if any(f.max_loss_breached for f in result.folds) else "NO"
        print()
        print("TOPSTEP 50K RULE CHECK")
        print("  Daily Loss Limit : $1,000")
        print("  Max Loss Limit   : $2,000 from highest end-of-day P&L, capped at $0")
        print(f"  Final account P&L: ${_fmt(final_fold.final_account_pnl)}")
        print(f"  Max-loss floor   : ${_fmt(final_fold.max_loss_threshold)}")
        print(f"  Cushion to floor : ${_fmt(cushion)}")
        print(f"  Max loss breached: {breached}")

    # Correlated exposure block
    _print_correlated_exposure(result.all_oos_trades, daily_loss_limit_usd)
    print()


def _print_correlated_exposure(trades: list[TradeResult], daily_loss_limit_usd: float) -> None:
    """Group OOS trades by session_date and analyse multi-symbol correlation."""
    if not trades:
        return

    # Group by date
    by_date: dict[date, list[TradeResult]] = {}
    for t in trades:
        by_date.setdefault(t.session_date, []).append(t)

    total_days = len(by_date)
    multi_symbol_days = sum(
        1 for ts in by_date.values()
        if len({t.symbol for t in ts}) >= 2
    )
    same_direction_days = sum(
        1 for ts in by_date.values()
        if len({t.symbol for t in ts}) >= 2
        and len({t.direction for t in ts}) == 1
    )
    daily_pnls = [sum(t.pnl_usd for t in ts) for ts in by_date.values()]
    avg_daily = sum(daily_pnls) / len(daily_pnls)
    best_daily = max(daily_pnls)
    worst_daily = min(daily_pnls)
    breached_days = sum(1 for p in daily_pnls if p <= -daily_loss_limit_usd)

    print()
    print("CORRELATED EXPOSURE ANALYSIS")
    print(f"  Total trading days           : {total_days}")
    print(f"  Multi-symbol days (≥2 syms)  : {multi_symbol_days}")
    print(f"  Same-direction days          : {same_direction_days}")
    print(f"  Avg daily portfolio P&L      : ${_fmt(avg_daily)}")
    print(f"  Best single day              : ${_fmt(best_daily)}")
    print(f"  Worst single day             : ${_fmt(worst_daily)}")
    print(f"  Days breaching ${daily_loss_limit_usd:,.0f} limit   : {breached_days}")


def write_csv(result: WalkForwardResult, output_dir: str) -> None:
    """Write trades.csv and folds.csv to *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)

    # trades.csv
    trades_path = os.path.join(output_dir, "trades.csv")
    if result.all_oos_trades:
        fields = list(TradeResult.__dataclass_fields__.keys())
        with open(trades_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for t in result.all_oos_trades:
                writer.writerow(asdict(t))
        print(f"Trades written to {trades_path} ({len(result.all_oos_trades)} rows)")

    # folds.csv
    folds_path = os.path.join(output_dir, "folds.csv")
    fold_fields = [
        "fold_idx", "test_start", "test_end",
        "total_trades", "win_rate", "total_pnl_usd", "profit_factor",
        "sharpe_ratio", "calmar_ratio", "max_drawdown_usd",
        "tp2_count", "tp1_only_count", "sl_count",
        "final_account_pnl", "max_loss_threshold", "max_loss_breached",
    ]
    with open(folds_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fold_fields)
        writer.writeheader()
        for fold in result.folds:
            m = fold.test_metrics
            writer.writerow({
                "fold_idx": fold.fold_idx,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "total_trades": m.total_trades,
                "win_rate": round(m.win_rate, 4),
                "total_pnl_usd": round(m.total_pnl_usd, 2),
                "profit_factor": round(m.profit_factor, 4) if m.profit_factor != float("inf") else "inf",
                "sharpe_ratio": round(m.sharpe_ratio, 4),
                "calmar_ratio": round(m.calmar_ratio, 4),
                "max_drawdown_usd": round(m.max_drawdown_usd, 2),
                "tp2_count": m.tp2_count,
                "tp1_only_count": m.tp1_only_count,
                "sl_count": m.sl_count,
                "final_account_pnl": round(fold.final_account_pnl, 2),
                "max_loss_threshold": round(fold.max_loss_threshold, 2),
                "max_loss_breached": fold.max_loss_breached,
            })
    print(f"Folds written to {folds_path} ({len(result.folds)} rows)")

    ledger_path = os.path.join(output_dir, "candidate_signal_ledger.csv")
    write_candidate_csv(result.signal_ledger, ledger_path)
    print(f"Candidate signal ledger written to {ledger_path} ({len(result.signal_ledger)} rows)")

    raw_ledger_path = os.path.join(output_dir, "raw_setup_ledger.csv")
    write_raw_setup_csv(result.raw_setup_ledger, raw_ledger_path)
    print(f"Raw setup ledger written to {raw_ledger_path} ({len(result.raw_setup_ledger)} rows)")

    rejection_path = os.path.join(output_dir, "setup_rejection_summary.csv")
    write_rejection_summary(result.raw_setup_ledger, rejection_path)
    print(f"Setup rejection summary written to {rejection_path}")
