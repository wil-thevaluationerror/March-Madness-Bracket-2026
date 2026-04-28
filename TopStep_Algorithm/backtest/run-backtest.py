from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import available_profiles, build_config
from backtest.dashboard import build_dashboard_payload, write_dashboard
from backtest.engine import SimulatedBacktestEngine
from data_pipeline.loader import load_dbn
from data_pipeline.preprocess import preprocess, select_primary_symbol
from execution.engine import ExecutionEngine
from execution.logging import EventLogger
from execution.order_manager import OrderManager
from execution.topstepx_adapter import TopstepXAdapter
from features.indicators import add_adx, add_atr, add_ema, add_vwap
from models.orders import TradingMode
from risk.engine import RiskEngine
from strategy.rules import generate_intents

BACKTEST_TRADING_DAYS = 30


def build_execution_engine(
    run_dir: Path,
    profile: str | None = None,
    enforce_live_risk_rules: bool = False,
) -> ExecutionEngine:
    config = build_config(profile)
    config.execution.mode = TradingMode.MOCK
    config.execution.trade_log_dir = str(run_dir)
    config.execution.enforce_live_risk_rules = enforce_live_risk_rules
    risk_engine = RiskEngine(config.risk, mode="backtest", enforce_live_risk_rules=enforce_live_risk_rules)
    adapter = TopstepXAdapter(mode=config.execution.mode)
    order_manager = OrderManager()
    logger = EventLogger(config.execution.trade_log_dir)
    engine = ExecutionEngine(config, risk_engine, adapter, order_manager, logger, mode="backtest")
    engine.startup()
    return engine


def resolve_data_file(project_root: Path) -> Path:
    matches = sorted((project_root / "data").glob("*.dbn.zst"))
    if not matches:
        raise FileNotFoundError("No .dbn.zst file found under data/.")
    return matches[0]


def create_run_dir(project_root: Path, tag: str | None = None) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"-{tag}" if tag else ""
    run_dir = project_root / "output" / "backtest" / f"{stamp}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def filter_to_recent_trading_days(df, days: int):
    if df.empty or "ts_event" not in df.columns:
        return df
    ordered = df.sort_values("ts_event").copy()
    trading_dates = ordered["ts_event"].dt.tz_convert("America/Chicago").dt.date.drop_duplicates()
    selected_dates = set(trading_dates.tail(days))
    return ordered[ordered["ts_event"].dt.tz_convert("America/Chicago").dt.date.isin(selected_dates)].copy()


def filter_to_date_range(df, start: date | None, end: date | None):
    """Filter bars to [start, end] inclusive, using Chicago-local dates."""
    if df.empty or "ts_event" not in df.columns:
        return df
    local_date = df["ts_event"].dt.tz_convert("America/Chicago").dt.date
    mask = (local_date >= start if start else True) & (local_date <= end if end else True)
    return df[mask].copy()


def get_trading_dates(df) -> list[date]:
    return sorted(df["ts_event"].dt.tz_convert("America/Chicago").dt.date.drop_duplicates().tolist())


def run_single_backtest(
    df_full: pd.DataFrame,
    profile: str | None,
    run_dir: Path,
    start: date | None = None,
    end: date | None = None,
    tag: str | None = None,
    enforce_live_risk_rules: bool = False,
) -> dict:
    """Run one backtest window and write outputs.  Returns the summary dict."""
    import pandas as pd

    df = filter_to_date_range(df_full, start, end)
    if df.empty:
        print(f"[warn] No data in window {start} → {end}, skipping.")
        return {}

    trader_config = build_config(profile)
    preferred_symbol = trader_config.strategy.preferred_symbol or trader_config.strategy.instrument_root_symbol
    df, data_diagnostics = select_primary_symbol(df, preferred_symbol=preferred_symbol)
    df = add_atr(df)
    df = add_adx(df)
    df = add_vwap(df)
    df = add_ema(df)
    intents = generate_intents(df, trader_config.strategy)

    execution_engine = build_execution_engine(run_dir, profile=profile, enforce_live_risk_rules=enforce_live_risk_rules)
    backtest_engine = SimulatedBacktestEngine(
        execution_engine,
        max_concurrent_positions=execution_engine.config.risk.max_concurrent_positions,
        concurrency_levels=(1,),
    )
    result = backtest_engine.run(df, intents)
    payload = build_dashboard_payload(
        intents=intents,
        result=result,
        events_path=run_dir / "events.jsonl",
        trade_ledger_path=run_dir / "trade_ledger.jsonl",
        data_diagnostics=data_diagnostics,
    )
    write_dashboard(run_dir / "dashboard.html", payload)
    (run_dir / "summary.json").write_text(json.dumps(payload["summary"], indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "analytics.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    preset = {
        "profile": profile,
        "start_date": str(start) if start else None,
        "end_date": str(end) if end else None,
        "tag": tag,
        "base_qty": trader_config.strategy.base_qty,
        "min_entry_signal_score": trader_config.strategy.min_entry_signal_score,
        "volume_entry_filter": trader_config.strategy.volume_entry_filter,
        "use_5min_atr_for_stops": trader_config.strategy.use_5min_atr_for_stops,
        "max_position_size": trader_config.risk.max_position_size,
        "max_concurrent_positions": trader_config.risk.max_concurrent_positions,
        "internal_daily_loss_limit": trader_config.risk.internal_daily_loss_limit,
        "risk_budget_threshold": trader_config.risk.risk_budget_threshold,
        "reentry_breakout_delta_min": trader_config.risk.reentry_breakout_delta_min,
        "reentry_signal_score_min": trader_config.risk.reentry_signal_score_min,
        "drawdown_risk_tiers": list(trader_config.risk.drawdown_risk_tiers),
    }
    (run_dir / "preset.json").write_text(json.dumps(preset, indent=2, sort_keys=True), encoding="utf-8")
    return payload["summary"]


def run_walkforward(
    df_full,
    profile: str | None,
    project_root: Path,
    is_days: int = 60,
    oos_days: int = 20,
    tag: str | None = None,
    enforce_live_risk_rules: bool = False,
) -> None:
    """
    Rolling walk-forward validation.

    Splits the full data set into successive IS (in-sample) + OOS (out-of-sample)
    windows.  Each window advances by oos_days so windows do not overlap OOS periods.

    This is the primary anti-overfitting check: a strategy should perform consistently
    across all OOS windows, not just the window the parameters were tuned on.
    """
    trading_dates = get_trading_dates(df_full)
    if len(trading_dates) < is_days + oos_days:
        print(f"[warn] Not enough trading days ({len(trading_dates)}) for WFO with IS={is_days} OOS={oos_days}.")
        return

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    wfo_dir = project_root / "output" / "backtest" / f"wfo-{stamp}{'-' + tag if tag else ''}"
    wfo_dir.mkdir(parents=True, exist_ok=True)

    windows: list[tuple[date, date, date, date]] = []
    cursor = 0
    while cursor + is_days + oos_days <= len(trading_dates):
        is_start = trading_dates[cursor]
        is_end = trading_dates[cursor + is_days - 1]
        oos_start = trading_dates[cursor + is_days]
        oos_end = trading_dates[min(cursor + is_days + oos_days - 1, len(trading_dates) - 1)]
        windows.append((is_start, is_end, oos_start, oos_end))
        cursor += oos_days  # advance by OOS window size to keep OOS periods non-overlapping

    oos_summaries = []
    for i, (is_start, is_end, oos_start, oos_end) in enumerate(windows):
        window_label = f"w{i+1:02d}_oos_{oos_start}_{oos_end}"
        oos_dir = wfo_dir / window_label
        oos_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[WFO {i+1}/{len(windows)}] IS: {is_start}→{is_end}  OOS: {oos_start}→{oos_end}")
        summary = run_single_backtest(df_full, profile, oos_dir, start=oos_start, end=oos_end, tag=window_label, enforce_live_risk_rules=enforce_live_risk_rules)
        if summary:
            summary["window"] = window_label
            summary["is_start"] = str(is_start)
            summary["is_end"] = str(is_end)
            summary["oos_start"] = str(oos_start)
            summary["oos_end"] = str(oos_end)
            oos_summaries.append(summary)
            pnl = summary.get("total_pnl", 0)
            wr = summary.get("win_rate", 0)
            trades = summary.get("total_trades", 0)
            dd = summary.get("max_drawdown", 0)
            print(f"  OOS: trades={trades}  win={wr:.1%}  P&L=${pnl:.2f}  MaxDD=${dd:.2f}")

    # Aggregate OOS report
    if oos_summaries:
        total_trades = sum(s.get("total_trades", 0) for s in oos_summaries)
        total_pnl = sum(s.get("total_pnl", 0) for s in oos_summaries)
        wins = sum(s.get("total_trades", 0) * s.get("win_rate", 0) for s in oos_summaries)
        agg_win_rate = wins / total_trades if total_trades > 0 else 0.0
        profitable_windows = sum(1 for s in oos_summaries if s.get("total_pnl", 0) > 0)
        max_dds = [s.get("max_drawdown", 0) for s in oos_summaries]
        worst_dd = min(max_dds) if max_dds else 0.0

        agg = {
            "windows": len(oos_summaries),
            "profitable_windows": profitable_windows,
            "profitable_window_rate": profitable_windows / len(oos_summaries),
            "total_trades_oos": total_trades,
            "total_pnl_oos": total_pnl,
            "avg_pnl_per_window": total_pnl / len(oos_summaries),
            "aggregate_win_rate": agg_win_rate,
            "worst_oos_drawdown": worst_dd,
            "per_window": oos_summaries,
        }
        report_path = wfo_dir / "wfo_report.json"
        report_path.write_text(json.dumps(agg, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\n{'='*60}")
        print(f"WALK-FORWARD SUMMARY ({len(oos_summaries)} OOS windows)")
        print(f"  Profitable windows : {profitable_windows}/{len(oos_summaries)}  ({agg['profitable_window_rate']:.0%})")
        print(f"  Aggregate P&L      : ${total_pnl:.2f}")
        print(f"  Avg P&L / window   : ${agg['avg_pnl_per_window']:.2f}")
        print(f"  Aggregate win rate : {agg_win_rate:.1%}")
        print(f"  Worst OOS drawdown : ${worst_dd:.2f}")
        print(f"  Report             : {report_path}")
        print(f"{'='*60}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a backtest for the TopstepX strategy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 30-day single run (most recent data)
  python run-backtest.py --profile topstep-50k-express

  # Specific date range
  python run-backtest.py --start-date 2026-01-01 --end-date 2026-02-28 --profile topstep-50k-express

  # Walk-forward validation (IS=60 trading days, OOS=20 trading days)
  python run-backtest.py --walk-forward --is-days 60 --oos-days 20 --profile topstep-50k-express
        """,
    )
    parser.add_argument("--days", type=int, default=BACKTEST_TRADING_DAYS,
                        help="Number of recent trading days (used when no date range given)")
    parser.add_argument("--start-date", type=date.fromisoformat,
                        help="Start date for backtest window (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=date.fromisoformat,
                        help="End date for backtest window (YYYY-MM-DD)")
    parser.add_argument("--profile", choices=available_profiles())
    parser.add_argument("--tag")
    parser.add_argument("--walk-forward", action="store_true",
                        help="Run rolling walk-forward OOS validation instead of a single window")
    parser.add_argument("--is-days", type=int, default=60,
                        help="In-sample window size in trading days (walk-forward mode)")
    parser.add_argument("--oos-days", type=int, default=20,
                        help="Out-of-sample window size in trading days (walk-forward mode)")
    parser.add_argument("--enforce-live-rules", action="store_true",
                        help=(
                            "Apply live-mode risk guards (daily loss limit, trade cap, cooldown) "
                            "during the backtest simulation.  Use to verify the model behaves "
                            "correctly under production constraints."
                        ))
    return parser.parse_args()


def main() -> None:
    import pandas as pd
    args = parse_args()
    data_file = resolve_data_file(PROJECT_ROOT)

    trader_config = build_config(args.profile)
    df = load_dbn(data_file)
    df = preprocess(df)

    if args.walk_forward:
        run_walkforward(
            df,
            args.profile,
            PROJECT_ROOT,
            is_days=args.is_days,
            oos_days=args.oos_days,
            tag=args.tag,
            enforce_live_risk_rules=args.enforce_live_rules,
        )
        return

    # Single-window backtest
    if args.start_date or args.end_date:
        df_window = filter_to_date_range(df, args.start_date, args.end_date)
        tag = args.tag or (
            f"{args.profile or 'default'}-{args.start_date or 'start'}_{args.end_date or 'end'}"
        )
    else:
        df_window = filter_to_recent_trading_days(df, args.days)
        tag = args.tag or f"{args.profile or 'default'}-{args.days}d"

    run_dir = create_run_dir(PROJECT_ROOT, tag=tag)
    preferred_symbol = trader_config.strategy.preferred_symbol or trader_config.strategy.instrument_root_symbol
    df_window, data_diagnostics = select_primary_symbol(df_window, preferred_symbol=preferred_symbol)
    df_window = add_atr(df_window)
    df_window = add_adx(df_window)
    df_window = add_vwap(df_window)
    df_window = add_ema(df_window)
    intents = generate_intents(df_window, trader_config.strategy)

    execution_engine = build_execution_engine(run_dir, profile=args.profile, enforce_live_risk_rules=args.enforce_live_rules)
    backtest_engine = SimulatedBacktestEngine(
        execution_engine,
        max_concurrent_positions=execution_engine.config.risk.max_concurrent_positions,
        concurrency_levels=(1,),
    )
    result = backtest_engine.run(df_window, intents)
    payload = build_dashboard_payload(
        intents=intents,
        result=result,
        events_path=run_dir / "events.jsonl",
        trade_ledger_path=run_dir / "trade_ledger.jsonl",
        data_diagnostics=data_diagnostics,
    )
    write_dashboard(run_dir / "dashboard.html", payload)
    (run_dir / "summary.json").write_text(json.dumps(payload["summary"], indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "analytics.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    preset = {
        "profile": args.profile,
        "start_date": str(args.start_date) if args.start_date else None,
        "end_date": str(args.end_date) if args.end_date else None,
        "days": args.days,
        "tag": args.tag,
        "base_qty": trader_config.strategy.base_qty,
        "min_entry_signal_score": trader_config.strategy.min_entry_signal_score,
        "volume_entry_filter": trader_config.strategy.volume_entry_filter,
        "use_5min_atr_for_stops": trader_config.strategy.use_5min_atr_for_stops,
        "max_position_size": trader_config.risk.max_position_size,
        "max_concurrent_positions": trader_config.risk.max_concurrent_positions,
        "internal_daily_loss_limit": trader_config.risk.internal_daily_loss_limit,
        "risk_budget_threshold": trader_config.risk.risk_budget_threshold,
        "reentry_breakout_delta_min": trader_config.risk.reentry_breakout_delta_min,
        "reentry_signal_score_min": trader_config.risk.reentry_signal_score_min,
        "drawdown_risk_tiers": list(trader_config.risk.drawdown_risk_tiers),
    }
    (run_dir / "preset.json").write_text(json.dumps(preset, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Dashboard written to {run_dir / 'dashboard.html'}")
    print(result)


if __name__ == "__main__":
    main()
