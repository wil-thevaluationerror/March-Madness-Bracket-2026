from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
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
from features.indicators import add_atr, add_ema, add_vwap
from models.orders import TradingMode
from risk.engine import RiskEngine
from strategy.rules import generate_intents

BACKTEST_TRADING_DAYS = 5


def build_execution_engine(run_dir: Path, profile: str | None = None) -> ExecutionEngine:
    config = build_config(profile)
    config.execution.mode = TradingMode.MOCK
    config.execution.trade_log_dir = str(run_dir)
    risk_engine = RiskEngine(config.risk, mode="backtest")
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


def filter_to_recent_trading_week(df, days: int):
    if df.empty or "ts_event" not in df.columns:
        return df
    ordered = df.sort_values("ts_event").copy()
    trading_dates = ordered["ts_event"].dt.tz_convert("America/Chicago").dt.date.drop_duplicates()
    selected_dates = set(trading_dates.tail(days))
    return ordered[ordered["ts_event"].dt.tz_convert("America/Chicago").dt.date.isin(selected_dates)].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a backtest for the TopstepX strategy.")
    parser.add_argument("--days", type=int, default=BACKTEST_TRADING_DAYS)
    parser.add_argument("--profile", choices=available_profiles())
    parser.add_argument("--tag")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = create_run_dir(PROJECT_ROOT, tag=args.tag)
    data_file = resolve_data_file(PROJECT_ROOT)
    trader_config = build_config(args.profile)
    df = load_dbn(data_file)
    df = preprocess(df)
    df = filter_to_recent_trading_week(df, args.days)
    preferred_symbol = trader_config.strategy.preferred_symbol or trader_config.strategy.instrument_root_symbol
    df, data_diagnostics = select_primary_symbol(df, preferred_symbol=preferred_symbol)
    df = add_atr(df)
    df = add_vwap(df)
    df = add_ema(df)
    intents = generate_intents(df, trader_config.strategy)

    execution_engine = build_execution_engine(run_dir, profile=args.profile)
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
    (run_dir / "concurrency-analysis.json").write_text(
        json.dumps(
            {
                "summary": payload["summary"],
                "concurrency_analysis": payload["concurrency_analysis"],
                "position_constraint_analysis": payload["position_constraint_analysis"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (run_dir / "preset.json").write_text(
        json.dumps(
            {
                "profile": args.profile,
                "days": args.days,
                "base_qty": trader_config.strategy.base_qty,
                "max_position_size": trader_config.risk.max_position_size,
                "max_concurrent_positions": trader_config.risk.max_concurrent_positions,
                "internal_daily_loss_limit": trader_config.risk.internal_daily_loss_limit,
                "risk_budget_threshold": trader_config.risk.risk_budget_threshold,
                "reentry_breakout_delta_min": trader_config.risk.reentry_breakout_delta_min,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"Dashboard written to {run_dir / 'dashboard.html'}")
    print(result)


if __name__ == "__main__":
    main()
