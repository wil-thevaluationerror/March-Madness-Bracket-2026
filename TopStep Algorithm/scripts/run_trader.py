from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import available_profiles, build_config
from execution.engine import ExecutionEngine
from execution.logging import EventLogger
from execution.order_manager import OrderManager
from execution.topstepx_adapter import TopstepXAdapter
from models.orders import Regime, TradingMode
from risk.engine import RiskEngine
from strategy.rules import SignalInput, build_order_intent


def build_runtime(mode: TradingMode, profile: str | None = None) -> ExecutionEngine:
    config = build_config(profile)
    config.execution.mode = mode
    risk_engine = RiskEngine(config.risk)
    adapter = TopstepXAdapter(mode=mode)
    order_manager = OrderManager()
    logger = EventLogger(config.execution.trade_log_dir)
    return ExecutionEngine(config, risk_engine, adapter, order_manager, logger)


def run_once(mode: TradingMode, profile: str | None = None) -> None:
    engine = build_runtime(mode, profile=profile)
    now = datetime.now(UTC)
    if not engine.startup(now=now):
        raise SystemExit("Startup failed: broker state not clean.")

    signal = SignalInput(
        symbol="MES",
        timestamp=now,
        regime=Regime.TREND_EXPANSION,
        long_signal=True,
        short_signal=False,
        entry_price=5200.0,
        stop_price=5195.0,
        target_price=5208.0,
        signal_score=1.0,
        qty=engine.config.strategy.base_qty,
    )
    intent = build_order_intent(signal, engine.config.strategy)
    if intent is not None:
        engine.submit_intent(intent, now=now)
        engine.drain_adapter_events()
    engine.heartbeat(now=now)
    engine.safe_shutdown()


def run_loop(mode: TradingMode, poll_seconds: int, profile: str | None = None) -> None:
    engine = build_runtime(mode, profile=profile)
    if not engine.startup():
        raise SystemExit("Startup failed: broker state not clean.")
    try:
        while True:
            engine.heartbeat()
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        engine.safe_shutdown(reason="keyboard_interrupt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TopstepX trading runtime.")
    parser.add_argument("--mode", choices=[mode.value for mode in TradingMode], default=TradingMode.MOCK.value)
    parser.add_argument("--once", action="store_true", help="Run a single demo cycle and exit.")
    parser.add_argument("--poll-seconds", type=int, default=3)
    parser.add_argument("--profile", choices=available_profiles())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = TradingMode(args.mode)
    if args.once:
        run_once(mode, profile=args.profile)
    else:
        run_loop(mode, args.poll_seconds, profile=args.profile)


if __name__ == "__main__":
    main()
