from __future__ import annotations

import argparse
import fcntl
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env file if present (no-op when python-dotenv is unavailable or file is absent)
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from config import available_profiles, build_config
from data_pipeline.live_feed import TopstepLiveFeed
from execution.engine import ExecutionEngine
from execution.logging import EventLogger
from execution.order_manager import OrderManager
from execution.topstepx_adapter import TopstepXAdapter
from models.orders import Regime, TradingMode
from risk.engine import RiskEngine
from strategy.diagnostics import StrategyDiagnosticsLogger
from strategy.rules import SignalInput, build_order_intent

_LIVE_BLOCKED_SYMBOLS = frozenset({"6B", "6E"})


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _lock_name(mode: TradingMode, profile: str | None) -> str:
    profile_name = profile or "default"
    safe_profile = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in profile_name)
    return f"run_trader_{mode.value}_{safe_profile}.lock"


def _acquire_runtime_lock(mode: TradingMode, profile: str | None):
    lock_dir = PROJECT_ROOT / "runtime_logs"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / _lock_name(mode, profile)
    lock_file = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit(f"Another trader is already running for mode={mode.value} profile={profile or 'default'}") from exc
    lock_file.write(f"pid={os.getpid()}\n")
    lock_file.flush()
    return lock_file


def _inject_credentials_from_env(config) -> None:
    """Overwrite TopstepConnectionConfig fields from environment variables when set."""
    mapping = {
        "TOPSTEP_USERNAME": "username",
        "TOPSTEP_API_KEY": "api_key",
        "TOPSTEP_ACCOUNT_ID": "account_id",
        "TOPSTEP_ENVIRONMENT": "environment",
        "TOPSTEP_API_BASE_URL": "api_base_url",
        "TOPSTEP_WEBSOCKET_URL": "websocket_url",
    }
    for env_var, field in mapping.items():
        value = os.environ.get(env_var, "").strip()
        if value:
            setattr(config.execution.topstep, field, value)


def build_runtime(mode: TradingMode, profile: str | None = None) -> ExecutionEngine:
    config = build_config(profile)
    config.execution.mode = mode
    selected_symbol = config.strategy.preferred_symbol or config.strategy.default_symbol
    if mode == TradingMode.LIVE and selected_symbol in _LIVE_BLOCKED_SYMBOLS:
        raise SystemExit("6B/6E live execution is not verified; use paper/mock only.")
    _inject_credentials_from_env(config)
    risk_engine = RiskEngine(config.risk)
    adapter = TopstepXAdapter(mode=mode, config=config.execution.topstep)
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


def _build_live_feed(
    engine: ExecutionEngine,
    diagnostics_logger: StrategyDiagnosticsLogger | None = None,
) -> TopstepLiveFeed | None:
    """
    Construct a TopstepLiveFeed for PAPER/LIVE modes.

    Resolves the front MES contract from the adapter's contract cache
    (populated during startup) and shares the adapter's JWT token so
    the feed never needs to authenticate independently.
    """
    from execution.topstep_live_adapter import LiveTopstepAdapter

    adapter = engine.adapter
    # Unwrap TopstepXAdapter router to get the underlying LiveTopstepAdapter
    inner = getattr(adapter, "_impl", None)
    if not isinstance(inner, LiveTopstepAdapter):
        return None

    # Reuse the contract the adapter already resolved for MES
    symbol = engine.config.strategy.preferred_symbol or engine.config.strategy.default_symbol
    contract = inner.contract_cache_by_symbol.get(symbol)
    if contract is None:
        # Force resolution so the cache is populated
        try:
            contract = inner._resolve_contract(symbol)
        except RuntimeError:
            logging.getLogger(__name__).warning("live_feed_contract_not_found symbol=%s", symbol)
            return None

    contract_id = str(contract["id"])
    token_provider = lambda: inner.access_token  # noqa: E731

    feed = TopstepLiveFeed(
        config=inner.config,
        token_provider=token_provider,
        contract_id=contract_id,
        symbol=symbol,
        diagnostics_callback=diagnostics_logger.write if diagnostics_logger is not None else None,
        risk_allowed_provider=lambda: engine.risk_engine.can_trade(datetime.now(UTC))[0],
    )
    if not feed.initialize():
        logging.getLogger(__name__).warning("live_feed_init_failed no bars returned (market closed?)")
        # Return the feed anyway; tick() will retry lazily each heartbeat
    return feed


def _log_runtime_config(engine: ExecutionEngine, mode: TradingMode, profile: str | None) -> None:
    log = logging.getLogger(__name__)
    cfg = engine.config
    symbol = cfg.strategy.preferred_symbol or cfg.strategy.default_symbol
    log.info(
        "runtime_startup mode=%s profile=%s symbol=%s base_qty=%d",
        mode.value,
        profile or "default",
        symbol,
        cfg.strategy.base_qty,
    )
    for window in cfg.session.session_windows:
        log.info(
            "session_window label=%s market_open=%s no_new_trades_after=%s force_flatten_at=%s tz=%s",
            window.label,
            window.market_open.strftime("%H:%M"),
            window.no_new_trades_after.strftime("%H:%M"),
            window.force_flatten_at.strftime("%H:%M"),
            cfg.session.timezone,
        )


def run_loop(
    mode: TradingMode,
    poll_seconds: int,
    profile: str | None = None,
    *,
    diagnostics_enabled: bool = False,
) -> None:
    if diagnostics_enabled and mode == TradingMode.LIVE:
        raise SystemExit("Strategy diagnostics are disabled for live mode; use paper/mock only.")
    engine = build_runtime(mode, profile=profile)
    if not engine.startup():
        raise SystemExit("Startup failed: broker state not clean.")

    _log_runtime_config(engine, mode, profile)

    diagnostics_logger: StrategyDiagnosticsLogger | None = None
    if diagnostics_enabled and mode in (TradingMode.MOCK, TradingMode.PAPER):
        diagnostics_path = Path(engine.config.execution.trade_log_dir) / "strategy_diagnostics.jsonl"
        diagnostics_logger = StrategyDiagnosticsLogger(diagnostics_path)
        logging.getLogger(__name__).info("strategy_diagnostics_enabled path=%s", diagnostics_path)

    feed: TopstepLiveFeed | None = None
    if mode in (TradingMode.PAPER, TradingMode.LIVE):
        feed = _build_live_feed(engine, diagnostics_logger=diagnostics_logger)

    try:
        while True:
            if feed is not None:
                intents = feed.tick(engine.config.strategy)
                for intent in intents:
                    engine.submit_intent(intent)
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
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG-level logging (shows API wire traffic).")
    parser.add_argument("--diagnostics", action="store_true", help="Write paper/mock strategy diagnostics JSONL.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = TradingMode(args.mode)
    lock_file = _acquire_runtime_lock(mode, args.profile) if mode != TradingMode.MOCK else None
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    diagnostics_enabled = args.diagnostics or _env_flag("STRATEGY_DIAGNOSTICS_ENABLED")
    try:
        if args.once:
            if diagnostics_enabled and mode == TradingMode.LIVE:
                raise SystemExit("Strategy diagnostics are disabled for live mode; use paper/mock only.")
            run_once(mode, profile=args.profile)
        else:
            run_loop(mode, args.poll_seconds, profile=args.profile, diagnostics_enabled=diagnostics_enabled)
    finally:
        if lock_file is not None:
            lock_file.close()


if __name__ == "__main__":
    main()
