"""London Session Liquidity Sweep — Walk-Forward Backtest CLI.

Usage
-----
Run from the repo root with PYTHONPATH set:

    PYTHONPATH=. python backtest/run_backtest.py \\
        --data-1m  data/6B_ohlcv-1m.dbn.zst  data/6E_ohlcv-1m.dbn.zst \\
        --data-1h  data/6B_ohlcv-1h.dbn.zst  data/6E_ohlcv-1h.dbn.zst \\
        --symbols  6B 6E \\
        --balance  50000 \\
        --output   results

To backtest with a specific calibrated profile (recommended):

    PYTHONPATH=. python backtest/run_backtest.py \\
        --data-1m  data/6B_ohlcv-1m.dbn.zst \\
        --symbols  6B \\
        --profile  topstep-50k-express-london-6b-paper \\
        --output   results
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure repo root is on the path when run as a script
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config import INSTRUMENTS
from backtest.data_loader import DataLoader
from backtest.reporter import print_results, write_csv
from backtest.simulator import SimulatorConfig
from backtest.walk_forward import WalkForward
from profiles import available_profiles, build_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="London Session Liquidity Sweep — Walk-Forward Backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-1m",
        nargs="+",
        required=True,
        metavar="PATH",
        help="One .dbn.zst 1-minute bar file per symbol (same order as --symbols).",
    )
    parser.add_argument(
        "--data-1h",
        nargs="+",
        default=None,
        metavar="PATH",
        help="Optional 1-hour bar files (same order as --symbols). "
             "If omitted, 1h bars are resampled from 1m.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        metavar="SYM",
        help="Instrument symbols matching the order of --data-1m (e.g. 6B 6E).",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=50_000.0,
        metavar="USD",
        help="Starting account balance in USD (default: 50000).",
    )
    parser.add_argument(
        "--profile",
        default=None,
        metavar="PROFILE",
        help=(
            "Trading profile name — loads calibrated strategy parameters. "
            f"Available: {', '.join(available_profiles())}. "
            "When omitted, conservative defaults are used."
        ),
    )
    parser.add_argument(
        "--train-months",
        type=int,
        default=6,
        metavar="N",
        help="Training window length in months (default: 6).",
    )
    parser.add_argument(
        "--test-months",
        type=int,
        default=1,
        metavar="N",
        help="Test window length in months / fold step (default: 1).",
    )
    parser.add_argument(
        "--output",
        default="results",
        metavar="DIR",
        help="Directory for CSV output (default: results).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


def _validate(args: argparse.Namespace) -> None:
    errors: list[str] = []

    if len(args.data_1m) != len(args.symbols):
        errors.append(
            f"--data-1m has {len(args.data_1m)} paths but --symbols has "
            f"{len(args.symbols)} entries — counts must match."
        )

    if args.data_1h is not None and len(args.data_1h) != len(args.symbols):
        errors.append(
            f"--data-1h has {len(args.data_1h)} paths but --symbols has "
            f"{len(args.symbols)} entries — counts must match."
        )

    for path in args.data_1m:
        if not os.path.exists(path):
            errors.append(f"1m data file not found: {path}")

    if args.data_1h:
        for path in args.data_1h:
            if not os.path.exists(path):
                errors.append(f"1h data file not found: {path}")

    for sym in args.symbols:
        if sym not in INSTRUMENTS:
            errors.append(
                f"Symbol '{sym}' is not in INSTRUMENTS. "
                f"Available: {sorted(INSTRUMENTS)}"
            )

    if args.profile is not None and args.profile not in available_profiles():
        errors.append(
            f"Unknown profile '{args.profile}'. "
            f"Available: {', '.join(available_profiles())}"
        )

    if errors:
        for err in errors:
            print(f"[ERROR] {err}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    args = _parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    _validate(args)

    # ── Build SimulatorConfig from profile (or use defaults) ─────────────────
    if args.profile is not None:
        trader_config = build_config(args.profile)
        sim_config = SimulatorConfig.from_trader_config(
            trader_config, account_balance=args.balance
        )
        logging.getLogger(__name__).info(
            "Profile '%s' loaded: ADX≥%.0f  ATR_pct≥%.2f  EMA_persist=%d bars  "
            "BE_trigger=%.2f  TP_mult=%.1f",
            args.profile,
            sim_config.adx_min_threshold,
            sim_config.atr_min_pct,
            sim_config.ema_trend_persistence_bars,
            sim_config.breakeven_trigger_atr,
            sim_config.target_atr_multiple,
        )
    else:
        sim_config = SimulatorConfig(
            account_balance=args.balance,
            max_trades_per_session=2,
            daily_loss_limit_usd=1_000.0,
            max_loss_limit_usd=2_000.0,
        )

    # ── Load data ────────────────────────────────────────────────────────────
    all_sessions: dict = {}
    for idx, (sym, path_1m) in enumerate(zip(args.symbols, args.data_1m)):
        path_1h = args.data_1h[idx] if args.data_1h else None
        loader = DataLoader(path_1m=path_1m, symbol=sym, path_1h=path_1h)
        loader.load()
        sessions = loader.build_sessions()
        if not sessions:
            print(f"[WARNING] No valid sessions found for {sym}. Skipping.", file=sys.stderr)
            continue
        all_sessions[sym] = sessions
        print(f"Loaded {len(sessions)} sessions for {sym}.")

    if not all_sessions:
        print("[ERROR] No sessions loaded for any symbol. Exiting.", file=sys.stderr)
        sys.exit(1)

    # ── Walk-forward ────────────────────────────────────────────────────────
    wf = WalkForward(
        sessions=all_sessions,
        sim_config=sim_config,
        train_months=args.train_months,
        test_months=args.test_months,
    )

    print(f"\nRunning walk-forward ({args.train_months}m train / {args.test_months}m test)…")
    result = wf.run()
    print(f"Completed {len(result.folds)} folds, {len(result.all_oos_trades)} OOS trades.\n")

    # ── Output ───────────────────────────────────────────────────────────────
    print_results(result, daily_loss_limit_usd=sim_config.daily_loss_limit_usd)
    write_csv(result, args.output)


if __name__ == "__main__":
    main()
