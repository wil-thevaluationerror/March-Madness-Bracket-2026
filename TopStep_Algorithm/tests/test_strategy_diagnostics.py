from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from backtest.config import StrategyConfig
from models.orders import TradingMode
from scripts.run_trader import run_loop
from strategy.diagnostics import StrategyDiagnosticsLogger
from strategy.rules import generate_intents


def _bar(ts: str, close: float, *, symbol: str = "6B", adx: float | None = 30.0) -> dict:
    row = {
        "ts_event": pd.Timestamp(ts, tz="America/Chicago"),
        "symbol": symbol,
        "open": close,
        "high": close + 0.0002,
        "low": close - 0.0002,
        "close": close,
        "volume": 1000,
        "ema_fast": close,
        "ema_slow": close,
        "vwap": close,
        "atr": 0.0002,
        "atr_median": 0.0002,
        "breakout_level": close + 0.0005,
        "breakdown_level": close - 0.0005,
    }
    if adx is not None:
        row["adx"] = adx
    return row


class StrategyDiagnosticsTestCase(unittest.TestCase):
    def test_diagnostic_logger_writes_required_no_trade_fields(self) -> None:
        with tempfile.TemporaryDirectory(prefix="strategy-diagnostics-") as tmpdir:
            path = Path(tmpdir) / "strategy_diagnostics.jsonl"
            logger = StrategyDiagnosticsLogger(path)
            config = StrategyConfig()
            frame = pd.DataFrame([_bar("2026-04-27 03:30:00", 1.2500)])

            intents = generate_intents(frame, config, diagnostics_callback=logger.write)

            self.assertEqual(intents, [])
            payload = json.loads(path.read_text(encoding="utf-8").strip())
            for field in (
                "ts",
                "event",
                "symbol",
                "bar_timestamp",
                "close",
                "session_allowed",
                "bars_loaded",
                "atr",
                "adx",
                "ema_trend_state",
                "vwap_condition",
                "breakout_condition",
                "signal_score",
                "decision",
                "no_trade_reason",
                "risk_allowed",
            ):
                self.assertIn(field, payload)
            self.assertEqual(payload["event"], "strategy_bar_diagnostic")
            self.assertEqual(payload["symbol"], "6B")
            self.assertEqual(payload["decision"], "no_trade")
            self.assertEqual(payload["no_trade_reason"], "no_breakout")

    def test_missing_optional_indicators_do_not_crash_diagnostics(self) -> None:
        records: list[dict] = []
        frame = pd.DataFrame(
            [
                {
                    "ts_event": pd.Timestamp("2026-04-27 03:30:00", tz="America/Chicago"),
                    "symbol": "6E",
                    "open": 1.1,
                    "high": 1.1001,
                    "low": 1.0999,
                    "close": 1.1,
                    "volume": 100,
                }
            ]
        )

        intents = generate_intents(frame, StrategyConfig(), diagnostics_callback=records.append)

        self.assertEqual(intents, [])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["symbol"], "6E")
        self.assertEqual(records[0]["decision"], "no_trade")

    def test_diagnostics_do_not_alter_signal_output(self) -> None:
        config = StrategyConfig(
            default_symbol="6B",
            min_entry_signal_score=0.0,
            volume_entry_filter=0.0,
            adx_min_threshold=0.0,
            ema_trend_persistence_bars=0,
            max_entry_extension_atr=2.0,
        )
        frame = pd.DataFrame(
            [
                {
                    **_bar("2026-04-27 03:30:00", 1.2500),
                    "high": 1.2504,
                    "ema_fast": 1.2501,
                    "ema_slow": 1.2500,
                    "vwap": 1.2500,
                },
                {
                    **_bar("2026-04-27 03:31:00", 1.2507),
                    "high": 1.2508,
                    "ema_fast": 1.2510,
                    "ema_slow": 1.2500,
                    "vwap": 1.2502,
                },
            ]
        )
        records: list[dict] = []

        without_diagnostics = generate_intents(frame, config)
        with_diagnostics = generate_intents(frame, config, diagnostics_callback=records.append)

        self.assertEqual(len(without_diagnostics), 1)
        self.assertEqual(len(with_diagnostics), 1)
        self.assertEqual(without_diagnostics[0].symbol, with_diagnostics[0].symbol)
        self.assertEqual(records[-1]["decision"], "trade")

    def test_live_mode_refuses_diagnostics_before_runtime_startup(self) -> None:
        with self.assertRaisesRegex(SystemExit, "diagnostics are disabled for live mode"):
            run_loop(TradingMode.LIVE, 3, diagnostics_enabled=True)


if __name__ == "__main__":
    unittest.main()
