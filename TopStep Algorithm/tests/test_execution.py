from __future__ import annotations

import shutil
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from config import PROFILE_TOPSTEP_50K_EXPRESS, TraderConfig, build_config
from backtest.dashboard import build_dashboard_payload
from backtest.engine import BacktestResult, SimulatedBacktestEngine
from data_pipeline.preprocess import preprocess, select_primary_symbol
from execution.engine import ExecutionEngine
from execution.logging import EventLogger
from execution.order_manager import OrderManager
from execution.scheduler import SessionScheduler
from execution.topstepx_adapter import TopstepXAdapter
from features.indicators import add_atr, add_ema, add_vwap
from models.orders import OrderState, PositionSnapshot, Regime, Side, TradingMode
from risk.engine import RiskEngine
from strategy.rules import SignalInput, build_order_intent, generate_intents


class ExecutionEngineTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="topstepx-tests-")
        self.config = TraderConfig()
        self.config.execution.trade_log_dir = self.tmpdir
        self.config.execution.intent_expiry_seconds = 60
        self.config.execution.max_orders_per_second = 10
        self.config.execution.reconnect_timeout_seconds = 5
        self.config.risk.max_position_size = 2
        self.config.risk.max_slippage_ticks = 2
        self.engine, self.adapter = self._build_engine(self.config)
        self.now = datetime(2026, 3, 20, 14, 0, tzinfo=UTC)
        started = self.engine.startup(now=self.now)
        self.assertTrue(started)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir)

    def _build_engine(self, config: TraderConfig) -> tuple[ExecutionEngine, TopstepXAdapter]:
        risk_engine = RiskEngine(config.risk)
        adapter = TopstepXAdapter(mode=TradingMode.MOCK)
        order_manager = OrderManager()
        logger = EventLogger(config.execution.trade_log_dir)
        engine = ExecutionEngine(config, risk_engine, adapter, order_manager, logger)
        return engine, adapter

    def _build_backtest_engine(self, config: TraderConfig) -> tuple[ExecutionEngine, TopstepXAdapter]:
        risk_engine = RiskEngine(config.risk, mode="backtest")
        adapter = TopstepXAdapter(mode=TradingMode.MOCK)
        order_manager = OrderManager()
        logger = EventLogger(config.execution.trade_log_dir)
        engine = ExecutionEngine(config, risk_engine, adapter, order_manager, logger, mode="backtest")
        return engine, adapter

    def _intent(self, *, regime: Regime = Regime.TREND_EXPANSION, qty: int = 1, symbol: str = "MES"):
        signal = SignalInput(
            symbol=symbol,
            timestamp=self.now,
            regime=regime,
            long_signal=True,
            short_signal=False,
            entry_price=5200.0,
            stop_price=5198.0,
            target_price=5204.0,
            signal_score=1.0,
            qty=qty,
        )
        return build_order_intent(signal, self.config.strategy)

    def test_entry_submission_is_fill_driven(self) -> None:
        intent = self._intent()
        assert intent is not None
        decision = self.engine.submit_intent(intent, now=self.now)
        self.assertTrue(decision.approved)
        assert decision.order_plan is not None
        self.assertIn(decision.order_plan.entry.order_id, self.adapter.orders)
        self.assertNotIn(decision.order_plan.stop.order_id, self.adapter.orders)
        self.assertIsNone(self.engine.order_manager.orders[decision.order_plan.stop.order_id].submitted_at)

        self.adapter.emit_fill(decision.order_plan.entry.order_id, 1, 5200.0)
        self.engine.drain_adapter_events()

        self.assertIn(decision.order_plan.stop.order_id, self.adapter.orders)
        self.assertEqual(self.adapter.orders[decision.order_plan.stop.order_id].qty, 1)
        self.assertEqual(self.adapter.orders[decision.order_plan.target.order_id].qty, 1)

    def test_partial_fill_resizes_children(self) -> None:
        intent = self._intent(qty=2)
        assert intent is not None
        decision = self.engine.submit_intent(intent, now=self.now)
        assert decision.order_plan is not None

        self.adapter.emit_fill(decision.order_plan.entry.order_id, 1, 5200.0)
        self.engine.drain_adapter_events()
        self.assertEqual(self.adapter.orders[decision.order_plan.stop.order_id].qty, 1)
        self.assertEqual(self.engine.risk_engine.position_for("MES").stop_covered_qty, 1)

        self.adapter.emit_fill(decision.order_plan.entry.order_id, 1, 5200.25)
        self.engine.drain_adapter_events()
        self.assertEqual(self.adapter.orders[decision.order_plan.stop.order_id].qty, 2)
        self.assertEqual(self.adapter.orders[decision.order_plan.target.order_id].qty, 2)
        self.assertEqual(self.engine.risk_engine.position_for("MES").stop_covered_qty, 2)

    def test_stop_coverage_sums_across_multiple_active_chains(self) -> None:
        config = TraderConfig()
        config.execution.trade_log_dir = self.tmpdir
        engine, adapter = self._build_backtest_engine(config)
        started = engine.startup(now=self.now)
        self.assertTrue(started)

        first_intent = self._intent()
        second_intent = self._intent()
        assert first_intent is not None
        assert second_intent is not None
        second_intent.intent_id = f"{second_intent.intent_id}-second"

        first = engine.submit_intent(first_intent, now=self.now)
        second = engine.submit_intent(second_intent, now=self.now + timedelta(minutes=1))
        assert first.order_plan is not None
        assert second.order_plan is not None

        adapter.emit_fill(first.order_plan.entry.order_id, 1, 5200.0)
        engine.drain_adapter_events()
        adapter.emit_fill(second.order_plan.entry.order_id, 1, 5200.5)
        engine.drain_adapter_events()

        self.assertEqual(engine.risk_engine.position_for("MES").qty, 2)
        self.assertEqual(engine.risk_engine.position_for("MES").stop_covered_qty, 2)
        self.assertFalse(engine.flatten_in_progress)

    def test_slippage_guard_rejects_entry(self) -> None:
        self.adapter.set_market_price("MES", 5201.0)
        intent = self._intent()
        assert intent is not None
        decision = self.engine.submit_intent(intent, now=self.now)
        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "entry_slippage_exceeded")

    def test_persisted_state_is_restored_on_restart(self) -> None:
        self.engine.processed_intents["persisted-intent"] = self.now
        self.engine._persist_state()

        new_config = TraderConfig()
        new_config.execution.trade_log_dir = self.tmpdir
        new_config.execution.intent_expiry_seconds = self.config.execution.intent_expiry_seconds
        new_engine, _ = self._build_engine(new_config)
        started = new_engine.startup(now=self.now)
        self.assertTrue(started)
        self.assertIn("persisted-intent", new_engine.processed_intents)

    def test_expired_processed_intents_are_pruned(self) -> None:
        self.engine.processed_intents["old"] = self.now - timedelta(seconds=120)
        self.engine.heartbeat(now=self.now)
        self.assertNotIn("old", self.engine.processed_intents)

    def test_reconnect_timeout_triggers_kill_switch_with_open_position(self) -> None:
        intent = self._intent()
        assert intent is not None
        decision = self.engine.submit_intent(intent, now=self.now)
        assert decision.order_plan is not None
        self.adapter.emit_fill(decision.order_plan.entry.order_id, 1, 5200.0)
        self.engine.drain_adapter_events()

        self.adapter.disconnect()
        self.adapter.reconnect_should_fail = True
        self.engine.heartbeat(now=self.now + timedelta(seconds=1))
        self.engine.heartbeat(now=self.now + timedelta(seconds=6))
        self.assertTrue(self.engine.risk_engine.state.kill_switch.armed)
        self.assertEqual(self.engine.risk_engine.state.kill_switch.reason, "reconnect_timeout")

    def test_flatten_all_is_idempotent(self) -> None:
        intent = self._intent()
        assert intent is not None
        decision = self.engine.submit_intent(intent, now=self.now)
        assert decision.order_plan is not None
        self.adapter.emit_fill(decision.order_plan.entry.order_id, 1, 5200.0)
        self.engine.drain_adapter_events()

        self.engine.flatten_all("manual", now=self.now)
        self.engine.flatten_all("manual", now=self.now)
        flatten_orders = [order for order in self.adapter.orders.values() if order.role == "flatten"]
        self.assertEqual(len(flatten_orders), 1)

    def test_rejected_protective_stop_triggers_kill_switch(self) -> None:
        self.adapter.reject_roles.add("stop")
        intent = self._intent()
        assert intent is not None
        decision = self.engine.submit_intent(intent, now=self.now)
        assert decision.order_plan is not None
        self.adapter.emit_fill(decision.order_plan.entry.order_id, 1, 5200.0)
        self.engine.drain_adapter_events()
        self.assertTrue(self.engine.risk_engine.state.kill_switch.armed)
        self.assertEqual(self.engine.risk_engine.state.kill_switch.reason, "protective_order_rejected")

    def test_reconciliation_mismatch_arms_kill_switch(self) -> None:
        self.adapter.positions["MES"] = PositionSnapshot(symbol="MES", qty=1, avg_price=5200.0)
        self.engine.heartbeat(now=self.now)
        self.assertTrue(self.engine.risk_engine.state.kill_switch.armed)
        self.assertEqual(self.engine.risk_engine.state.kill_switch.reason, "reconciliation_mismatch")

    def test_flatten_exit_closes_position_and_cancels_siblings(self) -> None:
        intent = self._intent()
        assert intent is not None
        decision = self.engine.submit_intent(intent, now=self.now)
        assert decision.order_plan is not None
        self.adapter.emit_fill(decision.order_plan.entry.order_id, 1, 5200.0)
        self.engine.drain_adapter_events()

        self.engine.handle_exit_signal("MES", "time_stop", now=self.now)
        flatten_orders = [order for order in self.adapter.orders.values() if order.role == "flatten"]
        self.assertEqual(len(flatten_orders), 1)
        self.adapter.emit_fill(flatten_orders[0].order_id, 1, 5201.0)
        self.engine.drain_adapter_events()

        position = self.engine.risk_engine.position_for("MES")
        self.assertTrue(position.is_flat)
        target_state = self.adapter.orders[decision.order_plan.target.order_id].state
        stop_state = self.adapter.orders[decision.order_plan.stop.order_id].state
        self.assertEqual(target_state, OrderState.CANCELED)
        self.assertEqual(stop_state, OrderState.CANCELED)

    def test_backtest_mode_bypasses_live_trade_count_limits(self) -> None:
        config = TraderConfig()
        config.execution.trade_log_dir = self.tmpdir
        config.risk.max_trades_per_day = 1
        engine, _ = self._build_backtest_engine(config)
        started = engine.startup(now=self.now)
        self.assertTrue(started)

        first_intent = self._intent()
        second_intent = self._intent()
        assert first_intent is not None
        assert second_intent is not None
        second_intent.intent_id = f"{second_intent.intent_id}-second"

        first_decision = engine.submit_intent(first_intent, now=self.now)
        self.assertTrue(first_decision.approved)
        engine.risk_engine.record_entry()

        second_decision = engine.submit_intent(second_intent, now=self.now + timedelta(minutes=1))
        self.assertTrue(second_decision.approved)

    def test_backtest_mode_ignores_cooldown_and_daily_lockout(self) -> None:
        config = TraderConfig()
        config.execution.trade_log_dir = self.tmpdir
        engine, _ = self._build_backtest_engine(config)
        started = engine.startup(now=self.now)
        self.assertTrue(started)

        for _ in range(config.risk.max_consecutive_losses + 1):
            engine.risk_engine.record_trade_close(-10.0, self.now)

        can_trade, reason = engine.risk_engine.can_trade(self.now)
        self.assertTrue(can_trade)
        self.assertEqual(reason, "ok")
        self.assertIsNone(engine.risk_engine.state.cooldown_until)
        self.assertFalse(engine.risk_engine.state.locked)

    def test_backtest_mode_bypasses_order_throttle_without_api_errors(self) -> None:
        config = TraderConfig()
        config.execution.trade_log_dir = self.tmpdir
        config.execution.max_orders_per_second = 1
        engine, adapter = self._build_backtest_engine(config)
        started = engine.startup(now=self.now)
        self.assertTrue(started)

        first_intent = self._intent()
        second_intent = self._intent()
        assert first_intent is not None
        assert second_intent is not None
        second_intent.intent_id = f"{second_intent.intent_id}-second"

        first = engine.submit_intent(first_intent, now=self.now)
        second = engine.submit_intent(second_intent, now=self.now)
        self.assertTrue(first.approved)
        self.assertTrue(second.approved)
        self.assertEqual(engine.risk_engine.state.api_error_count, 0)
        self.assertFalse(engine.risk_engine.state.kill_switch.armed)

        assert first.order_plan is not None
        assert second.order_plan is not None
        adapter.emit_fill(first.order_plan.entry.order_id, 1, 5200.0)
        adapter.emit_fill(second.order_plan.entry.order_id, 1, 5200.0)
        engine.drain_adapter_events()

        self.assertEqual(engine.risk_engine.state.api_error_count, 0)
        self.assertFalse(engine.risk_engine.state.kill_switch.armed)

    def test_denial_reasons_are_tracked_with_examples(self) -> None:
        intent = self._intent()
        assert intent is not None
        self.engine.flatten_in_progress = True

        decision = self.engine.submit_intent(intent, now=self.now)

        self.assertFalse(decision.approved)
        self.assertEqual(self.engine.denial_counts["flatten_in_progress"], 1)
        self.assertIn("flatten_in_progress", self.engine.denial_examples)
        example = self.engine.denial_examples["flatten_in_progress"][0]
        self.assertEqual(example["intent_id"], intent.intent_id)
        self.assertEqual(example["symbol"], intent.symbol)

    def test_dashboard_enriches_fills_and_builds_performance_matrix(self) -> None:
        intent = self._intent()
        assert intent is not None
        events_path = Path(self.tmpdir) / "events.jsonl"
        trade_ledger_path = Path(self.tmpdir) / "trade_ledger.jsonl"
        events_path.write_text(
            "\n".join(
                [
                    '{"event_type":"signal_emitted","payload":{"intent":{"intent_id":"intent-1","symbol":"MES","signal_ts":"2026-03-20T09:15:00-05:00","regime":"TREND_EXPANSION","side":"BUY"}}}',
                    '{"event_type":"entry_submitted","payload":{"entry":{"order_id":"entry-1","intent_id":"intent-1"},"intent":{"intent_id":"intent-1"}}}',
                    '{"event_type":"fill","payload":{"report":{"order_id":"entry-1","symbol":"MES","status":"FILLED","fill_qty":1,"fill_price":5200.0,"side":"BUY","timestamp":"2026-03-20T14:15:00+00:00"}}}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        trade_ledger_path.write_text(
            '{"timestamp":"2026-03-20T14:20:00+00:00","payload":{"symbol":"MES","intent_id":"intent-1","regime":"TREND_EXPANSION","signal_ts":"2026-03-20T09:15:00-05:00","reason":"profit_target","pnl":10.0,"total_trade_pnl":10.0,"report":{"fill_price":5204.0,"fill_qty":1,"timestamp":"2026-03-20T14:20:00+00:00"}}}\n',
            encoding="utf-8",
        )

        payload = build_dashboard_payload(
            intents=[intent],
            result=BacktestResult(
                fills=[],
                denied_intents=[],
                approved_intents=["intent-1"],
                denial_counts={},
                denial_examples={},
                concurrency_analysis={
                    1: {"trades": 1, "pnl": 10.0, "expectancy": 10.0, "win_rate": 1.0, "max_drawdown": 0.0, "blocked_signals": 0, "blocked_percentage": 0.0, "avg_concurrent_positions": 1.0, "max_concurrent_positions": 1, "avg_overlap_duration_minutes": 0.0, "overlap_pair_count": 0, "overlap_pnl_correlation": 0.0, "worst_clustered_loss": 0.0, "worst_clustered_loss_trade_count": 0, "max_simultaneous_drawdown": 0.0, "equity_curve": [{"exit_ts": "2026-03-20T14:20:00+00:00", "cumulative_pnl": 10.0, "drawdown": 0.0}], "drawdown_curve": [{"exit_ts": "2026-03-20T14:20:00+00:00", "drawdown": 0.0}], "overlap_distribution": [{"concurrent_positions": 1, "bars": 1, "percentage": 100.0}], "time_of_day": {"OPEN": {"trades": 1, "pnl": 10.0, "expectancy": 10.0, "win_rate": 1.0}}},
                    2: {"trades": 2, "pnl": 15.0, "expectancy": 7.5, "win_rate": 1.0, "max_drawdown": -5.0, "blocked_signals": 0, "blocked_percentage": 0.0, "avg_concurrent_positions": 1.5, "max_concurrent_positions": 2, "avg_overlap_duration_minutes": 1.0, "overlap_pair_count": 1, "overlap_pnl_correlation": 0.5, "worst_clustered_loss": -5.0, "worst_clustered_loss_trade_count": 1, "max_simultaneous_drawdown": -5.0, "equity_curve": [{"exit_ts": "2026-03-20T14:20:00+00:00", "cumulative_pnl": 15.0, "drawdown": 0.0}], "drawdown_curve": [{"exit_ts": "2026-03-20T14:20:00+00:00", "drawdown": 0.0}], "overlap_distribution": [{"concurrent_positions": 1, "bars": 1, "percentage": 50.0}, {"concurrent_positions": 2, "bars": 1, "percentage": 50.0}], "time_of_day": {"OPEN": {"trades": 2, "pnl": 15.0, "expectancy": 7.5, "win_rate": 1.0}}},
                },
            ),
            events_path=events_path,
            trade_ledger_path=trade_ledger_path,
        )

        self.assertEqual(payload["filled_samples"][0]["regime"], "TREND_EXPANSION")
        self.assertEqual(payload["filled_samples"][0]["time_bucket"], "OPEN")
        self.assertEqual(payload["trade_ledger"][0]["regime"], "TREND_EXPANSION")
        self.assertEqual(payload["trade_ledger"][0]["time_bucket"], "OPEN")
        self.assertEqual(payload["performance_matrix"][0]["regime"], "TREND_EXPANSION")
        self.assertEqual(payload["performance_matrix"][0]["time_bucket"], "OPEN")
        self.assertEqual(payload["performance_matrix"][0]["trade_count"], 1)
        self.assertEqual(payload["concurrency_analysis"][1]["trades"], 1)
        self.assertEqual(payload["position_constraint_analysis"]["levels"][1]["level"], 2)
        self.assertEqual(payload["concurrency_validation_rows"][1]["max_concurrent_positions"], 2)
        self.assertEqual(payload["concurrency_time_rows"][0]["time_bucket"], "OPEN")
        self.assertEqual(payload["summary"]["max_concurrent_positions"], 2)
        self.assertIn("blocked_by_rule", payload["exposure_control"])

    def test_backtest_engine_simulates_full_trade_lifecycle(self) -> None:
        config = TraderConfig()
        config.execution.trade_log_dir = self.tmpdir
        engine, _ = self._build_backtest_engine(config)
        started = engine.startup(now=self.now)
        self.assertTrue(started)

        intent = self._intent()
        assert intent is not None
        bars = pd.DataFrame(
            [
                {
                    "ts_event": pd.Timestamp("2026-03-20 08:45:00", tz="America/Chicago"),
                    "symbol": "MES",
                    "open": 5199.0,
                    "high": 5201.0,
                    "low": 5198.5,
                    "close": 5200.0,
                    "volume": 1000,
                },
                {
                    "ts_event": pd.Timestamp("2026-03-20 08:46:00", tz="America/Chicago"),
                    "symbol": "MES",
                    "open": 5200.0,
                    "high": 5204.5,
                    "low": 5199.75,
                    "close": 5204.0,
                    "volume": 1200,
                },
            ]
        )
        intent.signal_ts = bars.iloc[0]["ts_event"].to_pydatetime()

        result = SimulatedBacktestEngine(engine).run(bars, [intent])

        self.assertEqual(len(result.approved_intents), 1)
        self.assertEqual(len(result.trades), 1)
        trade = result.trades[0]
        self.assertEqual(trade.exit_reason, "take_profit")
        self.assertEqual(trade.time_bucket, "OPEN")
        self.assertEqual(trade.regime, "TREND_EXPANSION")
        self.assertEqual(trade.pnl, 20.0)
        self.assertGreaterEqual(trade.mfe, 20.0)

    def test_backtest_engine_simulates_short_trade_lifecycle(self) -> None:
        config = TraderConfig()
        config.execution.trade_log_dir = self.tmpdir
        engine, _ = self._build_backtest_engine(config)
        started = engine.startup(now=self.now)
        self.assertTrue(started)

        intent = self._intent()
        assert intent is not None
        intent.side = Side.SELL
        intent.entry_price = 5200.0
        intent.stop_price = 5202.0
        intent.target_price = 5196.0
        bars = pd.DataFrame(
            [
                {
                    "ts_event": pd.Timestamp("2026-03-20 08:45:00", tz="America/Chicago"),
                    "symbol": "MES",
                    "open": 5201.0,
                    "high": 5201.5,
                    "low": 5199.0,
                    "close": 5200.0,
                    "volume": 1000,
                },
                {
                    "ts_event": pd.Timestamp("2026-03-20 08:46:00", tz="America/Chicago"),
                    "symbol": "MES",
                    "open": 5200.0,
                    "high": 5200.25,
                    "low": 5195.5,
                    "close": 5196.0,
                    "volume": 1200,
                },
            ]
        )
        intent.signal_ts = bars.iloc[0]["ts_event"].to_pydatetime()

        result = SimulatedBacktestEngine(engine).run(bars, [intent])

        self.assertEqual(len(result.approved_intents), 1)
        self.assertEqual(len(result.trades), 1)
        trade = result.trades[0]
        self.assertEqual(trade.exit_reason, "take_profit")
        self.assertEqual(trade.side, "SELL")
        self.assertEqual(trade.pnl, 20.0)
        self.assertGreaterEqual(trade.mfe, 20.0)

    def test_generate_intents_emits_mirrored_short_breakdown(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "ts_event": pd.Timestamp("2026-03-20 08:30:00", tz="America/Chicago"),
                    "symbol": "MES",
                    "open": 101.0,
                    "high": 101.5,
                    "low": 100.5,
                    "close": 101.0,
                    "volume": 100.0,
                    "ema_fast": 101.0,
                    "ema_slow": 100.8,
                    "vwap": 100.9,
                    "atr": 1.0,
                    "atr_median": 1.0,
                },
                {
                    "ts_event": pd.Timestamp("2026-03-20 08:31:00", tz="America/Chicago"),
                    "symbol": "MES",
                    "open": 100.8,
                    "high": 101.0,
                    "low": 99.8,
                    "close": 100.0,
                    "volume": 150.0,
                    "ema_fast": 99.5,
                    "ema_slow": 100.4,
                    "vwap": 100.6,
                    "atr": 1.0,
                    "atr_median": 1.0,
                },
            ]
        )

        intents = generate_intents(frame, self.config.strategy)

        self.assertEqual(len(intents), 1)
        intent = intents[0]
        self.assertEqual(intent.side, Side.SELL)
        self.assertEqual(intent.stop_price, 101.0)
        self.assertEqual(intent.target_price, 98.0)
        self.assertEqual(intent.metadata["trend_state"], "ema_below_vwap_below")
        self.assertEqual(intent.metadata["breakout_level"], 100.5)

    def test_generate_intents_honors_strategy_base_qty(self) -> None:
        config = TraderConfig()
        config.strategy.base_qty = 2
        frame = pd.DataFrame(
            [
                {
                    "ts_event": pd.Timestamp("2026-03-20 08:30:00", tz="America/Chicago"),
                    "symbol": "MES",
                    "open": 100.0,
                    "high": 100.5,
                    "low": 99.75,
                    "close": 100.0,
                    "volume": 100.0,
                    "ema_fast": 100.0,
                    "ema_slow": 99.8,
                    "vwap": 99.9,
                    "atr": 1.0,
                    "atr_median": 1.0,
                },
                {
                    "ts_event": pd.Timestamp("2026-03-20 08:31:00", tz="America/Chicago"),
                    "symbol": "MES",
                    "open": 100.25,
                    "high": 101.25,
                    "low": 100.0,
                    "close": 101.0,
                    "volume": 140.0,
                    "ema_fast": 100.9,
                    "ema_slow": 100.2,
                    "vwap": 100.3,
                    "atr": 1.0,
                    "atr_median": 1.0,
                },
            ]
        )

        intents = generate_intents(frame, config.strategy)

        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].qty, 2)

    def test_add_atr_populates_5min_stop_sizing_series(self) -> None:
        frame = pd.DataFrame(
            [
                {"ts_event": pd.Timestamp("2026-03-20 08:30:00", tz="America/Chicago"), "symbol": "MES", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 100.0},
                {"ts_event": pd.Timestamp("2026-03-20 08:31:00", tz="America/Chicago"), "symbol": "MES", "open": 100.0, "high": 102.0, "low": 100.0, "close": 101.0, "volume": 100.0},
                {"ts_event": pd.Timestamp("2026-03-20 08:32:00", tz="America/Chicago"), "symbol": "MES", "open": 101.0, "high": 103.0, "low": 101.0, "close": 102.0, "volume": 100.0},
                {"ts_event": pd.Timestamp("2026-03-20 08:33:00", tz="America/Chicago"), "symbol": "MES", "open": 102.0, "high": 104.0, "low": 102.0, "close": 103.0, "volume": 100.0},
                {"ts_event": pd.Timestamp("2026-03-20 08:34:00", tz="America/Chicago"), "symbol": "MES", "open": 103.0, "high": 105.0, "low": 103.0, "close": 104.0, "volume": 100.0},
                {"ts_event": pd.Timestamp("2026-03-20 08:35:00", tz="America/Chicago"), "symbol": "MES", "open": 104.0, "high": 106.0, "low": 104.0, "close": 105.0, "volume": 100.0},
            ]
        )

        enriched = add_atr(frame)

        self.assertIn("atr_5min", enriched.columns)
        row_835 = enriched.loc[enriched["ts_event"] == pd.Timestamp("2026-03-20 08:35:00", tz="America/Chicago")].iloc[0]
        self.assertEqual(row_835["atr"], 2.0)
        self.assertEqual(row_835["atr_5min"], 4.0)

    def test_generate_intents_skips_extended_breakout_entries(self) -> None:
        config = TraderConfig()
        config.strategy.max_entry_extension_atr = 0.75
        frame = pd.DataFrame(
            [
                {
                    "ts_event": pd.Timestamp("2026-03-20 08:30:00", tz="America/Chicago"),
                    "symbol": "MES",
                    "open": 100.0,
                    "high": 100.5,
                    "low": 99.75,
                    "close": 100.0,
                    "volume": 100.0,
                    "ema_fast": 100.0,
                    "ema_slow": 99.8,
                    "vwap": 99.9,
                    "atr": 1.0,
                    "atr_median": 1.0,
                },
                {
                    "ts_event": pd.Timestamp("2026-03-20 08:31:00", tz="America/Chicago"),
                    "symbol": "MES",
                    "open": 100.5,
                    "high": 102.25,
                    "low": 100.5,
                    "close": 101.5,
                    "volume": 140.0,
                    "ema_fast": 101.2,
                    "ema_slow": 100.3,
                    "vwap": 100.4,
                    "atr": 1.0,
                    "atr_median": 1.0,
                },
            ]
        )

        intents = generate_intents(frame, config.strategy)

        self.assertEqual(len(intents), 0)

    def test_topstep_50k_express_profile_sets_safe_runtime_defaults(self) -> None:
        config = build_config(PROFILE_TOPSTEP_50K_EXPRESS)

        self.assertEqual(config.strategy.base_qty, 2)
        self.assertEqual(config.strategy.preferred_symbol, "MES")
        self.assertEqual(config.risk.max_position_size, 20)
        self.assertEqual(config.risk.max_concurrent_positions, 1)
        self.assertFalse(config.risk.enable_stacking)
        self.assertEqual(config.risk.internal_daily_loss_limit, 500.0)
        self.assertEqual(config.risk.risk_budget_threshold, 125.0)
        self.assertEqual(config.risk.reentry_breakout_delta_min, -1_000_000_000.0)
        self.assertEqual(config.risk.max_stop_distance_ticks, 80)

    def test_scheduler_supports_asia_pre_london_window(self) -> None:
        scheduler = SessionScheduler(self.config.session)

        asia_active = datetime(2026, 3, 20, 19, 0, tzinfo=UTC)
        asia_cutoff = datetime(2026, 3, 20, 6, 45, tzinfo=UTC)
        asia_flatten = datetime(2026, 3, 20, 6, 59, tzinfo=UTC)
        london_open = datetime(2026, 3, 20, 7, 5, tzinfo=UTC)

        self.assertTrue(scheduler.is_trading_session(asia_active))
        self.assertFalse(scheduler.is_past_new_trade_cutoff(asia_active))
        self.assertTrue(scheduler.is_trading_session(asia_cutoff))
        self.assertTrue(scheduler.is_past_new_trade_cutoff(asia_cutoff))
        self.assertTrue(scheduler.should_force_flatten(asia_flatten))
        self.assertFalse(scheduler.is_trading_session(london_open))

    def test_scheduler_does_not_apply_asia_flatten_to_day_session(self) -> None:
        scheduler = SessionScheduler(self.config.session)

        ny_open = datetime(2026, 3, 20, 15, 0, tzinfo=UTC)
        ny_flatten = datetime(2026, 3, 20, 20, 9, tzinfo=UTC)

        self.assertTrue(scheduler.is_trading_session(ny_open))
        self.assertFalse(scheduler.is_past_new_trade_cutoff(ny_open))
        self.assertFalse(scheduler.should_force_flatten(ny_open))
        self.assertTrue(scheduler.should_force_flatten(ny_flatten))

    def test_preprocess_keeps_bars_from_both_session_windows(self) -> None:
        frame = pd.DataFrame(
            [
                {"ts_event": pd.Timestamp("2026-03-20 00:30:00", tz="UTC"), "symbol": "MES", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
                {"ts_event": pd.Timestamp("2026-03-20 07:30:00", tz="UTC"), "symbol": "MES", "open": 2, "high": 2, "low": 2, "close": 2, "volume": 1},
                {"ts_event": pd.Timestamp("2026-03-20 13:30:00", tz="UTC"), "symbol": "MES", "open": 3, "high": 3, "low": 3, "close": 3, "volume": 1},
                {"ts_event": pd.Timestamp("2026-03-20 21:30:00", tz="UTC"), "symbol": "MES", "open": 4, "high": 4, "low": 4, "close": 4, "volume": 1},
            ]
        )

        filtered = preprocess(frame, self.config.session)

        kept = filtered["ts_event"].dt.tz_convert("America/Chicago").dt.strftime("%H:%M").tolist()
        self.assertEqual(kept, ["19:30", "08:30"])

    def test_preprocess_drops_out_of_session_bars(self) -> None:
        frame = pd.DataFrame(
            [
                {"ts_event": pd.Timestamp("2026-03-20 07:30:00", tz="UTC"), "symbol": "MES", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
                {"ts_event": pd.Timestamp("2026-03-20 12:00:00", tz="UTC"), "symbol": "MES", "open": 2, "high": 2, "low": 2, "close": 2, "volume": 1},
            ]
        )

        filtered = preprocess(frame, self.config.session)

        self.assertTrue(filtered.empty)

    def test_sequential_reentry_is_allowed_after_profitable_exit(self) -> None:
        config = TraderConfig()
        config.execution.trade_log_dir = self.tmpdir
        config.risk.max_concurrent_positions = 1
        config.risk.enable_stacking = False
        engine, adapter = self._build_backtest_engine(config)
        self.assertTrue(engine.startup(now=self.now))

        first = self._intent()
        second = self._intent()
        assert first is not None and second is not None
        first.intent_id = "intent-first"
        second.intent_id = "intent-second"
        first.signal_ts = self.now
        second.signal_ts = self.now + timedelta(minutes=3)
        first.metadata.update({"trend_state": "ema_above_vwap_above", "breakout_level": 5199.0, "volume_strength": 1.0})
        second.metadata.update({"trend_state": "ema_above_vwap_above", "breakout_level": 5201.0, "volume_strength": 1.3})

        first_decision = engine.submit_intent(first, now=self.now)
        self.assertTrue(first_decision.approved)
        assert first_decision.order_plan is not None
        adapter.emit_fill(first_decision.order_plan.entry.order_id, 1, 5200.0, timestamp=self.now)
        engine.drain_adapter_events()
        adapter.emit_fill(first_decision.order_plan.target.order_id, 1, 5204.0, timestamp=self.now + timedelta(minutes=1))
        engine.drain_adapter_events()

        second_decision = engine.submit_intent(second, now=self.now + timedelta(minutes=3))
        self.assertTrue(second_decision.approved)
        second_working_intent = engine.intent_registry[second.intent_id]
        self.assertTrue(second_working_intent.metadata["is_reentry"])
        self.assertEqual(second_working_intent.metadata["entry_number_in_trend"], 2)

    def test_reentry_requires_breakout_progression_before_other_checks(self) -> None:
        config = TraderConfig()
        config.execution.trade_log_dir = self.tmpdir
        config.risk.max_concurrent_positions = 1
        config.risk.enable_stacking = False
        config.risk.min_reentry_spacing_bars = 2
        engine, adapter = self._build_backtest_engine(config)
        self.assertTrue(engine.startup(now=self.now))

        first = self._intent()
        retry = self._intent()
        assert first is not None and retry is not None
        first.intent_id = "loss-first"
        retry.intent_id = "loss-retry"
        first.signal_ts = self.now
        retry.signal_ts = self.now + timedelta(minutes=5)
        first.metadata.update({"trend_state": "ema_above_vwap_above", "breakout_level": 5199.0, "volume_strength": 1.0})
        retry.metadata.update({"trend_state": "ema_above_vwap_above", "breakout_level": 5199.0, "volume_strength": 1.25})
        retry.signal_score = first.signal_score

        first_decision = engine.submit_intent(first, now=self.now)
        self.assertTrue(first_decision.approved)
        assert first_decision.order_plan is not None
        adapter.emit_fill(first_decision.order_plan.entry.order_id, 1, 5200.0, timestamp=self.now)
        engine.drain_adapter_events()
        adapter.emit_fill(first_decision.order_plan.stop.order_id, 1, 5198.0, timestamp=self.now + timedelta(minutes=1))
        engine.drain_adapter_events()

        retry_decision = engine.submit_intent(retry, now=self.now + timedelta(minutes=5))
        self.assertFalse(retry_decision.approved)
        self.assertEqual(retry_decision.reason, "reentry_breakout_delta")

    def test_select_primary_symbol_removes_spreads_and_other_contracts(self) -> None:
        frame = pd.DataFrame(
            [
                {"ts_event": pd.Timestamp("2026-03-20 08:30:00", tz="America/Chicago"), "symbol": "MESM6", "open": 10, "high": 11, "low": 9, "close": 10, "volume": 1},
                {"ts_event": pd.Timestamp("2026-03-20 08:31:00", tz="America/Chicago"), "symbol": "MESU6", "open": 20, "high": 21, "low": 19, "close": 20, "volume": 1},
                {"ts_event": pd.Timestamp("2026-03-20 08:32:00", tz="America/Chicago"), "symbol": "MESU6", "open": 21, "high": 22, "low": 20, "close": 21, "volume": 1},
                {"ts_event": pd.Timestamp("2026-03-20 08:30:00", tz="America/Chicago"), "symbol": "MESM6-MESU6", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
            ]
        )
        filtered, diagnostics = select_primary_symbol(frame, preferred_symbol="MES")
        self.assertEqual(filtered["symbol"].nunique(), 1)
        self.assertEqual(filtered["symbol"].iloc[0], "MESU6")
        self.assertEqual(diagnostics["non_outright_rows_dropped"], 1)
        self.assertEqual(diagnostics["selected_root_symbol"], "MES")

    def test_single_pass_concurrency_analysis_tracks_levels(self) -> None:
        config = TraderConfig()
        config.execution.trade_log_dir = self.tmpdir
        config.risk.enable_stacking = True
        config.risk.max_concurrent_positions = 2
        config.risk.max_positions_per_regime = 2
        config.risk.risk_budget_threshold = 1_000.0
        engine, _ = self._build_backtest_engine(config)
        self.assertTrue(engine.startup(now=self.now))

        first_intent = self._intent()
        second_intent = self._intent()
        assert first_intent is not None
        assert second_intent is not None
        first_intent.intent_id = "intent-1"
        second_intent.intent_id = "intent-2"
        second_intent.side = Side.SELL
        second_intent.regime = Regime.HIGH_VOL_BREAKOUT

        bars = pd.DataFrame(
            [
                {
                    "ts_event": pd.Timestamp("2026-03-20 08:45:00", tz="America/Chicago"),
                    "symbol": "MES",
                    "open": 5199.0,
                    "high": 5201.0,
                    "low": 5198.5,
                    "close": 5200.0,
                    "volume": 1000,
                },
                {
                    "ts_event": pd.Timestamp("2026-03-20 08:46:00", tz="America/Chicago"),
                    "symbol": "MES",
                    "open": 5200.0,
                    "high": 5204.5,
                    "low": 5199.75,
                    "close": 5204.0,
                    "volume": 1200,
                },
            ]
        )
        signal_ts = bars.iloc[0]["ts_event"].to_pydatetime()
        first_intent.signal_ts = signal_ts
        second_intent.signal_ts = signal_ts

        result = SimulatedBacktestEngine(engine, max_concurrent_positions=2, concurrency_levels=(1, 2)).run(
            bars,
            [first_intent, second_intent],
        )

        self.assertEqual(result.blocked_by_max_positions, 0)
        self.assertEqual(len(result.trades), 2)
        self.assertEqual(result.concurrency_analysis[2]["trades"], 2)
        self.assertEqual(result.concurrency_analysis[2]["blocked_signals"], 0)
        self.assertGreaterEqual(result.concurrency_analysis[2]["avg_concurrent_positions"], 1.0)
        self.assertIn("time_of_day", result.concurrency_analysis[2])

    def test_third_overlapping_trade_is_denied_by_two_position_cap(self) -> None:
        config = TraderConfig()
        config.execution.trade_log_dir = self.tmpdir
        config.risk.enable_stacking = True
        config.risk.max_concurrent_positions = 2
        config.risk.max_positions_per_regime = 3
        config.risk.risk_budget_threshold = 1_000.0
        engine, _ = self._build_backtest_engine(config)
        self.assertTrue(engine.startup(now=self.now))

        first_intent = self._intent(symbol="MESA")
        second_intent = self._intent(symbol="MESB")
        third_intent = self._intent(symbol="MESC")
        assert first_intent is not None
        assert second_intent is not None
        assert third_intent is not None
        first_intent.intent_id = "intent-1"
        second_intent.intent_id = "intent-2"
        third_intent.intent_id = "intent-3"
        first_intent.signal_score = 0.9
        second_intent.signal_score = 0.8
        third_intent.signal_score = 0.7
        second_intent.side = Side.SELL
        second_intent.regime = Regime.HIGH_VOL_BREAKOUT
        third_intent.side = Side.SELL
        third_intent.regime = Regime.LOW_VOL_COMPRESSION
        signal_ts = pd.Timestamp("2026-03-20 08:45:00", tz="America/Chicago").to_pydatetime()
        first_intent.signal_ts = signal_ts
        second_intent.signal_ts = signal_ts
        third_intent.signal_ts = signal_ts

        bars = pd.DataFrame(
            [
                {"ts_event": pd.Timestamp("2026-03-20 08:45:00", tz="America/Chicago"), "symbol": "MESA", "open": 5199.0, "high": 5201.0, "low": 5198.5, "close": 5200.0, "volume": 1000},
                {"ts_event": pd.Timestamp("2026-03-20 08:45:00", tz="America/Chicago"), "symbol": "MESB", "open": 5199.0, "high": 5201.0, "low": 5198.5, "close": 5200.0, "volume": 1000},
                {"ts_event": pd.Timestamp("2026-03-20 08:45:00", tz="America/Chicago"), "symbol": "MESC", "open": 5199.0, "high": 5201.0, "low": 5198.5, "close": 5200.0, "volume": 1000},
                {"ts_event": pd.Timestamp("2026-03-20 08:46:00", tz="America/Chicago"), "symbol": "MESA", "open": 5200.0, "high": 5204.5, "low": 5199.75, "close": 5204.0, "volume": 1200},
                {"ts_event": pd.Timestamp("2026-03-20 08:46:00", tz="America/Chicago"), "symbol": "MESB", "open": 5200.0, "high": 5204.5, "low": 5199.75, "close": 5204.0, "volume": 1200},
                {"ts_event": pd.Timestamp("2026-03-20 08:46:00", tz="America/Chicago"), "symbol": "MESC", "open": 5200.0, "high": 5204.5, "low": 5199.75, "close": 5204.0, "volume": 1200},
            ]
        )

        result = SimulatedBacktestEngine(engine, max_concurrent_positions=2, concurrency_levels=(1, 2)).run(
            bars,
            [first_intent, second_intent, third_intent],
        )

        self.assertEqual(len(result.approved_intents), 2)
        self.assertEqual(result.blocked_by_max_positions, 1)
        self.assertEqual(result.denial_counts["max_positions_reached"], 1)
        self.assertEqual(result.denial_examples["max_positions_reached"][0]["intent_id"], "intent-3")

    def test_higher_scored_signal_replaces_weaker_position_when_cap_is_full(self) -> None:
        config = TraderConfig()
        config.execution.trade_log_dir = self.tmpdir
        config.risk.enable_stacking = True
        config.risk.max_concurrent_positions = 2
        config.risk.max_positions_per_regime = 3
        config.risk.risk_budget_threshold = 1_000.0
        engine, _ = self._build_backtest_engine(config)
        self.assertTrue(engine.startup(now=self.now))

        first_intent = self._intent(symbol="MESA")
        second_intent = self._intent(symbol="MESB")
        third_intent = self._intent(symbol="MESC")
        assert first_intent is not None and second_intent is not None and third_intent is not None
        first_intent.intent_id = "intent-1"
        second_intent.intent_id = "intent-2"
        third_intent.intent_id = "intent-3"
        first_intent.signal_score = 0.45
        second_intent.signal_score = 0.85
        third_intent.signal_score = 0.95
        second_intent.side = Side.SELL
        second_intent.regime = Regime.HIGH_VOL_BREAKOUT
        third_intent.side = Side.SELL
        third_intent.regime = Regime.HIGH_VOL_BREAKOUT
        second_intent.stop_price = 5202.0
        second_intent.target_price = 5196.0
        third_intent.stop_price = 5203.0
        third_intent.target_price = 5196.0
        second_intent.metadata["atr"] = 1.0
        third_intent.metadata["atr"] = 1.0
        third_intent.entry_price = 5202.0
        first_ts = pd.Timestamp("2026-03-20 08:45:00", tz="America/Chicago").to_pydatetime()
        third_ts = pd.Timestamp("2026-03-20 08:46:00", tz="America/Chicago").to_pydatetime()
        first_intent.signal_ts = first_ts
        second_intent.signal_ts = first_ts
        third_intent.signal_ts = third_ts

        bars = pd.DataFrame(
            [
                {"ts_event": pd.Timestamp("2026-03-20 08:45:00", tz="America/Chicago"), "symbol": "MESA", "open": 5199.0, "high": 5200.5, "low": 5198.5, "close": 5200.0, "volume": 1000},
                {"ts_event": pd.Timestamp("2026-03-20 08:45:00", tz="America/Chicago"), "symbol": "MESB", "open": 5199.0, "high": 5200.5, "low": 5198.5, "close": 5200.0, "volume": 1000},
                {"ts_event": pd.Timestamp("2026-03-20 08:46:00", tz="America/Chicago"), "symbol": "MESA", "open": 5200.0, "high": 5201.0, "low": 5199.0, "close": 5200.5, "volume": 1000},
                {"ts_event": pd.Timestamp("2026-03-20 08:46:00", tz="America/Chicago"), "symbol": "MESB", "open": 5200.0, "high": 5201.0, "low": 5199.0, "close": 5200.5, "volume": 1000},
                {"ts_event": pd.Timestamp("2026-03-20 08:46:00", tz="America/Chicago"), "symbol": "MESC", "open": 5199.0, "high": 5201.0, "low": 5198.75, "close": 5200.5, "volume": 1000},
                {"ts_event": pd.Timestamp("2026-03-20 08:47:00", tz="America/Chicago"), "symbol": "MESA", "open": 5200.5, "high": 5201.0, "low": 5199.0, "close": 5200.0, "volume": 1000},
                {"ts_event": pd.Timestamp("2026-03-20 08:47:00", tz="America/Chicago"), "symbol": "MESB", "open": 5200.5, "high": 5201.0, "low": 5199.0, "close": 5200.0, "volume": 1000},
                {"ts_event": pd.Timestamp("2026-03-20 08:47:00", tz="America/Chicago"), "symbol": "MESC", "open": 5200.5, "high": 5204.5, "low": 5200.0, "close": 5204.0, "volume": 1200},
            ]
        )

        result = SimulatedBacktestEngine(engine, max_concurrent_positions=2, concurrency_levels=(1, 2)).run(
            bars,
            [first_intent, second_intent, third_intent],
        )

        self.assertEqual(len(result.approved_intents), 3)
        self.assertEqual(result.blocked_by_max_positions, 0)
        self.assertTrue(any(trade.intent_id == "intent-1" and trade.exit_reason == "replaced_by_higher_score" for trade in result.trades))
        self.assertTrue(any(trade.intent_id == "intent-3" for trade in result.trades))

    def test_same_symbol_replacement_does_not_stick_flatten_state(self) -> None:
        config = TraderConfig()
        config.execution.trade_log_dir = self.tmpdir
        config.risk.enable_stacking = True
        config.risk.max_concurrent_positions = 2
        config.risk.max_positions_per_regime = 3
        config.risk.risk_budget_threshold = 1_000.0
        engine, _ = self._build_backtest_engine(config)
        self.assertTrue(engine.startup(now=self.now))

        first_intent = self._intent(symbol="MES")
        second_intent = self._intent(symbol="MES")
        third_intent = self._intent(symbol="MES")
        assert first_intent is not None and second_intent is not None and third_intent is not None
        first_intent.intent_id = "intent-1"
        second_intent.intent_id = "intent-2"
        third_intent.intent_id = "intent-3"
        first_intent.signal_score = 0.45
        second_intent.signal_score = 0.85
        third_intent.signal_score = 0.95
        first_ts = pd.Timestamp("2026-03-20 08:45:00", tz="America/Chicago").to_pydatetime()
        second_ts = pd.Timestamp("2026-03-20 08:46:00", tz="America/Chicago").to_pydatetime()
        third_ts = pd.Timestamp("2026-03-20 08:47:00", tz="America/Chicago").to_pydatetime()
        first_intent.signal_ts = first_ts
        second_intent.signal_ts = second_ts
        third_intent.signal_ts = third_ts
        second_intent.metadata["atr"] = 1.0
        second_intent.entry_price = 5201.0
        third_intent.metadata["atr"] = 1.0
        third_intent.entry_price = 5202.0

        bars = pd.DataFrame(
            [
                {"ts_event": pd.Timestamp("2026-03-20 08:45:00", tz="America/Chicago"), "symbol": "MES", "open": 5199.0, "high": 5200.5, "low": 5198.5, "close": 5200.0, "volume": 1000},
                {"ts_event": pd.Timestamp("2026-03-20 08:46:00", tz="America/Chicago"), "symbol": "MES", "open": 5200.0, "high": 5201.0, "low": 5199.0, "close": 5200.5, "volume": 1000},
                {"ts_event": pd.Timestamp("2026-03-20 08:47:00", tz="America/Chicago"), "symbol": "MES", "open": 5200.5, "high": 5201.0, "low": 5200.0, "close": 5200.75, "volume": 1000},
                {"ts_event": pd.Timestamp("2026-03-20 08:48:00", tz="America/Chicago"), "symbol": "MES", "open": 5200.75, "high": 5204.5, "low": 5200.5, "close": 5204.0, "volume": 1200},
            ]
        )

        result = SimulatedBacktestEngine(engine, max_concurrent_positions=2, concurrency_levels=(1, 2)).run(
            bars,
            [first_intent, second_intent, third_intent],
        )

        self.assertEqual(len(result.approved_intents), 3)
        self.assertFalse(engine.flatten_in_progress)
        self.assertTrue(any(trade.intent_id == "intent-1" and trade.exit_reason == "replaced_by_higher_score" for trade in result.trades))

    def test_dashboard_reports_signal_quality(self) -> None:
        accepted = self._intent(symbol="MESA")
        rejected = self._intent(symbol="MESB")
        assert accepted is not None and rejected is not None
        accepted.intent_id = "accepted-1"
        rejected.intent_id = "rejected-1"
        accepted.signal_score = 1.2
        rejected.signal_score = 0.4
        signal_ts = pd.Timestamp("2026-03-20 08:45:00", tz="America/Chicago").to_pydatetime()
        accepted.signal_ts = signal_ts
        rejected.signal_ts = signal_ts

        events_path = Path(self.tmpdir) / "events.jsonl"
        trade_ledger_path = Path(self.tmpdir) / "trade_ledger.jsonl"
        events_path.write_text("", encoding="utf-8")
        trade_ledger_path.write_text("", encoding="utf-8")

        payload = build_dashboard_payload(
            intents=[accepted, rejected],
            result=BacktestResult(
                fills=[],
                denied_intents=["rejected-1"],
                approved_intents=["accepted-1"],
                denial_counts={"max_positions_reached": 1},
                denial_examples={
                    "max_positions_reached": [
                        {
                            "intent_id": "rejected-1",
                            "symbol": "MESB",
                            "signal_ts": signal_ts.isoformat(),
                            "signal_score": 0.4,
                            "entry_price": rejected.entry_price,
                            "stop_price": rejected.stop_price,
                            "target_price": rejected.target_price,
                            "regime": "TREND_EXPANSION",
                        }
                    ]
                },
                concurrency_analysis={
                    1: {"trades": 1, "pnl": 10.0, "expectancy": 10.0, "win_rate": 1.0, "max_drawdown": 0.0, "blocked_signals": 1, "blocked_percentage": 50.0, "avg_concurrent_positions": 1.0, "max_concurrent_positions": 1, "avg_overlap_duration_minutes": 0.0, "overlap_pair_count": 0, "overlap_pnl_correlation": 0.0, "worst_clustered_loss": 0.0, "worst_clustered_loss_trade_count": 0, "max_simultaneous_drawdown": 0.0, "equity_curve": [], "drawdown_curve": [], "overlap_distribution": [], "time_of_day": {}},
                    2: {"trades": 1, "pnl": 10.0, "expectancy": 10.0, "win_rate": 1.0, "max_drawdown": 0.0, "blocked_signals": 0, "blocked_percentage": 0.0, "avg_concurrent_positions": 1.0, "max_concurrent_positions": 2, "avg_overlap_duration_minutes": 0.0, "overlap_pair_count": 0, "overlap_pnl_correlation": 0.0, "worst_clustered_loss": 0.0, "worst_clustered_loss_trade_count": 0, "max_simultaneous_drawdown": 0.0, "equity_curve": [], "drawdown_curve": [], "overlap_distribution": [], "time_of_day": {}},
                },
                max_concurrent_positions=2,
                blocked_by_max_positions=1,
                percentage_blocked_by_max_positions=50.0,
            ),
            events_path=events_path,
            trade_ledger_path=trade_ledger_path,
        )

        self.assertAlmostEqual(payload["signal_quality"]["avg_score_accepted"], 1.2)
        self.assertAlmostEqual(payload["signal_quality"]["avg_score_rejected"], 0.4)
        self.assertEqual(payload["signal_quality"]["higher_score_rejected_pct"], 0.0)
        self.assertEqual(payload["top_rejected_signals"][0]["intent_id"], "rejected-1")

    def test_live_validation_denies_third_open_trade_at_cap(self) -> None:
        config = TraderConfig()
        config.execution.trade_log_dir = self.tmpdir
        config.risk.max_concurrent_positions = 2
        config.risk.max_active_intents_per_symbol = 3
        engine, adapter = self._build_engine(config)
        self.assertTrue(engine.startup(now=self.now))

        first = self._intent(symbol="MESA")
        second = self._intent(symbol="MESB")
        third = self._intent(symbol="MESC")
        assert first is not None and second is not None and third is not None

        first_decision = engine.submit_intent(first, now=self.now)
        second_decision = engine.submit_intent(second, now=self.now + timedelta(minutes=1))
        self.assertTrue(first_decision.approved)
        self.assertTrue(second_decision.approved)
        adapter.emit_fill(first_decision.order_plan.entry.order_id, 1, 5200.0, timestamp=self.now)
        adapter.emit_fill(second_decision.order_plan.entry.order_id, 1, 5200.0, timestamp=self.now + timedelta(minutes=1))
        engine.drain_adapter_events()

        third_decision = engine.submit_intent(third, now=self.now + timedelta(minutes=2))
        self.assertFalse(third_decision.approved)
        self.assertEqual(third_decision.reason, "max_positions_reached")

    def test_same_direction_stacking_blocks_quality_only_signal(self) -> None:
        config = TraderConfig()
        config.execution.trade_log_dir = self.tmpdir
        config.risk.enable_stacking = True
        config.risk.max_concurrent_positions = 2
        config.risk.max_active_intents_per_symbol = 3
        config.risk.max_positions_per_regime = 2
        config.risk.risk_budget_threshold = 1_000.0
        engine, adapter = self._build_engine(config)
        self.assertTrue(engine.startup(now=self.now))

        first = self._intent(symbol="MESA")
        second = self._intent(symbol="MESB")
        assert first is not None and second is not None
        first.signal_score = 0.65
        second.signal_score = 0.95

        first_decision = engine.submit_intent(first, now=self.now)
        self.assertTrue(first_decision.approved)
        adapter.emit_fill(first_decision.order_plan.entry.order_id, 1, 5200.0, timestamp=self.now)
        engine.drain_adapter_events()

        second_decision = engine.submit_intent(second, now=self.now + timedelta(minutes=1))
        self.assertFalse(second_decision.approved)
        self.assertEqual(second_decision.reason, "same_direction_blocked")

    def test_regime_limit_blocks_same_regime_even_opposite_direction(self) -> None:
        config = TraderConfig()
        config.execution.trade_log_dir = self.tmpdir
        config.risk.enable_stacking = True
        config.risk.max_concurrent_positions = 2
        config.risk.max_positions_per_regime = 1
        config.risk.max_active_intents_per_symbol = 3
        engine, adapter = self._build_engine(config)
        self.assertTrue(engine.startup(now=self.now))

        first = self._intent(symbol="MESA")
        second = self._intent(symbol="MESB")
        assert first is not None and second is not None
        second.side = Side.SELL

        first_decision = engine.submit_intent(first, now=self.now)
        self.assertTrue(first_decision.approved)
        adapter.emit_fill(first_decision.order_plan.entry.order_id, 1, 5200.0, timestamp=self.now)
        engine.drain_adapter_events()

        second_decision = engine.submit_intent(second, now=self.now + timedelta(minutes=5))
        self.assertFalse(second_decision.approved)
        self.assertEqual(second_decision.reason, "regime_limit")

    def test_risk_budget_blocks_excess_heat(self) -> None:
        config = TraderConfig()
        config.execution.trade_log_dir = self.tmpdir
        config.risk.enable_stacking = True
        config.risk.max_concurrent_positions = 2
        config.risk.max_positions_per_regime = 2
        config.risk.risk_budget_threshold = 3.0
        config.risk.max_cluster_risk_fraction = 1.0
        config.risk.max_active_intents_per_symbol = 3
        engine, adapter = self._build_engine(config)
        self.assertTrue(engine.startup(now=self.now))

        first = self._intent(symbol="MESA")
        second = self._intent(symbol="MESB")
        assert first is not None and second is not None
        first.entry_price = 5200.0
        first.stop_price = 5198.0
        second.entry_price = 5200.0
        second.stop_price = 5198.0
        second.side = Side.SELL
        second.regime = Regime.HIGH_VOL_BREAKOUT

        first_decision = engine.submit_intent(first, now=self.now)
        self.assertTrue(first_decision.approved)
        adapter.emit_fill(first_decision.order_plan.entry.order_id, 1, 5200.0, timestamp=self.now)
        engine.drain_adapter_events()

        second_decision = engine.submit_intent(second, now=self.now + timedelta(minutes=5))
        self.assertFalse(second_decision.approved)
        self.assertEqual(second_decision.reason, "risk_budget_exceeded")

    def test_second_position_scales_size_down(self) -> None:
        config = TraderConfig()
        config.execution.trade_log_dir = self.tmpdir
        config.risk.enable_stacking = True
        config.risk.max_concurrent_positions = 2
        config.risk.max_position_size = 2
        config.risk.max_positions_per_regime = 2
        config.risk.risk_budget_threshold = 1_000.0
        config.risk.max_active_intents_per_symbol = 3
        engine, adapter = self._build_engine(config)
        self.assertTrue(engine.startup(now=self.now))

        first = self._intent(symbol="MESA", qty=2)
        second = self._intent(symbol="MESB", qty=2)
        assert first is not None and second is not None
        second.side = Side.SELL
        second.regime = Regime.HIGH_VOL_BREAKOUT

        first_decision = engine.submit_intent(first, now=self.now)
        self.assertTrue(first_decision.approved)
        self.assertEqual(first_decision.normalized_qty, 2)
        adapter.emit_fill(first_decision.order_plan.entry.order_id, 2, 5200.0, timestamp=self.now)
        engine.drain_adapter_events()

        second_decision = engine.submit_intent(second, now=self.now + timedelta(minutes=5))
        self.assertTrue(second_decision.approved)
        self.assertEqual(second_decision.normalized_qty, 1)

    def test_same_direction_stacking_allows_after_spacing_window(self) -> None:
        config = TraderConfig()
        config.execution.trade_log_dir = self.tmpdir
        config.risk.enable_stacking = True
        config.risk.max_concurrent_positions = 2
        config.risk.max_active_intents_per_symbol = 3
        config.risk.max_positions_per_regime = 2
        config.risk.same_direction_spacing_bars = 5
        config.risk.risk_budget_threshold = 1_000.0
        engine, adapter = self._build_engine(config)
        self.assertTrue(engine.startup(now=self.now))

        first = self._intent(symbol="MESA")
        second = self._intent(symbol="MESB")
        assert first is not None and second is not None
        first.signal_score = 0.9
        second.signal_score = 1.1

        first_decision = engine.submit_intent(first, now=self.now)
        self.assertTrue(first_decision.approved)
        adapter.emit_fill(first_decision.order_plan.entry.order_id, 1, 5200.0, timestamp=self.now)
        engine.drain_adapter_events()

        second_decision = engine.submit_intent(second, now=self.now + timedelta(minutes=5))
        self.assertTrue(second_decision.approved)
        self.assertEqual(second_decision.reason, "stacking_allowed_spacing")

    def test_same_direction_stacking_allows_on_price_improvement(self) -> None:
        config = TraderConfig()
        config.execution.trade_log_dir = self.tmpdir
        config.risk.enable_stacking = True
        config.risk.max_concurrent_positions = 2
        config.risk.max_active_intents_per_symbol = 3
        config.risk.max_positions_per_regime = 2
        config.risk.same_direction_spacing_bars = 5
        config.risk.risk_budget_threshold = 1_000.0
        engine, adapter = self._build_engine(config)
        self.assertTrue(engine.startup(now=self.now))

        first = self._intent(symbol="MESA")
        second = self._intent(symbol="MESB")
        assert first is not None and second is not None
        first.signal_score = 0.9
        second.signal_score = 0.8
        first.metadata["atr"] = 4.0
        second.metadata["atr"] = 4.0
        second.entry_price = 5202.5

        first_decision = engine.submit_intent(first, now=self.now)
        self.assertTrue(first_decision.approved)
        adapter.emit_fill(first_decision.order_plan.entry.order_id, 1, 5200.0, timestamp=self.now)
        engine.drain_adapter_events()

        second_decision = engine.submit_intent(second, now=self.now + timedelta(minutes=1))
        self.assertTrue(second_decision.approved)
        self.assertEqual(second_decision.reason, "stacking_allowed_price")

    def test_same_direction_stacking_is_blocked_in_low_vol_regime(self) -> None:
        config = TraderConfig()
        config.execution.trade_log_dir = self.tmpdir
        config.risk.enable_stacking = True
        config.risk.max_concurrent_positions = 2
        config.risk.max_active_intents_per_symbol = 3
        config.risk.max_positions_per_regime = 2
        config.risk.risk_budget_threshold = 1_000.0
        engine, adapter = self._build_engine(config)
        self.assertTrue(engine.startup(now=self.now))

        first = self._intent(symbol="MESA", regime=Regime.TREND_EXPANSION)
        second = self._intent(symbol="MESB", regime=Regime.LOW_VOL_COMPRESSION)
        assert first is not None and second is not None
        second.signal_score = 1.2

        first_decision = engine.submit_intent(first, now=self.now)
        self.assertTrue(first_decision.approved)
        adapter.emit_fill(first_decision.order_plan.entry.order_id, 1, 5200.0, timestamp=self.now)
        engine.drain_adapter_events()

        second_decision = engine.submit_intent(second, now=self.now + timedelta(minutes=5))
        self.assertFalse(second_decision.approved)
        self.assertEqual(second_decision.reason, "stacking_blocked_regime")

    def test_same_direction_stacking_blocked_when_no_condition_is_met(self) -> None:
        config = TraderConfig()
        config.execution.trade_log_dir = self.tmpdir
        config.risk.enable_stacking = True
        config.risk.max_concurrent_positions = 2
        config.risk.max_active_intents_per_symbol = 3
        config.risk.max_positions_per_regime = 2
        config.risk.same_direction_spacing_bars = 5
        config.risk.risk_budget_threshold = 1_000.0
        engine, adapter = self._build_engine(config)
        self.assertTrue(engine.startup(now=self.now))

        first = self._intent(symbol="MESA")
        second = self._intent(symbol="MESB")
        assert first is not None and second is not None
        first.signal_score = 1.0
        second.signal_score = 0.8
        first.metadata["atr"] = 4.0
        second.metadata["atr"] = 4.0
        second.entry_price = 5200.5

        first_decision = engine.submit_intent(first, now=self.now)
        self.assertTrue(first_decision.approved)
        adapter.emit_fill(first_decision.order_plan.entry.order_id, 1, 5200.0, timestamp=self.now)
        engine.drain_adapter_events()

        second_decision = engine.submit_intent(second, now=self.now + timedelta(minutes=1))
        self.assertFalse(second_decision.approved)
        self.assertEqual(second_decision.reason, "same_direction_blocked")

    def test_same_direction_stack_depth_is_limited(self) -> None:
        config = TraderConfig()
        config.execution.trade_log_dir = self.tmpdir
        config.risk.enable_stacking = True
        config.risk.max_concurrent_positions = 3
        config.risk.max_position_size = 3
        config.risk.max_positions_per_regime = 3
        config.risk.max_active_intents_per_symbol = 3
        config.risk.risk_budget_threshold = 1_000.0
        config.risk.max_same_direction_entries_per_trend = 2
        engine, adapter = self._build_engine(config)
        self.assertTrue(engine.startup(now=self.now))

        first = self._intent(symbol="MESA")
        second = self._intent(symbol="MESB")
        third = self._intent(symbol="MESC")
        assert first is not None and second is not None and third is not None
        second.metadata["atr"] = 4.0
        second.entry_price = 5202.5
        third.metadata["atr"] = 4.0
        third.entry_price = 5204.0

        first_decision = engine.submit_intent(first, now=self.now)
        self.assertTrue(first_decision.approved)
        adapter.emit_fill(first_decision.order_plan.entry.order_id, 1, 5200.0, timestamp=self.now)
        engine.drain_adapter_events()

        second_decision = engine.submit_intent(second, now=self.now + timedelta(minutes=1))
        self.assertTrue(second_decision.approved)
        adapter.emit_fill(second_decision.order_plan.entry.order_id, 1, 5202.5, timestamp=self.now + timedelta(minutes=1))
        engine.drain_adapter_events()

        third_decision = engine.submit_intent(third, now=self.now + timedelta(minutes=2))
        self.assertFalse(third_decision.approved)
        self.assertEqual(third_decision.reason, "stack_depth_reached")

    def test_stacked_loss_triggers_stacking_cooldown(self) -> None:
        config = TraderConfig()
        config.execution.trade_log_dir = self.tmpdir
        config.risk.enable_stacking = True
        config.risk.max_concurrent_positions = 2
        config.risk.max_active_intents_per_symbol = 3
        config.risk.max_positions_per_regime = 2
        config.risk.risk_budget_threshold = 1_000.0
        config.risk.stacking_loss_cooldown_bars = 5
        engine, adapter = self._build_engine(config)
        self.assertTrue(engine.startup(now=self.now))

        first = self._intent(symbol="MESA")
        second = self._intent(symbol="MESB")
        third = self._intent(symbol="MESC")
        assert first is not None and second is not None and third is not None
        second.metadata["atr"] = 4.0
        second.entry_price = 5202.5
        second.stop_price = 5201.0
        second.target_price = 5206.0
        third.metadata["atr"] = 4.0
        third.entry_price = 5202.5

        first_decision = engine.submit_intent(first, now=self.now)
        self.assertTrue(first_decision.approved)
        adapter.emit_fill(first_decision.order_plan.entry.order_id, 1, 5200.0, timestamp=self.now)
        engine.drain_adapter_events()

        second_decision = engine.submit_intent(second, now=self.now + timedelta(minutes=1))
        self.assertTrue(second_decision.approved)
        adapter.emit_fill(second_decision.order_plan.entry.order_id, 1, 5202.5, timestamp=self.now + timedelta(minutes=1))
        engine.drain_adapter_events()
        assert second_decision.order_plan is not None
        adapter.emit_fill(second_decision.order_plan.stop.order_id, 1, 5201.0, timestamp=self.now + timedelta(minutes=2))
        engine.drain_adapter_events()

        third_decision = engine.submit_intent(third, now=self.now + timedelta(minutes=3))
        self.assertFalse(third_decision.approved)
        self.assertEqual(third_decision.reason, "stacking_disabled_recent_pnl")

    def test_stacked_risk_budget_blocks_new_stack(self) -> None:
        config = TraderConfig()
        config.execution.trade_log_dir = self.tmpdir
        config.risk.enable_stacking = True
        config.risk.max_active_intents_per_symbol = 3
        config.risk.max_concurrent_positions = 3
        config.risk.max_position_size = 3
        config.risk.max_positions_per_regime = 2
        config.risk.max_same_direction_entries_per_trend = 3
        config.risk.risk_budget_threshold = 10.0
        config.risk.stacked_risk_budget_fraction = 0.4
        config.risk.max_cluster_risk_fraction = 1.0
        config.risk.heat_decay_lambda = 0.0
        engine, adapter = self._build_engine(config)
        self.assertTrue(engine.startup(now=self.now))

        first = self._intent(symbol="MESA")
        second = self._intent(symbol="MESB")
        third = self._intent(symbol="MESC")
        assert first is not None and second is not None and third is not None
        second.signal_score = 0.9
        second.metadata["atr"] = 5.0
        second.entry_price = 5201.5
        second.stop_price = 5199.5
        third.signal_score = 0.95
        third.metadata["atr"] = 5.0
        third.entry_price = 5203.0
        third.stop_price = 5201.0

        first_decision = engine.submit_intent(first, now=self.now)
        self.assertTrue(first_decision.approved)
        adapter.emit_fill(first_decision.order_plan.entry.order_id, 1, 5200.0, timestamp=self.now)
        engine.drain_adapter_events()

        second_decision = engine.submit_intent(second, now=self.now + timedelta(minutes=1))
        self.assertTrue(second_decision.approved)
        adapter.emit_fill(second_decision.order_plan.entry.order_id, 1, 5201.5, timestamp=self.now + timedelta(minutes=1))
        engine.drain_adapter_events()

        third_decision = engine.submit_intent(third, now=self.now + timedelta(minutes=2))
        self.assertFalse(third_decision.approved)
        self.assertEqual(third_decision.reason, "stacked_risk_budget_exceeded")

    def test_stacking_volatility_filter_blocks_same_direction_entry(self) -> None:
        config = TraderConfig()
        config.execution.trade_log_dir = self.tmpdir
        config.risk.enable_stacking = True
        config.risk.max_concurrent_positions = 2
        config.risk.max_active_intents_per_symbol = 3
        config.risk.max_positions_per_regime = 2
        engine, adapter = self._build_engine(config)
        self.assertTrue(engine.startup(now=self.now))

        first = self._intent(symbol="MESA")
        second = self._intent(symbol="MESB")
        assert first is not None and second is not None
        second.signal_score = 0.95
        second.metadata["atr"] = 4.0
        second.metadata["atr_pct"] = 1.8
        second.entry_price = 5202.0

        first_decision = engine.submit_intent(first, now=self.now)
        self.assertTrue(first_decision.approved)
        adapter.emit_fill(first_decision.order_plan.entry.order_id, 1, 5200.0, timestamp=self.now)
        engine.drain_adapter_events()

        second_decision = engine.submit_intent(second, now=self.now + timedelta(minutes=1))
        self.assertFalse(second_decision.approved)
        self.assertEqual(second_decision.reason, "stacking_volatility_blocked")

    def test_consecutive_stacked_losses_disable_future_stacking(self) -> None:
        config = TraderConfig()
        config.execution.trade_log_dir = self.tmpdir
        config.risk.enable_stacking = True
        config.risk.max_active_intents_per_symbol = 4
        config.risk.max_positions_per_regime = 2
        config.risk.max_concurrent_positions = 3
        config.risk.max_position_size = 3
        config.risk.risk_budget_threshold = 1_000.0
        config.risk.stacking_recent_pnl_lookback = 10
        config.risk.stacking_consecutive_loss_limit = 2
        config.risk.max_consecutive_losses = 10
        config.risk.max_trades_per_day = 10
        engine, adapter = self._build_engine(config)
        self.assertTrue(engine.startup(now=self.now))

        base = self._intent(symbol="MESA")
        stack_one = self._intent(symbol="MESB")
        stack_two = self._intent(symbol="MESC")
        stack_three = self._intent(symbol="MESD")
        assert base and stack_one and stack_two and stack_three
        for offset, intent in enumerate((stack_one, stack_two, stack_three), start=1):
            intent.signal_score = 0.95
            intent.metadata["atr"] = 4.0
            intent.entry_price = 5200.0 + (offset * 2.0)
            intent.stop_price = intent.entry_price - 1.5

        first_decision = engine.submit_intent(base, now=self.now)
        self.assertTrue(first_decision.approved)
        adapter.emit_fill(first_decision.order_plan.entry.order_id, 1, 5200.0, timestamp=self.now)
        engine.drain_adapter_events()

        second_decision = engine.submit_intent(stack_one, now=self.now + timedelta(minutes=1))
        self.assertTrue(second_decision.approved)
        adapter.emit_fill(second_decision.order_plan.entry.order_id, 1, stack_one.entry_price, timestamp=self.now + timedelta(minutes=1))
        engine.drain_adapter_events()
        adapter.emit_fill(second_decision.order_plan.stop.order_id, 1, stack_one.stop_price, timestamp=self.now + timedelta(minutes=2))
        engine.drain_adapter_events()
        engine.risk_engine.state.recent_stacked_trade_pnls = [5.0]
        engine.risk_engine.state.stacking_disabled_until = None
        engine.risk_engine.state.stacking_disabled_reason = ""

        third_decision = engine.submit_intent(stack_two, now=self.now + timedelta(minutes=6))
        self.assertTrue(third_decision.approved)
        adapter.emit_fill(third_decision.order_plan.entry.order_id, 1, stack_two.entry_price, timestamp=self.now + timedelta(minutes=6))
        engine.drain_adapter_events()
        adapter.emit_fill(third_decision.order_plan.stop.order_id, 1, stack_two.stop_price, timestamp=self.now + timedelta(minutes=7))
        engine.drain_adapter_events()

        fourth_decision = engine.submit_intent(stack_three, now=self.now + timedelta(minutes=8))
        self.assertFalse(fourth_decision.approved)
        self.assertEqual(fourth_decision.reason, "stacking_disabled_loss_cluster")


if __name__ == "__main__":
    unittest.main()
