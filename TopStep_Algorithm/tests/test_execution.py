from __future__ import annotations

import shutil
import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from config import (
    PROFILE_TOPSTEP_50K_EXPRESS,
    PROFILE_TOPSTEP_50K_EXPRESS_LONDON,
    PROFILE_TOPSTEP_50K_EXPRESS_LONDON_6B_PAPER,
    PROFILE_TOPSTEP_50K_EXPRESS_LONDON_6E_PAPER,
    TraderConfig,
    build_config,
)
from backtest.dashboard import build_dashboard_payload
from backtest.engine import BacktestResult, SimulatedBacktestEngine
from data_pipeline.preprocess import preprocess, select_primary_symbol
from execution.engine import ExecutionEngine
from execution.logging import EventLogger
from execution.order_manager import OrderManager
from execution.scheduler import SessionScheduler
from execution.topstep_live_adapter import TopstepHttpTransport, TopstepRealtimeClient
from execution.topstepx_adapter import TopstepXAdapter
from features.indicators import add_atr, add_ema, add_vwap
from models.instruments import infer_symbol_root, known_instruments
from models.orders import BrokerOrder, OrderState, OrderType, PositionSnapshot, Regime, Side, TimeInForce, TradingMode
from risk.engine import RiskEngine
from strategy.rules import SignalInput, build_order_intent, generate_intents


class FakeTopstepTransport(TopstepHttpTransport):
    def __init__(self) -> None:
        self.token = "fake-jwt-token"
        self.accounts = [
            {
                "id": 212,
                "name": "acct-123",
                "balance": 12500.0,
                "canTrade": True,
                "isVisible": True,
            }
        ]
        self.contracts = {
            "CON.F.US.MES.M25": {
                "id": "CON.F.US.MES.M25",
                "name": "MESM25",
                "description": "Micro E-mini S&P 500: June 2025",
                "tickSize": 0.25,
                "tickValue": 1.25,
                "activeContract": True,
            },
            "CON.F.US.6B.M25": {
                "id": "CON.F.US.6B.M25",
                "name": "6BM25",
                "description": "British Pound futures: June 2025",
                "tickSize": 0.0001,
                "tickValue": 6.25,
                "activeContract": True,
            },
            "CON.F.US.6E.M25": {
                "id": "CON.F.US.6E.M25",
                "name": "6EM25",
                "description": "Euro FX futures: June 2025",
                "tickSize": 0.00005,
                "tickValue": 6.25,
                "activeContract": True,
            }
        }
        self.orders: dict[int, dict[str, object]] = {}
        self.order_history: list[dict[str, object]] = []
        self.place_payloads: list[dict[str, object]] = []
        self.positions: list[dict[str, object]] = []
        self.next_order_id = 9001

    def post(
        self,
        base_url: str,
        path: str,
        payload: dict[str, object],
        *,
        bearer_token: str | None = None,
        timeout_seconds: int = 10,
    ) -> dict[str, object]:
        del base_url, timeout_seconds
        if path == "/api/Auth/loginKey":
            return {"success": True, "token": self.token}
        if bearer_token != self.token:
            raise RuntimeError("http_error:401:unauthorized")
        if path == "/api/Account/search":
            return {"success": True, "accounts": list(self.accounts)}
        if path == "/api/Contract/search":
            search_text = str(payload.get("searchText") or "")
            contracts = [contract for contract in self.contracts.values() if search_text in str(contract["name"]) or search_text in str(contract["id"])]
            return {"success": True, "contracts": contracts}
        if path == "/api/Contract/searchById":
            contract = self.contracts[str(payload["contractId"])]
            return {"success": True, "contracts": [contract]}
        if path == "/api/Position/searchOpen":
            return {"success": True, "positions": list(self.positions)}
        if path == "/api/Order/searchOpen":
            open_statuses = {1, 6}
            return {"success": True, "orders": [dict(order) for order in self.orders.values() if int(order.get("status", 0)) in open_statuses]}
        if path == "/api/Order/search":
            return {"success": True, "orders": [dict(order) for order in self.order_history]}
        if path == "/api/Order/place":
            self.place_payloads.append(dict(payload))
            order_id = self.next_order_id
            self.next_order_id += 1
            order = {
                "id": order_id,
                "accountId": int(payload["accountId"]),
                "contractId": str(payload["contractId"]),
                "creationTimestamp": "2026-04-12T14:00:00+00:00",
                "updateTimestamp": "2026-04-12T14:00:00+00:00",
                "status": 1,
                "type": int(payload["type"]),
                "side": int(payload["side"]),
                "size": int(payload["size"]),
                "limitPrice": payload.get("limitPrice"),
                "stopPrice": payload.get("stopPrice"),
                "fillVolume": 0,
                "filledPrice": None,
                "customTag": payload.get("customTag"),
            }
            self.orders[order_id] = order
            self.order_history.append(dict(order))
            return {"success": True, "orderId": order_id}
        if path == "/api/Order/cancel":
            order_id = payload.get("orderId", payload.get("id"))
            order = self.orders[int(order_id)]
            order["status"] = 3
            order["updateTimestamp"] = "2026-04-12T14:00:10+00:00"
            self.order_history.append(dict(order))
            return {"success": True}
        if path == "/api/Order/modify":
            order = self.orders[int(payload["id"])]
            order["size"] = int(payload["size"])
            if "limitPrice" in payload:
                order["limitPrice"] = payload["limitPrice"]
            if "stopPrice" in payload:
                order["stopPrice"] = payload["stopPrice"]
            order["updateTimestamp"] = "2026-04-12T14:00:06+00:00"
            self.order_history.append(dict(order))
            return {"success": True}
        raise AssertionError(f"Unhandled path: {path}")

    def push_order_update(self, order: dict[str, object]) -> None:
        order_id = int(order["id"])
        self.orders[order_id] = dict(order)
        self.order_history.append(dict(order))


class FakeRealtimeClient(TopstepRealtimeClient):
    def __init__(self) -> None:
        self.running = False
        self.websocket_url: str | None = None
        self.access_token: str | None = None
        self.account_id: int | None = None
        self.callback = None

    def start(self, websocket_url: str, access_token: str, account_id: int, event_callback) -> None:
        self.running = True
        self.websocket_url = websocket_url
        self.access_token = access_token
        self.account_id = account_id
        self.callback = event_callback

    def stop(self) -> None:
        self.running = False

    def is_running(self) -> bool:
        return self.running

    def emit(self, target: str, payload: dict[str, object]) -> None:
        if self.callback is not None:
            self.callback(target, payload)


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

    def test_child_order_prices_are_rounded_to_instrument_tick(self) -> None:
        intent = self._intent()
        assert intent is not None
        intent.stop_price = 5198.12
        intent.target_price = 5204.13

        decision = self.engine.submit_intent(intent, now=self.now)
        self.assertTrue(decision.approved)
        assert decision.order_plan is not None

        self.adapter.emit_fill(decision.order_plan.entry.order_id, 1, 5200.0)
        self.engine.drain_adapter_events()

        stop = self.adapter.orders[decision.order_plan.stop.order_id]
        target = self.adapter.orders[decision.order_plan.target.order_id]
        self.assertEqual(stop.stop_price, 5198.0)
        self.assertEqual(target.price, 5204.25)

    def test_child_order_api_failure_flattens_without_crashing(self) -> None:
        intent = self._intent()
        assert intent is not None
        original_place_order = self.adapter.place_order

        def place_order(order):
            if order.role == "stop":
                raise RuntimeError("topstep_api_error:/api/Order/place:2:Invalid price. Price is outside allowed range.")
            return original_place_order(order)

        self.adapter.place_order = place_order
        decision = self.engine.submit_intent(intent, now=self.now)
        self.assertTrue(decision.approved)
        assert decision.order_plan is not None

        self.adapter.emit_fill(decision.order_plan.entry.order_id, 1, 5200.0)
        self.engine.drain_adapter_events()

        self.assertTrue(self.engine.risk_engine.state.kill_switch.armed)
        self.assertEqual(self.engine.risk_engine.state.kill_switch.reason, "protective_order_failed")
        self.assertTrue(self.engine.flatten_in_progress)
        self.assertTrue(any(order.role == "flatten" for order in self.adapter.orders.values()))

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

    def test_duplicate_fill_report_is_ignored(self) -> None:
        intent = self._intent()
        assert intent is not None
        decision = self.engine.submit_intent(intent, now=self.now)
        assert decision.order_plan is not None

        report = self.adapter.emit_fill(decision.order_plan.entry.order_id, 1, 5200.0, timestamp=self.now)
        self.adapter.poll_events()

        self.engine.handle_execution_report(report)
        self.engine.handle_execution_report(report)

        self.assertEqual(self.engine.risk_engine.position_for("MES").qty, 1)
        self.assertEqual(self.engine.order_manager.chains_by_parent[decision.order_plan.entry.order_id].filled_qty, 1)
        self.assertEqual(self.adapter.orders[decision.order_plan.stop.order_id].qty, 1)
        self.assertFalse(self.engine.flatten_in_progress)

    def test_flatten_uses_broker_position_size_when_internal_position_is_wrong(self) -> None:
        self.engine.risk_engine.record_fill("MES", -16, 5200.0, self.now)
        self.adapter.positions["MES"] = PositionSnapshot(symbol="MES", qty=-8, avg_price=5200.0)

        self.engine.handle_exit_signal("MES", "missing_protective_stop", now=self.now)

        flatten_orders = [order for order in self.adapter.orders.values() if order.role == "flatten"]
        self.assertEqual(len(flatten_orders), 1)
        self.assertEqual(flatten_orders[0].side, Side.BUY)
        self.assertEqual(flatten_orders[0].qty, 8)
        self.assertTrue(self.engine.risk_engine.state.kill_switch.armed)
        self.assertEqual(self.engine.risk_engine.state.kill_switch.reason, "broker_position_mismatch_on_exit")

    def test_flatten_is_not_submitted_when_broker_is_flat_even_if_internal_position_is_stale(self) -> None:
        self.engine.risk_engine.record_fill("MES", -16, 5200.0, self.now)
        self.adapter.positions.pop("MES", None)

        self.engine.handle_exit_signal("MES", "manual", now=self.now)

        flatten_orders = [order for order in self.adapter.orders.values() if order.role == "flatten"]
        self.assertEqual(flatten_orders, [])
        self.assertTrue(self.engine.risk_engine.position_for("MES").is_flat)
        self.assertTrue(self.engine.risk_engine.state.kill_switch.armed)

    def test_flatten_all_submits_exit_for_broker_only_position(self) -> None:
        self.adapter.positions["MES"] = PositionSnapshot(symbol="MES", qty=8, avg_price=5200.0)

        self.engine.flatten_all("manual", now=self.now)

        flatten_orders = [order for order in self.adapter.orders.values() if order.role == "flatten"]
        self.assertEqual(len(flatten_orders), 1)
        self.assertEqual(flatten_orders[0].side, Side.SELL)
        self.assertEqual(flatten_orders[0].qty, 8)
        self.assertTrue(self.engine.risk_engine.state.kill_switch.armed)
        self.assertEqual(self.engine.risk_engine.state.kill_switch.reason, "broker_position_mismatch_on_exit")

    def test_heartbeat_flattens_broker_only_position_on_reconciliation_mismatch(self) -> None:
        self.adapter.positions["MES"] = PositionSnapshot(symbol="MES", qty=-8, avg_price=5200.0)

        self.engine.heartbeat(now=self.now)

        flatten_orders = [order for order in self.adapter.orders.values() if order.role == "flatten"]
        self.assertEqual(len(flatten_orders), 1)
        self.assertEqual(flatten_orders[0].side, Side.BUY)
        self.assertEqual(flatten_orders[0].qty, 8)
        self.assertTrue(self.engine.risk_engine.state.kill_switch.armed)
        self.assertEqual(self.engine.risk_engine.state.kill_switch.reason, "reconciliation_mismatch")

    def test_missing_broker_stop_coverage_triggers_flatten(self) -> None:
        intent = self._intent()
        assert intent is not None
        decision = self.engine.submit_intent(intent, now=self.now)
        assert decision.order_plan is not None
        self.adapter.emit_fill(decision.order_plan.entry.order_id, 1, 5200.0)
        self.engine.drain_adapter_events()

        stop = self.adapter.orders[decision.order_plan.stop.order_id]
        self.adapter.orders[decision.order_plan.stop.order_id] = replace(stop, state=OrderState.CANCELED)

        self.engine._verify_stop_coverage("MES", self.now)

        self.assertTrue(self.engine.risk_engine.state.kill_switch.armed)
        self.assertEqual(self.engine.risk_engine.state.kill_switch.reason, "missing_protective_stop")
        flatten_orders = [order for order in self.adapter.orders.values() if order.role == "flatten"]
        self.assertEqual(len(flatten_orders), 1)
        self.assertEqual(flatten_orders[0].qty, 1)

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
        self.adapter._impl.reconnect_should_fail = True
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

    def test_startup_scrubs_stale_chain_and_open_intent_when_position_is_flat(self) -> None:
        # Reproduce the crash scenario captured in engine_state.json:
        # entry filled (8 SHORT MES), flatten filled while bot was offline.
        # The risk_engine position is correctly 0 but chain.closed_qty and
        # open_intents_by_symbol were not updated before the state was persisted.
        # On restart, startup() must scrub these so the next intent is not falsely
        # denied with max_positions_reached.
        #
        # We inject the stale state directly into the StateStore rather than
        # running a full trade cycle, because a full cycle would leave stop/target
        # orders in the adapter which the fresh adapter can't reconcile.
        import json

        stale_snapshot = {
            "active_position_intent": {},
            "counted_entry_orders": [],
            "disconnect_started_at": None,
            "flatten_in_progress": True,
            "intent_parent_map": {"stale-intent-001": "entry-stale001"},
            "processed_intents": {},
            "order_manager": {
                "chains_by_parent": {
                    "entry-stale001": {
                        "intent_id": "stale-intent-001",
                        "parent_order_id": "entry-stale001",
                        "stop_order_id": None,
                        "target_order_id": None,
                        "symbol": "MES",
                        "created_at": self.now.isoformat(),
                        "last_update": self.now.isoformat(),
                        "filled_qty": 1,    # entry filled
                        "closed_qty": 0,    # exit never tracked — the stale state
                    }
                },
                "intent_to_parent": {"stale-intent-001": "entry-stale001"},
                "orders": {
                    "entry-stale001": {
                        "order_id": "entry-stale001",
                        "symbol": "MES",
                        "side": "SELL",
                        "qty": 1,
                        "order_type": "MARKET",
                        "tif": "DAY",
                        "state": "FILLED",
                        "price": None,
                        "stop_price": None,
                        "filled_qty": 1,
                        "avg_fill_price": 5200.0,
                        "parent_order_id": None,
                        "role": "entry",
                        "broker_order_id": "broker-001",
                        "submitted_at": self.now.isoformat(),
                        "updated_at": self.now.isoformat(),
                        "intent_id": "stale-intent-001",
                        "reason": "test",
                    }
                },
            },
            "risk_state": {
                "daily_realized_pnl": 0.0,
                "equity_peak": 0.0,
                "trade_count": 0,
                "completed_trade_count": 0,
                "consecutive_losses": 0,
                "cooldown_until": None,
                "last_entry_at": None,
                "last_exit_at": None,
                "last_stacked_loss_at": None,
                "last_exit_regime": "",
                "last_exit_trend_state": "",
                "last_exit_signal_score": 0.0,
                "last_exit_breakout_level": 0.0,
                "last_exit_volume_strength": 0.0,
                "last_exit_side": "",
                "last_trade_was_loss": False,
                "current_trend_key": "",
                "entries_in_current_trend": 0,
                "stacking_disabled_until": None,
                "stacking_disabled_reason": "",
                "recent_stacked_trade_pnls": [],
                "consecutive_stacked_losses": 0,
                "stacking_disabled_events": 0,
                "api_error_count": 0,
                "open_intents_by_symbol": {"MES": 1},  # stale: never resolved
                "positions": {
                    "MES": {
                        "symbol": "MES",
                        "qty": 0,           # correctly flat
                        "avg_price": None,
                        "realized_pnl": 0.0,
                        "unrealized_pnl": 0.0,
                        "stop_covered_qty": 0,
                        "last_updated": None,
                    }
                },
                "open_trade_pnl_by_symbol": {},
                "locked": False,
                "kill_switch": {
                    "armed": False,
                    "reason": "",
                    "activated_at": None,
                    "requires_manual_reset": False,
                },
            },
        }

        state_path = Path(self.tmpdir) / "engine_state.json"
        state_path.write_text(json.dumps(stale_snapshot))

        new_config = TraderConfig()
        new_config.execution.trade_log_dir = self.tmpdir
        new_config.execution.intent_expiry_seconds = self.config.execution.intent_expiry_seconds
        new_engine, _ = self._build_engine(new_config)
        started = new_engine.startup(now=self.now)
        self.assertTrue(started, "startup must succeed when broker confirms clean book")

        # Startup must have scrubbed both stale counters.
        scrubbed_chain = new_engine.order_manager.chains_by_parent["entry-stale001"]
        self.assertEqual(scrubbed_chain.filled_qty, scrubbed_chain.closed_qty,
                         "chain.closed_qty must be advanced to filled_qty after startup scrub")
        self.assertEqual(new_engine.risk_engine.state.open_intents_by_symbol.get("MES", 0), 0,
                         "stale open intent must be cleared on startup")
        # A new intent must now pass max_concurrent_positions check.
        self.assertEqual(new_engine.order_manager.active_position_count(), 0)
        new_intent = self._intent()
        assert new_intent is not None
        new_decision = new_engine.submit_intent(new_intent, now=self.now)
        self.assertTrue(new_decision.approved, f"Expected approved, got: {new_decision.reason}")

    def test_flatten_fill_updates_chain_when_parent_id_is_missing(self) -> None:
        # When the broker adapter reconstructs a fill report without preserving
        # parent_order_id, on_order_update must still find the open chain via
        # symbol fallback so closed_qty is updated and the position can be reused.
        intent = self._intent()
        assert intent is not None
        decision = self.engine.submit_intent(intent, now=self.now)
        assert decision.order_plan is not None
        self.adapter.emit_fill(decision.order_plan.entry.order_id, 1, 5200.0)
        self.engine.drain_adapter_events()

        self.engine.handle_exit_signal("MES", "manual_flatten", now=self.now)
        flatten_orders = [order for order in self.adapter.orders.values() if order.role == "flatten"]
        self.assertEqual(len(flatten_orders), 1)

        # Drop parent_order_id to simulate the adapter losing the linkage on fill.
        flatten_id = flatten_orders[0].order_id
        original = self.adapter.orders[flatten_id]
        from dataclasses import replace as dc_replace
        self.adapter.orders[flatten_id] = dc_replace(original, parent_order_id=None)

        self.adapter.emit_fill(flatten_id, 1, 5201.0)
        self.engine.drain_adapter_events()

        chain = self.engine.order_manager.chains_by_parent[decision.order_plan.entry.order_id]
        self.assertEqual(chain.closed_qty, chain.filled_qty, "chain.closed_qty must equal filled_qty after flatten fill")
        self.assertEqual(self.engine.order_manager.active_position_count(), 0)

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

    def test_topstep_connection_config_defaults_exist(self) -> None:
        config = TraderConfig()
        self.assertEqual(config.execution.topstep.environment, "paper")
        self.assertEqual(config.execution.topstep.api_base_url, "https://api.topstepx.com")
        self.assertEqual(config.execution.topstep.missing_required_fields(), ("username", "api_key", "account_id"))

    def test_instrument_registry_includes_fx_and_index_contracts(self) -> None:
        instruments = known_instruments()
        for symbol in ("6B", "6E", "ES", "MES", "MNQ", "NQ"):
            self.assertIn(symbol, instruments)
            self.assertGreater(instruments[symbol].tick_size, 0)
            self.assertGreater(instruments[symbol].tick_value, 0)

        self.assertEqual(instruments["6B"].tick_size, 0.0001)
        self.assertEqual(instruments["6B"].tick_value, 6.25)
        self.assertEqual(instruments["6E"].tick_size, 0.00005)
        self.assertEqual(instruments["6E"].tick_value, 6.25)

    def test_infer_symbol_root_handles_digit_prefixed_futures(self) -> None:
        cases = {
            "6BM25": "6B",
            "6BU26": "6B",
            "6EM25": "6E",
            "6EU26": "6E",
            "ESM25": "ES",
            "MESU26": "MES",
            "MNQM25": "MNQ",
            "NQU26": "NQ",
        }
        for raw_symbol, expected_root in cases.items():
            with self.subTest(raw_symbol=raw_symbol):
                self.assertEqual(infer_symbol_root(raw_symbol), expected_root)

    def test_live_topstep_adapter_requires_connection_config(self) -> None:
        adapter = TopstepXAdapter(mode=TradingMode.LIVE)
        with self.assertRaisesRegex(RuntimeError, "topstep_config_incomplete"):
            adapter.connect()

    def test_live_topstep_adapter_can_connect_and_place_orders_via_transport(self) -> None:
        config = TraderConfig()
        config.execution.topstep.username = "demo-user"
        config.execution.topstep.api_key = "demo-key"
        config.execution.topstep.account_id = "acct-123"
        transport = FakeTopstepTransport()
        realtime = FakeRealtimeClient()
        adapter = TopstepXAdapter(
            mode=TradingMode.LIVE,
            config=config.execution.topstep,
            transport=transport,
            realtime_client=realtime,
        )
        adapter.connect()

        self.assertTrue(adapter.is_connected())
        self.assertEqual(adapter.get_account()["cash_balance"], 12500.0)
        self.assertTrue(realtime.is_running())
        self.assertEqual(realtime.account_id, 212)

        live_order = BrokerOrder(
            order_id="entry-live-1",
            symbol="MES",
            side=Side.BUY,
            qty=1,
            order_type=OrderType.MARKET,
            tif=TimeInForce.DAY,
            state=OrderState.PENDING,
            price=None,
            stop_price=None,
            role="entry",
        )
        submitted = adapter.place_order(live_order)
        self.assertEqual(submitted.broker_order_id, "9001")
        self.assertIn("entry-live-1", adapter.orders)
        self.assertNotIn("limitPrice", transport.place_payloads[-1])

        realtime.emit(
            "GatewayUserOrder",
            {
                "action": 1,
                "data": {
                    "id": 9001,
                    "accountId": 212,
                    "contractId": "CON.F.US.MES.M25",
                    "creationTimestamp": "2026-04-12T14:00:00+00:00",
                    "updateTimestamp": "2026-04-12T14:00:05+00:00",
                    "status": 2,
                    "type": 2,
                    "side": 0,
                    "size": 1,
                    "fillVolume": 1,
                    "filledPrice": 5200.25,
                    "customTag": "entry-live-1",
                },
            }
        )
        reports = adapter.poll_events()
        self.assertTrue(any(report.status == OrderState.FILLED for report in reports))
        self.assertEqual(adapter.orders["entry-live-1"].filled_qty, 1)

        realtime.emit("GatewayUserPosition", {"accountId": 212, "size": 1, "type": 1})

        canceled = adapter.cancel_order("entry-live-1")
        self.assertEqual(canceled.order_id, "entry-live-1")

    def test_live_topstep_adapter_normalizes_fx_contract_roots(self) -> None:
        config = TraderConfig()
        config.execution.topstep.username = "demo-user"
        config.execution.topstep.api_key = "demo-key"
        config.execution.topstep.account_id = "acct-123"
        adapter = TopstepXAdapter(
            mode=TradingMode.LIVE,
            config=config.execution.topstep,
            transport=FakeTopstepTransport(),
            realtime_client=FakeRealtimeClient(),
        )

        self.assertEqual(adapter._impl._normalize_symbol_root({"id": "CON.F.US.6B.M25", "name": "6BM25"}), "6B")
        self.assertEqual(adapter._impl._normalize_symbol_root({"id": "CON.F.US.6E.M25", "name": "6EM25"}), "6E")

    def test_live_topstep_adapter_blocks_unverified_fx_live_orders(self) -> None:
        config = TraderConfig()
        config.execution.topstep.username = "demo-user"
        config.execution.topstep.api_key = "demo-key"
        config.execution.topstep.account_id = "acct-123"
        adapter = TopstepXAdapter(
            mode=TradingMode.LIVE,
            config=config.execution.topstep,
            transport=FakeTopstepTransport(),
            realtime_client=FakeRealtimeClient(),
        )
        adapter.connect()

        order = BrokerOrder(
            order_id="entry-live-6b",
            symbol="6B",
            side=Side.BUY,
            qty=1,
            order_type=OrderType.MARKET,
            tif=TimeInForce.DAY,
            state=OrderState.PENDING,
            role="entry",
        )
        with self.assertRaisesRegex(RuntimeError, "6B/6E live execution is not verified"):
            adapter.place_order(order)

    def test_paper_topstep_adapter_allows_fx_orders(self) -> None:
        config = TraderConfig()
        config.execution.topstep.username = "demo-user"
        config.execution.topstep.api_key = "demo-key"
        config.execution.topstep.account_id = "acct-123"
        transport = FakeTopstepTransport()
        adapter = TopstepXAdapter(
            mode=TradingMode.PAPER,
            config=config.execution.topstep,
            transport=transport,
            realtime_client=FakeRealtimeClient(),
        )
        adapter.connect()

        submitted = adapter.place_order(
            BrokerOrder(
                order_id="entry-paper-6b",
                symbol="6B",
                side=Side.BUY,
                qty=1,
                order_type=OrderType.MARKET,
                tif=TimeInForce.DAY,
                state=OrderState.PENDING,
                role="entry",
            )
        )
        self.assertEqual(submitted.broker_order_id, "9001")
        self.assertEqual(transport.place_payloads[-1]["contractId"], "CON.F.US.6B.M25")

    def test_live_topstep_adapter_does_not_emit_reports_for_historical_orders_on_startup(self) -> None:
        config = TraderConfig()
        config.execution.topstep.username = "demo-user"
        config.execution.topstep.api_key = "demo-key"
        config.execution.topstep.account_id = "acct-123"
        transport = FakeTopstepTransport()
        transport.order_history.append(
            {
                "id": 8001,
                "accountId": 212,
                "contractId": "CON.F.US.MES.M25",
                "creationTimestamp": "2026-04-12T13:30:00+00:00",
                "updateTimestamp": "2026-04-12T13:30:05+00:00",
                "status": 2,
                "type": 2,
                "side": 0,
                "size": 1,
                "fillVolume": 1,
                "filledPrice": 5199.75,
                "customTag": "stale-entry",
            }
        )
        adapter = TopstepXAdapter(
            mode=TradingMode.LIVE,
            config=config.execution.topstep,
            transport=transport,
            realtime_client=FakeRealtimeClient(),
        )

        adapter.connect()

        self.assertEqual(adapter.poll_events(), [])
        self.assertIn("stale-entry", adapter.orders)
        self.assertEqual(adapter.orders["stale-entry"].state, OrderState.FILLED)

    def test_live_topstep_adapter_does_not_emit_reports_for_unknown_realtime_orders(self) -> None:
        config = TraderConfig()
        config.execution.topstep.username = "demo-user"
        config.execution.topstep.api_key = "demo-key"
        config.execution.topstep.account_id = "acct-123"
        transport = FakeTopstepTransport()
        realtime = FakeRealtimeClient()
        adapter = TopstepXAdapter(
            mode=TradingMode.LIVE,
            config=config.execution.topstep,
            transport=transport,
            realtime_client=realtime,
        )

        adapter.connect()
        realtime.emit(
            "GatewayUserOrder",
            {
                "id": 8002,
                "accountId": 212,
                "contractId": "CON.F.US.MES.M25",
                "creationTimestamp": "2026-04-12T13:30:00+00:00",
                "updateTimestamp": "2026-04-12T13:30:05+00:00",
                "status": 2,
                "type": 2,
                "side": 0,
                "size": 1,
                "fillVolume": 1,
                "filledPrice": 5199.75,
            },
        )

        self.assertEqual(adapter.poll_events(), [])
        self.assertIn("broker-8002", adapter.orders)
        self.assertEqual(adapter.orders["broker-8002"].state, OrderState.FILLED)

    def test_live_topstep_adapter_replaces_orders_via_transport(self) -> None:
        config = TraderConfig()
        config.execution.topstep.username = "demo-user"
        config.execution.topstep.api_key = "demo-key"
        config.execution.topstep.account_id = "acct-123"
        transport = FakeTopstepTransport()
        adapter = TopstepXAdapter(
            mode=TradingMode.LIVE,
            config=config.execution.topstep,
            transport=transport,
            realtime_client=FakeRealtimeClient(),
        )
        adapter.connect()

        submitted = adapter.place_order(
            BrokerOrder(
                order_id="target-live-1",
                symbol="MES",
                side=Side.SELL,
                qty=1,
                order_type=OrderType.LIMIT,
                tif=TimeInForce.DAY,
                state=OrderState.PENDING,
                price=5210.0,
                role="target",
            )
        )
        replaced = adapter.replace_order("target-live-1", qty=2, price=5212.0)
        self.assertEqual(replaced.qty, 2)
        self.assertEqual(replaced.price, 5212.0)
        self.assertEqual(submitted.broker_order_id, "9001")

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
        self.assertEqual(trade.exit_reason, "profit_target")
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
        self.assertEqual(trade.exit_reason, "profit_target")
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

    def test_trailing_drawdown_kill_switch_arms_at_configured_limit(self) -> None:
        """RiskEngine.can_trade() must arm the kill switch and return False when the
        trailing drawdown reaches the configured trailing_drawdown_kill_switch threshold."""
        from backtest.config import RiskLimits

        limits = RiskLimits()
        limits.trailing_drawdown_kill_switch = -1800.0
        engine = RiskEngine(limits, mode="backtest")
        now = datetime(2026, 4, 11, 9, 0, tzinfo=UTC)

        # Simulate: equity peaks at $500, then falls to -$1,300 → drawdown = -$1,800.
        engine.record_trade_close(500.0, now)   # equity_peak = 500, daily_pnl = 500
        engine.record_trade_close(-1800.0, now) # daily_pnl = -1300, drawdown = -1300 - 500 = -1800

        ok, reason = engine.can_trade(now)
        self.assertFalse(ok)
        self.assertEqual(reason, "trailing_drawdown_limit")
        self.assertTrue(engine.state.kill_switch.armed)
        self.assertEqual(engine.state.kill_switch.reason, "trailing_drawdown_limit")

    def test_trailing_drawdown_kill_switch_not_triggered_while_below_limit(self) -> None:
        """Kill switch must NOT fire when drawdown is within the safe range."""
        from backtest.config import RiskLimits

        limits = RiskLimits()
        limits.trailing_drawdown_kill_switch = -1800.0
        engine = RiskEngine(limits, mode="backtest")
        now = datetime(2026, 4, 11, 9, 0, tzinfo=UTC)

        # equity_peak = 500, daily_pnl = -1000, drawdown = -1500 (within limit).
        engine.record_trade_close(500.0, now)
        engine.record_trade_close(-1500.0, now)

        ok, reason = engine.can_trade(now)
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")

    def test_enforce_live_risk_rules_applies_daily_loss_limit_in_backtest(self) -> None:
        """With enforce_live_risk_rules=True, backtest mode must lock trading after
        the internal daily loss limit is breached (same as live mode)."""
        from backtest.config import RiskLimits

        limits = RiskLimits()
        limits.internal_daily_loss_limit = 500.0
        engine = RiskEngine(limits, mode="backtest", enforce_live_risk_rules=True)
        now = datetime(2026, 4, 11, 9, 0, tzinfo=UTC)

        # Just under the limit — should still be able to trade.
        engine.record_trade_close(-499.0, now)
        ok, _ = engine.can_trade(now)
        self.assertTrue(ok)

        # Breach the daily limit.
        engine.record_trade_close(-1.0, now)  # now at -$500 exactly (≤ -|500|)
        ok, reason = engine.can_trade(now)
        self.assertFalse(ok)
        self.assertEqual(reason, "daily_lockout")

    def test_topstep_50k_express_profile_sets_safe_runtime_defaults(self) -> None:
        config = build_config(PROFILE_TOPSTEP_50K_EXPRESS)

        self.assertEqual(config.strategy.base_qty, 8)
        self.assertEqual(config.strategy.preferred_symbol, "MES")
        self.assertEqual(config.risk.max_position_size, 20)
        self.assertEqual(config.risk.max_concurrent_positions, 1)
        self.assertFalse(config.risk.enable_stacking)
        self.assertEqual(config.risk.internal_daily_loss_limit, 600.0)
        self.assertEqual(config.risk.risk_budget_threshold, 125.0)
        # reentry_breakout_delta_min=0.0 requires the breakout level to have advanced
        # before allowing a reentry, preventing repeated entries at the same stall point.
        self.assertEqual(config.risk.reentry_breakout_delta_min, 0.0)
        self.assertEqual(config.risk.reentry_signal_score_min, 0.65)
        self.assertEqual(config.risk.max_stop_distance_ticks, 80)
        # Quality gate parameters
        # v6: signal quality raised to 0.45; volume filter at 0.9; 5-min ATR disabled.
        self.assertEqual(config.strategy.min_entry_signal_score, 0.45)
        self.assertEqual(config.strategy.volume_entry_filter, 0.9)
        self.assertFalse(config.strategy.use_5min_atr_for_stops)
        # Breakeven stop disabled (proven harmful in WFO v2/v3).
        self.assertEqual(config.strategy.breakeven_trigger_atr, 0.0)
        # 2:1 R:R reverted from 3:1 (3:1 targets unreachable in choppy windows).
        self.assertEqual(config.strategy.target_atr_multiple, 2.0)
        # ADX ≥ 25 regime filter.
        self.assertEqual(config.strategy.adx_min_threshold, 25.0)
        # EMA persistence: 3 consecutive bullish/bearish bars required before entry.
        self.assertEqual(config.strategy.ema_trend_persistence_bars, 3)
        # ATR volatility floor: skip entries when ATR < 85% of rolling median.
        self.assertEqual(config.strategy.atr_min_pct, 0.85)
        self.assertEqual(config.execution.commission_per_lot, 0.59)
        # Trailing drawdown kill switch: halt at $1,800 (buffer before $2,000 platform limit).
        self.assertEqual(config.risk.trailing_drawdown_kill_switch, -1800.0)
        # NY-only session: one window, cutoff at 12:30
        self.assertEqual(len(config.session.session_windows), 1)
        self.assertEqual(config.session.session_windows[0].label, "new_york")
        from datetime import time
        self.assertEqual(config.session.session_windows[0].no_new_trades_after, time(12, 30))
        # Holiday calendar: skip_dates must be populated for 2024-2027 range.
        self.assertGreater(len(config.session.skip_dates), 0)
        from datetime import date
        # Thanksgiving 2025 (Nov 27) must be excluded.
        self.assertIn(date(2025, 11, 27), config.session.skip_dates)
        # Pre-Christmas thin period: Dec 15, 2025 must be excluded.
        self.assertIn(date(2025, 12, 15), config.session.skip_dates)
        # Christmas Day 2025 must be excluded.
        self.assertIn(date(2025, 12, 25), config.session.skip_dates)

    def test_topstep_50k_express_london_profile_uses_london_utc_window(self) -> None:
        config = build_config(PROFILE_TOPSTEP_50K_EXPRESS_LONDON)

        self.assertEqual(config.strategy.base_qty, 8)
        self.assertEqual(config.strategy.preferred_symbol, "MES")
        self.assertEqual(config.risk.internal_daily_loss_limit, 600.0)
        self.assertEqual(config.strategy.atr_min_pct, 0.85)
        self.assertEqual(len(config.session.session_windows), 1)
        # London profiles use UTC so the scheduler window matches the signal engine's
        # London session detection (08:30–13:30 UTC) regardless of DST.
        self.assertEqual(config.session.timezone, "UTC")

        london_window = config.session.session_windows[0]
        self.assertEqual(london_window.label, "london")
        from datetime import time
        self.assertEqual(london_window.market_open, time(8, 30))
        self.assertEqual(london_window.no_new_trades_after, time(13, 0))
        self.assertEqual(london_window.force_flatten_at, time(13, 25))
        self.assertEqual(london_window.exchange_close, time(13, 30))

    def test_topstep_fx_london_profiles_are_paper_sized_and_single_symbol(self) -> None:
        config_6b = build_config(PROFILE_TOPSTEP_50K_EXPRESS_LONDON_6B_PAPER)
        config_6e = build_config(PROFILE_TOPSTEP_50K_EXPRESS_LONDON_6E_PAPER)

        self.assertEqual(config_6b.strategy.preferred_symbol, "6B")
        self.assertEqual(config_6b.strategy.instrument_root_symbol, "6B")
        self.assertEqual(config_6b.strategy.base_qty, 1)
        self.assertEqual(config_6b.risk.max_position_size, 1)

        self.assertEqual(config_6e.strategy.preferred_symbol, "6E")
        self.assertEqual(config_6e.strategy.instrument_root_symbol, "6E")
        self.assertEqual(config_6e.strategy.base_qty, 1)
        self.assertEqual(config_6e.risk.max_position_size, 1)

    def test_run_trader_blocks_fx_live_runtime_before_startup(self) -> None:
        from scripts.run_trader import build_runtime

        with self.assertRaisesRegex(SystemExit, "6B/6E live execution is not verified"):
            build_runtime(TradingMode.LIVE, profile=PROFILE_TOPSTEP_50K_EXPRESS_LONDON_6B_PAPER)
        engine = build_runtime(TradingMode.PAPER, profile=PROFILE_TOPSTEP_50K_EXPRESS_LONDON_6B_PAPER)
        self.assertEqual(engine.config.strategy.preferred_symbol, "6B")

    def test_scheduler_blocks_trading_on_holiday(self) -> None:
        """SessionScheduler.is_trading_session() must return False on skip_dates."""
        from backtest.config import SessionConfig, SessionWindow
        from execution.scheduler import SessionScheduler
        from datetime import date, time

        # Build a config with a single known skip date: Christmas Day 2025.
        session_cfg = SessionConfig()
        session_cfg.session_windows = (
            SessionWindow(
                label="new_york",
                market_open=time(8, 30),
                no_new_trades_after=time(12, 30),
                force_flatten_at=time(15, 8),
                exchange_close=time(15, 10),
            ),
        )
        session_cfg.skip_dates = (date(2025, 12, 25),)
        scheduler = SessionScheduler(session_cfg)

        # Christmas 2025 at 10:00 CT (16:00 UTC) — would normally be inside NY session.
        christmas_10am_ct = datetime(2025, 12, 25, 16, 0, tzinfo=UTC)
        self.assertFalse(scheduler.is_trading_session(christmas_10am_ct))
        self.assertTrue(scheduler.is_holiday(christmas_10am_ct))

        # Day before: Dec 24 should still trade (not in skip_dates here).
        day_before_10am = datetime(2025, 12, 24, 16, 0, tzinfo=UTC)
        self.assertTrue(scheduler.is_trading_session(day_before_10am))
        self.assertFalse(scheduler.is_holiday(day_before_10am))

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
