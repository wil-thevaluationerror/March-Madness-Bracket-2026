from __future__ import annotations

import sys
import types
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from api.market_data import Bar
from backtest import data_loader
from backtest.data_loader import DataLoader, SessionDay
from backtest.metrics import compute_metrics
from backtest.raw_setup_ledger import (
    RAW_SETUP_ENTRY_FEATURE_COLUMNS,
    RawSetupEvent,
    assert_no_raw_outcome_columns,
)
from backtest.reporter import write_csv
from backtest.signal_ledger import ENTRY_FEATURE_COLUMNS, assert_no_outcome_columns
from backtest.simulator import SessionSimulator, SimulatorConfig, TradeResult
from backtest.walk_forward import Fold, WalkForward, WalkForwardResult
from strategy.asian_range import compute_asian_range
from strategy.confluence import ConfluenceResult
from strategy.signal import TradeSignal, _compute_adx14


class _Metadata:
    def __init__(self, symbols: list[str]) -> None:
        self.symbols = symbols


class _Store:
    def __init__(self, frame: pd.DataFrame, symbols: list[str]) -> None:
        self._frame = frame
        self.metadata = _Metadata(symbols)

    def to_df(self) -> pd.DataFrame:
        return self._frame.copy()


def _install_fake_databento(
    monkeypatch: pytest.MonkeyPatch,
    frame: pd.DataFrame,
    symbols: list[str],
) -> None:
    class _DBNStore:
        @staticmethod
        def from_file(path: str) -> _Store:
            return _Store(frame, symbols)

    fake_db = types.SimpleNamespace(DBNStore=_DBNStore)
    monkeypatch.setitem(sys.modules, "databento", fake_db)


def test_load_raw_rejects_mismatched_parent_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame(
        {
            "ts_event": [pd.Timestamp("2026-01-02 00:00:00", tz="UTC")],
            "symbol": ["MESM6"],
            "open": [5000.0],
            "high": [5001.0],
            "low": [4999.0],
            "close": [5000.5],
            "volume": [100],
        }
    )
    _install_fake_databento(monkeypatch, frame, ["MES.FUT"])

    with pytest.raises(ValueError, match="does not contain 6B\\.FUT"):
        data_loader._load_raw("mes-only.dbn.zst", "6B")


def test_load_raw_filters_to_requested_outright_before_front_month_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ts = pd.Timestamp("2026-01-02 00:00:00", tz="UTC")
    frame = pd.DataFrame(
        {
            "ts_event": [ts, ts, ts],
            "symbol": ["MESH6", "6EH6", "6EH6-6EM6"],
            "open": [5000.0, 1.1000, 0.0010],
            "high": [5001.0, 1.1010, 0.0011],
            "low": [4999.0, 1.0990, 0.0009],
            "close": [5000.5, 1.1005, 0.0010],
            "volume": [10_000, 100, 1_000_000],
        }
    )
    _install_fake_databento(monkeypatch, frame, ["MES.FUT", "6E.FUT"])

    loaded, ts_col = data_loader._load_raw("mixed.dbn.zst", "6E")

    assert ts_col == "ts_event"
    assert len(loaded) == 1
    assert loaded.iloc[0]["symbol"] == "6EH6"
    assert loaded.iloc[0]["close"] == pytest.approx(1.1005)


def test_simulator_excludes_current_open_1h_bar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_1h_counts: list[int] = []

    class _FakeSignalEngine:
        def __init__(self, symbol: str, config=None) -> None:
            self.symbol = symbol

        def process_bar(self, bars_5m: list[Bar], bars_1h: list[Bar]) -> None:
            seen_1h_counts.append(len(bars_1h))
            return None

    monkeypatch.setattr("backtest.simulator.SignalEngine", _FakeSignalEngine)

    bars_5m = [
        Bar(datetime(2026, 1, 2, 8, 55, tzinfo=timezone.utc), 1.0, 1.0, 1.0, 1.0, 1),
        Bar(datetime(2026, 1, 2, 9, 0, tzinfo=timezone.utc), 1.0, 1.0, 1.0, 1.0, 1),
    ]
    bars_1h = [
        Bar(datetime(2026, 1, 2, 7, 0, tzinfo=timezone.utc), 1.0, 1.0, 1.0, 1.0, 1),
        Bar(datetime(2026, 1, 2, 8, 0, tzinfo=timezone.utc), 1.0, 1.0, 1.0, 1.0, 1),
        Bar(datetime(2026, 1, 2, 9, 0, tzinfo=timezone.utc), 1.0, 1.0, 1.0, 1.0, 1),
    ]
    session = SessionDay(
        symbol="6E",
        date=date(2026, 1, 2),
        bars_5m=bars_5m,
        bars_1h=bars_1h,
        london_start_idx=0,
        london_end_idx=2,
    )

    SessionSimulator().run(session)

    assert seen_1h_counts == [2, 2]


def _bars_5m(start: str, count: int, close_start: float = 1.1) -> list[Bar]:
    timestamps = pd.date_range(start, periods=count, freq="5min", tz="UTC")
    return [
        Bar(
            ts.to_pydatetime(),
            close_start + i * 0.0001,
            close_start + i * 0.0001 + 0.0002,
            close_start + i * 0.0001 - 0.0002,
            close_start + i * 0.0001,
            100,
        )
        for i, ts in enumerate(timestamps)
    ]


def test_asian_range_requires_complete_5m_window() -> None:
    partial = _bars_5m("2026-01-02 01:00:00", 71)
    complete = _bars_5m("2026-01-02 01:00:00", 72)

    assert compute_asian_range(partial, date(2026, 1, 2)) is None
    assert compute_asian_range(complete, date(2026, 1, 2)) is not None


def test_data_loader_skips_incomplete_london_session() -> None:
    loader = DataLoader(path_1m="unused.dbn.zst", symbol="6E")
    bars = _bars_5m("2026-01-02 08:30:00", 59)
    loader._df_5m = pd.DataFrame(
        {
            "ts_event": [b.timestamp for b in bars],
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
            "volume": [b.volume for b in bars],
        }
    )
    hourly = pd.date_range("2025-12-30 08:30:00", periods=80, freq="1h", tz="UTC")
    loader._df_1h = pd.DataFrame(
        {
            "ts_event": hourly,
            "open": [1.1] * len(hourly),
            "high": [1.101] * len(hourly),
            "low": [1.099] * len(hourly),
            "close": [1.1] * len(hourly),
            "volume": [100] * len(hourly),
        }
    )

    assert loader.build_sessions() == {}


def test_max_drawdown_pct_uses_supplied_account_balance() -> None:
    trade = TradeResult(
        symbol="6E",
        session_date=date(2026, 1, 2),
        direction="BUY",
        entry_price=1.1000,
        entry_time=datetime(2026, 1, 2, 9, 0, tzinfo=timezone.utc),
        stop_price=1.1010,
        tp1_price=1.1020,
        tp2_price=1.1030,
        tp1_filled=False,
        tp2_filled=False,
        sl_filled=True,
        exit_price=1.0990,
        exit_time=datetime(2026, 1, 2, 9, 5, tzinfo=timezone.utc),
        contracts=1,
        pnl_usd=-500.0,
        r_multiple=-1.0,
        confluence_type="OB",
        atr14=0.001,
        ema_slope=0.001,
    )

    metrics = compute_metrics([trade], account_balance=100_000.0)

    assert metrics.max_drawdown_usd == pytest.approx(-500.0)
    assert metrics.max_drawdown_pct == pytest.approx(-0.005)


def _trade(
    *,
    session_date: date,
    pnl: float,
    entry_hour: int = 9,
    symbol: str = "6E",
    stop_distance: float = 0.004,
) -> TradeResult:
    entry = 1.1000
    return TradeResult(
        symbol=symbol,
        session_date=session_date,
        direction="BUY",
        entry_price=entry,
        entry_time=datetime(2026, 1, session_date.day, entry_hour, 0, tzinfo=timezone.utc),
        stop_price=entry - stop_distance,
        tp1_price=entry + 0.001,
        tp2_price=entry + 0.002,
        tp1_filled=pnl > 0,
        tp2_filled=pnl > 500,
        sl_filled=pnl < 0,
        exit_price=entry + 0.001,
        exit_time=datetime(2026, 1, session_date.day, entry_hour, 5, tzinfo=timezone.utc),
        contracts=1,
        pnl_usd=pnl,
        r_multiple=1.0,
        confluence_type="FVG",
        atr14=0.001,
        ema_slope=0.001,
    )


def _empty_session(session_date: date, symbol: str = "6E") -> SessionDay:
    return SessionDay(
        symbol=symbol,
        date=session_date,
        bars_5m=[],
        bars_1h=[],
        london_start_idx=0,
        london_end_idx=0,
    )


def test_walk_forward_blocks_trade_when_stop_risk_breaches_topstep_daily_limit() -> None:
    wf = WalkForward({}, train_months=6, test_months=1)
    d = date(2026, 1, 2)
    wf._sim.run = lambda session: [
        _trade(session_date=d, pnl=100.0, stop_distance=0.00805)
    ]

    fold = wf._run_fold({}, {"6E": {d: _empty_session(d)}}, d, date(2026, 2, 2), 0)

    assert fold.test_trades == []
    assert fold.final_account_pnl == 0.0


def test_walk_forward_trails_topstep_max_loss_from_end_of_day_high_watermark() -> None:
    wf = WalkForward(
        {},
        sim_config=SimulatorConfig(daily_loss_limit_usd=5_000.0),
        train_months=6,
        test_months=1,
    )
    day1 = date(2026, 1, 2)
    day2 = date(2026, 1, 3)

    def fake_run(session: SessionDay) -> list[TradeResult]:
        if session.date == day1:
            return [_trade(session_date=day1, pnl=1_500.0, stop_distance=0.004)]
        return [_trade(session_date=day2, pnl=100.0, stop_distance=0.0168)]

    wf._sim.run = fake_run

    fold = wf._run_fold(
        {},
        {"6E": {day1: _empty_session(day1), day2: _empty_session(day2)}},
        day1,
        date(2026, 2, 2),
        0,
    )

    assert [t.session_date for t in fold.test_trades] == [day1]
    assert fold.final_account_pnl == pytest.approx(1_500.0)
    assert fold.max_loss_threshold == pytest.approx(-500.0)


def test_candidate_ledger_records_rejected_reason() -> None:
    wf = WalkForward({}, train_months=6, test_months=1)
    d = date(2026, 1, 2)
    wf._sim.run = lambda session: [
        _trade(session_date=d, pnl=100.0, stop_distance=0.00805)
    ]

    fold = wf._run_fold({}, {"6E": {d: _empty_session(d)}}, d, date(2026, 2, 2), 0)

    assert len(fold.signal_ledger) == 1
    assert fold.signal_ledger[0].accepted is False
    assert fold.signal_ledger[0].rejection_reason == "daily_loss_risk_exceeded"


def test_candidate_ledger_records_accepted_candidate() -> None:
    wf = WalkForward({}, train_months=6, test_months=1)
    d = date(2026, 1, 2)
    wf._sim.run = lambda session: [
        _trade(session_date=d, pnl=100.0, stop_distance=0.001)
    ]

    fold = wf._run_fold({}, {"6E": {d: _empty_session(d)}}, d, date(2026, 2, 2), 0)

    assert len(fold.signal_ledger) == 1
    assert fold.signal_ledger[0].accepted is True
    assert fold.signal_ledger[0].rejection_reason == ""


def test_candidate_ledger_rejects_outcome_columns() -> None:
    assert_no_outcome_columns(ENTRY_FEATURE_COLUMNS)
    with pytest.raises(ValueError, match="post-trade outcome"):
        assert_no_outcome_columns([*ENTRY_FEATURE_COLUMNS, "mfe_r"])


def test_raw_setup_ledger_rejects_outcome_feature_columns() -> None:
    assert_no_raw_outcome_columns(RAW_SETUP_ENTRY_FEATURE_COLUMNS)
    with pytest.raises(ValueError, match="post-trade outcome"):
        assert_no_raw_outcome_columns([*RAW_SETUP_ENTRY_FEATURE_COLUMNS, "pnl_usd"])


def test_raw_setup_records_rejected_setup_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    event = RawSetupEvent(
        event_id="evt-reject",
        timestamp=datetime(2026, 1, 2, 9, 0, tzinfo=timezone.utc),
        symbol="6E",
        timeframe="5m",
        session=date(2026, 1, 2),
        direction_candidate="BUY",
        setup_stage="ema_trend",
        setup_type="asian_range_sweep",
        rejected=True,
        rejection_stage="ema_trend",
        rejection_reason="ema_trend_filter_failed",
        rejection_category="market_structure_rejection",
    )

    class _FakeSignalEngine:
        def __init__(self, symbol: str, config=None) -> None:
            self._events = [event]

        def process_bar(self, bars_5m: list[Bar], bars_1h: list[Bar]) -> None:
            return None

        def pop_raw_setup_events(self) -> list[RawSetupEvent]:
            events = self._events
            self._events = []
            return events

    monkeypatch.setattr("backtest.simulator.SignalEngine", _FakeSignalEngine)
    bars = [
        Bar(datetime(2026, 1, 2, 9, 0, tzinfo=timezone.utc), 1.1, 1.1, 1.1, 1.1, 100)
    ]
    session = SessionDay("6E", date(2026, 1, 2), bars, [], 0, 1)

    trades = SessionSimulator().run(session)

    assert trades == []
    assert len(SessionSimulator().last_raw_setup_events) == 0
    sim = SessionSimulator()
    sim.run(session)
    assert len(sim.last_raw_setup_events) == 1
    assert sim.last_raw_setup_events[0].rejection_reason == "ema_trend_filter_failed"
    assert sim.last_raw_setup_events[0].rejection_stage == "ema_trend"


def test_raw_setup_records_accepted_setup_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    event = RawSetupEvent(
        event_id="evt-accept",
        timestamp=datetime(2026, 1, 2, 9, 0, tzinfo=timezone.utc),
        symbol="6E",
        timeframe="5m",
        session=date(2026, 1, 2),
        direction_candidate="BUY",
        setup_stage="final_candidate",
        setup_type="asian_range_sweep",
        confluence_type="FVG",
        passed=True,
        rejected=False,
        accepted_into_final_candidate=True,
    )

    class _FakeSignalEngine:
        def __init__(self, symbol: str, config=None) -> None:
            self._emitted = False
            self._events = [event]

        def process_bar(self, bars_5m: list[Bar], bars_1h: list[Bar]) -> TradeSignal | None:
            if self._emitted:
                return None
            self._emitted = True
            return TradeSignal(
                symbol="6E",
                direction="BUY",
                entry_price=1.1000,
                stop_price=1.0990,
                tp1_price=1.1010,
                tp2_price=1.1020,
                confluence=ConfluenceResult("FVG", "synthetic"),
                atr14=0.001,
                ema_slope=0.001,
                raw_setup_event_id="evt-accept",
            )

        def pop_raw_setup_events(self) -> list[RawSetupEvent]:
            events = self._events
            self._events = []
            return events

    monkeypatch.setattr("backtest.simulator.SignalEngine", _FakeSignalEngine)
    bars = [
        Bar(datetime(2026, 1, 2, 9, 0, tzinfo=timezone.utc), 1.1000, 1.1000, 1.1000, 1.1000, 100),
        Bar(datetime(2026, 1, 2, 9, 5, tzinfo=timezone.utc), 1.1000, 1.1025, 1.1000, 1.1020, 100),
    ]
    session = SessionDay("6E", date(2026, 1, 2), bars, [], 0, len(bars))

    sim = SessionSimulator()
    trades = sim.run(session)

    assert len(trades) == 1
    assert len(sim.last_raw_setup_events) == 1
    raw = sim.last_raw_setup_events[0]
    assert raw.accepted_into_final_candidate is True
    assert raw.accepted_into_trade is True
    assert raw.linked_trade_id
    assert raw.pnl_usd is not None


def test_signal_engine_known_setup_fails_at_asian_range_gate() -> None:
    from strategy.signal import SignalEngine

    bars = _bars_5m("2026-01-02 08:30:00", 15)
    engine = SignalEngine("6E")

    assert engine.process_bar(bars, []) is None
    events = engine.pop_raw_setup_events()

    assert len(events) == 1
    assert events[0].rejection_stage == "asian_range"
    assert events[0].rejection_reason == "missing_asian_range"


def test_simulator_tracks_mfe_mae_and_holding_path(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeSignalEngine:
        def __init__(self, symbol: str, config=None) -> None:
            self._emitted = False

        def process_bar(self, bars_5m: list[Bar], bars_1h: list[Bar]) -> TradeSignal | None:
            if self._emitted:
                return None
            self._emitted = True
            return TradeSignal(
                symbol="6E",
                direction="BUY",
                entry_price=1.1000,
                stop_price=1.0990,
                tp1_price=1.1010,
                tp2_price=1.1020,
                confluence=ConfluenceResult("FVG", "synthetic"),
                atr14=0.001,
                ema_slope=0.001,
                adx_at_entry=25.0,
                range_width_atr=3.0,
                sweep_depth_atr=0.5,
                distance_to_asian_mid=0.25,
            )

    monkeypatch.setattr("backtest.simulator.SignalEngine", _FakeSignalEngine)
    bars_5m = [
        Bar(datetime(2026, 1, 2, 9, 0, tzinfo=timezone.utc), 1.1000, 1.1000, 1.1000, 1.1000, 100),
        Bar(datetime(2026, 1, 2, 9, 5, tzinfo=timezone.utc), 1.1000, 1.1005, 1.0995, 1.1002, 100),
        Bar(datetime(2026, 1, 2, 9, 10, tzinfo=timezone.utc), 1.1002, 1.1025, 1.1005, 1.1020, 100),
    ]
    session = SessionDay(
        symbol="6E",
        date=date(2026, 1, 2),
        bars_5m=bars_5m,
        bars_1h=[],
        london_start_idx=0,
        london_end_idx=len(bars_5m),
    )

    trades = SessionSimulator().run(session)

    assert len(trades) == 1
    trade = trades[0]
    assert trade.exit_reason == "take_profit_2"
    assert trade.mfe_points == pytest.approx(0.0025)
    assert trade.mae_points == pytest.approx(0.0005)
    assert trade.mfe_r == pytest.approx(2.5)
    assert trade.mae_r == pytest.approx(0.5)
    assert trade.bars_held == 2
    assert trade.holding_minutes == pytest.approx(10.0)


def test_adx_is_bounded_to_standard_scale() -> None:
    bars = _bars_5m("2026-01-02 01:00:00", 40)

    adx = _compute_adx14(bars)

    assert 0.0 <= adx <= 100.0


def test_backtest_writes_raw_setup_outputs(tmp_path) -> None:
    d = date(2026, 1, 2)
    raw = RawSetupEvent(
        event_id="evt-report",
        timestamp=datetime(2026, 1, 2, 9, 0, tzinfo=timezone.utc),
        symbol="6E",
        timeframe="5m",
        session=d,
        direction_candidate="BUY",
        setup_stage="final_candidate",
        setup_type="asian_range_sweep",
        confluence_type="FVG",
        passed=True,
        rejected=False,
        accepted_into_final_candidate=True,
        accepted_into_trade=True,
    )
    trade = _trade(session_date=d, pnl=100.0)
    metrics = compute_metrics([trade])
    fold = Fold(
        fold_idx=0,
        train_start=d,
        train_end=d,
        test_start=d,
        test_end=date(2026, 2, 2),
        train_metrics=compute_metrics([]),
        test_metrics=metrics,
        test_trades=[trade],
        signal_ledger=[],
        raw_setup_ledger=[raw],
    )
    result = WalkForwardResult(
        folds=[fold],
        all_oos_trades=[trade],
        combined_oos_metrics=metrics,
        raw_setup_ledger=[raw],
    )

    write_csv(result, str(tmp_path))

    assert (tmp_path / "raw_setup_ledger.csv").exists()
    assert (tmp_path / "setup_rejection_summary.csv").exists()
