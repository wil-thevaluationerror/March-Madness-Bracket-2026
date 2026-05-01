from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

from backtest.simulator import TradeResult


ENTRY_FEATURE_COLUMNS = [
    "timestamp",
    "symbol",
    "direction",
    "session",
    "confluence_type",
    "setup_type",
    "accepted",
    "rejection_reason",
    "entry_candidate_price",
    "stop_candidate_price",
    "target_candidate_price",
    "atr_at_entry",
    "adx_at_entry",
    "ema_slope_at_entry",
    "vwap_distance_at_entry",
    "range_width_atr",
    "sweep_depth_atr",
    "distance_to_asian_mid",
    "time_since_sweep",
    "risk_points",
    "risk_usd",
    "daily_loss_gate_passed",
    "max_loss_gate_passed",
    "max_trades_gate_passed",
]

OUTCOME_COLUMNS = {
    "pnl_usd",
    "r_multiple",
    "exit_time",
    "exit_price",
    "mfe_points",
    "mae_points",
    "mfe_r",
    "mae_r",
    "holding_minutes",
    "exit_reason",
    "bars_held",
}


@dataclass(slots=True)
class CandidateSignalRecord:
    """Entry-time candidate setup record.

    This dataclass is intentionally limited to information known before final
    trade acceptance.  Realized outcome fields live on ``TradeResult`` and must
    not be added here.
    """

    timestamp: datetime
    symbol: str
    direction: str
    session: date
    confluence_type: str
    setup_type: str
    accepted: bool
    rejection_reason: str
    entry_candidate_price: float
    stop_candidate_price: float
    target_candidate_price: float
    atr_at_entry: float
    adx_at_entry: float
    ema_slope_at_entry: float
    vwap_distance_at_entry: float
    range_width_atr: float
    sweep_depth_atr: float
    distance_to_asian_mid: float
    time_since_sweep: int
    risk_points: float
    risk_usd: float
    daily_loss_gate_passed: bool
    max_loss_gate_passed: bool
    max_trades_gate_passed: bool


def record_from_trade(
    trade: TradeResult,
    *,
    accepted: bool,
    rejection_reason: str = "",
    daily_loss_gate_passed: bool = True,
    max_loss_gate_passed: bool = True,
    max_trades_gate_passed: bool = True,
) -> CandidateSignalRecord:
    return CandidateSignalRecord(
        timestamp=trade.entry_time,
        symbol=trade.symbol,
        direction=trade.direction,
        session=trade.session_date,
        confluence_type=trade.confluence_type,
        setup_type=trade.setup_type,
        accepted=accepted,
        rejection_reason=rejection_reason,
        entry_candidate_price=trade.entry_price,
        stop_candidate_price=trade.stop_price,
        target_candidate_price=trade.tp1_price,
        atr_at_entry=trade.atr14,
        adx_at_entry=trade.adx_at_entry,
        ema_slope_at_entry=trade.ema_slope,
        vwap_distance_at_entry=trade.vwap_distance_at_entry,
        range_width_atr=trade.range_width_atr,
        sweep_depth_atr=trade.sweep_depth_atr,
        distance_to_asian_mid=trade.distance_to_asian_mid,
        time_since_sweep=trade.time_since_sweep,
        risk_points=trade.risk_points,
        risk_usd=trade.risk_usd,
        daily_loss_gate_passed=daily_loss_gate_passed,
        max_loss_gate_passed=max_loss_gate_passed,
        max_trades_gate_passed=max_trades_gate_passed,
    )


def assert_no_outcome_columns(columns: Iterable[str]) -> None:
    overlap = sorted(set(columns) & OUTCOME_COLUMNS)
    if overlap:
        raise ValueError(
            "Candidate ledger contains post-trade outcome columns: "
            + ", ".join(overlap)
        )


def write_candidate_csv(records: list[CandidateSignalRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert_no_outcome_columns(ENTRY_FEATURE_COLUMNS)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ENTRY_FEATURE_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
