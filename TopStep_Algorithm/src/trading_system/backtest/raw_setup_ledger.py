from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Iterable


RAW_SETUP_ENTRY_FEATURE_COLUMNS = [
    "event_id",
    "timestamp",
    "symbol",
    "timeframe",
    "session",
    "direction_candidate",
    "setup_stage",
    "setup_type",
    "confluence_type",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "asian_high",
    "asian_low",
    "asian_mid",
    "distance_to_asian_high",
    "distance_to_asian_low",
    "distance_to_asian_mid",
    "range_width_points",
    "range_width_atr",
    "sweep_side",
    "sweep_depth_points",
    "sweep_depth_atr",
    "time_since_sweep",
    "atr_at_decision",
    "adx_at_decision",
    "ema_fast",
    "ema_slow",
    "ema_slope",
    "vwap",
    "vwap_distance",
    "hour_utc",
    "weekday",
    "month",
    "contract",
    "passed",
    "rejected",
    "rejection_stage",
    "rejection_reason",
    "rejection_category",
    "rule_name",
    "threshold_value",
    "observed_value",
    "accepted_into_final_candidate",
    "accepted_into_trade",
]

RAW_SETUP_OUTCOME_COLUMNS = {
    "linked_trade_id",
    "pnl_usd",
    "r_multiple",
    "exit_reason",
    "mfe_r",
    "mae_r",
    "holding_minutes",
}

RAW_SETUP_CSV_COLUMNS = [
    *RAW_SETUP_ENTRY_FEATURE_COLUMNS,
    "linked_trade_id",
    "pnl_usd",
    "r_multiple",
    "exit_reason",
    "mfe_r",
    "mae_r",
    "holding_minutes",
]


@dataclass(slots=True)
class RawSetupEvent:
    """Decision-time raw setup event plus isolated optional outcome fields.

    The fields in ``RAW_SETUP_ENTRY_FEATURE_COLUMNS`` are the only fields that
    may be used for entry-time ML features.  Outcome fields are populated only
    after simulation and must stay excluded from preprocessing.
    """

    event_id: str
    timestamp: datetime
    symbol: str
    timeframe: str
    session: date
    direction_candidate: str
    setup_stage: str
    setup_type: str
    confluence_type: str = ""
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: int | None = None
    asian_high: float | None = None
    asian_low: float | None = None
    asian_mid: float | None = None
    distance_to_asian_high: float | None = None
    distance_to_asian_low: float | None = None
    distance_to_asian_mid: float | None = None
    range_width_points: float | None = None
    range_width_atr: float | None = None
    sweep_side: str = ""
    sweep_depth_points: float | None = None
    sweep_depth_atr: float | None = None
    time_since_sweep: int | None = None
    atr_at_decision: float | None = None
    adx_at_decision: float | None = None
    ema_fast: float | None = None
    ema_slow: float | None = None
    ema_slope: float | None = None
    vwap: float | None = None
    vwap_distance: float | None = None
    hour_utc: int | None = None
    weekday: str = ""
    month: int | None = None
    contract: str = ""
    passed: bool = False
    rejected: bool = True
    rejection_stage: str = ""
    rejection_reason: str = ""
    rejection_category: str = ""
    rule_name: str = ""
    threshold_value: float | None = None
    observed_value: float | None = None
    accepted_into_final_candidate: bool = False
    accepted_into_trade: bool = False
    linked_trade_id: str = ""
    pnl_usd: float | None = None
    r_multiple: float | None = None
    exit_reason: str = ""
    mfe_r: float | None = None
    mae_r: float | None = None
    holding_minutes: float | None = None


def assert_no_raw_outcome_columns(columns: Iterable[str]) -> None:
    overlap = sorted(set(columns) & RAW_SETUP_OUTCOME_COLUMNS)
    if overlap:
        raise ValueError(
            "Raw setup feature matrix contains post-trade outcome columns: "
            + ", ".join(overlap)
        )


def with_trade_outcome(
    event: RawSetupEvent,
    *,
    linked_trade_id: str,
    pnl_usd: float,
    r_multiple: float,
    exit_reason: str,
    mfe_r: float,
    mae_r: float,
    holding_minutes: float,
) -> RawSetupEvent:
    return replace(
        event,
        accepted_into_trade=True,
        linked_trade_id=linked_trade_id,
        pnl_usd=pnl_usd,
        r_multiple=r_multiple,
        exit_reason=exit_reason,
        mfe_r=mfe_r,
        mae_r=mae_r,
        holding_minutes=holding_minutes,
    )


def with_account_rejection(
    event: RawSetupEvent,
    *,
    rejection_reason: str,
    rejection_stage: str = "account_gate",
    rejection_category: str = "account_rejection",
) -> RawSetupEvent:
    return replace(
        event,
        passed=False,
        rejected=True,
        accepted_into_trade=False,
        rejection_stage=rejection_stage,
        rejection_reason=rejection_reason,
        rejection_category=rejection_category,
        linked_trade_id="",
        pnl_usd=None,
        r_multiple=None,
        exit_reason="",
        mfe_r=None,
        mae_r=None,
        holding_minutes=None,
    )


def write_raw_setup_csv(records: list[RawSetupEvent], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert_no_raw_outcome_columns(RAW_SETUP_ENTRY_FEATURE_COLUMNS)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_SETUP_CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def _pct(part: int, total: int) -> float:
    return float(part / total) if total else 0.0


def write_rejection_summary(records: list[RawSetupEvent], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rejection_stage",
        "rejection_reason",
        "symbol",
        "direction_candidate",
        "session",
        "confluence_type",
        "setup_type",
        "count",
        "percent_of_raw_setups",
        "percent_passed_to_next_stage",
        "accepted_trade_count",
        "accepted_trade_rate",
        "avg_pnl_usd",
        "avg_r_multiple",
    ]
    total = len(records)
    groups: dict[tuple[str, str, str, str, date, str, str], list[RawSetupEvent]] = {}
    for record in records:
        key = (
            record.rejection_stage,
            record.rejection_reason,
            record.symbol,
            record.direction_candidate,
            record.session,
            record.confluence_type,
            record.setup_type,
        )
        groups.setdefault(key, []).append(record)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key, group in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            accepted = [r for r in group if r.accepted_into_trade]
            pnl_values = [r.pnl_usd for r in accepted if r.pnl_usd is not None]
            r_values = [r.r_multiple for r in accepted if r.r_multiple is not None]
            writer.writerow(
                {
                    "rejection_stage": key[0],
                    "rejection_reason": key[1],
                    "symbol": key[2],
                    "direction_candidate": key[3],
                    "session": key[4],
                    "confluence_type": key[5],
                    "setup_type": key[6],
                    "count": len(group),
                    "percent_of_raw_setups": _pct(len(group), total),
                    "percent_passed_to_next_stage": _pct(
                        sum(1 for r in group if r.passed), len(group)
                    ),
                    "accepted_trade_count": len(accepted),
                    "accepted_trade_rate": _pct(len(accepted), len(group)),
                    "avg_pnl_usd": sum(pnl_values) / len(pnl_values) if pnl_values else "",
                    "avg_r_multiple": sum(r_values) / len(r_values) if r_values else "",
                }
            )
