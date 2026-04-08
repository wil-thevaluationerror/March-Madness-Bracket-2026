from __future__ import annotations

import html
import json
from collections import Counter
from dataclasses import asdict, is_dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from backtest.engine import BacktestResult
from models.orders import OrderIntent


def _coerce(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return {key: _coerce(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _coerce(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _event_type(record: dict[str, Any]) -> str | None:
    event_type = record.get("event_type")
    if event_type is not None:
        return str(event_type)
    legacy_event = record.get("event")
    if legacy_event is not None:
        return str(legacy_event)
    return None


CENTRAL_TZ = ZoneInfo("America/Chicago")


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value)
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _time_bucket_central(value: Any) -> str:
    dt_value = _parse_dt(value)
    if dt_value is None:
        return "UNKNOWN"
    if dt_value.tzinfo is None:
        dt_value = dt_value.replace(tzinfo=CENTRAL_TZ)
    else:
        dt_value = dt_value.astimezone(CENTRAL_TZ)
    bucket_time = dt_value.time()
    if time(8, 30) <= bucket_time < time(10, 30):
        return "OPEN"
    if time(10, 30) <= bucket_time < time(12, 30):
        return "MIDDAY"
    if time(12, 30) <= bucket_time < time(14, 45):
        return "AFTERNOON"
    if time(14, 45) <= bucket_time < time(15, 0):
        return "LATE"
    return "OUTSIDE_SESSION"


def _extract_intent_context(events: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    intent_context: dict[str, dict[str, Any]] = {}
    order_to_intent: dict[str, str] = {}
    for record in events:
        event_type = _event_type(record)
        payload = record.get("payload", {})
        if event_type == "signal_emitted":
            intent = payload.get("intent", {})
            intent_id = intent.get("intent_id")
            if intent_id:
                intent_context[str(intent_id)] = {
                    "regime": intent.get("regime"),
                    "signal_ts": intent.get("signal_ts"),
                    "symbol": intent.get("symbol"),
                    "side": intent.get("side"),
                    "signal_score": intent.get("signal_score"),
                }
        elif event_type == "entry_submitted":
            entry = payload.get("entry", {})
            intent = payload.get("intent", {})
            order_id = entry.get("order_id")
            intent_id = entry.get("intent_id") or intent.get("intent_id")
            if order_id and intent_id:
                order_to_intent[str(order_id)] = str(intent_id)
    return intent_context, order_to_intent


def _build_performance_matrix(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matrix: dict[tuple[str, str], dict[str, Any]] = {}
    for trade in trades:
        regime = str(trade.get("regime") or "UNKNOWN")
        time_bucket = str(trade.get("time_bucket") or "UNKNOWN")
        key = (regime, time_bucket)
        row = matrix.setdefault(
            key,
            {
                "regime": regime,
                "time_bucket": time_bucket,
                "trade_count": 0,
                "win_count": 0,
                "loss_count": 0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "win_rate": 0.0,
            },
        )
        pnl = float(trade.get("total_trade_pnl", trade.get("pnl", 0.0)) or 0.0)
        row["trade_count"] += 1
        row["total_pnl"] += pnl
        if pnl > 0:
            row["win_count"] += 1
        elif pnl < 0:
            row["loss_count"] += 1
    rows = []
    for row in matrix.values():
        count = row["trade_count"]
        row["avg_pnl"] = row["total_pnl"] / count if count else 0.0
        row["win_rate"] = row["win_count"] / count if count else 0.0
        rows.append(row)
    return sorted(rows, key=lambda item: (item["regime"], item["time_bucket"]))


def _build_trade_rows(result: BacktestResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trade in result.trades:
        rows.append(
            {
                "intent_id": trade.intent_id,
                "symbol": trade.symbol,
                "side": trade.side,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "qty": trade.qty,
                "pnl": trade.pnl,
                "entry_ts": _coerce(trade.entry_ts),
                "exit_ts": _coerce(trade.exit_ts),
                "exit_reason": trade.exit_reason,
                "regime": trade.regime,
                "time_bucket": trade.time_bucket,
                "is_stacked": trade.is_stacked,
                "stacking_reason": trade.stacking_reason or "",
                "risk_multiplier": trade.risk_multiplier,
                "position_scale": trade.position_scale,
                "cluster_risk_usage_pct": trade.cluster_risk_usage_pct,
                "mae": trade.mae,
                "mfe": trade.mfe,
                "is_reentry": trade.is_reentry,
                "entry_number_in_trend": trade.entry_number_in_trend,
                "trend_key": trade.trend_key,
                "atr_at_entry": trade.atr_at_entry,
                "stop_distance": trade.stop_distance,
                "stop_distance_pct": trade.stop_distance_pct,
                "stop_distance_ticks": trade.stop_distance_ticks,
                "abnormal_stop": trade.abnormal_stop,
                "path": f"{trade.entry_price} -> {trade.exit_price}",
            }
        )
    return rows


def _build_equity_curve(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    curve: list[dict[str, Any]] = []
    cumulative_pnl = 0.0
    peak = 0.0
    for trade in sorted(trades, key=lambda item: str(item.get("exit_ts", ""))):
        cumulative_pnl += float(trade.get("pnl", 0.0) or 0.0)
        peak = max(peak, cumulative_pnl)
        drawdown = cumulative_pnl - peak
        curve.append(
            {
                "exit_ts": trade.get("exit_ts"),
                "cumulative_pnl": cumulative_pnl,
                "drawdown": drawdown,
                "trade_pnl": trade.get("pnl", 0.0),
            }
        )
    return curve


def _build_trade_metrics(trades: list[dict[str, Any]]) -> dict[str, float]:
    total_trades = len(trades)
    total_pnl = sum(float(trade.get("pnl", 0.0) or 0.0) for trade in trades)
    wins = sum(1 for trade in trades if float(trade.get("pnl", 0.0) or 0.0) > 0)
    avg_pnl = total_pnl / total_trades if total_trades else 0.0
    win_rate = wins / total_trades if total_trades else 0.0
    equity_curve = _build_equity_curve(trades)
    max_drawdown = min((point["drawdown"] for point in equity_curve), default=0.0)
    return {
        "total_trades": float(total_trades),
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "total_pnl": total_pnl,
        "max_drawdown": max_drawdown,
        "expectancy": avg_pnl,
    }


def _intent_bar_key(intent: OrderIntent) -> str:
    return _coerce(_parse_dt(intent.signal_ts) or intent.signal_ts)


def _build_signal_quality(intents: list[OrderIntent], result: BacktestResult) -> dict[str, Any]:
    approved_ids = set(result.approved_intents)
    denied_ids = set(result.denied_intents)
    approved = [intent for intent in intents if intent.intent_id in approved_ids]
    rejected = [intent for intent in intents if intent.intent_id in denied_ids]

    def _avg_score(items: list[OrderIntent]) -> float:
        if not items:
            return 0.0
        return sum(float(intent.signal_score) for intent in items) / len(items)

    histogram_bins = [
        {"label": "0.0-0.5", "min": 0.0, "max": 0.5},
        {"label": "0.5-1.0", "min": 0.5, "max": 1.0},
        {"label": "1.0-1.5", "min": 1.0, "max": 1.5},
        {"label": "1.5-2.0", "min": 1.5, "max": 2.0},
        {"label": "2.0+", "min": 2.0, "max": None},
    ]
    histogram_rows: list[dict[str, Any]] = []
    for spec in histogram_bins:
        lower = float(spec["min"])
        upper = spec["max"]
        approved_count = sum(
            1
            for intent in approved
            if float(intent.signal_score) >= lower and (upper is None or float(intent.signal_score) < float(upper))
        )
        rejected_count = sum(
            1
            for intent in rejected
            if float(intent.signal_score) >= lower and (upper is None or float(intent.signal_score) < float(upper))
        )
        histogram_rows.append(
            {
                "bin": spec["label"],
                "accepted": approved_count,
                "rejected": rejected_count,
            }
        )

    approved_by_bar: dict[str, list[OrderIntent]] = {}
    for intent in approved:
        approved_by_bar.setdefault(_intent_bar_key(intent), []).append(intent)
    higher_score_rejected = 0
    for intent in rejected:
        same_bar = approved_by_bar.get(_intent_bar_key(intent), [])
        if same_bar and float(intent.signal_score) > min(float(candidate.signal_score) for candidate in same_bar):
            higher_score_rejected += 1

    return {
        "avg_score_accepted": _avg_score(approved),
        "avg_score_rejected": _avg_score(rejected),
        "accepted_count": len(approved),
        "rejected_count": len(rejected),
        "higher_score_rejected_count": higher_score_rejected,
        "higher_score_rejected_pct": (higher_score_rejected / len(rejected)) * 100.0 if rejected else 0.0,
        "score_histogram": histogram_rows,
    }


def _build_position_constraint_analysis(
    single_summary: dict[str, Any],
    concurrency_analysis: dict[int, dict[str, Any]] | None,
) -> dict[str, Any]:
    blocked_signals = int(single_summary.get("position_open_blocked_signals", 0) or 0)
    analysis = {
        "position_open_blocked_signals": blocked_signals,
        "percentage_blocked": float(single_summary.get("position_open_percentage_blocked", 0.0) or 0.0),
        "single_position": {
            "trades": int(single_summary.get("total_trades", 0) or 0),
            "pnl": float(single_summary.get("total_pnl", 0.0) or 0.0),
            "expectancy": float(single_summary.get("expectancy", 0.0) or 0.0),
            "max_drawdown": float(single_summary.get("max_drawdown", 0.0) or 0.0),
        },
    }
    if not concurrency_analysis:
        analysis["levels"] = []
        return analysis
    base = concurrency_analysis.get(1, {})
    levels: list[dict[str, Any]] = []
    for level in sorted(concurrency_analysis):
        metrics = dict(concurrency_analysis[level])
        levels.append(
            {
                "level": level,
                "trades": int(metrics.get("trades", 0) or 0),
                "pnl": float(metrics.get("pnl", 0.0) or 0.0),
                "expectancy": float(metrics.get("expectancy", 0.0) or 0.0),
                "win_rate": float(metrics.get("win_rate", 0.0) or 0.0),
                "max_drawdown": float(metrics.get("max_drawdown", 0.0) or 0.0),
                "blocked_signals": int(metrics.get("blocked_signals", 0) or 0),
                "blocked_percentage": float(metrics.get("blocked_percentage", 0.0) or 0.0),
                "pnl_delta_vs_1": float(metrics.get("pnl", 0.0) or 0.0) - float(base.get("pnl", 0.0) or 0.0),
                "trade_delta_vs_1": int(metrics.get("trades", 0) or 0) - int(base.get("trades", 0) or 0),
                "avg_concurrent_positions": float(metrics.get("avg_concurrent_positions", 0.0) or 0.0),
                "max_concurrent_positions": int(metrics.get("max_concurrent_positions", 0) or 0),
                "avg_overlap_duration_minutes": float(metrics.get("avg_overlap_duration_minutes", 0.0) or 0.0),
                "overlap_pnl_correlation": float(metrics.get("overlap_pnl_correlation", 0.0) or 0.0),
                "worst_clustered_loss": float(metrics.get("worst_clustered_loss", 0.0) or 0.0),
                "worst_clustered_loss_trade_count": int(metrics.get("worst_clustered_loss_trade_count", 0) or 0),
                "max_simultaneous_drawdown": float(metrics.get("max_simultaneous_drawdown", 0.0) or 0.0),
            }
        )
    analysis["levels"] = levels
    return analysis


def _build_exposure_control_summary(
    summary: dict[str, Any],
    denial_counts: Counter[str],
    concurrency_analysis: dict[int, dict[str, Any]] | None,
) -> dict[str, Any]:
    generated = int(summary.get("generated_intents", 0) or 0)

    def _blocked(reason: str) -> dict[str, float | int]:
        count = int(denial_counts.get(reason, 0) or 0)
        return {
            "count": count,
            "percentage": (count / generated) * 100.0 if generated else 0.0,
        }

    baseline = {}
    if concurrency_analysis:
        baseline = dict(concurrency_analysis.get(int(summary.get("max_concurrent_positions", 2)), {}))
    before_drawdown = float(baseline.get("max_drawdown", summary.get("max_drawdown", 0.0)) or 0.0)
    before_trades = int(baseline.get("trades", summary.get("total_trades", 0)) or 0)

    return {
        "avg_concurrent_positions": float(summary.get("avg_concurrent_positions", 0.0) or 0.0),
        "max_observed_concurrent_positions": int(summary.get("max_observed_concurrent_positions", 0) or 0),
        "blocked_by_rule": {
            "same_direction_blocked": _blocked("same_direction_blocked"),
            "stacking_blocked_regime": _blocked("stacking_blocked_regime"),
            "spacing_limit": _blocked("spacing_limit"),
            "regime_limit": _blocked("regime_limit"),
            "risk_budget_exceeded": _blocked("risk_budget_exceeded"),
            "max_positions_reached": _blocked("max_positions_reached"),
        },
        "drawdown_comparison": {
            "before_exposure_controls": before_drawdown,
            "after_exposure_controls": float(summary.get("max_drawdown", 0.0) or 0.0),
            "delta": float(summary.get("max_drawdown", 0.0) or 0.0) - before_drawdown,
        },
        "trade_comparison": {
            "before_exposure_controls": before_trades,
            "after_exposure_controls": int(summary.get("total_trades", 0) or 0),
            "delta": int(summary.get("total_trades", 0) or 0) - before_trades,
        },
    }


def _max_drawdown_from_trades(trades: list[dict[str, Any]]) -> float:
    return min((point["drawdown"] for point in _build_equity_curve(trades)), default=0.0)


def _build_direction_stacking_summary(
    trades: list[dict[str, Any]],
    denied_reason_counts: Counter[str],
) -> dict[str, Any]:
    stacked_trades = [trade for trade in trades if bool(trade.get("is_stacked"))]
    non_stacked_trades = [trade for trade in trades if not bool(trade.get("is_stacked"))]
    stacked_pnl = sum(float(trade.get("pnl", 0.0) or 0.0) for trade in stacked_trades)
    same_direction_allowed = len(stacked_trades)
    same_direction_blocked = int(denied_reason_counts.get("same_direction_blocked", 0) or 0) + int(
        denied_reason_counts.get("stacking_blocked_regime", 0) or 0
    ) + int(denied_reason_counts.get("stacking_signal_threshold", 0) or 0) + int(
        denied_reason_counts.get("stacking_disabled_recent_pnl", 0) or 0
    ) + int(
        denied_reason_counts.get("stacking_disabled_loss_cluster", 0) or 0
    ) + int(
        denied_reason_counts.get("stacking_volatility_blocked", 0) or 0
    ) + int(
        denied_reason_counts.get("stacked_risk_budget_exceeded", 0) or 0
    ) + int(denied_reason_counts.get("stack_depth_reached", 0) or 0)
    same_direction_candidates = same_direction_allowed + same_direction_blocked
    stacked_entries_sorted = sorted(
        (_parse_dt(trade.get("entry_ts")) for trade in stacked_trades if _parse_dt(trade.get("entry_ts")) is not None),
        key=lambda dt: dt or datetime.min,
    )
    entry_distances: list[float] = []
    for earlier, later in zip(stacked_entries_sorted, stacked_entries_sorted[1:]):
        if earlier is not None and later is not None:
            entry_distances.append((later - earlier).total_seconds() / 60.0)
    stacking_reasons: Counter[str] = Counter()
    for trade in stacked_trades:
        stacking_reasons[str(trade.get("stacking_reason") or "unknown")] += 1
    stacking_reasons["regime"] += int(denied_reason_counts.get("stacking_blocked_regime", 0) or 0)
    return {
        "same_direction_allowed": same_direction_allowed,
        "same_direction_blocked": same_direction_blocked,
        "acceptance_rate": (same_direction_allowed / same_direction_candidates) if same_direction_candidates else 0.0,
        "stacked_trade_percentage": (same_direction_allowed / len(trades)) * 100.0 if trades else 0.0,
        "stacked_pnl": stacked_pnl,
        "stacked_drawdown": _max_drawdown_from_trades(stacked_trades),
        "non_stacked_drawdown": _max_drawdown_from_trades(non_stacked_trades),
        "avg_distance_between_stacked_entries_minutes": (
            sum(entry_distances) / len(entry_distances) if entry_distances else 0.0
        ),
        "price_activation_rate": (
            stacking_reasons.get("price", 0) / same_direction_allowed if same_direction_allowed else 0.0
        ),
        "stacking_reason_counts": dict(stacking_reasons),
    }


def _build_stacking_risk_control_summary(
    trades: list[dict[str, Any]],
    denied_reason_counts: Counter[str],
    result: BacktestResult,
) -> dict[str, Any]:
    stacked_trades = [trade for trade in trades if bool(trade.get("is_stacked"))]
    non_stacked_trades = [trade for trade in trades if not bool(trade.get("is_stacked"))]
    stacked_pnl = sum(float(trade.get("pnl", 0.0) or 0.0) for trade in stacked_trades)
    non_stacked_pnl = sum(float(trade.get("pnl", 0.0) or 0.0) for trade in non_stacked_trades)
    loss_cluster = 0.0
    worst_loss_cluster = 0.0
    for trade in sorted(stacked_trades, key=lambda row: str(row.get("exit_ts", ""))):
        pnl = float(trade.get("pnl", 0.0) or 0.0)
        if pnl < 0:
            loss_cluster += pnl
            worst_loss_cluster = min(worst_loss_cluster, loss_cluster)
        else:
            loss_cluster = 0.0
    regime_distribution = Counter(str(trade.get("regime") or "UNKNOWN") for trade in stacked_trades)
    blocked_reasons = {
        reason: int(denied_reason_counts.get(reason, 0) or 0)
        for reason in (
            "stacked_risk_budget_exceeded",
            "stacking_disabled_recent_pnl",
            "stacking_disabled_loss_cluster",
            "stacking_volatility_blocked",
            "stacking_blocked_regime",
            "stacking_signal_threshold",
        )
    }
    return {
        "avg_stacked_heat_usage_pct": float(result.avg_stacked_heat_usage_pct),
        "max_stacked_heat_usage_pct": float(result.max_stacked_heat_usage_pct),
        "stacked_pnl": stacked_pnl,
        "non_stacked_pnl": non_stacked_pnl,
        "stacked_drawdown": _max_drawdown_from_trades(stacked_trades),
        "non_stacked_drawdown": _max_drawdown_from_trades(non_stacked_trades),
        "worst_stacked_loss_cluster": worst_loss_cluster,
        "stacking_disabled_events": int(result.stacking_disabled_events),
        "stacking_disabled_bar_percentage": float(result.stacking_disabled_bar_percentage),
        "stacked_regime_distribution": dict(regime_distribution),
        "blocked_reasons": blocked_reasons,
    }


def _build_risk_shaping_summary(
    trades: list[dict[str, Any]],
    denied_reason_counts: Counter[str],
    result: BacktestResult,
) -> dict[str, Any]:
    stacked_trades = [trade for trade in trades if bool(trade.get("is_stacked"))]
    non_stacked_trades = [trade for trade in trades if not bool(trade.get("is_stacked"))]
    return {
        "avg_risk_multiplier": float(result.avg_risk_multiplier),
        "avg_position_scale": float(result.avg_position_scale),
        "max_cluster_risk_usage_pct": float(result.max_cluster_risk_usage_pct),
        "stacked_drawdown": _max_drawdown_from_trades(stacked_trades),
        "non_stacked_drawdown": _max_drawdown_from_trades(non_stacked_trades),
        "cluster_cap_blocked": int(denied_reason_counts.get("cluster_risk_exceeded", 0) or 0),
        "midday_stacking_blocked": int(denied_reason_counts.get("midday_stacking_blocked", 0) or 0),
        "cluster_risk_usage_curve": list(result.cluster_risk_usage_curve),
        "risk_multiplier_curve": list(result.risk_multiplier_curve),
        "avg_trade_risk_multiplier": (
            sum(float(trade.get("risk_multiplier", 0.0) or 0.0) for trade in trades) / len(trades) if trades else 0.0
        ),
        "avg_trade_position_scale": (
            sum(float(trade.get("position_scale", 0.0) or 0.0) for trade in trades) / len(trades) if trades else 0.0
        ),
    }


def _build_reentry_summary(trades: list[dict[str, Any]], result: BacktestResult) -> dict[str, Any]:
    reentry_trades = [trade for trade in trades if bool(trade.get("is_reentry"))]
    first_entry_trades = [trade for trade in trades if not bool(trade.get("is_reentry"))]
    return {
        "reentry_count": int(result.reentry_count),
        "reentry_pnl": float(result.reentry_pnl),
        "initial_entry_pnl": float(result.initial_entry_pnl),
        "reentry_win_rate": float(result.reentry_win_rate),
        "reentry_drawdown": float(result.reentry_drawdown),
        "first_entry_drawdown": _max_drawdown_from_trades(first_entry_trades),
        "avg_minutes_between_entries": float(result.avg_minutes_between_entries),
        "trades_per_trend": dict(result.trades_per_trend),
        "reentry_trade_percentage": (len(reentry_trades) / len(trades)) * 100.0 if trades else 0.0,
    }


def _build_stop_atr_diagnostics(
    trades: list[dict[str, Any]],
    result: BacktestResult,
    data_diagnostics: dict[str, Any] | None,
) -> dict[str, Any]:
    diagnostics = dict(result.stop_diagnostics)
    diagnostics["selected_symbol"] = (data_diagnostics or {}).get("selected_symbol")
    diagnostics["price_jump_flag_count"] = int((data_diagnostics or {}).get("price_jump_flag_count", 0) or 0)
    diagnostics["price_jump_flags"] = list((data_diagnostics or {}).get("price_jump_flags", []))
    diagnostics["largest_atr_trades"] = list(diagnostics.get("largest_atr_trades", []))
    diagnostics["flagged_trades"] = list(diagnostics.get("flagged_trades", []))
    return diagnostics


def _render_equity_curve_svg(equity_curve: list[dict[str, Any]]) -> str:
    if not equity_curve:
        return "<p class='empty'>No equity curve available.</p>"
    width = 760
    height = 240
    values = [float(point["cumulative_pnl"]) for point in equity_curve]
    min_value = min(values)
    max_value = max(values)
    span = max(max_value - min_value, 1.0)
    points: list[str] = []
    for index, value in enumerate(values):
        x = 20 + (index / max(len(values) - 1, 1)) * (width - 40)
        y = height - 20 - ((value - min_value) / span) * (height - 40)
        points.append(f"{x:.2f},{y:.2f}")
    return (
        f"<svg viewBox='0 0 {width} {height}' width='100%' height='{height}' role='img' aria-label='Equity curve'>"
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='#fffdf8' stroke='#d6d1c4' rx='12' />"
        f"<polyline fill='none' stroke='#0f766e' stroke-width='3' points='{' '.join(points)}' />"
        f"<text x='24' y='24' fill='#6a6f78' font-size='12'>Min {min_value:.2f}</text>"
        f"<text x='{width - 140}' y='24' fill='#6a6f78' font-size='12'>Max {max_value:.2f}</text>"
        "</svg>"
    )


def _render_curve_svg(points_data: list[dict[str, Any]], key: str, *, stroke: str, label: str) -> str:
    if not points_data:
        return "<p class='empty'>No curve available.</p>"
    width = 760
    height = 180
    values = [float(point.get(key, 0.0) or 0.0) for point in points_data]
    min_value = min(values)
    max_value = max(values)
    span = max(max_value - min_value, 1.0)
    points: list[str] = []
    for index, value in enumerate(values):
        x = 20 + (index / max(len(values) - 1, 1)) * (width - 40)
        y = height - 20 - ((value - min_value) / span) * (height - 40)
        points.append(f"{x:.2f},{y:.2f}")
    return (
        f"<svg viewBox='0 0 {width} {height}' width='100%' height='{height}' role='img' aria-label='{html.escape(label)}'>"
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='#fffdf8' stroke='#d6d1c4' rx='12' />"
        f"<polyline fill='none' stroke='{stroke}' stroke-width='3' points='{' '.join(points)}' />"
        f"<text x='24' y='24' fill='#6a6f78' font-size='12'>Min {min_value:.2f}</text>"
        f"<text x='{width - 140}' y='24' fill='#6a6f78' font-size='12'>Max {max_value:.2f}</text>"
        "</svg>"
    )


def build_dashboard_payload(
    *,
    intents: list[OrderIntent],
    result: BacktestResult,
    events_path: Path,
    trade_ledger_path: Path,
    comparison_payload: dict[str, Any] | None = None,
    data_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    events = _read_jsonl(events_path)
    trade_ledger = _read_jsonl(trade_ledger_path)
    intent_context, order_to_intent = _extract_intent_context(events)

    denied_reason_counts: Counter[str] = Counter()
    denied_samples: list[dict[str, Any]] = []
    filled_samples: list[dict[str, Any]] = []

    for record in events:
        event_type = _event_type(record)
        payload = record.get("payload", {})
        if event_type == "intent_denied":
            decision = payload.get("decision", {})
            intent = payload.get("intent", {})
            reason = str(decision.get("reason", "unknown"))
            denied_reason_counts[reason] += 1
            if len(denied_samples) < 25:
                denied_samples.append(
                    {
                        "intent_id": intent.get("intent_id"),
                        "symbol": intent.get("symbol"),
                        "signal_ts": intent.get("signal_ts"),
                        "signal_score": intent.get("signal_score"),
                        "entry_price": intent.get("entry_price"),
                        "stop_price": intent.get("stop_price"),
                        "target_price": intent.get("target_price"),
                        "reason": reason,
                        "regime": intent.get("regime"),
                    }
                )
        elif event_type == "fill":
            report = payload.get("report", {})
            order_id = report.get("order_id")
            intent_id = order_to_intent.get(str(order_id))
            context = intent_context.get(intent_id or "", {})
            if len(filled_samples) < 25:
                filled_samples.append(
                    {
                        "order_id": report.get("order_id"),
                        "intent_id": intent_id,
                        "symbol": report.get("symbol"),
                        "status": report.get("status"),
                        "fill_qty": report.get("fill_qty"),
                        "fill_price": report.get("fill_price"),
                        "side": report.get("side"),
                        "timestamp": report.get("timestamp"),
                        "regime": context.get("regime", "UNKNOWN"),
                        "time_bucket": _time_bucket_central(context.get("signal_ts") or report.get("timestamp")),
                        "signal_score": context.get("signal_score"),
                    }
                )

    total_intents = len(intents)
    denied_count = len(result.denied_intents)
    approved_count = len(result.approved_intents) if result.approved_intents else total_intents - denied_count
    fill_count = len(result.fills)
    concurrency_analysis = {int(level): dict(metrics) for level, metrics in result.concurrency_analysis.items()}
    trade_rows = _build_trade_rows(result)
    trade_metrics = _build_trade_metrics(trade_rows)
    equity_curve = _build_equity_curve(trade_rows)
    analytics_denial_counts = dict(result.denial_counts)
    analytics_denial_examples = dict(result.denial_examples)
    if analytics_denial_counts:
        denied_reason_counts = Counter(analytics_denial_counts)
    if analytics_denial_examples:
        denied_samples = []
        for reason, examples in analytics_denial_examples.items():
            for example in examples:
                if len(denied_samples) >= 25:
                    break
                denied_samples.append(
                    {
                        "intent_id": example.get("intent_id"),
                        "symbol": example.get("symbol"),
                        "signal_ts": example.get("signal_ts"),
                        "signal_score": example.get("signal_score"),
                        "entry_price": example.get("entry_price"),
                        "stop_price": example.get("stop_price"),
                        "target_price": example.get("target_price"),
                        "reason": reason,
                        "regime": example.get("regime"),
                    }
                )
            if len(denied_samples) >= 25:
                break

    summary = {
        "generated_intents": total_intents,
        "approved_intents": approved_count,
        "denied_intents": denied_count,
        "filled_orders": fill_count,
        "total_trades": int(trade_metrics["total_trades"]),
        "win_rate": trade_metrics["win_rate"],
        "avg_pnl": trade_metrics["avg_pnl"],
        "total_pnl": trade_metrics["total_pnl"],
        "max_drawdown": trade_metrics["max_drawdown"],
        "expectancy": trade_metrics["expectancy"],
        "approval_rate": approved_count / total_intents if total_intents else 0.0,
        "fill_rate": fill_count / total_intents if total_intents else 0.0,
        "max_concurrent_positions": result.max_concurrent_positions,
        "blocked_by_max_positions": result.blocked_by_max_positions,
        "percentage_blocked_by_max_positions": result.percentage_blocked_by_max_positions,
        "avg_concurrent_positions": result.avg_concurrent_positions,
        "max_observed_concurrent_positions": result.max_observed_concurrent_positions,
        "position_open_blocked_signals": int(concurrency_analysis.get(result.max_concurrent_positions, {}).get("blocked_signals", result.position_open_blocked_signals)),
        "position_open_percentage_blocked": float(concurrency_analysis.get(result.max_concurrent_positions, {}).get("blocked_percentage", result.percentage_blocked)),
        "top_denial_reasons": denied_reason_counts.most_common(10),
    }

    intent_samples = [
        {
            "intent_id": intent.intent_id,
            "symbol": intent.symbol,
            "signal_ts": _coerce(intent.signal_ts),
            "signal_score": intent.signal_score,
            "side": _coerce(intent.side),
            "entry_price": intent.entry_price,
            "stop_price": intent.stop_price,
            "target_price": intent.target_price,
            "regime": _coerce(intent.regime),
        }
        for intent in intents[:25]
    ]

    enriched_trade_ledger: list[dict[str, Any]] = []
    if trade_rows:
        enriched_trade_ledger = list(trade_rows)
    else:
        for record in trade_ledger:
            payload = dict(record.get("payload", {}))
            report = payload.get("report", {})
            intent_id = payload.get("intent_id")
            context = intent_context.get(str(intent_id), {})
            signal_ts = payload.get("signal_ts") or context.get("signal_ts")
            enriched_trade_ledger.append(
                {
                    "timestamp": record.get("timestamp"),
                    "symbol": payload.get("symbol"),
                    "intent_id": intent_id,
                    "strategy_id": payload.get("strategy_id"),
                    "reason": payload.get("reason"),
                    "pnl": payload.get("pnl"),
                    "total_trade_pnl": payload.get("total_trade_pnl"),
                    "fill_price": report.get("fill_price"),
                    "fill_qty": report.get("fill_qty"),
                    "report_ts": report.get("timestamp"),
                    "regime": payload.get("regime") or context.get("regime") or "UNKNOWN",
                    "time_bucket": _time_bucket_central(signal_ts or report.get("timestamp")),
                    "signal_ts": signal_ts,
                }
            )

    performance_matrix = _build_performance_matrix(enriched_trade_ledger)
    position_constraint_analysis = _build_position_constraint_analysis(
        summary,
        concurrency_analysis or None,
    )
    exposure_control = _build_exposure_control_summary(summary, denied_reason_counts, concurrency_analysis or None)
    direction_stacking = _build_direction_stacking_summary(trade_rows, denied_reason_counts)
    stacking_risk_control = _build_stacking_risk_control_summary(trade_rows, denied_reason_counts, result)
    risk_shaping = _build_risk_shaping_summary(trade_rows, denied_reason_counts, result)
    reentry_analysis = _build_reentry_summary(trade_rows, result)
    stop_atr_diagnostics = _build_stop_atr_diagnostics(trade_rows, result, data_diagnostics)
    signal_quality = _build_signal_quality(intents, result)
    top_rejected_signals = sorted(
        [
            {
                "intent_id": intent.intent_id,
                "symbol": intent.symbol,
                "signal_ts": _coerce(intent.signal_ts),
                "signal_score": intent.signal_score,
                "entry_price": intent.entry_price,
                "stop_price": intent.stop_price,
                "target_price": intent.target_price,
                "regime": _coerce(intent.regime),
            }
            for intent in intents
            if intent.intent_id in set(result.denied_intents)
        ],
        key=lambda row: (float(row["signal_score"]), str(row["intent_id"])),
        reverse=True,
    )[:10]
    concurrency_validation_rows: list[dict[str, Any]] = []
    concurrency_time_rows: list[dict[str, Any]] = []
    concurrency_curves: list[dict[str, Any]] = []
    for level in sorted(concurrency_analysis):
        metrics = dict(concurrency_analysis[level])
        concurrency_validation_rows.append(
            {
                "level": level,
                "avg_concurrent_positions": round(float(metrics.get("avg_concurrent_positions", 0.0) or 0.0), 3),
                "max_concurrent_positions": int(metrics.get("max_concurrent_positions", 0) or 0),
                "avg_overlap_duration_minutes": round(float(metrics.get("avg_overlap_duration_minutes", 0.0) or 0.0), 3),
                "overlap_pair_count": int(metrics.get("overlap_pair_count", 0) or 0),
                "overlap_pnl_correlation": round(float(metrics.get("overlap_pnl_correlation", 0.0) or 0.0), 4),
                "worst_clustered_loss": round(float(metrics.get("worst_clustered_loss", 0.0) or 0.0), 2),
                "worst_clustered_loss_trade_count": int(metrics.get("worst_clustered_loss_trade_count", 0) or 0),
                "max_simultaneous_drawdown": round(float(metrics.get("max_simultaneous_drawdown", 0.0) or 0.0), 2),
            }
        )
        for bucket, bucket_metrics in sorted(dict(metrics.get("time_of_day", {})).items()):
            concurrency_time_rows.append(
                {
                    "level": level,
                    "time_bucket": bucket,
                    "trades": int(bucket_metrics.get("trades", 0) or 0),
                    "pnl": round(float(bucket_metrics.get("pnl", 0.0) or 0.0), 2),
                    "expectancy": round(float(bucket_metrics.get("expectancy", 0.0) or 0.0), 4),
                    "win_rate": round(float(bucket_metrics.get("win_rate", 0.0) or 0.0), 4),
                }
            )
        concurrency_curves.append(
            {
                "level": level,
                "equity_svg": _render_curve_svg(
                    list(metrics.get("equity_curve", [])),
                    "cumulative_pnl",
                    stroke="#0f766e",
                    label=f"Equity curve level {level}",
                ),
                "drawdown_svg": _render_curve_svg(
                    list(metrics.get("drawdown_curve", [])),
                    "drawdown",
                    stroke="#b42318",
                    label=f"Drawdown curve level {level}",
                ),
                "overlap_distribution": list(metrics.get("overlap_distribution", [])),
            }
        )

    return {
        "summary": summary,
        "denial_counts": dict(denied_reason_counts),
        "denial_examples": analytics_denial_examples,
        "intent_samples": intent_samples,
        "denied_samples": denied_samples,
        "filled_samples": filled_samples,
        "trade_ledger": enriched_trade_ledger[:25],
        "trades": enriched_trade_ledger[:100],
        "performance_matrix": performance_matrix,
        "concurrency_analysis": concurrency_analysis,
        "position_constraint_analysis": position_constraint_analysis,
        "exposure_control": exposure_control,
        "direction_stacking": direction_stacking,
        "stacking_risk_control": stacking_risk_control,
        "risk_shaping": risk_shaping,
        "reentry_analysis": reentry_analysis,
        "stop_atr_diagnostics": stop_atr_diagnostics,
        "signal_quality": signal_quality,
        "top_rejected_signals": top_rejected_signals,
        "concurrency_validation_rows": concurrency_validation_rows,
        "concurrency_time_rows": concurrency_time_rows,
        "concurrency_curves": concurrency_curves,
        "comparison": comparison_payload,
        "trade_metrics": trade_metrics,
        "equity_curve": equity_curve,
        "equity_curve_svg": _render_equity_curve_svg(equity_curve),
        "raw_result": _coerce(result),
    }


def write_dashboard(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = payload["summary"]

    def render_rows(rows: list[dict[str, Any]], columns: list[str]) -> str:
        if not rows:
            return "<p class='empty'>No rows available.</p>"
        header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
        body_rows = []
        for row in rows:
            cells = "".join(f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in columns)
            body_rows.append(f"<tr>{cells}</tr>")
        return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"

    denial_reason_rows = [
        {"reason": reason, "count": count}
        for reason, count in summary["top_denial_reasons"]
    ]
    position_analysis = payload["position_constraint_analysis"]
    exposure_control = payload["exposure_control"]
    direction_stacking = payload["direction_stacking"]
    stacking_risk_control = payload["stacking_risk_control"]
    risk_shaping = payload["risk_shaping"]
    reentry_analysis = payload["reentry_analysis"]
    stop_atr_diagnostics = payload["stop_atr_diagnostics"]
    signal_quality = payload["signal_quality"]
    concurrency_rows = position_analysis.get("levels", [])
    concurrency_curves = payload.get("concurrency_curves", [])

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Backtest Dashboard</title>
  <style>
    :root {{
      --bg: #f5f1e8;
      --panel: #fffdf8;
      --ink: #1b1d21;
      --muted: #6a6f78;
      --accent: #0f766e;
      --danger: #b42318;
      --border: #d6d1c4;
    }}
    body {{
      margin: 0;
      padding: 32px;
      background: linear-gradient(180deg, #f1eadb 0%, #f8f5ee 100%);
      color: var(--ink);
      font-family: Georgia, 'Times New Roman', serif;
    }}
    h1, h2 {{
      margin: 0 0 12px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin: 24px 0;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 18px;
      box-shadow: 0 8px 24px rgba(27, 29, 33, 0.06);
    }}
    .metric {{
      font-size: 2rem;
      font-weight: bold;
      margin-top: 8px;
    }}
    .muted {{
      color: var(--muted);
    }}
    .section {{
      margin-top: 28px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
      font-size: 0.95rem;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid #ece6d7;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #efe7d5;
    }}
    tr:last-child td {{
      border-bottom: none;
    }}
    .tag {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      background: #dff3ef;
      color: var(--accent);
      font-size: 0.85rem;
    }}
    .danger {{
      color: var(--danger);
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: #15171b;
      color: #f5f7fa;
      padding: 16px;
      border-radius: 12px;
      overflow-x: auto;
      font-size: 0.85rem;
    }}
    .empty {{
      color: var(--muted);
      font-style: italic;
    }}
  </style>
</head>
<body>
  <h1>Backtest Dashboard</h1>
  <p class="muted">Detailed execution report for the latest simulation run, including approvals, denials, fill samples, and raw summary output.</p>

  <div class="grid">
    <div class="card"><div class="muted">Generated Intents</div><div class="metric">{summary['generated_intents']}</div></div>
    <div class="card"><div class="muted">Approved Intents</div><div class="metric">{summary['approved_intents']}</div></div>
    <div class="card"><div class="muted">Denied Intents</div><div class="metric danger">{summary['denied_intents']}</div></div>
    <div class="card"><div class="muted">Filled Orders</div><div class="metric">{summary['filled_orders']}</div></div>
    <div class="card"><div class="muted">Total Trades</div><div class="metric">{summary['total_trades']}</div></div>
    <div class="card"><div class="muted">Win Rate</div><div class="metric">{summary['win_rate']:.2%}</div></div>
    <div class="card"><div class="muted">Average PnL</div><div class="metric">{summary['avg_pnl']:.2f}</div></div>
    <div class="card"><div class="muted">Total PnL</div><div class="metric">{summary['total_pnl']:.2f}</div></div>
    <div class="card"><div class="muted">Max Drawdown</div><div class="metric danger">{summary['max_drawdown']:.2f}</div></div>
    <div class="card"><div class="muted">Approval Rate</div><div class="metric">{summary['approval_rate']:.2%}</div></div>
    <div class="card"><div class="muted">Fill Rate</div><div class="metric">{summary['fill_rate']:.2%}</div></div>
    <div class="card"><div class="muted">Position-Cap Blocks</div><div class="metric">{summary['blocked_by_max_positions']}</div></div>
    <div class="card"><div class="muted">Blocked % Of Signals</div><div class="metric">{summary['position_open_percentage_blocked']:.2f}%</div></div>
  </div>

  <div class="section">
    <h2>Position Cap</h2>
    <div class="grid">
      <div class="card"><div class="muted">Configured Max Concurrent Positions</div><div class="metric">{summary['max_concurrent_positions']}</div></div>
      <div class="card"><div class="muted">Blocked By Cap</div><div class="metric">{summary['blocked_by_max_positions']}</div></div>
      <div class="card"><div class="muted">Blocked % By Cap</div><div class="metric">{summary['percentage_blocked_by_max_positions']:.2f}%</div></div>
      <div class="card"><div class="muted">Total Trades Executed</div><div class="metric">{summary['total_trades']}</div></div>
    </div>
  </div>

  <div class="section">
    <h2>Re-Entry Analysis</h2>
    <div class="grid">
      <div class="card"><div class="muted">Re-Entries</div><div class="metric">{reentry_analysis['reentry_count']}</div></div>
      <div class="card"><div class="muted">Re-Entry PnL</div><div class="metric">{reentry_analysis['reentry_pnl']:.2f}</div></div>
      <div class="card"><div class="muted">First Entry PnL</div><div class="metric">{reentry_analysis['initial_entry_pnl']:.2f}</div></div>
      <div class="card"><div class="muted">Re-Entry Win Rate</div><div class="metric">{reentry_analysis['reentry_win_rate']:.2%}</div></div>
      <div class="card"><div class="muted">Re-Entry Drawdown</div><div class="metric danger">{reentry_analysis['reentry_drawdown']:.2f}</div></div>
      <div class="card"><div class="muted">Avg Minutes Between Entries</div><div class="metric">{reentry_analysis['avg_minutes_between_entries']:.1f}</div></div>
    </div>
    {render_rows(
      [{'trend_key': key, 'trades': value} for key, value in sorted(reentry_analysis['trades_per_trend'].items())[:10]],
      ['trend_key', 'trades']
    )}
  </div>

  <div class="section">
    <h2>Stop / ATR Diagnostics</h2>
    <div class="grid">
      <div class="card"><div class="muted">Selected Symbol</div><div class="metric">{html.escape(str(stop_atr_diagnostics.get('selected_symbol') or 'N/A'))}</div></div>
      <div class="card"><div class="muted">Avg ATR</div><div class="metric">{float(stop_atr_diagnostics.get('avg_atr', 0.0) or 0.0):.2f}</div></div>
      <div class="card"><div class="muted">Max ATR</div><div class="metric">{float(stop_atr_diagnostics.get('max_atr', 0.0) or 0.0):.2f}</div></div>
      <div class="card"><div class="muted">Avg Stop Distance</div><div class="metric">{float(stop_atr_diagnostics.get('avg_stop_distance', 0.0) or 0.0):.2f}</div></div>
      <div class="card"><div class="muted">Max Stop Distance</div><div class="metric">{float(stop_atr_diagnostics.get('max_stop_distance', 0.0) or 0.0):.2f}</div></div>
      <div class="card"><div class="muted">Max Stop Distance %</div><div class="metric">{float(stop_atr_diagnostics.get('max_stop_distance_pct', 0.0) or 0.0) * 100.0:.2f}%</div></div>
      <div class="card"><div class="muted">Abnormal Stops</div><div class="metric danger">{int(stop_atr_diagnostics.get('abnormal_stop_count', 0) or 0)}</div></div>
      <div class="card"><div class="muted">Abnormal Stop %</div><div class="metric danger">{float(stop_atr_diagnostics.get('abnormal_stop_pct', 0.0) or 0.0):.2f}%</div></div>
      <div class="card"><div class="muted">Price Jump Flags</div><div class="metric danger">{int(stop_atr_diagnostics.get('price_jump_flag_count', 0) or 0)}</div></div>
    </div>
    {render_rows(stop_atr_diagnostics['largest_atr_trades'], ['intent_id', 'symbol', 'entry_ts', 'atr', 'stop_distance', 'stop_distance_pct'])}
    {render_rows(stop_atr_diagnostics['flagged_trades'], ['intent_id', 'symbol', 'entry_ts', 'entry_price', 'stop_price', 'stop_distance', 'stop_distance_pct', 'stop_distance_ticks', 'atr'])}
    {render_rows(stop_atr_diagnostics['price_jump_flags'], ['ts_event', 'symbol', 'close', 'prev_close', 'price_jump_pct'])}
  </div>

  <div class="section">
    <h2>Signal Quality</h2>
    <div class="grid">
      <div class="card"><div class="muted">Avg Score Accepted</div><div class="metric">{signal_quality['avg_score_accepted']:.3f}</div></div>
      <div class="card"><div class="muted">Avg Score Rejected</div><div class="metric">{signal_quality['avg_score_rejected']:.3f}</div></div>
      <div class="card"><div class="muted">Accepted Signals</div><div class="metric">{signal_quality['accepted_count']}</div></div>
      <div class="card"><div class="muted">Rejected Signals</div><div class="metric">{signal_quality['rejected_count']}</div></div>
      <div class="card"><div class="muted">Higher-Score Rejected</div><div class="metric">{signal_quality['higher_score_rejected_count']}</div></div>
      <div class="card"><div class="muted">Higher-Score Rejected %</div><div class="metric">{signal_quality['higher_score_rejected_pct']:.2f}%</div></div>
    </div>
    <p class="muted">Accepted vs rejected score distribution under the current position cap and ranking rules.</p>
    {render_rows(signal_quality['score_histogram'], ["bin", "accepted", "rejected"])}
  </div>

  <div class="section">
    <h2>Exposure Control</h2>
    <div class="grid">
      <div class="card"><div class="muted">Avg Concurrent Positions</div><div class="metric">{exposure_control['avg_concurrent_positions']:.3f}</div></div>
      <div class="card"><div class="muted">Max Observed Concurrent</div><div class="metric">{exposure_control['max_observed_concurrent_positions']}</div></div>
      <div class="card"><div class="muted">Before Drawdown</div><div class="metric danger">{exposure_control['drawdown_comparison']['before_exposure_controls']:.2f}</div></div>
      <div class="card"><div class="muted">After Drawdown</div><div class="metric danger">{exposure_control['drawdown_comparison']['after_exposure_controls']:.2f}</div></div>
      <div class="card"><div class="muted">Before Trades</div><div class="metric">{exposure_control['trade_comparison']['before_exposure_controls']}</div></div>
      <div class="card"><div class="muted">After Trades</div><div class="metric">{exposure_control['trade_comparison']['after_exposure_controls']}</div></div>
    </div>
    {render_rows(
      [
        {'rule': rule, 'blocked_count': metrics['count'], 'blocked_percentage': f"{metrics['percentage']:.2f}%"}
        for rule, metrics in exposure_control['blocked_by_rule'].items()
      ],
      ['rule', 'blocked_count', 'blocked_percentage']
    )}
  </div>

  <div class="section">
    <h2>Direction Stacking</h2>
    <div class="grid">
      <div class="card"><div class="muted">Stacked Trades</div><div class="metric">{direction_stacking['same_direction_allowed']}</div></div>
      <div class="card"><div class="muted">Blocked Same-Direction Signals</div><div class="metric">{direction_stacking['same_direction_blocked']}</div></div>
      <div class="card"><div class="muted">Acceptance Rate</div><div class="metric">{direction_stacking['acceptance_rate']:.2%}</div></div>
      <div class="card"><div class="muted">Stacked Trade %</div><div class="metric">{direction_stacking['stacked_trade_percentage']:.2f}%</div></div>
      <div class="card"><div class="muted">Stacked PnL</div><div class="metric">{direction_stacking['stacked_pnl']:.2f}</div></div>
      <div class="card"><div class="muted">Stacked Drawdown</div><div class="metric danger">{direction_stacking['stacked_drawdown']:.2f}</div></div>
      <div class="card"><div class="muted">Non-Stacked Drawdown</div><div class="metric danger">{direction_stacking['non_stacked_drawdown']:.2f}</div></div>
      <div class="card"><div class="muted">Avg Stacked Entry Distance</div><div class="metric">{direction_stacking['avg_distance_between_stacked_entries_minutes']:.2f}m</div></div>
      <div class="card"><div class="muted">Price Activation Rate</div><div class="metric">{direction_stacking['price_activation_rate']:.2%}</div></div>
    </div>
    {render_rows(
      [
        {'stacking_reason': reason, 'count': count}
        for reason, count in direction_stacking['stacking_reason_counts'].items()
      ],
      ['stacking_reason', 'count']
    )}
  </div>

  <div class="section">
    <h2>Stacking Risk Control</h2>
    <div class="grid">
      <div class="card"><div class="muted">Avg Stacked Heat Usage</div><div class="metric">{stacking_risk_control['avg_stacked_heat_usage_pct']:.2f}%</div></div>
      <div class="card"><div class="muted">Max Stacked Heat Usage</div><div class="metric">{stacking_risk_control['max_stacked_heat_usage_pct']:.2f}%</div></div>
      <div class="card"><div class="muted">Stacked PnL</div><div class="metric">{stacking_risk_control['stacked_pnl']:.2f}</div></div>
      <div class="card"><div class="muted">Non-Stacked PnL</div><div class="metric">{stacking_risk_control['non_stacked_pnl']:.2f}</div></div>
      <div class="card"><div class="muted">Worst Stacked Loss Cluster</div><div class="metric danger">{stacking_risk_control['worst_stacked_loss_cluster']:.2f}</div></div>
      <div class="card"><div class="muted">Stacking Disabled Events</div><div class="metric">{stacking_risk_control['stacking_disabled_events']}</div></div>
      <div class="card"><div class="muted">% Time Stacking Disabled</div><div class="metric">{stacking_risk_control['stacking_disabled_bar_percentage']:.2f}%</div></div>
      <div class="card"><div class="muted">Stacked Drawdown</div><div class="metric danger">{stacking_risk_control['stacked_drawdown']:.2f}</div></div>
      <div class="card"><div class="muted">Non-Stacked Drawdown</div><div class="metric danger">{stacking_risk_control['non_stacked_drawdown']:.2f}</div></div>
    </div>
    {render_rows(
      [
        {'blocked_reason': reason, 'count': count}
        for reason, count in stacking_risk_control['blocked_reasons'].items()
      ],
      ['blocked_reason', 'count']
    )}
    {render_rows(
      [
        {'regime': regime, 'stacked_trades': count}
        for regime, count in stacking_risk_control['stacked_regime_distribution'].items()
      ],
      ['regime', 'stacked_trades']
    )}
  </div>

  <div class="section">
    <h2>Stacking Impact</h2>
    <div class="grid">
      <div class="card"><div class="muted">Stacked PnL</div><div class="metric">{stacking_risk_control['stacked_pnl']:.2f}</div></div>
      <div class="card"><div class="muted">Non-Stacked PnL</div><div class="metric">{stacking_risk_control['non_stacked_pnl']:.2f}</div></div>
      <div class="card"><div class="muted">Stacked Drawdown</div><div class="metric danger">{stacking_risk_control['stacked_drawdown']:.2f}</div></div>
      <div class="card"><div class="muted">Non-Stacked Drawdown</div><div class="metric danger">{stacking_risk_control['non_stacked_drawdown']:.2f}</div></div>
    </div>
  </div>

  <div class="section">
    <h2>Risk Shaping</h2>
    <div class="grid">
      <div class="card"><div class="muted">Avg Risk Multiplier</div><div class="metric">{risk_shaping['avg_risk_multiplier']:.3f}</div></div>
      <div class="card"><div class="muted">Avg Position Scale</div><div class="metric">{risk_shaping['avg_position_scale']:.3f}</div></div>
      <div class="card"><div class="muted">Avg Trade Risk Multiplier</div><div class="metric">{risk_shaping['avg_trade_risk_multiplier']:.3f}</div></div>
      <div class="card"><div class="muted">Avg Trade Position Scale</div><div class="metric">{risk_shaping['avg_trade_position_scale']:.3f}</div></div>
      <div class="card"><div class="muted">Max Cluster Risk Usage</div><div class="metric">{risk_shaping['max_cluster_risk_usage_pct']:.2f}%</div></div>
      <div class="card"><div class="muted">Cluster Cap Blocks</div><div class="metric">{risk_shaping['cluster_cap_blocked']}</div></div>
      <div class="card"><div class="muted">Midday Stacking Blocks</div><div class="metric">{risk_shaping['midday_stacking_blocked']}</div></div>
      <div class="card"><div class="muted">Stacked DD</div><div class="metric danger">{risk_shaping['stacked_drawdown']:.2f}</div></div>
      <div class="card"><div class="muted">Non-Stacked DD</div><div class="metric danger">{risk_shaping['non_stacked_drawdown']:.2f}</div></div>
    </div>
    <div class="section">
      <h3>Risk Multiplier Over Time</h3>
      {_render_curve_svg(risk_shaping['risk_multiplier_curve'], 'risk_multiplier', stroke='#8b5cf6', label='Risk multiplier over time')}
    </div>
    <div class="section">
      <h3>Cluster Risk Usage Over Time</h3>
      {_render_curve_svg(risk_shaping['cluster_risk_usage_curve'], 'cluster_risk_usage_pct', stroke='#d97706', label='Cluster risk usage over time')}
    </div>
  </div>

  <div class="section">
    <h2>Concurrency Analysis (Single Pass)</h2>
    <p class="muted">This section reuses the same bar stream and signal stream to simulate multiple position caps in parallel memory, without rerunning the full backtest.</p>
    <div class="grid">
      <div class="card"><div class="muted">Blocked Signals</div><div class="metric">{position_analysis['position_open_blocked_signals']}</div></div>
      <div class="card"><div class="muted">Blocked Percentage</div><div class="metric">{position_analysis['percentage_blocked']:.2f}%</div></div>
      <div class="card"><div class="muted">Positions Tested</div><div class="metric">{len(concurrency_rows)}</div></div>
      <div class="card"><div class="muted">Base Level</div><div class="metric">1</div></div>
    </div>
    {render_rows(concurrency_rows, ["level", "trades", "pnl", "expectancy", "win_rate", "max_drawdown", "blocked_signals", "blocked_percentage", "trade_delta_vs_1", "pnl_delta_vs_1"]) if concurrency_rows else "<p class='empty'>No concurrency analysis available.</p>"}
  </div>

  <div class="section">
    <h2>Concurrency Validation</h2>
    <p class="muted">Overlap, correlation, clustered-loss, and drawdown diagnostics for each concurrency level.</p>
    {render_rows(payload['concurrency_validation_rows'], ["level", "avg_concurrent_positions", "max_concurrent_positions", "avg_overlap_duration_minutes", "overlap_pair_count", "overlap_pnl_correlation", "worst_clustered_loss", "worst_clustered_loss_trade_count", "max_simultaneous_drawdown"])}
  </div>

  <div class="section">
    <h2>Concurrency By Time Of Day</h2>
    {render_rows(payload['concurrency_time_rows'], ["level", "time_bucket", "trades", "pnl", "expectancy", "win_rate"])}
  </div>

  <div class="section">
    <h2>Per-Level Curves</h2>
    {''.join(f"<div class='card'><h2>Positions = {curve['level']}</h2><p class='muted'>Equity curve</p>{curve['equity_svg']}<p class='muted'>Drawdown curve</p>{curve['drawdown_svg']}<p class='muted'>Overlap distribution</p>{render_rows(curve['overlap_distribution'], ['concurrent_positions', 'bars', 'percentage'])}</div>" for curve in concurrency_curves)}
  </div>

  <div class="section">
    <h2>Equity Curve</h2>
    {payload['equity_curve_svg']}
  </div>

  <div class="section">
    <h2>Denial Reasons</h2>
    {render_rows(denial_reason_rows, ["reason", "count"])}
  </div>

  <div class="section">
    <h2>Filled Trade Samples</h2>
    {render_rows(payload['filled_samples'], ["order_id", "intent_id", "symbol", "status", "fill_qty", "fill_price", "side", "regime", "time_bucket", "timestamp"])}
  </div>

  <div class="section">
    <h2>Performance Matrix</h2>
    {render_rows(payload['performance_matrix'], ["regime", "time_bucket", "trade_count", "win_count", "loss_count", "total_pnl", "avg_pnl", "win_rate"])}
  </div>

  <div class="section">
    <h2>Trade Lifecycle Samples</h2>
    {render_rows(payload['trades'], ["intent_id", "symbol", "side", "entry_ts", "entry_price", "exit_ts", "exit_price", "exit_reason", "pnl", "regime", "time_bucket", "is_stacked", "stacking_reason", "mae", "mfe", "path"])}
  </div>

  <div class="section">
    <h2>Denied Intent Samples</h2>
    {render_rows(payload['denied_samples'], ["intent_id", "symbol", "signal_ts", "signal_score", "entry_price", "stop_price", "target_price", "reason", "regime"])}
  </div>

  <div class="section">
    <h2>Intent Signal Samples</h2>
    {render_rows(payload['intent_samples'], ["intent_id", "symbol", "signal_ts", "signal_score", "side", "entry_price", "stop_price", "target_price", "regime"])}
  </div>

  <div class="section">
    <h2>Top 10 Highest Rejected Signals</h2>
    {render_rows(payload['top_rejected_signals'], ["intent_id", "symbol", "signal_ts", "signal_score", "entry_price", "stop_price", "target_price", "regime"])}
  </div>

  <div class="section">
    <h2>Trade Ledger Samples</h2>
    {render_rows(payload['trade_ledger'], ["intent_id", "symbol", "side", "entry_ts", "exit_ts", "entry_price", "exit_price", "pnl", "exit_reason", "regime", "time_bucket", "is_stacked", "stacking_reason", "mae", "mfe"])}
  </div>

  <div class="section">
    <h2>Raw Backtest Summary</h2>
    <pre>{html.escape(json.dumps(payload['raw_result'], indent=2, sort_keys=True))}</pre>
  </div>
</body>
</html>
"""
    output_path.write_text(html_doc, encoding="utf-8")
