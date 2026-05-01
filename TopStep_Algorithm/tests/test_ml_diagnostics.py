from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest.feature_importance import (
    _filter_frame,
    _prepare_features,
    _walk_forward_splits,
    analyse,
)


def _sample_trades() -> pd.DataFrame:
    rows: list[dict] = []
    for i, ts in enumerate(pd.date_range("2025-01-01 09:00", periods=120, freq="7D", tz="UTC")):
        confluence = "OB" if i % 5 == 0 else ("OB+FVG" if i % 3 == 0 else "FVG")
        pnl = 250.0 if confluence != "OB" else -200.0
        rows.append(
            {
                "symbol": "6E" if i % 2 else "6B",
                "session_date": ts.date(),
                "direction": "BUY" if i % 2 else "SELL",
                "entry_price": 1.1,
                "entry_time": ts,
                "stop_price": 1.099,
                "tp1_price": 1.101,
                "tp2_price": 1.102,
                "tp1_filled": pnl > 0,
                "tp2_filled": pnl > 300,
                "sl_filled": pnl < 0,
                "exit_price": 1.101,
                "exit_time": ts + pd.Timedelta(minutes=10),
                "contracts": 1,
                "pnl_usd": pnl,
                "r_multiple": pnl / 250.0,
                "confluence_type": confluence,
                "atr14": 0.001 + (i % 10) * 0.0001,
                "ema_slope": -0.0001 + (i % 7) * 0.00003,
                "setup_type": "asian_range_sweep",
                "adx_at_entry": 20 + i % 20,
                "vwap_distance_at_entry": 0.0001,
                "range_width_atr": 2.0,
                "sweep_depth_atr": 0.5,
                "distance_to_asian_mid": 0.2,
                "time_since_sweep": 0,
                "risk_points": 0.001,
                "risk_usd": 125.0,
                "mfe_points": 0.002,
                "mae_points": 0.0005,
                "mfe_r": 2.0,
                "mae_r": 0.5,
                "holding_minutes": 10,
                "exit_reason": "take_profit_2" if pnl > 0 else "stop_loss",
                "bars_held": 2,
            }
        )
    return pd.DataFrame(rows)


def test_walk_forward_splits_are_chronological_with_embargo() -> None:
    df = _prepare_features(_sample_trades())
    folds = _walk_forward_splits(df, train_months=6, test_months=2, embargo_days=2)

    assert folds
    for fold in folds:
        train_max = df.iloc[fold.train_idx]["entry_time"].max()
        test_min = df.iloc[fold.test_idx]["entry_time"].min()
        assert train_max < test_min
        assert test_min >= fold.train_end + pd.Timedelta(days=2)


def test_diagnostic_filters_exclude_ob_only_and_select_session() -> None:
    df = _prepare_features(_sample_trades())

    no_ob = _filter_frame(df, exclude_ob_only=True)
    open_only = _filter_frame(df, session_filter={"open"})

    assert "OB" not in set(no_ob["confluence_type"])
    assert set(open_only["session_bucket"]) == {"open"}


def test_analyse_writes_required_outputs(tmp_path: Path) -> None:
    trades = tmp_path / "trades.csv"
    _sample_trades().to_csv(trades, index=False)
    raw = tmp_path / "raw_setup_ledger.csv"
    pd.DataFrame(
        [
            {
                "event_id": f"evt-{i}",
                "timestamp": ts,
                "symbol": "6E",
                "timeframe": "5m",
                "session": ts.date(),
                "direction_candidate": "BUY",
                "setup_stage": "final_candidate" if i % 2 else "ema_trend",
                "setup_type": "asian_range_sweep",
                "confluence_type": "FVG" if i % 2 else "",
                "close": 1.1,
                "volume": 100,
                "range_width_atr": 2.0,
                "sweep_depth_atr": 0.5,
                "atr_at_decision": 0.001,
                "adx_at_decision": 25.0,
                "ema_slope": 0.001,
                "vwap_distance": 0.0001,
                "distance_to_asian_mid": 0.2,
                "passed": bool(i % 2),
                "rejected": not bool(i % 2),
                "rejection_stage": "" if i % 2 else "ema_trend",
                "rejection_reason": "" if i % 2 else "ema_trend_filter_failed",
                "rejection_category": "" if i % 2 else "market_structure_rejection",
                "accepted_into_final_candidate": bool(i % 2),
                "accepted_into_trade": bool(i % 4 == 1),
                "pnl_usd": 100.0 if i % 4 == 1 else "",
                "r_multiple": 1.0 if i % 4 == 1 else "",
                "exit_reason": "take_profit_2" if i % 4 == 1 else "",
                "mfe_r": 2.0 if i % 4 == 1 else "",
                "mae_r": 0.5 if i % 4 == 1 else "",
                "holding_minutes": 10 if i % 4 == 1 else "",
            }
            for i, ts in enumerate(pd.date_range("2025-01-01 09:00", periods=20, freq="7D", tz="UTC"))
        ]
    ).to_csv(raw, index=False)

    analyse(
        trades,
        tmp_path / "diag",
        train_months=6,
        test_months=2,
        embargo_days=1,
        raw_setups_csv=raw,
    )

    for filename in (
        "classification_metrics.csv",
        "probability_buckets.csv",
        "oof_classification_predictions.csv",
        "calibration.csv",
        "ablation_summary.csv",
        "confluence_performance.csv",
        "regime_performance.csv",
        "regime_dashboard.csv",
        "feature_drift_by_fold.csv",
        "fold_by_fold_summary.csv",
        "no_calendar_validation.csv",
        "feature_set_comparison.csv",
        "raw_setup_summary.csv",
        "setup_stage_funnel.csv",
        "rejection_reason_summary.csv",
        "accepted_vs_rejected_feature_drift.csv",
        "raw_to_trade_conversion.csv",
        "ml_validation_report.md",
    ):
        assert (tmp_path / "diag" / filename).exists()
    assert (tmp_path / "raw_setup_validation_report.md").exists()
