"""Time-ordered diagnostics for London sweep backtest trades.

This script intentionally does not use shuffled K-fold CV.  It treats the
trade ledger as a chronological sequence and evaluates models on expanding
walk-forward folds with preprocessing fit inside each fold.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

FEATURES = [
    "symbol",
    "direction",
    "confluence_type",
    "entry_hour",
    "weekday",
    "month",
    "contracts",
    "atr14",
    "atr_ticks",
    "ema_slope",
    "ema_slope_abs",
    "risk_ticks",
]
CALENDAR_FEATURES = {"weekday", "month", "entry_hour"}
SESSION_FEATURES = {"session_bucket"}
CATEGORICAL_FEATURES = ["symbol", "direction", "confluence_type", "weekday", "session_bucket"]
NUMERIC_FEATURES = [f for f in [*FEATURES, "session_bucket"] if f not in CATEGORICAL_FEATURES]
CLASSIFICATION_TARGETS = [
    "positive_pnl",
    "positive_r",
    "avoid_full_stop",
    "top_quartile_r_outcome",
    "tp1_or_better",
]


@dataclass(frozen=True)
class FoldSpec:
    fold_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_idx: np.ndarray
    test_idx: np.ndarray


def _prepare_features(trades: pd.DataFrame) -> pd.DataFrame:
    df = trades.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    df = df.sort_values(["entry_time", "exit_time", "symbol"]).reset_index(drop=True)
    df["entry_hour"] = df["entry_time"].dt.hour
    df["entry_minute"] = df["entry_time"].dt.minute
    entry_decimal = df["entry_hour"] + df["entry_minute"] / 60.0
    df["session_bucket"] = pd.cut(
        entry_decimal,
        bins=[8.49, 10.0, 11.5, 13.5],
        labels=["open", "mid", "late"],
        include_lowest=True,
    ).astype(str)
    df["weekday"] = df["entry_time"].dt.day_name()
    df["month"] = df["entry_time"].dt.month
    df["positive_pnl"] = (df["pnl_usd"] > 0).astype(int)
    df["positive_r"] = (df["r_multiple"] > 0).astype(int)
    sl_filled = df["sl_filled"].astype(bool) if "sl_filled" in df else pd.Series(False, index=df.index)
    tp1_filled = df["tp1_filled"].astype(bool) if "tp1_filled" in df else pd.Series(False, index=df.index)
    df["avoid_full_stop"] = (~(sl_filled & ~tp1_filled)).astype(int)
    df["tp1_or_better"] = tp1_filled.astype(int)
    top_quartile_threshold = df["r_multiple"].quantile(0.75)
    df["top_quartile_r_outcome"] = (df["r_multiple"] >= top_quartile_threshold).astype(int)
    df["ema_slope_abs"] = df["ema_slope"].abs()
    df["atr_ticks"] = np.where(
        df["symbol"].eq("6E"),
        df["atr14"] / 0.00005,
        df["atr14"] / 0.0001,
    )
    df["risk_ticks"] = np.where(
        df["symbol"].eq("6E"),
        (df["entry_price"] - df["stop_price"]).abs() / 0.00005,
        (df["entry_price"] - df["stop_price"]).abs() / 0.0001,
    )
    return df


def _walk_forward_splits(
    df: pd.DataFrame,
    *,
    train_months: int,
    test_months: int,
    embargo_days: int,
) -> list[FoldSpec]:
    first = df["entry_time"].min().normalize()
    last = df["entry_time"].max().normalize()
    train_start = first
    folds: list[FoldSpec] = []
    fold_idx = 0

    while True:
        train_end = train_start + relativedelta(months=train_months)
        test_start = train_end + pd.Timedelta(days=embargo_days)
        test_end = test_start + relativedelta(months=test_months)
        if test_start > last:
            break

        train_mask = (df["entry_time"] >= train_start) & (df["entry_time"] < train_end)
        test_mask = (df["entry_time"] >= test_start) & (df["entry_time"] < test_end)
        train_idx = np.flatnonzero(train_mask.to_numpy())
        test_idx = np.flatnonzero(test_mask.to_numpy())
        if len(train_idx) >= 20 and len(test_idx) >= 3:
            folds.append(
                FoldSpec(
                    fold_idx=fold_idx,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    train_idx=train_idx,
                    test_idx=test_idx,
                )
            )
            fold_idx += 1
        train_start += relativedelta(months=test_months)

    return folds


def _feature_parts(features: list[str]) -> tuple[list[str], list[str]]:
    categorical = [f for f in features if f in CATEGORICAL_FEATURES]
    numeric = [f for f in features if f not in categorical]
    return categorical, numeric


def _regression_pipeline(features: list[str]) -> Pipeline:
    categorical, numeric = _feature_parts(features)
    return Pipeline(
        [
            (
                "pre",
                ColumnTransformer(
                    [
                        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
                        ("num", StandardScaler(), numeric),
                    ]
                ),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    min_samples_leaf=5,
                    random_state=7,
                ),
            ),
        ]
    )


def _classification_pipeline(features: list[str]) -> Pipeline:
    categorical, numeric = _feature_parts(features)
    return Pipeline(
        [
            (
                "pre",
                ColumnTransformer(
                    [
                        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
                        ("num", StandardScaler(), numeric),
                    ]
                ),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    random_state=7,
                ),
            ),
        ]
    )


def _safe_auc(y_true: pd.Series, y_score: np.ndarray) -> float:
    if y_true.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_corr(a: pd.Series, b: pd.Series, method: str) -> float:
    if len(a) < 3 or a.nunique() < 2 or b.nunique() < 2:
        return float("nan")
    return float(a.corr(b, method=method))


def _fold_metrics(
    df: pd.DataFrame,
    folds: list[FoldSpec],
    features: list[str] = FEATURES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    predictions: list[pd.DataFrame] = []
    perm_rows: list[pd.DataFrame] = []

    for fold in folds:
        train = df.iloc[fold.train_idx]
        test = df.iloc[fold.test_idx]
        x_train = train[features]
        y_train = train["pnl_usd"]
        x_test = test[features]
        y_test = test["pnl_usd"]

        model = _regression_pipeline(features)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        pred_series = pd.Series(y_pred, index=test.index)

        zero_pred = np.zeros(len(y_test))
        mean_pred = np.full(len(y_test), float(y_train.mean()))
        sign_pred = (y_pred > 0).astype(int)
        y_cls = test["positive_pnl"]

        rows.append(
            {
                "fold_idx": fold.fold_idx,
                "train_start": fold.train_start.date(),
                "train_end": fold.train_end.date(),
                "test_start": fold.test_start.date(),
                "test_end": fold.test_end.date(),
                "train_trades": len(train),
                "test_trades": len(test),
                "train_mean_pnl": y_train.mean(),
                "test_mean_pnl": y_test.mean(),
                "model_r2": r2_score(y_test, y_pred),
                "zero_r2": r2_score(y_test, zero_pred),
                "train_mean_r2": r2_score(y_test, mean_pred),
                "model_mae": mean_absolute_error(y_test, y_pred),
                "zero_mae": mean_absolute_error(y_test, zero_pred),
                "directional_accuracy": accuracy_score(y_cls, sign_pred),
                "positive_precision": precision_score(y_cls, sign_pred, zero_division=0),
                "prediction_ic_pearson": _safe_corr(y_test, pred_series, "pearson"),
                "prediction_ic_spearman": _safe_corr(y_test, pred_series, "spearman"),
                "auc_positive_pnl": _safe_auc(y_cls, y_pred),
                "test_pnl": y_test.sum(),
                "test_win_rate": y_cls.mean(),
            }
        )

        pred_frame = test[
            [
                "symbol",
                "direction",
                "confluence_type",
                "entry_time",
                "exit_time",
                "pnl_usd",
                "r_multiple",
            ]
        ].copy()
        pred_frame.insert(0, "fold_idx", fold.fold_idx)
        pred_frame["predicted_pnl"] = y_pred
        pred_frame["predicted_positive"] = sign_pred
        predictions.append(pred_frame)

        if len(test) >= 5:
            permutation = permutation_importance(
                model,
                x_test,
                y_test,
                n_repeats=30,
                random_state=7,
                scoring="r2",
            )
            fold_perm = pd.DataFrame(
                {
                    "fold_idx": fold.fold_idx,
                    "feature": features,
                    "perm_r2_drop_mean": permutation.importances_mean,
                    "perm_r2_drop_std": permutation.importances_std,
                }
            )
            perm_rows.append(fold_perm)

    metrics = pd.DataFrame(rows)
    preds = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    permutation = pd.concat(perm_rows, ignore_index=True) if perm_rows else pd.DataFrame()
    return metrics, preds, permutation


def _aggregate_permutation(permutation: pd.DataFrame) -> pd.DataFrame:
    if permutation.empty:
        return pd.DataFrame(columns=["feature", "mean_test_r2_drop", "std_test_r2_drop", "folds"])
    return (
        permutation.groupby("feature")
        .agg(
            mean_test_r2_drop=("perm_r2_drop_mean", "mean"),
            std_test_r2_drop=("perm_r2_drop_mean", "std"),
            folds=("fold_idx", "nunique"),
        )
        .reset_index()
        .sort_values("mean_test_r2_drop", ascending=False)
    )


def _group_pnl(df: pd.DataFrame) -> pd.DataFrame:
    groups: list[pd.DataFrame] = []
    for col in ("symbol", "direction", "confluence_type", "entry_hour", "weekday", "month"):
        grouped = (
            df.groupby(col, dropna=False)
            .agg(
                trades=("pnl_usd", "size"),
                pnl=("pnl_usd", "sum"),
                avg_pnl=("pnl_usd", "mean"),
                win_rate=("pnl_usd", lambda s: (s > 0).mean()),
                avg_r=("r_multiple", "mean"),
                pnl_std=("pnl_usd", "std"),
            )
            .reset_index()
        )
        grouped.insert(0, "feature", col)
        groups.append(grouped.rename(columns={col: "value"}))
    return pd.concat(groups, ignore_index=True)


def _target_summary(df: pd.DataFrame) -> pd.DataFrame:
    q = df["pnl_usd"].quantile([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    return pd.DataFrame(
        [
            {
                "trades": len(df),
                "mean_pnl": df["pnl_usd"].mean(),
                "std_pnl": df["pnl_usd"].std(),
                "signal_to_noise": df["pnl_usd"].mean() / df["pnl_usd"].std()
                if df["pnl_usd"].std() > 0
                else 0.0,
                "win_rate": df["positive_pnl"].mean(),
                "mean_r": df["r_multiple"].mean(),
                "std_r": df["r_multiple"].std(),
                "p00": q.loc[0.0],
                "p10": q.loc[0.1],
                "p25": q.loc[0.25],
                "p50": q.loc[0.5],
                "p75": q.loc[0.75],
                "p90": q.loc[0.9],
                "p100": q.loc[1.0],
            }
        ]
    )


def _distribution_shift(
    df: pd.DataFrame,
    folds: list[FoldSpec],
    features: list[str] = FEATURES,
) -> pd.DataFrame:
    rows: list[dict] = []
    for fold in folds:
        train = df.iloc[fold.train_idx]
        test = df.iloc[fold.test_idx]
        categorical, numeric = _feature_parts(features)
        for col in numeric:
            train_std = train[col].std()
            rows.append(
                {
                    "fold_idx": fold.fold_idx,
                    "feature": col,
                    "type": "numeric",
                    "train_mean": train[col].mean(),
                    "test_mean": test[col].mean(),
                    "train_std": train_std,
                    "std_mean_diff": (test[col].mean() - train[col].mean()) / train_std
                    if train_std and train_std > 0
                    else 0.0,
                }
            )
        for col in categorical:
            train_freq = train[col].value_counts(normalize=True)
            test_freq = test[col].value_counts(normalize=True)
            values = sorted(set(train_freq.index) | set(test_freq.index))
            total_abs_drift = sum(abs(test_freq.get(v, 0.0) - train_freq.get(v, 0.0)) for v in values)
            rows.append(
                {
                    "fold_idx": fold.fold_idx,
                    "feature": col,
                    "type": "categorical",
                    "train_mean": np.nan,
                    "test_mean": np.nan,
                    "train_std": np.nan,
                    "std_mean_diff": total_abs_drift,
                }
            )
    return pd.DataFrame(rows)


def _profit_factor(pnl: pd.Series) -> float:
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = pnl[pnl <= 0].sum()
    return float(gross_profit / abs(gross_loss)) if gross_loss != 0 else float("inf")


def _max_drawdown(pnl: pd.Series) -> float:
    equity = pnl.cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    return float(drawdown.min()) if len(drawdown) else 0.0


def _performance_summary(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {
            "label": label,
            "trades": 0,
            "pnl": 0.0,
            "win_rate": 0.0,
            "avg_pnl": 0.0,
            "avg_r": 0.0,
            "median_r": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "avg_mfe_r": 0.0,
            "avg_mae_r": 0.0,
            "stop_out_rate": 0.0,
            "tp_hit_rate": 0.0,
        }
    return {
        "label": label,
        "trades": len(df),
        "pnl": df["pnl_usd"].sum(),
        "win_rate": (df["pnl_usd"] > 0).mean(),
        "avg_pnl": df["pnl_usd"].mean(),
        "avg_r": df["r_multiple"].mean(),
        "median_r": df["r_multiple"].median(),
        "profit_factor": _profit_factor(df["pnl_usd"]),
        "max_drawdown": _max_drawdown(df["pnl_usd"]),
        "avg_mfe_r": df["mfe_r"].mean() if "mfe_r" in df else 0.0,
        "avg_mae_r": df["mae_r"].mean() if "mae_r" in df else 0.0,
        "stop_out_rate": df["sl_filled"].mean() if "sl_filled" in df else 0.0,
        "tp_hit_rate": df["tp1_filled"].mean() if "tp1_filled" in df else 0.0,
    }


def _filter_frame(
    df: pd.DataFrame,
    *,
    exclude_ob_only: bool = False,
    include_confluence_types: set[str] | None = None,
    exclude_confluence_types: set[str] | None = None,
    min_atr_percentile: float | None = None,
    max_atr_percentile: float | None = None,
    session_filter: set[str] | None = None,
    symbol_filter: set[str] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    if exclude_ob_only:
        out = out[out["confluence_type"] != "OB"]
    if include_confluence_types:
        out = out[out["confluence_type"].isin(include_confluence_types)]
    if exclude_confluence_types:
        out = out[~out["confluence_type"].isin(exclude_confluence_types)]
    if min_atr_percentile is not None:
        out = out[out["atr14"] >= df["atr14"].quantile(min_atr_percentile)]
    if max_atr_percentile is not None:
        out = out[out["atr14"] <= df["atr14"].quantile(max_atr_percentile)]
    if session_filter:
        out = out[out["session_bucket"].isin(session_filter)]
    if symbol_filter:
        out = out[out["symbol"].isin(symbol_filter)]
    return out.reset_index(drop=True)


def _ablation_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        _performance_summary(df, "all_trades"),
        _performance_summary(_filter_frame(df, exclude_ob_only=True), "no_ob_only"),
        _performance_summary(_filter_frame(df, include_confluence_types={"FVG"}), "fvg_only"),
        _performance_summary(_filter_frame(df, include_confluence_types={"OB+FVG"}), "ob_fvg_only"),
        _performance_summary(_filter_frame(df, min_atr_percentile=0.67), "high_atr_regime"),
        _performance_summary(_filter_frame(df, max_atr_percentile=0.33), "low_atr_regime"),
    ]
    for bucket in ("open", "mid", "late"):
        rows.append(_performance_summary(_filter_frame(df, session_filter={bucket}), f"session_{bucket}"))
    return pd.DataFrame(rows)


def _regime_dashboard(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["atr_percentile_bucket"] = pd.qcut(out["atr14"].rank(method="first"), 3, labels=["low", "mid", "high"])
    adx_source = out["adx_at_entry"] if "adx_at_entry" in out else pd.Series(0.0, index=out.index)
    out["adx_bucket"] = pd.cut(adx_source, bins=[-np.inf, 20, 30, np.inf], labels=["low", "mid", "high"])
    out["ema_slope_bucket"] = pd.cut(out["ema_slope"], bins=[-np.inf, -0.00003, 0.00003, np.inf], labels=["down", "flat", "up"])
    rows: list[dict] = []
    for feature in (
        "atr_percentile_bucket",
        "adx_bucket",
        "ema_slope_bucket",
        "session_bucket",
        "weekday",
        "month",
        "symbol",
        "direction",
        "confluence_type",
        "setup_type",
    ):
        if feature not in out:
            continue
        for value, group in out.groupby(feature, dropna=False, observed=False):
            row = _performance_summary(group, f"{feature}={value}")
            row["feature"] = feature
            row["value"] = value
            rows.append(row)
    return pd.DataFrame(rows)


def _target_for_fold(train: pd.DataFrame, test: pd.DataFrame, target: str) -> tuple[pd.Series, pd.Series]:
    if target == "top_quartile_r_outcome":
        threshold = train["r_multiple"].quantile(0.75)
        return (train["r_multiple"] >= threshold).astype(int), (test["r_multiple"] >= threshold).astype(int)
    return train[target].astype(int), test[target].astype(int)


def _classification_diagnostics(
    df: pd.DataFrame,
    folds: list[FoldSpec],
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict] = []
    pred_rows: list[pd.DataFrame] = []
    bucket_rows: list[dict] = []
    calibration_rows: list[dict] = []

    for target in CLASSIFICATION_TARGETS:
        for fold in folds:
            train = df.iloc[fold.train_idx]
            test = df.iloc[fold.test_idx]
            y_train, y_test = _target_for_fold(train, test, target)
            if y_train.nunique() < 2 or y_test.empty:
                continue
            model = _classification_pipeline(features)
            model.fit(train[features], y_train)
            prob = model.predict_proba(test[features])[:, 1]
            pred = (prob >= 0.5).astype(int)
            auc = _safe_auc(y_test, prob)
            pr_auc = float(average_precision_score(y_test, prob)) if y_test.nunique() > 1 else float("nan")
            metric_rows.append(
                {
                    "target": target,
                    "fold_idx": fold.fold_idx,
                    "train_start": fold.train_start.date(),
                    "test_start": fold.test_start.date(),
                    "train_base_rate": y_train.mean(),
                    "test_base_rate": y_test.mean(),
                    "precision": precision_score(y_test, pred, zero_division=0),
                    "recall": recall_score(y_test, pred, zero_division=0),
                    "f1": f1_score(y_test, pred, zero_division=0),
                    "roc_auc": auc,
                    "pr_auc": pr_auc,
                    "brier": brier_score_loss(y_test, prob),
                }
            )
            pred_frame = test[["entry_time", "symbol", "direction", "confluence_type", "pnl_usd", "r_multiple"]].copy()
            pred_frame.insert(0, "target", target)
            pred_frame.insert(1, "fold_idx", fold.fold_idx)
            pred_frame["actual"] = y_test.to_numpy()
            pred_frame["probability"] = prob
            pred_frame["prediction"] = pred
            pred_rows.append(pred_frame)

            buckets = pd.qcut(pd.Series(prob).rank(method="first"), 5, labels=False, duplicates="drop")
            for bucket in sorted(pd.unique(buckets)):
                mask = buckets == bucket
                bucket_test = test.iloc[np.flatnonzero(mask.to_numpy())]
                actual = y_test.iloc[np.flatnonzero(mask.to_numpy())]
                bucket_prob = prob[mask.to_numpy()]
                bucket_rows.append(
                    {
                        "target": target,
                        "fold_idx": fold.fold_idx,
                        "bucket": int(bucket),
                        "count": len(bucket_test),
                        "avg_probability": float(np.mean(bucket_prob)),
                        "base_rate": float(actual.mean()) if len(actual) else 0.0,
                        "win_rate": float((bucket_test["pnl_usd"] > 0).mean()) if len(bucket_test) else 0.0,
                        "avg_r": bucket_test["r_multiple"].mean() if len(bucket_test) else 0.0,
                        "avg_pnl_usd": bucket_test["pnl_usd"].mean() if len(bucket_test) else 0.0,
                        "expected_value": bucket_test["pnl_usd"].mean() if len(bucket_test) else 0.0,
                    }
                )
                calibration_rows.append(
                    {
                        "target": target,
                        "fold_idx": fold.fold_idx,
                        "bucket": int(bucket),
                        "count": len(actual),
                        "mean_predicted_probability": float(np.mean(bucket_prob)),
                        "observed_rate": float(actual.mean()) if len(actual) else 0.0,
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    preds = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    buckets = pd.DataFrame(bucket_rows)
    calibration = pd.DataFrame(calibration_rows)
    return metrics, buckets, preds, calibration


def _feature_set_comparison(df: pd.DataFrame, folds: list[FoldSpec]) -> pd.DataFrame:
    feature_sets = {
        "all_features": FEATURES,
        "no_calendar": [f for f in FEATURES if f not in CALENDAR_FEATURES],
        "calendar_only": ["weekday", "month", "entry_hour"],
        "no_calendar_session_bucket": [
            *[f for f in FEATURES if f not in CALENDAR_FEATURES],
            "session_bucket",
        ],
    }
    rows: list[dict] = []
    for name, features in feature_sets.items():
        metrics, _, _ = _fold_metrics(df, folds, features)
        rows.append(
            {
                "feature_set": name,
                "folds": len(metrics),
                "mean_model_r2": metrics["model_r2"].mean(),
                "mean_precision": metrics["positive_precision"].mean(),
                "mean_directional_accuracy": metrics["directional_accuracy"].mean(),
                "mean_spearman_ic": metrics["prediction_ic_spearman"].mean(),
            }
        )
    return pd.DataFrame(rows)


def _empty_raw_outputs() -> dict[str, pd.DataFrame]:
    return {
        "raw_setup_summary": pd.DataFrame(),
        "setup_stage_funnel": pd.DataFrame(),
        "rejection_reason_summary": pd.DataFrame(),
        "accepted_vs_rejected_feature_drift": pd.DataFrame(),
        "raw_to_trade_conversion": pd.DataFrame(),
    }


def _prepare_raw_setups(raw_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_csv)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for col in ["passed", "rejected", "accepted_into_final_candidate", "accepted_into_trade"]:
        if col in df:
            df[col] = df[col].astype(str).str.lower().isin({"true", "1", "yes"})
    for col in [
        "rejection_stage",
        "rejection_reason",
        "rejection_category",
        "direction_candidate",
        "confluence_type",
        "setup_type",
    ]:
        if col in df:
            df[col] = df[col].fillna("")
    return df.sort_values(["timestamp", "symbol", "event_id"]).reset_index(drop=True)


def _raw_setup_diagnostics(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if raw.empty:
        return _empty_raw_outputs()

    total = len(raw)
    summary = pd.DataFrame(
        [
            {
                "raw_setups": total,
                "final_candidates": int(raw["accepted_into_final_candidate"].sum()),
                "accepted_trades": int(raw["accepted_into_trade"].sum()),
                "rejected_setups": int(raw["rejected"].sum()),
                "raw_to_final_candidate_rate": float(raw["accepted_into_final_candidate"].mean()),
                "raw_to_trade_rate": float(raw["accepted_into_trade"].mean()),
            }
        ]
    )

    stage = raw.groupby("setup_stage", dropna=False).agg(
        count=("event_id", "count"),
        passed_count=("passed", "sum"),
        rejected_count=("rejected", "sum"),
        final_candidate_count=("accepted_into_final_candidate", "sum"),
        accepted_trade_count=("accepted_into_trade", "sum"),
    ).reset_index()
    stage["percent_of_raw_setups"] = stage["count"] / total
    stage["accepted_trade_rate"] = stage["accepted_trade_count"] / stage["count"]
    stage = stage.sort_values(["count", "setup_stage"], ascending=[False, True])

    reason = raw.groupby(
        ["rejection_stage", "rejection_reason", "rejection_category"],
        dropna=False,
    ).agg(
        count=("event_id", "count"),
        accepted_trade_count=("accepted_into_trade", "sum"),
        avg_pnl_usd=("pnl_usd", "mean"),
        avg_r_multiple=("r_multiple", "mean"),
    ).reset_index()
    passed_mask = reason["rejection_reason"].eq("")
    reason.loc[passed_mask, "rejection_stage"] = "passed"
    reason.loc[passed_mask, "rejection_reason"] = "passed"
    reason.loc[passed_mask, "rejection_category"] = "passed"
    reason["percent_of_raw_setups"] = reason["count"] / total
    reason = reason.sort_values(["count", "rejection_reason"], ascending=[False, True])

    numeric_cols = [
        "close",
        "volume",
        "range_width_atr",
        "sweep_depth_atr",
        "atr_at_decision",
        "adx_at_decision",
        "ema_slope",
        "vwap_distance",
        "distance_to_asian_mid",
    ]
    drift_rows: list[dict] = []
    accepted = raw[raw["accepted_into_trade"]]
    rejected = raw[~raw["accepted_into_trade"]]
    for col in numeric_cols:
        if col not in raw:
            continue
        accepted_mean = pd.to_numeric(accepted[col], errors="coerce").mean()
        rejected_mean = pd.to_numeric(rejected[col], errors="coerce").mean()
        pooled_std = pd.to_numeric(raw[col], errors="coerce").std()
        drift_rows.append(
            {
                "feature": col,
                "accepted_mean": accepted_mean,
                "rejected_mean": rejected_mean,
                "difference": accepted_mean - rejected_mean,
                "std_mean_diff": (
                    (accepted_mean - rejected_mean) / pooled_std
                    if pooled_std and not np.isnan(pooled_std)
                    else np.nan
                ),
            }
        )
    drift = pd.DataFrame(drift_rows).sort_values(
        "std_mean_diff", key=lambda s: s.abs(), ascending=False
    )

    conversion = raw.groupby(
        ["symbol", "direction_candidate", "confluence_type", "setup_type"],
        dropna=False,
    ).agg(
        raw_setups=("event_id", "count"),
        final_candidates=("accepted_into_final_candidate", "sum"),
        accepted_trades=("accepted_into_trade", "sum"),
        avg_pnl_usd=("pnl_usd", "mean"),
        avg_r_multiple=("r_multiple", "mean"),
    ).reset_index()
    conversion["raw_to_final_candidate_rate"] = conversion["final_candidates"] / conversion["raw_setups"]
    conversion["raw_to_trade_rate"] = conversion["accepted_trades"] / conversion["raw_setups"]
    conversion = conversion.sort_values(["accepted_trades", "raw_setups"], ascending=False)

    return {
        "raw_setup_summary": summary,
        "setup_stage_funnel": stage,
        "rejection_reason_summary": reason,
        "accepted_vs_rejected_feature_drift": drift,
        "raw_to_trade_conversion": conversion,
    }


def _report_markdown(outputs: dict[str, pd.DataFrame]) -> str:
    fold_metrics = outputs["fold_metrics"]
    classification = outputs["classification_metrics"]
    buckets = outputs["probability_buckets"]
    ablations = outputs["ablation_summary"]
    feature_sets = outputs["feature_set_comparison"]

    best_target = "n/a"
    best_precision = float("nan")
    if not classification.empty:
        target_summary = classification.groupby("target")["precision"].mean().sort_values(ascending=False)
        best_target = str(target_summary.index[0])
        best_precision = float(target_summary.iloc[0])

    all_pf = float(ablations.loc[ablations["label"].eq("all_trades"), "profit_factor"].iloc[0])
    no_ob_pf = float(ablations.loc[ablations["label"].eq("no_ob_only"), "profit_factor"].iloc[0])
    no_calendar_r2 = float(feature_sets.loc[feature_sets["feature_set"].eq("no_calendar"), "mean_model_r2"].iloc[0])
    top_bucket_ev = float("nan")
    monotonic = False
    if not buckets.empty and best_target != "n/a":
        grouped = buckets[buckets["target"].eq(best_target)].groupby("bucket")["expected_value"].mean()
        if not grouped.empty:
            top_bucket_ev = float(grouped.iloc[-1])
            monotonic = bool(grouped.is_monotonic_increasing)

    status = "experimental_only"
    if best_precision > 0.50 and top_bucket_ev > 0 and monotonic and no_ob_pf >= all_pf and no_calendar_r2 > -0.05:
        status = "candidate_for_further_paper_validation"

    risks = [
        "- Calendar and regime drift remain material.",
        "- OB-only drag should be removed or explicitly gated before further ML work.",
        "- More data, preferably 6-12 additional months, is needed before live ML gating.",
    ]
    raw_summary = outputs.get("raw_setup_summary", pd.DataFrame())
    if raw_summary is not None and not raw_summary.empty:
        raw_row = raw_summary.iloc[0]
        raw_lines = [
            "",
            "## Raw Opportunity Set",
            f"- Raw setups captured: `{int(raw_row['raw_setups'])}`.",
            f"- Final candidates: `{int(raw_row['final_candidates'])}`.",
            f"- Accepted trades: `{int(raw_row['accepted_trades'])}`.",
            f"- Raw-to-trade conversion: `{raw_row['raw_to_trade_rate']:.3f}`.",
        ]
    else:
        raw_lines = [
            "",
            "## Raw Opportunity Set",
            "- Raw setup ledger was not supplied to this diagnostic run.",
        ]
        risks.insert(0, "- Current diagnostics did not include the raw opportunity set.")

    return "\n".join(
        [
            "# ML Validation Report",
            "",
            "## Executive Summary",
            f"- ML layer status: **{status}**.",
            f"- Mean chronological regression R2: `{fold_metrics['model_r2'].mean():.4f}`.",
            f"- Best classification target by precision: `{best_target}` (`{best_precision:.3f}`).",
            f"- No-calendar mean R2: `{no_calendar_r2:.4f}`.",
            f"- All-trades profit factor: `{all_pf:.2f}`.",
            f"- No-OB-only profit factor: `{no_ob_pf:.2f}`.",
            f"- Top probability bucket EV for best target: `{top_bucket_ev:.2f}`.",
            f"- Probability buckets monotonic: `{monotonic}`.",
            "",
            "## Decision",
            "The ML layer remains experimental. It should not gate live trading until "
            "chronological OOF precision, probability-bucket EV, and no-calendar validation "
            "are stable over a larger sample.",
            *raw_lines,
            "",
            "## Remaining Risks",
            *risks,
        ]
    )


def _raw_validation_report(outputs: dict[str, pd.DataFrame]) -> str:
    raw_summary = outputs.get("raw_setup_summary", pd.DataFrame())
    funnel = outputs.get("setup_stage_funnel", pd.DataFrame())
    reasons = outputs.get("rejection_reason_summary", pd.DataFrame())
    conversion = outputs.get("raw_to_trade_conversion", pd.DataFrame())
    ablations = outputs.get("ablation_summary", pd.DataFrame())

    if raw_summary.empty:
        return "# Raw Setup Validation Report\n\nRaw setup ledger was not supplied.\n"

    row = raw_summary.iloc[0]
    no_ob_pf = float(ablations.loc[ablations["label"].eq("no_ob_only"), "profit_factor"].iloc[0])
    all_pf = float(ablations.loc[ablations["label"].eq("all_trades"), "profit_factor"].iloc[0])
    top_reasons = reasons.head(8)
    reason_lines = [
        f"- `{r.rejection_reason or 'passed'}` at `{r.rejection_stage or 'passed'}`: {int(r['count'])}"
        for _, r in top_reasons.iterrows()
    ]
    funnel_lines = [
        f"- `{r.setup_stage}`: {int(r['count'])} rows, {int(r.accepted_trade_count)} accepted trades"
        for _, r in funnel.head(10).iterrows()
    ]
    ob_rows = conversion[conversion["confluence_type"].eq("OB")]
    fvg_rows = conversion[conversion["confluence_type"].isin(["FVG", "OB+FVG"])]
    ob_trade_count = int(ob_rows["accepted_trades"].sum()) if not ob_rows.empty else 0
    fvg_trade_count = int(fvg_rows["accepted_trades"].sum()) if not fvg_rows.empty else 0

    return "\n".join(
        [
            "# Raw Setup Validation Report",
            "",
            "## Executive Summary",
            f"- Raw setups captured: `{int(row.raw_setups)}`.",
            f"- Final candidates captured: `{int(row.final_candidates)}`.",
            f"- Accepted trades: `{int(row.accepted_trades)}`.",
            f"- Raw-to-trade conversion: `{row.raw_to_trade_rate:.3f}`.",
            f"- No-OB-only PF versus all trades: `{no_ob_pf:.2f}` vs `{all_pf:.2f}`.",
            "- ML status: **experimental_only**.",
            "",
            "## Rejection Funnel",
            *funnel_lines,
            "",
            "## Top Rejection Reasons",
            *reason_lines,
            "",
            "## Confluence Read",
            f"- OB accepted trades visible in raw conversion table: `{ob_trade_count}`.",
            f"- FVG/OB+FVG accepted trades visible in raw conversion table: `{fvg_trade_count}`.",
            "- OB-only weakness remains confirmed at the accepted-trade level; raw-level structural confirmation should be treated as directional until rejected setups receive counterfactual outcomes.",
            "",
            "## Current Selection Quality",
            "- The strategy is selecting sparsely from a much larger raw setup universe.",
            "- The raw ledger now makes gate-level drift and rejection concentration auditable.",
            "- Outcome fields are populated only for linked accepted trades and are excluded from entry-time feature columns.",
            "",
            "## Next Experiments",
            "- Add a side-effect-free signal probe to observe raw setups during open-position and max-trade-cap periods.",
            "- Build a counterfactual outcome simulator for rejected final candidates only, clearly separated from live features.",
            "- Re-run classification on the full raw setup universe once rejected setups have valid labels.",
        ]
    )


def analyse(
    trades_csv: Path,
    output_dir: Path,
    *,
    train_months: int = 3,
    test_months: int = 1,
    embargo_days: int = 1,
    exclude_ob_only: bool = False,
    include_confluence_types: set[str] | None = None,
    exclude_confluence_types: set[str] | None = None,
    min_atr_percentile: float | None = None,
    max_atr_percentile: float | None = None,
    session_filter: set[str] | None = None,
    symbol_filter: set[str] | None = None,
    raw_setups_csv: Path | None = None,
) -> dict[str, pd.DataFrame]:
    df = _prepare_features(pd.read_csv(trades_csv))
    df = _filter_frame(
        df,
        exclude_ob_only=exclude_ob_only,
        include_confluence_types=include_confluence_types,
        exclude_confluence_types=exclude_confluence_types,
        min_atr_percentile=min_atr_percentile,
        max_atr_percentile=max_atr_percentile,
        session_filter=session_filter,
        symbol_filter=symbol_filter,
    )
    folds = _walk_forward_splits(
        df,
        train_months=train_months,
        test_months=test_months,
        embargo_days=embargo_days,
    )
    if not folds:
        raise ValueError("No valid chronological folds generated for feature diagnostics.")

    fold_metrics, predictions, permutation = _fold_metrics(df, folds)
    classification_metrics, probability_buckets, oof_classification, calibration = (
        _classification_diagnostics(df, folds, FEATURES)
    )
    regime_dashboard = _regime_dashboard(df)
    feature_set_comparison = _feature_set_comparison(df, folds)
    outputs = {
        "fold_metrics": fold_metrics,
        "fold_by_fold_summary": fold_metrics,
        "oof_predictions": predictions,
        "permutation_importance_by_fold": permutation,
        "permutation_importance": _aggregate_permutation(permutation),
        "classification_metrics": classification_metrics,
        "probability_buckets": probability_buckets,
        "oof_classification_predictions": oof_classification,
        "calibration": calibration,
        "group_pnl": _group_pnl(df),
        "ablation_summary": _ablation_summary(df),
        "confluence_performance": regime_dashboard[regime_dashboard["feature"].eq("confluence_type")],
        "regime_performance": regime_dashboard,
        "regime_dashboard": regime_dashboard,
        "target_summary": _target_summary(df),
        "distribution_shift": _distribution_shift(df, folds),
        "feature_drift_by_fold": _distribution_shift(df, folds),
        "feature_set_comparison": feature_set_comparison,
        "no_calendar_validation": feature_set_comparison[
            feature_set_comparison["feature_set"].isin(
                ["all_features", "no_calendar", "calendar_only", "no_calendar_session_bucket"]
            )
        ],
    }
    raw_outputs = (
        _raw_setup_diagnostics(_prepare_raw_setups(raw_setups_csv))
        if raw_setups_csv is not None
        else _empty_raw_outputs()
    )
    outputs.update(raw_outputs)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    (output_dir / "ml_validation_report.md").write_text(_report_markdown(outputs), encoding="utf-8")
    if raw_setups_csv is not None:
        (output_dir.parent / "raw_setup_validation_report.md").write_text(
            _raw_validation_report(outputs),
            encoding="utf-8",
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trades", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--test-months", type=int, default=1)
    parser.add_argument("--embargo-days", type=int, default=1)
    parser.add_argument("--exclude-ob-only", action="store_true")
    parser.add_argument("--include-confluence-types", default="")
    parser.add_argument("--exclude-confluence-types", default="")
    parser.add_argument("--min-atr-percentile", type=float, default=None)
    parser.add_argument("--max-atr-percentile", type=float, default=None)
    parser.add_argument("--session-filter", default="")
    parser.add_argument("--symbol-filter", default="")
    parser.add_argument("--raw-setups", type=Path, default=None)
    args = parser.parse_args()

    def _csv_set(value: str) -> set[str] | None:
        parsed = {v.strip() for v in value.split(",") if v.strip()}
        return parsed or None

    outputs = analyse(
        args.trades,
        args.output,
        train_months=args.train_months,
        test_months=args.test_months,
        embargo_days=args.embargo_days,
        exclude_ob_only=args.exclude_ob_only,
        include_confluence_types=_csv_set(args.include_confluence_types),
        exclude_confluence_types=_csv_set(args.exclude_confluence_types),
        min_atr_percentile=args.min_atr_percentile,
        max_atr_percentile=args.max_atr_percentile,
        session_filter=_csv_set(args.session_filter),
        symbol_filter=_csv_set(args.symbol_filter),
        raw_setups_csv=args.raw_setups,
    )
    fold_metrics = outputs["fold_metrics"]
    importance = outputs["permutation_importance"]
    grouped = outputs["group_pnl"]
    target = outputs["target_summary"].iloc[0]
    shift = outputs["distribution_shift"]
    classification = outputs["classification_metrics"]
    ablations = outputs["ablation_summary"]
    raw_summary = outputs["raw_setup_summary"]

    print("Chronological walk-forward diagnostics")
    print(f"Folds: {len(fold_metrics)}")
    print(f"Mean model R2: {fold_metrics['model_r2'].mean():.4f}")
    print(f"Mean zero baseline R2: {fold_metrics['zero_r2'].mean():.4f}")
    print(f"Mean train-mean baseline R2: {fold_metrics['train_mean_r2'].mean():.4f}")
    print(f"Mean directional accuracy: {fold_metrics['directional_accuracy'].mean():.3f}")
    print(f"Mean positive precision: {fold_metrics['positive_precision'].mean():.3f}")
    print(f"Mean Spearman IC: {fold_metrics['prediction_ic_spearman'].mean():.4f}")
    print(
        "Target noise: "
        f"mean=${target['mean_pnl']:.2f}, std=${target['std_pnl']:.2f}, "
        f"SNR={target['signal_to_noise']:.3f}, win_rate={target['win_rate']:.3f}"
    )

    print("\nTime-aware permutation importance:")
    print(importance.head(12).to_string(index=False))
    print("\nBest P&L groups:")
    print(grouped.sort_values("pnl", ascending=False).head(12).to_string(index=False))
    print("\nWorst P&L groups:")
    print(grouped.sort_values("pnl", ascending=True).head(12).to_string(index=False))
    print("\nLargest train/test drift rows:")
    print(shift.sort_values("std_mean_diff", key=lambda s: s.abs(), ascending=False).head(12).to_string(index=False))
    if not classification.empty:
        print("\nClassification metrics by target:")
        print(classification.groupby("target")[["precision", "recall", "f1", "roc_auc", "pr_auc", "brier"]].mean().to_string())
    print("\nAblation summary:")
    print(ablations[["label", "trades", "pnl", "profit_factor", "max_drawdown", "win_rate", "avg_r"]].to_string(index=False))
    if not raw_summary.empty:
        row = raw_summary.iloc[0]
        print("\nRaw setup ledger:")
        print(
            f"raw_setups={int(row['raw_setups'])} "
            f"final_candidates={int(row['final_candidates'])} "
            f"accepted_trades={int(row['accepted_trades'])} "
            f"raw_to_trade_rate={row['raw_to_trade_rate']:.3f}"
        )
    print(f"\nReport written to {args.output / 'ml_validation_report.md'}")


if __name__ == "__main__":
    main()
