from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from march_madness_model import (
    FEATURE_SET_VARIANTS,
    SEEDS_FILE,
    TOURNEY_FILE,
    build_combined_training_data,
    build_team_metrics,
    evaluate_probability_predictions,
    fit_base_model,
    get_round_lookup,
    map_day_to_round,
    matchup_feature_frame,
    parse_seed,
)


PRODUCTION_FEATURE_SET = "replace_both_interactions"
PRODUCTION_C = 0.2
BACKTEST_YEARS = list(range(2018, 2026))
NO_SEED_FEATURES = [
    column
    for column in FEATURE_SET_VARIANTS[PRODUCTION_FEATURE_SET]
    if column not in {"seed_edge", "seed_a_offense_vs_b_defense"}
]
MEDIUM_GAP_FEATURE_VARIANTS: dict[str, list[str]] = {
    "production": [],
    "medium_gap_underdog_three_point_rate": ["medium_gap_underdog_three_point_rate"],
    "medium_gap_tempo_diff": ["medium_gap_tempo_diff"],
    "medium_gap_a_offense_vs_b_defense": ["medium_gap_a_offense_vs_b_defense"],
    "medium_gap_three_point_pct_diff": ["medium_gap_three_point_pct_diff"],
}
MEDIUM_GAP_CORRECTION_GRID = [0.01, 0.02, 0.03]


def add_medium_gap_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    if "seed_edge" not in enriched.columns:
        return enriched
    seed_gap_abs = enriched["seed_edge"].abs()
    medium_gap_indicator = ((seed_gap_abs >= 3) & (seed_gap_abs <= 5)).astype(float)
    enriched["medium_gap_indicator"] = medium_gap_indicator
    enriched["medium_gap_underdog_three_point_rate"] = medium_gap_indicator * enriched["underdog_three_point_rate"]
    enriched["medium_gap_tempo_diff"] = medium_gap_indicator * enriched["tempo_diff"]
    enriched["medium_gap_a_offense_vs_b_defense"] = medium_gap_indicator * enriched["a_offense_vs_b_defense"]
    enriched["medium_gap_three_point_pct_diff"] = medium_gap_indicator * enriched["three_point_pct_diff"]
    return enriched


def build_seed_oriented_tournament_games(team_stats: pd.DataFrame) -> pd.DataFrame:
    round_lookup = get_round_lookup()
    seeds = pd.read_csv(SEEDS_FILE)
    seeds["SeedNum"] = seeds["Seed"].map(parse_seed)
    seed_lookup = seeds[["Season", "TeamID", "Seed", "SeedNum"]]

    tourney = pd.read_csv(TOURNEY_FILE, usecols=["Season", "DayNum", "WTeamID", "LTeamID"])
    tourney["RoundNum"] = tourney["DayNum"].map(lambda day_num: map_day_to_round(day_num, round_lookup))
    tourney = (
        tourney.merge(seed_lookup, left_on=["Season", "WTeamID"], right_on=["Season", "TeamID"], how="left")
        .rename(columns={"Seed": "WSeed", "SeedNum": "WSeedNum"})
        .drop(columns=["TeamID"])
        .merge(seed_lookup, left_on=["Season", "LTeamID"], right_on=["Season", "TeamID"], how="left")
        .rename(columns={"Seed": "LSeed", "SeedNum": "LSeedNum"})
        .drop(columns=["TeamID"])
    )

    rows: list[dict[str, int | float | str]] = []
    for row in tourney.itertuples(index=False):
        w_seed = getattr(row, "WSeedNum")
        l_seed = getattr(row, "LSeedNum")
        if pd.notna(w_seed) and pd.notna(l_seed) and int(w_seed) != int(l_seed):
            if int(w_seed) < int(l_seed):
                team_a = int(row.WTeamID)
                team_b = int(row.LTeamID)
                team_a_seed = int(w_seed)
                team_b_seed = int(l_seed)
                target = 1
            else:
                team_a = int(row.LTeamID)
                team_b = int(row.WTeamID)
                team_a_seed = int(l_seed)
                team_b_seed = int(w_seed)
                target = 0
        else:
            team_a = int(row.WTeamID)
            team_b = int(row.LTeamID)
            team_a_seed = int(w_seed) if pd.notna(w_seed) else 0
            team_b_seed = int(l_seed) if pd.notna(l_seed) else 0
            target = 1

        rows.append(
            {
                "Season": int(row.Season),
                "DayNum": int(row.DayNum),
                "RoundNum": int(row.RoundNum),
                "TeamAID": team_a,
                "TeamBID": team_b,
                "TeamASeedNum": team_a_seed,
                "TeamBSeedNum": team_b_seed,
                "target": int(target),
            }
        )

    games = pd.DataFrame(rows)
    features = matchup_feature_frame(games[["Season", "TeamAID", "TeamBID", "RoundNum", "target"]], team_stats).drop(
        columns=["Season", "TeamAID", "TeamBID", "RoundNum", "target"], errors="ignore"
    )
    games = pd.concat([games.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    games["seed_gap"] = games["TeamBSeedNum"] - games["TeamASeedNum"]
    games["seed_matchup"] = games.apply(
        lambda r: f"{int(min(r['TeamASeedNum'], r['TeamBSeedNum']))} vs {int(max(r['TeamASeedNum'], r['TeamBSeedNum']))}",
        axis=1,
    )
    games["seed_gap_bucket"] = games["seed_gap"].map(
        lambda gap: "small_gap_1_2" if 1 <= gap <= 2 else ("medium_gap_3_5" if 3 <= gap <= 5 else ("large_gap_6_plus" if gap >= 6 else "equal_seed"))
    )
    games["actual_upset"] = 1 - games["target"]
    return games


def fit_walkforward_predictions(
    combined_X: pd.DataFrame,
    combined_y: pd.Series,
    combined_seasons: pd.Series,
    combined_source: pd.Series,
    tournament_games: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    prediction_frames: list[pd.DataFrame] = []
    for season in BACKTEST_YEARS:
        train_mask = ((combined_source == "tournament") & (combined_seasons < season)) | (
            (combined_source == "regular_season") & (combined_seasons <= season)
        )
        test_mask = tournament_games["Season"] == season
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        model = fit_base_model(
            combined_X.loc[train_mask, feature_columns],
            combined_y.loc[train_mask],
            seed_shrinkage=1.0,
            c_value=PRODUCTION_C,
        )
        season_games = tournament_games.loc[test_mask].copy()
        season_games["pred_high_seed_win_prob"] = model.predict_proba(season_games[feature_columns])[:, 1]
        prediction_frames.append(season_games)

    return pd.concat(prediction_frames, ignore_index=True)


def fit_walkforward_predictions_with_correction(
    combined_X: pd.DataFrame,
    combined_y: pd.Series,
    combined_seasons: pd.Series,
    combined_source: pd.Series,
    tournament_games: pd.DataFrame,
    feature_columns: list[str],
    medium_gap_lift: float,
) -> pd.DataFrame:
    predictions = fit_walkforward_predictions(
        combined_X,
        combined_y,
        combined_seasons,
        combined_source,
        tournament_games,
        feature_columns,
    )
    adjusted = predictions.copy()
    medium_gap_mask = adjusted["seed_gap_bucket"].eq("medium_gap_3_5")
    adjusted.loc[medium_gap_mask, "pred_high_seed_win_prob"] = (
        adjusted.loc[medium_gap_mask, "pred_high_seed_win_prob"] - medium_gap_lift
    ).clip(1e-6, 1 - 1e-6)
    return adjusted


def summarize_yearly(predictions: pd.DataFrame, model_name: str) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for season, season_df in predictions.groupby("Season", sort=True):
        rows.append(
            {
                "Season": int(season),
                "Model": model_name,
                **evaluate_probability_predictions(
                    season_df["target"], season_df["pred_high_seed_win_prob"], model_name
                ),
            }
        )
    overall = evaluate_probability_predictions(predictions["target"], predictions["pred_high_seed_win_prob"], model_name)
    rows.append({"Season": "Overall", "Model": model_name, **overall})
    return pd.DataFrame(rows)


def build_seed_matchup_table(predictions: pd.DataFrame) -> pd.DataFrame:
    seeded = predictions[predictions["seed_gap"] > 0].copy()
    table = (
        seeded.groupby("seed_matchup", dropna=False)
        .agg(
            games=("target", "size"),
            predicted_higher_seed_win=("pred_high_seed_win_prob", "mean"),
            actual_higher_seed_win=("target", "mean"),
        )
        .reset_index()
    )
    table["calibration_gap"] = table["predicted_higher_seed_win"] - table["actual_higher_seed_win"]
    return table.sort_values("seed_matchup").reset_index(drop=True)


def build_seed_gap_table(predictions: pd.DataFrame) -> pd.DataFrame:
    seeded = predictions[predictions["seed_gap"] > 0].copy()
    table = (
        seeded.groupby("seed_gap_bucket", dropna=False)
        .agg(
            games=("target", "size"),
            predicted_higher_seed_win=("pred_high_seed_win_prob", "mean"),
            actual_higher_seed_win=("target", "mean"),
        )
        .reset_index()
    )
    table["calibration_gap"] = table["predicted_higher_seed_win"] - table["actual_higher_seed_win"]
    bucket_order = ["small_gap_1_2", "medium_gap_3_5", "large_gap_6_plus"]
    table["bucket_order"] = table["seed_gap_bucket"].map({name: idx for idx, name in enumerate(bucket_order)})
    return table.sort_values("bucket_order").drop(columns=["bucket_order"]).reset_index(drop=True)


def build_upset_table(predictions: pd.DataFrame) -> pd.DataFrame:
    seeded = predictions[predictions["seed_gap"] > 0].copy()
    rows = [
        {
            "metric": "overall_upset_rate",
            "predicted": float((1 - seeded["pred_high_seed_win_prob"]).mean()),
            "actual": float(seeded["actual_upset"].mean()),
        }
    ]
    for bucket, bucket_df in seeded.groupby("seed_gap_bucket", sort=False):
        rows.append(
            {
                "metric": f"{bucket}_upset_rate",
                "predicted": float((1 - bucket_df["pred_high_seed_win_prob"]).mean()),
                "actual": float(bucket_df["actual_upset"].mean()),
            }
        )
    table = pd.DataFrame(rows)
    table["gap"] = table["predicted"] - table["actual"]
    return table


def build_distribution_table(predictions: pd.DataFrame) -> pd.DataFrame:
    probs = predictions["pred_high_seed_win_prob"]
    distribution = (
        probs.groupby(
            pd.cut(
                probs,
                bins=[0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0],
                include_lowest=True,
            )
        )
        .size()
        .reset_index(name="count")
        .rename(columns={"pred_high_seed_win_prob": "bucket"})
    )
    summary_rows = [
        {"metric": "pct_above_095", "value": float((probs > 0.95).mean())},
        {"metric": "pct_between_040_070", "value": float(((probs >= 0.4) & (probs <= 0.7)).mean())},
        {"metric": "mean_probability", "value": float(probs.mean())},
        {"metric": "actual_higher_seed_win_rate", "value": float(predictions["target"].mean())},
    ]
    summary = pd.DataFrame(summary_rows)
    distribution["bucket"] = distribution["bucket"].astype(str)
    return distribution, summary


def build_variant_metric_row(predictions: pd.DataFrame, model_name: str) -> dict[str, float | str]:
    overall = evaluate_probability_predictions(predictions["target"], predictions["pred_high_seed_win_prob"], model_name)
    medium_gap = predictions[predictions["seed_gap_bucket"] == "medium_gap_3_5"].copy()
    large_gap = predictions[predictions["seed_gap_bucket"] == "large_gap_6_plus"].copy()
    medium_gap_upset_pred = float((1 - medium_gap["pred_high_seed_win_prob"]).mean()) if not medium_gap.empty else 0.0
    medium_gap_upset_actual = float(medium_gap["actual_upset"].mean()) if not medium_gap.empty else 0.0
    large_gap_pred = float(large_gap["pred_high_seed_win_prob"].mean()) if not large_gap.empty else 0.0
    large_gap_actual = float(large_gap["target"].mean()) if not large_gap.empty else 0.0
    return {
        "variant": model_name,
        "log_loss": float(overall["log_loss"]),
        "brier_score": float(overall["brier_score"]),
        "ece": float(overall["ece"]),
        "medium_gap_predicted_upset_rate": medium_gap_upset_pred,
        "medium_gap_actual_upset_rate": medium_gap_upset_actual,
        "medium_gap_upset_gap": medium_gap_upset_pred - medium_gap_upset_actual,
        "large_gap_calibration_gap": large_gap_pred - large_gap_actual,
    }


def build_model_comparison_table(
    with_seed_predictions: pd.DataFrame,
    without_seed_predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for model_name, frame in [
        ("with_seed", with_seed_predictions),
        ("without_seed", without_seed_predictions),
    ]:
        rows.append(
            {
                "model": model_name,
                **evaluate_probability_predictions(frame["target"], frame["pred_high_seed_win_prob"], model_name),
            }
        )
    return pd.DataFrame(rows).sort_values(["log_loss", "brier_score", "ece"]).reset_index(drop=True)


def build_diagnosis(
    seed_gap_table: pd.DataFrame,
    upset_table: pd.DataFrame,
    comparison_table: pd.DataFrame,
) -> dict[str, object]:
    large_gap = seed_gap_table.loc[seed_gap_table["seed_gap_bucket"] == "large_gap_6_plus"]
    large_gap_calibration = float(large_gap["calibration_gap"].iloc[0]) if not large_gap.empty else 0.0
    overall_upset_gap = float(upset_table.loc[upset_table["metric"] == "overall_upset_rate", "gap"].iloc[0])
    with_seed = comparison_table.loc[comparison_table["model"] == "with_seed"].iloc[0]
    without_seed = comparison_table.loc[comparison_table["model"] == "without_seed"].iloc[0]
    seed_value = float(with_seed["log_loss"] - without_seed["log_loss"])

    if seed_value < -0.003 and abs(large_gap_calibration) < 0.03 and abs(overall_upset_gap) < 0.03:
        conclusion = "Model is well-calibrated and not seed-biased"
    elif large_gap_calibration > 0.03 or overall_upset_gap < -0.03 or seed_value >= 0:
        conclusion = "Model over-relies on seed and suppresses upset probability"
    else:
        conclusion = "Model uses seed constructively, but upset calibration should be watched"

    return {
        "conclusion": conclusion,
        "large_gap_calibration_gap": large_gap_calibration,
        "overall_upset_gap": overall_upset_gap,
        "with_seed_log_loss": float(with_seed["log_loss"]),
        "without_seed_log_loss": float(without_seed["log_loss"]),
    }


def build_medium_gap_refinement_report(
    combined_X: pd.DataFrame,
    combined_y: pd.Series,
    combined_seasons: pd.Series,
    combined_source: pd.Series,
    tournament_games: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    enriched_X = add_medium_gap_features(combined_X)
    enriched_games = add_medium_gap_features(tournament_games)
    base_features = FEATURE_SET_VARIANTS[PRODUCTION_FEATURE_SET]

    variant_predictions: dict[str, pd.DataFrame] = {}
    metric_rows: list[dict[str, float | str]] = []
    yearly_rows: list[pd.DataFrame] = []

    for variant_name, extra_features in MEDIUM_GAP_FEATURE_VARIANTS.items():
        feature_columns = base_features + extra_features
        preds = fit_walkforward_predictions(
            enriched_X,
            combined_y,
            combined_seasons,
            combined_source,
            enriched_games,
            feature_columns,
        )
        variant_predictions[variant_name] = preds
        metric_rows.append(build_variant_metric_row(preds, variant_name))
        yearly = summarize_yearly(preds, variant_name).rename(columns={"Model": "variant"})
        yearly_rows.append(yearly)

    for lift in MEDIUM_GAP_CORRECTION_GRID:
        variant_name = f"medium_gap_correction_{lift:.2f}"
        preds = fit_walkforward_predictions_with_correction(
            enriched_X,
            combined_y,
            combined_seasons,
            combined_source,
            enriched_games,
            base_features,
            lift,
        )
        variant_predictions[variant_name] = preds
        metric_rows.append(build_variant_metric_row(preds, variant_name))
        yearly = summarize_yearly(preds, variant_name).rename(columns={"Model": "variant"})
        yearly_rows.append(yearly)

    metrics_table = pd.DataFrame(metric_rows).sort_values(
        ["log_loss", "brier_score", "ece", "medium_gap_upset_gap"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    yearly_table = pd.concat(yearly_rows, ignore_index=True)

    production_row = metrics_table.loc[metrics_table["variant"] == "production"].iloc[0]
    eligible = metrics_table[
        (metrics_table["log_loss"] <= production_row["log_loss"] + 0.003)
        & (metrics_table["brier_score"] <= production_row["brier_score"] + 0.003)
        & (metrics_table["ece"] <= production_row["ece"] + 0.02)
        & (metrics_table["large_gap_calibration_gap"].sub(production_row["large_gap_calibration_gap"]).abs() <= 0.02)
    ].copy()
    eligible["medium_gap_improvement"] = eligible["medium_gap_upset_gap"].abs() - abs(production_row["medium_gap_upset_gap"])
    eligible = eligible.sort_values(
        ["medium_gap_improvement", "log_loss", "brier_score", "ece"],
        ascending=[True, True, True, True],
    )
    best_variant = str(eligible.iloc[0]["variant"]) if not eligible.empty else "production"
    promoted = best_variant != "production"

    comparison_rows = []
    for variant_name in ["production", best_variant] if promoted else ["production"]:
        preds = variant_predictions[variant_name]
        medium_gap = preds[preds["seed_gap_bucket"] == "medium_gap_3_5"]
        large_gap = preds[preds["seed_gap_bucket"] == "large_gap_6_plus"]
        comparison_rows.append(
            {
                "variant": variant_name,
                "medium_gap_predicted_upset_rate": float((1 - medium_gap["pred_high_seed_win_prob"]).mean()),
                "medium_gap_actual_upset_rate": float(medium_gap["actual_upset"].mean()),
                "medium_gap_upset_gap": float((1 - medium_gap["pred_high_seed_win_prob"]).mean() - medium_gap["actual_upset"].mean()),
                "large_gap_predicted_higher_seed_win": float(large_gap["pred_high_seed_win_prob"].mean()),
                "large_gap_actual_higher_seed_win": float(large_gap["target"].mean()),
                "large_gap_calibration_gap": float(large_gap["pred_high_seed_win_prob"].mean() - large_gap["target"].mean()),
            }
        )
    localized_comparison = pd.DataFrame(comparison_rows)

    decision = {
        "selected_variant": best_variant,
        "promote_refinement": promoted,
        "reason": (
            "Promote targeted refinement"
            if promoted
            else "Keep current production model unchanged"
        ),
    }
    return metrics_table, yearly_table, localized_comparison, decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Run walk-forward tournament backtest and seed-bias diagnostics.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/wilcroutwater/Documents/Playground/output/march_madness"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    team_stats = build_team_metrics()
    combined_X, combined_y, combined_seasons, combined_source = build_combined_training_data(team_stats)
    tournament_games = build_seed_oriented_tournament_games(team_stats)

    with_seed_predictions = fit_walkforward_predictions(
        combined_X,
        combined_y,
        combined_seasons,
        combined_source,
        tournament_games,
        FEATURE_SET_VARIANTS[PRODUCTION_FEATURE_SET],
    )
    without_seed_predictions = fit_walkforward_predictions(
        combined_X,
        combined_y,
        combined_seasons,
        combined_source,
        tournament_games,
        NO_SEED_FEATURES,
    )

    yearly_with_seed = summarize_yearly(with_seed_predictions, "with_seed")
    seed_matchup_table = build_seed_matchup_table(with_seed_predictions)
    seed_gap_table = build_seed_gap_table(with_seed_predictions)
    upset_table = build_upset_table(with_seed_predictions)
    distribution_table, distribution_summary = build_distribution_table(with_seed_predictions)
    comparison_table = build_model_comparison_table(with_seed_predictions, without_seed_predictions)
    diagnosis = build_diagnosis(seed_gap_table, upset_table, comparison_table)
    (
        medium_gap_refinement_metrics,
        medium_gap_refinement_yearly,
        medium_gap_localized_comparison,
        medium_gap_decision,
    ) = build_medium_gap_refinement_report(
        combined_X,
        combined_y,
        combined_seasons,
        combined_source,
        tournament_games,
    )

    (args.output_dir / "tournament_backtest_yearly.csv").write_text(yearly_with_seed.to_csv(index=False), encoding="utf-8")
    (args.output_dir / "tournament_backtest_predictions.csv").write_text(with_seed_predictions.to_csv(index=False), encoding="utf-8")
    (args.output_dir / "tournament_seed_matchup_calibration.csv").write_text(seed_matchup_table.to_csv(index=False), encoding="utf-8")
    (args.output_dir / "tournament_seed_gap_analysis.csv").write_text(seed_gap_table.to_csv(index=False), encoding="utf-8")
    (args.output_dir / "tournament_upset_analysis.csv").write_text(upset_table.to_csv(index=False), encoding="utf-8")
    (args.output_dir / "tournament_probability_distribution.csv").write_text(distribution_table.to_csv(index=False), encoding="utf-8")
    (args.output_dir / "tournament_distribution_summary.csv").write_text(distribution_summary.to_csv(index=False), encoding="utf-8")
    (args.output_dir / "tournament_model_comparison.csv").write_text(comparison_table.to_csv(index=False), encoding="utf-8")
    (args.output_dir / "tournament_yearly_with_without_seed.csv").write_text(
        pd.concat([summarize_yearly(with_seed_predictions, "with_seed"), summarize_yearly(without_seed_predictions, "without_seed")]).to_csv(index=False),
        encoding="utf-8",
    )
    (args.output_dir / "medium_gap_refinement_metrics.csv").write_text(
        medium_gap_refinement_metrics.to_csv(index=False), encoding="utf-8"
    )
    (args.output_dir / "medium_gap_refinement_yearly.csv").write_text(
        medium_gap_refinement_yearly.to_csv(index=False), encoding="utf-8"
    )
    (args.output_dir / "medium_gap_refinement_comparison.csv").write_text(
        medium_gap_localized_comparison.to_csv(index=False), encoding="utf-8"
    )
    (args.output_dir / "medium_gap_refinement_decision.json").write_text(
        json.dumps(medium_gap_decision, indent=2), encoding="utf-8"
    )
    (args.output_dir / "tournament_diagnosis.json").write_text(json.dumps(diagnosis, indent=2), encoding="utf-8")

    print(f"Saved walk-forward yearly metrics to {args.output_dir / 'tournament_backtest_yearly.csv'}")
    print(f"Saved tournament predictions to {args.output_dir / 'tournament_backtest_predictions.csv'}")
    print(f"Saved seed matchup calibration to {args.output_dir / 'tournament_seed_matchup_calibration.csv'}")
    print(f"Saved seed gap analysis to {args.output_dir / 'tournament_seed_gap_analysis.csv'}")
    print(f"Saved upset analysis to {args.output_dir / 'tournament_upset_analysis.csv'}")
    print(f"Saved probability distribution to {args.output_dir / 'tournament_probability_distribution.csv'}")
    print(f"Saved model comparison to {args.output_dir / 'tournament_model_comparison.csv'}")
    print(f"Saved medium-gap refinement metrics to {args.output_dir / 'medium_gap_refinement_metrics.csv'}")
    print(f"Saved medium-gap refinement comparison to {args.output_dir / 'medium_gap_refinement_comparison.csv'}")
    print(f"Saved medium-gap refinement decision to {args.output_dir / 'medium_gap_refinement_decision.json'}")
    print(f"Saved diagnosis to {args.output_dir / 'tournament_diagnosis.json'}")
    print(diagnosis["conclusion"])


if __name__ == "__main__":
    main()
