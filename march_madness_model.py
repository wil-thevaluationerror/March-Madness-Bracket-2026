from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


DATA_DIR = Path("/Users/wilcroutwater/Downloads/march-machine-learning-mania-2026")
REGULAR_SEASON_FILE = DATA_DIR / "MRegularSeasonDetailedResults.csv"
REGULAR_SEASON_COMPACT_FILE = DATA_DIR / "MRegularSeasonCompactResults.csv"
TOURNEY_FILE = DATA_DIR / "MNCAATourneyDetailedResults.csv"
TOURNEY_COMPACT_FILE = DATA_DIR / "MNCAATourneyCompactResults.csv"
SEEDS_FILE = DATA_DIR / "MNCAATourneySeeds.csv"
SLOTS_FILE = DATA_DIR / "MNCAATourneySlots.csv"
SEED_ROUND_SLOTS_FILE = DATA_DIR / "MNCAATourneySeedRoundSlots.csv"
MASSEY_FILE = DATA_DIR / "MMasseyOrdinals.csv"
TEAMS_FILE = DATA_DIR / "MTeams.csv"
TEAM_CONFERENCES_FILE = DATA_DIR / "MTeamConferences.csv"
CONFERENCES_FILE = DATA_DIR / "Conferences.csv"
CONFERENCE_TOURNEY_FILE = DATA_DIR / "MConferenceTourneyGames.csv"
GAME_CITIES_FILE = DATA_DIR / "MGameCities.csv"
CITIES_FILE = DATA_DIR / "Cities.csv"
SAMPLE_SUBMISSION_FILE = DATA_DIR / "SampleSubmissionStage2.csv"

ALL_FEATURE_COLUMNS = [
    "eff_margin_diff",
    "def_eff_edge",
    "off_eff_diff",
    "tov_rate_edge",
    "avg_margin_diff",
    "three_point_pct_diff",
    "three_point_rate_diff",
    "three_point_volatility_edge",
    "opp_three_point_volatility_edge",
    "momentum_edge",
    "predictive_rank_edge",
    "seed_edge",
    "seed_strength_interaction",
    "a_offense_vs_b_defense",
    "b_offense_vs_a_defense",
    "underdog_a_offense_vs_b_defense",
    "seed_a_offense_vs_b_defense",
    "tempo_diff",
    "abs_tempo_diff",
    "underdog_three_point_rate",
    "underdog_tempo_diff",
    "site_advantage",
    "defense_round_pressure",
]

BASELINE_FEATURE_COLUMNS = [
    "eff_margin_diff",
    "def_eff_edge",
    "off_eff_diff",
    "tov_rate_edge",
    "three_point_pct_diff",
    "seed_edge",
    "predictive_rank_edge",
    "site_advantage",
]

MATCHUP_REPLACEMENT_COLUMNS = [
    "a_offense_vs_b_defense",
    "b_offense_vs_a_defense",
    "underdog_three_point_rate",
    "underdog_tempo_diff",
]

FEATURE_SET_VARIANTS = {
    "baseline_compact": BASELINE_FEATURE_COLUMNS,
    "hybrid_minimal": [
        "predictive_rank_edge",
        "site_advantage",
        "seed_edge",
        "tov_rate_edge",
        "a_offense_vs_b_defense",
        "b_offense_vs_a_defense",
        "underdog_three_point_rate",
    ],
    "replace_eff_margin": [
        column for column in BASELINE_FEATURE_COLUMNS if column != "eff_margin_diff"
    ] + MATCHUP_REPLACEMENT_COLUMNS,
    "replace_predictive_rank": [
        column for column in BASELINE_FEATURE_COLUMNS if column != "predictive_rank_edge"
    ] + MATCHUP_REPLACEMENT_COLUMNS,
    "replace_both": [
        column
        for column in BASELINE_FEATURE_COLUMNS
        if column not in {"eff_margin_diff", "predictive_rank_edge"}
    ] + MATCHUP_REPLACEMENT_COLUMNS,
    "replace_both_interactions": [
        column
        for column in BASELINE_FEATURE_COLUMNS
        if column not in {"eff_margin_diff", "predictive_rank_edge"}
    ]
    + MATCHUP_REPLACEMENT_COLUMNS
    + [
        "underdog_a_offense_vs_b_defense",
        "seed_a_offense_vs_b_defense",
    ],
}

TEAM_SCORE_COMPONENT_WEIGHTS = {
    "strength_score": 0.44,
    "defense_score": 0.23,
    "stability_score": 0.14,
    "variance_upside_score": 0.07,
    "variance_risk_score": -0.05,
    "context_score": 0.05,
    "momentum_score": 0.02,
}


@dataclass
class MatchupResult:
    winner_team_id: int
    loser_team_id: int
    winner_prob: float


@dataclass
class CalibrationBundle:
    method: str
    calibrator: object
    params: dict[str, float | str]


@dataclass
class TreeBoostingModel:
    preprocessor: ColumnTransformer
    trees: list[tuple[DecisionTreeRegressor, np.ndarray]]
    feature_order: list[str]
    init_score: float
    learning_rate: float
    feature_importance_: np.ndarray

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        transformed = self.preprocessor.transform(X[self.feature_order])
        return transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = self._transform(X)
        raw_scores = np.full(X_arr.shape[0], self.init_score, dtype=float)
        for tree, feature_idx in self.trees:
            raw_scores += self.learning_rate * tree.predict(X_arr[:, feature_idx])
        return raw_scores

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        prob = logits_to_probabilities(self.decision_function(X))
        return np.column_stack([1.0 - prob, prob])


@dataclass
class BoostingArrayCache:
    X_array: np.ndarray
    y_array: np.ndarray
    feature_order: list[str]
    preprocessor: ColumnTransformer


PLATT_C_GRID = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
PLATT_BLEND_GRID = [0.15, 0.3, 0.45, 0.6]
PLATT_CLIP_VALUE = 2.5
SEED_SHRINKAGE_GRID = [1.0, 0.5, 0.25, 0.05]
MIN_PREGAME_GAMES = 5
TEMPERATURE_GRID = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0, 2.25, 2.5]
ENSEMBLE_ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
LOGISTIC_C_GRID = [0.2, 0.5, 1.0]
BOOSTING_STABILITY_SEEDS = [11, 23, 37, 51, 73]
BOOSTING_SUBSAMPLE_DROP_FRAC = 0.1
MEDIUM_GAP_PROBABILITY_LIFT = 0.03


def parse_seed(seed: str) -> int:
    match = re.search(r"(\d{2})", str(seed))
    if not match:
        raise ValueError(f"Unable to parse numeric seed from {seed!r}")
    return int(match.group(1))


def possessions(fga: pd.Series, oreb: pd.Series, turnovers: pd.Series, fta: pd.Series) -> pd.Series:
    return fga - oreb + turnovers + 0.475 * fta


def invert_location(location: str) -> str:
    mapping = {"H": "A", "A": "H", "N": "N"}
    return mapping.get(str(location), "N")


def load_latest_massey(system_name: str, day_cutoff: int = 133) -> pd.DataFrame:
    ordinals = pd.read_csv(MASSEY_FILE)
    ordinals = ordinals[ordinals["SystemName"] == system_name].copy()
    ordinals = ordinals[ordinals["RankingDayNum"] <= day_cutoff]
    ordinals["max_day"] = ordinals.groupby(["Season", "TeamID"])["RankingDayNum"].transform("max")
    ordinals = ordinals[ordinals["RankingDayNum"] == ordinals["max_day"]]
    return ordinals[["Season", "TeamID", "OrdinalRank"]].rename(columns={"OrdinalRank": f"{system_name.lower()}_rank"})


def season_zscore(df: pd.DataFrame, column: str) -> pd.Series:
    mean_col = df.groupby("Season")[column].transform("mean")
    std_col = df.groupby("Season")[column].transform("std").replace(0, 1).fillna(1)
    return (df[column] - mean_col) / std_col


def build_compact_team_features() -> pd.DataFrame:
    games = pd.read_csv(REGULAR_SEASON_COMPACT_FILE)
    winners = pd.DataFrame(
        {
            "Season": games["Season"],
            "DayNum": games["DayNum"],
            "TeamID": games["WTeamID"],
            "margin": games["WScore"] - games["LScore"],
            "Win": 1,
            "away_or_neutral_game": games["WLoc"].isin(["A", "N"]).astype(int),
            "away_or_neutral_win": games["WLoc"].isin(["A", "N"]).astype(int),
        }
    )
    losers = pd.DataFrame(
        {
            "Season": games["Season"],
            "DayNum": games["DayNum"],
            "TeamID": games["LTeamID"],
            "margin": games["LScore"] - games["WScore"],
            "Win": 0,
            "away_or_neutral_game": games["WLoc"].isin(["H", "N"]).astype(int),
            "away_or_neutral_win": 0,
        }
    )
    team_games = pd.concat([winners, losers], ignore_index=True)
    team_games["close_game"] = team_games["margin"].abs().le(5).astype(int)
    team_games["close_game_win"] = ((team_games["close_game"] == 1) & (team_games["Win"] == 1)).astype(int)

    compact_features = (
        team_games.groupby(["Season", "TeamID"], as_index=False)
        .agg(
            avg_margin=("margin", "mean"),
            close_games=("close_game", "sum"),
            close_game_wins=("close_game_win", "sum"),
            away_neutral_games=("away_or_neutral_game", "sum"),
            away_neutral_wins=("away_or_neutral_win", "sum"),
        )
    )
    compact_features["close_game_win_pct"] = (
        compact_features["close_game_wins"] / compact_features["close_games"].replace(0, pd.NA)
    )
    compact_features["road_neutral_win_pct"] = (
        compact_features["away_neutral_wins"] / compact_features["away_neutral_games"].replace(0, pd.NA)
    )

    recent10 = (
        team_games.sort_values(["Season", "TeamID", "DayNum"])
        .groupby(["Season", "TeamID"], group_keys=False)
        .tail(10)
        .groupby(["Season", "TeamID"], as_index=False)["Win"]
        .mean()
        .rename(columns={"Win": "recent10_win_pct"})
    )
    return compact_features.merge(recent10, on=["Season", "TeamID"], how="left")


def build_conference_features(base_team_stats: pd.DataFrame) -> pd.DataFrame:
    team_conf = pd.read_csv(TEAM_CONFERENCES_FILE)
    conf_names = pd.read_csv(CONFERENCES_FILE)
    conference_games = pd.read_csv(CONFERENCE_TOURNEY_FILE)

    winners = conference_games.rename(columns={"WTeamID": "TeamID", "LTeamID": "OppTeamID"})[
        ["Season", "ConfAbbrev", "DayNum", "TeamID", "OppTeamID"]
    ]
    winners["Win"] = 1
    losers = conference_games.rename(columns={"LTeamID": "TeamID", "WTeamID": "OppTeamID"})[
        ["Season", "ConfAbbrev", "DayNum", "TeamID", "OppTeamID"]
    ]
    losers["Win"] = 0
    conf_team_games = pd.concat([winners, losers], ignore_index=True)

    conf_tourney = (
        conf_team_games.groupby(["Season", "TeamID"], as_index=False)
        .agg(conference_tourney_games=("Win", "size"), conference_tourney_wins=("Win", "sum"))
    )
    conf_tourney["conference_tourney_win_pct"] = (
        conf_tourney["conference_tourney_wins"] / conf_tourney["conference_tourney_games"].replace(0, pd.NA)
    )

    conf_champs = (
        conference_games.sort_values(["Season", "ConfAbbrev", "DayNum"])
        .groupby(["Season", "ConfAbbrev"], as_index=False)
        .tail(1)[["Season", "ConfAbbrev", "WTeamID"]]
        .rename(columns={"WTeamID": "TeamID"})
    )
    conf_champs["conference_champion"] = 1

    conference_strength = base_team_stats.merge(team_conf, on=["Season", "TeamID"], how="left")
    conference_strength = (
        conference_strength.groupby(["Season", "ConfAbbrev"], as_index=False)["eff_margin"]
        .mean()
        .rename(columns={"eff_margin": "conference_strength"})
        .merge(conf_names, on="ConfAbbrev", how="left")
    )

    team_conference_features = team_conf.merge(conference_strength, on=["Season", "ConfAbbrev"], how="left")
    team_conference_features = team_conference_features.merge(conf_tourney, on=["Season", "TeamID"], how="left")
    team_conference_features = team_conference_features.merge(
        conf_champs[["Season", "TeamID", "conference_champion"]], on=["Season", "TeamID"], how="left"
    )
    return team_conference_features


def build_city_features() -> pd.DataFrame:
    game_cities = pd.read_csv(GAME_CITIES_FILE)
    cities = pd.read_csv(CITIES_FILE)
    winners = game_cities[["Season", "DayNum", "WTeamID", "CityID"]].rename(columns={"WTeamID": "TeamID"})
    losers = game_cities[["Season", "DayNum", "LTeamID", "CityID"]].rename(columns={"LTeamID": "TeamID"})
    team_cities = pd.concat([winners, losers], ignore_index=True).merge(cities, on="CityID", how="left")
    return (
        team_cities.groupby(["Season", "TeamID"], as_index=False)
        .agg(unique_city_count=("CityID", "nunique"), unique_state_count=("State", "nunique"))
    )


def build_seed_history_features() -> pd.DataFrame:
    tourney = pd.read_csv(TOURNEY_COMPACT_FILE)
    seeds = pd.read_csv(SEEDS_FILE)
    round_slots = get_round_lookup()
    seed_lookup = seeds.assign(seed_num=seeds["Seed"].map(parse_seed))[["Season", "TeamID", "seed_num"]]
    tourney = tourney.merge(seed_lookup, left_on=["Season", "WTeamID"], right_on=["Season", "TeamID"], how="left").rename(
        columns={"seed_num": "winner_seed_num"}
    )
    tourney = tourney.drop(columns=["TeamID"])
    tourney = tourney.merge(seed_lookup, left_on=["Season", "LTeamID"], right_on=["Season", "TeamID"], how="left").rename(
        columns={"seed_num": "loser_seed_num"}
    )
    tourney = tourney.drop(columns=["TeamID"])

    tourney["GameRound"] = tourney["DayNum"].map(lambda day_num: map_day_to_round(day_num, round_slots))
    winners = tourney[["winner_seed_num", "GameRound"]].rename(columns={"winner_seed_num": "seed_num"})
    winners["won"] = 1
    losers = tourney[["loser_seed_num", "GameRound"]].rename(columns={"loser_seed_num": "seed_num"})
    losers["won"] = 0
    seed_round = pd.concat([winners, losers], ignore_index=True).dropna(subset=["seed_num", "GameRound"])
    seed_round["seed_num"] = seed_round["seed_num"].astype(int)
    seed_round["GameRound"] = seed_round["GameRound"].astype(int)
    seed_round = (
        seed_round.groupby(["seed_num", "GameRound"], as_index=False)["won"]
        .mean()
        .rename(columns={"won": "seed_round_win_rate"})
    )
    seed_history = (
        seed_round.groupby("seed_num", as_index=False)["seed_round_win_rate"]
        .sum()
        .rename(columns={"seed_round_win_rate": "seed_history_score"})
    )
    return seed_history


def get_round_lookup() -> pd.DataFrame:
    round_lookup = pd.read_csv(SEED_ROUND_SLOTS_FILE)[["GameRound", "EarlyDayNum", "LateDayNum"]].drop_duplicates()
    return round_lookup.sort_values(["EarlyDayNum", "LateDayNum"]).reset_index(drop=True)


def map_day_to_round(day_num: int, round_lookup: pd.DataFrame) -> int:
    matched = round_lookup[(round_lookup["EarlyDayNum"] <= day_num) & (round_lookup["LateDayNum"] >= day_num)]
    if matched.empty:
        return 0
    return int(matched.iloc[0]["GameRound"])


def build_team_metrics() -> pd.DataFrame:
    games = pd.read_csv(REGULAR_SEASON_FILE)
    teams = pd.read_csv(TEAMS_FILE)[["TeamID", "TeamName"]]
    compact_features = build_compact_team_features()

    winners = pd.DataFrame(
        {
            "Season": games["Season"],
            "TeamID": games["WTeamID"],
            "OppTeamID": games["LTeamID"],
            "PointsFor": games["WScore"],
            "PointsAgainst": games["LScore"],
            "FGM": games["WFGM"],
            "FGA": games["WFGA"],
            "FGM3": games["WFGM3"],
            "FGA3": games["WFGA3"],
            "FTM": games["WFTM"],
            "FTA": games["WFTA"],
            "OR": games["WOR"],
            "DR": games["WDR"],
            "Ast": games["WAst"],
            "TO": games["WTO"],
            "Stl": games["WStl"],
            "Blk": games["WBlk"],
            "PF": games["WPF"],
            "OppFGM": games["LFGM"],
            "OppFGA": games["LFGA"],
            "OppFGM3": games["LFGM3"],
            "OppFGA3": games["LFGA3"],
            "OppFTM": games["LFTM"],
            "OppFTA": games["LFTA"],
            "OppOR": games["LOR"],
            "OppDR": games["LDR"],
            "OppAst": games["LAst"],
            "OppTO": games["LTO"],
            "OppStl": games["LStl"],
            "OppBlk": games["LBlk"],
            "OppPF": games["LPF"],
            "Win": 1,
        }
    )

    losers = pd.DataFrame(
        {
            "Season": games["Season"],
            "TeamID": games["LTeamID"],
            "OppTeamID": games["WTeamID"],
            "PointsFor": games["LScore"],
            "PointsAgainst": games["WScore"],
            "FGM": games["LFGM"],
            "FGA": games["LFGA"],
            "FGM3": games["LFGM3"],
            "FGA3": games["LFGA3"],
            "FTM": games["LFTM"],
            "FTA": games["LFTA"],
            "OR": games["LOR"],
            "DR": games["LDR"],
            "Ast": games["LAst"],
            "TO": games["LTO"],
            "Stl": games["LStl"],
            "Blk": games["LBlk"],
            "PF": games["LPF"],
            "OppFGM": games["WFGM"],
            "OppFGA": games["WFGA"],
            "OppFGM3": games["WFGM3"],
            "OppFGA3": games["WFGA3"],
            "OppFTM": games["WFTM"],
            "OppFTA": games["WFTA"],
            "OppOR": games["WOR"],
            "OppDR": games["WDR"],
            "OppAst": games["WAst"],
            "OppTO": games["WTO"],
            "OppStl": games["WStl"],
            "OppBlk": games["WBlk"],
            "OppPF": games["WPF"],
            "Win": 0,
        }
    )

    team_games = pd.concat([winners, losers], ignore_index=True)
    team_games["possessions"] = possessions(team_games["FGA"], team_games["OR"], team_games["TO"], team_games["FTA"])
    team_games["opp_possessions"] = possessions(
        team_games["OppFGA"], team_games["OppOR"], team_games["OppTO"], team_games["OppFTA"]
    )
    team_games["three_point_rate"] = team_games["FGA3"] / team_games["FGA"].replace(0, pd.NA)
    team_games["three_point_pct"] = team_games["FGM3"] / team_games["FGA3"].replace(0, pd.NA)
    team_games["opp_three_point_pct"] = team_games["OppFGM3"] / team_games["OppFGA3"].replace(0, pd.NA)
    team_games["three_point_scoring_share"] = (3 * team_games["FGM3"]) / team_games["PointsFor"].replace(0, pd.NA)

    season_stats = (
        team_games.groupby(["Season", "TeamID"], as_index=False)
        .agg(
            games=("Win", "size"),
            wins=("Win", "sum"),
            points_for=("PointsFor", "sum"),
            points_against=("PointsAgainst", "sum"),
            possessions=("possessions", "sum"),
            opp_possessions=("opp_possessions", "sum"),
            turnovers=("TO", "sum"),
            opp_turnovers=("OppTO", "sum"),
            three_point_rate=("three_point_rate", "mean"),
            three_point_pct=("three_point_pct", "mean"),
            three_point_pct_var=("three_point_pct", "var"),
            opp_three_point_pct_var=("opp_three_point_pct", "var"),
            three_point_dependency=("three_point_scoring_share", "mean"),
        )
    )
    season_stats["win_pct"] = season_stats["wins"] / season_stats["games"]
    season_stats["off_eff"] = 100 * season_stats["points_for"] / season_stats["possessions"]
    season_stats["def_eff"] = 100 * season_stats["points_against"] / season_stats["opp_possessions"]
    season_stats["eff_margin"] = season_stats["off_eff"] - season_stats["def_eff"]
    season_stats["tempo"] = season_stats["possessions"] / season_stats["games"].replace(0, pd.NA)
    season_stats["tov_rate"] = season_stats["turnovers"] / season_stats["possessions"]
    season_stats["opp_tov_rate"] = season_stats["opp_turnovers"] / season_stats["opp_possessions"]

    opp_eff = season_stats[["Season", "TeamID", "eff_margin"]].rename(
        columns={"TeamID": "OppTeamID", "eff_margin": "opp_eff_margin"}
    )
    schedule_strength = team_games[["Season", "TeamID", "OppTeamID"]].merge(
        opp_eff, on=["Season", "OppTeamID"], how="left"
    )
    schedule_strength = (
        schedule_strength.groupby(["Season", "TeamID"], as_index=False)["opp_eff_margin"]
        .mean()
        .rename(columns={"opp_eff_margin": "schedule_strength"})
    )
    season_stats = season_stats.merge(compact_features, on=["Season", "TeamID"], how="left")

    seeds = pd.read_csv(SEEDS_FILE)
    seeds["seed_num"] = seeds["Seed"].map(parse_seed)
    seeds = seeds[["Season", "TeamID", "Seed", "seed_num"]]

    pom = load_latest_massey("POM")

    team_stats = season_stats.merge(schedule_strength, on=["Season", "TeamID"], how="left")
    team_stats = team_stats.merge(teams, on="TeamID", how="left")
    team_stats = team_stats.merge(seeds, on=["Season", "TeamID"], how="left")
    team_stats = team_stats.merge(pom, on=["Season", "TeamID"], how="left")
    team_stats = team_stats.merge(build_conference_features(team_stats), on=["Season", "TeamID"], how="left")
    team_stats = team_stats.merge(build_city_features(), on=["Season", "TeamID"], how="left")
    team_stats = team_stats.merge(build_seed_history_features(), on="seed_num", how="left")

    team_stats["seed_num"] = team_stats.groupby("Season")["seed_num"].transform(lambda s: s.fillna(s.max() + 1))
    team_stats["pom_rank"] = team_stats.groupby("Season")["pom_rank"].transform(lambda s: s.fillna(s.median()))
    fill_defaults = {
        "avg_margin": 0.0,
        "close_game_win_pct": 0.5,
        "road_neutral_win_pct": 0.5,
        "recent10_win_pct": 0.5,
        "conference_strength": 0.0,
        "conference_tourney_win_pct": 0.0,
        "conference_champion": 0.0,
        "seed_history_score": 0.0,
        "unique_city_count": 0.0,
        "unique_state_count": 0.0,
        "three_point_rate": 0.0,
        "three_point_pct": 0.0,
        "three_point_pct_var": 0.0,
        "opp_three_point_pct_var": 0.0,
        "three_point_dependency": 0.0,
    }
    for column, value in fill_defaults.items():
        if column in team_stats.columns:
            team_stats[column] = pd.to_numeric(team_stats[column], errors="coerce").fillna(value)

    rating_columns = [
        "eff_margin",
        "off_eff",
        "def_eff",
        "pom_rank",
        "seed_num",
        "tov_rate",
        "schedule_strength",
        "conference_strength",
        "avg_margin",
        "three_point_rate",
        "three_point_pct",
        "three_point_pct_var",
        "opp_three_point_pct_var",
        "three_point_dependency",
        "unique_city_count",
    ]
    for column in rating_columns:
        mean_col = f"{column}_season_mean"
        std_col = f"{column}_season_std"
        team_stats[mean_col] = team_stats.groupby("Season")[column].transform("mean")
        team_stats[std_col] = team_stats.groupby("Season")[column].transform("std").replace(0, 1).fillna(1)
        team_stats[f"{column}_z"] = (team_stats[column] - team_stats[mean_col]) / team_stats[std_col]

    team_stats["close_game_win_pct_centered"] = team_stats["close_game_win_pct"] - 0.5
    team_stats["recent10_win_pct_centered"] = team_stats["recent10_win_pct"] - 0.5
    team_stats["road_neutral_win_pct_centered"] = team_stats["road_neutral_win_pct"] - 0.5
    team_stats["defense_quality_z"] = -team_stats["def_eff_z"]
    team_stats["resume_score"] = 0.80 * (-team_stats["pom_rank_z"]) + 0.20 * (-team_stats["seed_num_z"])
    team_stats["momentum_score"] = (
        0.85 * team_stats["recent10_win_pct_centered"] + 0.15 * team_stats["close_game_win_pct_centered"]
    )
    team_stats["strength_score"] = (
        0.45 * team_stats["eff_margin_z"]
        + 0.35 * team_stats["off_eff_z"]
        + 0.10 * team_stats["avg_margin_z"]
        + 0.10 * team_stats["resume_score"]
    )
    team_stats["defense_score"] = (
        0.65 * team_stats["defense_quality_z"] + 0.25 * team_stats["schedule_strength_z"]
        + 0.10 * (-team_stats["opp_three_point_pct_var_z"])
    )
    team_stats["variance_upside_score"] = (
        0.45 * team_stats["three_point_pct_z"]
        + 0.30 * team_stats["three_point_rate_z"]
        + 0.25 * team_stats["resume_score"]
    )
    team_stats["variance_risk_score"] = (
        0.45 * team_stats["three_point_pct_var_z"]
        + 0.35 * team_stats["three_point_dependency_z"]
        + 0.20 * team_stats["opp_three_point_pct_var_z"]
    )
    team_stats["stability_score"] = (
        0.60 * (-team_stats["tov_rate_z"])
        + 0.25 * team_stats["defense_quality_z"]
        + 0.15 * (-team_stats["three_point_pct_var_z"])
    )
    team_stats["context_score"] = (
        0.65 * team_stats["conference_strength_z"] + 0.35 * team_stats["road_neutral_win_pct_centered"]
    )
    team_stats["model_score"] = sum(weight * team_stats[column] for column, weight in TEAM_SCORE_COMPONENT_WEIGHTS.items())

    keep_columns = [
        "Season",
        "TeamID",
        "TeamName",
        "ConfAbbrev",
        "Description",
        "Seed",
        "seed_num",
        "win_pct",
        "off_eff",
        "def_eff",
        "eff_margin",
        "tempo",
        "tov_rate",
        "three_point_rate",
        "three_point_pct",
        "three_point_pct_var",
        "opp_three_point_pct_var",
        "three_point_dependency",
        "avg_margin",
        "close_game_win_pct",
        "road_neutral_win_pct",
        "recent10_win_pct",
        "schedule_strength",
        "conference_strength",
        "conference_tourney_win_pct",
        "conference_champion",
        "seed_history_score",
        "unique_city_count",
        "unique_state_count",
        "resume_score",
        "strength_score",
        "defense_score",
        "variance_upside_score",
        "variance_risk_score",
        "stability_score",
        "context_score",
        "momentum_score",
        "pom_rank",
        "model_score",
    ]
    return team_stats[keep_columns]


def compute_matchup_features_from_merged(merged: pd.DataFrame) -> pd.DataFrame:
    if "RoundNum" not in merged.columns:
        merged["RoundNum"] = 1
    if "SiteAdvantage" not in merged.columns:
        merged["SiteAdvantage"] = 0
    if "a_seed_num" not in merged.columns:
        merged["a_seed_num"] = 0.0
    if "b_seed_num" not in merged.columns:
        merged["b_seed_num"] = 0.0

    merged["eff_margin_diff"] = merged["a_eff_margin"] - merged["b_eff_margin"]
    merged["def_eff_edge"] = merged["b_def_eff"] - merged["a_def_eff"]
    merged["off_eff_diff"] = merged["a_off_eff"] - merged["b_off_eff"]
    merged["tov_rate_edge"] = merged["b_tov_rate"] - merged["a_tov_rate"]
    merged["avg_margin_diff"] = merged["a_avg_margin"] - merged["b_avg_margin"]
    merged["three_point_pct_diff"] = merged["a_three_point_pct"] - merged["b_three_point_pct"]
    merged["three_point_rate_diff"] = merged["a_three_point_rate"] - merged["b_three_point_rate"]
    merged["three_point_volatility_edge"] = merged["b_three_point_pct_var"] - merged["a_three_point_pct_var"]
    merged["opp_three_point_volatility_edge"] = (
        merged["b_opp_three_point_pct_var"] - merged["a_opp_three_point_pct_var"]
    )
    merged["momentum_edge"] = merged["a_momentum_score"] - merged["b_momentum_score"]
    merged["predictive_rank_edge"] = merged["b_pom_rank"] - merged["a_pom_rank"]
    merged["seed_edge"] = merged["b_seed_num"] - merged["a_seed_num"]
    merged["seed_strength_interaction"] = merged["seed_edge"] * merged["eff_margin_diff"]
    merged["a_offense_vs_b_defense"] = merged["a_off_eff"] - merged["b_def_eff"]
    merged["b_offense_vs_a_defense"] = merged["b_off_eff"] - merged["a_def_eff"]
    merged["tempo_diff"] = merged["a_tempo"] - merged["b_tempo"]
    merged["abs_tempo_diff"] = merged["tempo_diff"].abs()
    a_is_underdog = (merged["eff_margin_diff"] < 0).astype(float)
    b_is_underdog = (merged["eff_margin_diff"] > 0).astype(float)
    merged["underdog_three_point_rate"] = (
        a_is_underdog * merged["a_three_point_rate"] - b_is_underdog * merged["b_three_point_rate"]
    )
    merged["underdog_tempo_diff"] = (a_is_underdog - b_is_underdog) * merged["tempo_diff"]
    merged["underdog_a_offense_vs_b_defense"] = a_is_underdog * merged["a_offense_vs_b_defense"]
    merged["seed_a_offense_vs_b_defense"] = merged["seed_edge"] * merged["a_offense_vs_b_defense"]
    merged["site_advantage"] = merged["SiteAdvantage"].fillna(0.0)
    late_round_intensity = (merged["RoundNum"] - 1).clip(lower=0)
    merged["defense_round_pressure"] = merged["def_eff_edge"] * late_round_intensity
    return merged


def matchup_feature_frame(games: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
    a_stats = team_stats.add_prefix("a_")
    b_stats = team_stats.add_prefix("b_")
    merged = games.merge(
        a_stats,
        left_on=["Season", "TeamAID"],
        right_on=["a_Season", "a_TeamID"],
        how="left",
    ).merge(
        b_stats,
        left_on=["Season", "TeamBID"],
        right_on=["b_Season", "b_TeamID"],
        how="left",
    )
    return compute_matchup_features_from_merged(merged)


def build_regular_season_team_games() -> pd.DataFrame:
    games = pd.read_csv(REGULAR_SEASON_FILE)
    winner_site = np.select([games["WLoc"].eq("H"), games["WLoc"].eq("A")], [1, -1], default=0)
    loser_site = -winner_site

    winners = pd.DataFrame(
        {
            "Season": games["Season"],
            "DayNum": games["DayNum"],
            "TeamID": games["WTeamID"],
            "OppTeamID": games["LTeamID"],
            "PointsFor": games["WScore"],
            "PointsAgainst": games["LScore"],
            "FGM": games["WFGM"],
            "FGA": games["WFGA"],
            "FGM3": games["WFGM3"],
            "FGA3": games["WFGA3"],
            "FTM": games["WFTM"],
            "FTA": games["WFTA"],
            "OR": games["WOR"],
            "DR": games["WDR"],
            "Ast": games["WAst"],
            "TO": games["WTO"],
            "Stl": games["WStl"],
            "Blk": games["WBlk"],
            "PF": games["WPF"],
            "OppFGM": games["LFGM"],
            "OppFGA": games["LFGA"],
            "OppFGM3": games["LFGM3"],
            "OppFGA3": games["LFGA3"],
            "OppFTM": games["LFTM"],
            "OppFTA": games["LFTA"],
            "OppOR": games["LOR"],
            "OppDR": games["LDR"],
            "OppAst": games["LAst"],
            "OppTO": games["LTO"],
            "OppStl": games["LStl"],
            "OppBlk": games["LBlk"],
            "OppPF": games["LPF"],
            "Win": 1,
            "Loc": games["WLoc"],
            "SiteAdvantage": winner_site,
        }
    )
    losers = pd.DataFrame(
        {
            "Season": games["Season"],
            "DayNum": games["DayNum"],
            "TeamID": games["LTeamID"],
            "OppTeamID": games["WTeamID"],
            "PointsFor": games["LScore"],
            "PointsAgainst": games["WScore"],
            "FGM": games["LFGM"],
            "FGA": games["LFGA"],
            "FGM3": games["LFGM3"],
            "FGA3": games["LFGA3"],
            "FTM": games["LFTM"],
            "FTA": games["LFTA"],
            "OR": games["LOR"],
            "DR": games["LDR"],
            "Ast": games["LAst"],
            "TO": games["LTO"],
            "Stl": games["LStl"],
            "Blk": games["LBlk"],
            "PF": games["LPF"],
            "OppFGM": games["WFGM"],
            "OppFGA": games["WFGA"],
            "OppFGM3": games["WFGM3"],
            "OppFGA3": games["WFGA3"],
            "OppFTM": games["WFTM"],
            "OppFTA": games["WFTA"],
            "OppOR": games["WOR"],
            "OppDR": games["WDR"],
            "OppAst": games["WAst"],
            "OppTO": games["WTO"],
            "OppStl": games["WStl"],
            "OppBlk": games["WBlk"],
            "OppPF": games["WPF"],
            "Win": 0,
            "Loc": games["WLoc"].map(invert_location),
            "SiteAdvantage": loser_site,
        }
    )
    team_games = pd.concat([winners, losers], ignore_index=True)
    team_games["possessions"] = possessions(team_games["FGA"], team_games["OR"], team_games["TO"], team_games["FTA"])
    team_games["opp_possessions"] = possessions(
        team_games["OppFGA"], team_games["OppOR"], team_games["OppTO"], team_games["OppFTA"]
    )
    team_games["margin"] = team_games["PointsFor"] - team_games["PointsAgainst"]
    team_games["three_point_rate"] = team_games["FGA3"] / team_games["FGA"].replace(0, pd.NA)
    team_games["three_point_pct"] = team_games["FGM3"] / team_games["FGA3"].replace(0, pd.NA)
    team_games["opp_three_point_pct"] = team_games["OppFGM3"] / team_games["OppFGA3"].replace(0, pd.NA)
    team_games["away_or_neutral_game"] = team_games["Loc"].isin(["A", "N"]).astype(int)
    team_games["away_or_neutral_win"] = ((team_games["away_or_neutral_game"] == 1) & (team_games["Win"] == 1)).astype(int)
    team_games["close_game"] = team_games["margin"].abs().le(5).astype(int)
    team_games["close_game_win"] = ((team_games["close_game"] == 1) & (team_games["Win"] == 1)).astype(int)
    return team_games.sort_values(["Season", "TeamID", "DayNum", "OppTeamID"]).reset_index(drop=True)


def build_regular_season_team_snapshots() -> pd.DataFrame:
    team_games = build_regular_season_team_games()
    team_games["games_played_pre"] = team_games.groupby(["Season", "TeamID"]).cumcount()

    cumulative_columns = [
        "Win",
        "PointsFor",
        "PointsAgainst",
        "possessions",
        "opp_possessions",
        "TO",
        "FGA",
        "FGM3",
        "FGA3",
        "margin",
        "away_or_neutral_game",
        "away_or_neutral_win",
        "close_game",
        "close_game_win",
    ]
    for column in cumulative_columns:
        cumulative = team_games.groupby(["Season", "TeamID"])[column].cumsum()
        team_games[f"pre_{column}"] = cumulative - team_games[column]

    team_games["recent10_win_pct"] = (
        team_games.groupby(["Season", "TeamID"])["Win"]
        .transform(lambda s: s.shift().rolling(10, min_periods=1).mean())
        .fillna(0.5)
    )
    team_games["three_point_pct_var"] = (
        team_games.groupby(["Season", "TeamID"])["three_point_pct"]
        .transform(lambda s: s.shift().expanding(min_periods=5).var())
        .fillna(0.0)
    )
    team_games["opp_three_point_pct_var"] = (
        team_games.groupby(["Season", "TeamID"])["opp_three_point_pct"]
        .transform(lambda s: s.shift().expanding(min_periods=5).var())
        .fillna(0.0)
    )

    pre_games = team_games["pre_Win"].replace(0, pd.NA)
    pre_possessions = team_games["pre_possessions"].replace(0, pd.NA)
    pre_opp_possessions = team_games["pre_opp_possessions"].replace(0, pd.NA)
    pre_fga = team_games["pre_FGA"].replace(0, pd.NA)
    pre_fga3 = team_games["pre_FGA3"].replace(0, pd.NA)
    pre_away_neutral = team_games["pre_away_or_neutral_game"].replace(0, pd.NA)
    pre_close_games = team_games["pre_close_game"].replace(0, pd.NA)

    team_games["off_eff"] = 100 * team_games["pre_PointsFor"] / pre_possessions
    team_games["def_eff"] = 100 * team_games["pre_PointsAgainst"] / pre_opp_possessions
    team_games["eff_margin"] = team_games["off_eff"] - team_games["def_eff"]
    team_games["tempo"] = team_games["pre_possessions"] / pre_games
    team_games["tov_rate"] = team_games["pre_TO"] / pre_possessions
    team_games["avg_margin"] = team_games["pre_margin"] / pre_games
    team_games["three_point_rate"] = team_games["pre_FGA3"] / pre_fga
    team_games["three_point_pct"] = team_games["pre_FGM3"] / pre_fga3
    team_games["road_neutral_win_pct"] = team_games["pre_away_or_neutral_win"] / pre_away_neutral
    team_games["close_game_win_pct"] = team_games["pre_close_game_win"] / pre_close_games
    close_game_component = pd.to_numeric(team_games["close_game_win_pct"], errors="coerce").fillna(0.5)
    team_games["momentum_score"] = 0.85 * (team_games["recent10_win_pct"] - 0.5) + 0.15 * (close_game_component - 0.5)

    pom = pd.read_csv(MASSEY_FILE)
    pom = pom[pom["SystemName"] == "POM"][["Season", "RankingDayNum", "TeamID", "OrdinalRank"]].copy()
    merged_frames: list[pd.DataFrame] = []
    for season, season_games in team_games.groupby("Season", sort=True):
        season_games = season_games.sort_values(["DayNum", "TeamID"]).copy()
        season_pom = pom[pom["Season"] == season][["RankingDayNum", "TeamID", "OrdinalRank"]].sort_values(
            ["RankingDayNum", "TeamID"]
        ).copy()
        season_merged = pd.merge_asof(
            season_games,
            season_pom,
            left_on="DayNum",
            right_on="RankingDayNum",
            by="TeamID",
            direction="backward",
            allow_exact_matches=False,
        )
        merged_frames.append(season_merged)
    team_games = pd.concat(merged_frames, ignore_index=True).rename(columns={"OrdinalRank": "pom_rank"})

    fill_defaults = {
        "off_eff": 0.0,
        "def_eff": 0.0,
        "eff_margin": 0.0,
        "tempo": 0.0,
        "tov_rate": 0.0,
        "avg_margin": 0.0,
        "three_point_rate": 0.0,
        "three_point_pct": 0.0,
        "three_point_pct_var": 0.0,
        "opp_three_point_pct_var": 0.0,
        "road_neutral_win_pct": 0.5,
        "close_game_win_pct": 0.5,
        "recent10_win_pct": 0.5,
        "momentum_score": 0.0,
    }
    for column, value in fill_defaults.items():
        team_games[column] = pd.to_numeric(team_games[column], errors="coerce").fillna(value)
    team_games["pom_rank"] = pd.to_numeric(team_games["pom_rank"], errors="coerce")
    team_games["pom_rank"] = team_games.groupby("Season")["pom_rank"].transform(lambda s: s.fillna(s.median()))
    return team_games


def build_regular_season_training_data() -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    snapshots = build_regular_season_team_snapshots()
    games = pd.read_csv(REGULAR_SEASON_FILE, usecols=["Season", "DayNum", "WTeamID", "LTeamID", "WLoc"])

    winners = games.rename(columns={"WTeamID": "TeamAID", "LTeamID": "TeamBID"})[["Season", "DayNum", "TeamAID", "TeamBID"]]
    winners["target"] = 1
    winners["SiteAdvantage"] = np.select([games["WLoc"].eq("H"), games["WLoc"].eq("A")], [1, -1], default=0)
    winners["RoundNum"] = 1

    losers = games.rename(columns={"LTeamID": "TeamAID", "WTeamID": "TeamBID"})[["Season", "DayNum", "TeamAID", "TeamBID"]]
    losers["target"] = 0
    losers["SiteAdvantage"] = -winners["SiteAdvantage"]
    losers["RoundNum"] = 1

    matchup_rows = pd.concat([winners, losers], ignore_index=True)
    a_snapshots = snapshots.add_prefix("a_")
    b_snapshots = snapshots.add_prefix("b_")
    merged = matchup_rows.merge(
        a_snapshots,
        left_on=["Season", "DayNum", "TeamAID"],
        right_on=["a_Season", "a_DayNum", "a_TeamID"],
        how="left",
    ).merge(
        b_snapshots,
        left_on=["Season", "DayNum", "TeamBID"],
        right_on=["b_Season", "b_DayNum", "b_TeamID"],
        how="left",
    )
    merged = merged[
        (merged["a_games_played_pre"] >= MIN_PREGAME_GAMES) & (merged["b_games_played_pre"] >= MIN_PREGAME_GAMES)
    ].copy()
    features = compute_matchup_features_from_merged(merged)
    return (
        features[ALL_FEATURE_COLUMNS],
        features["target"].astype(int),
        features["Season"].astype(int),
        pd.Series("regular_season", index=features.index),
    )


def build_training_data(team_stats: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    round_lookup = get_round_lookup()
    tourney = pd.read_csv(TOURNEY_FILE, usecols=["Season", "DayNum", "WTeamID", "LTeamID"])
    tourney["RoundNum"] = tourney["DayNum"].map(lambda day_num: map_day_to_round(day_num, round_lookup))
    winners = tourney.rename(columns={"WTeamID": "TeamAID", "LTeamID": "TeamBID"})
    winners["target"] = 1
    losers = tourney.rename(columns={"LTeamID": "TeamAID", "WTeamID": "TeamBID"})
    losers["target"] = 0
    games = pd.concat([winners, losers], ignore_index=True)
    features = matchup_feature_frame(games[["Season", "TeamAID", "TeamBID", "RoundNum", "target"]], team_stats)
    return (
        features[ALL_FEATURE_COLUMNS],
        features["target"].astype(int),
        features["Season"].astype(int),
        pd.Series("tournament", index=features.index),
    )


def fit_base_model(X: pd.DataFrame, y: pd.Series, seed_shrinkage: float = 1.0, c_value: float = 0.2) -> Pipeline:
    feature_columns = list(X.columns)
    other_columns = [column for column in feature_columns if column != "seed_edge"]
    transformers = [
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            other_columns,
        )
    ]
    if "seed_edge" in feature_columns:
        transformers.append(
            (
                "seed",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        (
                            "shrink",
                            FunctionTransformer(
                                lambda values, factor=seed_shrinkage: values * factor,
                                validate=False,
                            ),
                        ),
                    ]
                ),
                ["seed_edge"],
            )
        )
    preprocessor = ColumnTransformer(
        transformers=transformers
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=2500,
                    random_state=42,
                ),
            ),
        ]
    ).fit(X, y)


def build_boosting_array_cache(X: pd.DataFrame, y: pd.Series) -> BoostingArrayCache:
    feature_columns = list(X.columns)
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                feature_columns,
            )
        ]
    )
    X_arr = preprocessor.fit_transform(X[feature_columns])
    X_arr = X_arr.toarray() if hasattr(X_arr, "toarray") else np.asarray(X_arr)
    y_arr = np.asarray(y, dtype=float)
    return BoostingArrayCache(
        X_array=X_arr,
        y_array=y_arr,
        feature_order=feature_columns,
        preprocessor=preprocessor,
    )


def compute_train_val_indices(y_array: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    all_idx = np.arange(len(y_array))
    train_idx, val_idx = train_test_split(
        all_idx,
        test_size=0.15,
        random_state=seed,
        stratify=y_array.astype(int),
    )
    return np.asarray(train_idx), np.asarray(val_idx)


def fit_boosting_model_from_array(
    cache: BoostingArrayCache,
    seed: int = 42,
    train_idx: np.ndarray | None = None,
    val_idx: np.ndarray | None = None,
) -> TreeBoostingModel:
    X_arr = cache.X_array
    y_arr = cache.y_array

    if train_idx is None or val_idx is None:
        train_idx, val_idx = compute_train_val_indices(y_arr, seed)

    X_train = X_arr[train_idx]
    X_val = X_arr[val_idx]
    y_train = y_arr[train_idx]
    y_val = y_arr[val_idx]

    base_rate = float(np.clip(y_train.mean(), 1e-6, 1 - 1e-6))
    init_score = float(np.log(base_rate / (1 - base_rate)))
    learning_rate = 0.05
    max_estimators = 225
    max_depth = 3
    subsample = 0.7
    colsample = 0.8
    patience = 20
    tol = 1e-4
    min_samples_leaf = 20
    rng = np.random.default_rng(seed)

    raw_train = np.full(len(y_train), init_score, dtype=float)
    raw_val = np.full(len(y_val), init_score, dtype=float)
    trees: list[tuple[DecisionTreeRegressor, np.ndarray]] = []
    feature_importance = np.zeros(X_arr.shape[1], dtype=float)
    best_val_loss = log_loss(y_val, logits_to_probabilities(raw_val), labels=[0, 1])
    best_tree_count = 0
    rounds_without_improvement = 0

    for iteration in range(max_estimators):
        residual = y_train - logits_to_probabilities(raw_train)
        row_idx = rng.choice(len(y_train), size=max(50, int(len(y_train) * subsample)), replace=False)
        feature_idx = np.sort(
            rng.choice(X_arr.shape[1], size=max(1, int(np.ceil(X_arr.shape[1] * colsample))), replace=False)
        )
        tree = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=seed + iteration,
        )
        tree.fit(X_train[row_idx][:, feature_idx], residual[row_idx])
        raw_train += learning_rate * tree.predict(X_train[:, feature_idx])
        raw_val += learning_rate * tree.predict(X_val[:, feature_idx])
        trees.append((tree, feature_idx))
        feature_importance[feature_idx] += np.abs(tree.feature_importances_)

        current_val_loss = log_loss(y_val, logits_to_probabilities(raw_val), labels=[0, 1])
        if current_val_loss + tol < best_val_loss:
            best_val_loss = current_val_loss
            best_tree_count = len(trees)
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1
            if rounds_without_improvement >= patience:
                break

    kept_trees = trees[:best_tree_count] if best_tree_count else trees
    if best_tree_count and best_tree_count < len(trees):
        feature_importance = np.zeros(X_arr.shape[1], dtype=float)
        for tree, feature_idx in kept_trees:
            feature_importance[feature_idx] += np.abs(tree.feature_importances_)

    return TreeBoostingModel(
        preprocessor=cache.preprocessor,
        trees=kept_trees,
        feature_order=cache.feature_order,
        init_score=init_score,
        learning_rate=learning_rate,
        feature_importance_=feature_importance,
    )


def fit_boosting_model(X: pd.DataFrame, y: pd.Series, seed: int = 42) -> TreeBoostingModel:
    cache = build_boosting_array_cache(X, y)
    return fit_boosting_model_from_array(cache, seed=seed)


def base_model_raw_score(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return model.decision_function(X)


def clip_probabilities(probabilities: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(probabilities, dtype=float), eps, 1 - eps)


def logits_to_probabilities(logits: np.ndarray) -> np.ndarray:
    logits_arr = np.asarray(logits, dtype=float)
    return clip_probabilities(1.0 / (1.0 + np.exp(-logits_arr)))


def expected_calibration_error(y_true: pd.Series, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    bucket_ids = np.digitize(y_prob_arr, bins[1:-1], right=True)
    ece = 0.0
    for bucket in range(n_bins):
        mask = bucket_ids == bucket
        if not mask.any():
            continue
        avg_conf = y_prob_arr[mask].mean()
        avg_acc = y_true_arr[mask].mean()
        ece += mask.mean() * abs(avg_conf - avg_acc)
    return float(ece)


def build_reliability_curve(y_true: pd.Series, y_prob: np.ndarray, label: str, n_bins: int = 10) -> pd.DataFrame:
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    return pd.DataFrame(
        {
            "label": label,
            "mean_predicted_probability": mean_pred,
            "fraction_positives": frac_pos,
        }
    )


def standardize_and_clip_scores(
    raw_scores: np.ndarray,
    mean_score: float,
    std_score: float,
    clip_value: float,
) -> np.ndarray:
    std = std_score if std_score > 1e-8 else 1.0
    standardized = (np.asarray(raw_scores, dtype=float) - mean_score) / std
    return np.clip(standardized, -clip_value, clip_value)


def fit_platt_model(raw_scores: np.ndarray, y_true: pd.Series, c_value: float, clip_value: float) -> dict[str, object]:
    mean_score = float(np.mean(raw_scores))
    std_score = float(np.std(raw_scores))
    transformed_scores = standardize_and_clip_scores(raw_scores, mean_score, std_score, clip_value).reshape(-1, 1)
    model = LogisticRegression(C=c_value, random_state=42, max_iter=2000)
    model.fit(transformed_scores, y_true)
    return {
        "mean_score": mean_score,
        "std_score": std_score,
        "clip_value": clip_value,
        "c_value": c_value,
        "model": model,
    }


def predict_platt_model(platt_model: dict[str, object], raw_scores: np.ndarray) -> np.ndarray:
    transformed_scores = standardize_and_clip_scores(
        raw_scores,
        float(platt_model["mean_score"]),
        float(platt_model["std_score"]),
        float(platt_model["clip_value"]),
    ).reshape(-1, 1)
    return clip_probabilities(platt_model["model"].predict_proba(transformed_scores)[:, 1])


def fit_ensemble_platt_models(
    raw_scores: np.ndarray,
    y_true: pd.Series,
    season_labels: pd.Series,
    c_value: float,
    clip_value: float,
) -> list[dict[str, object]]:
    models: list[dict[str, object]] = []
    for season in sorted(pd.Series(season_labels).unique()):
        season_mask = season_labels == season
        if season_mask.sum() < 10:
            continue
        models.append(fit_platt_model(raw_scores[season_mask], y_true.loc[season_mask], c_value, clip_value))
    if not models:
        models.append(fit_platt_model(raw_scores, y_true, c_value, clip_value))
    return models


def predict_ensemble_platt(models: list[dict[str, object]], raw_scores: np.ndarray) -> np.ndarray:
    ensemble_probs = np.column_stack([predict_platt_model(model, raw_scores) for model in models])
    return clip_probabilities(ensemble_probs.mean(axis=1))


def blend_probabilities(raw_probabilities: np.ndarray, calibrated_probabilities: np.ndarray, blend_weight: float) -> np.ndarray:
    return clip_probabilities((1 - blend_weight) * raw_probabilities + blend_weight * calibrated_probabilities)


def fit_temperature_scaler(raw_scores: np.ndarray, y_true: pd.Series) -> dict[str, float]:
    best_temperature = 1.0
    best_log_loss = float("inf")
    for temperature in TEMPERATURE_GRID:
        scaled_prob = logits_to_probabilities(np.asarray(raw_scores, dtype=float) / temperature)
        current_log_loss = log_loss(y_true, scaled_prob, labels=[0, 1])
        if current_log_loss < best_log_loss:
            best_log_loss = current_log_loss
            best_temperature = float(temperature)
    return {"temperature": best_temperature, "log_loss": best_log_loss}


def predict_temperature_scaled_proba(raw_scores: np.ndarray, temperature: float) -> np.ndarray:
    return logits_to_probabilities(np.asarray(raw_scores, dtype=float) / temperature)


def apply_medium_gap_probability_adjustment(features: pd.DataFrame, probabilities: np.ndarray, lift: float) -> np.ndarray:
    adjusted = np.asarray(probabilities, dtype=float).copy()
    if lift == 0.0 or "seed_edge" not in features.columns:
        return clip_probabilities(adjusted)
    seed_gap_abs = pd.Series(features["seed_edge"]).abs().to_numpy(dtype=float)
    medium_gap_mask = (seed_gap_abs >= 3.0) & (seed_gap_abs <= 5.0)
    adjusted[medium_gap_mask] = adjusted[medium_gap_mask] - lift
    return clip_probabilities(adjusted)


def predict_calibrated_proba(
    base_model: Pipeline,
    calibration_bundle: CalibrationBundle | None,
    X: pd.DataFrame,
    method: str = "raw",
) -> np.ndarray:
    raw_scores = base_model_raw_score(base_model, X)
    raw_probabilities = clip_probabilities(base_model.predict_proba(X)[:, 1])
    if method == "raw" or calibration_bundle is None:
        return raw_probabilities
    if method == "baseline_platt":
        return predict_platt_model(calibration_bundle.calibrator, raw_scores)
    if method == "regularized_platt":
        return predict_platt_model(calibration_bundle.calibrator, raw_scores)
    if method == "ensemble_platt":
        return predict_ensemble_platt(calibration_bundle.calibrator, raw_scores)
    if method == "hybrid_platt":
        ensemble_type = calibration_bundle.params.get("ensemble_type", "single")
        if ensemble_type == "ensemble":
            calibrated_probabilities = predict_ensemble_platt(calibration_bundle.calibrator, raw_scores)
        else:
            calibrated_probabilities = predict_platt_model(calibration_bundle.calibrator, raw_scores)
        return blend_probabilities(raw_probabilities, calibrated_probabilities, float(calibration_bundle.params["blend_weight"]))
    raise ValueError(f"Unknown prediction method: {method}")


def evaluate_probability_predictions(
    y_true: pd.Series,
    y_prob: np.ndarray,
    label: str,
) -> dict[str, float | str]:
    return {
        "label": label,
        "log_loss": log_loss(y_true, y_prob, labels=[0, 1]),
        "brier_score": brier_score_loss(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
    }


def build_matchup_feature_weights_report(model: Pipeline) -> pd.DataFrame:
    coefficients = model.named_steps["model"].coef_[0]
    feature_order: list[str] = []
    for _, _, columns in model.named_steps["preprocessor"].transformers_:
        if columns == "drop":
            continue
        if isinstance(columns, str):
            feature_order.append(columns)
        else:
            feature_order.extend(list(columns))
    weights = pd.DataFrame({"metric": feature_order, "coefficient": coefficients})
    weights["abs_coefficient"] = weights["coefficient"].abs()
    total_abs = weights["abs_coefficient"].sum()
    weights["normalized_weight_pct"] = 100 * weights["abs_coefficient"] / total_abs if total_abs else 0.0
    weights["direction"] = weights["coefficient"].apply(
        lambda value: "team A advantage" if value > 0 else "team B advantage"
    )
    return weights.sort_values("abs_coefficient", ascending=False)


def build_boosting_feature_importance_report(model: TreeBoostingModel) -> pd.DataFrame:
    importance_df = pd.DataFrame({"metric": model.feature_order, "importance": model.feature_importance_})
    importance_df["abs_importance"] = importance_df["importance"].abs()
    total_abs = importance_df["abs_importance"].sum()
    importance_df["normalized_importance_pct"] = (
        100 * importance_df["abs_importance"] / total_abs if total_abs else 0.0
    )
    return importance_df.sort_values("importance", ascending=False).reset_index(drop=True)


def build_model_comparison_report(calibration_metrics: pd.DataFrame) -> pd.DataFrame:
    raw_row = calibration_metrics[calibration_metrics["label"] == "raw"].sort_values(
        ["log_loss", "brier_score", "ece"]
    ).iloc[0]
    hybrid_rows = calibration_metrics[calibration_metrics["label"] == "hybrid_platt"].sort_values(
        ["log_loss", "brier_score", "ece"]
    )
    hybrid_row = hybrid_rows.iloc[0] if not hybrid_rows.empty else raw_row
    return pd.DataFrame(
        [
            {
                "model_variant": "current_hybrid_platt",
                "log_loss": hybrid_row["log_loss"],
                "brier_score": hybrid_row["brier_score"],
                "ece": hybrid_row["ece"],
                "c_value": hybrid_row["c_value"],
                "blend_weight": hybrid_row["blend_weight"],
                "ensemble_type": hybrid_row["ensemble_type"],
            },
            {
                "model_variant": "direct_logistic_regression",
                "log_loss": raw_row["log_loss"],
                "brier_score": raw_row["brier_score"],
                "ece": raw_row["ece"],
                "c_value": 0.0,
                "blend_weight": 0.0,
                "ensemble_type": "raw",
            },
        ]
    )


def build_temperature_scaling_report(
    X: pd.DataFrame,
    y: pd.Series,
    seasons: pd.Series,
    source_labels: pd.Series,
    prediction_season: int,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    available_seasons = sorted(int(season) for season in pd.Series(seasons[source_labels == "tournament"]).unique())
    fold_test_seasons = [season for season in available_seasons if prediction_season - 5 <= season <= prediction_season - 1]

    metrics_rows: list[dict[str, float | str]] = []
    reliability_frames: list[pd.DataFrame] = []
    distribution_frames: list[pd.DataFrame] = []

    for fold_test_season in fold_test_seasons:
        calibration_seasons = [season for season in [fold_test_season - 2, fold_test_season - 1] if season in available_seasons]
        if len(calibration_seasons) < 2:
            continue
        train_mask = ((source_labels == "tournament") & (seasons < min(calibration_seasons))) | (
            (source_labels == "regular_season") & (seasons <= fold_test_season)
        )
        calibration_mask = (source_labels == "tournament") & seasons.isin(calibration_seasons)
        fold_test_mask = (source_labels == "tournament") & (seasons == fold_test_season)
        if train_mask.sum() == 0 or calibration_mask.sum() == 0 or fold_test_mask.sum() == 0:
            continue

        fold_model = fit_base_model(X.loc[train_mask], y.loc[train_mask], seed_shrinkage=1.0)
        raw_calibration_scores = base_model_raw_score(fold_model, X.loc[calibration_mask])
        raw_test_scores = base_model_raw_score(fold_model, X.loc[fold_test_mask])
        raw_test_prob = logits_to_probabilities(raw_test_scores)
        temperature_fit = fit_temperature_scaler(raw_calibration_scores, y.loc[calibration_mask])
        temperature = float(temperature_fit["temperature"])
        temp_test_prob = predict_temperature_scaled_proba(raw_test_scores, temperature)

        metrics_rows.append(
            {
                "fold_test_season": fold_test_season,
                **evaluate_probability_predictions(y.loc[fold_test_mask], raw_test_prob, "raw_logistic"),
                "temperature": 1.0,
            }
        )
        metrics_rows.append(
            {
                "fold_test_season": fold_test_season,
                **evaluate_probability_predictions(y.loc[fold_test_mask], temp_test_prob, "temperature_scaled"),
                "temperature": temperature,
            }
        )
        reliability_frames.append(build_reliability_curve(y.loc[fold_test_mask], raw_test_prob, "raw_logistic"))
        reliability_frames.append(build_reliability_curve(y.loc[fold_test_mask], temp_test_prob, "temperature_scaled"))

        for label, probs in [("raw_logistic", raw_test_prob), ("temperature_scaled", temp_test_prob)]:
            distribution_frames.append(
                pd.DataFrame(
                    {
                        "label": label,
                        "fold_test_season": fold_test_season,
                        "probability": probs,
                        "bucket": pd.cut(
                            probs,
                            bins=[0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0],
                            include_lowest=True,
                        ).astype(str),
                    }
                )
            )

    metrics_df = pd.DataFrame(metrics_rows)
    summary_df = (
        metrics_df.groupby("label", dropna=False)[["log_loss", "brier_score", "ece", "temperature"]]
        .mean()
        .reset_index()
        .sort_values(["log_loss", "brier_score", "ece"])
    )
    final_train_mask = ((source_labels == "tournament") & (seasons < prediction_season)) | (
        (source_labels == "regular_season") & (seasons <= prediction_season)
    )
    final_calibration_mask = (source_labels == "tournament") & seasons.isin([prediction_season - 3, prediction_season - 2])
    final_model = fit_base_model(X.loc[final_train_mask], y.loc[final_train_mask], seed_shrinkage=1.0)
    final_raw_scores = base_model_raw_score(final_model, X.loc[final_calibration_mask])
    final_temperature_fit = fit_temperature_scaler(final_raw_scores, y.loc[final_calibration_mask])

    distribution_df = pd.concat(distribution_frames, ignore_index=True)
    distribution_summary = (
        distribution_df.groupby(["label", "bucket"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["label", "bucket"])
    )
    extreme_summary = distribution_df.groupby("label")["probability"].agg(
        mean_probability="mean",
        median_probability="median",
        pct_above_095=lambda s: float((s > 0.95).mean()),
        pct_below_005=lambda s: float((s < 0.05).mean()),
    ).reset_index()
    distribution_summary = distribution_summary.merge(extreme_summary, on="label", how="left")
    reliability_df = pd.concat(reliability_frames, ignore_index=True)
    return final_temperature_fit, summary_df, reliability_df, distribution_summary


def build_ensemble_report(
    X: pd.DataFrame,
    y: pd.Series,
    seasons: pd.Series,
    source_labels: pd.Series,
    prediction_season: int,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    available_seasons = sorted(int(season) for season in pd.Series(seasons[source_labels == "tournament"]).unique())
    fold_test_seasons = [season for season in available_seasons if prediction_season - 5 <= season <= prediction_season - 1]

    metrics_rows: list[dict[str, float | str]] = []
    distribution_frames: list[pd.DataFrame] = []
    reliability_curves: list[pd.DataFrame] = []

    for fold_test_season in fold_test_seasons:
        calibration_seasons = [season for season in [fold_test_season - 2, fold_test_season - 1] if season in available_seasons]
        if len(calibration_seasons) < 2:
            continue
        train_mask = ((source_labels == "tournament") & (seasons < min(calibration_seasons))) | (
            (source_labels == "regular_season") & (seasons <= fold_test_season)
        )
        calibration_mask = (source_labels == "tournament") & seasons.isin(calibration_seasons)
        fold_test_mask = (source_labels == "tournament") & (seasons == fold_test_season)
        if train_mask.sum() == 0 or calibration_mask.sum() == 0 or fold_test_mask.sum() == 0:
            continue

        fold_model = fit_base_model(X.loc[train_mask], y.loc[train_mask], seed_shrinkage=1.0)
        raw_calibration_scores = base_model_raw_score(fold_model, X.loc[calibration_mask])
        raw_test_scores = base_model_raw_score(fold_model, X.loc[fold_test_mask])

        temperature_fit = fit_temperature_scaler(raw_calibration_scores, y.loc[calibration_mask])
        logistic_prob = predict_temperature_scaled_proba(raw_test_scores, float(temperature_fit["temperature"]))

        regularized_platt = fit_platt_model(raw_calibration_scores, y.loc[calibration_mask], c_value=1.0, clip_value=PLATT_CLIP_VALUE)
        hybrid_calibrated = predict_platt_model(regularized_platt, raw_test_scores)
        hybrid_raw = logits_to_probabilities(raw_test_scores)
        hybrid_prob = blend_probabilities(hybrid_raw, hybrid_calibrated, 0.3)

        for alpha in ENSEMBLE_ALPHA_GRID:
            ensemble_prob = clip_probabilities(alpha * logistic_prob + (1 - alpha) * hybrid_prob)
            label = f"ensemble_alpha_{alpha:.2f}"
            metrics_rows.append(
                {
                    "fold_test_season": fold_test_season,
                    "alpha": alpha,
                    **evaluate_probability_predictions(y.loc[fold_test_mask], ensemble_prob, label),
                }
            )
            if alpha in (0.0, 0.5, 1.0):
                reliability_curves.append(build_reliability_curve(y.loc[fold_test_mask], ensemble_prob, label))
            distribution_frames.append(
                pd.DataFrame(
                    {
                        "label": label,
                        "alpha": alpha,
                        "fold_test_season": fold_test_season,
                        "probability": ensemble_prob,
                        "bucket": pd.cut(
                            ensemble_prob,
                            bins=[0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0],
                            include_lowest=True,
                        ).astype(str),
                    }
                )
            )

    metrics_df = pd.DataFrame(metrics_rows)
    summary_df = (
        metrics_df.groupby(["alpha", "label"], dropna=False)[["log_loss", "brier_score", "ece"]]
        .mean()
        .reset_index()
        .sort_values(["log_loss", "brier_score", "ece"])
    )
    best_row = summary_df.iloc[0]
    final_train_mask = ((source_labels == "tournament") & (seasons < prediction_season)) | (
        (source_labels == "regular_season") & (seasons <= prediction_season)
    )
    final_calibration_mask = (source_labels == "tournament") & seasons.isin([prediction_season - 3, prediction_season - 2])
    final_model = fit_base_model(X.loc[final_train_mask], y.loc[final_train_mask], seed_shrinkage=1.0)
    final_raw_scores = base_model_raw_score(final_model, X.loc[final_calibration_mask])
    final_temperature_fit = fit_temperature_scaler(final_raw_scores, y.loc[final_calibration_mask])

    distribution_df = pd.concat(distribution_frames, ignore_index=True)
    distribution_summary = (
        distribution_df.groupby(["alpha", "label", "bucket"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["alpha", "bucket"])
    )
    extreme_summary = distribution_df.groupby(["alpha", "label"])["probability"].agg(
        pct_above_095=lambda s: float((s > 0.95).mean()),
        pct_between_040_070=lambda s: float(((s >= 0.4) & (s <= 0.7)).mean()),
    ).reset_index()
    distribution_summary = distribution_summary.merge(extreme_summary, on=["alpha", "label"], how="left")
    reliability_df = pd.concat(reliability_curves, ignore_index=True)
    return (
        {"alpha": float(best_row["alpha"]), "temperature": float(final_temperature_fit["temperature"])},
        summary_df,
        reliability_df,
        distribution_summary,
    )


def build_boosting_benchmark_report(
    X: pd.DataFrame,
    y: pd.Series,
    seasons: pd.Series,
    source_labels: pd.Series,
    prediction_season: int,
) -> tuple[TreeBoostingModel, CalibrationBundle, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    available_seasons = sorted(int(season) for season in pd.Series(seasons[source_labels == "tournament"]).unique())
    fold_test_seasons = [season for season in available_seasons if prediction_season - 5 <= season <= prediction_season - 1]

    metrics_rows: list[dict[str, float | str]] = []
    reliability_frames: list[pd.DataFrame] = []
    distribution_frames: list[pd.DataFrame] = []

    for fold_test_season in fold_test_seasons:
        calibration_seasons = [season for season in [fold_test_season - 2, fold_test_season - 1] if season in available_seasons]
        if len(calibration_seasons) < 2:
            continue
        train_mask = ((source_labels == "tournament") & (seasons < min(calibration_seasons))) | (
            (source_labels == "regular_season") & (seasons <= fold_test_season)
        )
        calibration_mask = (source_labels == "tournament") & seasons.isin(calibration_seasons)
        fold_test_mask = (source_labels == "tournament") & (seasons == fold_test_season)
        if train_mask.sum() == 0 or calibration_mask.sum() == 0 or fold_test_mask.sum() == 0:
            continue

        logistic_model = fit_base_model(X.loc[train_mask], y.loc[train_mask], seed_shrinkage=1.0, c_value=0.2)
        logistic_prob = clip_probabilities(logistic_model.predict_proba(X.loc[fold_test_mask])[:, 1])

        fold_cache = build_boosting_array_cache(X.loc[train_mask], y.loc[train_mask])
        fold_train_idx, fold_val_idx = compute_train_val_indices(fold_cache.y_array, 42)
        boosting_model = fit_boosting_model_from_array(fold_cache, seed=42, train_idx=fold_train_idx, val_idx=fold_val_idx)
        boosting_raw_scores = base_model_raw_score(boosting_model, X.loc[fold_test_mask])
        boosting_raw_prob = clip_probabilities(boosting_model.predict_proba(X.loc[fold_test_mask])[:, 1])

        boosting_calibration_scores = base_model_raw_score(boosting_model, X.loc[calibration_mask])
        boosting_platt = fit_platt_model(
            boosting_calibration_scores,
            y.loc[calibration_mask],
            c_value=1.0,
            clip_value=PLATT_CLIP_VALUE,
        )
        boosting_platt_prob = predict_platt_model(boosting_platt, boosting_raw_scores)

        for label, probs in [
            ("logistic_baseline", logistic_prob),
            ("boosting_raw", boosting_raw_prob),
            ("boosting_platt", boosting_platt_prob),
        ]:
            metrics_rows.append(
                {
                    "fold_test_season": fold_test_season,
                    **evaluate_probability_predictions(y.loc[fold_test_mask], probs, label),
                }
            )
            reliability_frames.append(build_reliability_curve(y.loc[fold_test_mask], probs, label))
            distribution_frames.append(
                pd.DataFrame(
                    {
                        "label": label,
                        "fold_test_season": fold_test_season,
                        "probability": probs,
                        "bucket": pd.cut(
                            probs,
                            bins=[0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0],
                            include_lowest=True,
                        ).astype(str),
                    }
                )
            )

    metrics_df = pd.DataFrame(metrics_rows)
    summary_df = (
        metrics_df.groupby("label", dropna=False)[["log_loss", "brier_score", "ece"]]
        .mean()
        .reset_index()
        .sort_values(["log_loss", "brier_score", "ece"])
    )

    distribution_df = pd.concat(distribution_frames, ignore_index=True)
    distribution_summary = (
        distribution_df.groupby(["label", "bucket"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["label", "bucket"])
    )
    extreme_summary = distribution_df.groupby("label")["probability"].agg(
        mean_probability="mean",
        median_probability="median",
        pct_above_095=lambda s: float((s > 0.95).mean()),
        pct_between_040_070=lambda s: float(((s >= 0.4) & (s <= 0.7)).mean()),
    ).reset_index()
    distribution_summary = distribution_summary.merge(extreme_summary, on="label", how="left")
    reliability_df = pd.concat(reliability_frames, ignore_index=True)

    final_train_mask = ((source_labels == "tournament") & (seasons < prediction_season)) | (
        (source_labels == "regular_season") & (seasons <= prediction_season)
    )
    final_calibration_mask = (source_labels == "tournament") & seasons.isin([prediction_season - 3, prediction_season - 2])
    final_cache = build_boosting_array_cache(X.loc[final_train_mask], y.loc[final_train_mask])
    final_train_idx, final_val_idx = compute_train_val_indices(final_cache.y_array, 42)
    final_boosting_model = fit_boosting_model_from_array(final_cache, seed=42, train_idx=final_train_idx, val_idx=final_val_idx)
    final_boosting_scores = base_model_raw_score(final_boosting_model, X.loc[final_calibration_mask])
    final_platt = fit_platt_model(
        final_boosting_scores,
        y.loc[final_calibration_mask],
        c_value=1.0,
        clip_value=PLATT_CLIP_VALUE,
    )
    calibration_bundle = CalibrationBundle(
        method="baseline_platt",
        calibrator=final_platt,
        params={"c_value": 1.0, "blend_weight": 1.0, "clip_value": PLATT_CLIP_VALUE, "ensemble_type": "single"},
    )
    return final_boosting_model, calibration_bundle, summary_df, reliability_df, distribution_summary


def build_boosting_stability_report(
    X: pd.DataFrame,
    y: pd.Series,
    seasons: pd.Series,
    source_labels: pd.Series,
    prediction_season: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    available_seasons = sorted(int(season) for season in pd.Series(seasons[source_labels == "tournament"]).unique())
    fold_test_seasons = [season for season in available_seasons if prediction_season - 5 <= season <= prediction_season - 1]
    X_np = X.to_numpy(dtype=float, copy=False)
    y_np = np.asarray(y, dtype=float)
    seasons_np = np.asarray(seasons, dtype=int)
    source_np = np.asarray(source_labels)
    latest_test_season = prediction_season - 1
    latest_calibration_seasons = [
        season for season in [latest_test_season - 2, latest_test_season - 1] if season in available_seasons
    ]
    latest_train_mask = ((source_np == "tournament") & (seasons_np < min(latest_calibration_seasons))) | (
        (source_np == "regular_season") & (seasons_np <= latest_test_season)
    )
    latest_test_mask = (source_np == "tournament") & (seasons_np == latest_test_season)
    latest_train_idx = np.flatnonzero(latest_train_mask)
    latest_test_idx = np.flatnonzero(latest_test_mask)

    logistic_latest = fit_base_model(X.loc[latest_train_mask], y.loc[latest_train_mask], seed_shrinkage=1.0, c_value=0.2)
    logistic_latest_prob = clip_probabilities(logistic_latest.predict_proba(X.loc[latest_test_mask])[:, 1])
    logistic_latest_metrics = evaluate_probability_predictions(y.loc[latest_test_mask], logistic_latest_prob, "logistic_baseline")

    latest_cache = build_boosting_array_cache(X.iloc[latest_train_idx], y.iloc[latest_train_idx])
    latest_test_df = X.iloc[latest_test_idx]

    def run_seed_trial(seed: int) -> tuple[dict[str, float | int | str], pd.DataFrame]:
        train_idx, val_idx = compute_train_val_indices(latest_cache.y_array, seed)
        boosting_model = fit_boosting_model_from_array(latest_cache, seed=seed, train_idx=train_idx, val_idx=val_idx)
        boosting_prob = clip_probabilities(boosting_model.predict_proba(latest_test_df)[:, 1])
        metrics = evaluate_probability_predictions(y.iloc[latest_test_idx], boosting_prob, "boosting_raw")
        row = {
            "experiment": "seed_robustness",
            "seed": seed,
            "model": "boosting_raw",
            "test_season": latest_test_season,
            **metrics,
            "pct_above_095": float((boosting_prob > 0.95).mean()),
            "pct_between_040_070": float(((boosting_prob >= 0.4) & (boosting_prob <= 0.7)).mean()),
        }
        importance_df = build_boosting_feature_importance_report(boosting_model).reset_index(drop=True)
        importance_df["rank"] = np.arange(1, len(importance_df) + 1)
        importance_df["seed"] = seed
        importance_df["experiment"] = "seed_robustness"
        return row, importance_df

    with ThreadPoolExecutor(max_workers=min(4, len(BOOSTING_STABILITY_SEEDS))) as executor:
        seed_results = list(executor.map(run_seed_trial, BOOSTING_STABILITY_SEEDS))
    seed_rows = [row for row, _ in seed_results]
    importance_rows: list[pd.DataFrame] = [df for _, df in seed_results]

    def run_subsample_trial(seed: int) -> tuple[dict[str, float | int | str], pd.DataFrame]:
        rng = np.random.default_rng(1000 + seed)
        keep_count = max(100, int(np.floor(len(latest_train_idx) * (1 - BOOSTING_SUBSAMPLE_DROP_FRAC))))
        keep_pos = np.sort(rng.choice(np.arange(len(latest_train_idx)), size=keep_count, replace=False))
        subsample_cache = BoostingArrayCache(
            X_array=latest_cache.X_array[keep_pos],
            y_array=latest_cache.y_array[keep_pos],
            feature_order=latest_cache.feature_order,
            preprocessor=latest_cache.preprocessor,
        )
        train_idx, val_idx = compute_train_val_indices(subsample_cache.y_array, seed)
        subsample_model = fit_boosting_model_from_array(subsample_cache, seed=seed, train_idx=train_idx, val_idx=val_idx)
        subsample_prob = clip_probabilities(subsample_model.predict_proba(latest_test_df)[:, 1])
        metrics = evaluate_probability_predictions(y.iloc[latest_test_idx], subsample_prob, "boosting_subsample")
        row = {
            "experiment": "subsample_stability",
            "seed": seed,
            "model": "boosting_subsample",
            "test_season": latest_test_season,
            **metrics,
            "pct_above_095": float((subsample_prob > 0.95).mean()),
            "pct_between_040_070": float(((subsample_prob >= 0.4) & (subsample_prob <= 0.7)).mean()),
        }
        importance_df = build_boosting_feature_importance_report(subsample_model).reset_index(drop=True)
        importance_df["rank"] = np.arange(1, len(importance_df) + 1)
        importance_df["seed"] = seed
        importance_df["experiment"] = "subsample_stability"
        return row, importance_df

    with ThreadPoolExecutor(max_workers=min(4, len(BOOSTING_STABILITY_SEEDS))) as executor:
        subsample_results = list(executor.map(run_subsample_trial, BOOSTING_STABILITY_SEEDS))
    subsample_rows = [row for row, _ in subsample_results]
    importance_rows.extend(df for _, df in subsample_results)

    seed_subsample_df = pd.DataFrame(seed_rows + subsample_rows)
    logistic_summary_row = {
        "experiment": "logistic_baseline",
        "seed": -1,
        "model": "logistic_baseline",
        "test_season": latest_test_season,
        **logistic_latest_metrics,
        "pct_above_095": float((logistic_latest_prob > 0.95).mean()),
        "pct_between_040_070": float(((logistic_latest_prob >= 0.4) & (logistic_latest_prob <= 0.7)).mean()),
    }
    seed_subsample_with_baseline = pd.concat([pd.DataFrame([logistic_summary_row]), seed_subsample_df], ignore_index=True)
    seed_summary = (
        seed_subsample_with_baseline.groupby(["experiment", "model"], dropna=False)[
            ["log_loss", "brier_score", "ece", "pct_above_095", "pct_between_040_070"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    seed_summary.columns = [
        "experiment",
        "model",
        "log_loss_mean",
        "log_loss_std",
        "brier_score_mean",
        "brier_score_std",
        "ece_mean",
        "ece_std",
        "pct_above_095_mean",
        "pct_above_095_std",
        "pct_between_040_070_mean",
        "pct_between_040_070_std",
    ]

    split_cache: dict[int, tuple[BoostingArrayCache, np.ndarray, np.ndarray, np.ndarray]] = {}
    for fold_test_season in fold_test_seasons:
        calibration_seasons = [season for season in [fold_test_season - 2, fold_test_season - 1] if season in available_seasons]
        if len(calibration_seasons) < 2:
            continue
        train_mask = ((source_np == "tournament") & (seasons_np < min(calibration_seasons))) | (
            (source_np == "regular_season") & (seasons_np <= fold_test_season)
        )
        test_mask = (source_np == "tournament") & (seasons_np == fold_test_season)
        train_idx_global = np.flatnonzero(train_mask)
        test_idx_global = np.flatnonzero(test_mask)
        logistic_model = fit_base_model(X.loc[train_mask], y.loc[train_mask], seed_shrinkage=1.0, c_value=0.2)
        if fold_test_season not in split_cache:
            fold_cache = build_boosting_array_cache(X.iloc[train_idx_global], y.iloc[train_idx_global])
            fold_train_idx, fold_val_idx = compute_train_val_indices(fold_cache.y_array, 42)
            split_cache[fold_test_season] = (fold_cache, fold_train_idx, fold_val_idx, test_idx_global)
    def run_rolling_trial(fold_test_season: int) -> list[dict[str, float | int | str]]:
        fold_cache, fold_train_idx, fold_val_idx, fold_test_idx = split_cache[fold_test_season]
        calibration_seasons = [season for season in [fold_test_season - 2, fold_test_season - 1] if season in available_seasons]
        train_mask = ((source_np == "tournament") & (seasons_np < min(calibration_seasons))) | (
            (source_np == "regular_season") & (seasons_np <= fold_test_season)
        )
        logistic_model = fit_base_model(X.loc[train_mask], y.loc[train_mask], seed_shrinkage=1.0, c_value=0.2)
        boosting_model = fit_boosting_model_from_array(fold_cache, seed=42, train_idx=fold_train_idx, val_idx=fold_val_idx)
        logistic_prob = clip_probabilities(logistic_model.predict_proba(X.iloc[fold_test_idx])[:, 1])
        boosting_prob = clip_probabilities(boosting_model.predict_proba(X.iloc[fold_test_idx])[:, 1])
        rows: list[dict[str, float | int | str]] = []
        for model_name, probs in [("logistic_baseline", logistic_prob), ("boosting_raw", boosting_prob)]:
            rows.append(
                {
                    "test_season": fold_test_season,
                    "model": model_name,
                    **evaluate_probability_predictions(y.iloc[fold_test_idx], probs, model_name),
                    "pct_above_095": float((probs > 0.95).mean()),
                    "pct_between_040_070": float(((probs >= 0.4) & (probs <= 0.7)).mean()),
                }
            )
        return rows

    with ThreadPoolExecutor(max_workers=min(4, len(split_cache))) as executor:
        rolling_rows = [row for rows in executor.map(run_rolling_trial, list(split_cache.keys())) for row in rows]
    rolling_df = pd.DataFrame(rolling_rows)
    rolling_comparison = (
        rolling_df.pivot_table(
            index="test_season",
            columns="model",
            values=["log_loss", "brier_score", "ece"],
        )
        .reset_index()
    )
    rolling_comparison.columns = [
        "_".join([str(part) for part in col if str(part) != ""]).strip("_") for col in rolling_comparison.columns.to_flat_index()
    ]
    rolling_comparison["log_loss_improvement"] = (
        rolling_comparison["log_loss_logistic_baseline"] - rolling_comparison["log_loss_boosting_raw"]
    )
    rolling_comparison["brier_improvement"] = (
        rolling_comparison["brier_score_logistic_baseline"] - rolling_comparison["brier_score_boosting_raw"]
    )
    rolling_comparison["ece_improvement"] = (
        rolling_comparison["ece_logistic_baseline"] - rolling_comparison["ece_boosting_raw"]
    )

    importance_stability = pd.concat(importance_rows, ignore_index=True)
    importance_summary = (
        importance_stability.groupby("metric", dropna=False)[["importance", "rank"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    importance_summary.columns = ["metric", "importance_mean", "importance_std", "rank_mean", "rank_std"]
    top3_rate = (
        importance_stability.groupby("metric", dropna=False)["rank"]
        .apply(lambda s: float((s <= 3).mean()))
        .reset_index(name="top3_rate")
    )
    importance_summary = importance_summary.merge(top3_rate, on="metric", how="left")
    importance_summary = importance_summary.sort_values(["rank_mean", "importance_mean"], ascending=[True, False]).reset_index(drop=True)

    stable_seed_edge = bool(
        seed_summary.loc[seed_summary["model"] == "boosting_raw", "log_loss_mean"].iloc[0]
        < seed_summary.loc[seed_summary["model"] == "logistic_baseline", "log_loss_mean"].iloc[0]
    )
    stable_time_edge = bool((rolling_comparison["log_loss_improvement"] > 0).mean() >= 0.6)
    tight_variance = bool(seed_summary.loc[seed_summary["model"] == "boosting_raw", "log_loss_std"].iloc[0] <= 0.003)
    calibration_ok = bool(
        seed_summary.loc[seed_summary["model"] == "boosting_raw", "ece_mean"].iloc[0]
        <= seed_summary.loc[seed_summary["model"] == "logistic_baseline", "ece_mean"].iloc[0] + 0.005
    )
    verdict = "stable_to_promote" if stable_seed_edge and stable_time_edge and tight_variance and calibration_ok else "unstable_hold"
    verdict_df = pd.DataFrame(
        [
            {
                "verdict": verdict,
                "seed_edge_consistent": stable_seed_edge,
                "time_edge_consistent": stable_time_edge,
                "log_loss_std_tight": tight_variance,
                "calibration_ok": calibration_ok,
                "seed_log_loss_delta": float(
                    seed_summary.loc[seed_summary["model"] == "logistic_baseline", "log_loss_mean"].iloc[0]
                    - seed_summary.loc[seed_summary["model"] == "boosting_raw", "log_loss_mean"].iloc[0]
                ),
                "rolling_win_rate": float((rolling_comparison["log_loss_improvement"] > 0).mean()),
            }
        ]
    )
    return seed_summary, rolling_comparison, seed_subsample_with_baseline, importance_summary, verdict_df


def build_feature_set_report(
    X: pd.DataFrame,
    y: pd.Series,
    seasons: pd.Series,
    source_labels: pd.Series,
    prediction_season: int,
    feature_sets: dict[str, list[str]],
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], pd.DataFrame]:
    available_seasons = sorted(int(season) for season in pd.Series(seasons[source_labels == "tournament"]).unique())
    fold_test_seasons = [season for season in available_seasons if prediction_season - 5 <= season <= prediction_season - 1]

    metrics_rows: list[dict[str, float | str]] = []
    reliability_frames: list[pd.DataFrame] = []
    final_configs: dict[str, dict[str, float]] = {}

    for feature_set_name, feature_columns in feature_sets.items():
        for c_value in LOGISTIC_C_GRID:
            for fold_test_season in fold_test_seasons:
                calibration_seasons = [season for season in [fold_test_season - 2, fold_test_season - 1] if season in available_seasons]
                if len(calibration_seasons) < 2:
                    continue
                train_mask = ((source_labels == "tournament") & (seasons < min(calibration_seasons))) | (
                    (source_labels == "regular_season") & (seasons <= fold_test_season)
                )
                calibration_mask = (source_labels == "tournament") & seasons.isin(calibration_seasons)
                fold_test_mask = (source_labels == "tournament") & (seasons == fold_test_season)
                if train_mask.sum() == 0 or calibration_mask.sum() == 0 or fold_test_mask.sum() == 0:
                    continue

                fold_model = fit_base_model(
                    X.loc[train_mask, feature_columns], y.loc[train_mask], seed_shrinkage=1.0, c_value=c_value
                )
                raw_calibration_scores = base_model_raw_score(fold_model, X.loc[calibration_mask, feature_columns])
                raw_test_scores = base_model_raw_score(fold_model, X.loc[fold_test_mask, feature_columns])
                raw_test_prob = logits_to_probabilities(raw_test_scores)
                temperature_fit = fit_temperature_scaler(raw_calibration_scores, y.loc[calibration_mask])
                temp_test_prob = predict_temperature_scaled_proba(raw_test_scores, float(temperature_fit["temperature"]))

                metrics_rows.append(
                    {
                        "feature_set": feature_set_name,
                        "c_value": c_value,
                        "variant": "raw_logistic",
                        "fold_test_season": fold_test_season,
                        **evaluate_probability_predictions(y.loc[fold_test_mask], raw_test_prob, "raw_logistic"),
                        "temperature": 1.0,
                    }
                )
                metrics_rows.append(
                    {
                        "feature_set": feature_set_name,
                        "c_value": c_value,
                        "variant": "temperature_scaled",
                        "fold_test_season": fold_test_season,
                        **evaluate_probability_predictions(y.loc[fold_test_mask], temp_test_prob, "temperature_scaled"),
                        "temperature": float(temperature_fit["temperature"]),
                    }
                )
                reliability_frames.append(
                    build_reliability_curve(
                        y.loc[fold_test_mask], raw_test_prob, f"{feature_set_name}_c{c_value}_raw"
                    )
                )
                reliability_frames.append(
                    build_reliability_curve(
                        y.loc[fold_test_mask], temp_test_prob, f"{feature_set_name}_c{c_value}_temperature_scaled"
                    )
                )

        feature_rows = pd.DataFrame([row for row in metrics_rows if row["feature_set"] == feature_set_name])
        feature_summary = (
            feature_rows.groupby(["feature_set", "c_value", "variant"], dropna=False)[["log_loss", "brier_score", "ece", "temperature"]]
            .mean()
            .reset_index()
            .sort_values(["log_loss", "brier_score", "ece"])
        )
        best_row = feature_summary.iloc[0]
        final_train_mask = ((source_labels == "tournament") & (seasons < prediction_season)) | (
            (source_labels == "regular_season") & (seasons <= prediction_season)
        )
        final_calibration_mask = (source_labels == "tournament") & seasons.isin([prediction_season - 3, prediction_season - 2])
        final_model = fit_base_model(
            X.loc[final_train_mask, feature_columns],
            y.loc[final_train_mask],
            seed_shrinkage=1.0,
            c_value=float(best_row["c_value"]),
        )
        final_raw_scores = base_model_raw_score(final_model, X.loc[final_calibration_mask, feature_columns])
        final_temperature_fit = fit_temperature_scaler(final_raw_scores, y.loc[final_calibration_mask])
        final_configs[feature_set_name] = {
            "c_value": float(best_row["c_value"]),
            "temperature": float(final_temperature_fit["temperature"]),
            "use_temperature": bool(best_row["variant"] == "temperature_scaled"),
        }

    metrics_df = pd.DataFrame(metrics_rows)
    summary_df = (
        metrics_df.groupby(["feature_set", "c_value", "variant"], dropna=False)[["log_loss", "brier_score", "ece", "temperature"]]
        .mean()
        .reset_index()
        .sort_values(["log_loss", "brier_score", "ece"])
    )
    reliability_df = pd.concat(reliability_frames, ignore_index=True)
    return summary_df, final_configs, reliability_df


def build_team_strength_report(
    team_stats: pd.DataFrame,
    model: Pipeline,
    season: int,
    feature_columns: list[str],
) -> pd.DataFrame:
    season_stats = team_stats[team_stats["Season"] == season].copy()
    reference_values = {
        "eff_margin": season_stats["eff_margin"].mean(),
        "def_eff": season_stats["def_eff"].mean(),
        "off_eff": season_stats["off_eff"].mean(),
        "tov_rate": season_stats["tov_rate"].mean(),
        "avg_margin": season_stats["avg_margin"].mean(),
        "three_point_pct": season_stats["three_point_pct"].mean(),
        "three_point_rate": season_stats["three_point_rate"].mean(),
        "three_point_pct_var": season_stats["three_point_pct_var"].mean(),
        "opp_three_point_pct_var": season_stats["opp_three_point_pct_var"].mean(),
        "momentum_score": season_stats["momentum_score"].mean(),
        "pom_rank": season_stats["pom_rank"].mean(),
        "seed_num": season_stats["seed_num"].mean(),
        "tempo": season_stats["tempo"].mean(),
    }
    matchup_rows = pd.DataFrame(
        {
            "Season": season,
            "TeamAID": season_stats["TeamID"],
            "TeamBID": season_stats["TeamID"],
            "RoundNum": 1,
            "SiteAdvantage": 0.0,
            "a_eff_margin": season_stats["eff_margin"].values,
            "b_eff_margin": reference_values["eff_margin"],
            "a_def_eff": season_stats["def_eff"].values,
            "b_def_eff": reference_values["def_eff"],
            "a_off_eff": season_stats["off_eff"].values,
            "b_off_eff": reference_values["off_eff"],
            "a_tov_rate": season_stats["tov_rate"].values,
            "b_tov_rate": reference_values["tov_rate"],
            "a_avg_margin": season_stats["avg_margin"].values,
            "b_avg_margin": reference_values["avg_margin"],
            "a_three_point_pct": season_stats["three_point_pct"].values,
            "b_three_point_pct": reference_values["three_point_pct"],
            "a_three_point_rate": season_stats["three_point_rate"].values,
            "b_three_point_rate": reference_values["three_point_rate"],
            "a_three_point_pct_var": season_stats["three_point_pct_var"].values,
            "b_three_point_pct_var": reference_values["three_point_pct_var"],
            "a_opp_three_point_pct_var": season_stats["opp_three_point_pct_var"].values,
            "b_opp_three_point_pct_var": reference_values["opp_three_point_pct_var"],
            "a_momentum_score": season_stats["momentum_score"].values,
            "b_momentum_score": reference_values["momentum_score"],
            "a_pom_rank": season_stats["pom_rank"].values,
            "b_pom_rank": reference_values["pom_rank"],
            "a_seed_num": season_stats["seed_num"].values,
            "b_seed_num": reference_values["seed_num"],
            "a_tempo": season_stats["tempo"].values,
            "b_tempo": reference_values["tempo"],
        }
    )
    features = compute_matchup_features_from_merged(matchup_rows)
    season_stats["team_logit_rating"] = base_model_raw_score(model, features[feature_columns])
    season_stats["team_win_prob_vs_avg"] = clip_probabilities(model.predict_proba(features[feature_columns])[:, 1])
    season_stats["model_score"] = season_stats["team_logit_rating"]
    return season_stats.sort_values("team_logit_rating", ascending=False)


def predict_with_temperature(model: Pipeline, X: pd.DataFrame, temperature: float) -> np.ndarray:
    raw_scores = base_model_raw_score(model, X)
    return predict_temperature_scaled_proba(raw_scores, temperature)


def predict_hybrid_baseline(model: Pipeline, X: pd.DataFrame, calibration_bundle: CalibrationBundle) -> np.ndarray:
    return predict_calibrated_proba(model, calibration_bundle, X, calibration_bundle.method)


def predict_ensemble_proba(
    model: Pipeline,
    calibration_bundle: CalibrationBundle,
    X: pd.DataFrame,
    temperature: float,
    alpha: float,
) -> np.ndarray:
    logistic_prob = predict_with_temperature(model, X, temperature)
    hybrid_prob = predict_hybrid_baseline(model, X, calibration_bundle)
    return clip_probabilities(alpha * logistic_prob + (1 - alpha) * hybrid_prob)


def evaluate_model(
    X: pd.DataFrame,
    y: pd.Series,
    seasons: pd.Series,
    source_labels: pd.Series,
    eval_seasons: list[int],
    seed_shrinkage: float = 1.0,
) -> pd.DataFrame:
    metrics: list[dict[str, float]] = []
    for season in eval_seasons:
        train_mask = ((source_labels == "tournament") & (seasons < season)) | (
            (source_labels == "regular_season") & (seasons <= season)
        )
        test_mask = (source_labels == "tournament") & (seasons == season)
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        model = fit_base_model(X.loc[train_mask], y.loc[train_mask], seed_shrinkage=seed_shrinkage)
        probs = model.predict_proba(X.loc[test_mask])[:, 1]
        metrics.append(
            {
                "season": season,
                "log_loss": log_loss(y.loc[test_mask], probs, labels=[0, 1]),
                "brier_score": brier_score_loss(y.loc[test_mask], probs),
            }
        )
    return pd.DataFrame(metrics)


def build_calibration_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    seasons: pd.Series,
    source_labels: pd.Series,
    prediction_season: int,
    seed_shrinkage: float = 1.0,
) -> tuple[Pipeline, CalibrationBundle, pd.DataFrame, pd.DataFrame]:
    available_seasons = sorted(int(season) for season in pd.Series(seasons[source_labels == "tournament"]).unique())
    fold_test_seasons = [season for season in available_seasons if prediction_season - 5 <= season <= prediction_season - 1]
    candidate_rows: list[dict[str, float | str]] = []
    reliability_frames: list[pd.DataFrame] = []

    for fold_test_season in fold_test_seasons:
        calibration_seasons = [season for season in [fold_test_season - 2, fold_test_season - 1] if season in available_seasons]
        if len(calibration_seasons) < 2:
            continue
        train_mask = ((source_labels == "tournament") & (seasons < min(calibration_seasons))) | (
            (source_labels == "regular_season") & (seasons <= fold_test_season)
        )
        calibration_mask = (source_labels == "tournament") & seasons.isin(calibration_seasons)
        fold_test_mask = (source_labels == "tournament") & (seasons == fold_test_season)
        if train_mask.sum() == 0 or calibration_mask.sum() == 0 or fold_test_mask.sum() == 0:
            continue

        fold_base_model = fit_base_model(X.loc[train_mask], y.loc[train_mask], seed_shrinkage=seed_shrinkage)
        raw_calibration_scores = base_model_raw_score(fold_base_model, X.loc[calibration_mask])
        raw_test_scores = base_model_raw_score(fold_base_model, X.loc[fold_test_mask])
        raw_test_prob = clip_probabilities(fold_base_model.predict_proba(X.loc[fold_test_mask])[:, 1])

        candidate_rows.append(
            {
                "fold_test_season": fold_test_season,
                **evaluate_probability_predictions(y.loc[fold_test_mask], raw_test_prob, "raw"),
                "c_value": np.nan,
                "blend_weight": 0.0,
                "ensemble_type": "raw",
            }
        )
        reliability_frames.append(build_reliability_curve(y.loc[fold_test_mask], raw_test_prob, "raw"))

        baseline_platt = fit_platt_model(raw_calibration_scores, y.loc[calibration_mask], c_value=1.0, clip_value=PLATT_CLIP_VALUE)
        baseline_platt_prob = predict_platt_model(baseline_platt, raw_test_scores)
        candidate_rows.append(
            {
                "fold_test_season": fold_test_season,
                **evaluate_probability_predictions(y.loc[fold_test_mask], baseline_platt_prob, "baseline_platt"),
                "c_value": 1.0,
                "blend_weight": 1.0,
                "ensemble_type": "single",
            }
        )
        reliability_frames.append(build_reliability_curve(y.loc[fold_test_mask], baseline_platt_prob, "baseline_platt"))

        for c_value in PLATT_C_GRID:
            regularized_platt = fit_platt_model(raw_calibration_scores, y.loc[calibration_mask], c_value=c_value, clip_value=PLATT_CLIP_VALUE)
            regularized_prob = predict_platt_model(regularized_platt, raw_test_scores)
            candidate_rows.append(
                {
                    "fold_test_season": fold_test_season,
                    **evaluate_probability_predictions(y.loc[fold_test_mask], regularized_prob, "regularized_platt"),
                    "c_value": c_value,
                    "blend_weight": 1.0,
                    "ensemble_type": "single",
                }
            )

            ensemble_models = fit_ensemble_platt_models(
                raw_calibration_scores,
                y.loc[calibration_mask],
                seasons.loc[calibration_mask],
                c_value=c_value,
                clip_value=PLATT_CLIP_VALUE,
            )
            ensemble_prob = predict_ensemble_platt(ensemble_models, raw_test_scores)
            candidate_rows.append(
                {
                    "fold_test_season": fold_test_season,
                    **evaluate_probability_predictions(y.loc[fold_test_mask], ensemble_prob, "ensemble_platt"),
                    "c_value": c_value,
                    "blend_weight": 1.0,
                    "ensemble_type": "ensemble",
                }
            )

            for blend_weight in PLATT_BLEND_GRID:
                hybrid_single_prob = blend_probabilities(raw_test_prob, regularized_prob, blend_weight)
                candidate_rows.append(
                    {
                        "fold_test_season": fold_test_season,
                        **evaluate_probability_predictions(y.loc[fold_test_mask], hybrid_single_prob, "hybrid_platt"),
                        "c_value": c_value,
                        "blend_weight": blend_weight,
                        "ensemble_type": "single",
                    }
                )
                hybrid_ensemble_prob = blend_probabilities(raw_test_prob, ensemble_prob, blend_weight)
                candidate_rows.append(
                    {
                        "fold_test_season": fold_test_season,
                        **evaluate_probability_predictions(y.loc[fold_test_mask], hybrid_ensemble_prob, "hybrid_platt"),
                        "c_value": c_value,
                        "blend_weight": blend_weight,
                        "ensemble_type": "ensemble",
                    }
                )

    candidate_df = pd.DataFrame(candidate_rows)
    calibration_metrics = (
        candidate_df.groupby(["label", "c_value", "blend_weight", "ensemble_type"], dropna=False)[["log_loss", "brier_score", "ece"]]
        .mean()
        .reset_index()
        .sort_values(["log_loss", "brier_score", "ece"])
    )
    best_row = calibration_metrics.iloc[0]
    best_label = str(best_row["label"])
    best_c = None if pd.isna(best_row["c_value"]) else float(best_row["c_value"])
    best_blend = None if pd.isna(best_row["blend_weight"]) else float(best_row["blend_weight"])
    best_ensemble_type = str(best_row["ensemble_type"])

    selected_reliability_frames: list[pd.DataFrame] = []
    for fold_test_season in fold_test_seasons:
        calibration_seasons = [season for season in [fold_test_season - 2, fold_test_season - 1] if season in available_seasons]
        if len(calibration_seasons) < 2:
            continue
        train_mask = ((source_labels == "tournament") & (seasons < min(calibration_seasons))) | (
            (source_labels == "regular_season") & (seasons <= fold_test_season)
        )
        calibration_mask = (source_labels == "tournament") & seasons.isin(calibration_seasons)
        fold_test_mask = (source_labels == "tournament") & (seasons == fold_test_season)
        if train_mask.sum() == 0 or calibration_mask.sum() == 0 or fold_test_mask.sum() == 0:
            continue

        fold_base_model = fit_base_model(X.loc[train_mask], y.loc[train_mask], seed_shrinkage=seed_shrinkage)
        raw_calibration_scores = base_model_raw_score(fold_base_model, X.loc[calibration_mask])
        raw_test_scores = base_model_raw_score(fold_base_model, X.loc[fold_test_mask])
        raw_test_prob = clip_probabilities(fold_base_model.predict_proba(X.loc[fold_test_mask])[:, 1])
        selected_reliability_frames.append(build_reliability_curve(y.loc[fold_test_mask], raw_test_prob, "raw"))

        if best_label == "raw":
            continue
        if best_label == "baseline_platt":
            selected_prob = predict_platt_model(
                fit_platt_model(raw_calibration_scores, y.loc[calibration_mask], c_value=1.0, clip_value=PLATT_CLIP_VALUE),
                raw_test_scores,
            )
        elif best_label == "regularized_platt":
            selected_prob = predict_platt_model(
                fit_platt_model(raw_calibration_scores, y.loc[calibration_mask], c_value=best_c, clip_value=PLATT_CLIP_VALUE),
                raw_test_scores,
            )
        elif best_label == "ensemble_platt":
            selected_prob = predict_ensemble_platt(
                fit_ensemble_platt_models(
                    raw_calibration_scores,
                    y.loc[calibration_mask],
                    seasons.loc[calibration_mask],
                    c_value=best_c,
                    clip_value=PLATT_CLIP_VALUE,
                ),
                raw_test_scores,
            )
        else:
            if best_ensemble_type == "ensemble":
                calibrated_prob = predict_ensemble_platt(
                    fit_ensemble_platt_models(
                        raw_calibration_scores,
                        y.loc[calibration_mask],
                        seasons.loc[calibration_mask],
                        c_value=best_c,
                        clip_value=PLATT_CLIP_VALUE,
                    ),
                    raw_test_scores,
                )
            else:
                calibrated_prob = predict_platt_model(
                    fit_platt_model(raw_calibration_scores, y.loc[calibration_mask], c_value=best_c, clip_value=PLATT_CLIP_VALUE),
                    raw_test_scores,
                )
            selected_prob = blend_probabilities(raw_test_prob, calibrated_prob, best_blend)
        selected_reliability_frames.append(build_reliability_curve(y.loc[fold_test_mask], selected_prob, best_label))

    reliability_df = pd.concat(selected_reliability_frames, ignore_index=True)
    final_train_mask = ((source_labels == "tournament") & (seasons < prediction_season)) | (
        (source_labels == "regular_season") & (seasons <= prediction_season)
    )
    final_calibration_mask = (source_labels == "tournament") & seasons.isin([prediction_season - 3, prediction_season - 2])
    final_base_model = fit_base_model(X.loc[final_train_mask], y.loc[final_train_mask], seed_shrinkage=seed_shrinkage)
    final_raw_scores = base_model_raw_score(final_base_model, X.loc[final_calibration_mask])

    if best_label == "raw":
        calibration_bundle = CalibrationBundle(
            method="raw",
            calibrator=None,
            params={"c_value": 0.0, "blend_weight": 0.0, "clip_value": PLATT_CLIP_VALUE, "ensemble_type": "raw"},
        )
    elif best_label == "baseline_platt":
        calibration_bundle = CalibrationBundle(
            method="baseline_platt",
            calibrator=fit_platt_model(final_raw_scores, y.loc[final_calibration_mask], c_value=1.0, clip_value=PLATT_CLIP_VALUE),
            params={"c_value": 1.0, "blend_weight": 1.0, "clip_value": PLATT_CLIP_VALUE, "ensemble_type": "single"},
        )
    elif best_label == "regularized_platt":
        calibration_bundle = CalibrationBundle(
            method="regularized_platt",
            calibrator=fit_platt_model(final_raw_scores, y.loc[final_calibration_mask], c_value=best_c, clip_value=PLATT_CLIP_VALUE),
            params={"c_value": best_c, "blend_weight": 1.0, "clip_value": PLATT_CLIP_VALUE, "ensemble_type": "single"},
        )
    elif best_label == "ensemble_platt":
        calibration_bundle = CalibrationBundle(
            method="ensemble_platt",
            calibrator=fit_ensemble_platt_models(
                final_raw_scores,
                y.loc[final_calibration_mask],
                seasons.loc[final_calibration_mask],
                c_value=best_c,
                clip_value=PLATT_CLIP_VALUE,
            ),
            params={"c_value": best_c, "blend_weight": 1.0, "clip_value": PLATT_CLIP_VALUE, "ensemble_type": "ensemble"},
        )
    else:
        if best_ensemble_type == "ensemble":
            final_calibrator = fit_ensemble_platt_models(
                final_raw_scores,
                y.loc[final_calibration_mask],
                seasons.loc[final_calibration_mask],
                c_value=best_c,
                clip_value=PLATT_CLIP_VALUE,
            )
        else:
            final_calibrator = fit_platt_model(final_raw_scores, y.loc[final_calibration_mask], c_value=best_c, clip_value=PLATT_CLIP_VALUE)
        calibration_bundle = CalibrationBundle(
            method="hybrid_platt",
            calibrator=final_calibrator,
            params={"c_value": best_c, "blend_weight": best_blend, "clip_value": PLATT_CLIP_VALUE, "ensemble_type": best_ensemble_type},
        )

    return final_base_model, calibration_bundle, calibration_metrics, reliability_df


def run_seed_shrinkage_sweep(team_stats: pd.DataFrame, prediction_season: int) -> tuple[float, pd.DataFrame]:
    sweep_rows: list[dict[str, float | str]] = []
    for shrinkage in SEED_SHRINKAGE_GRID:
        X, y, seasons, source_labels = build_training_data(team_stats)
        _, calibration_bundle, calibration_metrics, _ = build_calibration_pipeline(
            X, y, seasons, source_labels, prediction_season, seed_shrinkage=shrinkage
        )
        best_metrics = calibration_metrics.iloc[0].to_dict()
        sweep_rows.append(
            {
                "seed_shrinkage": shrinkage,
                "selected_method": calibration_bundle.method,
                "selected_c_value": calibration_bundle.params.get("c_value", 0.0),
                "selected_blend_weight": calibration_bundle.params.get("blend_weight", 0.0),
                "selected_ensemble_type": calibration_bundle.params.get("ensemble_type", "raw"),
                "log_loss": best_metrics["log_loss"],
                "brier_score": best_metrics["brier_score"],
                "ece": best_metrics["ece"],
            }
        )
    sweep_df = pd.DataFrame(sweep_rows).sort_values(["log_loss", "brier_score", "ece"])
    best_shrinkage = float(sweep_df.iloc[0]["seed_shrinkage"])
    return best_shrinkage, sweep_df


def build_combined_training_data(team_stats: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    tournament_X, tournament_y, tournament_seasons, tournament_source = build_training_data(team_stats)
    regular_X, regular_y, regular_seasons, regular_source = build_regular_season_training_data()
    return (
        pd.concat([tournament_X, regular_X], ignore_index=True),
        pd.concat([tournament_y, regular_y], ignore_index=True),
        pd.concat([tournament_seasons, regular_seasons], ignore_index=True),
        pd.concat([tournament_source, regular_source], ignore_index=True),
    )


def build_current_season_regular_season_exports(season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    games = pd.read_csv(REGULAR_SEASON_FILE)
    season_games = games[games["Season"] == season].copy()
    season_games["EstimatedPossessions"] = (
        possessions(season_games["WFGA"], season_games["WOR"], season_games["WTO"], season_games["WFTA"])
        + possessions(season_games["LFGA"], season_games["LOR"], season_games["LTO"], season_games["LFTA"])
    ) / 2.0
    season_games["GameDateLabel"] = season_games["DayNum"].map(lambda day: f"{season}-Day{int(day):03d}")
    cleaned_games = season_games.rename(
        columns={
            "WTeamID": "WinnerTeamID",
            "LTeamID": "LoserTeamID",
            "WScore": "WinnerScore",
            "LScore": "LoserScore",
            "WLoc": "WinnerLocation",
        }
    )

    team_snapshots = build_regular_season_team_snapshots()
    season_snapshots = team_snapshots[team_snapshots["Season"] == season].copy()
    matchup_games = pd.read_csv(REGULAR_SEASON_FILE, usecols=["Season", "DayNum", "WTeamID", "LTeamID", "WLoc"])
    matchup_games = matchup_games[matchup_games["Season"] == season].copy()

    winners = matchup_games.rename(columns={"WTeamID": "TeamAID", "LTeamID": "TeamBID"})[["Season", "DayNum", "TeamAID", "TeamBID"]]
    winners["target"] = 1
    winners["SiteAdvantage"] = np.select([matchup_games["WLoc"].eq("H"), matchup_games["WLoc"].eq("A")], [1, -1], default=0)
    winners["RoundNum"] = 1

    losers = matchup_games.rename(columns={"LTeamID": "TeamAID", "WTeamID": "TeamBID"})[["Season", "DayNum", "TeamAID", "TeamBID"]]
    losers["target"] = 0
    losers["SiteAdvantage"] = -winners["SiteAdvantage"]
    losers["RoundNum"] = 1

    model_rows = pd.concat([winners, losers], ignore_index=True)
    a_snapshots = season_snapshots.add_prefix("a_")
    b_snapshots = season_snapshots.add_prefix("b_")
    merged = model_rows.merge(
        a_snapshots,
        left_on=["Season", "DayNum", "TeamAID"],
        right_on=["a_Season", "a_DayNum", "a_TeamID"],
        how="left",
    ).merge(
        b_snapshots,
        left_on=["Season", "DayNum", "TeamBID"],
        right_on=["b_Season", "b_DayNum", "b_TeamID"],
        how="left",
    )
    merged = merged[
        (merged["a_games_played_pre"] >= MIN_PREGAME_GAMES) & (merged["b_games_played_pre"] >= MIN_PREGAME_GAMES)
    ].copy()
    features = compute_matchup_features_from_merged(merged)
    model_ready = features[
        ["Season", "DayNum", "TeamAID", "TeamBID", "target", "SiteAdvantage", "RoundNum"] + ALL_FEATURE_COLUMNS
    ].copy()
    return cleaned_games, model_ready


def build_performance_comparison(
    tournament_only_metrics: pd.DataFrame,
    expanded_metrics: pd.DataFrame,
) -> pd.DataFrame:
    baseline = tournament_only_metrics.iloc[0].to_dict()
    expanded = expanded_metrics.iloc[0].to_dict()
    return pd.DataFrame(
        [
            {
                "dataset_variant": "tournament_only",
                "selected_method": baseline["label"],
                "log_loss": baseline["log_loss"],
                "brier_score": baseline["brier_score"],
                "ece": baseline["ece"],
                "c_value": baseline["c_value"],
                "blend_weight": baseline["blend_weight"],
                "ensemble_type": baseline["ensemble_type"],
            },
            {
                "dataset_variant": "tournament_plus_regular_season",
                "selected_method": expanded["label"],
                "log_loss": expanded["log_loss"],
                "brier_score": expanded["brier_score"],
                "ece": expanded["ece"],
                "c_value": expanded["c_value"],
                "blend_weight": expanded["blend_weight"],
                "ensemble_type": expanded["ensemble_type"],
            },
        ]
    )


def predict_submission(
    model: Pipeline,
    calibration: CalibrationBundle,
    team_stats: pd.DataFrame,
    season: int,
    seed_shrinkage: float,
    temperature: float | None = None,
    ensemble_alpha: float | None = None,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    submission = pd.read_csv(SAMPLE_SUBMISSION_FILE)
    season_submission = submission[submission["ID"].str.startswith(f"{season}_")].copy()
    season_submission[["Season", "TeamAID", "TeamBID"]] = season_submission["ID"].str.split("_", expand=True).astype(int)
    season_submission["RoundNum"] = 1
    features = matchup_feature_frame(
        season_submission[["Season", "TeamAID", "TeamBID", "RoundNum"]],
        team_stats,
    )
    selected_columns = feature_columns or ALL_FEATURE_COLUMNS
    selected_features = features[selected_columns]
    if ensemble_alpha is not None and temperature is not None:
        raw_pred = predict_ensemble_proba(
            model, calibration, selected_features, temperature, ensemble_alpha
        )
    elif temperature is None:
        raw_pred = predict_calibrated_proba(
            model, calibration, selected_features, calibration.method
        )
    else:
        raw_pred = predict_with_temperature(model, selected_features, temperature)
    season_submission["Pred"] = apply_medium_gap_probability_adjustment(
        selected_features,
        raw_pred,
        MEDIUM_GAP_PROBABILITY_LIFT,
    )
    return season_submission[["ID", "Pred"]]


def win_probability(
    model: Pipeline,
    calibration: CalibrationBundle,
    team_stats: pd.DataFrame,
    season: int,
    team_a: int,
    team_b: int,
    round_num: int,
    seed_shrinkage: float,
    temperature: float | None = None,
    ensemble_alpha: float | None = None,
    feature_columns: list[str] | None = None,
) -> float:
    game = pd.DataFrame({"Season": [season], "TeamAID": [team_a], "TeamBID": [team_b], "RoundNum": [round_num]})
    features = matchup_feature_frame(game, team_stats)
    selected_columns = feature_columns or ALL_FEATURE_COLUMNS
    selected_features = features[selected_columns]
    if ensemble_alpha is not None and temperature is not None:
        raw_prob = predict_ensemble_proba(model, calibration, selected_features, temperature, ensemble_alpha)
    elif temperature is None:
        raw_prob = predict_calibrated_proba(model, calibration, selected_features, calibration.method)
    else:
        raw_prob = predict_with_temperature(model, selected_features, temperature)
    adjusted_prob = apply_medium_gap_probability_adjustment(
        selected_features,
        raw_prob,
        MEDIUM_GAP_PROBABILITY_LIFT,
    )
    return float(adjusted_prob[0])


def simulate_bracket(
    model: Pipeline,
    calibration: CalibrationBundle,
    team_stats: pd.DataFrame,
    season: int,
    seed_shrinkage: float,
    temperature: float | None = None,
    ensemble_alpha: float | None = None,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    teams = pd.read_csv(TEAMS_FILE)[["TeamID", "TeamName"]]
    seeds = pd.read_csv(SEEDS_FILE)
    slots = pd.read_csv(SLOTS_FILE)
    season_seeds = seeds[seeds["Season"] == season].copy()
    season_slots = slots[slots["Season"] == season].copy()

    seed_to_team = dict(zip(season_seeds["Seed"], season_seeds["TeamID"]))
    stats_index = team_stats[team_stats["Season"] == season].set_index("TeamID")
    team_names = dict(zip(teams["TeamID"], teams["TeamName"]))
    slot_rows = season_slots.set_index("Slot").to_dict("index")
    resolved: dict[str, MatchupResult] = {}

    def resolve(entry: str) -> MatchupResult:
        if entry in resolved:
            return resolved[entry]
        if entry in seed_to_team:
            team_id = int(seed_to_team[entry])
            result = MatchupResult(team_id, team_id, 1.0)
            resolved[entry] = result
            return result
        slot = slot_rows[entry]
        strong = resolve(slot["StrongSeed"])
        weak = resolve(slot["WeakSeed"])
        round_num = int(entry[1]) if entry.startswith("R") and len(entry) > 1 and entry[1].isdigit() else 0
        p_strong = win_probability(
            model,
            calibration,
            team_stats,
            season,
            strong.winner_team_id,
            weak.winner_team_id,
            round_num,
            seed_shrinkage,
            temperature=temperature,
            ensemble_alpha=ensemble_alpha,
            feature_columns=feature_columns,
        )
        if p_strong >= 0.5:
            result = MatchupResult(strong.winner_team_id, weak.winner_team_id, p_strong)
        else:
            result = MatchupResult(weak.winner_team_id, strong.winner_team_id, 1.0 - p_strong)
        resolved[entry] = result
        return result

    bracket_rows = []
    for _, row in season_slots.iterrows():
        outcome = resolve(row["Slot"])
        winner_seed = stats_index.loc[outcome.winner_team_id, "Seed"] if outcome.winner_team_id in stats_index.index else None
        loser_seed = stats_index.loc[outcome.loser_team_id, "Seed"] if outcome.loser_team_id in stats_index.index else None
        bracket_rows.append(
            {
                "Season": season,
                "Slot": row["Slot"],
                "StrongSeed": row["StrongSeed"],
                "WeakSeed": row["WeakSeed"],
                "WinnerTeamID": outcome.winner_team_id,
                "WinnerTeamName": team_names.get(outcome.winner_team_id, str(outcome.winner_team_id)),
                "WinnerSeed": winner_seed,
                "LoserTeamID": outcome.loser_team_id,
                "LoserTeamName": team_names.get(outcome.loser_team_id, str(outcome.loser_team_id)),
                "LoserSeed": loser_seed,
                "WinnerProb": outcome.winner_prob,
            }
        )
    return pd.DataFrame(bracket_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a March Madness prediction model and 2026 bracket.")
    parser.add_argument("--season", type=int, default=2026, help="Season to score and simulate.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/wilcroutwater/Documents/Playground/output/march_madness"),
        help="Directory for prediction outputs.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    team_stats = build_team_metrics()
    tournament_X, tournament_y, tournament_seasons, tournament_source = build_training_data(team_stats)
    combined_X, combined_y, combined_seasons, combined_source = build_combined_training_data(team_stats)

    feature_set_metrics, feature_set_configs, feature_set_reliability = build_feature_set_report(
        combined_X, combined_y, combined_seasons, combined_source, args.season, FEATURE_SET_VARIANTS
    )
    feature_set_comparison = (
        feature_set_metrics.sort_values(["log_loss", "brier_score", "ece"])
        .groupby("feature_set", as_index=False)
        .first()
        .sort_values(["log_loss", "brier_score", "ece"])
    )
    selected_feature_row = feature_set_comparison.iloc[0]
    selected_feature_set = str(selected_feature_row["feature_set"])
    selected_feature_columns = FEATURE_SET_VARIANTS[selected_feature_set]
    selected_feature_config = feature_set_configs[selected_feature_set]

    _, _, tournament_only_calibration_metrics, _ = build_calibration_pipeline(
        tournament_X[selected_feature_columns], tournament_y, tournament_seasons, tournament_source, args.season
    )
    _, calibration_bundle, calibration_metrics, reliability_df = build_calibration_pipeline(
        combined_X[selected_feature_columns], combined_y, combined_seasons, combined_source, args.season
    )
    performance_comparison = build_performance_comparison(tournament_only_calibration_metrics, calibration_metrics)
    model_comparison = build_model_comparison_report(calibration_metrics)
    temperature_fit, temperature_metrics, temperature_reliability, temperature_distribution = build_temperature_scaling_report(
        combined_X[selected_feature_columns], combined_y, combined_seasons, combined_source, args.season
    )
    ensemble_fit, ensemble_metrics, ensemble_reliability, ensemble_distribution = build_ensemble_report(
        combined_X[selected_feature_columns], combined_y, combined_seasons, combined_source, args.season
    )
    (
        boosting_model,
        boosting_calibration_bundle,
        boosting_metrics,
        boosting_reliability,
        boosting_distribution,
    ) = build_boosting_benchmark_report(
        combined_X[selected_feature_columns], combined_y, combined_seasons, combined_source, args.season
    )
    (
        boosting_stability_summary,
        boosting_rolling_comparison,
        boosting_stability_runs,
        boosting_importance_stability,
        boosting_stability_verdict,
    ) = build_boosting_stability_report(
        combined_X[selected_feature_columns], combined_y, combined_seasons, combined_source, args.season
    )

    final_train_mask = ((combined_source == "tournament") & (combined_seasons < args.season)) | (
        (combined_source == "regular_season") & (combined_seasons <= args.season)
    )
    model = fit_base_model(
        combined_X.loc[final_train_mask, selected_feature_columns],
        combined_y.loc[final_train_mask],
        seed_shrinkage=1.0,
        c_value=float(selected_feature_config["c_value"]),
    )
    selected_temperature = float(selected_feature_config["temperature"])
    selected_alpha = float(ensemble_fit["alpha"])
    raw_temp_row = temperature_metrics[temperature_metrics["label"] == "raw_logistic"].iloc[0]
    scaled_temp_row = temperature_metrics[temperature_metrics["label"] == "temperature_scaled"].iloc[0]
    use_temperature_scaling = bool(selected_feature_config["use_temperature"])
    ensemble_best_row = ensemble_metrics.iloc[0]
    use_ensemble = False

    recent_eval = evaluate_model(
        combined_X[selected_feature_columns],
        combined_y,
        combined_seasons,
        combined_source,
        [2022, 2023, 2024, 2025],
    )
    raw_bundle = CalibrationBundle(
        method="raw",
        calibrator=None,
        params={"c_value": 0.0, "blend_weight": 0.0, "clip_value": 0.0, "ensemble_type": "raw"},
    )
    submission_predictions = predict_submission(
        model,
        raw_bundle,
        team_stats,
        args.season,
        seed_shrinkage=1.0,
        temperature=selected_temperature if use_temperature_scaling else None,
        ensemble_alpha=None,
        feature_columns=selected_feature_columns,
    )
    bracket = simulate_bracket(
        model,
        raw_bundle,
        team_stats,
        args.season,
        seed_shrinkage=1.0,
        temperature=selected_temperature if use_temperature_scaling else None,
        ensemble_alpha=None,
        feature_columns=selected_feature_columns,
    )
    cleaned_regular_games, current_regular_model_ready = build_current_season_regular_season_exports(args.season)
    team_strength_report = build_team_strength_report(team_stats, model, args.season, selected_feature_columns)

    submission_path = args.output_dir / f"submission_{args.season}.csv"
    bracket_path = args.output_dir / f"bracket_{args.season}.csv"
    team_stats_path = args.output_dir / f"team_strengths_{args.season}.csv"
    eval_path = args.output_dir / "model_backtest.csv"
    model_score_weights_path = args.output_dir / "model_score_component_weights.csv"
    matchup_weights_path = args.output_dir / "matchup_feature_weights.csv"
    logistic_coefficients_path = args.output_dir / "logistic_feature_coefficients.csv"
    calibration_metrics_path = args.output_dir / "calibration_metrics.csv"
    reliability_path = args.output_dir / "calibration_reliability.csv"
    calibration_choice_path = args.output_dir / "calibration_choice.json"
    performance_comparison_path = args.output_dir / "dataset_performance_comparison.csv"
    feature_set_comparison_path = args.output_dir / "feature_set_comparison.csv"
    feature_set_reliability_path = args.output_dir / "feature_set_reliability.csv"
    model_comparison_path = args.output_dir / "model_comparison.csv"
    temperature_metrics_path = args.output_dir / "temperature_scaling_metrics.csv"
    temperature_reliability_path = args.output_dir / "temperature_scaling_reliability.csv"
    temperature_distribution_path = args.output_dir / "temperature_scaling_distribution.csv"
    ensemble_metrics_path = args.output_dir / "ensemble_metrics.csv"
    ensemble_reliability_path = args.output_dir / "ensemble_reliability.csv"
    ensemble_distribution_path = args.output_dir / "ensemble_distribution.csv"
    boosting_metrics_path = args.output_dir / "boosting_model_comparison.csv"
    boosting_reliability_path = args.output_dir / "boosting_reliability.csv"
    boosting_distribution_path = args.output_dir / "boosting_distribution.csv"
    boosting_importance_path = args.output_dir / "boosting_feature_importance.csv"
    boosting_stability_summary_path = args.output_dir / "boosting_stability_summary.csv"
    boosting_rolling_comparison_path = args.output_dir / "boosting_rolling_comparison.csv"
    boosting_stability_runs_path = args.output_dir / "boosting_stability_runs.csv"
    boosting_importance_stability_path = args.output_dir / "boosting_importance_stability.csv"
    boosting_stability_verdict_path = args.output_dir / "boosting_stability_verdict.csv"
    cleaned_regular_games_path = args.output_dir / f"regular_season_games_{args.season}.csv"
    current_regular_model_path = args.output_dir / f"regular_season_matchups_{args.season}.csv"
    ingestion_summary_path = args.output_dir / "regular_season_ingestion_summary.json"

    submission_predictions.to_csv(submission_path, index=False)
    bracket.to_csv(bracket_path, index=False)
    team_strength_report.to_csv(team_stats_path, index=False)
    recent_eval.to_csv(eval_path, index=False)
    logistic_coefficients = build_matchup_feature_weights_report(model)
    logistic_coefficients.to_csv(matchup_weights_path, index=False)
    logistic_coefficients.to_csv(logistic_coefficients_path, index=False)
    logistic_coefficients.to_csv(model_score_weights_path, index=False)
    calibration_metrics.to_csv(calibration_metrics_path, index=False)
    reliability_df.to_csv(reliability_path, index=False)
    performance_comparison.to_csv(performance_comparison_path, index=False)
    feature_set_comparison.to_csv(feature_set_comparison_path, index=False)
    feature_set_reliability.to_csv(feature_set_reliability_path, index=False)
    model_comparison.to_csv(model_comparison_path, index=False)
    temperature_metrics.to_csv(temperature_metrics_path, index=False)
    temperature_reliability.to_csv(temperature_reliability_path, index=False)
    temperature_distribution.to_csv(temperature_distribution_path, index=False)
    ensemble_metrics.to_csv(ensemble_metrics_path, index=False)
    ensemble_reliability.to_csv(ensemble_reliability_path, index=False)
    ensemble_distribution.to_csv(ensemble_distribution_path, index=False)
    boosting_metrics.to_csv(boosting_metrics_path, index=False)
    boosting_reliability.to_csv(boosting_reliability_path, index=False)
    boosting_distribution.to_csv(boosting_distribution_path, index=False)
    build_boosting_feature_importance_report(boosting_model).to_csv(boosting_importance_path, index=False)
    boosting_stability_summary.to_csv(boosting_stability_summary_path, index=False)
    boosting_rolling_comparison.to_csv(boosting_rolling_comparison_path, index=False)
    boosting_stability_runs.to_csv(boosting_stability_runs_path, index=False)
    boosting_importance_stability.to_csv(boosting_importance_stability_path, index=False)
    boosting_stability_verdict.to_csv(boosting_stability_verdict_path, index=False)
    cleaned_regular_games.to_csv(cleaned_regular_games_path, index=False)
    current_regular_model_ready.to_csv(current_regular_model_path, index=False)
    calibration_choice_path.write_text(
        json.dumps(
            {
                "selected_method": (
                    "temperature_scaled_logistic_regression" if use_temperature_scaling else "raw_logistic_regression"
                ),
                "params": {
                    "regularization_c": float(selected_feature_config["c_value"]),
                    "penalty": "l2",
                    "temperature": selected_temperature,
                    "temperature_selected_for_production": use_temperature_scaling,
                    "medium_gap_probability_lift": MEDIUM_GAP_PROBABILITY_LIFT,
                    "ensemble_alpha": selected_alpha,
                    "ensemble_selected_for_production": use_ensemble,
                    "feature_set": selected_feature_set,
                    "feature_count": len(selected_feature_columns),
                },
                "comparison_baseline": {
                    "feature_set": "baseline_compact",
                    "metrics": feature_set_comparison[feature_set_comparison["feature_set"] == "baseline_compact"]
                    .iloc[0]
                    .to_dict(),
                },
                "dataset_variant": "tournament_plus_regular_season",
                "boosting_benchmark": boosting_metrics.iloc[0].to_dict(),
                "boosting_stability": boosting_stability_verdict.iloc[0].to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    ingestion_summary_path.write_text(
        json.dumps(
            {
                "current_season": args.season,
                "current_regular_season_games": int(len(cleaned_regular_games)),
                "current_regular_season_model_rows": int(len(current_regular_model_ready)),
                "historical_tournament_training_games": int(len(tournament_X) // 2),
                "historical_regular_season_training_games": int((combined_source == "regular_season").sum() // 2),
                "leakage_controls": [
                    "Regular-season matchup features are built from pregame cumulative team stats shifted before each game.",
                    "Pregame POM ranks are merged with backward asof logic and no same-day forward lookup.",
                    "Calibration and holdout evaluation use tournament games only.",
                    "Final training uses regular-season data only through the prediction season and tournament data only from prior seasons.",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    champion = bracket.loc[bracket["Slot"] == "R6CH"].iloc[0]
    print(f"Saved predictions to {submission_path}")
    print(f"Saved bracket picks to {bracket_path}")
    print(f"Saved 2026 team strengths to {team_stats_path}")
    print(f"Saved backtest metrics to {eval_path}")
    print(f"Saved compatibility weights file to {model_score_weights_path}")
    print(f"Saved matchup feature weights to {matchup_weights_path}")
    print(f"Saved logistic feature coefficients to {logistic_coefficients_path}")
    print(f"Saved calibration metrics to {calibration_metrics_path}")
    print(f"Saved calibration reliability data to {reliability_path}")
    print(f"Saved dataset performance comparison to {performance_comparison_path}")
    print(f"Saved feature set comparison to {feature_set_comparison_path}")
    print(f"Saved feature set reliability to {feature_set_reliability_path}")
    print(f"Saved model comparison to {model_comparison_path}")
    print(f"Saved temperature metrics to {temperature_metrics_path}")
    print(f"Saved temperature reliability to {temperature_reliability_path}")
    print(f"Saved temperature distribution to {temperature_distribution_path}")
    print(f"Saved ensemble metrics to {ensemble_metrics_path}")
    print(f"Saved ensemble reliability to {ensemble_reliability_path}")
    print(f"Saved ensemble distribution to {ensemble_distribution_path}")
    print(f"Saved boosting comparison to {boosting_metrics_path}")
    print(f"Saved boosting reliability to {boosting_reliability_path}")
    print(f"Saved boosting distribution to {boosting_distribution_path}")
    print(f"Saved boosting feature importance to {boosting_importance_path}")
    print(f"Saved boosting stability summary to {boosting_stability_summary_path}")
    print(f"Saved boosting rolling comparison to {boosting_rolling_comparison_path}")
    print(f"Saved boosting stability runs to {boosting_stability_runs_path}")
    print(f"Saved boosting importance stability to {boosting_importance_stability_path}")
    print(f"Saved boosting stability verdict to {boosting_stability_verdict_path}")
    print(f"Saved cleaned regular season games to {cleaned_regular_games_path}")
    print(f"Saved model-ready regular season matchups to {current_regular_model_path}")
    print(f"Saved ingestion summary to {ingestion_summary_path}")
    print(
        f"Selected feature set: {selected_feature_set} ({len(selected_feature_columns)} features). "
        f"Temperature scaling evaluated at T={selected_temperature:.2f}; "
        f"selected for production: {use_temperature_scaling}. "
        f"Ensemble alpha evaluated at {selected_alpha:.2f}; selected for production: {use_ensemble}. "
        "Compact matchup feature comparison completed."
    )
    print(
        f"Champion pick: {champion['WinnerTeamName']} over {champion['LoserTeamName']} "
        f"({champion['WinnerProb']:.3f} win probability)"
    )


if __name__ == "__main__":
    main()
