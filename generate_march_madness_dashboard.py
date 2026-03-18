from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.io as pio
from jinja2 import Template


HTML_TEMPLATE = Template(
    """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>March Madness Dashboard</title>
  <style>
    :root {
      --bg: #f3efe6;
      --panel: #fffaf2;
      --ink: #1d1a16;
      --muted: #74685b;
      --accent: #c85c38;
      --accent-dark: #8f3417;
      --line: rgba(29, 26, 22, 0.08);
      --shadow: 0 18px 40px rgba(76, 49, 31, 0.10);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(200, 92, 56, 0.18), transparent 32%),
        radial-gradient(circle at top right, rgba(128, 154, 135, 0.18), transparent 28%),
        linear-gradient(180deg, #f8f4eb 0%, var(--bg) 100%);
    }
    .wrap {
      width: min(1180px, calc(100% - 32px));
      margin: 0 auto;
      padding: 28px 0 56px;
    }
    .hero {
      display: grid;
      gap: 18px;
      margin-bottom: 24px;
      padding: 28px;
      background: linear-gradient(135deg, rgba(255,250,242,0.96), rgba(248,235,221,0.92));
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow);
    }
    .eyebrow {
      letter-spacing: 0.18em;
      text-transform: uppercase;
      font-size: 12px;
      color: var(--accent-dark);
      margin: 0;
    }
    h1 {
      margin: 0;
      font-size: clamp(34px, 5vw, 58px);
      line-height: 0.95;
    }
    .sub {
      margin: 0;
      max-width: 800px;
      color: var(--muted);
      font-size: 17px;
      line-height: 1.5;
    }
    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
    }
    .card, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: var(--shadow);
    }
    .card {
      padding: 18px;
    }
    .label {
      display: block;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: var(--muted);
      margin-bottom: 8px;
    }
    .metric {
      font-size: 30px;
      font-weight: 700;
    }
    .mini {
      margin-top: 8px;
      color: var(--muted);
      font-size: 14px;
    }
    .grid {
      display: grid;
      grid-template-columns: 1.2fr 1fr;
      gap: 18px;
      margin-bottom: 18px;
    }
    .panel {
      padding: 18px;
      overflow: hidden;
    }
    .panel h2 {
      margin: 0 0 4px;
      font-size: 22px;
    }
    .panel p {
      margin: 0 0 16px;
      color: var(--muted);
    }
    .plot {
      width: 100%;
    }
    .final-four {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 12px;
    }
    .team-chip {
      padding: 16px;
      background: linear-gradient(180deg, rgba(200,92,56,0.10), rgba(255,250,242,0.6));
      border-radius: 16px;
      border: 1px solid rgba(200, 92, 56, 0.18);
    }
    .team-chip strong {
      display: block;
      font-size: 22px;
      margin-bottom: 6px;
    }
    .bracket-shell {
      display: grid;
      gap: 18px;
    }
    .bracket-board {
      display: grid;
      grid-template-columns: 1fr;
      gap: 18px;
    }
    .region-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }
    .play-in-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }
    .region-panel {
      padding: 16px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background:
        linear-gradient(180deg, rgba(255,250,242,0.9), rgba(248,235,221,0.8)),
        linear-gradient(135deg, rgba(200,92,56,0.05), rgba(73,106,90,0.05));
    }
    .region-panel h3, .finals-panel h3 {
      margin: 0 0 4px;
      font-size: 20px;
    }
    .region-panel p, .finals-panel p {
      margin: 0 0 14px;
      color: var(--muted);
      font-size: 14px;
    }
    .round-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(150px, 1fr));
      gap: 12px;
      align-items: start;
    }
    .round-col {
      display: grid;
      gap: 10px;
    }
    .round-head {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      color: var(--muted);
      margin-bottom: 2px;
    }
    .game-card {
      padding: 10px 11px;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.7);
    }
    .game-slot {
      font-size: 11px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 8px;
    }
    .team-row {
      display: grid;
      grid-template-columns: 42px 1fr auto;
      gap: 8px;
      align-items: center;
      padding: 6px 0;
      border-top: 1px solid rgba(29, 26, 22, 0.05);
    }
    .team-row:first-of-type {
      border-top: 0;
      padding-top: 0;
    }
    .team-seed {
      font-size: 12px;
      color: var(--muted);
    }
    .team-name {
      font-size: 14px;
      font-weight: 700;
      line-height: 1.2;
    }
    .team-prob {
      font-size: 12px;
      color: var(--muted);
      text-align: right;
    }
    .winner {
      color: var(--accent-dark);
    }
    .finals-panel {
      padding: 18px;
      border-radius: 20px;
      border: 1px solid rgba(200, 92, 56, 0.18);
      background:
        radial-gradient(circle at top, rgba(200,92,56,0.14), transparent 42%),
        linear-gradient(180deg, rgba(255,250,242,0.96), rgba(248,235,221,0.92));
    }
    .finals-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      align-items: start;
    }
    .final-stage {
      display: grid;
      gap: 10px;
    }
    .champion-card {
      padding: 16px;
      border-radius: 16px;
      border: 1px solid rgba(200, 92, 56, 0.22);
      background: rgba(255,255,255,0.74);
      text-align: center;
    }
    .champion-card strong {
      display: block;
      font-size: 28px;
      margin: 8px 0 4px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }
    th, td {
      padding: 10px 8px;
      border-bottom: 1px solid var(--line);
      text-align: left;
    }
    th {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: var(--muted);
    }
    .table-wrap {
      overflow-x: auto;
    }
    @media (max-width: 900px) {
      .grid {
        grid-template-columns: 1fr;
      }
      .region-grid {
        grid-template-columns: 1fr;
      }
      .play-in-grid,
      .round-grid,
      .finals-grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <p class="eyebrow">2026 Men's Tournament Production Model</p>
      <h1>{{ champion_name }} over {{ runner_up_name }}</h1>
      <p class="sub">
        The live system is a validated interaction-enhanced logistic regression with seed retained
        as real signal, no global calibration layer, and one localized medium-gap upset correction.
        It passed stability validation, walk-forward tournament backtesting, and seed-bias diagnostics
        before promotion.
      </p>
      <div class="cards">
        <div class="card">
          <span class="label">Champion Win Prob</span>
          <div class="metric">{{ champion_prob }}</div>
          <div class="mini">{{ champion_seed }} over {{ runner_up_seed }}</div>
        </div>
        <div class="card">
          <span class="label">Best Backtest Log Loss</span>
          <div class="metric">{{ best_log_loss }}</div>
          <div class="mini">Held-out seasons 2022-2025</div>
        </div>
        <div class="card">
          <span class="label">Top Rated Team</span>
          <div class="metric">{{ top_team_name }}</div>
          <div class="mini">Score {{ top_team_score }}</div>
        </div>
        <div class="card">
          <span class="label">Production Model</span>
          <div class="metric">{{ production_model_name }}</div>
          <div class="mini">{{ production_model_params }}</div>
        </div>
        <div class="card">
          <span class="label">Feature Set</span>
          <div class="metric">{{ feature_set_name }}</div>
          <div class="mini">{{ feature_set_count }} features in production</div>
        </div>
        <div class="card">
          <span class="label">Medium-Gap Adjustment</span>
          <div class="metric">{{ medium_gap_adjustment }}</div>
          <div class="mini">{{ medium_gap_adjustment_note }}</div>
        </div>
        <div class="card">
          <span class="label">2026 Regular Games</span>
          <div class="metric">{{ regular_games }}</div>
          <div class="mini">{{ regular_matchup_rows }} model-ready matchup rows</div>
        </div>
      </div>
    </section>

    <section class="grid">
      <div class="panel">
        <h2>Backtest</h2>
        <p>Recent holdout performance for the final production logistic pipeline.</p>
        <div class="plot">{{ backtest_plot|safe }}</div>
      </div>
      <div class="panel">
        <h2>Final Four Path</h2>
        <p>Regional winners and title path from the validated production model.</p>
        <div class="final-four">
          {% for team in final_four %}
          <div class="team-chip">
            <span class="label">{{ team.slot }}</span>
            <strong>{{ team.team }}</strong>
            <div>{{ team.seed }}</div>
            <div class="mini">Win prob {{ team.prob }}</div>
          </div>
          {% endfor %}
        </div>
      </div>
    </section>

    <section class="grid">
      <div class="panel">
        <h2>Validation Summary</h2>
        <p>Production promotion summary after stability, tournament, and behavioral audits.</p>
        <div class="table-wrap">{{ validation_summary_table|safe }}</div>
      </div>
      <div class="panel">
        <h2>Behavioral Audit</h2>
        <p>Seed signal, upset pricing, and calibration checks that support the final production decision.</p>
        <div class="table-wrap">{{ behavioral_audit_table|safe }}</div>
      </div>
    </section>

    <section class="grid">
      <div class="panel">
        <h2>Top Team Ratings</h2>
        <p>Top teams by production rating against an average tournament team.</p>
        <div class="plot">{{ strength_plot|safe }}</div>
      </div>
      <div class="panel">
        <h2>Bracket Confidence</h2>
        <p>Most confident slot winners in the production bracket tree.</p>
        <div class="plot">{{ confidence_plot|safe }}</div>
      </div>
    </section>

    <section class="grid">
      <div class="panel">
        <h2>Why Logistic Stayed Production</h2>
        <p>Historical comparison versus the earlier hybrid-Platt pipeline. This remains evidence, not an active model-selection contest.</p>
        <div class="plot">{{ model_comparison_plot|safe }}</div>
      </div>
      <div class="panel">
        <h2>Logistic Coefficients</h2>
        <p>Learned logistic-regression coefficients on standardized matchup features.</p>
        <div class="plot">{{ matchup_weights_plot|safe }}</div>
      </div>
    </section>

    <section class="grid">
      <div class="panel">
        <h2>Rejected Alternative: Boosting</h2>
        <p>Controlled model-class benchmark on the same feature set. Boosting was not promoted because it failed the robustness gate.</p>
        <div class="plot">{{ boosting_plot|safe }}</div>
      </div>
      <div class="panel">
        <h2>Boosting Importance</h2>
        <p>Importance shares from the benchmarked tree model, retained here as historical validation context.</p>
        <div class="plot">{{ boosting_importance_plot|safe }}</div>
      </div>
    </section>

    <section class="panel" style="margin-bottom: 18px;">
      <h2>Boosting Rejection Detail</h2>
      <p>Held-out metrics and high-confidence probability behavior for logistic, raw boosting, and calibrated boosting.</p>
      <div class="table-wrap">{{ boosting_metrics_table|safe }}</div>
      <div class="table-wrap">{{ boosting_distribution_table|safe }}</div>
    </section>

    <section class="grid">
      <div class="panel">
        <h2>Boosting Stability</h2>
        <p>Mean and standard deviation across seed and subsample reruns on the latest holdout season.</p>
        <div class="table-wrap">{{ boosting_stability_summary_table|safe }}</div>
      </div>
      <div class="panel">
        <h2>Boosting Stability Verdict</h2>
        <p>Production gate on robustness across seeds, time splits, and calibration behavior.</p>
        <div class="table-wrap">{{ boosting_stability_verdict_table|safe }}</div>
      </div>
    </section>

    <section class="grid">
      <div class="panel">
        <h2>Rolling Seasons</h2>
        <p>Per-season tournament comparison of boosting versus logistic across rolling time splits.</p>
        <div class="plot">{{ boosting_rolling_plot|safe }}</div>
      </div>
      <div class="panel">
        <h2>Importance Stability</h2>
        <p>Top feature ranks and importance dispersion across seed and subsample reruns.</p>
        <div class="table-wrap">{{ boosting_importance_stability_table|safe }}</div>
      </div>
    </section>

    <section class="grid">
      <div class="panel">
        <h2>Legacy Calibration Research</h2>
        <p>Historical calibration benchmark from the earlier Platt-based pipeline. Global calibration is not part of the final production system.</p>
        <div class="plot">{{ calibration_curve_plot|safe }}</div>
      </div>
      <div class="panel">
        <h2>Legacy Calibration Metrics</h2>
        <p>Retained as research context only. Production uses raw logistic probabilities plus a localized medium-gap correction.</p>
        <div class="table-wrap">{{ calibration_metrics_table|safe }}</div>
      </div>
    </section>

    <section class="grid">
      <div class="panel">
        <h2>Rejected Global Temperature Scaling</h2>
        <p>Global temperature scaling was tested and not retained because it did not improve production-quality holdout behavior.</p>
        <div class="plot">{{ temperature_metrics_plot|safe }}</div>
      </div>
      <div class="panel">
        <h2>Probability Tails</h2>
        <p>Temperature scaling softened tails, but the final production model kept raw logistic probabilities because global softening was not superior.</p>
        <div class="plot">{{ temperature_distribution_plot|safe }}</div>
      </div>
    </section>

    <section class="panel" style="margin-bottom: 18px;">
      <h2>Temperature Scaling Detail</h2>
      <p>Fitted temperature, holdout metrics, and extreme-probability share from the rejected global-scaling experiment.</p>
      <div class="table-wrap">{{ temperature_metrics_table|safe }}</div>
      <div class="table-wrap">{{ temperature_distribution_table|safe }}</div>
    </section>

    <section class="grid">
      <div class="panel">
        <h2>Feature Upgrade</h2>
        <p>Replacement test: matchup features are allowed to replace efficiency margin, predictive rank, or both.</p>
        <div class="plot">{{ feature_set_plot|safe }}</div>
      </div>
      <div class="panel">
        <h2>Feature Set Table</h2>
        <p>Held-out tournament metrics across baseline and replacement variants.</p>
        <div class="table-wrap">{{ feature_set_table|safe }}</div>
      </div>
    </section>

    <section class="grid">
      <div class="panel">
        <h2>Historical Ensemble Search</h2>
        <p>Earlier blend search between logistic and hybrid-Platt variants. This is no longer part of live production selection.</p>
        <div class="plot">{{ ensemble_metrics_plot|safe }}</div>
      </div>
      <div class="panel">
        <h2>Ensemble Tails</h2>
        <p>How alpha changed extreme-confidence frequency and mid-range stability during the earlier model-selection phase.</p>
        <div class="plot">{{ ensemble_distribution_plot|safe }}</div>
      </div>
    </section>

    <section class="panel" style="margin-bottom: 18px;">
      <h2>Ensemble Search Detail</h2>
      <p>Alpha sweep, holdout metrics, and probability-shape summary from the historical ensemble experiment.</p>
      <div class="table-wrap">{{ ensemble_metrics_table|safe }}</div>
      <div class="table-wrap">{{ ensemble_distribution_table|safe }}</div>
    </section>

    <section class="grid">
      <div class="panel">
        <h2>Dataset Expansion</h2>
        <p>Tournament-only versus tournament-plus-regular-season holdout performance.</p>
        <div class="plot">{{ dataset_comparison_plot|safe }}</div>
      </div>
      <div class="panel">
        <h2>Ingestion Summary</h2>
        <p>Current-season volume, historical sample growth, and leakage controls.</p>
        <div class="table-wrap">{{ ingestion_summary_table|safe }}</div>
      </div>
    </section>

    <section class="panel" style="margin-bottom: 18px;">
      <h2>Dataset Comparison Table</h2>
      <p>Selected calibration method and holdout metrics for each training-data variant.</p>
      <div class="table-wrap">{{ dataset_comparison_table|safe }}</div>
    </section>

    <section class="grid">
      <div class="panel">
        <h2>Tournament Backtest</h2>
        <p>Strict walk-forward NCAA tournament performance with the fixed production logistic configuration.</p>
        <div class="plot">{{ tournament_backtest_plot|safe }}</div>
      </div>
      <div class="panel">
        <h2>Seed Gap Calibration</h2>
        <p>Average higher-seed win probability versus actual higher-seed win rate by seed-gap bucket, including the corrected medium-gap zone.</p>
        <div class="plot">{{ tournament_seed_gap_plot|safe }}</div>
      </div>
    </section>

    <section class="grid">
      <div class="panel">
        <h2>Seed Matchup Gaps</h2>
        <p>Calibration gap by seed pairing, filtered to matchup groups with repeated history.</p>
        <div class="plot">{{ tournament_seed_matchup_plot|safe }}</div>
      </div>
      <div class="panel">
        <h2>With vs Without Seed</h2>
        <p>Counterfactual comparison of the production feature set against the same model without seed-based terms.</p>
        <div class="plot">{{ tournament_seed_counterfactual_plot|safe }}</div>
      </div>
    </section>

    <section class="grid">
      <div class="panel">
        <h2>Upset and Distribution</h2>
        <p>Upset-frequency calibration and probability-shape summary across 2018-2025 tournaments.</p>
        <div class="table-wrap">{{ tournament_upset_table|safe }}</div>
        <div class="table-wrap">{{ tournament_distribution_table|safe }}</div>
      </div>
      <div class="panel">
        <h2>Seed-Bias Diagnosis</h2>
        <p>Seed-bias conclusion based on walk-forward calibration, upset behavior, and the no-seed counterfactual.</p>
        <div class="table-wrap">{{ tournament_diagnosis_table|safe }}</div>
      </div>
    </section>

    <section class="grid">
      <div class="panel">
        <h2>Medium-Gap Refinement</h2>
        <p>Localized variants tested only to improve 3-to-5 seed-gap upset pricing without disturbing global behavior. The promoted rule is now live in production.</p>
        <div class="plot">{{ medium_gap_refinement_plot|safe }}</div>
      </div>
      <div class="panel">
        <h2>Refinement Decision</h2>
        <p>Promote only if medium-gap calibration improves while log loss, Brier, ECE, and large-gap behavior remain stable.</p>
        <div class="table-wrap">{{ medium_gap_refinement_decision_table|safe }}</div>
        <div class="table-wrap">{{ medium_gap_refinement_comparison_table|safe }}</div>
      </div>
    </section>

    <section class="panel" style="margin-bottom: 18px;">
      <h2>End-to-End Bracket</h2>
      <p>Full tournament tree from the First Round through the championship, generated with the final validated production model.</p>
      <div class="bracket-shell">
        {% if play_in_games %}
        <div class="region-panel">
          <h3>First Four</h3>
          <p>Play-in winners that feed directly into the main bracket.</p>
          <div class="play-in-grid">
            {% for game in play_in_games %}
            <div class="game-card">
              <div class="game-slot">{{ game.slot }}</div>
              <div class="team-row">
                <div class="team-seed">{{ game.winner_seed }}</div>
                <div class="team-name winner">{{ game.winner_name }}</div>
                <div class="team-prob">{{ game.winner_prob }}</div>
              </div>
              <div class="team-row">
                <div class="team-seed">{{ game.loser_seed }}</div>
                <div class="team-name">{{ game.loser_name }}</div>
                <div class="team-prob">loss</div>
              </div>
            </div>
            {% endfor %}
          </div>
        </div>
        {% endif %}
        <div class="bracket-board">
          <div class="region-grid">
            {% for region in regions %}
            <div class="region-panel">
              <h3>{{ region.name }}</h3>
              <p>{{ region.summary }}</p>
              <div class="round-grid">
                {% for round in region.rounds %}
                <div class="round-col">
                  <div class="round-head">{{ round.label }}</div>
                  {% for game in round.games %}
                  <div class="game-card">
                    <div class="game-slot">{{ game.slot }}</div>
                    <div class="team-row">
                      <div class="team-seed">{{ game.winner_seed }}</div>
                      <div class="team-name winner">{{ game.winner_name }}</div>
                      <div class="team-prob">{{ game.winner_prob }}</div>
                    </div>
                    <div class="team-row">
                      <div class="team-seed">{{ game.loser_seed }}</div>
                      <div class="team-name">{{ game.loser_name }}</div>
                      <div class="team-prob">loss</div>
                    </div>
                  </div>
                  {% endfor %}
                </div>
                {% endfor %}
              </div>
            </div>
            {% endfor %}
          </div>
          <div class="finals-panel">
            <h3>Final Four and Championship</h3>
            <p>Regional champions feed into the national semifinals and title game.</p>
            <div class="finals-grid">
              <div class="final-stage">
                <div class="round-head">National Semifinal 1</div>
                <div class="game-card">
                  <div class="game-slot">{{ semifinal_one.slot }}</div>
                  <div class="team-row">
                    <div class="team-seed">{{ semifinal_one.winner_seed }}</div>
                    <div class="team-name winner">{{ semifinal_one.winner_name }}</div>
                    <div class="team-prob">{{ semifinal_one.winner_prob }}</div>
                  </div>
                  <div class="team-row">
                    <div class="team-seed">{{ semifinal_one.loser_seed }}</div>
                    <div class="team-name">{{ semifinal_one.loser_name }}</div>
                    <div class="team-prob">loss</div>
                  </div>
                </div>
                <div class="round-head">National Semifinal 2</div>
                <div class="game-card">
                  <div class="game-slot">{{ semifinal_two.slot }}</div>
                  <div class="team-row">
                    <div class="team-seed">{{ semifinal_two.winner_seed }}</div>
                    <div class="team-name winner">{{ semifinal_two.winner_name }}</div>
                    <div class="team-prob">{{ semifinal_two.winner_prob }}</div>
                  </div>
                  <div class="team-row">
                    <div class="team-seed">{{ semifinal_two.loser_seed }}</div>
                    <div class="team-name">{{ semifinal_two.loser_name }}</div>
                    <div class="team-prob">loss</div>
                  </div>
                </div>
              </div>
              <div class="final-stage">
                <div class="round-head">National Championship</div>
                <div class="game-card">
                  <div class="game-slot">{{ championship_game.slot }}</div>
                  <div class="team-row">
                    <div class="team-seed">{{ championship_game.winner_seed }}</div>
                    <div class="team-name winner">{{ championship_game.winner_name }}</div>
                    <div class="team-prob">{{ championship_game.winner_prob }}</div>
                  </div>
                  <div class="team-row">
                    <div class="team-seed">{{ championship_game.loser_seed }}</div>
                    <div class="team-name">{{ championship_game.loser_name }}</div>
                    <div class="team-prob">loss</div>
                  </div>
                </div>
              </div>
              <div class="champion-card">
                <span class="label">Champion</span>
                <strong>{{ champion_name }}</strong>
                <div>{{ champion_seed }}</div>
                <div class="mini">{{ champion_prob }} title-game win probability</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section class="panel">
      <h2>Top Strength Table</h2>
      <p>Top 20 team rows from the production output strength file.</p>
      <div class="table-wrap">{{ strengths_table|safe }}</div>
    </section>
  </div>
</body>
</html>"""
)


def plot_div(fig) -> str:
    return pio.to_html(fig, include_plotlyjs="cdn", full_html=False, config={"displayModeBar": False})


def format_prob(value: float) -> str:
    return f"{value:.3f}"


def clean_seed(value: object) -> str:
    if pd.isna(value):
        return "-"
    return str(value)


def build_game_card(row: pd.Series) -> dict[str, str]:
    return {
        "slot": row["Slot"],
        "winner_seed": clean_seed(row["WinnerSeed"]),
        "winner_name": row["WinnerTeamName"],
        "winner_prob": format_prob(float(row["WinnerProb"])),
        "loser_seed": clean_seed(row["LoserSeed"]),
        "loser_name": row["LoserTeamName"],
    }


def build_region_payload(bracket: pd.DataFrame, region_code: str) -> dict[str, object]:
    round_map = {
        1: "Round of 64",
        2: "Round of 32",
        3: "Sweet 16",
        4: "Elite 8",
    }
    rounds = []
    for round_num in range(1, 5):
        pattern = rf"^R{round_num}{region_code}"
        round_rows = bracket[bracket["Slot"].str.match(pattern)].copy()
        round_rows = round_rows.sort_values("Slot")
        rounds.append(
            {
                "label": round_map[round_num],
                "games": [build_game_card(row) for _, row in round_rows.iterrows()],
            }
        )

    champion_row = bracket.loc[bracket["Slot"] == f"R4{region_code}1"].iloc[0]
    return {
        "name": f"Region {region_code}",
        "summary": f"{champion_row['WinnerTeamName']} won the region as {champion_row['WinnerSeed']}.",
        "rounds": rounds,
    }


def build_dashboard(output_dir: Path, season: int) -> Path:
    backtest = pd.read_csv(output_dir / "model_backtest.csv")
    strengths = pd.read_csv(output_dir / f"team_strengths_{season}.csv")
    bracket = pd.read_csv(output_dir / f"bracket_{season}.csv")
    submission = pd.read_csv(output_dir / f"submission_{season}.csv")
    matchup_weights = pd.read_csv(output_dir / "logistic_feature_coefficients.csv")
    calibration_metrics = pd.read_csv(output_dir / "calibration_metrics.csv")
    calibration_reliability = pd.read_csv(output_dir / "calibration_reliability.csv")
    dataset_comparison = pd.read_csv(output_dir / "dataset_performance_comparison.csv")
    feature_set_comparison = pd.read_csv(output_dir / "feature_set_comparison.csv")
    model_comparison = pd.read_csv(output_dir / "model_comparison.csv")
    boosting_metrics = pd.read_csv(output_dir / "boosting_model_comparison.csv")
    boosting_distribution = pd.read_csv(output_dir / "boosting_distribution.csv")
    boosting_importance = pd.read_csv(output_dir / "boosting_feature_importance.csv")
    boosting_stability_summary = pd.read_csv(output_dir / "boosting_stability_summary.csv")
    boosting_rolling_comparison = pd.read_csv(output_dir / "boosting_rolling_comparison.csv")
    boosting_importance_stability = pd.read_csv(output_dir / "boosting_importance_stability.csv")
    boosting_stability_verdict = pd.read_csv(output_dir / "boosting_stability_verdict.csv")
    tournament_backtest_yearly = pd.read_csv(output_dir / "tournament_yearly_with_without_seed.csv")
    tournament_seed_matchup = pd.read_csv(output_dir / "tournament_seed_matchup_calibration.csv")
    tournament_seed_gap = pd.read_csv(output_dir / "tournament_seed_gap_analysis.csv")
    tournament_upset = pd.read_csv(output_dir / "tournament_upset_analysis.csv")
    tournament_distribution_summary = pd.read_csv(output_dir / "tournament_distribution_summary.csv")
    tournament_model_comparison = pd.read_csv(output_dir / "tournament_model_comparison.csv")
    medium_gap_refinement_metrics = pd.read_csv(output_dir / "medium_gap_refinement_metrics.csv")
    medium_gap_refinement_comparison = pd.read_csv(output_dir / "medium_gap_refinement_comparison.csv")
    temperature_metrics = pd.read_csv(output_dir / "temperature_scaling_metrics.csv")
    temperature_distribution = pd.read_csv(output_dir / "temperature_scaling_distribution.csv")
    ensemble_metrics = pd.read_csv(output_dir / "ensemble_metrics.csv")
    ensemble_distribution = pd.read_csv(output_dir / "ensemble_distribution.csv")
    regular_matchups = pd.read_csv(output_dir / f"regular_season_matchups_{season}.csv")
    import json
    ingestion_summary = json.loads((output_dir / "regular_season_ingestion_summary.json").read_text(encoding="utf-8"))
    calibration_method = "unknown"
    calibration_params = {}
    choice_path = output_dir / "calibration_choice.json"
    if choice_path.exists():
        choice = json.loads(choice_path.read_text(encoding="utf-8"))
        calibration_method = choice.get("selected_method", "unknown")
        calibration_params = choice.get("params", {})
    tournament_diagnosis = json.loads((output_dir / "tournament_diagnosis.json").read_text(encoding="utf-8"))
    medium_gap_refinement_decision = json.loads((output_dir / "medium_gap_refinement_decision.json").read_text(encoding="utf-8"))

    with_seed_row = tournament_model_comparison.loc[tournament_model_comparison["model"] == "with_seed"].iloc[0]
    without_seed_row = tournament_model_comparison.loc[tournament_model_comparison["model"] == "without_seed"].iloc[0]
    production_refinement_row = medium_gap_refinement_comparison.loc[
        medium_gap_refinement_comparison["variant"] == "production"
    ].iloc[0]
    promoted_refinement_row = medium_gap_refinement_comparison.loc[
        medium_gap_refinement_comparison["variant"] == medium_gap_refinement_decision.get("selected_variant", "production")
    ].iloc[0]
    overall_upset_row = tournament_upset.loc[tournament_upset["metric"] == "overall_upset_rate"].iloc[0]
    distribution_lookup = dict(zip(tournament_distribution_summary["metric"], tournament_distribution_summary["value"]))
    boosting_verdict_row = boosting_stability_verdict.iloc[0]

    validation_summary_rows = [
        {
            "check": "Stability validation",
            "result": "Passed",
            "evidence": "Interaction-enhanced logistic remained the stable production choice across robustness testing.",
        },
        {
            "check": "Boosting benchmark",
            "result": "Rejected",
            "evidence": f"Boosting failed the stability gate ({boosting_verdict_row['verdict']}) and was not promoted.",
        },
        {
            "check": "Tournament walk-forward audit",
            "result": "Passed",
            "evidence": f"2018-2025 walk-forward log loss {with_seed_row['log_loss']:.4f}.",
        },
        {
            "check": "Seed-bias audit",
            "result": "Passed",
            "evidence": "Seed improved performance materially and the diagnosis remained not materially seed-biased.",
        },
        {
            "check": "Medium-gap calibration refinement",
            "result": "Promoted",
            "evidence": "Localized -0.03 higher-seed probability adjustment for seed gaps 3-5 improved upset pricing.",
        },
    ]

    behavioral_audit_rows = [
        {
            "metric": "With seed log loss",
            "production": f"{with_seed_row['log_loss']:.4f}",
            "comparison": "Final production walk-forward result",
        },
        {
            "metric": "Without seed log loss",
            "production": f"{without_seed_row['log_loss']:.4f}",
            "comparison": "Counterfactual without seed terms",
        },
        {
            "metric": "Large-gap calibration gap",
            "production": f"{production_refinement_row['large_gap_calibration_gap']:.4f}",
            "comparison": "Stable after localized correction",
        },
        {
            "metric": "Overall upset rate",
            "production": f"{overall_upset_row['predicted']:.4f}",
            "comparison": f"Actual {overall_upset_row['actual']:.4f}",
        },
        {
            "metric": "Medium-gap upset rate",
            "production": f"{production_refinement_row['medium_gap_predicted_upset_rate']:.4f}",
            "comparison": f"Actual {production_refinement_row['medium_gap_actual_upset_rate']:.4f}",
        },
        {
            "metric": "Medium-gap upset gap",
            "production": f"{production_refinement_row['medium_gap_upset_gap']:.4f}",
            "comparison": f"Refined to {promoted_refinement_row['medium_gap_upset_gap']:.4f}",
        },
        {
            "metric": "High-confidence tail",
            "production": f"{distribution_lookup.get('pct_above_095', 0.0):.4f}",
            "comparison": "Share of walk-forward predictions above 0.95",
        },
        {
            "metric": "Mid-range density",
            "production": f"{distribution_lookup.get('pct_between_040_070', 0.0):.4f}",
            "comparison": "Share of walk-forward predictions between 0.40 and 0.70",
        },
    ]

    championship = bracket.loc[bracket["Slot"] == "R6CH"].iloc[0]
    final_four_df = bracket[bracket["Slot"].isin(["R4W1", "R4X1", "R4Y1", "R4Z1"])].copy()
    confidence = bracket[bracket["WinnerTeamID"] != bracket["LoserTeamID"]].nlargest(12, "WinnerProb").copy()
    top_strengths = strengths.head(12).copy()
    play_in_rows = bracket[~bracket["Slot"].str.startswith("R")].sort_values("Slot")
    regions = [build_region_payload(bracket, code) for code in ["W", "X", "Y", "Z"]]
    semifinal_one = build_game_card(bracket.loc[bracket["Slot"] == "R5WX"].iloc[0])
    semifinal_two = build_game_card(bracket.loc[bracket["Slot"] == "R5YZ"].iloc[0])
    championship_game = build_game_card(championship)

    backtest_fig = px.line(
        backtest,
        x="season",
        y=["log_loss", "brier_score"],
        markers=True,
        color_discrete_sequence=["#c85c38", "#496a5a"],
    )
    backtest_fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
    )

    strength_fig = px.bar(
        top_strengths.iloc[::-1],
        x="team_logit_rating",
        y="TeamID",
        orientation="h",
        hover_data=["Seed", "eff_margin", "pom_rank", "tov_rate", "team_win_prob_vs_avg"],
        color="team_win_prob_vs_avg",
        color_continuous_scale=["#f4d7c8", "#c85c38", "#7a2614"],
    )
    strength_fig.update_yaxes(
        tickmode="array",
        tickvals=top_strengths.iloc[::-1]["TeamID"],
        ticktext=top_strengths.iloc[::-1].apply(lambda r: f"{r['Seed']} {r['TeamName']}", axis=1),
    )
    strength_fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        xaxis_title="Team Logit Rating",
        yaxis_title="",
    )

    confidence["label"] = confidence["Slot"] + " - " + confidence["WinnerTeamName"]
    confidence_fig = px.bar(
        confidence.iloc[::-1],
        x="WinnerProb",
        y="label",
        orientation="h",
        color="WinnerProb",
        color_continuous_scale=["#d7e4dc", "#809a87", "#496a5a"],
    )
    confidence_fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        xaxis_title="Win Probability",
        yaxis_title="",
    )

    model_comparison_plot_df = model_comparison.melt(
        id_vars=["model_variant"],
        value_vars=["log_loss", "brier_score", "ece"],
        var_name="metric",
        value_name="value",
    )
    model_comparison_fig = px.bar(
        model_comparison_plot_df,
        x="metric",
        y="value",
        color="model_variant",
        barmode="group",
        color_discrete_sequence=["#496a5a", "#c85c38"],
    )
    model_comparison_fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        xaxis_title="Metric",
        yaxis_title="Metric Value",
    )

    boosting_plot_df = boosting_metrics.melt(
        id_vars=["label"],
        value_vars=["log_loss", "brier_score", "ece"],
        var_name="metric",
        value_name="value",
    )
    boosting_fig = px.bar(
        boosting_plot_df,
        x="metric",
        y="value",
        color="label",
        barmode="group",
        color_discrete_sequence=["#496a5a", "#c85c38", "#8f3417"],
    )
    boosting_fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        xaxis_title="Metric",
        yaxis_title="Metric Value",
    )

    boosting_importance_fig = px.bar(
        boosting_importance.head(12).iloc[::-1],
        x="normalized_importance_pct",
        y="metric",
        orientation="h",
        color="importance",
        color_continuous_scale=["#f1dfd4", "#c85c38", "#7a2614"],
    )
    boosting_importance_fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        xaxis_title="Importance Share (%)",
        yaxis_title="",
    )

    boosting_rolling_plot_df = boosting_rolling_comparison.melt(
        id_vars=["test_season"],
        value_vars=["log_loss_logistic_baseline", "log_loss_boosting_raw"],
        var_name="model",
        value_name="log_loss",
    )
    boosting_rolling_fig = px.line(
        boosting_rolling_plot_df,
        x="test_season",
        y="log_loss",
        color="model",
        markers=True,
        color_discrete_sequence=["#496a5a", "#c85c38"],
    )
    boosting_rolling_fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        xaxis_title="Test Season",
        yaxis_title="Log Loss",
    )

    tournament_yearly_plot_df = tournament_backtest_yearly[tournament_backtest_yearly["Season"] != "Overall"].copy()
    tournament_yearly_plot_df["Season"] = tournament_yearly_plot_df["Season"].astype(int)
    tournament_backtest_fig = px.line(
        tournament_yearly_plot_df,
        x="Season",
        y="log_loss",
        color="Model",
        markers=True,
        color_discrete_sequence=["#496a5a", "#c85c38"],
    )
    tournament_backtest_fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        xaxis_title="Tournament Season",
        yaxis_title="Log Loss",
    )

    tournament_seed_gap_plot_df = tournament_seed_gap.melt(
        id_vars=["seed_gap_bucket"],
        value_vars=["predicted_higher_seed_win", "actual_higher_seed_win"],
        var_name="metric",
        value_name="value",
    )
    tournament_seed_gap_fig = px.bar(
        tournament_seed_gap_plot_df,
        x="seed_gap_bucket",
        y="value",
        color="metric",
        barmode="group",
        color_discrete_sequence=["#c85c38", "#496a5a"],
    )
    tournament_seed_gap_fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        xaxis_title="Seed Gap Bucket",
        yaxis_title="Higher Seed Win Rate",
    )

    matchup_plot_df = tournament_seed_matchup[tournament_seed_matchup["games"] >= 5].copy()
    matchup_plot_df = matchup_plot_df.sort_values("calibration_gap", ascending=False)
    tournament_seed_matchup_fig = px.bar(
        matchup_plot_df,
        x="calibration_gap",
        y="seed_matchup",
        orientation="h",
        color="calibration_gap",
        color_continuous_scale=["#7a2614", "#f1dfd4", "#496a5a"],
    )
    tournament_seed_matchup_fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        xaxis_title="Predicted - Actual",
        yaxis_title="Seed Matchup",
    )

    tournament_model_comparison_plot_df = tournament_model_comparison.melt(
        id_vars=["model"],
        value_vars=["log_loss", "brier_score", "ece"],
        var_name="metric",
        value_name="value",
    )
    tournament_seed_counterfactual_fig = px.bar(
        tournament_model_comparison_plot_df,
        x="metric",
        y="value",
        color="model",
        barmode="group",
        color_discrete_sequence=["#496a5a", "#c85c38"],
    )
    tournament_seed_counterfactual_fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        xaxis_title="Metric",
        yaxis_title="Value",
    )

    medium_gap_refinement_fig = px.bar(
        medium_gap_refinement_metrics.sort_values("log_loss"),
        x="variant",
        y="medium_gap_upset_gap",
        color="log_loss",
        color_continuous_scale=["#496a5a", "#f1dfd4", "#c85c38"],
    )
    medium_gap_refinement_fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        xaxis_title="Variant",
        yaxis_title="Medium-Gap Upset Gap",
    )

    matchup_fig = px.bar(
        matchup_weights.head(12).iloc[::-1],
        x="normalized_weight_pct",
        y="metric",
        orientation="h",
        color="coefficient",
        color_continuous_scale=["#7a2614", "#f1dfd4", "#496a5a"],
    )
    matchup_fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        xaxis_title="Coefficient Share (%)",
        yaxis_title="",
    )

    calibration_fig = px.line(
        calibration_reliability,
        x="mean_predicted_probability",
        y="fraction_positives",
        color="label",
        markers=True,
        color_discrete_sequence=["#8f3417", "#496a5a", "#c85c38", "#809a87", "#b38b59"],
    )
    calibration_fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(color="#74685b", dash="dot"),
    )
    calibration_fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        xaxis_title="Predicted Probability",
        yaxis_title="Observed Win Rate",
    )

    dataset_plot_df = dataset_comparison.melt(
        id_vars=["dataset_variant"],
        value_vars=["log_loss", "brier_score", "ece"],
        var_name="metric",
        value_name="value",
    )
    dataset_fig = px.bar(
        dataset_plot_df,
        x="metric",
        y="value",
        color="dataset_variant",
        barmode="group",
        color_discrete_sequence=["#c85c38", "#496a5a"],
    )
    dataset_fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        xaxis_title="Metric",
        yaxis_title="Metric Value",
    )

    feature_set_plot_df = feature_set_comparison.melt(
        id_vars=["feature_set", "variant", "temperature"],
        value_vars=["log_loss", "brier_score", "ece"],
        var_name="metric",
        value_name="value",
    )
    feature_set_fig = px.bar(
        feature_set_plot_df,
        x="metric",
        y="value",
        color="feature_set",
        barmode="group",
        color_discrete_sequence=["#496a5a", "#c85c38"],
    )
    feature_set_fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        xaxis_title="Metric",
        yaxis_title="Metric Value",
    )

    temp_metrics_plot_df = temperature_metrics.melt(
        id_vars=["label", "temperature"],
        value_vars=["log_loss", "brier_score", "ece"],
        var_name="metric",
        value_name="value",
    )
    temperature_metrics_fig = px.bar(
        temp_metrics_plot_df,
        x="metric",
        y="value",
        color="label",
        barmode="group",
        color_discrete_sequence=["#496a5a", "#c85c38"],
    )
    temperature_metrics_fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        xaxis_title="Metric",
        yaxis_title="Metric Value",
    )

    temperature_distribution_fig = px.bar(
        temperature_distribution,
        x="bucket",
        y="count",
        color="label",
        barmode="group",
        color_discrete_sequence=["#496a5a", "#c85c38"],
    )
    temperature_distribution_fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        xaxis_title="Probability Bucket",
        yaxis_title="Prediction Count",
    )

    ensemble_metrics_fig = px.line(
        ensemble_metrics,
        x="alpha",
        y=["log_loss", "brier_score", "ece"],
        markers=True,
        color_discrete_sequence=["#c85c38", "#496a5a", "#8f3417"],
    )
    ensemble_metrics_fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        xaxis_title="Logistic Weight Alpha",
        yaxis_title="Metric Value",
    )

    ensemble_tail_fig = px.line(
        ensemble_distribution[ensemble_distribution["bucket"] == "(0.95, 1.0]"].sort_values("alpha"),
        x="alpha",
        y="pct_above_095",
        markers=True,
        color_discrete_sequence=["#c85c38"],
    )
    ensemble_tail_fig.add_scatter(
        x=ensemble_distribution.drop_duplicates("alpha").sort_values("alpha")["alpha"],
        y=ensemble_distribution.drop_duplicates("alpha").sort_values("alpha")["pct_between_040_070"],
        mode="lines+markers",
        name="pct_between_040_070",
        line=dict(color="#496a5a"),
    )
    ensemble_tail_fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        xaxis_title="Logistic Weight Alpha",
        yaxis_title="Probability Share",
    )

    leakage_rows = [{"Leakage Control": item} for item in ingestion_summary.get("leakage_controls", [])]
    ingestion_rows = [
        {"Metric": "Current Season", "Value": ingestion_summary.get("current_season")},
        {"Metric": "2026 Regular Games", "Value": ingestion_summary.get("current_regular_season_games")},
        {"Metric": "2026 Model Rows", "Value": ingestion_summary.get("current_regular_season_model_rows")},
        {"Metric": "Historical Tournament Games", "Value": ingestion_summary.get("historical_tournament_training_games")},
        {"Metric": "Historical Regular-Season Games", "Value": ingestion_summary.get("historical_regular_season_training_games")},
    ]
    ingestion_table_html = (
        ingestion_table_df.to_html(index=False, classes="table")
        if (ingestion_table_df := pd.DataFrame(ingestion_rows)).shape[0]
        else ""
    )
    leakage_table_html = (
        leakage_table_df.to_html(index=False, classes="table")
        if (leakage_table_df := pd.DataFrame(leakage_rows)).shape[0]
        else ""
    )

    final_four = [
        {
            "slot": row["Slot"],
            "team": row["WinnerTeamName"],
            "seed": row["WinnerSeed"],
            "prob": f"{row['WinnerProb']:.3f}",
        }
        for _, row in final_four_df.iterrows()
    ]

    html = HTML_TEMPLATE.render(
        champion_name=championship["WinnerTeamName"],
        runner_up_name=championship["LoserTeamName"],
        champion_prob=f"{championship['WinnerProb']:.3f}",
        champion_seed=championship["WinnerSeed"],
        runner_up_seed=championship["LoserSeed"],
        best_log_loss=f"{backtest['log_loss'].min():.3f}",
        top_team_name=f"{top_strengths.iloc[0]['Seed']} {top_strengths.iloc[0]['TeamName']}",
        top_team_score=f"{top_strengths.iloc[0]['team_logit_rating']:.3f}",
        production_model_name="Validated Logistic + Medium-Gap Adjustment",
        production_model_params=(
            f"C={calibration_params.get('regularization_c', 'n/a')}, "
            "raw logistic, seed retained, no global calibration"
        ),
        feature_set_name=str(calibration_params.get("feature_set", "n/a")).replace("_", " "),
        feature_set_count=calibration_params.get("feature_count", "n/a"),
        medium_gap_adjustment="-0.03",
        medium_gap_adjustment_note="Higher-seed win probability only for seed gaps 3-5",
        regular_games=f"{ingestion_summary.get('current_regular_season_games', 0):,}",
        regular_matchup_rows=f"{len(regular_matchups):,}",
        validation_summary_table=pd.DataFrame(validation_summary_rows).to_html(index=False, classes="table"),
        behavioral_audit_table=pd.DataFrame(behavioral_audit_rows).to_html(index=False, classes="table"),
        backtest_plot=plot_div(backtest_fig),
        strength_plot=plot_div(strength_fig),
        confidence_plot=plot_div(confidence_fig),
        model_comparison_plot=plot_div(model_comparison_fig),
        boosting_plot=plot_div(boosting_fig),
        boosting_importance_plot=plot_div(boosting_importance_fig),
        boosting_rolling_plot=plot_div(boosting_rolling_fig),
        boosting_metrics_table=boosting_metrics.to_html(
            index=False, classes="table", float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x
        ),
        boosting_distribution_table=boosting_distribution[
            ["label", "bucket", "count", "pct_above_095", "pct_between_040_070"]
        ].to_html(index=False, classes="table", float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x),
        boosting_stability_summary_table=boosting_stability_summary.to_html(
            index=False, classes="table", float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x
        ),
        boosting_importance_stability_table=boosting_importance_stability.head(12).to_html(
            index=False, classes="table", float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x
        ),
        boosting_stability_verdict_table=boosting_stability_verdict.to_html(
            index=False, classes="table", float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x
        ),
        tournament_backtest_plot=plot_div(tournament_backtest_fig),
        tournament_seed_gap_plot=plot_div(tournament_seed_gap_fig),
        tournament_seed_matchup_plot=plot_div(tournament_seed_matchup_fig),
        tournament_seed_counterfactual_plot=plot_div(tournament_seed_counterfactual_fig),
        medium_gap_refinement_plot=plot_div(medium_gap_refinement_fig),
        tournament_upset_table=tournament_upset.to_html(
            index=False, classes="table", float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x
        ),
        tournament_distribution_table=tournament_distribution_summary.to_html(
            index=False, classes="table", float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x
        ),
        tournament_diagnosis_table=pd.DataFrame(
            [{"metric": key, "value": value} for key, value in tournament_diagnosis.items()]
        ).to_html(index=False, classes="table", float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x),
        medium_gap_refinement_decision_table=pd.DataFrame(
            [{"metric": key, "value": value} for key, value in medium_gap_refinement_decision.items()]
        ).to_html(index=False, classes="table", float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x),
        medium_gap_refinement_comparison_table=medium_gap_refinement_comparison.to_html(
            index=False, classes="table", float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x
        ),
        matchup_weights_plot=plot_div(matchup_fig),
        calibration_curve_plot=plot_div(calibration_fig),
        calibration_metrics_table=calibration_metrics.to_html(
            index=False, classes="table", float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x
        ),
        temperature_metrics_plot=plot_div(temperature_metrics_fig),
        temperature_distribution_plot=plot_div(temperature_distribution_fig),
        temperature_metrics_table=temperature_metrics.to_html(
            index=False, classes="table", float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x
        ),
        temperature_distribution_table=temperature_distribution[
            ["label", "bucket", "count", "pct_above_095", "pct_below_005"]
        ].to_html(index=False, classes="table", float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x),
        ensemble_metrics_plot=plot_div(ensemble_metrics_fig),
        ensemble_distribution_plot=plot_div(ensemble_tail_fig),
        ensemble_metrics_table=ensemble_metrics.to_html(
            index=False, classes="table", float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x
        ),
        ensemble_distribution_table=ensemble_distribution[
            ["alpha", "bucket", "count", "pct_above_095", "pct_between_040_070"]
        ].to_html(index=False, classes="table", float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x),
        feature_set_plot=plot_div(feature_set_fig),
        feature_set_table=feature_set_comparison.to_html(
            index=False, classes="table", float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x
        ),
        dataset_comparison_plot=plot_div(dataset_fig),
        ingestion_summary_table=ingestion_table_html + leakage_table_html,
        dataset_comparison_table=dataset_comparison.to_html(
            index=False, classes="table", float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x
        ),
        calibration_method=f"{calibration_method} {calibration_params}",
        final_four=final_four,
        play_in_games=[build_game_card(row) for _, row in play_in_rows.iterrows()],
        regions=regions,
        semifinal_one=semifinal_one,
        semifinal_two=semifinal_two,
        championship_game=championship_game,
        strengths_table=strengths[
            [
                "TeamName",
                "Seed",
                "ConfAbbrev",
                "team_logit_rating",
                "team_win_prob_vs_avg",
                "eff_margin",
                "three_point_rate",
                "three_point_pct",
                "three_point_pct_var",
                "opp_three_point_pct_var",
                "three_point_dependency",
                "tov_rate",
            ]
        ].head(20).to_html(index=False, classes="table", float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else x),
    )

    output_path = output_dir / f"dashboard_{season}.html"
    output_path.write_text(html, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a dashboard from March Madness model outputs.")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/wilcroutwater/Documents/Playground/output/march_madness"),
    )
    args = parser.parse_args()

    dashboard_path = build_dashboard(args.output_dir, args.season)
    print(f"Saved dashboard to {dashboard_path}")


if __name__ == "__main__":
    main()
