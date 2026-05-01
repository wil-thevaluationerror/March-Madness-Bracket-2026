"""
TopStep Evaluation Bootstrap Simulator
=======================================

Answers the core question: given the observed WFO per-window P&L distribution,
what is the realistic probability of hitting +$3,000 before the trailing drawdown
reaches -$2,000?

Usage
-----
    # From the repo root:
    python -m backtest.eval_simulator

    # Or with custom WFO data (JSON produced by run-backtest.py --walk-forward):
    python -m backtest.eval_simulator --wfo-report output/backtest/wfo-<stamp>/wfo_report.json

    # Adjust simulation parameters:
    python -m backtest.eval_simulator --trials 50000 --target 3000 --drawdown-limit 2000

Algorithm
---------
Each simulated evaluation attempt is modelled as a 1-D random walk driven by the
observed per-window (20-trading-day) P&L distribution:

1. Draw a random per-window P&L from the bootstrap sample (with replacement).
2. Apply it to the running cumulative P&L.
3. Update the trailing equity peak: peak = max(peak, cumulative_pnl).
4. Compute current trailing drawdown: dd = cumulative_pnl - peak.
5. Stop if:
   - cumulative_pnl >= target  → PASS
   - dd <= -drawdown_limit     → FAIL
6. Repeat N=10,000 times.

The trailing drawdown (step 3-4) matches TopStep's actual rule: the limit is based
on the high-water mark of the account equity, not on the starting balance.  This
makes the problem harder than a fixed-barrier random walk because after making
$1,000 the model can only lose $1,000 more before failing.

"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Default WFO distribution — v10 per-window OOS P&L (12 windows, 20-day each)
# v10: 8 MES lots, holiday calendar, trailing-DD kill switch.
# Drawdown tiers reduce lot exposure in bad windows; actual scaling vs 2-lot
# baseline is ~2.3× aggregate (not the naive 4×).
# Update this when a new WFO run produces better estimates.
# ---------------------------------------------------------------------------
DEFAULT_WFO_PNL = [-541, 126, -250, 762, -549, 51, 32, 1787, -996, -551, 1571, 407]

# Baseline lot size for the default WFO distribution.
DEFAULT_BASE_LOTS = 8

# TopStep 50K Express evaluation parameters
DEFAULT_PROFIT_TARGET = 3_000.0
DEFAULT_DRAWDOWN_LIMIT = 2_000.0

# Monthly subscription cost.  This is a separate out-of-pocket expense —
# it does NOT reduce the $3,000 trading profit target or the funded-account
# payout.  It is reported as "expected subscription cost to pass" so you know
# how much you'll spend on the evaluation before getting funded.
DEFAULT_MONTHLY_FEE = 110.0
TRADING_DAYS_PER_MONTH = 20


def simulate_single_trial(
    pnl_distribution: list[float],
    profit_target: float,
    drawdown_limit: float,
    rng: random.Random,
) -> tuple[bool, int, float]:
    """
    Run one evaluation attempt.

    Returns
    -------
    (passed, windows_used, min_drawdown_seen)
    """
    cumulative = 0.0
    equity_peak = 0.0
    min_drawdown = 0.0
    windows = 0

    while True:
        # Draw a random window P&L from the bootstrap sample
        delta = rng.choice(pnl_distribution)
        cumulative += delta
        windows += 1

        # Advance trailing equity peak
        equity_peak = max(equity_peak, cumulative)

        # Compute trailing drawdown (always <= 0)
        drawdown = cumulative - equity_peak
        min_drawdown = min(min_drawdown, drawdown)

        if cumulative >= profit_target:
            return True, windows, min_drawdown
        if drawdown <= -drawdown_limit:
            return False, windows, min_drawdown


def run_simulation(
    pnl_distribution: list[float],
    n_trials: int,
    profit_target: float,
    drawdown_limit: float,
    monthly_fee: float = DEFAULT_MONTHLY_FEE,
    seed: int | None = None,
) -> dict:
    """
    Run N independent simulated evaluation attempts.

    Returns a dict with all metrics needed for the report.
    """
    rng = random.Random(seed)

    passes: list[int] = []    # windows to pass, for passing trials
    failures: list[int] = []  # windows to fail, for failing trials
    all_min_dd: list[float] = []
    pass_min_dd: list[float] = []

    for _ in range(n_trials):
        passed, windows, min_dd = simulate_single_trial(
            pnl_distribution, profit_target, drawdown_limit, rng
        )
        all_min_dd.append(min_dd)
        if passed:
            passes.append(windows)
            pass_min_dd.append(min_dd)
        else:
            failures.append(windows)

    n_pass = len(passes)
    n_fail = len(failures)
    pass_rate = n_pass / n_trials

    # Windows → trading days → months
    def windows_to_days(w: int) -> int:
        return w * 20  # each window = 20 OOS trading days

    def windows_to_months(w: float) -> float:
        return (w * 20) / TRADING_DAYS_PER_MONTH

    pass_windows = sorted(passes) if passes else [0]
    fail_windows = sorted(failures) if failures else [0]

    median_pass_windows = statistics.median(pass_windows) if passes else float("nan")
    p10_pass_windows = pass_windows[int(0.10 * len(pass_windows))] if passes else float("nan")
    p90_pass_windows = pass_windows[int(0.90 * len(pass_windows))] if passes else float("nan")

    # Subscription cost to pass.
    # The monthly fee is a separate out-of-pocket expense — it does NOT reduce
    # the $3,000 profit target or funded-account payouts.  Reported so you know
    # what you'll spend on subscriptions before getting funded.
    #
    # Expected subscription spend per attempt = median_months × monthly_fee.
    # Expected spend until first pass accounts for retries:
    #   E[attempts] = 1 / pass_rate  (geometric distribution)
    #   Expected total subscription = E[attempts] × cost_per_attempt
    median_pass_months = windows_to_months(median_pass_windows) if passes else float("nan")
    sub_cost_per_attempt = median_pass_months * monthly_fee
    expected_attempts = 1.0 / pass_rate if pass_rate > 0 else float("inf")
    expected_total_sub_cost = expected_attempts * sub_cost_per_attempt if pass_rate > 0 else float("inf")

    # Probability of reaching ≥$500 of drawdown during passing trials
    deep_dd_in_passes = sum(1 for dd in pass_min_dd if dd <= -500.0)
    p_deep_dd_given_pass = deep_dd_in_passes / n_pass if n_pass > 0 else float("nan")

    return {
        "input": {
            "n_trials": n_trials,
            "profit_target": profit_target,
            "drawdown_limit": drawdown_limit,
            "monthly_fee": monthly_fee,
            "pnl_distribution": pnl_distribution,
            "distribution_mean": statistics.mean(pnl_distribution),
            "distribution_stdev": statistics.stdev(pnl_distribution),
            "distribution_n": len(pnl_distribution),
        },
        "results": {
            "pass_rate": pass_rate,
            "n_pass": n_pass,
            "n_fail": n_fail,
            "median_windows_to_pass": median_pass_windows,
            "p10_windows_to_pass": p10_pass_windows,
            "p90_windows_to_pass": p90_pass_windows,
            "median_days_to_pass": windows_to_days(int(median_pass_windows)) if passes else None,
            "median_months_to_pass": round(median_pass_months, 1) if passes else None,
            "p_deep_drawdown_given_pass": p_deep_dd_given_pass,
            "expected_attempts_to_pass": round(expected_attempts, 1),
            "sub_cost_per_attempt_usd": round(sub_cost_per_attempt, 0) if passes else None,
            "expected_total_sub_cost_usd": round(expected_total_sub_cost, 0) if pass_rate > 0 else None,
            "worst_drawdown_pct": min(all_min_dd) / drawdown_limit,
        },
    }


def print_report(results: dict) -> None:
    inp = results["input"]
    res = results["results"]

    dist = inp["pnl_distribution"]
    monthly_fee = inp.get("monthly_fee", DEFAULT_MONTHLY_FEE)
    print("\n" + "=" * 62)
    print("  TopStep 50K Express — Evaluation Pass Probability Report")
    print("=" * 62)
    print(f"\n  Distribution : {len(dist)} WFO windows")
    print(f"  Mean / Std   : ${inp['distribution_mean']:.2f} / ${inp['distribution_stdev']:.2f} per window")
    print(f"  Trials       : {inp['n_trials']:,}")
    print(f"  Target       : +${inp['profit_target']:,.0f}  (trading P&L to unlock funded account)")
    print(f"  DD Limit     : -${inp['drawdown_limit']:,.0f} (trailing)")
    print(f"  Monthly sub  : ${monthly_fee:.0f}  (separate expense; does not reduce the profit target)")
    print()
    print(f"  ── Pass Probability ──────────────────────────────────")
    print(f"  P(pass)                : {res['pass_rate']:.1%}  ({res['n_pass']:,} / {inp['n_trials']:,})")
    print()
    print(f"  ── Time to Pass (given a win) ────────────────────────")
    print(f"  Median windows         : {res['median_windows_to_pass']:.0f}  "
          f"(~{res['median_months_to_pass']} months, {res['median_days_to_pass']} trading days)")
    print(f"  10th–90th percentile   : {res['p10_windows_to_pass']:.0f} – {res['p90_windows_to_pass']:.0f} windows")
    print()
    print(f"  ── Subscription Cost to Get Funded ──────────────────")
    print(f"  Sub cost / attempt     : ~${res['sub_cost_per_attempt_usd']:,.0f}  "
          f"(${monthly_fee:.0f}/month × {res['median_months_to_pass']} months)")
    print(f"  Expected attempts      : {res['expected_attempts_to_pass']:.1f}  (1 / pass_rate)")
    print(f"  Expected total sub cost: ~${res['expected_total_sub_cost_usd']:,.0f}  "
          f"(before first funded account)")
    print()
    print(f"  ── Drawdown Risk ─────────────────────────────────────")
    worst_dd_abs = abs(res["worst_drawdown_pct"] * inp["drawdown_limit"])
    print(f"  P(DD ≥ $500 | pass)    : {res['p_deep_drawdown_given_pass']:.1%}  "
          f"(drawdown hits -$500 even in winning attempts)")
    print(f"  Worst simulated DD     : -${worst_dd_abs:,.0f}  "
          f"(window-resolution artifact — the -$1,800 kill switch prevents this in practice)")
    print()
    print(f"  ── Verdict ───────────────────────────────────────────")
    pr = res["pass_rate"]
    sub_cost = res["expected_total_sub_cost_usd"] or float("inf")
    median_months = res["median_months_to_pass"] or float("inf")

    if pr >= 0.60:
        math_verdict = f"STRONG  ({pr:.1%} pass rate)"
    elif pr >= 0.55:
        math_verdict = f"GOOD  ({pr:.1%} pass rate)"
    elif pr >= 0.50:
        math_verdict = f"MARGINAL ({pr:.1%} pass rate)"
    else:
        math_verdict = f"NEGATIVE ({pr:.1%} pass rate)"

    if sub_cost <= 500:
        cost_verdict = f"LOW  (~${sub_cost:,.0f} expected in subscriptions)"
    elif sub_cost <= 1500:
        cost_verdict = f"MODERATE  (~${sub_cost:,.0f} expected in subscriptions)"
    elif sub_cost <= 3000:
        cost_verdict = f"HIGH  (~${sub_cost:,.0f} expected in subscriptions)"
    else:
        cost_verdict = f"VERY HIGH  (~${sub_cost:,.0f} expected in subscriptions)"

    print(f"  Mathematical edge      : {math_verdict}")
    print(f"  Subscription cost      : {cost_verdict}")
    print()

    if pr >= 0.55 and sub_cost <= 1500:
        print("  → GO: positive edge and manageable subscription cost.")
        print("    Confirm kill switch and holiday exclusion are active before live.")
    elif pr >= 0.55 and sub_cost <= 3000:
        print(f"  → CONDITIONAL GO: positive edge; budget ~{median_months:.0f} months of")
        print(f"    subscription (${sub_cost:,.0f} expected) before getting funded.")
        print("    Improving win rate further would compress the timeline.")
    elif pr >= 0.55:
        print("  → CAUTION: positive edge but long expected evaluation period.")
        print(f"    ~${sub_cost:,.0f} in subscriptions expected before funded.")
        print("    Consider lot-size increase to compress the timeline.")
    elif pr >= 0.50:
        print("  → MARGINAL: slightly positive edge but evaluation period is very long.")
        print("    Win rate or lot-size improvement recommended before committing.")
    else:
        print("  → NO-GO: model does not have a positive edge at current win rate.")
        print("    Win rate must exceed the commission-adjusted breakeven before going live.")
    print("=" * 62 + "\n")


def load_wfo_pnl(wfo_report_path: str) -> list[float]:
    """Extract per-window OOS P&L from a wfo_report.json file."""
    data = json.loads(Path(wfo_report_path).read_text(encoding="utf-8"))
    windows = data.get("per_window", [])
    if not windows:
        raise ValueError(f"No per_window data in {wfo_report_path}")
    return [float(w["total_pnl"]) for w in windows]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap Monte Carlo simulator for TopStep evaluation pass probability.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default run with built-in v7 WFO distribution:
  python -m backtest.eval_simulator

  # Use fresh WFO results:
  python -m backtest.eval_simulator --wfo-report output/backtest/wfo-<stamp>/wfo_report.json

  # Increase trials for tighter confidence intervals:
  python -m backtest.eval_simulator --trials 100000
        """,
    )
    parser.add_argument(
        "--wfo-report",
        help="Path to wfo_report.json from a completed walk-forward run. "
             "If omitted, uses the built-in v7 WFO distribution.",
    )
    parser.add_argument("--trials", type=int, default=10_000, help="Number of Monte Carlo trials (default 10000)")
    parser.add_argument("--target", type=float, default=DEFAULT_PROFIT_TARGET, help="Profit target in dollars")
    parser.add_argument("--drawdown-limit", type=float, default=DEFAULT_DRAWDOWN_LIMIT,
                        help="Trailing drawdown limit in dollars (absolute value, e.g. 2000)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility (default 42)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of human-readable report")
    parser.add_argument(
        "--monthly-fee",
        type=float,
        default=DEFAULT_MONTHLY_FEE,
        metavar="DOLLARS",
        help=(
            f"Monthly subscription cost in dollars (default ${DEFAULT_MONTHLY_FEE:.0f}). "
            "This is a separate out-of-pocket expense — it does NOT reduce the profit target "
            "or funded-account payouts. Reported as expected subscription cost to get funded."
        ),
    )
    parser.add_argument(
        "--lot-scale",
        type=float,
        default=1.0,
        metavar="FACTOR",
        help=(
            "Scale per-window P&L by FACTOR before simulation. "
            "Use to project economics at a different lot size without re-running WFO. "
            "E.g. --lot-scale 3.0 projects 6-lot performance from a 2-lot WFO baseline. "
            "NOTE: this is a naive linear scale that does NOT model how drawdown tiers "
            "reduce exposure in bad windows — actual results will be somewhat better in "
            "losing windows. Run a full WFO at the target lot size for accurate numbers."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.wfo_report:
        pnl_dist = load_wfo_pnl(args.wfo_report)
        print(f"[info] Loaded {len(pnl_dist)} windows from {args.wfo_report}")
    else:
        pnl_dist = DEFAULT_WFO_PNL
        print("[info] Using built-in v8 WFO distribution (12 windows).")

    if args.lot_scale != 1.0:
        pnl_dist = [p * args.lot_scale for p in pnl_dist]
        scaled_mean = sum(pnl_dist) / len(pnl_dist)
        print(
            f"[info] Applied lot-scale {args.lot_scale}x "
            f"(naive linear; ignores drawdown-tier capping in bad windows). "
            f"Scaled mean: ${scaled_mean:.2f}/window."
        )

    results = run_simulation(
        pnl_distribution=pnl_dist,
        n_trials=args.trials,
        profit_target=args.target,
        drawdown_limit=args.drawdown_limit,
        monthly_fee=args.monthly_fee,
        seed=args.seed,
    )

    if args.json:
        json.dump(results, sys.stdout, indent=2)
        print()
    else:
        print_report(results)


if __name__ == "__main__":
    main()
