"""Scores backtest_pick_strategies_v3.py's output: does the real
production fix (avg_batting_order hard gate removed, replaced by
Expected_PA feeding hitters.assemble_hitters/
matchup.compute_matchup_hit_probability's trials count) actually close
the gap to c1 (the user's own WAVE-vs-PAVE method), on the same real,
no-lookahead, full-season methodology round 1 used?
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from scipy.stats import fisher_exact

from mlb_metrics import config, evaluation

STRATEGIES = {
    "backtest-v3-c1-wave-vs-pave": "c1: WAVE vs PAVE (user's manual method)",
    "backtest-v3-c3-new-heuristic": "c3: NEW production heuristic (Expected_PA, no hard lineup gate)",
}


def _score_slice(df: pd.DataFrame, model_version: str) -> dict:
    subset = df[df["model_version"] == model_version]
    top_picks = subset[subset["rank"] <= config.DAILY_PICK_MAX]
    resolved = evaluation.resolved_only(top_picks)
    n = len(resolved)
    successes = int(resolved["actual_hit"].sum()) if n else 0
    hit_rate = successes / n if n else float("nan")
    lo, hi = evaluation.wilson_confidence_interval(successes, n)

    day_success_rate = evaluation.top_k_hit_rate(subset, k=config.DAILY_PICK_MAX, require_all=True)
    longest = evaluation.longest_streak(df, metric="Game_Hit_Probability", max_picks=config.DAILY_PICK_MAX, model_version=model_version)

    return {"n": n, "successes": successes, "hit_rate": hit_rate, "wilson_ci_low": lo, "wilson_ci_high": hi,
            "day_success_rate": day_success_rate, "longest_streak": longest}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="data/backtest/pick_strategy_backtest_v3.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input, parse_dates=["date"])
    dates = sorted(df["date"].unique())
    midpoint = dates[len(dates) // 2]
    first_half = df[df["date"] < midpoint]
    second_half = df[df["date"] >= midpoint]

    scores = {mv: _score_slice(df, mv) for mv in STRATEGIES}
    c1_mv, c3_mv = list(STRATEGIES.keys())
    p_c1_vs_c3 = fisher_exact([
        [scores[c1_mv]["successes"], scores[c1_mv]["n"] - scores[c1_mv]["successes"]],
        [scores[c3_mv]["successes"], scores[c3_mv]["n"] - scores[c3_mv]["successes"]],
    ])[1]

    rows = []
    for model_version, label in STRATEGIES.items():
        full = _score_slice(df, model_version)
        first = _score_slice(first_half, model_version)
        second = _score_slice(second_half, model_version)
        rows.append({"strategy": label, "window": "full_season", **full})
        rows.append({"strategy": label, "window": "first_half", **first})
        rows.append({"strategy": label, "window": "second_half", **second})

    summary = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(summary.to_string(index=False))
    print(f"\nc1 vs c3 fisher exact p-value (full season): {p_c1_vs_c3:.4f}")

    output_path = os.path.join(os.path.dirname(args.input), "pick_strategy_backtest_v3_summary.csv")
    summary.to_csv(output_path, index=False)
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
