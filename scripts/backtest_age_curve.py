"""Validate the Age Curves page's projection method against real history:
for a sample of past player-seasons with a known real next season, project
that next season using only comparable data available at or before that
test season's own year (see age_curve.backtest_projection_accuracy), and
report how close the projection came to what actually happened - one
report per metric in config.AGE_CURVE_METRICS (AVG/OBP/SLG/OPS), since
each is its own independent curve/comparable-search (see age_curve.py's
module docstring).

This is exploratory validation for a for-fun/insight page, not a live
prediction model - the bar is "does this produce plausible, better-than-
guessing projections," reported honestly either way (see this project's
established pattern: config.py's MATCHUP_PARK_FACTOR_WEIGHT/
GAME_PICK_SUSCEPTIBILITY_WEIGHT docstrings).

Needs data/raw/lahman/{people,batting}.parquet - run scripts/fetch_lahman.py
first.

Usage:
    python scripts/backtest_age_curve.py
    python scripts/backtest_age_curve.py --test-year-start 2010 --test-year-end 2019
    python scripts/backtest_age_curve.py --metric OPS
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from mlb_metrics import age_curve, config, lahman_data


def run_one_metric(historical_seasons: pd.DataFrame, test_seasons: pd.DataFrame, metric: str, k: int, age_window: int, sample_size: int) -> None:
    print(f"\n=== {metric} ===")

    result = age_curve.backtest_projection_accuracy(historical_seasons, test_seasons, metric=metric, k=k, age_window=age_window)
    resolved = result.dropna(subset=["projected_value_mean"])

    if resolved.empty:
        print("No scorable test seasons produced a projection.")
        return

    mae = resolved["abs_error"].mean()
    correlation = float(np.corrcoef(resolved["projected_value_mean"], resolved["actual_next_value"])[0, 1])
    baseline_mae = (resolved["actual_next_value"] - resolved["actual_next_value"].mean()).abs().mean()

    print(f"Scored {len(resolved)}/{len(test_seasons)} test seasons (rest had no comparable with a resolvable next season).")
    print(f"Mean absolute error: {mae:.4f} {metric} points")
    print(f"Correlation (projected vs. actual next-season {metric}): {correlation:.3f}")
    print(f"Naive baseline (always guess the sample's mean next-season {metric}) MAE: {baseline_mae:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--test-year-start", type=int, default=2010)
    parser.add_argument("--test-year-end", type=int, default=2019)
    parser.add_argument("--sample-size", type=int, default=500, help="Random sample of test seasons (this loops row by row).")
    parser.add_argument("--k", type=int, default=config.AGE_CURVE_K_NEIGHBORS)
    parser.add_argument("--age-window", type=int, default=config.AGE_CURVE_AGE_WINDOW)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metric", choices=config.AGE_CURVE_METRICS, help="Only backtest this one metric (default: all of them).")
    args = parser.parse_args()

    people = lahman_data.load_persisted_lahman_table(args.raw_dir, "people")
    batting = lahman_data.load_persisted_lahman_table(args.raw_dir, "batting")
    if people is None or batting is None:
        print(f"No persisted Lahman data in {args.raw_dir}/lahman/ - run scripts/fetch_lahman.py first.")
        return

    historical_seasons = age_curve.build_historical_seasons(batting, people)
    print(f"{len(historical_seasons)} qualified historical hitter-seasons (AB >= {config.AGE_CURVE_MIN_AB}).")

    in_range = historical_seasons[
        (historical_seasons["yearID"] >= args.test_year_start) & (historical_seasons["yearID"] <= args.test_year_end)
    ]
    test_seasons = in_range.sample(n=min(args.sample_size, len(in_range)), random_state=args.seed)
    print(f"Backtesting against {len(test_seasons)} sampled seasons from {args.test_year_start}-{args.test_year_end}...")

    metrics = [args.metric] if args.metric else config.AGE_CURVE_METRICS
    for metric in metrics:
        run_one_metric(historical_seasons, test_seasons, metric, args.k, args.age_window, args.sample_size)


if __name__ == "__main__":
    main()
