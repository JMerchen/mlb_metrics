"""Validate the Age Curves page's projection method against real history:
for a sample of past player-seasons with a known real next season, project
that next season using only comparable data available at or before that
test season's own year (see age_curve.backtest_projection_accuracy), and
report how close the projection came to what actually happened.

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
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from mlb_metrics import age_curve, config, lahman_data


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--test-year-start", type=int, default=2010)
    parser.add_argument("--test-year-end", type=int, default=2019)
    parser.add_argument("--sample-size", type=int, default=500, help="Random sample of test seasons (this loops row by row).")
    parser.add_argument("--k", type=int, default=config.AGE_CURVE_K_NEIGHBORS)
    parser.add_argument("--age-window", type=int, default=config.AGE_CURVE_AGE_WINDOW)
    parser.add_argument("--seed", type=int, default=0)
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

    result = age_curve.backtest_projection_accuracy(historical_seasons, test_seasons, k=args.k, age_window=args.age_window)
    resolved = result.dropna(subset=["projected_ops_mean"])

    if resolved.empty:
        print("No scorable test seasons produced a projection.")
        return

    mae = resolved["abs_error"].mean()
    correlation = float(np.corrcoef(resolved["projected_ops_mean"], resolved["actual_next_OPS"])[0, 1])
    baseline_mae = (resolved["actual_next_OPS"] - resolved["actual_next_OPS"].mean()).abs().mean()

    print(f"\nScored {len(resolved)}/{len(test_seasons)} test seasons (rest had no comparable with a resolvable next season).")
    print(f"Mean absolute error: {mae:.4f} OPS points")
    print(f"Correlation (projected vs. actual next-season OPS): {correlation:.3f}")
    print(f"Naive baseline (always guess the sample's mean next-season OPS) MAE: {baseline_mae:.4f}")


if __name__ == "__main__":
    main()
