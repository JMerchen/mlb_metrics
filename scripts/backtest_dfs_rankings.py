"""Validate dfs.py's DK-points projections against real historical
outcomes - see dfs_backtest.py's module docstring for the exact
methodology, no-lookahead discipline, and (for pitchers) the honesty
caveat about the FIP-based ER estimate not being validated against real
earned runs.

This is exploratory validation for a rankings page, not a live pick model
- the bar is "does this produce plausible, better-than-guessing
projections," reported honestly either way, same as every other backtest
in this project.

Needs data/raw/statcast_<season>.parquet (see scripts/wave.py).

Usage:
    python scripts/backtest_dfs_rankings.py
    python scripts/backtest_dfs_rankings.py --days 30
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from mlb_metrics import config, dfs_backtest, pitcher_matchup


def _report(label: str, projected, actual) -> None:
    if len(projected) < 2:
        print(f"{label}: not enough scored rows ({len(projected)}) to report MAE/correlation.")
        return
    mae = (projected - actual).abs().mean()
    # Actual_DK_Points_Modeled can come back as an object-dtype Series (real
    # Python floats, but pd.concat across many per-date frames doesn't
    # always coerce it back to float64) - np.corrcoef crashes on an
    # object-dtype array with a real dataset at scale, so cast explicitly
    # rather than relying on pandas' own dtype.
    correlation = float(np.corrcoef(projected.to_numpy(dtype=float), actual.to_numpy(dtype=float))[0, 1])
    baseline_mae = (actual - actual.mean()).abs().mean()
    print(f"{label}: MAE {mae:.4f} vs. naive-baseline MAE {baseline_mae:.4f}, correlation {correlation:.3f} (n={len(projected)})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--season", type=int, default=config.SEASON_START.year)
    parser.add_argument("--days", type=int, default=20)
    parser.add_argument(
        "--pitcher-matchup", action="store_true",
        help="Also report pitcher_matchup.py's opponent-offense-adjustment grid search "
             "(correlation/MAE per weight in config.PITCHER_MATCHUP_WEIGHT_GRID, vs. the "
             "weight=0.0 unadjusted baseline) - see pitcher_matchup.py's module docstring.",
    )
    args = parser.parse_args()

    result = dfs_backtest.backtest_dfs_projections(args.raw_dir, args.season, args.days)
    hitters = result["hitters"]
    pitchers = result["pitchers"]

    print(f"\n=== Hitters (hit-type + BB/HBP/RBI DK points - Runs/SB still excluded from both sides, not modeled) ===")
    if hitters.empty:
        print("No scorable hitter-days.")
    else:
        _report("DK_Points_Hitter", hitters["DK_Points_Hitter"], hitters["Actual_DK_Points_Modeled"])

    print(f"\n=== Pitchers (IP/K/BB/H - real; ER via FIP estimate on both sides, NOT validated against real ER) ===")
    if pitchers.empty:
        print("No scorable pitcher-days.")
    else:
        _report("Expected_IP vs. Actual_IP", pitchers["Expected_IP"], pitchers["Actual_IP"])
        _report("Expected_K vs. Actual_K", pitchers["Expected_K"], pitchers["Actual_K"])
        _report("Expected_BB vs. Actual_BB", pitchers["Expected_BB"], pitchers["Actual_BB"])
        _report("Expected_H_Allowed vs. Actual_H", pitchers["Expected_H_Allowed"], pitchers["Actual_H"])
        print()
        _report("DK_Points_Pitcher (combined, ER estimate on both sides)", pitchers["DK_Points_Pitcher"], pitchers["Actual_DK_Points_Modeled"])

    if args.pitcher_matchup:
        print(f"\n=== Opponent offense adjustment (pitcher_matchup.py) - Expected_H_Allowed/Expected_ER scaled by "
              f"opponent's team_bases_pg vs. league average, per weight in config.PITCHER_MATCHUP_WEIGHT_GRID ===")
        matchup_result = pitcher_matchup.backtest_pitcher_matchup_signal(args.raw_dir, args.season, args.days)
        n = matchup_result.get("n", 0)
        if n < 2 or "by_weight" not in matchup_result:
            print(f"Not enough scored pitcher-days (n={n}) to report a weight grid.")
        else:
            baseline = matchup_result["by_weight"].get(0.0, matchup_result["by_weight"].get(0))
            print(f"n={n} pitcher-days scored")
            for weight, metrics in matchup_result["by_weight"].items():
                marker = " (baseline, weight=0.0)" if weight == 0.0 else ""
                print(f"  weight={weight}: correlation {metrics['correlation']}, MAE {metrics['mae']:.4f}{marker}")
            if baseline is not None:
                best_weight = min(matchup_result["by_weight"], key=lambda w: matchup_result["by_weight"][w]["mae"])
                print(f"Lowest-MAE weight in grid: {best_weight} "
                      f"(baseline weight=0.0 MAE {baseline['mae']:.4f})")


if __name__ == "__main__":
    main()
