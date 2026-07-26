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

from mlb_metrics import config, dfs_backtest


def _report(label: str, projected, actual) -> None:
    if len(projected) < 2:
        print(f"{label}: not enough scored rows ({len(projected)}) to report MAE/correlation.")
        return
    mae = (projected - actual).abs().mean()
    correlation = float(np.corrcoef(projected, actual)[0, 1])
    baseline_mae = (actual - actual.mean()).abs().mean()
    print(f"{label}: MAE {mae:.4f} vs. naive-baseline MAE {baseline_mae:.4f}, correlation {correlation:.3f} (n={len(projected)})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--season", type=int, default=config.SEASON_START.year)
    parser.add_argument("--days", type=int, default=20)
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


if __name__ == "__main__":
    main()
