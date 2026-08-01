"""Validate nfl_dfs.py/nfl_dst.py's DK-points projections against real
historical outcomes - see nfl_dfs_backtest.py's module docstring for the
exact no-lookahead methodology.

This is exploratory validation, same bar as every other backtest in this
project: "does this beat naive baseline AND a simpler heuristic by a
real margin," reported honestly either way, not just when it passes.

Needs data/raw/nfl/{weekly,team_stats,schedules}_<season>.parquet for
every season in --seasons (see scripts/fetch_nfl_historical.py).

Usage:
    python scripts/backtest_nfl_dfs_rankings.py
    python scripts/backtest_nfl_dfs_rankings.py --seasons 2022 2023 2024 2025
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from mlb_metrics import config, nfl_data, nfl_dfs_backtest


def _report(label: str, projected: pd.Series, actual: pd.Series) -> None:
    if len(projected) < 2:
        print(f"{label}: not enough scored rows ({len(projected)}) to report MAE/correlation.")
        return
    mae = (projected - actual).abs().mean()
    correlation = float(np.corrcoef(projected.to_numpy(dtype=float), actual.to_numpy(dtype=float))[0, 1])
    baseline_mae = (actual - actual.mean()).abs().mean()
    print(f"{label}: MAE {mae:.4f} vs. naive-baseline MAE {baseline_mae:.4f}, correlation {correlation:.3f} (n={len(projected)})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default=config.NFL_RAW_DATA_DIR)
    parser.add_argument("--seasons", type=int, nargs="+", default=config.NFL_HISTORICAL_SEASONS)
    parser.add_argument("--weeks", type=int, default=None, help="Cap the backtest to the most recent N real weeks (default: every available week).")
    args = parser.parse_args()

    weekly_frames, team_stats_frames, schedule_frames = [], [], []
    for season in args.seasons:
        weekly = nfl_data.load_persisted_table(args.raw_dir, "weekly", season)
        team_stats = nfl_data.load_persisted_table(args.raw_dir, "team_stats", season)
        schedules = nfl_data.load_persisted_table(args.raw_dir, "schedules", season)
        if weekly is None or team_stats is None or schedules is None:
            print(f"Season {season}: missing persisted data in {args.raw_dir} - run scripts/fetch_nfl_historical.py first. Skipping.")
            continue
        weekly_frames.append(weekly)
        team_stats_frames.append(team_stats)
        schedule_frames.append(schedules)

    if not weekly_frames:
        print(f"No persisted NFL seasons found in {args.raw_dir} for {args.seasons} - nothing to backtest.")
        return

    weekly_df = pd.concat(weekly_frames, ignore_index=True)
    team_stats_df = pd.concat(team_stats_frames, ignore_index=True)
    schedules_df = pd.concat(schedule_frames, ignore_index=True)

    result = nfl_dfs_backtest.backtest_nfl_dfs_projections(weekly_df, team_stats_df, schedules_df, weeks=args.weeks)

    for label, frame, proj_col, actual_col in [
        ("QB", result["qb"], "DK_Points_QB", "Actual_DK_Points_QB"),
        ("Skill (RB/WR/TE)", result["skill"], "DK_Points_Skill", "Actual_DK_Points_Skill"),
        ("DST", result["dst"], "DK_Points_DST", "Actual_DK_Points_DST"),
    ]:
        scored = frame.dropna(subset=[proj_col, actual_col])
        _report(label, scored[proj_col], scored[actual_col])


if __name__ == "__main__":
    main()
