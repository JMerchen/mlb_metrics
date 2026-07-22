"""Backfill historical Automated Game Picks from confidence.csv/pave.csv
git history plus already-persisted Statcast, and report how well the
model would have performed.

Uses ACTUAL starting pitchers and ACTUAL final scores (both already fully
contained in persisted Statcast), not statsapi's "probable" pitcher
announcements or a live final-score fetch - neither was ever persisted to
git history (see schedule.py). This runs entirely offline: no network
access needed at all, unlike the hitter-pick backtest's --fetch-missing.

Usage:
    python scripts/run_game_picks_backtest.py               # backfill last 40 days + report
    python scripts/run_game_picks_backtest.py --days 60      # backfill a longer/shorter window
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from mlb_metrics import config, game_evaluation, game_picks_backtest, game_predictions, pipeline


def main():
    parser = argparse.ArgumentParser(description="Backfill and score historical Automated Game Picks.")
    parser.add_argument("--predictions-log", default="data/predictions/game_predictions.csv")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--docs-data-dir", default="docs/data")
    parser.add_argument("--repo-dir", default=".", help="Git checkout to replay confidence.csv/pave.csv history from.")
    parser.add_argument("--days", type=int, default=40, help="How many of the most recent daily commits to replay.")
    parser.add_argument("--season", type=int, default=config.SEASON_START.year)
    args = parser.parse_args()

    historical_picks = game_picks_backtest.reconstruct_historical_game_picks(
        repo_dir=args.repo_dir, raw_dir=args.raw_dir, season=args.season, days=args.days,
    )
    print(f"Reconstructed {len(historical_picks)} historical game picks from the last {args.days} day(s).")
    game_predictions.append_game_predictions(historical_picks, args.predictions_log)

    pipeline.write_game_picks_export(args.predictions_log, args.docs_data_dir)

    log = (
        pd.read_csv(args.predictions_log, parse_dates=["date"])
        if os.path.exists(args.predictions_log)
        else pd.DataFrame()
    )
    if log.empty:
        print("No game picks logged yet.")
        return

    picks, summary = game_evaluation.build_game_picks_export(log)
    print(f"\n{summary.loc[0, 'n_games_resolved']}/{len(log)} logged game picks resolved.")
    print(summary.to_string(index=False))
    print(f"\nWrote docs/data/game_picks_picks.csv and game_picks_summary.csv to {args.docs_data_dir}")


if __name__ == "__main__":
    main()
