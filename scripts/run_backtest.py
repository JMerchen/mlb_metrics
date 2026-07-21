"""Backfill historical picks from git history, resolve them against real
outcomes, and report how well the metrics actually predict hits.

Usage:
    python scripts/run_backtest.py                      # reconstruct + resolve + report
    python scripts/run_backtest.py --skip-reconstruct    # just resolve/score the existing log
    python scripts/run_backtest.py --fetch-missing       # also fetch outcome data Statcast doesn't
                                                          # have persisted locally yet (needs network
                                                          # access to baseballsavant.mlb.com)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from mlb_metrics import config, data, evaluation, git_backtest, predictions


def main():
    parser = argparse.ArgumentParser(description="Reconstruct, resolve, and score historical picks.")
    parser.add_argument("--predictions-log", default="data/predictions/predictions.csv")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--summary-out", default="docs/data/backtest_summary.csv")
    parser.add_argument("--repo-dir", default=".", help="Git checkout to replay docs/data/wave.csv history from.")
    parser.add_argument("--top-n", type=int, default=config.BACKTEST_TOP_N)
    parser.add_argument("--min-plate-appearances", type=int, default=config.BACKTEST_MIN_PLATE_APPEARANCES)
    parser.add_argument("--metric", default="Game_Hit_Probability")
    parser.add_argument(
        "--skip-reconstruct",
        action="store_true",
        help="Don't replay docs/data/wave.csv's git history, just resolve/score the existing log.",
    )
    parser.add_argument(
        "--fetch-missing",
        action="store_true",
        help="Fetch any historical dates not already persisted in --raw-dir directly from Statcast "
        "(one range request covering every missing date).",
    )
    args = parser.parse_args()

    if not args.skip_reconstruct:
        historical_picks = git_backtest.reconstruct_historical_picks(
            repo_dir=args.repo_dir,
            top_n=args.top_n,
            min_plate_appearances=args.min_plate_appearances,
            metric=args.metric,
        )
        print(f"Reconstructed {len(historical_picks)} historical picks from git history.")
        predictions.append_predictions(historical_picks, args.predictions_log)

    log = (
        pd.read_csv(args.predictions_log, parse_dates=["date"])
        if os.path.exists(args.predictions_log)
        else pd.DataFrame()
    )
    pending_dates = (
        sorted(log.loc[log["actual_hit"].isna(), "date"].dt.date.unique()) if not log.empty else []
    )

    persisted = data.load_persisted_statcast(args.raw_dir, config.SEASON_START.year)
    have_dates = set(persisted["game_date"].dt.date.unique()) if persisted is not None else set()
    missing_dates = [d for d in pending_dates if d not in have_dates]

    if missing_dates and args.fetch_missing:
        start, end = min(missing_dates), max(missing_dates)
        print(f"Fetching {start}..{end} from Statcast to cover {len(missing_dates)} missing date(s)...")
        fresh = data.fetch_statcast_range(start, end)
        persisted = data.persist_raw_statcast(fresh, args.raw_dir, config.SEASON_START.year)
    elif missing_dates:
        preview = missing_dates[:5]
        suffix = "..." if len(missing_dates) > 5 else ""
        print(
            f"{len(missing_dates)} pending pick date(s) have no outcome data in {args.raw_dir} "
            f"and --fetch-missing was not set; they'll stay unresolved: {preview}{suffix}"
        )

    if persisted is not None:
        completed = data.completed_events(persisted, ["game_date", "batter", "events"])
        predictions.resolve_predictions(args.predictions_log, completed)

    log = (
        pd.read_csv(args.predictions_log, parse_dates=["date"])
        if os.path.exists(args.predictions_log)
        else pd.DataFrame()
    )
    if log.empty:
        print("No predictions logged yet.")
        return

    summary = evaluation.summarize(log)
    os.makedirs(os.path.dirname(args.summary_out) or ".", exist_ok=True)
    summary.to_csv(args.summary_out, index=False)

    n_resolved = int(log["actual_hit"].notna().sum())
    n_total = len(log)
    print(f"\n{n_resolved}/{n_total} logged picks resolved.")
    print(summary.to_string(index=False))
    print(f"\nWrote summary to {args.summary_out}")


if __name__ == "__main__":
    main()
