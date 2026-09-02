"""Runs the real, no-lookahead nfl_game_picks_backtest.py replay against a
real historical NFL season (default: 2025 - the "last year" the approved
plan's train/test methodology is scoped to) and prints the honest
train(weeks 3-7)/test(weeks 8-18) report - the same model-vs-market/
beat-closing-line numbers the README's NFL section quotes.

Needs data/raw/nfl/{schedules,team_stats,weekly,snap_counts,
rosters_weekly}_<season>.parquet already fetched (see
scripts/fetch_nfl_historical.py). `pbp_<season>.parquet` (real
play-by-play, feeding points_per_drive) is NOT part of that historical
backfill (see nfl_data.fetch_pbp's own docstring - real ~20MB/season, not
bulk-backfilled) - this script fetches it live and caches it locally on
first use, then reads the cached copy on every later run (same "fetch
once, cache locally" pattern, not re-fetched every run once present).
Does NOT write to data/predictions/nfl_game_predictions.csv - see
nfl_game_picks_backtest.py's own module docstring for why this backtest
is never backfilled into the live log.

Usage:
    python scripts/run_nfl_game_picks_backtest.py
    python scripts/run_nfl_game_picks_backtest.py --season 2024
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from mlb_metrics import config, nfl_data, nfl_game_picks_backtest as backtest


def _load_or_fetch_pbp(raw_dir: str, season: int) -> pd.DataFrame:
    """Real play-by-play is not part of the bulk historical backfill (see
    nfl_data.fetch_pbp's own docstring) - reads the locally-cached copy if
    one exists, otherwise fetches it live and caches it for next time."""
    cached = nfl_data.load_persisted_table(raw_dir, "pbp", season)
    if cached is not None:
        return cached
    print(f"No cached pbp_{season}.parquet - fetching real play-by-play live (this can take a minute)...")
    pbp = nfl_data.fetch_pbp([season])
    nfl_data.persist_table(pbp, raw_dir, "pbp", season)
    return pbp


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default=config.NFL_RAW_DATA_DIR)
    parser.add_argument("--season", type=int, default=2025)
    args = parser.parse_args()

    def _load(table):
        path = os.path.join(args.raw_dir, f"{table}_{args.season}.parquet")
        return pd.read_parquet(path)

    schedules = _load("schedules")
    team_stats = _load("team_stats")
    weekly = _load("weekly")
    snap_counts = _load("snap_counts")
    rosters = _load("rosters_weekly")
    pbp = _load_or_fetch_pbp(args.raw_dir, args.season)

    replay = backtest.replay_season(schedules, team_stats, weekly, snap_counts, rosters, pbp, season=args.season)
    print(f"Replayed {len(replay)} real {args.season} REG games (weeks 3-18, no lookahead).")

    report = backtest.build_backtest_report(replay)
    pd.set_option("display.width", 200)
    print()
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
