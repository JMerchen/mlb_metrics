"""Build docs/data/team_schedule.csv for the Stats page's schedule view:
every team's regular-season games with final scores, grouped into series
(team_schedule.team_games), and each side's closing moneyline where
data/raw/market_odds/espn_mlb_odds_<season>.parquet has one
(scripts/backfill_historical_odds.py).

Usage:
    python scripts/build_team_schedules.py
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import config, data, team_schedule


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--data-dir", default="docs/data")
    parser.add_argument("--season", type=int, default=config.SEASON_START.year)
    args = parser.parse_args()

    statcast = data.load_persisted_statcast(args.raw_dir, args.season, columns=team_schedule.STATCAST_COLUMNS)
    if statcast is None or statcast.empty:
        print("No persisted Statcast for this season yet - nothing to write.")
        return
    odds_path = os.path.join(args.raw_dir, "market_odds", f"espn_mlb_odds_{args.season}.parquet")
    odds = pd.read_parquet(odds_path) if os.path.exists(odds_path) else None
    games = team_schedule.team_games(statcast, odds)
    path = os.path.join(args.data_dir, "team_schedule.csv")
    games.to_csv(path, index=False)
    per_team = games.groupby("team")["series"].max()
    with_line = int(games["moneyline"].notna().sum())
    print(f"Wrote {path} ({len(games)} team-games; {per_team.min()}-{per_team.max()} series per team; "
          f"{with_line} with a moneyline).")


if __name__ == "__main__":
    main()
