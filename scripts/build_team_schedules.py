"""Build docs/data/team_schedule.csv for the Stats page's schedule view:
every team's regular-season games with final scores, grouped into series
(team_schedule.team_games).

Usage:
    python scripts/build_team_schedules.py
"""

import argparse
import os
import sys

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
    games = team_schedule.team_games(statcast)
    path = os.path.join(args.data_dir, "team_schedule.csv")
    games.to_csv(path, index=False)
    per_team = games.groupby("team")["series"].max()
    print(f"Wrote {path} ({len(games)} team-games; {per_team.min()}-{per_team.max()} series per team).")


if __name__ == "__main__":
    main()
