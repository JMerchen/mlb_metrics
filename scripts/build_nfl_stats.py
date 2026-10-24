"""Build the NFL Stats tab's data from the persisted nflverse tables
(nfl_season_views.py):

    docs/data/nfl_team_schedule.csv   every team's weeks, scores and byes
    docs/data/nfl_team_stats.csv      record, points and EPA per play
    docs/data/nfl_player_stats.csv    QB/RB/WR/TE season views, plain and
                                      "vs. defenses faced"

Needs data/raw/nfl/{schedules,weekly,pbp}_<season>.parquet, which
scripts/nfl_pipeline.py persists each week.

Usage:
    python scripts/build_nfl_stats.py [--season 2026]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import config, nfl_data, nfl_season_views

PBP_COLUMNS = ["season_type", "posteam", "defteam", "epa", "pass", "rush", "interception", "fumble_lost", "success"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw/nfl")
    parser.add_argument("--data-dir", default="docs/data")
    parser.add_argument("--season", type=int, default=config.NFL_SEASON)
    args = parser.parse_args()

    schedules = nfl_data.load_persisted_table(args.raw_dir, "schedules", args.season)
    if schedules is None or schedules.empty:
        print(f"No persisted {args.season} schedule - nothing to write.")
        return
    schedule = nfl_season_views.team_schedule(schedules)
    schedule.to_csv(os.path.join(args.data_dir, "nfl_team_schedule.csv"), index=False)

    pbp = nfl_data.load_persisted_table(args.raw_dir, "pbp", args.season)
    if pbp is not None and not pbp.empty:
        stats = nfl_season_views.team_stats(schedule, pbp[[c for c in PBP_COLUMNS if c in pbp.columns]])
        stats.round(4).to_csv(os.path.join(args.data_dir, "nfl_team_stats.csv"), index=False)

    weekly = nfl_data.load_persisted_table(args.raw_dir, "weekly", args.season)
    if weekly is not None and not weekly.empty:
        players = nfl_season_views.player_views(weekly)
        players.round(4).to_csv(os.path.join(args.data_dir, "nfl_player_stats.csv"), index=False)
        print(f"Wrote NFL stats: {schedule['team'].nunique()} team schedules, {len(players)} players.")


if __name__ == "__main__":
    main()
