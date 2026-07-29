"""Build docs/data/hit_streaks.csv: each recently-active batter's real
current consecutive-games-with-a-hit streak (hitters.compute_current_hit_streaks),
for the dashboard's Beat the Streak "Hit Streaks" subtab. Purely
informational - independent of predictions.select_picks and the official
Beat the Streak picks, which this does not touch.

Needs only data/raw/statcast_<season>.parquet - no schedule fetch, no
today's wave.csv/pave.csv/confidence.csv.

Usage:
    python scripts/build_hit_streaks.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import config, data, hitters


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--data-dir", default="docs/data")
    parser.add_argument("--season", type=int, default=config.SEASON_START.year)
    args = parser.parse_args()

    persisted = data.load_persisted_statcast(args.raw_dir, args.season)
    if persisted is None:
        print(f"No persisted Statcast in {args.raw_dir} for {args.season} - run scripts/wave.py first.")
        return

    data_with_game_id = data.assign_game_ids(persisted)
    streaks = hitters.compute_current_hit_streaks(data_with_game_id)

    if streaks.empty:
        print("No recently-active batters found - leaving yesterday's hit_streaks.csv in place, if any.")
        return

    names = data.get_name_register()[["key_mlbam", "name_first", "name_last"]]
    latest_team = data.latest_team_for_batters(persisted)
    streaks = streaks.merge(names, on="key_mlbam", how="left").merge(latest_team, on="key_mlbam", how="left")
    streaks = streaks.sort_values("Current_Hit_Streak", ascending=False)

    os.makedirs(args.data_dir, exist_ok=True)
    streaks.to_csv(os.path.join(args.data_dir, "hit_streaks.csv"), index=False)
    print(f"Wrote hit_streaks.csv ({len(streaks)} rows).")


if __name__ == "__main__":
    main()
