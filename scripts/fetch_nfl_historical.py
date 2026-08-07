"""One-time/manual backfill of complete past NFL seasons (see
config.NFL_HISTORICAL_SEASONS) to data/raw/nfl/*.parquet, via
nfl_data.py's fetch/persist functions.

Unlike scripts/wave.py's daily Statcast pull, this data is historical and
static - a completed season's stats don't change (see nfl_data.py's
module docstring for why the CURRENT/in-progress season instead gets a
fresh overwrite on every live-pipeline run, a separate concern from this
script). Run this OCCASIONALLY (once, to seed data/raw/nfl/, and again
only if config.NFL_HISTORICAL_SEASONS is ever widened) - not as part of
any recurring workflow. Already-persisted seasons are skipped, not
re-fetched, so re-running this after widening the season range only
fetches the new seasons.

Usage:
    python scripts/fetch_nfl_historical.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import config, nfl_data

TABLES = [
    ("weekly", nfl_data.fetch_weekly_stats),
    ("schedules", nfl_data.fetch_schedules),
    ("injuries", nfl_data.fetch_injuries),
    ("rosters_weekly", nfl_data.fetch_rosters_weekly),
    ("team_stats", nfl_data.fetch_team_stats),
    ("snap_counts", nfl_data.fetch_snap_counts),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default=config.NFL_RAW_DATA_DIR)
    parser.add_argument("--seasons", type=int, nargs="+", default=config.NFL_HISTORICAL_SEASONS)
    args = parser.parse_args()

    for season in args.seasons:
        for table_name, fetch in TABLES:
            if nfl_data.load_persisted_table(args.raw_dir, table_name, season) is not None:
                print(f"{table_name} {season}: already persisted, skipping")
                continue
            print(f"{table_name} {season}: fetching...")
            df = fetch([season])
            nfl_data.persist_table(df, args.raw_dir, table_name, season)
            print(f"  wrote {len(df)} rows to {nfl_data.table_path(args.raw_dir, table_name, season)}")


if __name__ == "__main__":
    main()
