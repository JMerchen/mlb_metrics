"""Convert and persist Lahman's Baseball Database (People/Batting/Pitching,
read from the manually-refreshed Lahman_Raw/*.csv at the repo root - see
lahman_data.py's module docstring for why not a live fetch) to
data/raw/lahman/*.parquet.

Unlike the daily Statcast pull, this data is historical and mostly static -
it only updates roughly once a season, when a completed season is added to
Lahman's database. Run this OCCASIONALLY (e.g. once a season, after
refreshing Lahman_Raw/*.csv per Lahman_Raw/Readme.md), not as part of
daily_update.yml - see age_curve.py/lahman_data.py, which consume the
persisted output for the Age Curves page.

Usage:
    python scripts/fetch_lahman.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import lahman_data


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--lahman-raw-dir", default="Lahman_Raw")
    args = parser.parse_args()

    for name, fetch in (
        ("people", lahman_data.fetch_people),
        ("batting", lahman_data.fetch_batting),
        ("pitching", lahman_data.fetch_pitching),
    ):
        print(f"Reading Lahman {name} from {args.lahman_raw_dir}...")
        df = fetch(args.lahman_raw_dir)
        lahman_data.persist_lahman_table(df, args.raw_dir, name)
        print(f"  wrote {len(df)} rows to {lahman_data.lahman_table_path(args.raw_dir, name)}")


if __name__ == "__main__":
    main()
