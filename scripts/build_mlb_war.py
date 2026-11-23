"""Writes docs/data/mlb_war_<season>.json - our wins above replacement
for every MLB player that season - and docs/data/mlb_war.json, the list
of seasons there are files for, for the table on the MLB Stats page
(src/mlb_metrics/mlb_war.py has the method and the why; docs/mlb_war.js
draws it).

Usage:
  python scripts/build_mlb_war.py [--season 2026] [--seasons 2025 2026]

About 40 seconds a season.
"""

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mlb_metrics import mlb_war  # noqa: E402


def latest_season(raw_dir: str) -> int | None:
    """The latest season with pitches on disk."""
    years = [int(m.group(1)) for m in (re.search(r"statcast_(\d{4})", path) for path in glob.glob(os.path.join(raw_dir, "statcast_*.parquet"))) if m]
    return max(years) if years else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--data-dir", default="docs/data")
    parser.add_argument("--season", type=int, default=None, help="Default: the latest season with pitches on disk.")
    parser.add_argument("--seasons", type=int, nargs="+", help="Build these seasons instead (earlier seasons never change).")
    args = parser.parse_args()
    season = args.season or latest_season(args.raw_dir)
    for season in args.seasons or ([season] if season else []):
        data = mlb_war.build(season, args.raw_dir)
        if not data["players"]:
            print(f"No {season} regular-season pitches on disk - nothing to write.")
            continue
        with open(os.path.join(args.data_dir, f"mlb_war_{season}.json"), "w") as f:
            json.dump(data, f, separators=(",", ":"), ensure_ascii=False)
        print(f"Wrote {season} MLB WAR: {len(data['players'])} players through {data['through']}.")
    seasons = sorted((int(re.search(r"_(\d{4})\.json$", path).group(1))
                      for path in glob.glob(os.path.join(args.data_dir, "mlb_war_*.json"))), reverse=True)
    with open(os.path.join(args.data_dir, "mlb_war.json"), "w") as f:
        json.dump({"seasons": seasons}, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
