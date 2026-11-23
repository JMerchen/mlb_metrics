"""Writes docs/data/nfl_player_value_<season>.json - our wins above
replacement for every rated NFL player that season - and
docs/data/nfl_player_value.json, the list of seasons there are files for,
for the table on the NFL Stats tab (src/mlb_metrics/nfl_player_value.py
has the method and the why; docs/nfl_player_value.js draws it).

Usage:
  python scripts/build_nfl_player_value.py [--season 2026] [--seasons 2023 2024 2025]
"""

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mlb_metrics import config, nfl_player_value  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default=config.NFL_RAW_DATA_DIR)
    parser.add_argument("--data-dir", default="docs/data")
    parser.add_argument("--season", type=int, default=config.NFL_SEASON)
    parser.add_argument("--seasons", type=int, nargs="+", help="Build these seasons instead (earlier seasons never change).")
    args = parser.parse_args()
    for season in args.seasons or [args.season]:
        data = nfl_player_value.build(season, args.raw_dir)
        if not data["players"]:
            print(f"No {season} play-by-play on disk - nothing to write.")
            continue
        with open(os.path.join(args.data_dir, f"nfl_player_value_{season}.json"), "w") as f:
            json.dump(data, f, separators=(",", ":"))
        print(f"Wrote {season} player value: {len(data['players'])} players through week {data['through_week']}.")
    seasons = sorted((int(re.search(r"_(\d{4})\.json$", path).group(1))
                      for path in glob.glob(os.path.join(args.data_dir, "nfl_player_value_*.json"))), reverse=True)
    with open(os.path.join(args.data_dir, "nfl_player_value.json"), "w") as f:
        json.dump({"seasons": seasons}, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
