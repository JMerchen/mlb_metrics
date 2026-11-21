"""Writes docs/data/nfl_play_calling.json: how each team's drives open -
the first calls in order and whether each worked - on offense and
against its defense, for the play-calling tree on the NFL Stats tab
(src/mlb_metrics/nfl_play_sequencing.py has the why;
docs/nfl_play_calling.js draws it).

Usage:
  python scripts/build_nfl_play_calling.py [--season 2026]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mlb_metrics import config, nfl_play_sequencing  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default=config.NFL_RAW_DATA_DIR)
    parser.add_argument("--data-dir", default="docs/data")
    parser.add_argument("--season", type=int, default=config.NFL_SEASON)
    args = parser.parse_args()
    plays = nfl_play_sequencing.load_plays(args.raw_dir, args.season)
    if plays.empty:
        print(f"No {args.season} play-by-play on disk - nothing to write.")
        return
    data = nfl_play_sequencing.build(plays, args.season)
    with open(os.path.join(args.data_dir, "nfl_play_calling.json"), "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"Wrote play-calling trees: {len(data['teams'])} teams through week {data['through_week']}.")


if __name__ == "__main__":
    main()
