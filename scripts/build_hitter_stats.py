"""Build docs/data/hitter_stats.csv for the Stats page's hitter table:
each hitter's CAVE (share of games with a hit, recency-blended like WAVE),
WOPS (OPS, recency-blended like WAVE) and the "vs. pitchers faced"
versions of WAVE, CAVE and WOPS (hitter_views.py).

Needs data/raw/statcast_<season>_*.parquet.

Usage:
    python scripts/build_hitter_stats.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import config, data, hitter_views


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--data-dir", default="docs/data")
    parser.add_argument("--season", type=int, default=config.SEASON_START.year)
    args = parser.parse_args()

    statcast = data.load_persisted_statcast(args.raw_dir, args.season, columns=hitter_views.STATCAST_COLUMNS)
    if statcast is None or statcast.empty:
        print("No persisted Statcast for this season yet - nothing to write.")
        return
    table = hitter_views.hitter_views(statcast)
    path = os.path.join(args.data_dir, "hitter_stats.csv")
    table.round(4).to_csv(path, index=False)
    print(f"Wrote {path} ({len(table)} hitters).")


if __name__ == "__main__":
    main()
