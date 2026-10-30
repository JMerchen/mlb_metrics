"""Build the What if? game simulation's team profiles (nfl_sim.py):

    docs/data/nfl_sim_profiles.json   each offense's play style and the
                                      league's kicking game

Needs data/raw/nfl/pbp_<season>.parquet (and last season's), and
rosters_weekly_<season>.parquet for who's active where, which
scripts/nfl_pipeline.py persists each week.

Usage:
    python scripts/build_nfl_sim_profiles.py [--season 2026]
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import config, nfl_data, nfl_sim


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw/nfl")
    parser.add_argument("--data-dir", default="docs/data")
    parser.add_argument("--season", type=int, default=config.NFL_SEASON)
    args = parser.parse_args()

    tables = []
    for season in (args.season - 1, args.season):
        pbp = nfl_data.load_persisted_table(args.raw_dir, "pbp", season)
        if pbp is not None and not pbp.empty:
            tables.append(pbp[[c for c in nfl_sim.PBP_COLUMNS if c in pbp.columns]])
    if not tables:
        print(f"No persisted {args.season} play-by-play - nothing to write.")
        return
    rosters = nfl_data.load_persisted_table(args.raw_dir, "rosters_weekly", args.season)
    if rosters is not None:
        rosters = rosters[[c for c in nfl_sim.ROSTER_COLUMNS if c in rosters.columns]]
    profiles = nfl_sim.build_profiles(pd.concat(tables, ignore_index=True), args.season, rosters)
    nfl_sim.write_profiles(profiles, os.path.join(args.data_dir, "nfl_sim_profiles.json"))
    print(f"Wrote simulation profiles: {len(profiles['teams'])} teams through week {profiles['through_week']}.")


if __name__ == "__main__":
    main()
