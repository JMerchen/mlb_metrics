"""Build the MLB game simulation's player profiles (mlb_sim.py):

    docs/data/mlb_sim_profiles.json   each pitcher's pitches and each
                                      hitter's results against them, team
                                      rosters, the league's base running

Needs the stored Statcast pitches (data/raw/statcast_<season>_<month>.parquet)
for this season and last; uses Stuff+ and Decision+ from
scripts/build_pitch_metrics.py (data/pitch_metrics/) when they're there.

Usage:
    python scripts/build_mlb_sim_profiles.py [--season 2026]
"""

import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import data, mlb_sim


def load_pitches(raw_dir: str, season: int) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(raw_dir, f"statcast_{season - 1}_*.parquet"))
                   + glob.glob(os.path.join(raw_dir, f"statcast_{season}_*.parquet")))
    tables = []
    for path in files:
        df = pd.read_parquet(path, columns=None)
        tables.append(df[[c for c in mlb_sim.STATCAST_COLUMNS if c in df.columns]])
    if not tables:
        return pd.DataFrame(columns=mlb_sim.STATCAST_COLUMNS)
    pitches = pd.concat(tables, ignore_index=True)
    pitches["game_date"] = pd.to_datetime(pitches["game_date"])
    return pitches


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--data-dir", default="docs/data")
    parser.add_argument("--season", type=int, default=pd.Timestamp.today().year)
    parser.add_argument("--metrics-dir", default="data/pitch_metrics")
    args = parser.parse_args()

    pitches = load_pitches(args.raw_dir, args.season)
    if pitches.empty or not (pitches["game_date"].dt.year == args.season).any():
        print(f"No stored {args.season} Statcast pitches - nothing to write.")
        return
    try:
        names = data.get_name_register()[["key_mlbam", "name_first", "name_last"]]
    except Exception as e:  # noqa: BLE001 - names are a nicety; ids still work
        print(f"No name register ({e}); hitters keep their ids.")
        names = None
    def optional(name):
        path = os.path.join(args.metrics_dir, name)
        return pd.read_csv(path) if os.path.exists(path) else None

    profiles = mlb_sim.build_profiles(pitches, args.season, names, stuff=optional("stuff_by_type.csv"),
                                      decision=optional("decision_by_batter.csv"))
    mlb_sim.write_profiles(profiles, os.path.join(args.data_dir, "mlb_sim_profiles.json"))
    print(f"Wrote MLB simulation profiles: {len(profiles['teams'])} teams, {len(profiles['pitchers'])} pitchers, "
          f"{len(profiles['batters'])} hitters, through {profiles['through']}.")


if __name__ == "__main__":
    main()
