"""Build Stuff+ and Decision+ (pitch_metrics.py) from the stored Statcast
pitches of this season and last:

    docs/data/stuff_plus.csv          each pitcher's Stuff+ this season,
                                      overall and by pitch type
    docs/data/decision_plus.csv       each hitter's Decision+ this season,
                                      with his decision runs, chase and
                                      zone-swing rates
    data/pitch_metrics/stuff_by_type.csv     both seasons pooled, by pitcher
    data/pitch_metrics/decision_by_batter.csv and pitch type / by hitter
                                      (the MLB simulation's priors)

Usage:
    python scripts/build_pitch_metrics.py [--season 2026]
"""

import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import pitch_metrics as pm

COLUMNS = [
    "game_pk", "game_date", "game_type", "pitch_type", "release_speed", "pfx_x", "pfx_z", "release_spin_rate",
    "spin_axis", "release_extension", "release_pos_x", "release_pos_z", "arm_angle", "vx0", "vy0", "vz0", "ax", "ay",
    "az", "p_throws", "stand", "pitcher", "batter", "delta_run_exp", "description", "plate_x", "plate_z", "sz_top",
    "sz_bot", "balls", "strikes", "outs_when_up", "on_1b", "on_2b", "on_3b", "launch_speed", "launch_angle",
]


def load_pitches(raw_dir: str, season: int) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(raw_dir, f"statcast_{season - 1}_*.parquet"))
                   + glob.glob(os.path.join(raw_dir, f"statcast_{season}_*.parquet")))
    tables = []
    for path in files:
        df = pd.read_parquet(path)
        df = df[df["game_type"] == "R"]
        tables.append(df[[c for c in COLUMNS if c in df.columns]])
    if not tables:
        return pd.DataFrame(columns=COLUMNS)
    p = pd.concat(tables, ignore_index=True)
    p["game_date"] = pd.to_datetime(p["game_date"])
    return p[p["pitch_type"].map(pm.PITCH_GROUP).notna()].reset_index(drop=True)


def _wide_types(by_type: pd.DataFrame) -> pd.DataFrame:
    grades = by_type.pivot(index="pitcher", columns="pitch_type", values="stuff_plus").add_prefix("stuff_")
    counts = by_type.pivot(index="pitcher", columns="pitch_type", values="pitches").add_prefix("pitches_")
    return grades.round(1).join(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--data-dir", default="docs/data")
    parser.add_argument("--model-dir", default="data/pitch_metrics")
    parser.add_argument("--season", type=int, default=pd.Timestamp.today().year)
    args = parser.parse_args()

    p = load_pitches(args.raw_dir, args.season)
    if p.empty or not (p["game_date"].dt.year == args.season).any():
        print(f"No stored {args.season} Statcast pitches - nothing to write.")
        return
    os.makedirs(args.model_dir, exist_ok=True)

    pred = pm.stuff_predictions(p, pm.stuff_features(p))
    by_type, by_pitcher = pm.stuff_plus(p, pred, season=args.season)
    stuff = by_pitcher.set_index("pitcher").join(_wide_types(by_type)).reset_index().rename(columns={"pitcher": "key_mlbam"})
    stuff.round({"stuff_rv": 5, "stuff_plus": 1}).to_csv(os.path.join(args.data_dir, "stuff_plus.csv"), index=False)
    pooled_type, _ = pm.stuff_plus(p, pred)
    pooled_type.round({"stuff_rv": 5, "stuff_plus": 1}).to_csv(os.path.join(args.model_dir, "stuff_by_type.csv"), index=False)

    frame = pm.decision_frame(p)
    values = pm.decision_values(frame, pm.decision_models(frame))
    cols = {"batter": "key_mlbam"}
    season = pm.decision_plus(values, season=args.season).rename(columns=cols)
    season.round(4).to_csv(os.path.join(args.data_dir, "decision_plus.csv"), index=False)
    pm.decision_plus(values).rename(columns=cols).round(4).to_csv(os.path.join(args.model_dir, "decision_by_batter.csv"), index=False)
    print(f"Wrote Stuff+ for {len(stuff)} pitchers ({len(by_type)} pitch types) and Decision+ for {len(season)} hitters, "
          f"{args.season}.")


if __name__ == "__main__":
    main()
