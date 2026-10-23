"""Build docs/data/hitter_stats.csv for the Stats page's hitter table: each
hitter's WOPS (wops.py - expected OPS for his next game against today's
probable starter, or an average starter when his team has no game) and how
many of his last config.GAMES_WITH_HIT_WINDOW games had a hit.

Needs data/raw/statcast_<season>_*.parquet and data/models/wops_model.joblib
(scripts/train_wops_model.py); reads docs/data/probable_pitchers.csv when
it is for today.

Usage:
    python scripts/build_hitter_stats.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import joblib
import pandas as pd

from mlb_metrics import config, data, schedule, wops


def todays_probables(path, today):
    """probable_pitchers.csv's rows when they're for `today`, else None."""
    if not os.path.exists(path):
        return None
    probables = pd.read_csv(path)
    if probables.empty or "key_mlbam" not in probables.columns or str(probables["date"].iloc[0])[:10] != str(today):
        return None
    return probables


def build(statcast, model, probables=None):
    features = wops.current_features(statcast, probables)
    features["WOPS"] = wops.predict(model["models"], features).round(3)
    names = {}
    if probables is not None:
        names = dict(zip(probables["key_mlbam"], probables["pitcher_name"]))
    features["wops_starter"] = features["starter"].map(names).fillna("")
    recent = wops.games_with_hit(statcast, config.GAMES_WITH_HIT_WINDOW)
    out = features.rename(columns={"batter": "key_mlbam"})[["key_mlbam", "team", "opponent", "wops_starter", "WOPS"]]
    out = out.rename(columns={"opponent": "wops_opponent"})
    return out.merge(recent, on="key_mlbam", how="left")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--data-dir", default="docs/data")
    parser.add_argument("--season", type=int, default=config.SEASON_START.year)
    parser.add_argument("--model", default=config.WOPS_MODEL_PATH)
    args = parser.parse_args()

    statcast = data.load_persisted_statcast(args.raw_dir, args.season, columns=wops.STATCAST_COLUMNS)
    if statcast is None or statcast.empty:
        print("No persisted Statcast for this season yet - nothing to write.")
        return
    if not os.path.exists(args.model):
        print(f"No WOPS model at {args.model} - run scripts/train_wops_model.py first.")
        return
    probables = todays_probables(os.path.join(args.data_dir, "probable_pitchers.csv"), schedule.today_local())
    table = build(statcast, joblib.load(args.model), probables)
    path = os.path.join(args.data_dir, "hitter_stats.csv")
    table.to_csv(path, index=False)
    print(f"Wrote {path} ({len(table)} hitters{'' if probables is None else ', against today' + chr(39) + 's starters'}).")


if __name__ == "__main__":
    main()
