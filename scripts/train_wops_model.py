"""Train WOPS (wops.py): gradient-boosted expected OPS for a hitter's next
game from his recent form and the opposing starter.

First a walk-forward check - fit on every season before the last one
available, predict every game of the last one - against two baselines
(the hitter's season-to-date OPS and his last-30-games OPS, both shrunk
the same way), then the model that ships is fit on every season and saved
to config.WOPS_MODEL_PATH.

RESULT (2026-10-23, fit 2021-2025, tested on all 49,627 hitter-games of
2026, PA-weighted):
    WOPS               MSE 0.4575   correlation 0.079
    season OPS         MSE 0.4622   correlation 0.050
    last-30 OPS        MSE 0.4641   correlation 0.043
    league average     MSE 0.4603
Single-game OPS is mostly noise, so every correlation is small; WOPS is
the only one that beats predicting the league average, and it is well
calibrated (its lowest tenth of predictions averaged .646 and posted
.657, its highest .801 and .809). Without the starter and platoon
features the correlation drops to 0.062, so the pitcher earns its place.

Usage:
    python scripts/train_wops_model.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import joblib
import numpy as np
import pandas as pd

from mlb_metrics import config, data, wops


def season_frames(raw_dir, seasons):
    frames = {}
    for season in seasons:
        statcast = data.load_game_model_history(raw_dir, [season])
        if statcast.empty:
            continue
        frames[season] = wops.training_frame(statcast)
        print(f"{season}: {len(frames[season])} hitter-games")
    return frames


def report(test, predictions):
    test = test[test["PA"] > 0]
    actual = test["OB"] / test["PA"] + np.where(test["AB"] > 0, test["TB"] / test["AB"].clip(lower=1), 0)
    weights = test["PA"]

    def score(name, p):
        p = pd.Series(p, index=test.index)
        mse = np.average((actual - p) ** 2, weights=weights)
        pm, am = np.average(p, weights=weights), np.average(actual, weights=weights)
        cov = np.average((p - pm) * (actual - am), weights=weights)
        var = np.average((p - pm) ** 2, weights=weights) * np.average((actual - am) ** 2, weights=weights)
        corr = cov / np.sqrt(var) if var > 0 else float("nan")
        print(f"  {name:16s} MSE {mse:.4f}   correlation {corr:.3f}")

    score("WOPS", predictions.loc[test.index])
    score("season OPS", test["hit_obp_season"] + test["hit_slg_season"])
    score("last-30 OPS", test["hit_obp_30"] + test["hit_slg_30"])
    score("league average", np.full(len(test), wops.realized_ops(test)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--first-season", type=int, default=2021)
    parser.add_argument("--last-season", type=int, default=config.SEASON_START.year)
    parser.add_argument("--output", default=config.WOPS_MODEL_PATH)
    args = parser.parse_args()

    frames = season_frames(args.raw_dir, range(args.first_season, args.last_season + 1))
    seasons = sorted(frames)
    if len(seasons) >= 2:
        train = pd.concat([frames[s] for s in seasons[:-1]], ignore_index=True)
        test = frames[seasons[-1]]
        print(f"Walk-forward: fit {seasons[0]}-{seasons[-2]}, test {seasons[-1]} ({int((test['PA'] > 0).sum())} hitter-games)")
        report(test, wops.predict(wops.fit(train), test))

    everything = pd.concat(frames.values(), ignore_index=True)
    models = wops.fit(everything)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    joblib.dump({"models": models, "feature_columns": wops.FEATURE_COLUMNS, "seasons": seasons}, args.output)
    print(f"Saved {args.output} (fit on {seasons[0]}-{seasons[-1]}, {len(everything)} hitter-games)")


if __name__ == "__main__":
    main()
