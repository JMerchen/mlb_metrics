"""Build docs/data/hitter_hit_predictions.csv: today's PA-qualified hitters
ranked by dfs_ml's trained hit-probability model
(config.HITTER_HIT_PROBABILITY_MODEL_PATH, scripts/train_hitter_hit_model.py),
for the dashboard's Beat the Streak "Model Odds" subtab.

Purely informational - this is an ADDITIONAL, independent view alongside
the official Beat the Streak picks (predictions.select_picks's
Approach/Matchup_Approach heuristic, unchanged). It does not gate, filter,
or replace anything in the live pick-selection path.

Mirrors scripts/build_dfs_rankings.py's resilience shape exactly (same
wave.csv/pave.csv/confidence.csv + schedule-fetch dependency, same
leave-yesterday's-output-in-place failure handling). If the model artifact
itself is missing/not yet trained, writes nothing and prints why, rather
than crashing - matches ml_models.load_model's own fallback discipline.

Usage:
    python scripts/build_hitter_hit_predictions.py
"""

import argparse
import datetime
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from mlb_metrics import config, dfs_ml, matchup, schedule


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="docs/data")
    parser.add_argument("--as-of-date", type=datetime.date.fromisoformat, default=datetime.date.today())
    args = parser.parse_args()

    wave_path = os.path.join(args.data_dir, "wave.csv")
    pave_path = os.path.join(args.data_dir, "pave.csv")
    confidence_path = os.path.join(args.data_dir, "confidence.csv")
    if not (os.path.exists(wave_path) and os.path.exists(pave_path) and os.path.exists(confidence_path)):
        print(f"No wave.csv/pave.csv/confidence.csv in {args.data_dir} - run scripts/wave.py first.")
        return

    wave = pd.read_csv(wave_path)
    pave = pd.read_csv(pave_path)
    confidence = pd.read_csv(confidence_path)

    try:
        schedule_df = schedule.fetch_probable_pitchers(args.as_of_date)
    except Exception as exc:
        print(f"WARNING: failed to fetch today's schedule/probable pitchers ({exc}); "
              f"leaving yesterday's hitter_hit_predictions.csv in place, if any.")
        return

    if schedule_df.empty:
        print("No games scheduled today - leaving yesterday's hitter_hit_predictions.csv in place, if any.")
        return

    matchup_probability = matchup.compute_matchup_hit_probability(wave, pave, confidence, schedule_df)
    hitter_features = dfs_ml.build_hitter_features(wave, pave, confidence, schedule_df, matchup_probability)
    hitter_features = hitter_features.merge(wave[["key_mlbam", "name_first", "name_last", "team"]], on="key_mlbam", how="left")
    hitter_features = hitter_features.merge(schedule_df[["team", "opponent"]].drop_duplicates(), on="team", how="left")

    hitter_features["Total_PA"] = hitter_features["PA_L"] + hitter_features["PA_R"]
    qualified = hitter_features[hitter_features["Total_PA"] >= config.BACKTEST_MIN_PLATE_APPEARANCES].copy()

    predictions = dfs_ml.predict_hitter_hit_probability(qualified)
    if predictions.empty:
        print(f"No model artifact at {config.HITTER_HIT_PROBABILITY_MODEL_PATH} - "
              f"leaving yesterday's hitter_hit_predictions.csv in place, if any.")
        return

    result = qualified[["key_mlbam", "name_first", "name_last", "team", "opponent", "is_home"]].merge(
        predictions, on="key_mlbam", how="inner"
    )
    result = result.sort_values("Model_Hit_Probability", ascending=False)

    os.makedirs(args.data_dir, exist_ok=True)
    result.to_csv(os.path.join(args.data_dir, "hitter_hit_predictions.csv"), index=False)
    print(f"Wrote hitter_hit_predictions.csv ({len(result)} rows) for {args.as_of_date}.")


if __name__ == "__main__":
    main()
