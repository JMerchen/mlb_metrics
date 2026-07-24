"""Build docs/data/dfs_hitters.csv / docs/data/dfs_pitchers.csv from the
SAME wave.csv/pave.csv/confidence.csv scripts/wave.py already wrote
earlier in this run, plus a fresh schedule fetch (today's matchups can
change through the day - same reason pipeline.run() fetches its own
schedule fresh rather than reusing anything cached) and
pitcher_form.compute_pitcher_dfs_form (built from persisted raw Statcast,
the same input source pitchers.compute_pave uses).

Run this DAILY, immediately after scripts/wave.py, as part of
daily_update.yml - unlike Age Curves (historical Lahman comparables barely
move day to day), DFS rankings are entirely about TODAY's slate:
probable starters, opponents, and Matchup_Hit_Probability all change every
single day. A weekly cadence like age_curves_update.yml would serve stale
matchups on 6 of 7 days.

A failed schedule fetch (see schedule.py's module docstring - statsapi is
a genuinely separate, occasionally-flaky data source) leaves yesterday's
dfs_*.csv files in place rather than writing anything - same resilience
tradeoff pipeline.run() already accepts for probable_pitchers.csv.

After the heuristic projections are built, dfs_ml.apply_ml_overrides
swaps in a trained ML model's prediction for any of
DK_Points_Hitter/Expected_H_Allowed/Expected_BB wherever
scripts/train_dfs_ml_models.py has validated one (config.DFS_*_MODEL_PATH
exists and beat both the naive baseline and the heuristic on a held-out
backtest) - a missing/not-yet-trained artifact silently keeps the
existing heuristic, so this script's behavior never depends on whether
that weekly training job has run yet.

Usage:
    python scripts/build_dfs_rankings.py
"""

import argparse
import datetime
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from mlb_metrics import config, data, dfs, dfs_ml, matchup, pitcher_form, pipeline, schedule


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--data-dir", default="docs/data")
    parser.add_argument("--season", type=int, default=config.SEASON_START.year)
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
              f"leaving yesterday's dfs_hitters.csv/dfs_pitchers.csv in place, if any.")
        return

    if schedule_df.empty:
        print("No games scheduled today - leaving yesterday's dfs_hitters.csv/dfs_pitchers.csv in place, if any.")
        return

    persisted = data.load_persisted_statcast(args.raw_dir, args.season)
    if persisted is None:
        print(f"No persisted Statcast in {args.raw_dir} for {args.season} - run scripts/wave.py first.")
        return

    data_with_game_id = data.assign_game_ids(persisted)
    roles = data.label_pitcher_roles(data_with_game_id)
    pdf_with_role = pipeline.build_pitcher_events_with_role(data_with_game_id, roles)

    pitcher_form_df = pitcher_form.compute_pitcher_dfs_form(pdf_with_role)
    matchup_probability = matchup.compute_matchup_hit_probability(wave, pave, confidence, schedule_df)

    hitters = dfs.compute_hitter_dk_points(wave, matchup_probability, schedule_df)
    pitchers = dfs.compute_pitcher_dk_points(pave, pitcher_form_df, schedule_df)

    hitter_features = dfs_ml.build_hitter_features(wave, pave, confidence, schedule_df, matchup_probability)
    hitters, pitchers = dfs_ml.apply_ml_overrides(hitters, hitter_features, pitchers)

    os.makedirs(args.data_dir, exist_ok=True)
    hitters.to_csv(os.path.join(args.data_dir, "dfs_hitters.csv"), index=False)
    pitchers.to_csv(os.path.join(args.data_dir, "dfs_pitchers.csv"), index=False)
    print(
        f"Wrote dfs_hitters.csv ({len(hitters)} rows) and dfs_pitchers.csv ({len(pitchers)} rows) "
        f"for {args.as_of_date}."
    )


if __name__ == "__main__":
    main()
