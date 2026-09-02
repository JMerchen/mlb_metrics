"""Fit + report phase for a RECALIBRATION of nfl_game_picks.py's raw
composite heuristic (compute_game_win_probabilities) - direct mirror of
scripts/train_game_pick_calibration.py (see that script's own docstring
for the full reasoning: a rescaling of an already-computed probability
estimate, not a from-scratch replacement model).

nfl_game_picks_backtest.py's own real 2025 replay already surfaced the
exact same problem MLB's original calibration follow-up found: the raw
heuristic's spread (std ~0.03 across the real 2025 season) is far
narrower than the real market's own spread on the same games (std
~0.20) - see scripts/run_nfl_game_picks_backtest.py's own printed report.
This script fits ml_models.fit_probability_calibration (isotonic or
sigmoid/Platt, picked via real walk-forward CV over weeks - not guessed)
against nfl_game_picks_backtest.replay_season's real resolved outcomes.

Save gate (same policy as the MLB script): saved to
config.NFL_GAME_PICK_CALIBRATION_MODEL_PATH ONLY if the calibrated
probability beats the raw (uncalibrated) heuristic - the current live
model for this signal - on an untouched final holdout (the most recent
config.NFL_GAME_PICK_ML_FINAL_HOLDOUT_WEEKS real weeks). Reported
honestly either way; an existing artifact is left untouched if this run
doesn't clear the bar.

**Scope**: wired live via nfl_game_picks.apply_calibration - loading
gracefully falls back to the raw, uncalibrated heuristic when no artifact
exists yet (or hasn't cleared its own bar).

Needs data/raw/nfl/{schedules,team_stats,weekly,snap_counts,
rosters_weekly}_<season>.parquet already fetched (scripts/fetch_nfl_historical.py).
Fully offline.

Usage:
    python scripts/train_nfl_game_pick_calibration.py
    python scripts/train_nfl_game_pick_calibration.py --season 2024
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from mlb_metrics import config, ml_models, nfl_game_picks_backtest as backtest


def _split_holdout(rows: pd.DataFrame, holdout_weeks: int):
    weeks = sorted(rows["week"].unique())
    if len(weeks) <= holdout_weeks:
        return rows.iloc[0:0], rows, weeks  # not enough weeks for a real holdout
    cutoff_weeks = set(weeks[-holdout_weeks:])
    holdout = rows[rows["week"].isin(cutoff_weeks)]
    train_pool = rows[~rows["week"].isin(cutoff_weeks)]
    return train_pool, holdout, weeks


def _cv_log_loss(train_pool: pd.DataFrame, method: str, min_train_weeks: int, test_block_weeks: int) -> float:
    """Mean walk-forward-CV log_loss for one candidate calibration
    method, fit fresh on each fold's own train slice (no lookahead) -
    direct mirror of train_game_pick_calibration.py's own `_cv_log_loss`,
    with `week` in place of `date`."""
    weeks = train_pool["week"].reset_index(drop=True)
    raw = train_pool["home_win_probability"].reset_index(drop=True)
    y = train_pool["home_won"].reset_index(drop=True)

    splitter = ml_models.WalkForwardDateSplit(weeks, min_train_weeks, test_block_weeks)
    fold_losses = []
    for train_idx, test_idx in splitter.split():
        fold_calibrator = ml_models.fit_probability_calibration(raw.iloc[train_idx], y.iloc[train_idx], method=method)
        fold_predicted = fold_calibrator.predict(raw.iloc[test_idx].to_numpy())
        fold_result = ml_models.evaluate_classifier_predictions(y.iloc[test_idx], fold_predicted)
        fold_losses.append(fold_result["log_loss"])

    if not fold_losses:
        return float("nan")
    return sum(fold_losses) / len(fold_losses)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default=config.NFL_RAW_DATA_DIR)
    parser.add_argument("--season", type=int, default=2025)
    args = parser.parse_args()

    def _load(table):
        return pd.read_parquet(os.path.join(args.raw_dir, f"{table}_{args.season}.parquet"))

    schedules, team_stats, weekly = _load("schedules"), _load("team_stats"), _load("weekly")
    snap_counts, rosters = _load("snap_counts"), _load("rosters_weekly")

    print(f"Replaying {args.season} season (no lookahead)...")
    rows = backtest.replay_season(schedules, team_stats, weekly, snap_counts, rosters, season=args.season)

    if rows.empty:
        print("No replayed games - nothing to fit.")
        return

    # Same real, honest NaN handling as train_game_pick_calibration.py's
    # own script - a degenerate zero-variance week (see
    # nfl_game_picks_backtest.py's module docstring) has no real
    # calibration target to fit on, dropped and reported, not zero-filled.
    before = len(rows)
    rows = rows[rows["home_win_probability"].notna()].copy()
    if before - len(rows):
        print(f"Dropped {before - len(rows)} row(s) with a NaN raw home_win_probability - can't calibrate on those.")

    train_pool, holdout, weeks = _split_holdout(rows, config.NFL_GAME_PICK_ML_FINAL_HOLDOUT_WEEKS)
    print(f"{len(weeks)} distinct weeks, {len(train_pool)} train-pool rows, {len(holdout)} holdout rows")
    if holdout.empty or train_pool.empty:
        print("Not enough distinct weeks for a real holdout split - skipping.")
        return

    print("\n=== Walk-forward CV: isotonic vs sigmoid (Platt) ===")
    cv_results = {}
    for method in ("isotonic", "sigmoid"):
        cv_log_loss = _cv_log_loss(
            train_pool, method,
            config.NFL_GAME_PICK_ML_WALK_FORWARD_MIN_TRAIN_WEEKS, config.NFL_GAME_PICK_ML_WALK_FORWARD_TEST_BLOCK_WEEKS,
        )
        cv_results[method] = cv_log_loss
        print(f"  {method}: mean CV log_loss={cv_log_loss:.4f}")

    valid_results = {m: v for m, v in cv_results.items() if v == v}  # drop NaN
    if not valid_results:
        print("No CV fold had both a train and test slice - not enough history yet to validate. Not saved.")
        return
    best_method = min(valid_results, key=valid_results.get)
    print(f"  -> best by CV: {best_method}")

    print(f"\n=== Real final holdout ({len(holdout)} rows, {config.NFL_GAME_PICK_ML_FINAL_HOLDOUT_WEEKS} most recent weeks) ===")
    calibrator = ml_models.fit_probability_calibration(
        train_pool["home_win_probability"], train_pool["home_won"], method=best_method
    )
    calibrated_proba = calibrator.predict(holdout["home_win_probability"].to_numpy())
    calibrated_result = ml_models.evaluate_classifier_predictions(holdout["home_won"], calibrated_proba)
    raw_result = ml_models.evaluate_classifier_predictions(
        holdout["home_won"], holdout["home_win_probability"].clip(0, 1)
    )

    print(
        f"  Calibrated ({best_method}): log_loss={calibrated_result['log_loss']:.4f}, "
        f"brier={calibrated_result['brier_score']:.4f}, roc_auc={calibrated_result['roc_auc']:.4f}, "
        f"accuracy={calibrated_result['accuracy']:.4f} (n={calibrated_result['n']})"
    )
    print(f"  Naive baseline:             log_loss={calibrated_result['baseline_log_loss']:.4f} (always predict base rate)")
    print(
        f"  Raw heuristic (uncalibrated): log_loss={raw_result['log_loss']:.4f}, "
        f"brier={raw_result['brier_score']:.4f}, roc_auc={raw_result['roc_auc']:.4f}, "
        f"accuracy={raw_result['accuracy']:.4f}"
    )
    print(f"  Spread (std): calibrated={calibrated_proba.std():.4f}, raw={holdout['home_win_probability'].std():.4f}")

    beats_raw = calibrated_result["log_loss"] < raw_result["log_loss"]
    if beats_raw:
        final_calibrator = ml_models.fit_probability_calibration(
            rows["home_win_probability"], rows["home_won"], method=best_method
        )
        ml_models.save_model(final_calibrator, config.NFL_GAME_PICK_CALIBRATION_MODEL_PATH)
        print(
            f"\n  -> SAVED to {config.NFL_GAME_PICK_CALIBRATION_MODEL_PATH} "
            f"(beats the raw heuristic on the real holdout - refit on the FULL replayed season "
            f"before saving; wired live via nfl_game_picks.apply_calibration)"
        )
    else:
        print(
            "\n  -> NOT saved (does not beat the raw heuristic on the real holdout - "
            "reported honestly). Any existing artifact at "
            f"{config.NFL_GAME_PICK_CALIBRATION_MODEL_PATH} is left untouched, not deleted - only a real "
            "improvement overwrites it."
        )


if __name__ == "__main__":
    main()
