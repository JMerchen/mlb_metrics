"""Train, grid-search, and validate ML replacements for dfs.py's three
weak heuristic signals (config.py's DFS backtest numbers: DK_Points_Hitter
correlation -0.004; Expected_H_Allowed MAE worse than baseline;
Expected_BB correlation 0.182) - see dfs_ml.py's module docstring for the
feature sets and ml_models.py's for the walk-forward CV mechanism.

Methodology (no-lookahead, nested holdout - see the Plan this followed):
1. dfs_backtest.assemble_ml_training_rows assembles the FULL persisted
   history's feature+label table (every row's features come only from
   data strictly before that row's game date).
2. The last config.ML_FINAL_HOLDOUT_DATES dates are set aside entirely -
   the grid search below never sees them. This is the same-size window
   the original heuristic-only backtest used (scripts/backtest_dfs_rankings.py),
   so it's a fair head-to-head comparison.
3. Grid search (ml_models.grid_search_walk_forward, walk-forward blocked
   CV) runs ONLY on the earlier "train pool" dates, picking the
   best-scoring hyperparameters/model family per signal.
4. The selected model is refit on the full train pool and evaluated ONCE
   on the untouched holdout - that's the number reported and compared
   against the heuristic's own performance on the identical holdout
   window (dfs_backtest.backtest_dfs_projections with the same `days`).
5. A model is only saved to its live-serving path
   (config.DFS_*_MODEL_PATH, loaded by dfs_ml.apply_ml_overrides) if it
   does at least as well as the existing heuristic - "our current
   model" for this signal - on the holdout (2026-08-24 policy: "as long
   as it beats our current model, save it"; the naive baseline is still
   printed for context but is no longer a required bar). Otherwise the
   heuristic keeps serving live traffic and this prints why, honestly,
   rather than shipping a model that doesn't actually help.

Needs data/raw/statcast_<season>.parquet (see scripts/wave.py).

Usage:
    python scripts/train_dfs_ml_models.py
    python scripts/train_dfs_ml_models.py --season 2026
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from mlb_metrics import config, dfs_backtest, dfs_ml, ml_models


def _split_holdout(df: pd.DataFrame, holdout_dates: int):
    dates = sorted(df["date"].unique())
    if len(dates) <= holdout_dates:
        return df.iloc[0:0], df, dates  # not enough dates for a real holdout
    cutoff_dates = set(dates[-holdout_dates:])
    holdout = df[df["date"].isin(cutoff_dates)]
    train_pool = df[~df["date"].isin(cutoff_dates)]
    return train_pool, holdout, dates


def _report(label: str, result: dict) -> bool:
    """Prints the result and returns whether it beat the naive baseline."""
    if result["n"] < 2:
        print(f"  {label}: not enough holdout rows ({result['n']}) to report.")
        return False
    beats_baseline = result["mae"] < result["baseline_mae"]
    verdict = "BEATS baseline" if beats_baseline else "does NOT beat baseline"
    print(
        f"  {label}: MAE {result['mae']:.4f} vs. naive-baseline MAE {result['baseline_mae']:.4f}, "
        f"correlation {result['correlation']:.3f} (n={result['n']}) - {verdict}"
    )
    return beats_baseline


def train_hitter_model(train_rows: pd.DataFrame, holdout: pd.DataFrame, heuristic_mae: float) -> None:
    print("\n=== DFS Hitter: DK_Points_Hitter ===")
    X_train = dfs_ml.hitter_feature_matrix(train_rows)
    y_train = train_rows["Actual_DK_Points_Modeled"]
    dates_train = train_rows["date"]

    ridge_search = ml_models.grid_search_walk_forward(
        X_train, y_train, dates_train, Ridge(), {"alpha": config.DFS_HITTER_RIDGE_ALPHA_GRID},
        config.ML_WALK_FORWARD_MIN_TRAIN_DATES_HITTER, config.ML_WALK_FORWARD_TEST_BLOCK_DATES_HITTER,
    )
    gbm_search = ml_models.grid_search_walk_forward(
        X_train, y_train, dates_train, HistGradientBoostingRegressor(), config.DFS_HITTER_GBM_PARAM_GRID,
        config.ML_WALK_FORWARD_MIN_TRAIN_DATES_HITTER, config.ML_WALK_FORWARD_TEST_BLOCK_DATES_HITTER,
    )
    print(f"  Ridge best_params={ridge_search.best_params_} cv_score={ridge_search.best_score_:.4f}")
    print(f"  GBM   best_params={gbm_search.best_params_} cv_score={gbm_search.best_score_:.4f}")
    best_search = ridge_search if ridge_search.best_score_ >= gbm_search.best_score_ else gbm_search
    best_name = "Ridge" if best_search is ridge_search else "HistGradientBoostingRegressor"
    print(f"  Selected: {best_name}")

    X_holdout = dfs_ml.hitter_feature_matrix(holdout)
    predicted = pd.Series(best_search.predict(X_holdout)).clip(lower=0)
    result = ml_models.evaluate_predictions(holdout["Actual_DK_Points_Modeled"], predicted)
    _report("Model (holdout)", result)
    print(f"  Heuristic (holdout) MAE: {heuristic_mae:.4f}")

    # Save gate (2026-08-24 policy: "as long as it beats our current
    # model, save it" - the naive baseline is still printed above for
    # context by _report, but is no longer a required bar. "Our current
    # model" is the live heuristic (dfs.py) this artifact would replace.
    if result["mae"] <= heuristic_mae:
        ml_models.save_model(best_search.best_estimator_, config.DFS_HITTER_MODEL_PATH)
        print(f"  -> SAVED to {config.DFS_HITTER_MODEL_PATH} (beats the heuristic - now live)")
    else:
        print("  -> NOT saved (heuristic keeps serving - see dfs.py's docstring, this is reported honestly)")


def _train_pitcher_component(
    train_rows: pd.DataFrame, holdout: pd.DataFrame, label: str, target_col: str, model_path: str, heuristic_mae: float
) -> None:
    print(f"\n=== DFS Pitcher: {label} ===")
    X_train = dfs_ml.pitcher_feature_matrix(train_rows)
    y_train = train_rows[target_col]
    dates_train = train_rows["date"]

    search = ml_models.grid_search_walk_forward(
        X_train, y_train, dates_train, Ridge(), {"alpha": config.DFS_PITCHER_RIDGE_ALPHA_GRID},
        config.ML_WALK_FORWARD_MIN_TRAIN_DATES_PITCHER, config.ML_WALK_FORWARD_TEST_BLOCK_DATES_PITCHER,
    )
    print(f"  Ridge best_params={search.best_params_} cv_score={search.best_score_:.4f}")

    X_holdout = dfs_ml.pitcher_feature_matrix(holdout)
    predicted = pd.Series(search.predict(X_holdout)).clip(lower=0)
    result = ml_models.evaluate_predictions(holdout[target_col], predicted)
    _report("Model (holdout)", result)
    print(f"  Heuristic (holdout) MAE: {heuristic_mae:.4f}")

    # Save gate (2026-08-24 policy: see train_hitter_model's own comment above.
    if result["mae"] <= heuristic_mae:
        ml_models.save_model(search.best_estimator_, model_path)
        print(f"  -> SAVED to {model_path} (beats the heuristic - now live)")
    else:
        print("  -> NOT saved (heuristic keeps serving - see dfs.py's docstring, this is reported honestly)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument(
        "--days", type=int, default=None,
        help="Trim training-row assembly to the most recent N dates instead of the full persisted "
             "history - each date recomputes the full pipeline, which is expensive. Full history is "
             "the default (None) and the honest choice when time allows; trimming is a documented, "
             "not-silent fallback (see this script's module docstring).",
    )
    args = parser.parse_args()

    season = args.season or config.SEASON_START.year
    print(f"Assembling ML training rows (season={season}, days={args.days or 'all'})...")
    rows = dfs_backtest.assemble_ml_training_rows(args.raw_dir, season=season, days=args.days)

    heuristic = dfs_backtest.backtest_dfs_projections(args.raw_dir, season=season, days=config.ML_FINAL_HOLDOUT_DATES)
    heuristic_hitter_mae = (
        (heuristic["hitters"]["DK_Points_Hitter"] - heuristic["hitters"]["Actual_DK_Points_Modeled"]).abs().mean()
        if not heuristic["hitters"].empty else float("inf")
    )
    heuristic_h_allowed_mae = (
        (heuristic["pitchers"]["Expected_H_Allowed"] - heuristic["pitchers"]["Actual_H"]).abs().mean()
        if not heuristic["pitchers"].empty else float("inf")
    )
    heuristic_bb_mae = (
        (heuristic["pitchers"]["Expected_BB"] - heuristic["pitchers"]["Actual_BB"]).abs().mean()
        if not heuristic["pitchers"].empty else float("inf")
    )

    hitters = rows["hitters"]
    if hitters.empty:
        print("No hitter training rows assembled - skipping hitter model.")
    else:
        train_pool, holdout, dates = _split_holdout(hitters, config.ML_FINAL_HOLDOUT_DATES)
        print(f"Hitters: {len(dates)} distinct dates, {len(train_pool)} train-pool rows, {len(holdout)} holdout rows")
        if holdout.empty or train_pool.empty:
            print("Not enough distinct dates for a real holdout split - skipping hitter model.")
        else:
            train_hitter_model(train_pool, holdout, heuristic_hitter_mae)

    pitchers = rows["pitchers"]
    if pitchers.empty:
        print("No pitcher training rows assembled - skipping pitcher models.")
    else:
        train_pool, holdout, dates = _split_holdout(pitchers, config.ML_FINAL_HOLDOUT_DATES)
        print(f"Pitchers: {len(dates)} distinct dates, {len(train_pool)} train-pool rows, {len(holdout)} holdout rows")
        if holdout.empty or train_pool.empty:
            print("Not enough distinct dates for a real holdout split - skipping pitcher models.")
        else:
            _train_pitcher_component(
                train_pool, holdout, "Expected_H_Allowed", "Actual_H",
                config.DFS_PITCHER_H_ALLOWED_MODEL_PATH, heuristic_h_allowed_mae,
            )
            _train_pitcher_component(
                train_pool, holdout, "Expected_BB", "Actual_BB",
                config.DFS_PITCHER_BB_MODEL_PATH, heuristic_bb_mae,
            )


if __name__ == "__main__":
    main()
