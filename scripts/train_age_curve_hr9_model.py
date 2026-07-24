"""Train, grid-search, and validate an ML replacement for age_curve.py's
weakest metric: HR9 (real backtest - see README's Age Curves section: MAE
indistinguishable from the naive baseline). Tries a genuinely
multi-dimensional regression (age, IP, K9, BB9, HR9, FIP all at once)
against the existing single-dimension KNN comparable search - see
age_curve_ml.py's module docstring for the honest expectation (single-
season HR9 is substantially luck/park/defense-driven, so this may simply
not be fixable by ANY model built from a pitcher's own rate stats).

Methodology (no-lookahead, conservative single-model holdout - see
config.AGE_CURVE_HR9_TEST_YEAR_START's docstring for why this is even
MORE conservative than age_curve.backtest_projection_accuracy's own
per-row cutoff):
1. Trains ONLY on seasons strictly before config.AGE_CURVE_HR9_TEST_YEAR_START
   (1871 through 2009) - the entire 2010-2019 test decade is held out from
   training, not just each individual test row's own future.
2. Year-blocked grid search (age_curve_ml.grid_search_year_blocked) picks
   hyperparameters using ONLY that pre-2010 training data.
3. The selected model is refit on the full pre-2010 pool and evaluated
   ONCE on the same 500-season 2010-2019 sample
   scripts/backtest_age_curve.py already uses for HR9 (same seed, same
   sample size) - a real apples-to-apples comparison against the existing
   heuristic's reported numbers, recomputed fresh here rather than
   trusting stale config.py comments.
4. Only saved to the live-serving path (config.AGE_CURVE_HR9_MODEL_PATH,
   loaded by scripts/build_age_curves.py) if it clears the naive baseline
   AND does at least as well as the existing KNN heuristic - otherwise KNN
   keeps serving HR9 and this prints why, honestly.

Needs data/raw/lahman/{people,batting,pitching}.parquet - run
scripts/fetch_lahman.py first.

Usage:
    python scripts/train_age_curve_hr9_model.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from mlb_metrics import age_curve, age_curve_ml, config, lahman_data, ml_models


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw")
    args = parser.parse_args()

    people = lahman_data.load_persisted_lahman_table(args.raw_dir, "people")
    batting = lahman_data.load_persisted_lahman_table(args.raw_dir, "batting")
    pitching = lahman_data.load_persisted_lahman_table(args.raw_dir, "pitching")
    if people is None or batting is None or pitching is None:
        print(f"No persisted Lahman data in {args.raw_dir}/lahman/ - run scripts/fetch_lahman.py first.")
        return

    historical_pitcher_seasons = age_curve.build_historical_pitcher_seasons(pitching, people)
    print(f"{len(historical_pitcher_seasons)} qualified historical pitcher-seasons (IP >= {config.AGE_CURVE_MIN_IP}).")

    # The same 2010-2019, 500-sampled test seasons scripts/backtest_age_curve.py
    # uses for HR9 - recomputed fresh here for both the heuristic's and the
    # model's numbers, rather than trusting old config.py comments to still
    # be current.
    in_range = historical_pitcher_seasons[
        (historical_pitcher_seasons["yearID"] >= config.AGE_CURVE_HR9_TEST_YEAR_START)
        & (historical_pitcher_seasons["yearID"] <= config.AGE_CURVE_HR9_TEST_YEAR_END)
    ]
    test_seasons = in_range.sample(
        n=min(config.AGE_CURVE_HR9_TEST_SAMPLE_SIZE, len(in_range)), random_state=config.AGE_CURVE_HR9_TEST_SEED
    )

    print(f"\n=== Heuristic (KNN) baseline, recomputed fresh on {len(test_seasons)} sampled seasons ===")
    heuristic_result = age_curve.backtest_projection_accuracy(historical_pitcher_seasons, test_seasons, metric="HR9")
    heuristic_resolved = heuristic_result.dropna(subset=["projected_value_mean"])
    if heuristic_resolved.empty:
        print("No scorable heuristic test seasons - aborting.")
        return
    heuristic_mae = heuristic_resolved["abs_error"].mean()
    heuristic_corr = float(np.corrcoef(heuristic_resolved["projected_value_mean"], heuristic_resolved["actual_next_value"])[0, 1])
    print(f"KNN: MAE {heuristic_mae:.4f}, correlation {heuristic_corr:.3f} (n={len(heuristic_resolved)}/{len(test_seasons)})")

    table = age_curve_ml.build_hr9_training_table(historical_pitcher_seasons)
    train_pool = table[table["yearID"] < config.AGE_CURVE_HR9_TEST_YEAR_START]
    holdout = table.merge(
        test_seasons[["playerID", "yearID"]], on=["playerID", "yearID"], how="inner"
    )
    print(
        f"\nTrain pool: {len(train_pool)} rows (years < {config.AGE_CURVE_HR9_TEST_YEAR_START}); "
        f"holdout: {len(holdout)}/{len(test_seasons)} sampled seasons resolved to a real next season."
    )
    if train_pool.empty or holdout.empty:
        print("Not enough rows for a real train/holdout split - aborting.")
        return

    X_train = train_pool[age_curve_ml.FEATURE_COLUMNS]
    y_train = train_pool["next_HR9"]
    years_train = train_pool["yearID"]

    ridge_search = age_curve_ml.grid_search_year_blocked(
        X_train, y_train, years_train, Ridge(), {"alpha": config.AGE_CURVE_HR9_RIDGE_ALPHA_GRID},
        config.ML_WALK_FORWARD_MIN_TRAIN_YEARS, config.ML_WALK_FORWARD_TEST_BLOCK_YEARS,
    )
    gbm_search = age_curve_ml.grid_search_year_blocked(
        X_train, y_train, years_train, HistGradientBoostingRegressor(), config.AGE_CURVE_HR9_GBM_PARAM_GRID,
        config.ML_WALK_FORWARD_MIN_TRAIN_YEARS, config.ML_WALK_FORWARD_TEST_BLOCK_YEARS,
    )
    print(f"\nRidge best_params={ridge_search.best_params_} cv_score={ridge_search.best_score_:.4f} (n_splits={ridge_search.n_splits_})")
    print(f"GBM   best_params={gbm_search.best_params_} cv_score={gbm_search.best_score_:.4f} (n_splits={gbm_search.n_splits_})")
    best_search = ridge_search if ridge_search.best_score_ >= gbm_search.best_score_ else gbm_search
    best_name = "Ridge" if best_search is ridge_search else "HistGradientBoostingRegressor"
    print(f"Selected: {best_name}")

    X_holdout = holdout[age_curve_ml.FEATURE_COLUMNS]
    predicted = pd.Series(best_search.predict(X_holdout))
    result = ml_models.evaluate_predictions(holdout["next_HR9"], predicted)
    beats_baseline = result["mae"] < result["baseline_mae"]
    print(
        f"\n=== Model (holdout) ===\nMAE {result['mae']:.4f} vs. naive-baseline MAE {result['baseline_mae']:.4f}, "
        f"correlation {result['correlation']:.3f} (n={result['n']})"
    )
    print(f"Heuristic (KNN, same holdout) MAE: {heuristic_mae:.4f}, correlation {heuristic_corr:.3f}")

    if beats_baseline and result["mae"] <= heuristic_mae:
        ml_models.save_model(best_search.best_estimator_, config.AGE_CURVE_HR9_MODEL_PATH)
        print(f"\n-> SAVED to {config.AGE_CURVE_HR9_MODEL_PATH} (beats baseline and KNN - now live for HR9)")
    else:
        print("\n-> NOT saved (KNN keeps serving HR9 - reported honestly, see age_curve_ml.py's module docstring)")


if __name__ == "__main__":
    main()
