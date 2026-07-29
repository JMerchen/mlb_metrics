"""Two deliverables from the hitter hit log (dfs_backtest.assemble_hitter_hit_log,
see its docstring and README's "Hitter hit log" section) - real answers to
the "why is Beat the Streak survival only ~50%, and which features are
actually significant" question the earlier ad hoc analysis (n=64 resolved
picks) couldn't answer with confidence:

1. **Significance report** (statsmodels.Logit): one univariate Logit per
   feature in dfs_ml.HITTER_FEATURE_COLUMNS ("individually"), plus one
   multivariate Logit with all of them together ("combined"), fit on the
   FULL real history (~31k rows across ~120 dates as of 2026-07-28 - no
   holdout split needed here, since statistical inference isn't a
   held-out-accuracy question). Features are standardized (z-scored)
   first so coefficient magnitudes are comparable across very different
   scales (WAVE ~0-0.4 vs Total_PA ~0-30) - p-values themselves are
   scale-invariant. A feature that's constant in the current data (zero
   std) is excluded and reported as such rather than crashing on a
   divide-by-zero.

2. **Walk-forward-validated predictive model** (sklearn LogisticRegression +
   ml_models.py's WalkForwardDateSplit machinery): the SAME no-lookahead,
   nested-holdout methodology scripts/train_dfs_ml_models.py already uses
   for the three live DFS ML signals. The selected model is evaluated
   ONCE on an untouched final holdout and must beat BOTH a naive baseline
   (always predict the base hit rate) AND the existing Game_Hit_Probability
   heuristic column (the natural "heuristic to beat" here) before being
   saved to config.HITTER_HIT_PROBABILITY_MODEL_PATH.

**Scope**: this fits and reports. The saved model artifact is NOT wired
into dfs_ml.apply_ml_overrides or predictions.select_picks - whether/how
this ever feeds live Beat the Streak picks is a separate, later decision.

**PA filter**: dfs_backtest.assemble_hitter_hit_log deliberately carries
EVERY hitter with a game that date, including ones with a handful of
career plate appearances at that point - a WAVE/Game_Hit_Probability
built on a tiny sample is mostly noise, and fitting on it would let that
noise drag down real coefficients. This script filters to
Total_PA >= --min-plate-appearances (default config.BACKTEST_MIN_PLATE_APPEARANCES,
the SAME gate predictions.select_picks already applies before a hitter is
ever eligible to be a pick) right before fitting, for both the
significance report and the predictive model - the log itself stays
unfiltered so future consumers can pick their own threshold.

Needs data/raw/statcast_<season>.parquet (see scripts/wave.py).

Usage:
    python scripts/train_hitter_hit_model.py
    python scripts/train_hitter_hit_model.py --season 2026
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

from mlb_metrics import config, dfs_backtest, dfs_ml, ml_models


def _split_holdout(df: pd.DataFrame, holdout_dates: int):
    dates = sorted(df["date"].unique())
    if len(dates) <= holdout_dates:
        return df.iloc[0:0], df, dates  # not enough dates for a real holdout
    cutoff_dates = set(dates[-holdout_dates:])
    holdout = df[df["date"].isin(cutoff_dates)]
    train_pool = df[~df["date"].isin(cutoff_dates)]
    return train_pool, holdout, dates


def _standardize(X: pd.DataFrame) -> pd.DataFrame:
    """z-score every column; drops any column with zero variance in this
    data (would otherwise divide by zero / be perfectly collinear with the
    intercept) and prints which, if any, were dropped."""
    std = X.std()
    constant_columns = std[std == 0].index.tolist()
    if constant_columns:
        print(f"  Excluding constant-in-this-data feature(s) from the significance report: {constant_columns}")
    X = X.drop(columns=constant_columns)
    return (X - X.mean()) / X.std()


def _fit_logit_report(y: pd.Series, X: pd.DataFrame, label: str):
    """Returns (table, result) on success, or None (never raises) if the
    fit fails - e.g. a near-perfectly-collinear multivariate design (see
    this script's module docstring re: Consistency/Approach) - so a single
    failed fit is reported and skipped rather than crashing the whole
    script."""
    try:
        result = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
    except Exception as exc:
        print(f"  {label}: fit failed ({exc})")
        return None
    table = pd.DataFrame({
        "coef": result.params, "std_err": result.bse, "z": result.tvalues, "p_value": result.pvalues,
    }).drop(index="const")
    return table, result


def significance_report(rows: pd.DataFrame) -> None:
    print("\n=== Feature significance report (statsmodels.Logit, full history) ===")
    y = rows["Got_Hit"]
    # hitter_feature_matrix already fills every NaN/pd.NA (a feature that
    # wasn't computable yet on an early-history date, e.g. Park_Factor
    # before enough home games are on record) with 0 - but pd.NA-only
    # columns from those early dates leave the concatenated column
    # dtype=object even after filling, which statsmodels' Logit can't
    # consume (sklearn's LogisticRegression silently coerces past this,
    # which is why the walk-forward model below isn't affected). .astype
    # (float) only retags the dtype; every value is already numeric.
    X = _standardize(dfs_ml.hitter_feature_matrix(rows).astype(float))
    print(f"  n={len(rows)}, base hit rate={y.mean():.4f}")

    print("\n  -- Individually (one univariate Logit per feature) --")
    univariate_rows = []
    for column in X.columns:
        fit = _fit_logit_report(y, X[[column]], column)
        if fit is None:
            continue
        table, _ = fit
        univariate_rows.append(table.loc[column])
    if univariate_rows:
        univariate_table = pd.DataFrame(univariate_rows)
        print(univariate_table.to_string(float_format=lambda v: f"{v:.4f}"))

    print("\n  -- Combined (one multivariate Logit, all features together) --")
    fit = _fit_logit_report(y, X, "multivariate")
    if fit is not None:
        table, result = fit
        print(table.to_string(float_format=lambda v: f"{v:.4f}"))
        print(f"\n  pseudo R^2={result.prsquared:.4f}, log-likelihood={result.llf:.2f}, n={int(result.nobs)}")


def predictive_model(train_pool: pd.DataFrame, holdout: pd.DataFrame) -> None:
    print("\n=== Walk-forward-validated predictive model (sklearn LogisticRegression) ===")
    X_train = dfs_ml.hitter_feature_matrix(train_pool)
    y_train = train_pool["Got_Hit"]
    dates_train = train_pool["date"]

    search = ml_models.grid_search_walk_forward(
        X_train, y_train, dates_train, LogisticRegression(max_iter=1000),
        {"C": config.HITTER_HIT_LOGIT_C_GRID},
        config.ML_WALK_FORWARD_MIN_TRAIN_DATES_HITTER, config.ML_WALK_FORWARD_TEST_BLOCK_DATES_HITTER,
        scoring="neg_log_loss",
    )
    print(f"  Logistic Regression best_params={search.best_params_} cv_score={search.best_score_:.4f}")

    X_holdout = dfs_ml.hitter_feature_matrix(holdout)
    predicted_proba = search.predict_proba(X_holdout)[:, 1]
    result = ml_models.evaluate_classifier_predictions(holdout["Got_Hit"], predicted_proba)
    heuristic_result = ml_models.evaluate_classifier_predictions(
        holdout["Got_Hit"], holdout["Game_Hit_Probability"].clip(0, 1)
    )

    print(
        f"  Model (holdout):     log_loss={result['log_loss']:.4f}, brier={result['brier_score']:.4f}, "
        f"roc_auc={result['roc_auc']:.4f}, accuracy={result['accuracy']:.4f} (n={result['n']})"
    )
    print(f"  Naive baseline:      log_loss={result['baseline_log_loss']:.4f} (always predict base rate)")
    print(
        f"  Game_Hit_Probability heuristic: log_loss={heuristic_result['log_loss']:.4f}, "
        f"brier={heuristic_result['brier_score']:.4f}, roc_auc={heuristic_result['roc_auc']:.4f}"
    )

    beats_baseline = result["log_loss"] < result["baseline_log_loss"]
    beats_heuristic = result["log_loss"] < heuristic_result["log_loss"]
    if beats_baseline and beats_heuristic:
        ml_models.save_model(search.best_estimator_, config.HITTER_HIT_PROBABILITY_MODEL_PATH)
        print(
            f"  -> SAVED to {config.HITTER_HIT_PROBABILITY_MODEL_PATH} "
            f"(beats baseline and Game_Hit_Probability - artifact only, NOT wired into live picks)"
        )
    else:
        print("  -> NOT saved (does not beat baseline and/or the Game_Hit_Probability heuristic - reported honestly)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument(
        "--days", type=int, default=None,
        help="Trim the hit log to the most recent N dates instead of the full persisted history - "
             "each date recomputes the full pipeline, which is expensive. Full history is the "
             "default (None), matching train_dfs_ml_models.py's own --days flag.",
    )
    parser.add_argument(
        "--min-plate-appearances", type=int, default=config.BACKTEST_MIN_PLATE_APPEARANCES,
        help="Exclude hit-log rows below this Total_PA before fitting - a hitter with only a "
             "handful of career plate appearances at that point has a mostly-noise WAVE/"
             "Game_Hit_Probability. Defaults to the same gate predictions.select_picks already "
             "applies (config.BACKTEST_MIN_PLATE_APPEARANCES).",
    )
    args = parser.parse_args()

    season = args.season or config.SEASON_START.year
    print(f"Assembling hitter hit log (season={season}, days={args.days or 'all'})...")
    rows = dfs_backtest.assemble_hitter_hit_log(args.raw_dir, season=season, days=args.days)

    if rows.empty:
        print("No hitter hit log rows assembled - nothing to fit.")
        return

    unfiltered_n = len(rows)
    rows = rows[rows["Total_PA"] >= args.min_plate_appearances]
    print(
        f"Filtered to Total_PA >= {args.min_plate_appearances}: {len(rows)} of {unfiltered_n} rows kept."
    )
    if rows.empty:
        print("No rows clear the plate-appearance threshold - nothing to fit.")
        return

    significance_report(rows)

    train_pool, holdout, dates = _split_holdout(rows, config.ML_FINAL_HOLDOUT_DATES)
    print(f"\n{len(dates)} distinct dates, {len(train_pool)} train-pool rows, {len(holdout)} holdout rows")
    if holdout.empty or train_pool.empty:
        print("Not enough distinct dates for a real holdout split - skipping the predictive model.")
        return

    predictive_model(train_pool, holdout)


if __name__ == "__main__":
    main()
