"""Real fix for a real, confirmed structural defect in
`nfl_game_picks.compute_game_win_probabilities`: its ratio formula
(`home_rating / (home_rating + away_rating)`) has NO free scale
parameter, and both ratings are z-normalized composites clustered around
1.0 with a real, confirmed cross-team std of only ~0.075 - so even the
single best real team hosting the single worst real team (the most
extreme mismatch possible) works out to ~59%. The model structurally
CANNOT express a real blowout's true confidence, no matter how much real
historical data exists to learn from (a real, concrete complaint: a
"barely competent" rookie-QB team against "a juggernaut that eventually
won the Super Bowl" came out "almost a coin flip"). Recalibration alone
can't fix this either - `train_nfl_game_pick_calibration.py` already
tried rescaling this same compressed range and failed to beat the raw
heuristic on a real holdout.

Real precedent for the actual fix already exists in this codebase on the
MLB side (`train_game_pick_model.py` - fits a real walk-forward-validated
LogisticRegression/HistGradientBoostingClassifier directly on the same
raw composite ingredients the ratio formula uses, giving the model a real
LEARNED scale a fixed ratio never has) - but MLB's own version is
explicitly parked, never wired into live picks. This script ports that
exact methodology to NFL and - unlike MLB's own version - actually wires
the validated result into live picks via `nfl_game_picks.apply_ml_model`
if a real candidate clears the save-gate.

Two real feature-set candidates (`nfl_game_picks.build_game_features`'s
minimal composite-only set, and `build_game_features_disaggregated`'s
richer per-signal set - see that function's own docstring), each swept
through LogisticRegression and HistGradientBoostingClassifier via the
same real walk-forward CV every other ML fit in this project uses
(`ml_models.grid_search_walk_forward`), keyed on a real `season*100+week`
period so the walk continues across season boundaries (same trick the
pooled NFL calibration exploration used). Final holdout is the full real
most recent season (2025, all 18 real weeks) rather than a token few-week
slice - config.NFL_GAME_PICK_ML_WIN_PROBABILITY_FINAL_HOLDOUT_WEEKS
documents why, now that 9 real prior seasons of train data exist.

**Save gate**: the winning candidate must beat BOTH a naive baseline
(always predict the base home-win rate) AND today's real live heuristic
(the ratio + validated home-field term, computed independently for the
SAME holdout games) on holdout log_loss. Also prints a real, explicit
predicted-probability SPREAD comparison (min/p5/median/p95/max) between
the heuristic and the new model on the same holdout - the direct,
concrete check for whether this fixes the actual blowout-confidence
complaint, not just the aggregate metrics.

If a candidate clears the gate, it is refit on the FULL real dataset
(train + holdout combined) before saving - same "use every real data
point once validated" precedent `train_nfl_game_pick_calibration.py`
already establishes in this exact codebase area - and saved as a real
`{"model", "feature_columns"}` dict to
config.NFL_GAME_PICK_WIN_PROBABILITY_MODEL_PATH, which
`nfl_game_picks.apply_ml_model` loads live (graceful no-op if this script
is never run, or nothing clears the gate).

Needs data/raw/nfl/{schedules,team_stats,weekly,snap_counts,
rosters_weekly,pbp}_<season>.parquet for every season in SEASONS -
already fully cached (2016-2025, see PR #98). Fully offline.

Usage:
    python scripts/train_nfl_game_pick_model.py
"""

import os

import pandas as pd
import statsmodels.api as sm
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from mlb_metrics import config, ml_models, nfl_game_picks, nfl_game_picks_backtest as backtest

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "nfl")
SEASONS = list(range(2016, 2026))
HOLDOUT_SEASONS = {max(SEASONS)}  # config.NFL_GAME_PICK_ML_WIN_PROBABILITY_FINAL_HOLDOUT_WEEKS's own real season

FEATURE_SET_CANDIDATES = {
    "composite": (nfl_game_picks.build_game_features, nfl_game_picks.GAME_PICK_FEATURE_COLUMNS),
    "disaggregated": (nfl_game_picks.build_game_features_disaggregated, nfl_game_picks.DISAGGREGATED_FEATURE_COLUMNS),
}


def _load_all(table: str) -> pd.DataFrame:
    return pd.concat(
        [pd.read_parquet(os.path.join(RAW_DIR, f"{table}_{season}.parquet")) for season in SEASONS],
        ignore_index=True,
    )


def _standardize(X: pd.DataFrame) -> pd.DataFrame:
    """z-score every column; drops any zero-variance column (same guard
    train_game_pick_model.py's own `_standardize` uses)."""
    std = X.std()
    constant_columns = std[std == 0].index.tolist()
    if constant_columns:
        print(f"  Excluding constant-in-this-data feature(s): {constant_columns}")
    X = X.drop(columns=constant_columns)
    return (X - X.mean()) / X.std()


def _fit_logit_report(y: pd.Series, X: pd.DataFrame, label: str):
    try:
        result = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
    except Exception as exc:
        print(f"  {label}: fit failed ({exc})")
        return None
    table = pd.DataFrame({
        "coef": result.params, "std_err": result.bse, "z": result.tvalues, "p_value": result.pvalues,
    }).drop(index="const")
    return table, result


def significance_report(rows: pd.DataFrame, feature_columns: list, label: str) -> None:
    print(f"\n=== Feature significance report ({label}, statsmodels.Logit, full log) ===")
    y = rows["home_won"].astype(float)
    X = _standardize(rows[feature_columns].astype(float))
    print(f"  n={len(rows)}, base home-win rate={y.mean():.4f}")

    print("\n  -- Individually (one univariate Logit per feature) --")
    univariate_rows = []
    for column in X.columns:
        fit = _fit_logit_report(y, X[[column]], column)
        if fit is None:
            continue
        table, _ = fit
        univariate_rows.append(table.loc[column])
    if univariate_rows:
        print(pd.DataFrame(univariate_rows).to_string(float_format=lambda v: f"{v:.4f}"))

    print("\n  -- Combined (one multivariate Logit, all features together) --")
    fit = _fit_logit_report(y, X, "multivariate")
    if fit is not None:
        table, result = fit
        print(table.to_string(float_format=lambda v: f"{v:.4f}"))
        print(f"\n  pseudo R^2={result.prsquared:.4f}, log-likelihood={result.llf:.2f}, n={int(result.nobs)}")


def _select_best_model(X_train, y_train, periods_train):
    logit_search = ml_models.grid_search_walk_forward(
        X_train, y_train, periods_train, LogisticRegression(max_iter=1000),
        {"C": config.NFL_GAME_PICK_LOGIT_C_GRID},
        config.NFL_GAME_PICK_ML_WALK_FORWARD_MIN_TRAIN_WEEKS, config.NFL_GAME_PICK_ML_WALK_FORWARD_TEST_BLOCK_WEEKS,
        scoring="neg_log_loss",
    )
    gbm_search = ml_models.grid_search_walk_forward(
        X_train, y_train, periods_train, HistGradientBoostingClassifier(),
        config.NFL_GAME_PICK_GBM_PARAM_GRID,
        config.NFL_GAME_PICK_ML_WALK_FORWARD_MIN_TRAIN_WEEKS, config.NFL_GAME_PICK_ML_WALK_FORWARD_TEST_BLOCK_WEEKS,
        scoring="neg_log_loss",
    )
    print(f"  Logistic Regression best_params={logit_search.best_params_} cv_score={logit_search.best_score_:.4f}")
    print(f"  GBM                 best_params={gbm_search.best_params_} cv_score={gbm_search.best_score_:.4f}")
    best_search = logit_search if logit_search.best_score_ >= gbm_search.best_score_ else gbm_search
    best_name = "LogisticRegression" if best_search is logit_search else "HistGradientBoostingClassifier"
    print(f"  Selected: {best_name}")
    return best_search, best_name


def _spread_report(label: str, proba: pd.Series) -> None:
    print(
        f"  {label:22s} min={proba.min():.3f} p5={proba.quantile(0.05):.3f} p50={proba.quantile(0.5):.3f} "
        f"p95={proba.quantile(0.95):.3f} max={proba.max():.3f} std={proba.std():.4f}"
    )


def evaluate_candidate(label: str, feature_columns: list, train_pool: pd.DataFrame, holdout: pd.DataFrame) -> dict:
    print(f"\n=== {label} candidate: walk-forward-validated model ===")
    X_train = nfl_game_picks.game_feature_matrix(train_pool, feature_columns)
    y_train = train_pool["home_won"].astype(float)

    best_search, best_name = _select_best_model(X_train, y_train, train_pool["period"])

    X_holdout = nfl_game_picks.game_feature_matrix(holdout, feature_columns)
    predicted_proba = best_search.predict_proba(X_holdout)[:, 1]
    result = ml_models.evaluate_classifier_predictions(holdout["home_won"], predicted_proba)
    heuristic_result = ml_models.evaluate_classifier_predictions(
        holdout["home_won"], holdout["home_win_probability"].clip(0, 1)
    )

    print(
        f"  Model ({best_name}, holdout): log_loss={result['log_loss']:.4f} brier={result['brier_score']:.4f} "
        f"roc_auc={result['roc_auc']:.4f} accuracy={result['accuracy']:.4f} (n={result['n']})"
    )
    print(f"  Naive baseline:                    log_loss={result['baseline_log_loss']:.4f}")
    print(
        f"  Live heuristic (ratio+home-field): log_loss={heuristic_result['log_loss']:.4f} "
        f"brier={heuristic_result['brier_score']:.4f} roc_auc={heuristic_result['roc_auc']:.4f} "
        f"accuracy={heuristic_result['accuracy']:.4f}"
    )

    print("\n  Predicted-probability spread (does this fix the blowout-confidence ceiling?):")
    _spread_report("Live heuristic", holdout["home_win_probability"])
    _spread_report(f"{label} model", pd.Series(predicted_proba, index=holdout.index))

    most_lopsided = holdout.assign(_pred=predicted_proba).reindex(
        holdout["home_win_probability"].sub(0.5).abs().sort_values(ascending=False).index
    ).head(5)
    print("\n  5 most lopsided real holdout games (by how far the live heuristic strayed from 50%):")
    print(
        most_lopsided[["home_team", "away_team", "home_win_probability", "_pred", "home_won"]]
        .rename(columns={"home_win_probability": "heuristic_prob", "_pred": f"{label}_model_prob"})
        .to_string(index=False, float_format=lambda v: f"{v:.3f}")
    )

    beats_baseline = result["log_loss"] < result["baseline_log_loss"]
    beats_heuristic = result["log_loss"] < heuristic_result["log_loss"]
    clears_bar = beats_baseline and beats_heuristic
    print(f"\n  Beats naive baseline: {beats_baseline} | Beats live heuristic: {beats_heuristic} | CLEARS BAR: {clears_bar}")

    return {
        "label": label, "model_name": best_name, "estimator": best_search.best_estimator_,
        "feature_columns": feature_columns, "holdout_log_loss": result["log_loss"],
        "heuristic_log_loss": heuristic_result["log_loss"], "clears_bar": clears_bar,
    }


def main():
    print(f"Loading real {SEASONS[0]}-{SEASONS[-1]} NFL data...")
    schedules = _load_all("schedules")
    team_stats = _load_all("team_stats")
    weekly = _load_all("weekly")
    snap_counts = _load_all("snap_counts")
    rosters = _load_all("rosters_weekly")
    pbp = _load_all("pbp")

    print("Building real no-lookahead multi-season history (season_aware=False - today's validated live behavior)...")
    snapshots = backtest.build_multi_season_history(schedules, team_stats, weekly, snap_counts, rosters, pbp, SEASONS)
    print(f"{len(snapshots)} real replayed weeks.")

    # Real live heuristic probability for every real replayed game - the
    # SAME games the ML candidates are scored against, for a fair "beats
    # today's live model" comparison (not a re-derived approximation).
    heuristic_rows = []
    for snap in snapshots:
        probs = nfl_game_picks.compute_game_win_probabilities(
            snap["master"], snap["qb_continuity"], snap["weekly"], snap["this_week_games"]
        )
        heuristic_rows.append(probs[["game_id", "home_win_probability"]])
    heuristic = pd.concat(heuristic_rows, ignore_index=True)

    candidate_logs = {}
    for name, (feature_fn, feature_columns) in FEATURE_SET_CANDIDATES.items():
        rows = backtest.assemble_nfl_game_pick_log(snapshots, feature_fn)
        rows = rows.merge(heuristic, on="game_id", how="left")
        rows["period"] = rows["season"] * 100 + rows["week"]
        candidate_logs[name] = rows
        significance_report(rows, feature_columns, name)

    results = []
    for name, (feature_fn, feature_columns) in FEATURE_SET_CANDIDATES.items():
        rows = candidate_logs[name]
        holdout = rows[rows["season"].isin(HOLDOUT_SEASONS)]
        train_pool = rows[~rows["season"].isin(HOLDOUT_SEASONS)]
        print(f"\n{name}: {len(train_pool)} train-pool rows, {len(holdout)} holdout rows (season {sorted(HOLDOUT_SEASONS)})")
        if holdout.empty or train_pool.empty:
            print("  Not enough rows for a real holdout split - skipping.")
            continue
        results.append(evaluate_candidate(name, feature_columns, train_pool, holdout))

    print("\n" + "=" * 100)
    winners = [r for r in results if r["clears_bar"]]
    if not winners:
        print("NO candidate cleared the real save-gate (beat both naive baseline AND today's live heuristic).")
        print("Reporting honestly: the ML win-probability fix is NOT validated by this backtest.")
        if results:
            best = min(results, key=lambda r: r["holdout_log_loss"])
            print(
                f"\nClosest candidate (lowest holdout log_loss, NOT validated): {best['label']} "
                f"({best['model_name']}), log_loss={best['holdout_log_loss']:.4f} vs. "
                f"heuristic {best['heuristic_log_loss']:.4f}"
            )
        print("\nnfl_pipeline.py's live path stays on today's ratio+home-field heuristic, unchanged.")
        return

    winner = min(winners, key=lambda r: r["holdout_log_loss"])
    print(
        f"Best validated candidate: {winner['label']} ({winner['model_name']}), "
        f"holdout log_loss={winner['holdout_log_loss']:.4f} vs. heuristic {winner['heuristic_log_loss']:.4f}"
    )

    # Refit the winning (feature-set, model-family, hyperparameters)
    # combo on the FULL real dataset (train + holdout combined) before
    # saving - same "use every real data point once validated" precedent
    # train_nfl_game_pick_calibration.py already establishes in this
    # exact codebase area.
    rows = candidate_logs[winner["label"]]
    X_full = nfl_game_picks.game_feature_matrix(rows, winner["feature_columns"])
    y_full = rows["home_won"].astype(float)
    final_estimator = clone(winner["estimator"]).fit(X_full, y_full)

    artifact = {"model": final_estimator, "feature_columns": winner["feature_columns"]}
    ml_models.save_model(artifact, config.NFL_GAME_PICK_WIN_PROBABILITY_MODEL_PATH)
    print(
        f"\n-> SAVED to {config.NFL_GAME_PICK_WIN_PROBABILITY_MODEL_PATH} "
        f"(refit on the full {len(rows):,}-game real dataset) - now live via nfl_game_picks.apply_ml_model."
    )


if __name__ == "__main__":
    main()
