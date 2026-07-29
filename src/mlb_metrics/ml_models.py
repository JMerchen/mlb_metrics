"""Shared machine-learning infrastructure for the weak-signal follow-up
(DFS hitter/pitcher projections, Age Curves HR9) - written once here rather
than reimplemented per signal, the same way matchup.py's log5/clip-and-blend
helpers are shared between the hitter and team-level models.

The one thing every one of these training jobs needs and sklearn doesn't
provide out of the box: a walk-forward, date-respecting cross-validation
split. sklearn's default K-fold (and GridSearchCV's default CV) shuffles
rows randomly, which would let near-identical rows from the same date (or
adjacent dates - a batter's form barely changes day to day) land in both
the train and test fold, leaking information a live model would never
actually have. Every other backtest in this project
(dfs_backtest.backtest_dfs_projections, game_picks_backtest, age_curve's
backtest_projection_accuracy) enforces "only data strictly before the test
date" - WalkForwardDateSplit is that same discipline, wired into sklearn's
CV protocol so GridSearchCV respects it automatically.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV


class WalkForwardDateSplit:
    """sklearn-compatible CV splitter (get_n_splits/split). Blocks
    consecutive TEST dates together into folds, growing the training window
    forward in time - fold 1 trains on the earliest `min_train_dates` dates
    and tests on the next `test_block_dates`; fold 2 trains on everything up
    through fold 1's test block and tests on the following
    `test_block_dates`; and so on. Never puts a later date in train while an
    earlier date is in test.

    `dates` must be a sequence aligned POSITIONALLY with whatever X is
    passed to `.split()` - GridSearchCV's `split(X, y, groups)` call doesn't
    carry dates itself, so this stores dates at construction time and
    assumes the caller passes X/y to grid_search_walk_forward in the same
    row order `dates` was built from.
    """

    def __init__(self, dates, min_train_dates: int, test_block_dates: int):
        self.dates = pd.Series(dates).reset_index(drop=True)
        self.min_train_dates = min_train_dates
        self.test_block_dates = test_block_dates

    def _unique_sorted_dates(self):
        return sorted(self.dates.unique())

    def split(self, X=None, y=None, groups=None):
        unique_dates = self._unique_sorted_dates()
        cursor = self.min_train_dates
        while cursor < len(unique_dates):
            train_dates = set(unique_dates[:cursor])
            test_dates = set(unique_dates[cursor:cursor + self.test_block_dates])
            train_idx = np.flatnonzero(self.dates.isin(train_dates).to_numpy())
            test_idx = np.flatnonzero(self.dates.isin(test_dates).to_numpy())
            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx
            cursor += self.test_block_dates

    def get_n_splits(self, X=None, y=None, groups=None):
        unique_dates = self._unique_sorted_dates()
        n = 0
        cursor = self.min_train_dates
        while cursor < len(unique_dates):
            n += 1
            cursor += self.test_block_dates
        return n


def grid_search_walk_forward(
    X, y, dates, estimator, param_grid, min_train_dates, test_block_dates,
    scoring: str = "neg_mean_absolute_error",
) -> GridSearchCV:
    """Fits a GridSearchCV using WalkForwardDateSplit as the CV splitter -
    the one place a non-default (date-respecting) CV needs to be wired in,
    so every signal's training script gets no-lookahead grid search without
    hand-rolling the split logic per signal. Returns the fitted
    GridSearchCV (`.best_estimator_`/`.best_params_`/`.cv_results_` are all
    available on the result, same as any sklearn GridSearchCV)."""
    splitter = WalkForwardDateSplit(dates, min_train_dates, test_block_dates)
    search = GridSearchCV(estimator, param_grid, cv=splitter, scoring=scoring)
    search.fit(X, y)
    return search


def evaluate_predictions(actual: pd.Series, predicted: pd.Series) -> dict:
    """MAE, correlation, naive-baseline MAE (always guess actual's own
    mean), and n - the exact metric set scripts/backtest_dfs_rankings.py's
    _report and scripts/backtest_age_curve.py's run_one_metric already use,
    factored out once so a model-vs-heuristic comparison is apples-to-apples
    against the identical formula, not a re-derived one."""
    actual = pd.Series(actual).reset_index(drop=True)
    predicted = pd.Series(predicted).reset_index(drop=True)
    n = len(actual)
    if n == 0:
        return {"mae": float("nan"), "baseline_mae": float("nan"), "correlation": float("nan"), "n": 0}

    mae = (actual - predicted).abs().mean()
    baseline_mae = (actual - actual.mean()).abs().mean()
    correlation = actual.corr(predicted) if n > 1 else float("nan")
    return {"mae": mae, "baseline_mae": baseline_mae, "correlation": correlation, "n": n}


def evaluate_classifier_predictions(actual: pd.Series, predicted_proba: pd.Series) -> dict:
    """accuracy (@0.5 threshold), log_loss, brier_score, roc_auc,
    baseline_log_loss (always predict actual's own base rate), and n - the
    classification analog of evaluate_predictions, for a binary label and a
    predicted probability in [0, 1]. roc_auc is NaN (not raised) when
    `actual` is single-class, the same "can't compute, don't crash" stance
    evaluate_predictions takes on an empty input."""
    actual = pd.Series(actual).reset_index(drop=True)
    predicted_proba = pd.Series(predicted_proba).reset_index(drop=True)
    n = len(actual)
    if n == 0:
        return {
            "accuracy": float("nan"), "log_loss": float("nan"), "brier_score": float("nan"),
            "roc_auc": float("nan"), "baseline_log_loss": float("nan"), "n": 0,
        }

    base_rate = actual.mean()
    predicted_class = (predicted_proba >= 0.5).astype(int)
    baseline_proba = pd.Series(base_rate, index=actual.index)

    return {
        "accuracy": accuracy_score(actual, predicted_class),
        "log_loss": log_loss(actual, predicted_proba, labels=[0, 1]),
        "brier_score": brier_score_loss(actual, predicted_proba),
        "roc_auc": roc_auc_score(actual, predicted_proba) if actual.nunique() > 1 else float("nan"),
        "baseline_log_loss": log_loss(actual, baseline_proba, labels=[0, 1]),
        "n": n,
    }


def save_model(model, path: str) -> None:
    """joblib.dump, creating parent directories as needed - matches this
    project's existing pattern of committing binary data artifacts to git
    (data/raw/*.parquet, data/raw/lahman/*.parquet)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str):
    """Returns None (never raises) on any load failure - missing file,
    corrupt file, a scikit-learn version mismatch - so callers can use the
    same "fall back to the heuristic" pattern pipeline.run() already uses
    for a failed schedule fetch, rather than crashing the daily build."""
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None
