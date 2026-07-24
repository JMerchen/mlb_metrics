"""ML-based replacement for age_curve.py's weakest metric: HR9 (real
backtest, config.py's Age Curves section - MAE indistinguishable from the
naive "guess the sample mean" baseline). The existing KNN comparable
search (age_curve.find_comparables) uses ONE dimension (HR9's own value)
to find nearest-age comparable seasons; this tries a genuinely
multi-dimensional regression instead - age, IP, K9, BB9, HR9, FIP all at
once - to see whether a model can actually do better than "look at
similar pitchers' HR9 alone."

Honest expectation, stated up front (not just after the fact): single-
season HR9 is substantially driven by batted-ball luck, park, and defense,
none of which any model built purely from a pitcher's own aggregate rate
stats can see - this is plausibly the least fixable of this project's four
flagged weak signals in ANY reasonable model, not just under-modeled by
KNN specifically. This module exists to make a real, honestly-reported
attempt, not to promise a fix.

Deliberately does NOT add multi-year lag/trend features (e.g. previous
season's own HR9) for v1 - keeps the comparison to "same KNN inputs, but
via regression instead of nearest-neighbor averaging" clean. Lag features
are a documented possible follow-up, not built here.

Year-blocked cross-validation (build_year_blocked_splits) is the
Lahman-1871-2025-scale analogue of ml_models.WalkForwardDateSplit - blocks
consecutive SEASONS together instead of individual dates, since a
player's within-season stats obviously don't apply day-to-day the way
DFS's daily signals do."""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

FEATURE_COLUMNS = ["age", "IP", "K9", "BB9", "HR9", "FIP"]


class YearBlockedSplit:
    """sklearn-compatible CV splitter, grouping by `yearID` (a season)
    instead of a single date - Lahman spans 1871-2025, so this naturally
    produces far more folds than DFS's ~116-day walk-forward split, which
    is why a wider grid search is comparatively more defensible here (see
    config.AGE_CURVE_HR9_RIDGE_ALPHA_GRID's docstring) - it's about real
    fold count, not a contradiction of the general small-sample caution
    elsewhere in this project's ML work."""

    def __init__(self, years, min_train_years: int, test_block_years: int):
        self.years = pd.Series(years).reset_index(drop=True)
        self.min_train_years = min_train_years
        self.test_block_years = test_block_years

    def _unique_sorted_years(self):
        return sorted(self.years.unique())

    def split(self, X=None, y=None, groups=None):
        unique_years = self._unique_sorted_years()
        cursor = self.min_train_years
        while cursor < len(unique_years):
            train_years = set(unique_years[:cursor])
            test_years = set(unique_years[cursor:cursor + self.test_block_years])
            train_idx = np.flatnonzero(self.years.isin(train_years).to_numpy())
            test_idx = np.flatnonzero(self.years.isin(test_years).to_numpy())
            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx
            cursor += self.test_block_years

    def get_n_splits(self, X=None, y=None, groups=None):
        unique_years = self._unique_sorted_years()
        n = 0
        cursor = self.min_train_years
        while cursor < len(unique_years):
            n += 1
            cursor += self.test_block_years
        return n


def build_year_blocked_splits(years, min_train_years: int, test_block_years: int) -> YearBlockedSplit:
    return YearBlockedSplit(years, min_train_years, test_block_years)


def build_hr9_training_table(historical_pitcher_seasons: pd.DataFrame) -> pd.DataFrame:
    """One row per (playerID, yearID) qualified pitcher-season that ALSO
    has a resolvable next season (age_curve.build_historical_pitcher_seasons'
    own (playerID, yearID+1) row) - features are this season's own
    FEATURE_COLUMNS (no lookahead: entirely this season's own stats, never
    next season's), label is next season's real HR9. Seasons with no
    qualified next season (retirement, injury, released, or fell below
    AGE_CURVE_MIN_IP) are excluded from training, the same
    survivorship-aware exclusion age_curve.project_next_season already
    applies to the KNN path - not trained to predict a made-up value for
    players who didn't have a real next season on record."""
    next_seasons = historical_pitcher_seasons.rename(
        columns={"yearID": "next_year", "HR9": "next_HR9"}
    )[["playerID", "next_year", "next_HR9"]]

    table = historical_pitcher_seasons.assign(next_year=historical_pitcher_seasons["yearID"] + 1).merge(
        next_seasons, on=["playerID", "next_year"], how="inner"
    )
    return table[["playerID", "yearID"] + FEATURE_COLUMNS + ["next_HR9"]]


def grid_search_year_blocked(X, y, years, estimator, param_grid, min_train_years, test_block_years, scoring="neg_mean_absolute_error") -> GridSearchCV:
    """Same idea as ml_models.grid_search_walk_forward, but keyed on
    YearBlockedSplit instead of WalkForwardDateSplit."""
    splitter = build_year_blocked_splits(years, min_train_years, test_block_years)
    search = GridSearchCV(estimator, param_grid, cv=splitter, scoring=scoring)
    search.fit(X, y)
    return search


def project_next_season_ml(model, player_row: pd.Series) -> dict:
    """Point projection for HR9's ML path, in the same dict shape
    age_curve.project_next_season returns so scripts/build_age_curves.py
    can use either interchangeably for a given (player, metric) row.

    Unlike the KNN path, a single regression has no natural empirical
    spread across neighbors - projected_value_p25/p75 are set equal to
    the point projection itself (a documented simplification, not a
    fabricated distribution) rather than a real spread;
    n_comparables/n_with_next_season are NaN (not applicable to a
    regression path) rather than reused in a way that would misleadingly
    imply a comparable count."""
    X = pd.DataFrame([player_row[FEATURE_COLUMNS].to_dict()])
    predicted = float(model.predict(X)[0])
    return {
        "n_comparables": float("nan"),
        "n_with_next_season": float("nan"),
        "projected_value_mean": predicted,
        "projected_value_p25": predicted,
        "projected_value_p75": predicted,
    }
