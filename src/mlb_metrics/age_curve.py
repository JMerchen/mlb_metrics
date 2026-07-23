"""KNN-based next-season projection for the Age Curves page
(docs/age-curves.html) - an exploratory feature, entirely separate from
the daily pick pipeline (Game_Hit_Probability/Automated Game Picks).

Given a current player's age and season OPS (traditional_stats.py, not
WAVE/PAVE - Lahman's historical seasons have no Statcast-derived signal to
compare against), finds the K historical hitter-seasons (from Lahman, via
lahman_data.py) at the same age with the closest OPS, then looks at what
those comparable players actually did the FOLLOWING season - that
distribution of real outcomes is the projection, not a single confident
point estimate. Comparables with no next season (retired, hurt, released)
are excluded and that excluded fraction is reported explicitly -
survivorship bias made visible, not hidden.

No ML library is used (no scikit-learn dependency exists in this project -
see requirements.txt): K-nearest-neighbors on a single dimension is a
straightforward sort, matching this project's established pattern of
hand-implementing its own statistics (log5, z-scoring) rather than pulling
in a library for one call.

v1 scope, deliberately not overbuilt: hitters only, a single similarity
dimension (OPS), no era/park adjustment (comparing raw OPS across
different offensive eras/ballparks - e.g. the high-offense late-1990s vs.
a modern low-average era, or Coors Field vs. a pitcher's park - is a known
simplification, not fixed here). Expandable later, not designed for that
up front.
"""

import pandas as pd

from mlb_metrics import config, lahman_data


def build_historical_seasons(batting: pd.DataFrame, people: pd.DataFrame, min_at_bats: int = config.AGE_CURVE_MIN_AB) -> pd.DataFrame:
    """One row per qualified (playerID, yearID) historical season:
    [playerID, yearID, age, AB, OPS]. Lahman's Batting table has one row
    per player per team-stint per year (a mid-season trade splits a
    player's year across rows) - summed here first so a traded player's
    full season counts as one season, not two partial ones. Unqualified
    (min_at_bats) seasons are dropped - a tiny-sample season can otherwise
    show an OPS of 0 or 3.000+ on pure noise."""
    counting_cols = ["AB", "H", "2B", "3B", "HR", "BB", "HBP", "SF"]
    batting = batting.copy()
    for col in counting_cols:
        if col not in batting.columns:
            batting[col] = 0
        batting[col] = batting[col].fillna(0)

    season = batting.groupby(["playerID", "yearID"], as_index=False)[counting_cols].sum()

    # Total bases = H (every hit counts once) + one extra base per double,
    # two extra per triple, three extra per home run.
    season["TB"] = season["H"] + season["2B"] + 2 * season["3B"] + 3 * season["HR"]
    ab_safe = season["AB"].replace(0, pd.NA)
    season["SLG"] = (season["TB"] / ab_safe).fillna(0)

    plate_appearances = (season["AB"] + season["BB"] + season["HBP"] + season["SF"]).replace(0, pd.NA)
    season["OBP"] = ((season["H"] + season["BB"] + season["HBP"]) / plate_appearances).fillna(0)
    season["OPS"] = season["OBP"] + season["SLG"]

    season = season[season["AB"] >= min_at_bats]

    aged = lahman_data.attach_age(season, people)
    return aged[["playerID", "yearID", "age", "AB", "OPS"]]


def find_comparables(
    current_age: int,
    current_ops: float,
    historical_seasons: pd.DataFrame,
    k: int = config.AGE_CURVE_K_NEIGHBORS,
    age_window: int = config.AGE_CURVE_AGE_WINDOW,
) -> pd.DataFrame:
    """The `k` historical seasons (from `historical_seasons` - see
    build_historical_seasons) within `age_window` years of `current_age`,
    ranked by absolute OPS distance, nearest first. Ties are broken by
    whichever row appears first in `historical_seasons` (a stable sort),
    not randomly."""
    pool = historical_seasons[
        (historical_seasons["age"] >= current_age - age_window)
        & (historical_seasons["age"] <= current_age + age_window)
    ].copy()
    pool["distance"] = (pool["OPS"] - current_ops).abs()
    return pool.sort_values("distance", kind="stable").head(k)


def project_next_season(comparables: pd.DataFrame, historical_seasons: pd.DataFrame) -> dict:
    """Looks up each comparable's own (playerID, yearID + 1) row in
    `historical_seasons` - i.e. what they actually did the following
    season - and summarizes the resolved outcomes.

    Returns a dict: n_comparables, n_with_next_season (comparables who
    played a qualified next season at all; the rest didn't - retirement,
    injury, release, or simply falling below AGE_CURVE_MIN_AB - and are
    excluded from the projection rather than silently assumed to have
    continued at the same level), projected_ops_mean, projected_ops_p25,
    projected_ops_p75 (NaN for all three if no comparable has a next
    season)."""
    next_seasons = historical_seasons.rename(columns={"yearID": "next_year", "OPS": "next_OPS"})[
        ["playerID", "next_year", "next_OPS"]
    ]
    lookup = comparables.assign(next_year=comparables["yearID"] + 1).merge(
        next_seasons, on=["playerID", "next_year"], how="left"
    )
    resolved = lookup.dropna(subset=["next_OPS"])

    return {
        "n_comparables": len(comparables),
        "n_with_next_season": len(resolved),
        "projected_ops_mean": float(resolved["next_OPS"].mean()) if len(resolved) else float("nan"),
        "projected_ops_p25": float(resolved["next_OPS"].quantile(0.25)) if len(resolved) else float("nan"),
        "projected_ops_p75": float(resolved["next_OPS"].quantile(0.75)) if len(resolved) else float("nan"),
    }


def backtest_projection_accuracy(
    historical_seasons: pd.DataFrame,
    test_seasons: pd.DataFrame,
    k: int = config.AGE_CURVE_K_NEIGHBORS,
    age_window: int = config.AGE_CURVE_AGE_WINDOW,
) -> pd.DataFrame:
    """Validates the projection METHOD (not a live prediction - this is an
    exploratory page, not part of the daily pick pipeline) against real
    history: for each row in `test_seasons` (a sample of real past
    player-seasons that have a known actual next season), projects that
    next season using ONLY comparable seasons from a year at or before the
    test season's own year (excluding the test player's own row) - a
    stricter, no-lookahead comparable pool, matching this project's
    established backtest discipline elsewhere (git_backtest.py,
    game_picks_backtest.py) - then compares the projection to what
    actually happened.

    Test seasons with no real next-season row (the player's career simply
    ended there within Lahman's data) can't be scored and are skipped -
    there's nothing to compare the projection against.

    Returns one row per scored test season: [playerID, yearID,
    actual_next_OPS, projected_ops_mean, n_with_next_season, abs_error]."""
    next_lookup = historical_seasons.rename(columns={"yearID": "next_year", "OPS": "actual_next_OPS"})[
        ["playerID", "next_year", "actual_next_OPS"]
    ]
    scorable = test_seasons.assign(next_year=test_seasons["yearID"] + 1).merge(
        next_lookup, on=["playerID", "next_year"], how="inner"
    )

    rows = []
    for _, row in scorable.iterrows():
        pool = historical_seasons[
            (historical_seasons["yearID"] <= row["yearID"])
            & ~(
                (historical_seasons["playerID"] == row["playerID"])
                & (historical_seasons["yearID"] == row["yearID"])
            )
        ]
        comparables = find_comparables(row["age"], row["OPS"], pool, k=k, age_window=age_window)
        projection = project_next_season(comparables, historical_seasons)
        rows.append(
            {
                "playerID": row["playerID"],
                "yearID": row["yearID"],
                "actual_next_OPS": row["actual_next_OPS"],
                "projected_ops_mean": projection["projected_ops_mean"],
                "n_with_next_season": projection["n_with_next_season"],
            }
        )

    result = pd.DataFrame(rows, columns=["playerID", "yearID", "actual_next_OPS", "projected_ops_mean", "n_with_next_season"])
    result["abs_error"] = (result["projected_ops_mean"] - result["actual_next_OPS"]).abs()
    return result


def league_age_curve(historical_seasons: pd.DataFrame) -> pd.DataFrame:
    """One row per age: [age, OPS, n_seasons] - the league-average OPS at
    that age across every qualified historical season, for the Age Curves
    page's background "typical aging curve" chart line."""
    curve = historical_seasons.groupby("age", as_index=False).agg(OPS=("OPS", "mean"), n_seasons=("OPS", "size"))
    return curve.sort_values("age").reset_index(drop=True)
