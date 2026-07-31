"""Opponent-offense adjustment for DFS pitcher projections
(dfs.compute_pitcher_dk_points, docs/dfs.html pitcher table) - pitchers,
unlike hitters, previously had NO opponent-quality signal anywhere, not
even in the base mean projection (K9/BB9/HR9/IP_per_start are all windowed
averages of the PITCHER'S OWN recent form). A real, user-observed gap: a
pitcher's boom-for-the-price profile projected identically whether facing
a last-place offense or a first-place one.

Opponent_Offense_Ratio blends the opponent's real team_bases_pg
(teams.compute_offensive_edge's pure-offense half - see that function's
docstring for why the already-existing offensive_edge/true_power are NOT
reusable here, both contaminated by netting out a STALE opponent from the
team's own last game, not today's actual matchup) against the league
average, at a weight validated by backtest_pitcher_matchup_signal below
rather than guessed - see config.PITCHER_MATCHUP_OFFENSE_WEIGHT's
docstring for the real numbers and go/no-go decision from that backtest.

v1 scope: only Expected_H_Allowed/Expected_ER are scaled by the ratio
(compute_opponent_adjusted_pitcher_points). Expected_K/Expected_BB/
Expected_IP are left unadjusted - a tougher offense plausibly draws more
walks or shortens a start too, but there's no existing per-opponent signal
in this project for either of those yet, and scaling H_Allowed/ER alone
(the two categories a bases-scored rate most directly predicts) is the
smallest, most defensible first step.
"""

import pandas as pd

from mlb_metrics import config


def compute_opponent_offense_ratio(
    opponent_bases_pg: pd.Series, league_bases_pg: float | pd.Series, weight: float
) -> pd.Series:
    """Ratio of an opponent's real team_bases_pg to the league average,
    blended toward a neutral 1.0 by `weight` (0 = fully off, 1 = the full
    unblended ratio), then clipped to config.PITCHER_MATCHUP_OFFENSE_CLIP.
    weight=0.0 returns EXACTLY 1.0 for every row regardless of
    `opponent_bases_pg` - the built-in null hypothesis that reproduces
    today's unadjusted heuristic, not just an approximation of it.
    `league_bases_pg` is a single scalar for live use (one as-of-date
    confidence.csv snapshot); backtest_pitcher_matchup_signal below passes
    a per-row Series instead, since each backtest date has its OWN
    no-lookahead league average, not one shared across the whole sample."""
    raw_ratio = opponent_bases_pg / league_bases_pg
    blended = 1 + weight * (raw_ratio - 1)
    lo, hi = config.PITCHER_MATCHUP_OFFENSE_CLIP
    return blended.clip(lo, hi)


def attach_opponent_offense(
    pitchers: pd.DataFrame, confidence: pd.DataFrame, weight: float
) -> pd.DataFrame:
    """Merges each pitcher's opponent's team_bases_pg (via `pitchers`'
    `opponent` -> `confidence`'s `team`) and adds Opponent_Bases_PG/
    Opponent_Offense_Ratio columns. An opponent missing from `confidence`
    (shouldn't happen for a real scheduled game, but a test fixture or a
    stale/partial confidence.csv could lack one) falls back to the league
    average bases_pg - a neutral ratio, not a dropped row, same fallback
    philosophy as MATCHUP_LEAGUE_PAVE_FALLBACK elsewhere in this project."""
    league_bases_pg = confidence["team_bases_pg"].mean()
    opponent_bases = confidence[["team", "team_bases_pg"]].rename(
        columns={"team": "opponent", "team_bases_pg": "Opponent_Bases_PG"}
    )
    merged = pitchers.merge(opponent_bases, on="opponent", how="left")
    merged["Opponent_Bases_PG"] = merged["Opponent_Bases_PG"].fillna(league_bases_pg)
    merged["Opponent_Offense_Ratio"] = compute_opponent_offense_ratio(
        merged["Opponent_Bases_PG"], league_bases_pg, weight
    )
    return merged


def compute_opponent_adjusted_pitcher_points(pitchers: pd.DataFrame, offense_ratio: pd.Series) -> pd.DataFrame:
    """Returns a copy of `pitchers` with Expected_H_Allowed/Expected_ER
    scaled by `offense_ratio` and DK_Points_Pitcher recomputed from the
    scaled components (Expected_K/Expected_BB/Expected_IP left as-is - see
    module docstring's v1 scope note). Uses the same DK_DK_PITCHER_*_POINTS
    weights as dfs.compute_pitcher_dk_points itself, so this stays in sync
    with that formula rather than duplicating a stale copy."""
    df = pitchers.copy()
    df["Expected_H_Allowed"] = df["Expected_H_Allowed"] * offense_ratio
    df["Expected_ER"] = df["Expected_ER"] * offense_ratio
    df["DK_Points_Pitcher"] = (
        df["Expected_IP"] * config.DFS_DK_PITCHER_IP_POINTS
        + df["Expected_K"] * config.DFS_DK_PITCHER_K_POINTS
        + df["Expected_BB"] * config.DFS_DK_PITCHER_BB_POINTS
        + df["Expected_H_Allowed"] * config.DFS_DK_PITCHER_H_POINTS
        + df["Expected_ER"] * config.DFS_DK_PITCHER_ER_POINTS
    )
    return df


def backtest_pitcher_matchup_signal(
    raw_dir: str = "data/raw",
    season: int | None = None,
    days: int = 20,
    weight_grid: list[float] | None = None,
) -> dict:
    """No-lookahead backtest, one row per (date, pitcher) scored, reusing
    dfs_backtest.backtest_dfs_projections' own pitcher output (already
    widened to carry Opponent_Bases_PG/League_Bases_PG per row - see that
    function's docstring). For each weight in `weight_grid`, recomputes
    Opponent_Offense_Ratio and the adjusted DK_Points_Pitcher, then reports
    correlation and mean absolute error against that date's REAL
    Actual_DK_Points_Modeled.

    Unlike the hitter-side boom backtests (dfs_ceiling.py), this asks a
    mean-projection-ACCURACY question, not a boom-flagging question -
    "top-decile capture rate" doesn't have a clean analog here, so
    correlation + MAE are the reported metrics; a real go/no-go bar is
    beating weight=0.0's own correlation AND MAE, not just correlating with
    the outcome at all (any weight will correlate somewhat, since it's
    built from real projections).

    Returns {"n": 0} (too little scored data) or
    {"n": int, "by_weight": {weight: {"correlation": float, "mae": float}}}."""
    from mlb_metrics import dfs_backtest

    weight_grid = weight_grid if weight_grid is not None else config.PITCHER_MATCHUP_WEIGHT_GRID
    heuristic = dfs_backtest.backtest_dfs_projections(raw_dir, season, days)
    df = heuristic["pitchers"]
    if df.empty:
        return {"n": 0}

    required = {"Opponent_Bases_PG", "League_Bases_PG", "Expected_IP", "Expected_K", "Expected_BB", "Actual_DK_Points_Modeled"}
    if not required.issubset(df.columns):
        return {"n": 0}

    df = df.dropna(subset=list(required)).copy()
    n = len(df)
    if n < 2:
        return {"n": n}

    by_weight = {}
    for weight in weight_grid:
        ratio = compute_opponent_offense_ratio(df["Opponent_Bases_PG"], df["League_Bases_PG"], weight)
        adjusted = compute_opponent_adjusted_pitcher_points(df, ratio)
        correlation = adjusted["DK_Points_Pitcher"].corr(adjusted["Actual_DK_Points_Modeled"])
        mae = (adjusted["DK_Points_Pitcher"] - adjusted["Actual_DK_Points_Modeled"]).abs().mean()
        by_weight[weight] = {
            "correlation": float(correlation) if pd.notna(correlation) else None,
            "mae": float(mae),
        }

    return {"n": n, "by_weight": by_weight}
