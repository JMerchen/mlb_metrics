"""Opponent-defense adjustment for NFL skill-player projections
(consumed by nfl_dfs.py, Phase 4) - direct structural port of
pitcher_matchup.py's opponent-offense adjustment, applied in the
opposite direction: there, a PITCHER's projection is scaled by the
opponent OFFENSE's quality; here, a QB/RB/WR/TE's projection is scaled
by the opponent DEFENSE's quality (nfl_teams.compute_defense_rolling_rates'
output).

Hard invariant (the guiding principle this whole module exists to
enforce): a player's opponent for `attach_matchup_adjustment` ALWAYS
comes from `current_week_schedule_df` - never from
`defense_rates_df` itself, which deliberately carries no opponent
column at all (see nfl_teams.py's docstring). This is the exact bug
class teams.compute_offensive_edge's own docstring warns about (netting
out a team's own STALE last-played opponent instead of the real upcoming
one) - structurally impossible to reintroduce here since there's no
"last opponent" column anywhere in this module's inputs to accidentally
reach for.

Separate ratios per category (pass-yards-allowed for QB/WR/TE,
rush-yards-allowed for RB, receptions-allowed as an additional
informational column for pass-catchers) rather than one composite
number - mirrors dfs.py's per-category philosophy (Expected_H_Allowed/
Expected_ER are separate columns, not blended into one score) rather
than inventing a new composite metric.
"""

import pandas as pd

from mlb_metrics import config

_RATIO_CATEGORIES = {
    "pass_yards_allowed_per_game": "Opponent_Pass_Yards_Allowed_Ratio",
    "rush_yards_allowed_per_game": "Opponent_Rush_Yards_Allowed_Ratio",
    "receptions_allowed_per_game": "Opponent_Receptions_Allowed_Ratio",
}


def compute_opponent_adjustment_ratio(
    opponent_rate: pd.Series, league_rate: float | pd.Series, weight: float
) -> pd.Series:
    """Ratio of an opponent defense's real allowed-rate to the league
    average, blended toward a neutral 1.0 by `weight` (0 = fully off, 1 =
    the full unblended ratio), clipped to config.NFL_MATCHUP_OFFENSE_CLIP.
    weight=0.0 returns EXACTLY 1.0 for every row regardless of
    `opponent_rate` - the built-in null hypothesis that reproduces
    today's unadjusted projection, not just an approximation of it.
    `league_rate` is a single scalar for live use; a backtest instead
    passes a per-row Series, since each backtest date has its OWN
    no-lookahead league average, not one shared across the whole sample
    (same convention as pitcher_matchup.compute_opponent_offense_ratio)."""
    raw_ratio = opponent_rate / league_rate
    blended = 1 + weight * (raw_ratio - 1)
    lo, hi = config.NFL_MATCHUP_OFFENSE_CLIP
    return blended.clip(lo, hi)


def _team_opponents(current_week_schedule_df: pd.DataFrame) -> pd.DataFrame:
    """One row per team playing in `current_week_schedule_df`: [team,
    opponent]. Both home and away sides are exploded so every playing
    team gets its real upcoming opponent - the ONLY source of "opponent"
    this module ever reads from (see module docstring)."""
    home = current_week_schedule_df[["home_team", "away_team"]].rename(
        columns={"home_team": "team", "away_team": "opponent"}
    )
    away = current_week_schedule_df[["away_team", "home_team"]].rename(
        columns={"away_team": "team", "home_team": "opponent"}
    )
    return pd.concat([home, away], ignore_index=True)


def attach_matchup_adjustment(
    players_df: pd.DataFrame,
    defense_rates_df: pd.DataFrame,
    current_week_schedule_df: pd.DataFrame,
    weight: float,
) -> pd.DataFrame:
    """Merges each player's THIS-WEEK opponent (via `players_df`'s `team`
    -> `current_week_schedule_df`) and attaches, per category in
    _RATIO_CATEGORIES: the opponent's raw allowed-rate and its blended
    ratio. An opponent missing from `defense_rates_df` (a test fixture,
    or a stale/partial rates table) falls back to the league average
    rate - a neutral ratio, not a dropped row, same fallback philosophy
    as pitcher_matchup.attach_opponent_offense."""
    opponents = _team_opponents(current_week_schedule_df)
    merged = players_df.merge(opponents, on="team", how="left")

    defense_by_opponent = defense_rates_df.rename(columns={"team": "opponent"})
    merged = merged.merge(
        defense_by_opponent[["opponent", *_RATIO_CATEGORIES.keys()]], on="opponent", how="left"
    )

    for rate_col, ratio_col in _RATIO_CATEGORIES.items():
        league_rate = defense_rates_df[rate_col].mean()
        merged[rate_col] = merged[rate_col].fillna(league_rate)
        merged[ratio_col] = compute_opponent_adjustment_ratio(merged[rate_col], league_rate, weight)

    return merged
