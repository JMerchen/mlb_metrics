"""Preseason bestball draft-strategy signal - a genuinely different
question from the rest of the NFL DFS pipeline (`nfl_passing.py`,
`nfl_rush_rec.py`, `nfl_dfs.py`), which all answer "how many points will
this player score THIS upcoming week" via recency-weighted rolling
windows. Bestball drafting instead needs "how much value/risk did this
player represent LAST SEASON, as a whole" - a real, REALIZED season
total, not a forward-looking projection, and no rolling-window blend
(there's no "as of date" mid-draft; you have the complete season to look
back on).

Reuses `nfl_dfs_backtest.compute_actual_qb_dk_points`/
`compute_actual_skill_dk_points` directly for the points math - those
already compute REAL, realized full-PPR DK points from a real box score
"through the SAME real DK formulas" (that module's own docstring), which
is exactly what a season-total realized-points figure needs. Building a
second points calculator here would just duplicate that logic.

Games played (vs. that player's team's real games that season) is used
as a deliberately simple, honest injury-history proxy - not a full
medical history, just "how much of the season were they actually
available for" - per this feature's explicit scope (a full injury
database is out of scope; games-played is a cheap, real signal derived
entirely from already-persisted data, see config.py's NFL section for
what's on disk).
"""

import pandas as pd

from mlb_metrics import nfl_dfs_backtest, nfl_rush_rec

POSITIONS = ("QB",) + nfl_rush_rec.SKILL_POSITIONS


def compute_player_games_played(weekly_df: pd.DataFrame, schedules_df: pd.DataFrame, season: int) -> pd.DataFrame:
    """[player_id, season, team, games_played, possible_games, games_missed]
    for one real season, regular season only (season_type == "REG" on
    `weekly_df`, game_type == "REG" on `schedules_df` - postseason games
    would inflate both sides inconsistently, since not every team makes
    the playoffs).

    `games_played` is a real row count on `weekly_df` (a real week
    absent from that table means the player didn't play that week -
    bye/injury/inactive - same convention nfl_passing.py/nfl_rush_rec.py
    already rely on). `possible_games` is that player's TEAM's real game
    count that season (varies by season - 16 games through 2020, 17 from
    2021 on; always derived from the real schedule, never hardcoded).
    `team` is whichever team a player suited up for most that season -
    a mid-season trade is a real but rare edge case this deliberately
    doesn't try to split correctly; a preseason draft-strategy snapshot
    doesn't need per-team-stint precision, just "were they healthy.\""""
    season_weekly = weekly_df[(weekly_df["season"] == season) & (weekly_df["season_type"] == "REG")]

    games_played = season_weekly.groupby("player_id").size().rename("games_played")
    team = season_weekly.groupby("player_id")["team"].agg(lambda s: s.value_counts().idxmax()).rename("team")

    reg_schedule = schedules_df[(schedules_df["season"] == season) & (schedules_df["game_type"] == "REG")]
    team_games = reg_schedule.groupby("home_team").size().add(reg_schedule.groupby("away_team").size(), fill_value=0)

    result = pd.concat([games_played, team], axis=1).reset_index()
    result["possible_games"] = result["team"].map(team_games).fillna(0).astype(int)
    result["games_missed"] = (result["possible_games"] - result["games_played"]).clip(lower=0)
    result["season"] = season
    return result[["player_id", "season", "team", "games_played", "possible_games", "games_missed"]]


def compute_season_realized_dk_points(weekly_df: pd.DataFrame, position_group: str, season: int) -> pd.DataFrame:
    """[player_id, dk_points_total] - real full-PPR DK points actually
    scored across an entire real regular season, by summing
    nfl_dfs_backtest's real-week-by-real-week actual-points functions
    (each row in `weekly_df` is already one real player-week, so no
    per-week loop is needed - the same real formula just runs once over
    every row and gets summed by player)."""
    season_weekly = weekly_df[(weekly_df["season"] == season) & (weekly_df["season_type"] == "REG")]

    if position_group == "QB":
        per_week = nfl_dfs_backtest.compute_actual_qb_dk_points(season_weekly)
        points_col = "Actual_DK_Points_QB"
    elif position_group == "SKILL":
        per_week = nfl_dfs_backtest.compute_actual_skill_dk_points(season_weekly)
        points_col = "Actual_DK_Points_Skill"
    else:
        raise ValueError(f'position_group must be "QB" or "SKILL", got {position_group!r}')

    totals = per_week.groupby("player_id")[points_col].sum().reset_index()
    return totals.rename(columns={points_col: "dk_points_total"})


def build_bestball_rankings(
    weekly_df: pd.DataFrame,
    schedules_df: pd.DataFrame,
    season: int,
    prior_season: int | None = None,
    prior_weekly_df: pd.DataFrame | None = None,
    prior_schedules_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per real QB/RB/WR/TE who recorded a real stat line in
    `season`'s regular season: name, position, team, real games played/
    missed, real season-total and per-game full-PPR DK points - ranked
    by `dk_points_total` descending (a real, realized value signal, not
    a blended "draft score" - `dk_points_per_game` and `games_missed`
    are kept as separate honest columns rather than folded into one
    number, since there's no backtestable ground truth for what the
    "right" health/talent tradeoff weighting would even be).

    DST is intentionally excluded - bestball drafting doesn't need DST
    optimization the way weekly DFS does.

    If `prior_season`/`prior_weekly_df`/`prior_schedules_df` are given,
    adds `games_missed_prior_season` - a cheap, real repeat-injury-risk
    read using data already persisted for that season too."""
    games = compute_player_games_played(weekly_df, schedules_df, season)
    qb_points = compute_season_realized_dk_points(weekly_df, "QB", season)
    skill_points = compute_season_realized_dk_points(weekly_df, "SKILL", season)
    points = pd.concat([qb_points, skill_points], ignore_index=True)

    season_weekly = weekly_df[(weekly_df["season"] == season) & (weekly_df["season_type"] == "REG")]
    names = season_weekly[["player_id", "player_display_name", "position"]].drop_duplicates("player_id")
    names = names[names["position"].isin(POSITIONS)]

    result = names.merge(games, on="player_id", how="left").merge(points, on="player_id", how="inner")
    result["dk_points_per_game"] = result["dk_points_total"] / result["games_played"].replace(0, pd.NA)

    if prior_season is not None and prior_weekly_df is not None and prior_schedules_df is not None:
        prior_games = compute_player_games_played(prior_weekly_df, prior_schedules_df, prior_season)
        prior_games = prior_games[["player_id", "games_missed"]].rename(
            columns={"games_missed": "games_missed_prior_season"}
        )
        result = result.merge(prior_games, on="player_id", how="left")

    return result.sort_values("dk_points_total", ascending=False).reset_index(drop=True)
