"""DST (Defense/Special Teams) DK scoring - a genuinely different shape
from the player-level modules (a team-level unit, no per-player rolling
stat), so its own parallel module rather than forced through
nfl_passing.py/nfl_rush_rec.py's abstraction.

Box-score categories (sacks/INTs/fumble recoveries/safeties/blocked
kicks/defensive+return TDs) come straight from nfl_data.fetch_team_stats
(nflreadpy's load_team_stats) - confirmed live to already carry
ready-made per-team-per-week defense/special-teams stats, so NO
play-by-play aggregation is needed here, contrary to the original plan's
assumption that PBP aggregation would likely be required (a real,
positive scoping simplification - see nfl_data.py's module docstring).

`defensive_tds` sums THREE separate real columns - `def_tds`,
`fumble_recovery_tds`, `special_teams_tds` - confirmed live (spot-checked
against real 2025 per-team-week rows) to be non-overlapping: a game with
a fumble-recovery TD but `def_tds == 0` exists in the real data, proving
`def_tds` does NOT already include fumble-recovery TDs. DK scores every
one of these events identically (+6), so getting the TOTAL right matters
here, not preserving which specific column each TD came from.

`blocked_kicks` sums `fg_blocked` + `pt_blocked` + `pat_blocked` -
deliberately excludes the rare `gwfg_blocked` (blocked game-winning FG,
an overtime-only special case) as a documented, accepted v1 gap rather
than risking a double-count with `fg_blocked`.

Points allowed is a separate signal (`compute_points_allowed_rolling`,
derived from nfl_data.fetch_schedules' real final scores, not
team_stats) - mapped through DK's real non-linear bucket table
(config.NFL_DK_DST_POINTS_ALLOWED_BUCKETS, confirmed live via web
search). v1 maps the WINDOWED MEAN through the bucket table as a
documented first-pass approximation (DK's real table is defined on a
single game's discrete final score, not a rolling average) - flagged for
its own backtest column (Phase 7), same category of approximation as
nfl_dfs.py's yardage-bonus expected-value choice. Real DK rule (also not
modeled here, a known v1 gap): points allowed only counts points
surrendered while the DST unit itself was on the field (a pick-six is
charged to the DEFENSE that let it happen, not this team's own DST) -
a real game's final score doesn't make that distinction.

Games-back windows (config.NFL_DEFENSE_WINDOWS, shared with
nfl_teams.py's opponent-strength rolling rates - see that module's own
windows docstring for the "games-back, not day-count" reasoning) - not a
literal reuse of nfl_teams.compute_defense_rolling_rates' OUTPUT (that
table only carries pass/rush-yards-allowed/receptions-allowed, not these
box-score categories), but the same windowing pattern applied to a
different input table.
"""

import pandas as pd

from mlb_metrics import config

BOX_SCORE_STAT_COLS = ["sacks", "interceptions", "fumbles_recovered", "safeties", "blocked_kicks", "defensive_tds"]

DST_DFS_COLUMNS = [
    "team", "games",
    "sacks_per_game", "interceptions_per_game", "fumbles_recovered_per_game",
    "safeties_per_game", "blocked_kicks_per_game", "defensive_tds_per_game",
    "points_allowed_per_game", "Points_Allowed_Bonus", "DK_Points_DST",
]


def _rank_by_recency(team_df: pd.DataFrame) -> pd.DataFrame:
    """Adds `_recency_rank`: 0 for a team's own most recent row, 1 for
    their next-most-recent, etc. - independent per team."""
    ordered = team_df.sort_values(["season", "week"], ascending=False)
    return ordered.assign(_recency_rank=ordered.groupby("team").cumcount())


def compute_dst_box_score_rates(team_stats_df: pd.DataFrame) -> pd.DataFrame:
    """One row per team: per-game rates for each of BOX_SCORE_STAT_COLS,
    blended across config.NFL_DEFENSE_WINDOWS, plus `games` (full-history
    game count, unweighted)."""
    df = team_stats_df.copy()
    df["sacks"] = df["def_sacks"]
    df["interceptions"] = df["def_interceptions"]
    df["fumbles_recovered"] = df["fumble_recovery_opp"]
    df["safeties"] = df["def_safeties"]
    df["blocked_kicks"] = df["fg_blocked"] + df["pt_blocked"] + df["pat_blocked"]
    df["defensive_tds"] = df["def_tds"] + df["fumble_recovery_tds"] + df["special_teams_tds"]

    ranked = _rank_by_recency(df)
    blended = {f"{col}_per_game": None for col in BOX_SCORE_STAT_COLS}
    full_games = None

    for games_back, weight in config.NFL_DEFENSE_WINDOWS:
        window_df = ranked if games_back is None else ranked[ranked["_recency_rank"] < games_back]
        agg = window_df.groupby("team", as_index=False).agg(
            games=("_recency_rank", "size"), **{col: (col, "sum") for col in BOX_SCORE_STAT_COLS}
        )
        base = agg[["team"]]
        for col in BOX_SCORE_STAT_COLS:
            rate = (agg[col] / agg["games"]).where(agg["games"] > 0, 0)
            contribution = base.assign(rate=rate.values).set_index("team")["rate"] * weight
            key = f"{col}_per_game"
            blended[key] = contribution if blended[key] is None else blended[key].add(contribution, fill_value=0)
        if games_back is None:
            full_games = agg[["team", "games"]]

    result = pd.DataFrame(blended)
    result.index.name = "team"
    result = result.reset_index()
    result = result.merge(full_games, on="team", how="left")

    return result[["team", "games"] + [f"{col}_per_game" for col in BOX_SCORE_STAT_COLS]]


def compute_points_allowed(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, season, week): real points allowed that game
    (the opponent's final score) - derived from nfl_data.fetch_schedules,
    not team_stats. Games with no final score yet (an upcoming/in-progress
    game) are excluded, not zero-filled."""
    home = schedule_df[["home_team", "away_score", "season", "week"]].rename(
        columns={"home_team": "team", "away_score": "points_allowed"}
    )
    away = schedule_df[["away_team", "home_score", "season", "week"]].rename(
        columns={"away_team": "team", "home_score": "points_allowed"}
    )
    return pd.concat([home, away], ignore_index=True).dropna(subset=["points_allowed"])


def compute_points_allowed_rolling(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """One row per team: blended per-game points-allowed rate across
    config.NFL_DEFENSE_WINDOWS, plus `games`."""
    points = compute_points_allowed(schedule_df)
    ranked = _rank_by_recency(points)

    blended = None
    full_games = None
    for games_back, weight in config.NFL_DEFENSE_WINDOWS:
        window_df = ranked if games_back is None else ranked[ranked["_recency_rank"] < games_back]
        agg = window_df.groupby("team", as_index=False).agg(
            games=("points_allowed", "size"), total=("points_allowed", "sum")
        )
        rate = (agg["total"] / agg["games"]).where(agg["games"] > 0, 0)
        contribution = agg[["team"]].assign(rate=rate.values).set_index("team")["rate"] * weight
        blended = contribution if blended is None else blended.add(contribution, fill_value=0)
        if games_back is None:
            full_games = agg[["team", "games"]]

    result = blended.reset_index()
    result.columns = ["team", "points_allowed_per_game"]
    result = result.merge(full_games, on="team", how="left")

    return result[["team", "games", "points_allowed_per_game"]]


def compute_points_allowed_bonus(points_allowed_per_game: pd.Series) -> pd.Series:
    """Maps a (possibly fractional, windowed-mean) points-allowed value
    through DK's real points-allowed bucket table
    (config.NFL_DK_DST_POINTS_ALLOWED_BUCKETS) - see module docstring for
    the windowed-mean-vs-discrete-score approximation this makes."""
    result = pd.Series(index=points_allowed_per_game.index, dtype=float)
    for upper, points in config.NFL_DK_DST_POINTS_ALLOWED_BUCKETS:
        mask = result.isna() & (True if upper is None else points_allowed_per_game <= upper)
        result[mask] = points
    return result


def compute_dst_dk_points(box_score_rates: pd.DataFrame, points_allowed_rolling: pd.DataFrame) -> pd.DataFrame:
    """One row per team in `box_score_rates`
    (compute_dst_box_score_rates' output): DK_Points_DST from box-score
    categories plus the points-allowed bucket bonus. A team missing from
    `points_allowed_rolling` (a test fixture, or a stale/partial schedule)
    gets Points_Allowed_Bonus of 0 (the bucket table's neutral middle
    entry, 21-27 points allowed) rather than a dropped row."""
    merged = box_score_rates.merge(points_allowed_rolling[["team", "points_allowed_per_game"]], on="team", how="left")
    merged["points_allowed_per_game"] = merged["points_allowed_per_game"].fillna(23.5)
    merged["Points_Allowed_Bonus"] = compute_points_allowed_bonus(merged["points_allowed_per_game"])

    merged["DK_Points_DST"] = (
        merged["sacks_per_game"] * config.NFL_DK_DST_SACK_POINTS
        + merged["interceptions_per_game"] * config.NFL_DK_DST_INT_POINTS
        + merged["fumbles_recovered_per_game"] * config.NFL_DK_DST_FUMBLE_REC_POINTS
        + merged["safeties_per_game"] * config.NFL_DK_DST_SAFETY_POINTS
        + merged["blocked_kicks_per_game"] * config.NFL_DK_DST_BLOCKED_KICK_POINTS
        + merged["defensive_tds_per_game"] * config.NFL_DK_DST_TD_POINTS
        + merged["Points_Allowed_Bonus"]
    )

    return merged[DST_DFS_COLUMNS].sort_values("DK_Points_DST", ascending=False)
