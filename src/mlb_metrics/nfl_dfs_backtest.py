"""No-lookahead backtest for the NFL DFS pipeline - structural port of
dfs_backtest.py, week-by-week instead of date-by-day. For each (season,
week) in a historical range, recomputes that week's QB/skill/DST DK
projections using ONLY data strictly before that week (the same
games-back windowing every live module already uses, just fed a
truncated history slice), then compares against that week's REAL
realized box score - scored through the SAME real DK formulas
(nfl_dfs.py/nfl_dst.py), but using the ACTUAL discrete outcome (did they
really clear 100/300 yards in that specific game) rather than the
expected-value bonus rate the live projection uses. That's the whole
point of a backtest: comparing an EV estimate against the real discrete
truth it's estimating.

Chronological ordering across seasons uses `season * 100 + week` as a
sort key - real, sound for all persisted regular+postseason data
(nflreadpy's own `season`/`week` columns, confirmed live - see
nfl_data.py's module docstring), since no NFL season has 100 weeks.
"""

import pandas as pd

from mlb_metrics import config, nfl_dfs, nfl_dst, nfl_passing, nfl_rush_rec


def _sort_key(df: pd.DataFrame) -> pd.Series:
    return df["season"] * 100 + df["week"]


def compute_actual_qb_dk_points(week_weekly_df: pd.DataFrame) -> pd.DataFrame:
    """[player_id, Actual_DK_Points_QB] - real DK QB points for ONE real
    week's box score (weekly_df filtered to position == "QB" and a single
    season+week) - same NFL_DK_* weights as nfl_dfs.compute_qb_dk_points,
    but using the REAL discrete 300+-yard/2pt outcome that game, not a
    windowed expected-value rate."""
    df = week_weekly_df[week_weekly_df["position"] == "QB"].copy()
    cleared_300 = (df["passing_yards"] >= 300).astype(int)
    points = (
        df["passing_yards"] * config.NFL_DK_PASS_YARD_POINTS
        + df["passing_tds"] * config.NFL_DK_PASS_TD_POINTS
        + df["passing_interceptions"] * config.NFL_DK_INTERCEPTION_POINTS
        + df["rushing_yards"] * config.NFL_DK_RUSH_YARD_POINTS
        + df["rushing_tds"] * config.NFL_DK_RUSH_TD_POINTS
        + cleared_300 * config.NFL_DK_300_PASS_YARD_BONUS
        + df["passing_2pt_conversions"] * config.NFL_DK_2PT_POINTS
    )
    return pd.DataFrame({"player_id": df["player_id"], "Actual_DK_Points_QB": points})


def compute_actual_skill_dk_points(week_weekly_df: pd.DataFrame) -> pd.DataFrame:
    """[player_id, Actual_DK_Points_Skill] - real DK RB/WR/TE points for
    ONE real week's box score, same shape as compute_actual_qb_dk_points
    above (real discrete 100+-yard bonuses, not an expected-value rate)."""
    df = week_weekly_df[week_weekly_df["position"].isin(nfl_rush_rec.SKILL_POSITIONS)].copy()
    fumbles_lost = df["rushing_fumbles_lost"] + df["receiving_fumbles_lost"]
    cleared_rush_100 = (df["rushing_yards"] >= 100).astype(int)
    cleared_rec_100 = (df["receiving_yards"] >= 100).astype(int)
    two_pt_total = df["rushing_2pt_conversions"] + df["receiving_2pt_conversions"]
    points = (
        df["rushing_yards"] * config.NFL_DK_RUSH_YARD_POINTS
        + df["rushing_tds"] * config.NFL_DK_RUSH_TD_POINTS
        + df["receiving_yards"] * config.NFL_DK_RECEIVING_YARD_POINTS
        + df["receiving_tds"] * config.NFL_DK_RECEIVING_TD_POINTS
        + df["receptions"] * config.NFL_DK_RECEPTION_POINTS
        + fumbles_lost * config.NFL_DK_FUMBLE_LOST_POINTS
        + cleared_rush_100 * config.NFL_DK_100_YARD_BONUS
        + cleared_rec_100 * config.NFL_DK_100_YARD_BONUS
        + two_pt_total * config.NFL_DK_2PT_POINTS
    )
    return pd.DataFrame({"player_id": df["player_id"], "Actual_DK_Points_Skill": points})


def compute_actual_dst_dk_points(week_team_stats_df: pd.DataFrame, week_schedule_df: pd.DataFrame) -> pd.DataFrame:
    """[team, Actual_DK_Points_DST] - real DK DST points for ONE real
    week, using that week's REAL final score for points-allowed (mapped
    through DK's real bucket table on the true single-game score - exact
    here, unlike the live projection's windowed-mean approximation - see
    nfl_dst.py's module docstring)."""
    df = week_team_stats_df.copy()
    sacks = df["def_sacks"]
    interceptions = df["def_interceptions"]
    fumbles_recovered = df["fumble_recovery_opp"]
    safeties = df["def_safeties"]
    blocked_kicks = df["fg_blocked"] + df["pt_blocked"] + df["pat_blocked"]
    defensive_tds = df["def_tds"] + df["fumble_recovery_tds"] + df["special_teams_tds"]

    points_allowed = nfl_dst.compute_points_allowed(week_schedule_df)[["team", "points_allowed"]]
    df = df[["team"]].assign(
        sacks=sacks.values, interceptions=interceptions.values, fumbles_recovered=fumbles_recovered.values,
        safeties=safeties.values, blocked_kicks=blocked_kicks.values, defensive_tds=defensive_tds.values,
    ).merge(points_allowed, on="team", how="left")
    df["Points_Allowed_Bonus"] = nfl_dst.compute_points_allowed_bonus(df["points_allowed"])

    points = (
        df["sacks"] * config.NFL_DK_DST_SACK_POINTS
        + df["interceptions"] * config.NFL_DK_DST_INT_POINTS
        + df["fumbles_recovered"] * config.NFL_DK_DST_FUMBLE_REC_POINTS
        + df["safeties"] * config.NFL_DK_DST_SAFETY_POINTS
        + df["blocked_kicks"] * config.NFL_DK_DST_BLOCKED_KICK_POINTS
        + df["defensive_tds"] * config.NFL_DK_DST_TD_POINTS
        + df["Points_Allowed_Bonus"]
    )
    return pd.DataFrame({"team": df["team"], "Actual_DK_Points_DST": points})


def _compute_week_outputs(weekly_df: pd.DataFrame, team_stats_df: pd.DataFrame, schedules_df: pd.DataFrame, season: int, week: int) -> dict | None:
    """One (season, week)'s worth of no-lookahead recomputation. Returns
    None when there's no usable history, or no real box score that week
    to compare against (both callers skip the week in that case)."""
    target_key = season * 100 + week

    history_weekly = weekly_df[_sort_key(weekly_df) < target_key]
    if history_weekly.empty:
        return None
    history_team_stats = team_stats_df[_sort_key(team_stats_df) < target_key]
    history_schedules = schedules_df[_sort_key(schedules_df) < target_key]

    this_week_weekly = weekly_df[(weekly_df["season"] == season) & (weekly_df["week"] == week)]
    if this_week_weekly.empty:
        return None
    this_week_team_stats = team_stats_df[(team_stats_df["season"] == season) & (team_stats_df["week"] == week)]
    this_week_schedule = schedules_df[(schedules_df["season"] == season) & (schedules_df["week"] == week)]

    qb_rolling = nfl_passing.compute_qb_rolling_stats(history_weekly)
    projected_qb = nfl_dfs.compute_qb_dk_points(qb_rolling, history_weekly)

    skill_rolling = nfl_rush_rec.compute_skill_rolling_stats(history_weekly)
    projected_skill = nfl_dfs.compute_skill_dk_points(skill_rolling, history_weekly)

    dst_rates = nfl_dst.compute_dst_box_score_rates(history_team_stats)
    points_allowed_rolling = nfl_dst.compute_points_allowed_rolling(history_schedules)
    projected_dst = nfl_dst.compute_dst_dk_points(dst_rates, points_allowed_rolling)

    actual_qb = compute_actual_qb_dk_points(this_week_weekly)
    actual_skill = compute_actual_skill_dk_points(this_week_weekly)
    actual_dst = compute_actual_dst_dk_points(this_week_team_stats, this_week_schedule)

    return {
        "projected_qb": projected_qb, "actual_qb": actual_qb,
        "projected_skill": projected_skill, "actual_skill": actual_skill,
        "projected_dst": projected_dst, "actual_dst": actual_dst,
    }


def backtest_nfl_dfs_projections(
    weekly_df: pd.DataFrame, team_stats_df: pd.DataFrame, schedules_df: pd.DataFrame, weeks: int | None = None
) -> dict[str, pd.DataFrame]:
    """No-lookahead backtest: for each real (season, week) with usable
    prior history in `weekly_df`/`team_stats_df`/`schedules_df` (already-
    persisted/fetched historical data - see scripts/fetch_nfl_historical.py),
    recomputes that week's QB/skill/DST DK projections using ONLY data
    strictly before it, then compares against that week's REAL realized
    DK points. Returns {"qb": DataFrame, "skill": DataFrame, "dst":
    DataFrame}, one row per (season, week, player/team) scored.
    `weeks` (default None, i.e. every available week) caps the backtest
    to the most recent N real weeks - mirrors dfs_backtest.backtest_dfs_projections's
    own `days` parameter."""
    all_weeks = weekly_df[["season", "week"]].drop_duplicates()
    all_weeks = all_weeks.assign(_key=_sort_key(all_weeks)).sort_values("_key")
    if weeks:
        all_weeks = all_weeks.tail(weeks)

    qb_rows, skill_rows, dst_rows = [], [], []
    for _, row in all_weeks.iterrows():
        season, week = int(row["season"]), int(row["week"])
        outputs = _compute_week_outputs(weekly_df, team_stats_df, schedules_df, season, week)
        if outputs is None:
            continue

        qb_scored = outputs["projected_qb"].merge(outputs["actual_qb"], on="player_id", how="inner")
        if not qb_scored.empty:
            qb_scored = qb_scored.copy()
            qb_scored["season"], qb_scored["week"] = season, week
            qb_rows.append(qb_scored)

        skill_scored = outputs["projected_skill"].merge(outputs["actual_skill"], on="player_id", how="inner")
        if not skill_scored.empty:
            skill_scored = skill_scored.copy()
            skill_scored["season"], skill_scored["week"] = season, week
            skill_rows.append(skill_scored)

        dst_scored = outputs["projected_dst"].merge(outputs["actual_dst"], on="team", how="inner")
        if not dst_scored.empty:
            dst_scored = dst_scored.copy()
            dst_scored["season"], dst_scored["week"] = season, week
            dst_rows.append(dst_scored)

    return {
        "qb": pd.concat(qb_rows, ignore_index=True) if qb_rows else pd.DataFrame(),
        "skill": pd.concat(skill_rows, ignore_index=True) if skill_rows else pd.DataFrame(),
        "dst": pd.concat(dst_rows, ignore_index=True) if dst_rows else pd.DataFrame(),
    }
