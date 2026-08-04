"""Estimated DraftKings NFL Classic fantasy points for QBs and RB/WR/TE,
built on nfl_passing.compute_qb_rolling_stats/nfl_rush_rec.compute_skill_rolling_stats'
per-game blended rate output. DraftKings' scoring rules (config.NFL_DK_*)
were confirmed live via web search (not from memory) - see README.md's
NFL DFS section for sources.

Unlike dfs.py's MLB hitter scoring (which had to approximate non-linear
hit-type value from a single linear Expected_Bases signal), DK's real NFL
categories map directly onto real per-game rate stats this project
already computes - the real risk here is entirely upstream, in
nfl_passing.py/nfl_rush_rec.py's windowed projections, not in this
formula.

## The yardage-bonus wrinkle

DK's 100+ rushing/receiving and 300+ passing yard bonuses are step
functions, not linear in a windowed MEAN. A player averaging exactly 100
rushing yards/game doesn't reliably clear 100 in any GIVEN game (some
games well under, some well over) - crediting the full +3 bonus every
week off the mean would systematically overstate a bursty player's
points. v1 instead computes each player's own historical RATE of
clearing the threshold (fraction of their own real past games, blended
across the same games-back windows as the rest of their projection) and
multiplies by the bonus value - an expected-value approach, structurally
identical to hitters.compute_wave's own hit-rate-into-probability
pattern. Flagged unvalidated (no NFL backtest exists yet - see Phase 7);
tested against a naive "no bonus term" baseline once nfl_dfs_backtest.py
exists.

2-point conversions get the same expected-value-rate treatment (they're
also a rare, discrete event a windowed MEAN of raw counts handles fine
via the same per-game-rate machinery already used for every other
category here, so no special-casing is needed there - only the yardage
bonuses need a boolean-threshold rate instead of a raw-count rate).
"""

import pandas as pd

from mlb_metrics import config, nfl_rush_rec

QB_DFS_COLUMNS = [
    "player_id", "games",
    "passing_yards_per_game", "passing_tds_per_game", "passing_interceptions_per_game",
    "rushing_yards_per_game", "rushing_tds_per_game",
    "Expected_300_Bonus_Rate", "Expected_2PT", "DK_Points_QB",
]

SKILL_DFS_COLUMNS = [
    "player_id", "position", "games",
    "rushing_yards_per_game", "rushing_tds_per_game",
    "receiving_yards_per_game", "receiving_tds_per_game", "receptions_per_game",
    "fumbles_lost_per_game",
    "Expected_Rush_100_Bonus_Rate", "Expected_Rec_100_Bonus_Rate", "Expected_2PT",
    "DK_Points_Skill",
]


def _rank_by_recency(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """Adds `_recency_rank`: 0 for a player's own most recent game, 1 for
    their next-most-recent, etc. - independent per player_id. Same
    convention as nfl_passing.py/nfl_rush_rec.py's identically-named
    helper, duplicated here rather than imported since this module needs
    it applied to a DIFFERENT derived column each time (a boolean
    threshold flag, not a real stat total)."""
    ordered = weekly_df.sort_values(["season", "week"], ascending=False)
    return ordered.assign(_recency_rank=ordered.groupby("player_id").cumcount())


def _blended_rate(ranked_df: pd.DataFrame, windows, value_col: str) -> pd.Series:
    """Per-player blended per-game rate of `value_col` (a real stat OR a
    0/1 "cleared this threshold" flag), across `windows` - same
    games-back windowing shape as nfl_passing.py/nfl_rush_rec.py,
    generalized to any single column so this module doesn't need to
    duplicate a whole rolling-stats module per new signal."""
    blended = None
    for games_back, weight in windows:
        window_df = ranked_df if games_back is None else ranked_df[ranked_df["_recency_rank"] < games_back]
        agg = window_df.groupby("player_id", as_index=False).agg(
            games=("game_id", "nunique"), total=(value_col, "sum")
        )
        rate = (agg["total"] / agg["games"]).where(agg["games"] > 0, 0)
        contribution = agg[["player_id"]].assign(rate=rate.values).set_index("player_id")["rate"] * weight
        blended = contribution if blended is None else blended.add(contribution, fill_value=0)
    return blended if blended is not None else pd.Series(dtype=float)


def compute_qb_dk_points(qb_rolling: pd.DataFrame, weekly_df: pd.DataFrame) -> pd.DataFrame:
    """One row per QB in `qb_rolling` (nfl_passing.compute_qb_rolling_stats'
    output): DK_Points_QB from the linear categories (pass yards/TDs/
    INTs, rush yards/TDs) plus expected-value 300+ pass-yard bonus and
    2-point-conversion terms derived directly from `weekly_df` (raw
    per-game rows, needed for the threshold-rate computation - see module
    docstring)."""
    qb_weekly = weekly_df[weekly_df["position"] == "QB"].copy()
    qb_weekly["cleared_300"] = (qb_weekly["passing_yards"] >= 300).astype(int)
    ranked = _rank_by_recency(qb_weekly)

    result = qb_rolling.set_index("player_id").copy()
    result["Expected_300_Bonus_Rate"] = _blended_rate(ranked, config.NFL_QB_WINDOWS, "cleared_300")
    result["Expected_2PT"] = _blended_rate(ranked, config.NFL_QB_WINDOWS, "passing_2pt_conversions")
    bonus_cols = ["Expected_300_Bonus_Rate", "Expected_2PT"]
    result[bonus_cols] = result[bonus_cols].fillna(0)

    result["DK_Points_QB"] = (
        result["passing_yards_per_game"] * config.NFL_DK_PASS_YARD_POINTS
        + result["passing_tds_per_game"] * config.NFL_DK_PASS_TD_POINTS
        + result["passing_interceptions_per_game"] * config.NFL_DK_INTERCEPTION_POINTS
        + result["rushing_yards_per_game"] * config.NFL_DK_RUSH_YARD_POINTS
        + result["rushing_tds_per_game"] * config.NFL_DK_RUSH_TD_POINTS
        + result["Expected_300_Bonus_Rate"] * config.NFL_DK_300_PASS_YARD_BONUS
        + result["Expected_2PT"] * config.NFL_DK_2PT_POINTS
    )

    return result.reset_index()[QB_DFS_COLUMNS].sort_values("DK_Points_QB", ascending=False)


def compute_skill_dk_points(skill_rolling: pd.DataFrame, weekly_df: pd.DataFrame) -> pd.DataFrame:
    """One row per RB/WR/TE in `skill_rolling`
    (nfl_rush_rec.compute_skill_rolling_stats' output): DK_Points_Skill
    from the linear categories (rush/rec yards/TDs, receptions, fumbles
    lost) plus expected-value 100+ rush-yard, 100+ rec-yard, and
    2-point-conversion terms - both 100+ bonuses are computed and scored
    SEPARATELY (a player can clear both in the same real game), never
    combined into one "100+ total yards" flag."""
    skill_weekly = weekly_df[weekly_df["position"].isin(nfl_rush_rec.SKILL_POSITIONS)].copy()
    skill_weekly["cleared_rush_100"] = (skill_weekly["rushing_yards"] >= 100).astype(int)
    skill_weekly["cleared_rec_100"] = (skill_weekly["receiving_yards"] >= 100).astype(int)
    skill_weekly["two_pt_total"] = (
        skill_weekly["rushing_2pt_conversions"] + skill_weekly["receiving_2pt_conversions"]
    )
    ranked = _rank_by_recency(skill_weekly)

    result = skill_rolling.set_index("player_id").copy()
    result["Expected_Rush_100_Bonus_Rate"] = _blended_rate(ranked, config.NFL_SKILL_WINDOWS, "cleared_rush_100")
    result["Expected_Rec_100_Bonus_Rate"] = _blended_rate(ranked, config.NFL_SKILL_WINDOWS, "cleared_rec_100")
    result["Expected_2PT"] = _blended_rate(ranked, config.NFL_SKILL_WINDOWS, "two_pt_total")
    bonus_cols = ["Expected_Rush_100_Bonus_Rate", "Expected_Rec_100_Bonus_Rate", "Expected_2PT"]
    result[bonus_cols] = result[bonus_cols].fillna(0)

    result["DK_Points_Skill"] = (
        result["rushing_yards_per_game"] * config.NFL_DK_RUSH_YARD_POINTS
        + result["rushing_tds_per_game"] * config.NFL_DK_RUSH_TD_POINTS
        + result["receiving_yards_per_game"] * config.NFL_DK_RECEIVING_YARD_POINTS
        + result["receiving_tds_per_game"] * config.NFL_DK_RECEIVING_TD_POINTS
        + result["receptions_per_game"] * config.NFL_DK_RECEPTION_POINTS
        + result["fumbles_lost_per_game"] * config.NFL_DK_FUMBLE_LOST_POINTS
        + result["Expected_Rush_100_Bonus_Rate"] * config.NFL_DK_100_YARD_BONUS
        + result["Expected_Rec_100_Bonus_Rate"] * config.NFL_DK_100_YARD_BONUS
        + result["Expected_2PT"] * config.NFL_DK_2PT_POINTS
    )

    return result.reset_index()[SKILL_DFS_COLUMNS].sort_values("DK_Points_Skill", ascending=False)
