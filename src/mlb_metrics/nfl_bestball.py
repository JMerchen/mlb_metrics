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

from mlb_metrics import config, nfl_dfs_backtest, nfl_rush_rec

POSITIONS = ("QB",) + nfl_rush_rec.SKILL_POSITIONS

# Standard-deviation bucket edges for compute_position_scarcity's bell-curve
# breakdown - the standard "empirical rule" bands (within 1 SD, 1-2, 2-3,
# beyond 3) rather than a dynamically-sized set, since a real position pool
# (dozens to a couple hundred players) essentially never has anyone beyond
# +/-3 SD and a fixed, familiar shape is easier to read as an actual bell
# curve. `pd.cut`'s bin edges (right-inclusive): (-inf,-3], (-3,-2], (-2,-1],
# (-1,1], (1,2], (2,3], (3,inf) - the central bin is intentionally the widest
# (-1 to +1) to match "within one standard deviation" as a single band.
SCARCITY_BUCKET_EDGES = [-float("inf"), -3, -2, -1, 1, 2, 3, float("inf")]
SCARCITY_BUCKET_LABELS = [
    "below_-3sd",
    "-3sd_to_-2sd",
    "-2sd_to_-1sd",
    "within_1sd",
    "1sd_to_2sd",
    "2sd_to_3sd",
    "above_3sd",
]


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


def compute_position_scarcity(
    rankings_df: pd.DataFrame,
    min_games: int = config.NFL_BESTBALL_SCARCITY_MIN_GAMES,
    value_column: str = config.NFL_BESTBALL_SCARCITY_VALUE_COLUMN,
) -> pd.DataFrame:
    """One row per position (`POSITIONS` - QB/RB/WR/TE) describing how real
    `value_column` production is actually distributed - a "how many
    difference-makers exist at this position, and how many replacement-level
    guys" read for draft strategy, not just a single overall rank list.

    `total_players`: every real player at that position in `rankings_df`
    (i.e. everyone who recorded a real stat line last season), regardless of
    playing time - the position's real total pool size.

    `qualified_players`/`mean`/`std`: restricted to players with
    `games_played >= min_games` - a small handful of huge-rate, tiny-sample
    games (e.g. a Week 17 injury fill-in's one big game) would otherwise
    skew what "typical starter" production even looks like, so the mean/std
    describing the position's real shape are computed only over players who
    played enough of the season to be read as a real starter/role player,
    not a cameo (see config.NFL_BESTBALL_SCARCITY_MIN_GAMES's own docstring
    for the exact reasoning/default). `std` uses population (ddof=0), not
    sample, standard deviation - this describes the actual observed shape of
    this real, complete season's qualified population, not an estimate of
    some larger population it was sampled from.

    The bucket columns (`SCARCITY_BUCKET_LABELS`) count QUALIFIED players
    only, bucketed by how many standard deviations their own `value_column`
    sits from that position's own mean - a bell curve in table form. A
    position with fewer than 2 qualified players gets NaN mean/std and
    all-zero bucket counts (nothing real to compute a spread from). A
    position with 2+ qualified players but a std of exactly 0 (everyone
    qualified tied) gets a real mean/std but still all-zero bucket counts,
    since a z-score is undefined when the distribution has no spread -
    either way this is an honest "not enough real data" result rather than
    a divide-by-zero crash or a fabricated one."""
    rows = []
    for position in POSITIONS:
        pos_df = rankings_df[rankings_df["position"] == position]
        total_players = len(pos_df)
        qualified = pos_df[pos_df["games_played"] >= min_games]
        n_qualified = len(qualified)

        bucket_counts = {label: 0 for label in SCARCITY_BUCKET_LABELS}
        mean = qualified[value_column].mean() if n_qualified else float("nan")
        std = qualified[value_column].std(ddof=0) if n_qualified > 1 else float("nan")

        if n_qualified > 1 and std > 0:
            z_scores = (qualified[value_column] - mean) / std
            bucketed = pd.cut(z_scores, bins=SCARCITY_BUCKET_EDGES, labels=SCARCITY_BUCKET_LABELS)
            counts = bucketed.value_counts()
            bucket_counts = {label: int(counts.get(label, 0)) for label in SCARCITY_BUCKET_LABELS}

        rows.append(
            {
                "position": position,
                "total_players": total_players,
                "qualified_players": n_qualified,
                "mean_dk_points": mean,
                "std_dk_points": std,
                **bucket_counts,
            }
        )

    return pd.DataFrame(rows)
