"""RB/WR/TE rolling form: rush attempts/yards/TDs, targets/receptions/
receiving yards/TDs, and fumbles lost, per game, blended across
config.NFL_SKILL_WINDOWS. RB/WR/TE are handled together (not split into
separate modules the way MLB splits hitters/pitchers) because all three
share one stat surface and DK's FLEX pool - a TE's rushing columns are
just always 0, same shape pitcher_form.py doesn't need for hitters.

Games-back windows, NOT day-count ones - see nfl_passing.py's module
docstring (same reasoning, same nflreadpy bye-week-is-absent-row
behavior) and config.NFL_SKILL_WINDOWS's own docstring.

`fumbles_lost` is a derived sum of `rushing_fumbles_lost` +
`receiving_fumbles_lost` - nflreadpy also has a separate
`sack_fumbles_lost` column, but that's a QB-being-sacked stat (see
nfl_passing.py), not a skill-player one, so it's deliberately excluded
here.

All column names (carries, rushing_yards, rushing_tds, targets,
receptions, receiving_yards, receiving_tds, rushing_fumbles_lost,
receiving_fumbles_lost, player_id, position, season, week, game_id) are
confirmed live against real nflreadpy 0.1.5 output - see nfl_data.py's
module docstring for the cited run.
"""

import pandas as pd

from mlb_metrics import config

SKILL_POSITIONS = ("RB", "WR", "TE")

STAT_COLS = [
    "carries",
    "rushing_yards",
    "rushing_tds",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "fumbles_lost",
]


def _rank_by_recency(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """Adds `_recency_rank`: 0 for a player's own most recent game, 1 for
    their next-most-recent, etc. - independent per player_id."""
    ordered = weekly_df.sort_values(["season", "week"], ascending=False)
    return ordered.assign(_recency_rank=ordered.groupby("player_id").cumcount())


def _window_agg(ranked_df: pd.DataFrame, games_back) -> pd.DataFrame:
    window_df = ranked_df if games_back is None else ranked_df[ranked_df["_recency_rank"] < games_back]
    return window_df.groupby("player_id", as_index=False).agg(
        games=("game_id", "nunique"), **{col: (col, "sum") for col in STAT_COLS}
    )


def compute_skill_rolling_stats(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """One row per RB/WR/TE (weekly_df filtered to position in
    SKILL_POSITIONS): per-game rates for each of STAT_COLS (named
    `<col>_per_game`), blended across config.NFL_SKILL_WINDOWS, plus
    `games` and `position` - the player's full-history (unweighted) game
    count and their (current/most-recent) listed position, respectively.
    `games` is used by the DK-scoring consumer (nfl_dfs.py, Phase 4) to
    apply config.NFL_SKILL_MIN_GAMES's small-sample qualifier, same
    "expose the count, let the caller qualify" pattern
    pitcher_form.compute_pitcher_dfs_form already uses for
    DFS_PITCHER_MIN_STARTS. `position` is needed downstream for FLEX/
    slot eligibility (Phase 5) since RB/WR/TE are computed together here.

    `weekly_df` may span multiple seasons - windows are purely games-back
    over each player's own rows, season boundaries don't reset or
    interrupt them."""
    skill_df = weekly_df[weekly_df["position"].isin(SKILL_POSITIONS)].copy()
    skill_df["fumbles_lost"] = skill_df["rushing_fumbles_lost"] + skill_df["receiving_fumbles_lost"]
    ranked = _rank_by_recency(skill_df)

    blended = {f"{col}_per_game": None for col in STAT_COLS}
    full_games = None

    for games_back, weight in config.NFL_SKILL_WINDOWS:
        agg = _window_agg(ranked, games_back)
        base = agg[["player_id"]]
        for col in STAT_COLS:
            rate = (agg[col] / agg["games"]).where(agg["games"] > 0, 0)
            contribution = base.assign(rate=rate.values).set_index("player_id")["rate"] * weight
            key = f"{col}_per_game"
            blended[key] = contribution if blended[key] is None else blended[key].add(contribution, fill_value=0)
        if games_back is None:
            full_games = agg[["player_id", "games"]]

    result = pd.DataFrame(blended)
    result.index.name = "player_id"
    result = result.reset_index()
    result = result.merge(full_games, on="player_id", how="left")

    latest_position = ranked[ranked["_recency_rank"] == 0][["player_id", "position"]]
    result = result.merge(latest_position, on="player_id", how="left")

    return result[["player_id", "position", "games"] + [f"{col}_per_game" for col in STAT_COLS]]
