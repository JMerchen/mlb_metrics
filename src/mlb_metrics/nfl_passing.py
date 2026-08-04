"""QB rolling form: attempts/completions/passing yards/TDs/interceptions
per game, plus a QB's own rush production (carries/rushing yards/rushing
TDs - some QBs contribute meaningful rushing DK points, e.g. a mobile QB,
so excluding it would badly understate their projection), blended across
config.NFL_QB_WINDOWS.

Games-back windows, NOT day-count ones like hitters.py's WAVE_WINDOWS -
see config.NFL_QB_WINDOWS's own docstring and the plan's "windows are
game-count, not calendar-count" guiding principle. nfl_data.fetch_weekly_stats
(nflreadpy's load_player_stats) already omits any week a player didn't
play (bye, injury, inactive) - confirmed live (a real QB's row for a bye
week is simply absent, week numbers skip it entirely) - so ranking a
player's own rows by recency and slicing the most recent N games
naturally skips unplayed weeks without any separate bye-detection logic.

All column names here (attempts, completions, passing_yards, passing_tds,
passing_interceptions, carries, rushing_yards, rushing_tds, player_id,
position, season, week, game_id) are confirmed live against real
nflreadpy 0.1.5 output - see nfl_data.py's module docstring for the cited
run.
"""

import pandas as pd

from mlb_metrics import config

STAT_COLS = [
    "attempts",
    "completions",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "carries",
    "rushing_yards",
    "rushing_tds",
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


def compute_qb_rolling_stats(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """One row per QB (weekly_df filtered to position == "QB"): per-game
    rates for each of STAT_COLS (named `<col>_per_game`), blended across
    config.NFL_QB_WINDOWS, plus `games` - the QB's full-history (all rows
    in `weekly_df`, unweighted) game count, used by the DK-scoring
    consumer (nfl_dfs.py, Phase 4) to apply config.NFL_QB_MIN_GAMES's
    small-sample qualifier, same "expose the count, let the caller
    qualify" pattern pitcher_form.compute_pitcher_dfs_form already uses
    for DFS_PITCHER_MIN_STARTS.

    `weekly_df` may span multiple seasons (e.g. the current season plus
    however much history the caller chooses to pass in) - windows are
    purely games-back over each QB's own rows, season boundaries don't
    reset or interrupt them."""
    qb_df = weekly_df[weekly_df["position"] == "QB"]
    ranked = _rank_by_recency(qb_df)

    blended = {f"{col}_per_game": None for col in STAT_COLS}
    full_games = None

    for games_back, weight in config.NFL_QB_WINDOWS:
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

    return result[["player_id", "games"] + [f"{col}_per_game" for col in STAT_COLS]]
