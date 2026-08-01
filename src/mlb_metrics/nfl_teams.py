"""Team defense rolling rates: how many pass yards, rush yards, and
receptions a defense has actually allowed recently - the building block
`nfl_matchup.py` looks up by a player's UPCOMING opponent (never a
team's own last-played opponent - see that module's docstring for why
this distinction matters, the exact bug class teams.compute_offensive_edge
had to be split apart to avoid for MLB, see that function's own
docstring).

Derived entirely from nfl_data.fetch_weekly_stats's own `opponent_team`
column - a player's `opponent_team` for a given week IS the defense they
played against that week, so a week's worth of one team's REALIZED
offensive stats, grouped by `opponent_team`, is exactly what the
opposing DEFENSE allowed that week. No separate schedule join is needed
to compute what was allowed (only nfl_matchup.py's lookup of what's
COMING UP needs the schedule).

Games-back windows (config.NFL_DEFENSE_WINDOWS), not day-count ones -
same reasoning as nfl_passing.py/nfl_rush_rec.py.
"""

import pandas as pd

from mlb_metrics import config

ALLOWED_STAT_COLS = ["pass_yards_allowed", "rush_yards_allowed", "receptions_allowed"]


def compute_team_week_allowed(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (defending team, season, week): total pass yards, rush
    yards, and receptions allowed - summed across every offensive
    player whose `opponent_team` was that defense that week."""
    agg = weekly_df.groupby(["opponent_team", "season", "week"], as_index=False).agg(
        pass_yards_allowed=("passing_yards", "sum"),
        rush_yards_allowed=("rushing_yards", "sum"),
        receptions_allowed=("receptions", "sum"),
    )
    return agg.rename(columns={"opponent_team": "team"})


def _rank_by_recency(team_week_df: pd.DataFrame) -> pd.DataFrame:
    """Adds `_recency_rank`: 0 for a team's own most recent game allowed,
    1 for their next-most-recent, etc. - independent per team."""
    ordered = team_week_df.sort_values(["season", "week"], ascending=False)
    return ordered.assign(_recency_rank=ordered.groupby("team").cumcount())


def _window_agg(ranked_df: pd.DataFrame, games_back) -> pd.DataFrame:
    window_df = ranked_df if games_back is None else ranked_df[ranked_df["_recency_rank"] < games_back]
    return window_df.groupby("team", as_index=False).agg(
        games=("_recency_rank", "size"), **{col: (col, "sum") for col in ALLOWED_STAT_COLS}
    )


def compute_defense_rolling_rates(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """One row per team: per-game rates for each of ALLOWED_STAT_COLS
    (named `<col>_per_game`), blended across config.NFL_DEFENSE_WINDOWS,
    plus `games` (full-history game count, unweighted). `weekly_df` may
    span multiple seasons - windows are purely games-back over each
    team's own allowed-stats rows, season boundaries don't reset or
    interrupt them."""
    team_week = compute_team_week_allowed(weekly_df)
    ranked = _rank_by_recency(team_week)

    blended = {f"{col}_per_game": None for col in ALLOWED_STAT_COLS}
    full_games = None

    for games_back, weight in config.NFL_DEFENSE_WINDOWS:
        agg = _window_agg(ranked, games_back)
        base = agg[["team"]]
        for col in ALLOWED_STAT_COLS:
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

    return result[["team", "games"] + [f"{col}_per_game" for col in ALLOWED_STAT_COLS]]
