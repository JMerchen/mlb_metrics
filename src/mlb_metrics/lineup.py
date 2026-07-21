"""Batting-order consistency: is a batter actually a regular, playing high
enough in the order to get real at-bats - as opposed to a bench player
having a hot week or a recent call-up riding an unsustainable streak.

Derived entirely from data.assign_batting_order's output, itself derived
from Statcast data already being persisted (no external dependency, and it
works retroactively for any date the raw data already covers). This is a
game-count window (see config.LINEUP_WINDOW_GAMES), not the day-count
windows the rest of the pipeline's metrics use - roster usage doesn't follow
a calendar cadence the way rolling batting stats do, so it doesn't fit
hitters.py's `_blend_windows` machinery and instead follows teams.py's
existing rolling-by-game_id pattern.
"""

import pandas as pd

from mlb_metrics import config

STARTER_MAX_BATTING_ORDER = 9


def compute_lineup_consistency(batting_order: pd.DataFrame, latest_team: pd.DataFrame) -> pd.DataFrame:
    """Per batter, over their CURRENT team's last config.LINEUP_WINDOW_GAMES
    games (via `latest_team` - a batter traded mid-season has their window
    reset to the new team only, not blended with games played for the old
    one): `avg_batting_order` (mean slot across games they started - null if
    they never started for this team in the window) and `start_rate`
    (starts / min(team games in window, LINEUP_WINDOW_GAMES)).

    batting_order: data.assign_batting_order's output.
    latest_team: [key_mlbam, team] - e.g. data.latest_team_for_batters's output.
    """
    window = config.LINEUP_WINDOW_GAMES

    team_games = (
        batting_order[["team", "game_id"]]
        .drop_duplicates()
        .sort_values(["team", "game_id"], ascending=[True, False])
    )
    team_games["recency_rank"] = team_games.groupby("team").cumcount()
    recent_team_games = team_games[team_games["recency_rank"] < window]

    team_games_in_window = (
        recent_team_games.groupby("team", as_index=False)
        .size()
        .rename(columns={"size": "team_games_in_window"})
    )

    starts = batting_order[batting_order["batting_order"] <= STARTER_MAX_BATTING_ORDER].merge(
        recent_team_games[["team", "game_id"]], on=["team", "game_id"]
    )
    per_batter = starts.groupby(["team", "batter"], as_index=False).agg(
        avg_batting_order=("batting_order", "mean"),
        games_started=("game_id", "count"),
    )
    per_batter = per_batter.merge(team_games_in_window, on="team", how="left")
    per_batter["start_rate"] = per_batter["games_started"] / per_batter["team_games_in_window"].clip(lower=1)

    result = latest_team.rename(columns={"key_mlbam": "batter"}).merge(
        per_batter[["team", "batter", "avg_batting_order", "start_rate"]], on=["team", "batter"], how="left"
    )
    result["start_rate"] = result["start_rate"].fillna(0)
    # avg_batting_order is intentionally left null for a batter who never
    # started for their current team in the window - there's no slot to
    # average, and a null correctly fails the "top half" qualifier downstream.

    return result.rename(columns={"batter": "key_mlbam"})[["key_mlbam", "avg_batting_order", "start_rate"]]
