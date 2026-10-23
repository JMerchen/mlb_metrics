"""Each team's regular season game by game, grouped into series, for the
Stats page's schedule view: who they played, home or away, the final
score and which series it belonged to.

Final scores come from Statcast's post-pitch scores (the highest
post_home_score / post_away_score in the game - scores never go down).
A series is a run of consecutive games against the same opponent in the
same park; more than SERIES_GAP_DAYS between games starts a new one even
then (two separate visits).
"""

import numpy as np
import pandas as pd

STATCAST_COLUMNS = ["game_pk", "game_date", "game_type", "home_team", "away_team", "post_home_score", "post_away_score"]
SERIES_GAP_DAYS = 2


def final_scores(statcast: pd.DataFrame) -> pd.DataFrame:
    """One row per regular-season game: [game_pk, date, home_team,
    away_team, home_score, away_score]."""
    games = statcast[statcast["game_type"] == "R"]
    return games.groupby("game_pk", as_index=False).agg(
        date=("game_date", "min"), home_team=("home_team", "first"), away_team=("away_team", "first"),
        home_score=("post_home_score", "max"), away_score=("post_away_score", "max"),
    )


def team_games(statcast: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game, oldest first: [team, date, game_pk,
    opponent, home, runs_for, runs_against, result, run_diff, game_no,
    series]. `series` counts from 1 for each team."""
    scores = final_scores(statcast)
    home = scores.assign(team=scores["home_team"], opponent=scores["away_team"], home=1,
                         runs_for=scores["home_score"], runs_against=scores["away_score"])
    away = scores.assign(team=scores["away_team"], opponent=scores["home_team"], home=0,
                         runs_for=scores["away_score"], runs_against=scores["home_score"])
    games = pd.concat([home, away], ignore_index=True)
    games["runs_for"] = games["runs_for"].astype(int)
    games["runs_against"] = games["runs_against"].astype(int)
    games["run_diff"] = games["runs_for"] - games["runs_against"]
    games["result"] = np.where(games["run_diff"] > 0, "W", "L")
    games["date"] = pd.to_datetime(games["date"])
    games = games.sort_values(["team", "date", "game_pk"]).reset_index(drop=True)
    games["game_no"] = games.groupby("team").cumcount() + 1

    previous = games.groupby("team")[["opponent", "home", "date"]].shift(1)
    new_series = (
        previous["opponent"].isna()
        | (previous["opponent"] != games["opponent"])
        | (previous["home"] != games["home"])
        | ((games["date"] - previous["date"]).dt.days > SERIES_GAP_DAYS)
    )
    games["series"] = new_series.astype(int).groupby(games["team"]).cumsum()
    games["date"] = games["date"].dt.strftime("%Y-%m-%d")
    return games[["team", "date", "game_pk", "opponent", "home", "runs_for", "runs_against", "result", "run_diff", "game_no", "series"]]
