"""Each team's regular season game by game, grouped into series, for the
Stats page's schedule view: who they played, home or away, the final
score and which series it belonged to.

Final scores come from Statcast's post-pitch scores (the highest
post_home_score / post_away_score in the game - scores never go down).
A series is a run of consecutive games against the same opponent in the
same park; more than SERIES_GAP_DAYS between games starts a new one even
then (two separate visits).

Each game can also carry both sides' closing moneyline (game_moneylines,
from the ESPN odds scripts/backfill_historical_odds.py saves), which the
page uses for "one unit on this team's opponent every game".
"""

import numpy as np
import pandas as pd

from . import historical_odds, market_odds

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


def game_moneylines(scores: pd.DataFrame, odds_rows: pd.DataFrame | None) -> pd.DataFrame:
    """[game_pk, home_moneyline, away_moneyline] for the games in `scores`
    (final_scores) that have a closing line in `odds_rows` (the rows
    backfill_historical_odds.py saves: one per game per sportsbook).

    The odds carry no game_pk, so games are matched on the date and the
    two teams (ESPN's ARI and CHW are our AZ and CWS). A doubleheader is
    matched in order - first game to first start time - and only when
    both sides list the same number of games that day; otherwise those
    games are left without a line rather than guessed at."""
    columns = ["game_pk", "home_moneyline", "away_moneyline"]
    if odds_rows is None or odds_rows.empty or scores.empty:
        return pd.DataFrame(columns=columns)
    lines = historical_odds.consensus_closing_lines(odds_rows)
    if lines.empty:
        return pd.DataFrame(columns=columns)
    lines = lines.assign(home_team=lines["home_team"].map(market_odds._apply_abbrev_fixup),
                         away_team=lines["away_team"].map(market_odds._apply_abbrev_fixup),
                         date=pd.to_datetime(lines["date"]).dt.strftime("%Y-%m-%d"))
    key = ["date", "home_team", "away_team"]
    lines = lines.sort_values(key + ["start_time", "event_id"])
    lines["nth"] = lines.groupby(key).cumcount()
    lines["listed"] = lines.groupby(key)["event_id"].transform("size")
    games = scores.assign(date=pd.to_datetime(scores["date"]).dt.strftime("%Y-%m-%d")).sort_values(key + ["game_pk"])
    games["nth"] = games.groupby(key).cumcount()
    games["listed"] = games.groupby(key)["game_pk"].transform("size")
    matched = games.merge(lines[key + ["nth", "listed", "home_ml", "away_ml"]], on=key + ["nth", "listed"], how="inner")
    matched = matched.rename(columns={"home_ml": "home_moneyline", "away_ml": "away_moneyline"})[columns].reset_index(drop=True)
    for column in columns[1:]:
        matched[column] = pd.to_numeric(matched[column], errors="coerce").round(0)
    return matched


def team_games(statcast: pd.DataFrame, odds_rows: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per team per game, oldest first: [team, date, game_pk,
    opponent, home, runs_for, runs_against, result, run_diff, game_no,
    series, moneyline, opponent_moneyline]. `series` counts from 1 for
    each team. The moneylines are each side's closing line (American
    odds), blank for a game with none in `odds_rows`."""
    scores = final_scores(statcast)
    scores = scores.merge(game_moneylines(scores, odds_rows), on="game_pk", how="left")
    home = scores.assign(team=scores["home_team"], opponent=scores["away_team"], home=1,
                         runs_for=scores["home_score"], runs_against=scores["away_score"],
                         moneyline=scores["home_moneyline"], opponent_moneyline=scores["away_moneyline"])
    away = scores.assign(team=scores["away_team"], opponent=scores["home_team"], home=0,
                         runs_for=scores["away_score"], runs_against=scores["home_score"],
                         moneyline=scores["away_moneyline"], opponent_moneyline=scores["home_moneyline"])
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
    return games[["team", "date", "game_pk", "opponent", "home", "runs_for", "runs_against", "result", "run_diff",
                  "game_no", "series", "moneyline", "opponent_moneyline"]]
