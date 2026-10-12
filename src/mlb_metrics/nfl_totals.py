"""Predicted points for each NFL game, from WAVE-style offense and defense EPA.

User request (2026-10-12): a predicted point total for every game, built
from each offense's EPA per play and each defense's EPA allowed per play,
blended across recency windows the way the MLB side's WAVE blends a
hitter's full-season, 81-, 30- and 10-day rates.

For every team, from its own earlier regular-season games (crossing into
the prior season, so week 1 still has a full window):
    off_epa    EPA per play on offense, blended over config.NFL_TOTALS_WINDOWS
               (games back), each window shrunk toward the league rate by
               config.NFL_TOTALS_SHRINK_PLAYS pseudo-plays
    def_epa    the same for EPA allowed per play on defense
    off_plays  offensive plays per game (pace), same windows
    def_plays  plays allowed per game
A team's points in a game are then a linear function of its offense and
the opponent's defense:
    points = const + b1*off_epa + b2*opp_def_epa + b3*off_plays
             + b4*opp_def_plays + b5*home
with the coefficients in config.NFL_TOTALS_COEFFICIENTS, fit by
scripts/backtest_nfl_totals.py. The game's total is the two teams' sum.

Plays are dropbacks and designed runs (`pass` or `rush` flagged, kneels and
spikes dropped) - the same set the category ratings backtest uses.

Validated in scripts/backtest_nfl_totals.py (season by season, refit on
earlier seasons only) - see that script's docstring for the numbers. In
short: off by 10.8 points a game on average against the market's 10.4
and a flat average's 11.0, with nothing the market's number doesn't
already carry - shown next to Vegas's total, never turned into bet
advice.
"""

import pandas as pd

from mlb_metrics import config

# Schedules keep each franchise's code at the time (OAK, SD, STL);
# play-by-play is restated in today's codes.
NFL_TEAM_ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

TOTALS_FEATURE_COLUMNS = ["off_epa", "opp_def_epa", "off_plays", "opp_def_plays", "home"]

PBP_COLUMNS = ["game_id", "season", "week", "season_type", "posteam", "defteam", "home_team",
                "epa", "pass", "rush", "qb_kneel", "qb_spike"]

# Pseudo-games of league-average pace added to each plays-per-game window.
_PACE_PSEUDO_GAMES = 2.0


def _canonical(team: pd.Series) -> pd.Series:
    return team.replace(NFL_TEAM_ALIASES)


def team_game_epa(pbp: pd.DataFrame, schedules: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (regular-season game, offense): EPA summed over its
    plays, the play count, and - when `schedules` has a final score - the
    points it scored. Columns: game_id, season, week, team, opp, home,
    epa_sum, plays, points."""
    columns = ["game_id", "season", "week", "team", "opp", "home", "epa_sum", "plays", "points"]
    if pbp is None or pbp.empty or not set(PBP_COLUMNS).issubset(pbp.columns):
        return pd.DataFrame(columns=columns)
    plays = pbp[(pbp["season_type"] == "REG") & pbp["epa"].notna() & pbp["posteam"].notna()]
    plays = plays[(plays["qb_kneel"] != 1) & (plays["qb_spike"] != 1)]
    plays = plays[(plays["pass"] == 1) | (plays["rush"] == 1)]
    games = (
        plays.groupby(["game_id", "season", "week", "posteam", "defteam", "home_team"])
        .agg(epa_sum=("epa", "sum"), plays=("epa", "size"))
        .reset_index()
        .rename(columns={"posteam": "team", "defteam": "opp"})
    )
    games["home"] = (games["team"] == games["home_team"]).astype(float)
    games["points"] = float("nan")
    if schedules is not None and not schedules.empty:
        final = schedules[schedules["home_score"].notna() & schedules["away_score"].notna()]
        scores = pd.concat([
            pd.DataFrame({"game_id": final["game_id"], "team": _canonical(final["home_team"]), "points": final["home_score"]}),
            pd.DataFrame({"game_id": final["game_id"], "team": _canonical(final["away_team"]), "points": final["away_score"]}),
        ])
        games = games.drop(columns="points").merge(scores, on=["game_id", "team"], how="left")
    return games[columns]


def _blend(rows: pd.DataFrame, value: str, count: str, prior: float, pseudo: float, windows) -> float:
    """WAVE-style blend of rows[value].sum() / rows[count].sum() over the
    most recent N rows for each (N, weight) window (N None = all rows),
    each window shrunk toward `prior` with `pseudo` pseudo-observations."""
    total = 0.0
    for n_games, weight in windows:
        window = rows if n_games is None else rows.tail(n_games)
        total += weight * (window[value].sum() + pseudo * prior) / (window[count].sum() + pseudo)
    return total


def wave_ratings(
    team_games: pd.DataFrame,
    windows=None,
    shrink_plays: float | None = None,
) -> pd.DataFrame:
    """Each team's current offense and defense rating from every game in
    `team_games` (pass only games before the one being predicted). Columns:
    team, off_epa, def_epa, off_plays, def_plays."""
    windows = config.NFL_TOTALS_WINDOWS if windows is None else windows
    shrink_plays = config.NFL_TOTALS_SHRINK_PLAYS if shrink_plays is None else shrink_plays
    columns = ["team", "off_epa", "def_epa", "off_plays", "def_plays"]
    if team_games is None or team_games.empty:
        return pd.DataFrame(columns=columns)
    games = team_games.sort_values(["season", "week"]).assign(games=1.0)
    league_epa = games["epa_sum"].sum() / games["plays"].sum()
    league_pace = games["plays"].mean()

    def side(key: str, prefix: str) -> pd.DataFrame:
        rows = []
        for team, own in games.groupby(key):
            rows.append({
                "team": team,
                f"{prefix}_epa": _blend(own, "epa_sum", "plays", league_epa, shrink_plays, windows),
                f"{prefix}_plays": _blend(own, "plays", "games", league_pace, _PACE_PSEUDO_GAMES, windows),
            })
        return pd.DataFrame(rows, columns=["team", f"{prefix}_epa", f"{prefix}_plays"])

    return side("team", "off").merge(side("opp", "def"), on="team", how="outer")[columns]


def matchup_features(ratings: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game (home offense, away offense) with
    TOTALS_FEATURE_COLUMNS. `games` needs game_id, home_team, away_team.
    A team with no rating (no earlier games) gets NaN features."""
    games = games.assign(home_team=_canonical(games["home_team"]), away_team=_canonical(games["away_team"]))
    sides = pd.concat([
        pd.DataFrame({"game_id": games["game_id"], "team": games["home_team"], "opp": games["away_team"], "home": 1.0}),
        pd.DataFrame({"game_id": games["game_id"], "team": games["away_team"], "opp": games["home_team"], "home": 0.0}),
    ], ignore_index=True)
    offense = ratings[["team", "off_epa", "off_plays"]]
    defense = ratings[["team", "def_epa", "def_plays"]].rename(
        columns={"team": "opp", "def_epa": "opp_def_epa", "def_plays": "opp_def_plays"}
    )
    return sides.merge(offense, on="team", how="left").merge(defense, on="opp", how="left")


def predict_team_points(features: pd.DataFrame, coefficients: dict | None = None) -> pd.Series:
    """Linear points prediction for rows carrying TOTALS_FEATURE_COLUMNS."""
    coefficients = config.NFL_TOTALS_COEFFICIENTS if coefficients is None else coefficients
    points = pd.Series(coefficients["const"], index=features.index, dtype=float)
    for column in TOTALS_FEATURE_COLUMNS:
        points = points + coefficients[column] * features[column]
    return points


def predict_game_points(
    team_games: pd.DataFrame,
    games: pd.DataFrame,
    coefficients: dict | None = None,
) -> pd.DataFrame:
    """Predicted points for each game in `games` from the earlier games in
    `team_games` (team_game_epa's output). Columns: game_id,
    predicted_home_points, predicted_away_points, predicted_total; NaN for
    a game where either team has no history."""
    columns = ["game_id", "predicted_home_points", "predicted_away_points", "predicted_total"]
    if games is None or games.empty:
        return pd.DataFrame(columns=columns)
    features = matchup_features(wave_ratings(team_games), games)
    features["points"] = predict_team_points(features, coefficients)
    home = features[features["home"] == 1.0][["game_id", "points"]].rename(columns={"points": "predicted_home_points"})
    away = features[features["home"] == 0.0][["game_id", "points"]].rename(columns={"points": "predicted_away_points"})
    out = home.merge(away, on="game_id")
    out["predicted_total"] = out["predicted_home_points"] + out["predicted_away_points"]
    return out[columns]
