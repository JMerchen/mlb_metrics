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
plus two things about the game itself, both known before kickoff:
    qb_epa     the listed starting QB's own EPA per dropback over his last
               config.NFL_TOTALS_QB_GAMES games, shrunk toward a
               replacement-level rate (a QB with no history is replacement
               level) - so a backup starting moves the number, which a
               team-level rating built on the starter's games can't see
    dome       1 when the game is under a roof (dome, or retractable roof
               closed); a stadium whose roof isn't announced yet takes its
               usual setting from earlier games there
A team's points in a game are then a linear function of its offense and
the opponent's defense:
    points = const + b1*off_epa + b2*opp_def_epa + b3*off_plays
             + b4*opp_def_plays + b5*home + b6*dome + b7*qb_epa
with the coefficients in config.NFL_TOTALS_COEFFICIENTS, fit by
scripts/backtest_nfl_totals.py. The game's total is the two teams' sum.
Temperature and wind help on history but aren't published before the
game (nflverse leaves them blank for upcoming games), so they're left out.

Plays are dropbacks and designed runs (`pass` or `rush` flagged, kneels and
spikes dropped) - the same set the category ratings backtest uses.

Validated in scripts/backtest_nfl_totals.py (season by season, refit on
earlier seasons only) - see that script's docstring for the numbers. Shown
next to Vegas's total, never turned into bet advice.
"""

import pandas as pd

from mlb_metrics import config

# Schedules keep each franchise's code at the time (OAK, SD, STL);
# play-by-play is restated in today's codes.
NFL_TEAM_ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

TOTALS_FEATURE_COLUMNS = ["off_epa", "opp_def_epa", "off_plays", "opp_def_plays", "home", "dome", "qb_epa"]

PBP_COLUMNS = ["game_id", "season", "week", "season_type", "posteam", "defteam", "home_team",
                "epa", "pass", "rush", "qb_kneel", "qb_spike"]

QB_PBP_COLUMNS = ["game_id", "season", "week", "season_type", "posteam", "defteam",
                  "passer_player_id", "qb_epa", "qb_dropback"]

_ROOFED = {"dome", "closed"}

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


def qb_game_epa(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (regular-season game, passer): EPA summed over his
    dropbacks and the dropback count, with his team and the defense he
    faced. Columns: game_id, season, week, qb, team, opp, epa_sum,
    dropbacks."""
    columns = ["game_id", "season", "week", "qb", "team", "opp", "epa_sum", "dropbacks"]
    if pbp is None or pbp.empty or not set(QB_PBP_COLUMNS).issubset(pbp.columns):
        return pd.DataFrame(columns=columns)
    dropbacks = pbp[
        (pbp["season_type"] == "REG") & (pbp["qb_dropback"] == 1)
        & pbp["passer_player_id"].notna() & pbp["qb_epa"].notna()
    ]
    return (
        dropbacks.groupby(["game_id", "season", "week", "passer_player_id", "posteam", "defteam"])
        .agg(epa_sum=("qb_epa", "sum"), dropbacks=("qb_epa", "size"))
        .reset_index()
        .rename(columns={"passer_player_id": "qb", "posteam": "team", "defteam": "opp"})[columns]
    )


def starter_qb_epa(qb_games: pd.DataFrame, qb_ids: pd.Series) -> pd.Series:
    """Each listed starter's EPA per dropback over his last
    config.NFL_TOTALS_QB_GAMES games in `qb_games` (pass only earlier
    games), shrunk toward replacement level (the league rate minus
    config.NFL_TOTALS_QB_REPLACEMENT_GAP) with
    config.NFL_TOTALS_QB_SHRINK_DROPBACKS pseudo-dropbacks. A starter with
    no dropbacks - or no listed starter - is replacement level. NaN for
    every row when `qb_games` is empty (nothing to rate anyone against)."""
    if qb_games is None or qb_games.empty:
        return pd.Series(float("nan"), index=qb_ids.index)
    replacement = qb_games["epa_sum"].sum() / qb_games["dropbacks"].sum() - config.NFL_TOTALS_QB_REPLACEMENT_GAP
    shrink = config.NFL_TOTALS_QB_SHRINK_DROPBACKS
    recent = (
        qb_games.sort_values(["season", "week"])
        .groupby("qb").tail(config.NFL_TOTALS_QB_GAMES)
        .groupby("qb")[["epa_sum", "dropbacks"]].sum()
    )
    rating = (recent["epa_sum"] + shrink * replacement) / (recent["dropbacks"] + shrink)
    return qb_ids.map(rating).fillna(replacement).astype(float)


def dome_flags(games: pd.DataFrame, history_schedules: pd.DataFrame | None = None) -> pd.Series:
    """1.0 for a game under a roof (`roof` dome or closed), 0.0 outdoors
    or open. A game whose roof is blank (a retractable roof not announced
    yet) takes the most common setting from earlier games at the same
    stadium (or home team, without stadium ids); 0.0 with no history."""
    roof = games["roof"] if "roof" in games.columns else pd.Series(pd.NA, index=games.index)
    flags = roof.isin(_ROOFED).astype(float).where(roof.notna())
    if (flags.isna().any() and history_schedules is not None and not history_schedules.empty
            and "roof" in history_schedules.columns):
        key = "stadium_id" if "stadium_id" in games.columns and "stadium_id" in history_schedules.columns else "home_team"
        known = history_schedules[history_schedules["roof"].notna()]
        usual = known.assign(roofed=known["roof"].isin(_ROOFED).astype(float)).groupby(key)["roofed"].mean().round()
        flags = flags.fillna(games[key].map(usual))
    return flags.fillna(0.0)


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


def matchup_features(
    ratings: pd.DataFrame,
    games: pd.DataFrame,
    qb_games: pd.DataFrame | None = None,
    history_schedules: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Two rows per game (home offense, away offense) with
    TOTALS_FEATURE_COLUMNS. `games` needs game_id, home_team, away_team,
    and for the game-level features home_qb_id/away_qb_id and roof (plus
    stadium_id when it has one). A team with no rating (no earlier games)
    gets NaN features; so does every row's qb_epa without `qb_games`."""
    games = games.assign(home_team=_canonical(games["home_team"]), away_team=_canonical(games["away_team"]))
    dome = dome_flags(games, history_schedules)
    no_qb = pd.Series(pd.NA, index=games.index, dtype=object)
    sides = pd.concat([
        pd.DataFrame({"game_id": games["game_id"], "team": games["home_team"], "opp": games["away_team"], "home": 1.0,
                      "dome": dome, "starter": games.get("home_qb_id", no_qb)}),
        pd.DataFrame({"game_id": games["game_id"], "team": games["away_team"], "opp": games["home_team"], "home": 0.0,
                      "dome": dome, "starter": games.get("away_qb_id", no_qb)}),
    ], ignore_index=True)
    sides["qb_epa"] = starter_qb_epa(qb_games, sides["starter"])
    offense = ratings[["team", "off_epa", "off_plays"]]
    defense = ratings[["team", "def_epa", "def_plays"]].rename(
        columns={"team": "opp", "def_epa": "opp_def_epa", "def_plays": "opp_def_plays"}
    )
    return sides.merge(offense, on="team", how="left").merge(defense, on="opp", how="left")


EPA_DISPLAY_COLUMNS = [
    "home_off_epa", "home_off_rank", "home_def_epa", "home_def_rank",
    "away_off_epa", "away_off_rank", "away_def_epa", "away_def_rank",
]


def ranked_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """wave_ratings' output with league ranks: off_rank 1 = most EPA per
    play gained, def_rank 1 = least EPA per play allowed."""
    ranked = ratings.copy()
    ranked["off_rank"] = ranked["off_epa"].rank(ascending=False, method="min")
    ranked["def_rank"] = ranked["def_epa"].rank(ascending=True, method="min")
    return ranked


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
    qb_games: pd.DataFrame | None = None,
    history_schedules: pd.DataFrame | None = None,
    coefficients: dict | None = None,
) -> pd.DataFrame:
    """Predicted points for each game in `games` from the earlier games in
    `team_games` (team_game_epa's output) and `qb_games` (qb_game_epa's),
    with `history_schedules` filling in roofs not announced yet. Columns:
    game_id, predicted_home_points, predicted_away_points, predicted_total;
    NaN for a game where either team has no history, or without
    `qb_games`. Also each side's offense and defense EPA per play and
    league rank (EPA_DISPLAY_COLUMNS), for showing on the game card."""
    columns = ["game_id", "predicted_home_points", "predicted_away_points", "predicted_total", *EPA_DISPLAY_COLUMNS]
    if games is None or games.empty:
        return pd.DataFrame(columns=columns)
    ratings = wave_ratings(team_games)
    features = matchup_features(ratings, games, qb_games, history_schedules)
    features["points"] = predict_team_points(features, coefficients)
    home = features[features["home"] == 1.0][["game_id", "points"]].rename(columns={"points": "predicted_home_points"})
    away = features[features["home"] == 0.0][["game_id", "points"]].rename(columns={"points": "predicted_away_points"})
    out = home.merge(away, on="game_id")
    out["predicted_total"] = out["predicted_home_points"] + out["predicted_away_points"]
    # Each side's own ratings, for showing next to the prediction.
    ranked = ranked_ratings(ratings)[["team", "off_epa", "off_rank", "def_epa", "def_rank"]]
    sides = games[["game_id"]].assign(home=_canonical(games["home_team"]), away=_canonical(games["away_team"]))
    for side in ("home", "away"):
        sides = sides.merge(
            ranked.rename(columns={"team": side, **{c: f"{side}_{c}" for c in ranked.columns if c != "team"}}),
            on=side, how="left",
        )
    out = out.merge(sides.drop(columns=["home", "away"]), on="game_id", how="left")
    return out[columns]
