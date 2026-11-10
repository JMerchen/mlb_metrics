"""College basketball (Division I men) team ratings and game odds.

User request (2026-11-10): "create a college basketball tab to try and
predict March madness."

Data: hoopR's ESPN schedules and team box scores (cbb_data.py), every
Division I game since 2008. A game's pace is its possessions (field-goal
attempts - offensive rebounds + turnovers + 0.475 x free-throw attempts,
the two teams averaged), and each side's efficiency is its points per 100
possessions.

Ratings (fit_ratings): every team gets an offense number and a defense
number, in points per 100 possessions better than the average Division I
team, solved together over all the season's games so each performance is
judged against the opponent it came against and where it was played:

    efficiency = league average + offense(team) - defense(opponent) + home edge

by ridge regression, each rating pulled toward a prior (last season's
rating, carried over in part) with the weight of a few games - early in a
season the prior does most of the work, by January the games do. Tempo
(possessions per game) is solved the same way.

A game (predict_games): the two net ratings' difference, times the pace
the two teams figure to play at, plus the home edge, is the expected
margin; the win chance is read off a normal curve around it.

Settings and their backtest are in config.py (CBB_*) and
scripts/backtest_cbb_ratings.py.
"""

import re

import numpy as np
import pandas as pd
from scipy.stats import norm

from mlb_metrics import config

TOURNAMENT_PATTERN = r"men's basketball championship"
ROUND_NAMES = ["First Four", "Round of 64", "Round of 32", "Sweet 16", "Elite 8", "Final Four", "Championship"]
_ROUND_PATTERNS = [
    (r"first four|opening round|play-?in", "First Four"),
    (r"national championship|championship game", "Championship"),
    (r"final four|national semifinal", "Final Four"),
    (r"elite (8|eight)|regional final", "Elite 8"),
    (r"sweet (16|sixteen)|regional semifinal", "Sweet 16"),
    (r"2nd round|second round", "Round of 32"),
    (r"1st round|first round", "Round of 64"),
]
REGIONS = ["East", "West", "South", "Midwest"]

GAME_COLUMNS = [
    "game_id", "season", "date", "home_id", "away_id", "home_score", "away_score", "neutral", "possessions",
    "home_d1", "away_d1", "tournament", "round", "region", "home_seed", "away_seed",
]


def _possessions(team_box: pd.DataFrame) -> pd.Series:
    """Each game's possessions, the two teams' estimates averaged, indexed
    by game_id. Empty without box scores."""
    if team_box is None or team_box.empty:
        return pd.Series(dtype=float)
    box = team_box
    turnovers = pd.to_numeric(box["total_turnovers"], errors="coerce") if "total_turnovers" in box.columns \
        else pd.Series(float("nan"), index=box.index)
    if "turnovers" in box.columns:
        turnovers = turnovers.fillna(pd.to_numeric(box["turnovers"], errors="coerce"))
    num = lambda c: pd.to_numeric(box[c], errors="coerce")
    estimate = (num("field_goals_attempted") - num("offensive_rebounds") + turnovers
                + config.CBB_FREE_THROW_POSSESSION_WEIGHT * num("free_throws_attempted"))
    estimate = estimate.where(estimate > 30)   # a garbled box score isn't a pace
    return estimate.groupby(box["game_id"].astype(str)).mean()


def tournament_round(headline) -> tuple[str | None, str | None]:
    """(round, region) of an NCAA tournament game from its schedule note -
    "Men's Basketball Championship - East Region - Sweet 16" - with the
    rounds named as ROUND_NAMES. (None, None) for any other game. Before
    2016 the notes called the round of 64 the "2nd Round" and the round of
    32 the "3rd Round" (2011-15); prepare_games corrects for that from the
    number of games in each."""
    text = "" if headline is None or (isinstance(headline, float) and np.isnan(headline)) else str(headline).lower()
    if not re.search(TOURNAMENT_PATTERN, text) or "women" in text or "nit" in text.split():
        return None, None
    found = re.search(r"\b(east|west|south|midwest) region", text)   # \b: "midwest region" isn't "west region"
    region = found.group(1).capitalize() if found else None
    if "3rd round" in text or "third round" in text:
        return "3rd Round", region
    for pattern, name in _ROUND_PATTERNS:
        if re.search(pattern, text):
            return name, region
    return "Round of 64", region


def prepare_games(schedules: pd.DataFrame, team_box: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per completed game (GAME_COLUMNS): who played, the score,
    whether it was on a neutral floor, its possessions (from the box score;
    a game without one takes the pace its total points imply), whether each
    side is a Division I team (it has a conference), and for an NCAA
    tournament game its round, region and seeds (seeds are in the data
    from 2023 on)."""
    s = schedules
    done = s[s["status_type_completed"].fillna(False).astype(bool)].copy()
    done = done[pd.to_numeric(done["home_score"], errors="coerce").notna()
                & pd.to_numeric(done["away_score"], errors="coerce").notna()]
    out = pd.DataFrame({
        "game_id": done["game_id"].astype(str),
        "season": done["season"].astype(int),
        "date": pd.to_datetime(done["game_date"].astype(str).str[:10]),
        "home_id": done["home_id"].astype(int),
        "away_id": done["away_id"].astype(int),
        "home_score": pd.to_numeric(done["home_score"]).astype(float),
        "away_score": pd.to_numeric(done["away_score"]).astype(float),
        "neutral": done["neutral_site"].fillna(False).astype(bool),
        "home_d1": done["home_conference_id"].notna(),
        "away_d1": done["away_conference_id"].notna(),
    })
    out = out[(out["home_score"] + out["away_score"]) > 0]
    out = out.drop_duplicates("game_id")
    possessions = out["game_id"].map(_possessions(team_box))
    implied = (out["home_score"] + out["away_score"]) / 2 / config.CBB_POINTS_PER_POSSESSION
    out["possessions"] = possessions.fillna(implied).clip(lower=40)

    notes = done.drop_duplicates("game_id").set_index(done.drop_duplicates("game_id")["game_id"].astype(str))
    headline = out["game_id"].map(notes["notes_headline"]) if "notes_headline" in notes.columns else pd.Series(None, index=out.index)
    postseason = out["game_id"].map(notes["season_type"]) == 3 if "season_type" in notes.columns else True
    rounds = [tournament_round(h) if ok else (None, None) for h, ok in zip(headline, np.broadcast_to(postseason, len(out)))]
    out["round"] = [r for r, _ in rounds]
    out["region"] = [g for _, g in rounds]
    # 2011-15: "2nd Round" was the round of 64 and "3rd Round" the round of
    # 32. A season with a "3rd Round" used that naming.
    for season, group in out.groupby("season"):
        if (group["round"] == "3rd Round").any():
            out.loc[group.index[group["round"] == "Round of 32"], "round"] = "Round of 64"
            out.loc[group.index[group["round"] == "Round of 64"], "round"] = "First Four"
            out.loc[group.index[group["round"] == "3rd Round"], "round"] = "Round of 32"
    out["tournament"] = out["round"].notna()
    for side in ("home", "away"):
        column = f"{side}_current_rank"
        seed = pd.to_numeric(out["game_id"].map(notes[column]), errors="coerce") if column in notes.columns \
            else pd.Series(float("nan"), index=out.index)
        out[f"{side}_seed"] = seed.where(out["tournament"] & seed.between(1, 16))
    return out[GAME_COLUMNS].sort_values(["date", "game_id"]).reset_index(drop=True)


def team_names(schedules: pd.DataFrame) -> pd.DataFrame:
    """[team_id, team, short, abbreviation, conference_id, color, logo] for
    every Division I team in `schedules` (a team with a conference)."""
    frames = []
    for side in ("home", "away"):
        columns = {f"{side}_id": "team_id", f"{side}_location": "team", f"{side}_short_display_name": "short",
                   f"{side}_abbreviation": "abbreviation", f"{side}_conference_id": "conference_id",
                   f"{side}_color": "color", f"{side}_logo": "logo"}
        frames.append(schedules[[c for c in columns if c in schedules.columns]].rename(columns=columns))
    teams = pd.concat(frames, ignore_index=True)
    teams = teams[teams["conference_id"].notna()].drop_duplicates("team_id")
    teams["team_id"] = teams["team_id"].astype(int)
    teams["conference_id"] = teams["conference_id"].astype(int)
    return teams.reset_index(drop=True)


def preseason_ranked(schedules: pd.DataFrame) -> set:
    """The teams ranked 1-25 by the AP poll when they played their first
    game of the season (the schedule carries each side's rank at game
    time; 99 is unranked). Empty for a season whose data has no ranks
    (before 2023)."""
    if "home_current_rank" not in schedules.columns:
        return set()
    first = pd.concat([
        schedules[["game_date", f"{side}_id", f"{side}_current_rank"]].set_axis(["date", "team_id", "rank"], axis=1)
        for side in ("home", "away")
    ])
    first = first[first["team_id"] > 0].sort_values("date").drop_duplicates("team_id")
    rank = pd.to_numeric(first["rank"], errors="coerce")
    return {int(t) for t in first.loc[rank.between(1, 25), "team_id"]}


RATING_COLUMNS = ["team_id", "offense", "defense", "net", "tempo", "games", "wins", "losses"]


def fit_ratings(
    games: pd.DataFrame,
    prior: pd.DataFrame | None = None,
    as_of=None,
    ridge: float | None = None,
    prior_weight: float | None = None,
    carryover: float | None = None,
    half_life_days: float | None = None,
    margin_cap: float | None = None,
    home_edge_games: float | None = None,
    home_edge: float | None = None,
    ranked=None,
    ranked_bonus: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Ratings from one season's `games` (prepare_games rows; only games
    between two Division I teams count), as of `as_of` (games before that
    date; all of them when None).

    Returns (ratings, league): ratings is RATING_COLUMNS - offense and
    defense in points per 100 possessions better than average (a good
    defense is positive), net their sum, tempo in possessions per game
    against an average team - and league holds "efficiency" (the average
    points per 100 possessions), "home_edge" (points per 100 possessions a
    home team gains on each end), "tempo" (average possessions) and
    "games".

    `prior` is last season's ratings: a team's prior is `carryover` of each
    of its numbers, held with the weight of `prior_weight` games; a team
    without one (new to Division I) starts at average with weight `ridge`.
    `half_life_days` weights recent games more (None = all the same),
    `margin_cap` caps how far above or below expectation one game's
    efficiency can count, in points per 100 possessions, and
    `ranked` is the teams in the preseason AP top 25 (preseason_ranked):
    each one's prior net rating is raised by `ranked_bonus`.
    `home_edge_games` is how firmly the home edge is held at `home_edge`
    (config.CBB_HOME_EDGE_PRIOR; in November the home teams are mostly big
    programs hosting small ones, and a freely fit edge soaks that up)."""
    ridge = config.CBB_RIDGE_GAMES if ridge is None else ridge
    prior_weight = config.CBB_PRIOR_GAMES if prior_weight is None else prior_weight
    carryover = config.CBB_PRIOR_CARRYOVER if carryover is None else carryover
    half_life_days = config.CBB_HALF_LIFE_DAYS if half_life_days is None else half_life_days
    margin_cap = config.CBB_MARGIN_CAP if margin_cap is None else margin_cap
    home_edge_games = config.CBB_HOME_EDGE_PRIOR_GAMES if home_edge_games is None else home_edge_games
    home_edge = config.CBB_HOME_EDGE_PRIOR if home_edge is None else home_edge
    ranked_bonus = config.CBB_AP_PRESEASON_BONUS if ranked_bonus is None else ranked_bonus

    g = games[games["home_d1"] & games["away_d1"]]
    if as_of is not None:
        g = g[g["date"] < pd.Timestamp(as_of)]
    prior_teams = prior["team_id"].astype(int).tolist() if prior is not None and not prior.empty else []
    ranked = {int(t) for t in ranked} if ranked is not None else set()
    teams = sorted(set(g["home_id"]) | set(g["away_id"]) | set(prior_teams) | ranked)
    index = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    league = {"efficiency": 100 * config.CBB_POINTS_PER_POSSESSION, "home_edge": home_edge,
              "tempo": config.CBB_AVERAGE_TEMPO, "games": int(len(g))}
    if n == 0:
        return pd.DataFrame(columns=RATING_COLUMNS), league

    # Priors: `carryover` of last season's number, or average for a new team.
    prior_off, prior_def, prior_tempo = np.zeros(n), np.zeros(n), np.zeros(n)
    strength = np.full(n, float(ridge))
    if prior_teams:
        p = prior.set_index(prior["team_id"].astype(int))
        for t in prior_teams:
            i = index[t]
            prior_off[i], prior_def[i] = carryover * p.at[t, "offense"], carryover * p.at[t, "defense"]
            prior_tempo[i] = config.CBB_TEMPO_CARRYOVER * (p.at[t, "tempo"] - config.CBB_AVERAGE_TEMPO) \
                if "tempo" in p.columns and pd.notna(p.at[t, "tempo"]) else 0.0
            strength[i] = prior_weight
    # A team in the preseason AP top 25 starts higher than its carryover:
    # the poll knows who transferred in and who left.
    for t in ranked:
        i = index[t]
        prior_off[i] += ranked_bonus / 2
        prior_def[i] += ranked_bonus / 2
        strength[i] = prior_weight

    games_played = np.zeros(n)
    wins, losses = np.zeros(n), np.zeros(n)
    if len(g):
        h = g["home_id"].map(index).to_numpy()
        a = g["away_id"].map(index).to_numpy()
        poss = g["possessions"].to_numpy(dtype=float)
        site = np.where(g["neutral"].to_numpy(), 0.0, 1.0)
        home_eff = 100 * g["home_score"].to_numpy() / poss
        away_eff = 100 * g["away_score"].to_numpy() / poss
        weight = poss / poss.mean()
        if half_life_days:
            last = pd.Timestamp(as_of) if as_of is not None else g["date"].max() + pd.Timedelta(days=1)
            age = (last - g["date"]).dt.days.to_numpy(dtype=float)
            weight = weight * 0.5 ** (age / half_life_days)
        np.add.at(games_played, h, 1), np.add.at(games_played, a, 1)
        home_won = g["home_score"].to_numpy() > g["away_score"].to_numpy()
        np.add.at(wins, h, home_won), np.add.at(wins, a, ~home_won)
        np.add.at(losses, h, ~home_won), np.add.at(losses, a, home_won)

        # Unknowns: offense (n), defense (n), league average, home edge.
        k = 2 * n + 2
        rows = len(g)
        X = np.zeros((2 * rows, k))
        r = np.arange(rows)
        X[r, h], X[r, n + a], X[r, 2 * n], X[r, 2 * n + 1] = 1, -1, 1, site          # home team's offense
        X[rows + r, a], X[rows + r, n + h], X[rows + r, 2 * n], X[rows + r, 2 * n + 1] = 1, -1, 1, -site
        y = np.concatenate([home_eff, away_eff])
        w = np.concatenate([weight, weight])
        penalty = np.concatenate([strength, strength, [0.0, home_edge_games]])
        target = np.concatenate([prior_off, prior_def, [0.0, home_edge]])

        def solve(y_values):
            A = X.T @ (X * w[:, None]) + np.diag(penalty)
            A[2 * n, 2 * n] += 1e-9
            return np.linalg.solve(A, X.T @ (w * y_values) + penalty * target)

        beta = solve(y)
        if margin_cap:
            fitted = X @ beta
            beta = solve(fitted + np.clip(y - fitted, -margin_cap, margin_cap))
        offense, defense = beta[:n], beta[n:2 * n]
        league["efficiency"], league["home_edge"] = float(beta[2 * n]), float(beta[2 * n + 1])

        # Tempo: possessions = average + tempo(home) + tempo(away).
        T = np.zeros((rows, n + 1))
        T[r, h], T[r, a], T[r, n] = 1, 1, 1
        tw = weight / weight.mean()
        t_penalty = np.concatenate([np.full(n, config.CBB_TEMPO_RIDGE_GAMES), [0.0]])
        t_target = np.concatenate([prior_tempo, [0.0]])
        t_beta = np.linalg.solve(T.T @ (T * tw[:, None]) + np.diag(t_penalty) + 1e-9 * np.eye(n + 1),
                                 T.T @ (tw * poss) + t_penalty * t_target)
        tempo, league["tempo"] = t_beta[:n], float(t_beta[n])
    else:
        offense, defense, tempo = prior_off, prior_def, prior_tempo

    ratings = pd.DataFrame({
        "team_id": teams, "offense": offense, "defense": defense, "net": offense + defense,
        "tempo": league["tempo"] + tempo, "games": games_played.astype(int),
        "wins": wins.astype(int), "losses": losses.astype(int),
    })
    return ratings[RATING_COLUMNS], league


def predict_games(home_net, away_net, home_tempo, away_tempo, neutral, league: dict,
                  sigma: float | None = None) -> pd.DataFrame:
    """Expected margin (home minus away, in points), expected possessions
    and the home win chance for games between teams with these net ratings
    and tempos. Arrays or scalars; `neutral` true takes the home edge out."""
    sigma = config.CBB_MARGIN_SIGMA if sigma is None else sigma
    home_net, away_net = np.atleast_1d(np.asarray(home_net, dtype=float)), np.atleast_1d(np.asarray(away_net, dtype=float))
    possessions = np.atleast_1d(np.asarray(home_tempo, dtype=float) + np.asarray(away_tempo, dtype=float) - league["tempo"])
    edge = np.where(np.atleast_1d(np.asarray(neutral, dtype=bool)), 0.0, 2 * league["home_edge"])
    margin = (home_net - away_net + edge) / 100 * possessions
    return pd.DataFrame({"margin": margin, "possessions": possessions,
                         "home_win_probability": norm.cdf(margin / sigma)})
