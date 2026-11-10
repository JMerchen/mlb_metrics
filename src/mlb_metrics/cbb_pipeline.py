"""The daily college basketball run: ratings, game picks and bracket odds
for the site's College Hoops page (docs/data/cbb.json), and the log every
game pick is graded from (data/predictions/cbb_game_predictions.csv).

Each run: pull the season's schedule and box scores (cbb_data.py), refit
the ratings from every game played (cbb_ratings.py; last season's ratings
are the prior), grade the picks whose games have finished, log a pick for
every game between Division I teams in the next few days (the first pick
logged for a game is the one that's graded), and simulate the bracket
(cbb_bracket.py) - the real one once it's out, a projection before then.
"""

import datetime
import json
import os
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import norm

from mlb_metrics import cbb_bracket, cbb_data, cbb_ratings, config

RAW_DIR = "data/raw/cbb"
LOG_PATH = "data/predictions/cbb_game_predictions.csv"
BACKTEST_PATH = "data/backtest/cbb_tournament_backtest.csv"
LOG_COLUMNS = [
    "date", "season", "game_id", "home_id", "away_id", "home_team", "away_team", "neutral",
    "home_win_probability", "predicted_margin", "predicted_home_points", "predicted_away_points",
    "model_version", "home_score", "away_score",
]
CONFERENCES = {
    1: "America East", 2: "ACC", 3: "Atlantic 10", 4: "Big East", 5: "Big Sky", 6: "Big South", 7: "Big Ten",
    8: "Big 12", 9: "Big West", 10: "CAA", 11: "Conference USA", 12: "Ivy", 13: "MAAC", 14: "MAC", 16: "MEAC",
    18: "Missouri Valley", 19: "NEC", 20: "Ohio Valley", 21: "Pac-12", 22: "Patriot", 23: "SEC", 24: "SoCon",
    25: "Southland", 26: "SWAC", 27: "Sun Belt", 29: "WCC", 30: "UAC", 43: "Independent", 44: "Mountain West",
    45: "Horizon", 46: "ASUN", 49: "Summit", 62: "American",
}
CONFIDENCE_BANDS = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]


def _json_safe(value):
    """`value` with every NaN (a blank in the data) as None: a browser
    won't parse NaN."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def today_eastern() -> datetime.date:
    return datetime.datetime.now(ZoneInfo("America/New_York")).date()


def season_ratings(season: int, raw_dir: str, schedules: pd.DataFrame, team_box: pd.DataFrame):
    """(games, ratings, league) for `season` from its schedule and box
    scores, with last season's final ratings as the prior (themselves fit
    on the season before's)."""
    prior = None
    for earlier in (season - 2, season - 1):
        try:
            past = cbb_ratings.prepare_games(cbb_data.load_table("schedules", earlier, raw_dir),
                                             cbb_data.load_table("team_box", earlier, raw_dir))
        except Exception as exc:
            print(f"WARNING: no {earlier} college basketball data ({exc}); ratings start without it.")
            continue
        prior, _ = cbb_ratings.fit_ratings(past, prior=prior)
    games = cbb_ratings.prepare_games(schedules, team_box)
    ratings, league = cbb_ratings.fit_ratings(games, prior=prior, ranked=cbb_ratings.preseason_ranked(schedules))
    if prior is not None:
        ratings["last_season_net"] = ratings["team_id"].map(prior.set_index("team_id")["net"])
    return games, ratings, league


def upcoming_games(schedules: pd.DataFrame, today: datetime.date, days: int) -> pd.DataFrame:
    """The season's unplayed games between Division I teams from `today`
    through `days` days on: [game_id, date, home_id, away_id, neutral]."""
    s = schedules
    date = pd.to_datetime(s["game_date"].astype(str).str[:10])
    window = (date >= pd.Timestamp(today)) & (date <= pd.Timestamp(today) + pd.Timedelta(days=days))
    waiting = ~s["status_type_completed"].fillna(False).astype(bool)
    d1 = s["home_conference_id"].notna() & s["away_conference_id"].notna()
    rows = s[window & waiting & d1 & (s["home_id"] > 0) & (s["away_id"] > 0)]
    return pd.DataFrame({
        "game_id": rows["game_id"].astype(str), "date": date[rows.index].dt.strftime("%Y-%m-%d"),
        "home_id": rows["home_id"].astype(int), "away_id": rows["away_id"].astype(int),
        "neutral": rows["neutral_site"].fillna(False).astype(bool),
    }).drop_duplicates("game_id").reset_index(drop=True)


def price_games(games: pd.DataFrame, ratings: pd.DataFrame, league: dict, names: pd.Series, season: int) -> pd.DataFrame:
    """LOG_COLUMNS rows for `games` (upcoming_games), priced from the
    ratings. A game with a team the ratings don't know is left out."""
    r = ratings.set_index("team_id")
    known = games[games["home_id"].isin(r.index) & games["away_id"].isin(r.index)]
    p = cbb_ratings.predict_games(r["net"].reindex(known["home_id"]), r["net"].reindex(known["away_id"]),
                                  r["tempo"].reindex(known["home_id"]), r["tempo"].reindex(known["away_id"]),
                                  known["neutral"], league)
    edge = np.where(known["neutral"], 0.0, league["home_edge"])
    possessions = p["possessions"].to_numpy()
    home_points = possessions / 100 * (league["efficiency"] + r["offense"].reindex(known["home_id"]).to_numpy()
                                       - r["defense"].reindex(known["away_id"]).to_numpy() + edge)
    away_points = possessions / 100 * (league["efficiency"] + r["offense"].reindex(known["away_id"]).to_numpy()
                                       - r["defense"].reindex(known["home_id"]).to_numpy() - edge)
    out = pd.DataFrame({
        "date": known["date"].to_numpy(), "season": season, "game_id": known["game_id"].to_numpy(),
        "home_id": known["home_id"].to_numpy(), "away_id": known["away_id"].to_numpy(),
        "home_team": known["home_id"].map(names).to_numpy(), "away_team": known["away_id"].map(names).to_numpy(),
        "neutral": known["neutral"].to_numpy(),
        "home_win_probability": p["home_win_probability"].round(4).to_numpy(),
        "predicted_margin": p["margin"].round(1).to_numpy(),
        "predicted_home_points": home_points.round(1), "predicted_away_points": away_points.round(1),
        "model_version": config.CBB_MODEL_VERSION, "home_score": float("nan"), "away_score": float("nan"),
    })
    return out[LOG_COLUMNS]


def update_log(log_path: str, new_picks: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Adds `new_picks` to the log (a game already in it keeps its first
    pick) and fills in the final score of every logged game that has been
    played (`games` is prepare_games' output). Returns the whole log."""
    if os.path.exists(log_path):
        log = pd.read_csv(log_path, dtype={"game_id": str})
        fresh = new_picks[~new_picks["game_id"].astype(str).isin(set(log["game_id"]))]
        log = pd.concat([log, fresh], ignore_index=True) if len(fresh) else log
    else:
        log = new_picks.copy()
    log["game_id"] = log["game_id"].astype(str)
    final = games.set_index("game_id")
    for side in ("home", "away"):
        score = log["game_id"].map(final[f"{side}_score"])
        log[f"{side}_score"] = pd.to_numeric(log[f"{side}_score"], errors="coerce").fillna(score)
    log = log.sort_values(["date", "game_id"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    log[LOG_COLUMNS].to_csv(log_path, index=False)
    return log[LOG_COLUMNS]


def pick_record(log: pd.DataFrame, season: int) -> dict:
    """How the season's graded picks have done: games, right, the average
    miss of the margin, and by how sure the pick was (`bands`: label,
    games, right)."""
    done = log[(log["season"] == season) & log["home_score"].notna() & log["away_score"].notna()]
    done = done[done["home_score"] != done["away_score"]]
    p = done["home_win_probability"].astype(float)
    home_won = done["home_score"] > done["away_score"]
    right = (p >= 0.5) == home_won
    sure = p.where(p >= 0.5, 1 - p)
    bands = []
    for low, high in CONFIDENCE_BANDS:
        inside = (sure >= low) & ((sure < high) | (high >= 1.0))
        bands.append({"label": f"{round(low * 100)}–{round(high * 100)}%", "low": low, "high": high,
                      "games": int(inside.sum()), "right": int(right[inside].sum())})
    miss = (done["home_score"] - done["away_score"] - done["predicted_margin"].astype(float)).abs()
    return {"games": int(len(done)), "right": int(right.sum()),
            "margin_miss": round(float(miss.mean()), 1) if len(done) else None, "bands": bands}


def build_bracket(season: int, schedules: pd.DataFrame, games: pd.DataFrame, ratings: pd.DataFrame,
                  league: dict, teams: pd.DataFrame) -> dict:
    """The bracket block of cbb.json: kind ("official" or "projected"),
    which regions meet in the semifinals (and whether that's known), and
    each team's seed, region and chance of reaching each round."""
    field = cbb_bracket.official_field(schedules)
    kind = "official" if field is not None else "projected"
    if field is None:
        if len(ratings.merge(teams[["team_id"]], on="team_id")) < 68:
            # Not enough rated teams to fill a bracket (only in a cut-down test season).
            return {"kind": kind, "final_four_pairs": [], "final_four_pairs_known": True,
                    "sims": config.CBB_BRACKET_SIMS, "teams": [], "results": []}
        field = cbb_bracket.projected_field(ratings, teams)
    pairs, pairs_known = cbb_bracket.final_four_pairs(season, kind, schedules, field)
    results = cbb_bracket.played_results(games) if kind == "official" else None
    odds = cbb_bracket.simulate(field, ratings, league, pairs, results=results)
    rows = field.merge(odds, on="team_id")
    return {
        "kind": kind, "final_four_pairs": pairs, "final_four_pairs_known": bool(pairs_known),
        "sims": config.CBB_BRACKET_SIMS,
        "teams": [{"team_id": int(row["team_id"]), "seed": int(row["seed"]), "region": row["region"],
                   "play_in": bool(row["play_in"]), "bid": row["bid"],
                   **{c: round(float(row[c]), 4) for c in cbb_bracket.ROUND_COLUMNS}}
                  for _, row in rows.iterrows()],
        "results": [{"round": g["round"], "home_id": int(g["home_id"]), "away_id": int(g["away_id"]),
                     "home_score": int(g["home_score"]), "away_score": int(g["away_score"])}
                    for _, g in games[games["tournament"]].iterrows()] if kind == "official" else [],
    }


def build_payload(season: int, schedules: pd.DataFrame, games: pd.DataFrame, ratings: pd.DataFrame, league: dict,
                  log: pd.DataFrame, today: datetime.date, backtest: pd.DataFrame | None = None) -> dict:
    """Everything the College Hoops page reads, as one dict."""
    teams = cbb_ratings.team_names(schedules)
    rated = ratings.merge(teams, on="team_id", how="inner").sort_values("net", ascending=False).reset_index(drop=True)
    rated["rank"] = np.arange(1, len(rated) + 1)
    rated["offense_rank"] = rated["offense"].rank(ascending=False, method="min").astype(int)
    rated["defense_rank"] = rated["defense"].rank(ascending=False, method="min").astype(int)
    played = games[games["home_d1"] & games["away_d1"]]
    team_rows = []
    for _, t in rated.iterrows():
        last = t.get("last_season_net")
        team_rows.append({
            "team_id": int(t["team_id"]), "team": t["team"], "abbreviation": t.get("abbreviation"),
            "conference": CONFERENCES.get(int(t["conference_id"]), "Other"), "color": t.get("color"),
            "rank": int(t["rank"]), "net": round(float(t["net"]), 1),
            "offense": round(float(t["offense"]), 1), "defense": round(float(t["defense"]), 1),
            "offense_rank": int(t["offense_rank"]), "defense_rank": int(t["defense_rank"]),
            "tempo": round(float(t["tempo"]), 1), "wins": int(t["wins"]), "losses": int(t["losses"]),
            "last_season_net": None if last is None or pd.isna(last) else round(float(last), 1),
        })
    window = log[(log["season"] == season)
                 & (log["date"] >= (pd.Timestamp(today) - pd.Timedelta(days=config.CBB_RESULT_DAYS)).strftime("%Y-%m-%d"))
                 & (log["date"] <= (pd.Timestamp(today) + pd.Timedelta(days=config.CBB_PICK_DAYS)).strftime("%Y-%m-%d"))]
    num = lambda v: None if pd.isna(v) else float(v)
    game_rows = [{"date": g["date"], "game_id": str(g["game_id"]), "home_id": int(g["home_id"]), "away_id": int(g["away_id"]),
                  "neutral": bool(g["neutral"]), "home_win_probability": float(g["home_win_probability"]),
                  "predicted_home_points": num(g["predicted_home_points"]), "predicted_away_points": num(g["predicted_away_points"]),
                  "home_score": num(g["home_score"]), "away_score": num(g["away_score"])}
                 for _, g in window.iterrows()]
    return {
        "season": season, "updated": today.isoformat(),
        "through": played["date"].max().strftime("%Y-%m-%d") if len(played) else None,
        "games_played": int(len(played)),
        "league": {"efficiency": round(league["efficiency"], 2), "home_edge": round(league["home_edge"], 2),
                   "tempo": round(league["tempo"], 2), "sigma": config.CBB_MARGIN_SIGMA},
        "teams": team_rows, "games": game_rows, "record": pick_record(log, season),
        "bracket": build_bracket(season, schedules, games, ratings, league, teams),
        "backtest": [] if backtest is None else json.loads(backtest.to_json(orient="records")),
    }


def run(season: int | None = None, raw_dir: str = RAW_DIR, output_dir: str = "docs/data", log_path: str = LOG_PATH,
        today: datetime.date | None = None) -> dict | None:
    season = season or config.CBB_SEASON
    today = today or today_eastern()
    try:
        schedules = cbb_data.fetch_table("schedules", season)
        team_box = cbb_data.fetch_table("team_box", season)
    except Exception as exc:
        print(f"Could not fetch the {season} college basketball data ({exc}) - nothing updated this run.")
        return None
    games, ratings, league = season_ratings(season, raw_dir, schedules, team_box)
    names = cbb_ratings.team_names(schedules).set_index("team_id")["team"]
    picks = price_games(upcoming_games(schedules, today, config.CBB_PICK_DAYS), ratings, league, names, season)
    log = update_log(log_path, picks, games)
    backtest = pd.read_csv(BACKTEST_PATH) if os.path.exists(BACKTEST_PATH) else None
    payload = build_payload(season, schedules, games, ratings, league, log, today, backtest)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "cbb.json"), "w") as fh:
        json.dump(_json_safe(payload), fh, separators=(",", ":"), allow_nan=False)
    record = payload["record"]
    print(f"College basketball {season}: {payload['games_played']} games through {payload['through']}, "
          f"{len(payload['teams'])} teams rated, {len(picks)} games priced, "
          f"picks right {record['right']} of {record['games']}, bracket {payload['bracket']['kind']}.")
    return payload
