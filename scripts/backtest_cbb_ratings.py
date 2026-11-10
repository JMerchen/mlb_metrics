"""Backtest of the college basketball ratings (cbb_ratings.py) as a game
predictor, and of the settings in config.py (CBB_*).

Every season from 2009 on is replayed week by week: the ratings are refit
each Monday from the games before it (starting from last season's final
ratings as the prior, fit the same way), and that week's games between
Division I teams are predicted from them. So no game is ever predicted
with itself or anything after it in the ratings.

Scored: how far the expected margin missed (MAE), the log loss of the win
chance, and how often the favorite won - over every game, by part of the
season, and on NCAA tournament games two ways: with the ratings as they
stood that week ("live"), and with every tournament game predicted from
the ratings of Selection Sunday ("bracket"), which is what filling out a
bracket uses. Where the data has seeds (2023 on), picking the better seed
is the yardstick.

Usage:
    PYTHONPATH=src python3 scripts/backtest_cbb_ratings.py [--sweep] [--table] [--cache DIR]
(--cache defaults to data/raw/cbb; seasons not there are downloaded into it.)

RESULT: see config.py's CBB_* comments (filled in from this script).
"""

import argparse
import os
import sys
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")   # one thread a worker: the solves are small
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import cbb_bracket, cbb_data, cbb_ratings, config  # noqa: E402

DEFAULT_CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "cbb")
SEASONS = list(range(config.CBB_FIRST_SEASON, config.CBB_SEASON))
_GAMES = {}
_RANKED = {}     # season -> the preseason AP top 25


def load_games(cache: str) -> dict:
    games = {}
    for season in SEASONS:
        schedules = cbb_data.load_table("schedules", season, cache)
        games[season] = cbb_ratings.prepare_games(schedules, cbb_data.load_table("team_box", season, cache))
        _RANKED[season] = cbb_ratings.preseason_ranked(schedules)
    return games


def replay(games_by_season: dict, **settings) -> pd.DataFrame:
    """One row per predicted game: season, date, tournament, round, seeds,
    the actual margin (home minus away) and, from that week's ratings, the
    expected margin; `bracket_margin` is the expected margin from the
    ratings of the Monday of the tournament's first week, for tournament
    games."""
    out, prior = [], None
    for season in SEASONS:
        games = games_by_season[season]
        d1 = games[games["home_d1"] & games["away_d1"]]
        if prior is not None and len(d1):
            frozen = None
            first_tournament = d1.loc[d1["tournament"], "date"].min()
            week = d1["date"].min().normalize()
            while week <= d1["date"].max():
                this_week = d1[(d1["date"] >= week) & (d1["date"] < week + pd.Timedelta(days=7))]
                if len(this_week):
                    ratings, league = cbb_ratings.fit_ratings(games, prior=prior, as_of=week,
                                                              ranked=_RANKED.get(season), **settings)
                    r = ratings.set_index("team_id")
                    if pd.notna(first_tournament) and frozen is None and week + pd.Timedelta(days=7) > first_tournament:
                        frozen = (r, league)
                    known = this_week[this_week["home_id"].isin(r.index) & this_week["away_id"].isin(r.index)]
                    p = cbb_ratings.predict_games(r["net"].reindex(known["home_id"]), r["net"].reindex(known["away_id"]),
                                                  r["tempo"].reindex(known["home_id"]), r["tempo"].reindex(known["away_id"]),
                                                  known["neutral"], league, sigma=1.0)
                    rows = known[["game_id", "season", "date", "tournament", "round", "home_seed", "away_seed", "neutral"]].copy()
                    rows["actual"] = (known["home_score"] - known["away_score"]).to_numpy()
                    rows["margin"] = p["margin"].to_numpy()
                    rows["bracket_margin"] = float("nan")
                    if frozen is not None:
                        fr, fl = frozen
                        t = known[known["tournament"] & known["home_id"].isin(fr.index) & known["away_id"].isin(fr.index)]
                        fp = cbb_ratings.predict_games(fr["net"].reindex(t["home_id"]), fr["net"].reindex(t["away_id"]),
                                                       fr["tempo"].reindex(t["home_id"]), fr["tempo"].reindex(t["away_id"]),
                                                       t["neutral"], fl, sigma=1.0)
                        rows.loc[t.index, "bracket_margin"] = fp["margin"].to_numpy()
                    out.append(rows)
                week += pd.Timedelta(days=7)
        prior, _ = cbb_ratings.fit_ratings(games, prior=prior, ranked=_RANKED.get(season), **settings)
    return pd.concat(out, ignore_index=True)


def tournament_table(games_by_season: dict, cache: str) -> pd.DataFrame:
    """One row per past tournament, every game picked from the ratings as
    they stood before its first game: games and how many our favorite won
    (and the log loss of our win chances), the same against "the better
    seed wins" where the data has seeds, the champion, where we ranked it
    and - where the bracket can be rebuilt from the data (2023 on) - the
    title chance we gave it. This is the table on the site
    (data/backtest/cbb_tournament_backtest.csv)."""
    rows, prior = [], None
    for season in SEASONS:
        games = games_by_season[season]
        tournament = games[games["tournament"]]
        if prior is not None and len(tournament) >= 60:
            schedules = cbb_data.load_table("schedules", season, cache)
            names = cbb_ratings.team_names(schedules).set_index("team_id")["team"]
            ratings, league = cbb_ratings.fit_ratings(games, prior=prior, as_of=tournament["date"].min(),
                                                      ranked=_RANKED.get(season))
            r = ratings.set_index("team_id")
            t = tournament[tournament["home_id"].isin(r.index) & tournament["away_id"].isin(r.index)]
            p = cbb_ratings.predict_games(r["net"].reindex(t["home_id"]), r["net"].reindex(t["away_id"]),
                                          r["tempo"].reindex(t["home_id"]), r["tempo"].reindex(t["away_id"]),
                                          t["neutral"], league)
            home_won = (t["home_score"] > t["away_score"]).to_numpy()
            chance = np.clip(p["home_win_probability"].to_numpy(), 1e-6, 1 - 1e-6)
            right = (chance >= 0.5) == home_won
            seeded = (t["home_seed"].notna() & t["away_seed"].notna() & (t["home_seed"] != t["away_seed"])).to_numpy()
            chalk = ((t["home_seed"] < t["away_seed"]).to_numpy() == home_won)
            final = tournament[tournament["round"] == "Championship"]
            champion = None
            if len(final):
                g = final.iloc[0]
                champion = int(g["home_id"] if g["home_score"] > g["away_score"] else g["away_id"])
            row = {"season": season, "games": int(len(t)), "right": int(right.sum()),
                   "log_loss": round(float(-np.mean(np.where(home_won, np.log(chance), np.log(1 - chance)))), 4),
                   "seed_games": int(seeded.sum()), "seed_model_right": int(right[seeded].sum()),
                   "seed_chalk_right": int(chalk[seeded].sum()),
                   "champion": names.get(champion) if champion is not None else None,
                   "champion_rank": int((r["net"] > r.at[champion, "net"]).sum() + 1) if champion in r.index else None,
                   "champion_title_chance": None}
            field = cbb_bracket.official_field(schedules)
            if field is not None and champion is not None:
                pairs, _ = cbb_bracket.final_four_pairs(season, "official", schedules, field)
                odds = cbb_bracket.simulate(field, ratings, league, pairs).set_index("team_id")
                row["champion_title_chance"] = round(float(odds.at[champion, "champion"]), 4)
            rows.append(row)
        prior, _ = cbb_ratings.fit_ratings(games, prior=prior, ranked=_RANKED.get(season))
    return pd.DataFrame(rows)


def log_loss(margin, actual, sigma) -> float:
    p = np.clip(norm.cdf(np.asarray(margin) / sigma), 1e-6, 1 - 1e-6)
    won = np.asarray(actual) > 0
    return float(-np.mean(np.where(won, np.log(p), np.log(1 - p))))


def score(rows: pd.DataFrame, column: str = "margin", sigma: float | None = None) -> dict:
    rows = rows.dropna(subset=[column])
    if rows.empty:
        return {"games": 0}
    miss = rows["actual"] - rows[column]
    sigma = float(np.sqrt(np.mean(miss ** 2))) if sigma is None else sigma
    return {"games": len(rows), "mae": float(miss.abs().mean()), "sigma": sigma,
            "log_loss": log_loss(rows[column], rows["actual"], sigma),
            "right": float(((rows[column] > 0) == (rows["actual"] > 0)).mean())}


def _run(item):
    name, settings = item
    rows = replay(_GAMES, **settings)
    return name, rows


def summarize(name: str, rows: pd.DataFrame) -> dict:
    scored = rows[rows["season"] >= SEASONS[0] + 2]      # the first replayed season has a thin prior
    all_games = score(scored)
    sigma = all_games["sigma"]
    month = scored["date"].dt.month
    early, mid, late = scored[month.isin([11, 12])], scored[month.isin([1, 2])], scored[month.isin([3, 4])]
    tournament = scored[scored["tournament"]]
    return {"name": name, "games": all_games["games"], "mae": all_games["mae"], "sigma": sigma,
            "log_loss": all_games["log_loss"], "right": all_games["right"],
            "ll_nov_dec": score(early, sigma=sigma)["log_loss"], "ll_jan_feb": score(mid, sigma=sigma)["log_loss"],
            "ll_mar_apr": score(late, sigma=sigma)["log_loss"],
            "ll_nov_dec_2023on": score(early[early["season"] >= 2023], sigma=sigma)["log_loss"],
            "tourney_games": len(tournament), "tourney_ll_live": score(tournament, sigma=sigma)["log_loss"],
            "tourney_ll_bracket": score(tournament, "bracket_margin", sigma=sigma).get("log_loss"),
            "tourney_right_bracket": score(tournament, "bracket_margin", sigma=sigma).get("right")}


SWEEP = {
    "config": {},
    "ridge 2": {"ridge": 2.0}, "ridge 8": {"ridge": 8.0},
    "prior 3 games": {"prior_weight": 3.0}, "prior 10 games": {"prior_weight": 10.0}, "prior 16 games": {"prior_weight": 16.0},
    "carryover .5": {"carryover": 0.5}, "carryover .85": {"carryover": 0.85},
    "half-life 45d": {"half_life_days": 45.0}, "half-life 90d": {"half_life_days": 90.0}, "half-life 180d": {"half_life_days": 180.0},
    "cap 25": {"margin_cap": 25.0}, "cap 40": {"margin_cap": 40.0},
    "home edge free": {"home_edge_games": 0.0}, "home edge 2000": {"home_edge_games": 2000.0},
    "home edge fixed": {"home_edge_games": 1e7},
}
# The follow-up round, around the first one's best (home edge held fixed).
SWEEP2 = {
    "config": {},
    "carry .85 prior 6": {"carryover": 0.85, "prior_weight": 6.0},
    "carry .85 prior 4": {"carryover": 0.85, "prior_weight": 4.0},
    "carry 1.0 prior 4": {"carryover": 1.0, "prior_weight": 4.0},
    "carry 1.0 prior 6": {"carryover": 1.0, "prior_weight": 6.0},
    "carry .85 prior 8": {"carryover": 0.85, "prior_weight": 8.0},
    "home edge 2.2": {"home_edge": 2.2}, "home edge 3.0": {"home_edge": 3.0},
}
# The preseason AP poll (ranks are in the data from 2023 on, so only those
# seasons' early games can differ).
SWEEP3 = {"config": {}, "no AP bonus": {"ranked_bonus": 0.0}, "AP bonus 3": {"ranked_bonus": 3.0}, "AP bonus 6": {"ranked_bonus": 6.0}}


def main():
    global _GAMES
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep2", action="store_true")
    parser.add_argument("--sweep3", action="store_true")
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--out", default=None, help="write the config replay's game rows here (csv)")
    parser.add_argument("--table", action="store_true",
                        help="only build the past-tournaments table for the site and write it")
    args = parser.parse_args()
    _GAMES = load_games(args.cache)
    print({season: len(g) for season, g in _GAMES.items()})
    if args.table:
        table = tournament_table(_GAMES, args.cache)
        pd.set_option("display.width", 250)
        print(table.to_string(index=False))
        print(f"all: right {table['right'].sum()} of {table['games'].sum()} ({table['right'].sum() / table['games'].sum():.3f}),"
              f" log loss {np.average(table['log_loss'], weights=table['games']):.4f};"
              f" seeded games {table['seed_games'].sum()}: ours {table['seed_model_right'].sum()}, better seed {table['seed_chalk_right'].sum()}")
        path = os.path.join(os.path.dirname(__file__), "..", "data", "backtest", "cbb_tournament_backtest.csv")
        table.to_csv(path, index=False)
        return
    variants = SWEEP if args.sweep else SWEEP2 if args.sweep2 else SWEEP3 if args.sweep3 else {"config": {}}
    pd.set_option("display.width", 250)
    results = {}
    with Pool(min(4, len(variants))) as pool:
        for name, rows in pool.imap_unordered(_run, variants.items()):
            results[name] = rows
            print(pd.DataFrame([summarize(name, rows)]).round(4).to_string(index=False, header=len(results) == 1), flush=True)

    rows = results["config"]
    rows = rows[rows["season"] >= SEASONS[0] + 2]
    sigma = score(rows)["sigma"]
    t = rows[rows["tournament"]].dropna(subset=["bracket_margin"])
    by_round = t.groupby("round").apply(lambda g: pd.Series(score(g, "bracket_margin", sigma)), include_groups=False)
    print("\nTournament games from Selection Sunday's ratings, by round:")
    print(by_round.reindex(cbb_ratings.ROUND_NAMES)[["games", "right", "log_loss", "mae"]].round(3).to_string())
    by_season = t.groupby("season").apply(lambda g: pd.Series(score(g, "bracket_margin", sigma)), include_groups=False)
    print(by_season[["games", "right", "log_loss"]].round(3).to_string())
    seeded = t.dropna(subset=["home_seed", "away_seed"])
    seeded = seeded[seeded["home_seed"] != seeded["away_seed"]]
    if len(seeded):
        better_seed_won = ((seeded["home_seed"] < seeded["away_seed"]) == (seeded["actual"] > 0)).mean()
        ours = ((seeded["bracket_margin"] > 0) == (seeded["actual"] > 0)).mean()
        print(f"\nSeeded games with different seeds ({seeded['season'].min()}-{seeded['season'].max()}, {len(seeded)}):"
              f" better seed won {better_seed_won:.3f}, our favorite won {ours:.3f}")
    # Calibration of the win chance.
    p = norm.cdf(rows["margin"] / sigma)
    sure = np.maximum(p, 1 - p)
    right = (rows["margin"] > 0) == (rows["actual"] > 0)
    bands = pd.cut(sure, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], include_lowest=True)
    print("\nHow often the favorite won, by how sure we were:")
    print(pd.DataFrame({"games": right.groupby(bands, observed=True).size(), "said": pd.Series(sure).groupby(bands, observed=True).mean().round(3),
                        "right": right.groupby(bands, observed=True).mean().round(3)}).to_string())
    if args.out:
        results["config"].to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
