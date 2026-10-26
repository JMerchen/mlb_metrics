"""Backtest how QB+ (nfl_units / nfl_totals) rates a quarterback: which
pre-game rating best predicts how he actually passes in his next game?

Every regular-season game a QB took at least MIN_DROPBACKS dropbacks in,
2017 on, is predicted from his earlier dropbacks only. Candidates:

  current        his EPA per dropback this season, shrunk 150 dropbacks
                 toward replacement level (league - 0.10) - QB+ today
  recency        the same over his last 16 games, any season (QB+'s
                 "recent games" view)
  int_reg_K      interceptions regressed: his non-interception EPA plus an
                 interception charge at his INT rate shrunk toward the
                 league's with K pseudo-dropbacks, priced at the league's
                 average EPA per interception (K = inf: league INT rate)
  own_prior      shrunk toward his own last season (itself shrunk toward the
                 league) instead of replacement level, when he had one
  combos of the two

Scored, weighted by each game's dropbacks, against that game's EPA per
dropback (what QB+ means) and yards per attempt, with a paired test on
per-game squared error against `current`. League rates come from the
previous season, so nothing peeks ahead.

RESULT (2026-10-26, 4,549 QB-games 2017-2026, correlation with the game's
EPA per dropback; p = paired test of squared error against `current`):

    current (this season, toward replacement)   .241   weeks 1-4 .168
    recency (last 16 games)                     .281   weeks 1-4 .235   p < .001
    own_prior (toward his own last season)      .280   weeks 1-4 .241   p < .001
    int_reg_0 ... int_reg_inf                   .240 down to .232      (worse)

History is what's missing from the season view; discounting interceptions
hurts - in the NFL a QB's interceptions say something about his next game.
Adopted as nfl_units.DEFAULT_VIEW moving off "season" (see
scripts/backtest_nfl_unit_views.py); the game picks and points models
already rate QBs over their last 16 games.

Usage:
    python scripts/backtest_nfl_qb_rating.py [--first 2016] [--last 2026]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from mlb_metrics import config, nfl_data

MIN_DROPBACKS = 15
COLUMNS = ["season", "week", "season_type", "game_id", "passer_player_id", "passer_player_name", "posteam",
           "qb_dropback", "qb_epa", "interception", "sack", "passing_yards", "pass_attempt"]


def qb_games(raw_dir, seasons):
    """One row per QB per game: dropbacks, EPA, interceptions and their
    EPA, sacks, attempts, passing yards."""
    frames = []
    for season in seasons:
        pbp = nfl_data.load_persisted_table(raw_dir, "pbp", season)
        if pbp is None:
            continue
        pbp = pbp[[c for c in COLUMNS if c in pbp.columns]]
        d = pbp[(pbp["season_type"] == "REG") & (pbp["qb_dropback"] == 1) & pbp["passer_player_id"].notna() & pbp["qb_epa"].notna()].copy()
        d["int_epa"] = d["qb_epa"].where(d["interception"] == 1, 0.0)
        d["att"] = ((d["pass_attempt"] == 1) & (d["sack"] != 1)).astype(int)
        frames.append(d.groupby(["season", "week", "game_id", "passer_player_id", "posteam"]).agg(
            name=("passer_player_name", "first"), db=("qb_epa", "size"), epa=("qb_epa", "sum"),
            ints=("interception", "sum"), int_epa=("int_epa", "sum"), sacks=("sack", "sum"),
            att=("att", "sum"), yards=("passing_yards", "sum")).reset_index())
    games = pd.concat(frames, ignore_index=True).rename(columns={"passer_player_id": "qb"})
    return games.sort_values(["qb", "season", "week"]).reset_index(drop=True)


def league_by_season(games):
    """Each season's league rates, to be used by the NEXT season."""
    g = games.groupby("season")[["db", "epa", "ints", "int_epa"]].sum()
    lg = pd.DataFrame({"epa_db": g["epa"] / g["db"], "int_rate": g["ints"] / g["db"], "int_cost": g["int_epa"] / g["ints"]})
    lg.index = lg.index + 1
    return lg


def prior_sums(games, cols, scope):
    """Sums of `cols` over each QB's earlier games: this season ("season"),
    his last 16 games ("recent"), or all of last season ("last_season")."""
    if scope == "season":
        return games.groupby(["qb", "season"])[cols].cumsum() - games[cols]
    if scope == "recent":
        return games.groupby("qb")[cols].transform(lambda s: s.shift(1).rolling(config.NFL_TOTALS_QB_GAMES, min_periods=1).sum()).fillna(0)
    per_season = games.groupby(["qb", "season"])[cols].sum().reset_index()
    per_season["season"] += 1
    return games[["qb", "season"]].merge(per_season, on=["qb", "season"], how="left")[cols].fillna(0).set_index(games.index)


def ratings(games, lg):
    cols = ["db", "epa", "ints", "int_epa"]
    season = prior_sums(games, cols, "season")
    recent = prior_sums(games, cols, "recent")
    last = prior_sums(games, cols, "last_season")
    L = games[["season"]].join(lg, on="season")
    rep = L["epa_db"] - config.NFL_TOTALS_QB_REPLACEMENT_GAP
    k = config.NFL_TOTALS_QB_SHRINK_DROPBACKS

    def shrink(epa, db, target):
        return (epa + k * target) / (db + k)

    def int_regressed(s, k_int):
        rate = L["int_rate"] if np.isinf(k_int) else (s["ints"] + k_int * L["int_rate"]) / (s["db"] + k_int)
        return s["epa"] - s["int_epa"] + s["db"] * rate * L["int_cost"]

    # His own last season, shrunk toward the league with 300 dropbacks, as
    # the target when he has one (100+ dropbacks); replacement otherwise.
    last_rating = (last["epa"] + 300 * L["epa_db"]) / (last["db"] + 300)
    own_target = last_rating.where(last["db"] >= 100, rep)
    last_rating_int = (int_regressed(last, 500) + 300 * L["epa_db"]) / (last["db"] + 300)
    own_target_int = last_rating_int.where(last["db"] >= 100, rep)

    out = pd.DataFrame(index=games.index)
    out["current"] = shrink(season["epa"], season["db"], rep)
    out["recency"] = shrink(recent["epa"], recent["db"], rep)
    for k_int in (0, 250, 500, 1000, np.inf):
        out[f"int_reg_{k_int}"] = shrink(int_regressed(season, k_int), season["db"], rep)
    out["own_prior"] = shrink(season["epa"], season["db"], own_target)
    out["own_prior_int_reg_500"] = shrink(int_regressed(season, 500), season["db"], own_target_int)
    out["recency_int_reg_500"] = shrink(int_regressed(recent, 500), recent["db"], rep)
    return out


def score(games, preds, first_scored):
    from scipy.stats import ttest_rel

    keep = (games["db"] >= MIN_DROPBACKS) & (games["season"] >= first_scored) & preds.notna().all(axis=1)
    g, p = games[keep], preds[keep]
    actual = g["epa"] / g["db"]
    ypa = g["yards"] / g["att"].replace(0, np.nan)
    w = g["db"]

    def wcorr(x, y, w):
        ok = x.notna() & y.notna()
        x, y, w = x[ok], y[ok], w[ok]
        mx, my = np.average(x, weights=w), np.average(y, weights=w)
        return np.average((x - mx) * (y - my), weights=w) / np.sqrt(np.average((x - mx) ** 2, weights=w) * np.average((y - my) ** 2, weights=w))

    base_err = (actual - p["current"]) ** 2 * w
    rows = []
    for col in p.columns:
        err = (actual - p[col]) ** 2 * w
        t, pv = ttest_rel(err, base_err) if col != "current" else (np.nan, np.nan)
        early = g["week"] <= 4
        rows.append({"rating": col, "epa_corr": wcorr(p[col], actual, w), "epa_mse": err.sum() / w.sum(),
                     "epa_corr_wk1_4": wcorr(p[col][early], actual[early], w[early]),
                     "epa_corr_wk5+": wcorr(p[col][~early], actual[~early], w[~early]),
                     "ypa_corr": wcorr(p[col], ypa, w), "p_vs_current": pv})
    return pd.DataFrame(rows).set_index("rating"), len(g)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw/nfl")
    parser.add_argument("--first", type=int, default=2016)
    parser.add_argument("--last", type=int, default=config.NFL_SEASON)
    args = parser.parse_args()

    games = qb_games(args.raw_dir, range(args.first, args.last + 1))
    lg = league_by_season(games)
    preds = ratings(games, lg)
    table, n = score(games, preds, args.first + 1)
    pd.set_option("display.width", 160)
    print(f"{n} QB-games ({args.first + 1}-{args.last}, {MIN_DROPBACKS}+ dropbacks)")
    print(table.round(4).to_string())

    # The example that prompted this: Seattle's Darnold going into week 7, 2026.
    target = games[(games["name"] == "S.Darnold") & (games["season"] == args.last)]
    if not target.empty:
        last = target.index[-1]
        print("\nDarnold's rating after his latest game (EPA per dropback):")
        print(preds.loc[last].round(3).to_string())


if __name__ == "__main__":
    main()
