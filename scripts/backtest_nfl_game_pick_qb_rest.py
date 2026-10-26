"""Backtest two additions to the NFL game-pick model, prompted by where it
falls short of Vegas's closing line (2018-2026, 2,212 games: log loss
.646 against Vegas's .611). Vegas rates a team about 0.3 logits lower than
we do when its starting QB changed from the previous game, and the
results side with Vegas; and Vegas prices rest days we ignore.

  qb_plus      each starter's QB+ rating (nfl_totals.starter_qb_epa: EPA per
               dropback over his last 16 games, shrunk toward replacement
               level - what QB+ backtested best as), home and away
  qb_delta     the starter's QB+ minus the QB+ of whoever started the
               team's previous game (0 when it's the same QB)
  rest_diff    home rest days minus away rest days (schedule)

Variants, each fit as the live model is (standardized LogisticRegression,
C picked by walk-forward validation 2018-2024), then scored on 2025 and
2026 (trained on every season before), with paired tests against today's.

RESULT (2026-10-26, 2,221 games 2018-2026, log loss, lower is better):

    variant                               validation  2025    2026    all     QB-changed games
    today                                   .6480    .6272   .6526   .6456       .6511
    qb_plus instead of qb_adjustment        .6356    .6288   .6428   .6351       .6378
    + qb_plus (keeping qb_adjustment)       .6368    .6286   .6428   .6360       .6394
    + qb_delta                              .6473    .6244   .6603   .6450       .6466
    + rest_diff                             .6485    .6275   .6511   .6460       .6510
    Vegas's closing line                             (2025-26 .6224)  .6113       .6008

QB+ in place of the QB-continuity adjustment was better in 8 of 9 seasons
(paired p < .0001), closing about a third of the gap to Vegas; rest days
and the change-of-starter term added nothing. Shipped as
nfl_game_picks.build_game_features_qb_plus, which rates QBs from the
weekly player stats the live pipeline already has (passing EPA per
attempt-or-sack, correlation .97 with the play-by-play version here): all
9 seasons better, .6458 -> .6368, accuracy 62.9% -> 64.1%, p = .0001.

Usage:
    python scripts/backtest_nfl_game_pick_qb_rest.py [--rows-cache rows.parquet]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

import grid_search_nfl_game_picks as grid
from mlb_metrics import nfl_data, nfl_game_picks, nfl_totals

SEASONS = list(range(2016, 2027))
VALIDATION_SEASONS = list(range(2018, 2025))
HOLDOUTS = [2025, 2026]
BASE = list(nfl_game_picks.DISAGGREGATED_FEATURE_COLUMNS)
NO_QB_ADJ = [c for c in BASE if not c.endswith("qb_adjustment")]
VARIANTS = {
    "today": BASE,
    "+ qb_plus": BASE + ["home_qb_plus", "away_qb_plus"],
    "qb_plus instead of qb_adjustment": NO_QB_ADJ + ["home_qb_plus", "away_qb_plus"],
    "+ qb_delta": BASE + ["home_qb_delta", "away_qb_delta"],
    "+ qb_plus + qb_delta": BASE + ["home_qb_plus", "away_qb_plus", "home_qb_delta", "away_qb_delta"],
    "+ rest_diff": BASE + ["rest_diff"],
    "+ qb_plus + qb_delta + rest_diff": BASE + ["home_qb_plus", "away_qb_plus", "home_qb_delta", "away_qb_delta", "rest_diff"],
}


def qb_features(raw_dir: str, schedules: pd.DataFrame) -> pd.DataFrame:
    """Per game: home/away starter QB+ and its change from the team's
    previous starter, each from games before that week only."""
    qb_games = pd.concat([nfl_totals.qb_game_epa(nfl_data.load_persisted_table(raw_dir, "pbp", s)[nfl_totals.QB_PBP_COLUMNS])
                          for s in SEASONS], ignore_index=True)
    games = schedules[schedules["game_type"] == "REG"].copy()
    # Previous game's starter for each team.
    starts = pd.concat([
        games[["season", "week", "home_team", "home_qb_id"]].set_axis(["season", "week", "team", "qb"], axis=1),
        games[["season", "week", "away_team", "away_qb_id"]].set_axis(["season", "week", "team", "qb"], axis=1),
    ]).sort_values(["team", "season", "week"])
    starts["prev_qb"] = starts.groupby("team")["qb"].shift()
    for side in ("home", "away"):
        games = games.merge(starts[["season", "week", "team", "prev_qb"]].rename(
            columns={"team": f"{side}_team", "prev_qb": f"{side}_prev_qb"}), on=["season", "week", f"{side}_team"], how="left")
    out = []
    order = qb_games["season"] * 100 + qb_games["week"]
    for (season, week), wk in games.groupby(["season", "week"]):
        history = qb_games[order < season * 100 + week]
        rate = {}
        for col in ("home_qb_id", "away_qb_id", "home_prev_qb", "away_prev_qb"):
            rate[col] = nfl_totals.starter_qb_epa(history, wk[col])
        out.append(pd.DataFrame({
            "game_id": wk["game_id"],
            "home_qb_plus": rate["home_qb_id"], "away_qb_plus": rate["away_qb_id"],
            "home_qb_delta": np.where(wk["home_qb_id"] == wk["home_prev_qb"], 0.0, rate["home_qb_id"] - rate["home_prev_qb"]),
            "away_qb_delta": np.where(wk["away_qb_id"] == wk["away_prev_qb"], 0.0, rate["away_qb_id"] - rate["away_prev_qb"]),
            "rest_diff": (wk["home_rest"] - wk["away_rest"]).astype(float),
            "qb_changed": ((wk["home_qb_id"] != wk["home_prev_qb"]) | (wk["away_qb_id"] != wk["away_prev_qb"])).astype(int),
        }))
    feats = pd.concat(out, ignore_index=True)
    return feats.fillna({"home_qb_delta": 0.0, "away_qb_delta": 0.0, "rest_diff": 0.0})


def _model(c):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))


def predict(rows, columns, c, seasons):
    out = []
    for season in seasons:
        train, test = rows[rows["season"] < season], rows[rows["season"] == season]
        model = _model(c).fit(nfl_game_picks.game_feature_matrix(train, columns), train["home_won"].astype(int))
        p = model.predict_proba(nfl_game_picks.game_feature_matrix(test, columns))[:, 1]
        out.append(pd.DataFrame({"game_id": test["game_id"].values, "season": season,
                                 "y": test["home_won"].astype(int).values, "p": p}))
    return pd.concat(out, ignore_index=True).set_index("game_id")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default=grid.RAW_DIR)
    parser.add_argument("--rows-cache", default=None, help="parquet of today's feature rows (built if missing)")
    args = parser.parse_args()

    grid.SEASONS = SEASONS
    if args.rows_cache and os.path.exists(args.rows_cache):
        rows = pd.read_parquet(args.rows_cache)
    else:
        grid._init()
        rows = grid.build_rows(grid.BASELINE)
        rows = rows[rows["home_won"].notna()]
        if args.rows_cache:
            rows.to_parquet(args.rows_cache)
    schedules = grid._load("schedules")
    rows = rows.merge(qb_features(args.raw_dir, schedules), on="game_id", how="left")
    h, a = schedules["home_moneyline"], schedules["away_moneyline"]
    imp = lambda ml: np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))  # noqa: E731
    vegas = pd.Series(imp(h) / (imp(h) + imp(a)), index=schedules["game_id"])

    results, preds = [], {}
    for name, columns in VARIANTS.items():
        best = None
        for c in grid.C_GRID:
            v = predict(rows, columns, c, VALIDATION_SEASONS)
            ll = grid._ll(v["y"], v["p"]).mean()
            if best is None or ll < best[1]:
                best = (c, ll, v)
        c, val_ll, v = best
        preds[name] = pd.concat([v, predict(rows, columns, c, HOLDOUTS)])
        results.append({"variant": name, "C": c, "validation": val_ll})

    base = preds["today"]
    for r in results:
        p = preds[r["variant"]].loc[base.index]
        for label, mask in (("2025", p["season"] == 2025), ("2026", p["season"] == 2026),
                            ("2025-26", p["season"].isin(HOLDOUTS)), ("all", p["season"] >= 2018)):
            ll, lb = grid._ll(p.loc[mask, "y"], p.loc[mask, "p"]), grid._ll(base.loc[mask, "y"], base.loc[mask, "p"])
            r[label] = ll.mean()
            if label == "2025-26":
                r["holdout_p"] = ttest_rel(ll, lb).pvalue if r["variant"] != "today" else np.nan
        changed = rows.set_index("game_id").loc[p.index, "qb_changed"] == 1
        r["qb_changed_games"] = grid._ll(p.loc[changed, "y"], p.loc[changed, "p"]).mean()
        r["accuracy_all"] = ((p["p"] > .5) == (p["y"] == 1)).mean()
    table = pd.DataFrame(results).set_index("variant")
    v = vegas.reindex(base.index)
    ok = v.notna()
    pd.set_option("display.width", 220)
    print(f"{len(base)} games 2018-2026 ({ok.sum()} with a closing line)\n")
    print(table.round(4).to_string())
    print(f"\nVegas on the same games: all {grid._ll(base.loc[ok, 'y'], v[ok]).mean():.4f}, "
          f"2025-26 {grid._ll(base.loc[ok & base['season'].isin(HOLDOUTS), 'y'], v[ok & base['season'].isin(HOLDOUTS)]).mean():.4f}, "
          f"QB-changed games {grid._ll(base.loc[ok & changed, 'y'], v[ok & changed]).mean():.4f}, "
          f"accuracy {((v[ok] > .5) == (base.loc[ok, 'y'] == 1)).mean():.4f}")


if __name__ == "__main__":
    main()
