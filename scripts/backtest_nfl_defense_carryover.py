"""Backtest: should last season count more in an NFL team's defensive
rating?

Prompted by the 2026 Cowboys (2026-11-01: "maybe less of a pull towards
average if more extreme? We need to be open to having units that might be
historically bad or historically good"). Measured first, 2017-25, splitting
each season at weeks 4-10: an extreme unit keeps LESS of its gap from
average for the rest of the season than a middling one (defense 2+ sd: 20%
against 25% overall), so a lighter pull for extremes is ruled out - but a
defense extreme last season AND this season keeps 41%, and the rest of a
season's defense is best predicted by 0.22 x this season so far + 0.18 x
last season, with no extra term for extremes. The live rating
(nfl_team_strength._recency_window_blend: 50% last 3 games, 30% last 7,
20% every game in the data, last season included) gives last season about
0.16 of this season's weight at week 8.

This rebuilds the live model's features week by week with no lookahead
(nfl_game_picks_backtest.build_multi_season_history, as
scripts/train_nfl_game_pick_model.py does), replaces only defensive_edge
with each candidate blend of the EPA each defense allowed, refits the live
model's logistic regression (C as live) on the seasons before each test
season, and compares 2019-2026 predictions with the live blend's:

    live          the live recency blend over last season and this one
    prior_k       this season's recency blend, plus last season's average
                  counted as k extra games: (n x this + k x last) / (n + k)
    mean_k        the same with this season's plain average (no recency)

Usage:
    python scripts/backtest_nfl_defense_carryover.py
"""

import importlib.util
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import config, ml_models, nfl_game_picks, nfl_team_strength  # noqa: E402
from mlb_metrics import nfl_game_picks_backtest as backtest  # noqa: E402

SEASONS = list(range(2016, 2027))
TEST_SEASONS = list(range(2019, 2027))
CACHE = os.path.join("data", "backtest", "nfl_defense_carryover_snapshots.pkl")
VARIANTS = {"live": None, **{f"prior_{k}": ("recency", k) for k in (4, 8, 12, 17)},
            **{f"mean_{k}": ("mean", k) for k in (8, 12)}}


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _blend(kind, k):
    """A replacement for _season_aware_blend on the defense series only."""
    live = nfl_team_strength._season_aware_blend

    def blend(stats_df, group_col, value_col, current_season, **kwargs):
        if value_col != "epa_allowed_adjusted":
            return live(stats_df, group_col, value_col, current_season, **kwargs)
        current = stats_df[stats_df["season"] == current_season]
        last = stats_df[stats_df["season"] == current_season - 1].groupby(group_col)[value_col].mean()
        if kind == "recency":
            now = nfl_team_strength._recency_window_blend(current, group_col, value_col, config.NFL_TEAM_STRENGTH_WINDOWS)
        else:
            now = current.groupby(group_col)[value_col].mean()
        n = current.groupby(group_col)[value_col].size()
        teams = now.index.union(last.index)
        now, n = now.reindex(teams), n.reindex(teams).fillna(0)
        last = last.reindex(teams).fillna(last.mean() if len(last) else 0.0)
        out = (n * now.fillna(0) + k * last) / (n + k)
        out.index.name = group_col
        return out

    return blend


def _snapshots():
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    train = _load_script("train_nfl_game_pick_model")
    train.SEASONS = SEASONS
    tables = {t: train._load_all(t) for t in ("schedules", "team_stats", "weekly", "snap_counts", "rosters_weekly", "pbp")}
    snaps = backtest.build_multi_season_history(
        tables["schedules"], tables["team_stats"], tables["weekly"], tables["snap_counts"],
        tables["rosters_weekly"], tables["pbp"], SEASONS,
    )
    stats = tables["team_stats"]
    stats = stats[stats["season_type"] == "REG"]
    for snap in snaps:
        g = snap["this_week_games"].iloc[0]
        season, week = int(g["season"]), int(g["week"])
        snap["history_stats"] = stats[(stats["season"] == season - 1) | ((stats["season"] == season) & (stats["week"] < week))]
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump(snaps, f)
    return snaps


def feature_rows(snaps, variant):
    rows = []
    for snap in snaps:
        master = snap["master"].copy()
        if variant is not None:
            season = int(snap["this_week_games"].iloc[0]["season"])
            live = nfl_team_strength._season_aware_blend
            nfl_team_strength._season_aware_blend = _blend(*variant)
            try:
                edge = nfl_team_strength.compute_team_offense_defense_edge(snap["history_stats"], current_season=season)
            finally:
                nfl_team_strength._season_aware_blend = live
            d = master[["team"]].merge(edge[["team", "defensive_edge"]], on="team", how="left")["defensive_edge"]
            master["defensive_edge"] = (1 + (d - d.mean()) / d.std() * config.NFL_NORMALIZATION_Z_SCALE).to_numpy()
        features = nfl_game_picks.build_game_features_qb_plus(master, snap["qb_continuity"], snap["weekly"], snap["this_week_games"])
        outcome = snap["this_week_games"][["game_id", "home_score", "away_score"]]
        f = features.merge(outcome, on="game_id")
        f = f[f["home_score"] != f["away_score"]]
        f["home_won"] = (f["home_score"] > f["away_score"]).astype(float)
        rows.append(f)
    return pd.concat(rows, ignore_index=True)


def walk_forward(rows, c):
    cols = nfl_game_picks.QB_PLUS_FEATURE_COLUMNS
    preds = []
    for season in TEST_SEASONS:
        train, test = rows[rows["season"] < season], rows[rows["season"] == season]
        if test.empty:
            continue
        model = LogisticRegression(C=c, max_iter=1000).fit(nfl_game_picks.game_feature_matrix(train, cols), train["home_won"])
        preds.append(test[["game_id", "season", "home_won"]].assign(
            p=model.predict_proba(nfl_game_picks.game_feature_matrix(test, cols))[:, 1]))
    return pd.concat(preds, ignore_index=True)


def main():
    snaps = _snapshots()
    print(f"{len(snaps)} replayed weeks")
    artifact = ml_models.load_model(config.NFL_GAME_PICK_WIN_PROBABILITY_MODEL_PATH)
    c = artifact["model"].get_params()["C"] if artifact else 1.0
    results = {name: walk_forward(feature_rows(snaps, v), c) for name, v in VARIANTS.items()}
    base = results["live"].set_index("game_id")
    rng = np.random.default_rng(0)
    print(f"\nWalk-forward {TEST_SEASONS[0]}-{TEST_SEASONS[-1]} (logistic regression, C={c}), {len(base)} games")
    for name, r in results.items():
        r = r.set_index("game_id").loc[base.index]
        y, p, b = r["home_won"].to_numpy(), r["p"].to_numpy(), base["p"].to_numpy()
        diffs = []
        for _ in range(1000):
            i = rng.integers(0, len(y), len(y))
            diffs.append(log_loss(y[i], p[i]) - log_loss(y[i], b[i]))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        seasons_better = sum(log_loss(g["home_won"], g["p"]) < log_loss(g["home_won"], base.loc[g.index, "p"])
                             for _, g in r.groupby("season"))
        print(f"  {name:9s} log loss {log_loss(y, p):.5f} (vs live {log_loss(y, p) - log_loss(y, b):+.5f}, 95% CI {lo:+.5f} to {hi:+.5f})"
              f"  Brier {brier_score_loss(y, p):.5f}  better in {seasons_better}/{r['season'].nunique()} seasons")


if __name__ == "__main__":
    main()
