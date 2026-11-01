"""Backtest: does the NFL win model need a better measure of defense?

User (2026-11-01): "Dallas also hasn't kept a team under 20 since week 11
last year... our model does not capture just how bad they are." The live
model (nfl_game_picks.QB_PLUS_FEATURE_COLUMNS) sees defense only through
defensive_edge - EPA allowed per GAME, so pace and possessions mix in -
and through the points-based Pythagorean ratings; points per drive enters
for each OFFENSE only. Candidates, each z-scored across teams as the live
features are and recency-blended the live way (nfl_team_strength
._recency_window_blend over last season and this one):

    live            the live features
    def_ppd         + points allowed per drive, each side
    def_epa_play    defensive_edge as EPA allowed per PLAY
    both            the two together
    off_def_epa_play     offensive_edge and defensive_edge both per PLAY
    off_def_epa_play_ppd that, plus points allowed per drive
    pp_opp_w             per play, each game net of w x how far the
                         opponent's offense (for a defense) or defense
                         (for an offense) had been from the league
                         before that game

Walk-forward 2019-2026 with the live logistic regression (C as live),
refit on earlier seasons; reports log loss, Brier, a paired bootstrap
interval against live, seasons better, and the 474 games with a defense
1.5+ sd from average.

Usage:
    python scripts/backtest_nfl_defense_features.py
"""

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import config, ml_models, nfl_game_picks, nfl_team_strength  # noqa: E402

spec = importlib.util.spec_from_file_location("carry", os.path.join(os.path.dirname(__file__), "backtest_nfl_defense_carryover.py"))
carry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(carry)

PBP_COLUMNS = ["game_id", "play_id", "season", "week", "season_type", "posteam", "defteam", "fixed_drive",
               "posteam_score", "posteam_score_post", "epa", "play_type", "qb_kneel", "qb_spike"]


def defense_games() -> pd.DataFrame:
    """One row per (game, defense): points and drives allowed, EPA and plays allowed."""
    frames = []
    for season in carry.SEASONS:
        p = pd.read_parquet(os.path.join("data", "raw", "nfl", f"pbp_{season}.parquet"), columns=PBP_COLUMNS)
        p = p[(p["season_type"] == "REG") & p["posteam"].notna() & p["defteam"].notna()].sort_values(["game_id", "play_id"])
        drives = p[p["fixed_drive"].notna()].groupby(["game_id", "season", "week", "defteam", "fixed_drive"], as_index=False).agg(
            start=("posteam_score", "first"), end=("posteam_score_post", "last"))
        drives["points"] = drives["end"] - drives["start"]
        ppd = drives.groupby(["game_id", "season", "week", "defteam"], as_index=False).agg(
            points=("points", "sum"), drives=("fixed_drive", "nunique"))
        plays = p[p["play_type"].isin(["run", "pass"]) & (p["qb_kneel"] != 1) & (p["qb_spike"] != 1)].dropna(subset=["epa"])
        epa = plays.groupby(["game_id", "season", "week", "defteam"], as_index=False).agg(epa=("epa", "sum"), plays=("epa", "size"))
        own = plays.groupby(["game_id", "season", "week", "posteam"], as_index=False).agg(off_epa=("epa", "sum"), off_plays=("epa", "size"))
        own = own.rename(columns={"posteam": "defteam"})  # keyed by team, like the rest
        frames.append(ppd.merge(epa, on=["game_id", "season", "week", "defteam"], how="outer")
                      .merge(own, on=["game_id", "season", "week", "defteam"], how="outer"))
    g = pd.concat(frames, ignore_index=True).rename(columns={"defteam": "team"})
    g["ppd_allowed"] = g["points"] / g["drives"]
    g["epa_play_allowed"] = g["epa"] / g["plays"]
    g["epa_play"] = g["off_epa"] / g["off_plays"]
    # Each game's opponent, and how good its offense and defense had been
    # before that game (last season plus this one's earlier weeks), against
    # the league: what an average team would have done that day.
    opp = g[["game_id", "team"]].merge(g[["game_id", "team"]], on="game_id", suffixes=("", "_opp"))
    g = g.merge(opp[opp["team"] != opp["team_opp"]].rename(columns={"team_opp": "opp"}), on=["game_id", "team"], how="left")
    base = []
    for (season, week), _ in g.groupby(["season", "week"]):
        prior = g[(g["season"] == season - 1) | ((g["season"] == season) & (g["week"] < week))]
        if prior.empty:
            continue
        lg_off, lg_def = prior["epa_play"].mean(), prior["epa_play_allowed"].mean()
        means = prior.groupby("team")[["epa_play", "epa_play_allowed"]].mean()
        base.append(pd.DataFrame({"season": season, "week": week, "opp": means.index,
                                  "opp_off_gap": means["epa_play"].to_numpy() - lg_off,
                                  "opp_def_gap": means["epa_play_allowed"].to_numpy() - lg_def}))
    g = g.merge(pd.concat(base, ignore_index=True), on=["season", "week", "opp"], how="left")
    g[["opp_off_gap", "opp_def_gap"]] = g[["opp_off_gap", "opp_def_gap"]].fillna(0.0)
    return g


VARIANTS = ("live", "def_ppd", "def_epa_play", "both", "off_def_epa_play", "off_def_epa_play_ppd",
            "pp_opp_0.5", "pp_opp_1.0")


def _z(series: pd.Series) -> pd.Series:
    return 1 + (series - series.mean()) / series.std() * config.NFL_NORMALIZATION_Z_SCALE


def rows_for(snaps, games, variant):
    out = []
    for snap in snaps:
        g0 = snap["this_week_games"].iloc[0]
        season, week = int(g0["season"]), int(g0["week"])
        hist = games[(games["season"] == season - 1) | ((games["season"] == season) & (games["week"] < week))]
        master = snap["master"].copy()
        extra = []
        if variant in ("def_ppd", "both", "off_def_epa_play_ppd"):
            v = nfl_team_strength._recency_window_blend(hist, "team", "ppd_allowed", config.NFL_TEAM_STRENGTH_WINDOWS)
            master["ppd_allowed"] = _z(master["team"].map(v)).to_numpy()
            extra.append("ppd_allowed")
        if variant in ("def_epa_play", "both", "off_def_epa_play", "off_def_epa_play_ppd"):
            v = nfl_team_strength._recency_window_blend(hist, "team", "epa_play_allowed", config.NFL_TEAM_STRENGTH_WINDOWS)
            master["defensive_edge"] = _z(master["team"].map(v)).to_numpy()
        if variant in ("off_def_epa_play", "off_def_epa_play_ppd"):
            v = nfl_team_strength._recency_window_blend(hist, "team", "epa_play", config.NFL_TEAM_STRENGTH_WINDOWS)
            master["offensive_edge"] = _z(master["team"].map(v)).to_numpy()
        if variant.startswith("pp_opp_"):
            w = float(variant.rsplit("_", 1)[1])
            adj = hist.assign(off_adj=hist["epa_play"] - w * hist["opp_def_gap"],
                              def_adj=hist["epa_play_allowed"] - w * hist["opp_off_gap"])
            for col, edge in (("off_adj", "offensive_edge"), ("def_adj", "defensive_edge")):
                v = nfl_team_strength._recency_window_blend(adj, "team", col, config.NFL_TEAM_STRENGTH_WINDOWS)
                master[edge] = _z(master["team"].map(v)).to_numpy()
        f = nfl_game_picks.build_game_features_qb_plus(master, snap["qb_continuity"], snap["weekly"], snap["this_week_games"])
        for col in extra:
            by_team = master.set_index("team")[col]
            f[f"home_{col}"] = f["home_team"].map(by_team).to_numpy()
            f[f"away_{col}"] = f["away_team"].map(by_team).to_numpy()
        res = snap["this_week_games"][["game_id", "home_score", "away_score"]]
        f = f.merge(res, on="game_id")
        f = f[f["home_score"] != f["away_score"]]
        f["home_won"] = (f["home_score"] > f["away_score"]).astype(float)
        out.append(f)
    rows = pd.concat(out, ignore_index=True)
    cols = list(nfl_game_picks.QB_PLUS_FEATURE_COLUMNS)
    if variant in ("def_ppd", "both", "off_def_epa_play_ppd"):
        cols += ["home_ppd_allowed", "away_ppd_allowed"]
    return rows, cols


def walk_forward(rows, cols, c):
    preds = []
    for season in carry.TEST_SEASONS:
        train, test = rows[rows["season"] < season], rows[rows["season"] == season]
        if test.empty:
            continue
        model = LogisticRegression(C=c, max_iter=1000).fit(train[cols].fillna(1.0), train["home_won"])
        preds.append(test[["game_id", "season", "home_won"]].assign(p=model.predict_proba(test[cols].fillna(1.0))[:, 1]))
    return pd.concat(preds, ignore_index=True).set_index("game_id")


def main():
    snaps = carry._snapshots()
    games = defense_games()
    artifact = ml_models.load_model(config.NFL_GAME_PICK_WIN_PROBABILITY_MODEL_PATH)
    c = artifact["model"].get_params()["C"] if artifact else 1.0
    results = {}
    live_rows = None
    for variant in VARIANTS:
        rows, cols = rows_for(snaps, games, variant)
        if variant == "live":
            live_rows = rows
        results[variant] = walk_forward(rows, cols, c)
    z = config.NFL_NORMALIZATION_Z_SCALE
    extreme = set(live_rows.loc[((live_rows["home_defensive_edge"] - 1).abs() >= 1.5 * z)
                                | ((live_rows["away_defensive_edge"] - 1).abs() >= 1.5 * z), "game_id"])
    base = results["live"]
    rng = np.random.default_rng(0)
    print(f"Walk-forward {carry.TEST_SEASONS[0]}-{carry.TEST_SEASONS[-1]}, {len(base)} games, C={c}")
    for name, r in results.items():
        r = r.loc[base.index]
        y, p, b = r["home_won"].to_numpy(), r["p"].to_numpy(), base["p"].to_numpy()
        diffs, bdiffs = [], []
        for _ in range(1000):
            i = rng.integers(0, len(y), len(y))
            diffs.append(log_loss(y[i], p[i]) - log_loss(y[i], b[i]))
            bdiffs.append(brier_score_loss(y[i], p[i]) - brier_score_loss(y[i], b[i]))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        blo, bhi = np.percentile(bdiffs, [2.5, 97.5])
        better = sum(log_loss(g["home_won"], g["p"]) < log_loss(g["home_won"], base.loc[g.index, "p"]) for _, g in r.groupby("season"))
        e = r[r.index.isin(extreme)]
        print(f"  {name:13s} log loss {log_loss(y, p):.5f} ({log_loss(y, p) - log_loss(y, b):+.5f}, 95% CI {lo:+.5f} to {hi:+.5f})"
              f"  Brier {brier_score_loss(y, p):.5f} (CI {blo:+.5f} to {bhi:+.5f})  better {better}/{r['season'].nunique()}"
              f" | extreme-D games {log_loss(e['home_won'], e['p']):.4f}")


if __name__ == "__main__":
    main()
