"""Trains and validates the v4 NFL win model: a ridge regression of the
final margin on Elo, pass and rush EPA per play, and each starter's QB+
(nfl_game_picks.ELO_FEATURE_COLUMNS, ml_models.MarginWinModel).

User request (2026-11-10): "look at the nfl models and see which ones you
can improve either by bringing in new data not currently taken into
account or hyper parameters or honestly any change you can think of."

Every week of 2017-2026 is replayed with only what was known before it
(nfl_game_picks_backtest.build_multi_season_history - Elo from every
earlier season on file), and both models are scored walk-forward on
2019-2026, refit each season on the earlier ones:
    live (v3)   logistic regression (C = 3) on nfl_game_picks.QB_PLUS_FEATURE_COLUMNS
    v4          the margin model on ELO_FEATURE_COLUMNS
Reported: log loss, Brier and accuracy next to Vegas's closing moneyline
on the same games, a paired bootstrap interval for v4 minus v3, and how
many seasons v4 was better in. v4 is then fit on every replayed game and
saved to config.NFL_GAME_PICK_WIN_PROBABILITY_MODEL_PATH if it beat v3.

What was tried on the way (walk-forward 2019-2026, 1,972 games; change in
log loss against v3's 0.6383; Vegas 0.6112):
    Elo added to v3's inputs                          -0.0045
    a margin-of-victory rating (SRS) instead          -0.0031
    pass and rush EPA split out                       -0.0032
    fitting the margin instead of win/loss            -0.0010
    Elo + QB+ + pass/rush, margin model (v4)          -0.0066, 7 of 8 seasons
    v4 plus success rate / points per drive /
      turnovers / Pythagorean / pace / rest /
      division game / explosive plays / EPA without
      turnovers                                       within 0.0006 of v4
    v4 with opponent-adjusted pass/rush EPA (every
      offense and defense solved together, later
      games included)                                 +0.0012 to +0.0016 worse
    v4 without QB+                                    +0.0052 worse
    Elo K 15-30, keeping 1/2-3/4 between seasons,
      home edge 30-65, no margin scaling              within 0.0006 of v4
    garbage-time-free EPA, early-down EPA, the
      points model's ratings                          no help

Usage:
    PYTHONPATH=src python3 scripts/train_nfl_elo_margin_model.py [--no-save]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import config, ml_models, nfl_game_picks, nfl_game_picks_backtest as backtest  # noqa: E402

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "nfl")
SEASONS = list(range(2016, 2027))
TEST_SEASONS = list(range(2019, 2027))
TABLES = ("schedules", "team_stats", "weekly", "snap_counts", "rosters_weekly", "pbp")


def _load(table: str) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for season in SEASONS
              if os.path.exists(path := os.path.join(RAW_DIR, f"{table}_{season}.parquet"))]
    return pd.concat(frames, ignore_index=True)


def game_rows(snapshots: list, schedules: pd.DataFrame) -> pd.DataFrame:
    """One row per replayed game: both models' features, the margin, who
    won and the market's no-vig home win chance."""
    live = backtest.assemble_nfl_game_pick_log(snapshots, nfl_game_picks.build_game_features_qb_plus)
    new = backtest.assemble_nfl_game_pick_log(snapshots, nfl_game_picks.build_game_features_elo)
    rows = live.merge(new[["game_id", *[c for c in nfl_game_picks.ELO_FEATURE_COLUMNS if c not in live.columns]]], on="game_id")
    rows = rows.merge(schedules[["game_id", "home_score", "away_score", "home_moneyline", "away_moneyline"]], on="game_id")
    rows = rows[rows["home_score"] != rows["away_score"]].copy()
    rows["margin"] = rows["home_score"] - rows["away_score"]

    def implied(moneyline):
        moneyline = pd.to_numeric(moneyline, errors="coerce")
        return np.where(moneyline < 0, -moneyline / (-moneyline + 100), 100 / (moneyline + 100))

    home, away = implied(rows["home_moneyline"]), implied(rows["away_moneyline"])
    rows["market"] = home / (home + away)
    return rows


def walk_forward(rows: pd.DataFrame) -> pd.DataFrame:
    live_cols, new_cols = nfl_game_picks.QB_PLUS_FEATURE_COLUMNS, nfl_game_picks.ELO_FEATURE_COLUMNS
    out = []
    for season in TEST_SEASONS:
        train, test = rows[rows["season"] < season], rows[rows["season"] == season]
        if test.empty or train.empty:
            continue
        live = LogisticRegression(C=3.0, max_iter=2000).fit(train[live_cols].fillna(1.0), train["home_won"])
        new = ml_models.MarginWinModel(alpha=config.NFL_MARGIN_MODEL_ALPHA).fit(
            nfl_game_picks.game_feature_matrix(train, new_cols), train["margin"])
        out.append(test[["game_id", "season", "home_won", "market"]].assign(
            live=live.predict_proba(test[live_cols].fillna(1.0))[:, 1],
            new=new.predict_proba(nfl_game_picks.game_feature_matrix(test, new_cols))[:, 1]))
    return pd.concat(out, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    tables = {t: _load(t) for t in TABLES}
    snapshots = backtest.build_multi_season_history(
        tables["schedules"], tables["team_stats"], tables["weekly"], tables["snap_counts"],
        tables["rosters_weekly"], tables["pbp"], SEASONS)
    rows = game_rows(snapshots, tables["schedules"])
    print(f"{len(snapshots)} replayed weeks, {len(rows)} games {rows['season'].min()}-{rows['season'].max()}")
    missing = rows[nfl_game_picks.ELO_FEATURE_COLUMNS].isna().any(axis=1).sum()
    print(f"rows with a missing v4 input: {missing}")

    scored = walk_forward(rows)
    y = scored["home_won"].to_numpy()
    print(f"\nWalk-forward {TEST_SEASONS[0]}-{TEST_SEASONS[-1]}, {len(scored)} games")
    for name in ("live", "new", "market"):
        p = scored[name].clip(0.01, 0.99).to_numpy()
        print(f"  {name:7s} log loss {log_loss(y, p):.4f}  Brier {brier_score_loss(y, p):.4f}  right {((p > 0.5) == (y == 1)).mean():.3f}")
    point = lambda p: -(y * np.log(p) + (1 - y) * np.log(1 - p))
    diff = point(scored["new"].clip(0.01, 0.99).to_numpy()) - point(scored["live"].clip(0.01, 0.99).to_numpy())
    rng = np.random.default_rng(0)
    boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
    by_season = scored.groupby("season").apply(lambda g: pd.Series({
        "live": log_loss(g["home_won"], g["live"]), "new": log_loss(g["home_won"], g["new"]),
        "market": log_loss(g["home_won"], g["market"]), "games": len(g)}), include_groups=False)
    better = int((by_season["new"] < by_season["live"]).sum())
    print(f"  v4 - v3: {diff.mean():+.4f} (95% CI {np.percentile(boots, 2.5):+.4f} to {np.percentile(boots, 97.5):+.4f}),"
          f" better in {better} of {len(by_season)} seasons")
    print(by_season.round(4).to_string())

    final = ml_models.MarginWinModel(alpha=config.NFL_MARGIN_MODEL_ALPHA).fit(
        nfl_game_picks.game_feature_matrix(rows, nfl_game_picks.ELO_FEATURE_COLUMNS), rows["margin"])
    ridge = final.pipeline_.named_steps["ridge"]
    print(f"\nFit on all {len(rows)} games: sd of the miss {final.sigma_:.2f} points; points per standard deviation of each input:")
    print({c: round(float(b), 2) for c, b in zip(nfl_game_picks.ELO_FEATURE_COLUMNS, ridge.coef_)},
          "home edge", round(float(ridge.intercept_), 2))
    if args.no_save:
        return
    if diff.mean() >= 0:
        print("v4 did not beat v3 - not saved.")
        return
    ml_models.save_model({"model": final, "feature_columns": nfl_game_picks.ELO_FEATURE_COLUMNS},
                         config.NFL_GAME_PICK_WIN_PROBABILITY_MODEL_PATH)
    print(f"-> saved to {config.NFL_GAME_PICK_WIN_PROBABILITY_MODEL_PATH}")


if __name__ == "__main__":
    main()
