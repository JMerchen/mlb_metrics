"""Trains and validates the v4 NFL win model: the plain average
(ml_models.BlendedWinModel) of v3's logistic regression and a ridge
regression of the final margin on Elo, pass and rush EPA per play, and
each starter's QB+ (nfl_game_picks.ELO_FEATURE_COLUMNS, ml_models.MarginWinModel).

User request (2026-11-10): "look at the nfl models and see which ones you
can improve either by bringing in new data not currently taken into
account or hyper parameters or honestly any change you can think of."

Every week of 2017-2026 is replayed with only what was known before it
(nfl_game_picks_backtest.build_multi_season_history - Elo from every
earlier season on file), and the models are scored walk-forward on
2019-2026, refit each season on the earlier ones:
    live (v3)   logistic regression (C = 3) on nfl_game_picks.QB_PLUS_FEATURE_COLUMNS
    margin      the margin model on ELO_FEATURE_COLUMNS
    new (v4)    the two averaged
Reported: log loss, Brier and accuracy next to Vegas's closing moneyline
on the same games, a paired bootstrap interval for each against v3, and how
many seasons it was better in. Both are then fit on every replayed game
and the average saved to config.NFL_GAME_PICK_WIN_PROBABILITY_MODEL_PATH
if it beat v3, with the site's track-record numbers
(config.NFL_PICK_CONFIDENCE_BACKTEST_PATH).

RESULT (2026-11-10; 1,999 games, Vegas 0.6094):
    v3                    log loss 0.6362, right 64.5%
    margin model alone    0.6310 (-0.0052, 95% CI -0.0117 to +0.0015), 6 of 8 seasons, right 65.2%
    v4, the two averaged  0.6307 (-0.0054, 95% CI -0.0090 to -0.0021), 7 of 8 seasons, right 64.2%
  The average is saved: the same gain, steadier (2026 so far is the one
  season it trails v3, 0.6222 to 0.6205 over 134 games).
  How often v4's picks were right by how sure it was: 50-55% 48%,
  55-60% 58%, 60-65% 67%, 65-70% 72%, 70-75% 71%, 75%+ 79%. With Vegas's
  favorite 68% of 1,739; against it 43% of 260.

What was tried on the way (an earlier replay of the same seasons, 1,972
games; change in log loss against v3):
    Elo added to v3's inputs                          -0.0045
    a margin-of-victory rating (SRS) instead          -0.0031
    pass and rush EPA split out                       -0.0032
    fitting the margin instead of win/loss            -0.0010
    Elo + QB+ + pass/rush, margin model               -0.0058, 6 of 8 seasons
    that averaged with v3 (v4)                        -0.0057, 8 of 8 seasons
    weighted 70/30 toward the margin model            -0.0064, 7 of 8 seasons
    margin model plus success rate / points per
      drive / turnovers / Pythagorean / pace / rest /
      division game / explosive plays / EPA without
      turnovers                                       within 0.0006 of it
    with opponent-adjusted pass/rush EPA (every
      offense and defense solved together, later
      games included)                                 +0.0012 to +0.0016 worse
    without QB+                                       +0.0052 worse
    Elo K 15-30, keeping 1/2-3/4 between seasons,
      home edge 30-65, no margin scaling              within 0.0006
    garbage-time-free EPA, early-down EPA, the
      points model's ratings                          no help
  Against the spread (the same replay, closing lines): no margin tried -
  this model's, v3's, the points model's, a blend - beat it. Our side won
  49-51% of ~1,900 games at every gap from 0 to 6 points, and none added
  to the spread in a regression of the real margin on both (p 0.5-0.98).
  So the site shows our margin next to the spread as context, not a lean
  (config.NFL_WIN_MARGIN_SIGMA).

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

from mlb_metrics import config, ml_models, nfl_game_evaluation, nfl_game_picks, nfl_game_picks_backtest as backtest  # noqa: E402

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
            margin=new.predict_proba(nfl_game_picks.game_feature_matrix(test, new_cols))[:, 1]))
    scored = pd.concat(out, ignore_index=True)
    scored["new"] = (scored["live"] + scored["margin"]) / 2
    return scored


def write_confidence_backtest(scored: pd.DataFrame) -> pd.DataFrame:
    """The site's track-record numbers from the walk-forward picks, as the
    live pipeline makes them: the model's chance, replaced by Vegas's where
    the two are config.NFL_GAME_PICK_MARKET_DISAGREEMENT_THRESHOLD or more
    apart (nfl_game_picks.apply_market_tiebreak)."""
    far = (scored["new"] - scored["market"]).abs() >= config.NFL_GAME_PICK_MARKET_DISAGREEMENT_THRESHOLD
    shown = scored["new"].where(~far, scored["market"])
    counts = nfl_game_evaluation.confidence_counts(shown, scored["home_won"], scored["market"])
    path = os.path.join(os.path.dirname(__file__), "..", config.NFL_PICK_CONFIDENCE_BACKTEST_PATH)
    counts.to_csv(path, index=False)
    print(counts.assign(rate=(counts["right"] / counts["games"]).round(3)).to_string(index=False))
    return counts


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
    for name in ("live", "margin", "new", "market"):
        p = scored[name].clip(0.01, 0.99).to_numpy()
        print(f"  {name:7s} log loss {log_loss(y, p):.4f}  Brier {brier_score_loss(y, p):.4f}  right {((p > 0.5) == (y == 1)).mean():.3f}")
    point = lambda p: -(y * np.log(p) + (1 - y) * np.log(1 - p))
    by_season = scored.groupby("season").apply(lambda g: pd.Series({
        **{name: log_loss(g["home_won"], g[name].clip(0.01, 0.99)) for name in ("live", "margin", "new", "market")},
        "games": len(g)}), include_groups=False)
    rng = np.random.default_rng(0)
    diffs = {}
    for name, label in (("margin", "margin model alone"), ("new", "v4 (the two averaged)")):
        diff = point(scored[name].clip(0.01, 0.99).to_numpy()) - point(scored["live"].clip(0.01, 0.99).to_numpy())
        boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
        better = int((by_season[name] < by_season["live"]).sum())
        print(f"  {label} - v3: {diff.mean():+.4f} (95% CI {np.percentile(boots, 2.5):+.4f} to"
              f" {np.percentile(boots, 97.5):+.4f}), better in {better} of {len(by_season)} seasons")
        diffs[name] = diff
    diff = diffs["new"]
    print(by_season.round(4).to_string())
    # Picks by how sure the model was (the site's track-record table).
    sure = np.maximum(scored["new"], 1 - scored["new"])
    right = (scored["new"] > 0.5) == (scored["home_won"] == 1)
    bands = pd.cut(sure, [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 1.0], include_lowest=True)
    print(pd.DataFrame({"games": right.groupby(bands, observed=True).size(), "right": right.groupby(bands, observed=True).mean().round(3),
                        "said": sure.groupby(bands, observed=True).mean().round(3)}).to_string())
    scored.to_csv(os.path.join(os.path.dirname(__file__), "..", "data", "backtest", "nfl_win_model_v4_walk_forward.csv"), index=False)
    if not args.no_save and diff.mean() < 0:
        write_confidence_backtest(scored)

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
    live_cols = nfl_game_picks.QB_PLUS_FEATURE_COLUMNS
    logistic = LogisticRegression(C=3.0, max_iter=2000).fit(rows[live_cols].fillna(1.0), rows["home_won"])
    blend = ml_models.BlendedWinModel([(logistic, live_cols), (final, nfl_game_picks.ELO_FEATURE_COLUMNS)])
    ml_models.save_model({"model": blend, "feature_columns": nfl_game_picks.BLEND_FEATURE_COLUMNS},
                         config.NFL_GAME_PICK_WIN_PROBABILITY_MODEL_PATH)
    print(f"-> saved to {config.NFL_GAME_PICK_WIN_PROBABILITY_MODEL_PATH}")


if __name__ == "__main__":
    main()
