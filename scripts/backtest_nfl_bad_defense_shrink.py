"""Backtest: pull bad defenses toward the league less in the points model.

User request (2026-11-08): Dallas has allowed 20+ in every game this season
("the most assured thing in sports right now") - "there has to be something
that captures just how horrible that defense is." Against the worst 3-10%
of defenses the walk-forward points model predicted 24.0 points and teams
scored 25.7, and it implied 32% of those games would stay under 20 when 25%
did.

Variants of config.NFL_TOTALS_SHRINK_FADE_BAD_DEFENSE - the shrink fade for
a defense worse than the league (smaller = less pull toward average for an
extreme one); the regression refit season by season as in
backtest_nfl_totals.py. Reports, on 2018-2025: MAE of the game total, the
leans 5+ points from the closing total, and for team-games against the
worst 3-10% and worst 3% of defenses (by rating that week) the average
predicted and real points and how often the team stayed under 20.

RESULT (2026-11-09, 2,127 games 2018-2025): no difference at all.
    fade      MAE     leans 5+        worst 3-10%: pred/real, <20 implied/real   worst 3%
    live      10.475  62.0% of 237    24.0 / 25.7, 0.32 / 0.25                 24.0 / 24.3, 0.33 / 0.32
    0.02      10.475  62.2% of 238    24.0 / 25.6, 0.33 / 0.25                 24.0 / 24.3, 0.33 / 0.32
    0.01      10.475  61.9% of 239    24.0 / 25.6, 0.33 / 0.25                 24.0 / 24.3, 0.33 / 0.32
    0.005     10.475  62.2% of 238    24.0 / 25.6, 0.33 / 0.25                 23.9 / 24.3, 0.33 / 0.32
The fade already all but removes the pull for an extreme defense (one 0.09
EPA a play from the league gets a tenth of it), so the pull isn't what
holds the prediction down. And the very worst defenses are already called
right: the worst 3% allowed 24.3 points against 24.0 predicted, and held
teams under 20 in 32% of games against 33% implied. The 3-10% band's gap
(1.6 points) doesn't carry to the extreme, so it reads as noise rather
than a pattern to fit. Kept: NFL_TOTALS_SHRINK_FADE_BAD_DEFENSE = None.

Usage:
    python scripts/backtest_nfl_bad_defense_shrink.py
"""

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import backtest_nfl_totals as bt  # noqa: E402
from mlb_metrics import config, nfl_totals  # noqa: E402

VARIANTS = [None, 0.02, 0.015, 0.01, 0.005]


def run(team_games, qb_games, schedules, fade_bad):
    config.NFL_TOTALS_SHRINK_FADE_BAD_DEFENSE = fade_bad
    features = bt.weekly_features(team_games, qb_games, schedules).dropna(
        subset=nfl_totals.TOTALS_FEATURE_COLUMNS + ["points"])
    scored = []
    for season in bt.SCORED_SEASONS:
        model = bt.fit(features[features["season"] < season])
        test = features[features["season"] == season].copy()
        test["pred"] = model.predict(sm.add_constant(test[nfl_totals.TOTALS_FEATURE_COLUMNS], has_constant="add"))
        scored.append(test)
    rows = pd.concat(scored)
    rows["week_key"] = rows["game_id"].str[:7]
    rows["def_pct"] = rows.groupby("week_key")["opp_def_epa"].rank(pct=True)
    games = rows.groupby("game_id").agg(pred=("pred", "sum"), total=("points", "sum"), n=("points", "size")).reset_index()
    games = games[games["n"] == 2].merge(schedules[["game_id", "season", "total_line"]], on="game_id").dropna()
    return rows, games


def main():
    team_games, qb_games, schedules = bt.load()
    base = None
    for fade_bad in VARIANTS:
        rows, games = run(team_games, qb_games, schedules, fade_bad)
        sd = (rows["points"] - rows["pred"]).std()
        miss = (games["pred"] - games["total"]).abs()
        lean = games[(games["pred"] - games["total_line"]).abs() >= config.NFL_TOTAL_LEAN_MIN_GAP]
        lean = lean[lean["total"] != lean["total_line"]]
        right = ((lean["pred"] > lean["total_line"]) == (lean["total"] > lean["total_line"])).mean()
        line = [f"bad-defense fade {str(fade_bad):5s}: MAE {miss.mean():.3f}, leans 5+ {right:.3f} of {len(lean)}"]
        for label, lo, hi in (("worst 3-10%", 0.90, 0.97), ("worst 3%", 0.97, 1.01)):
            b = rows[(rows["def_pct"] > lo) & (rows["def_pct"] <= hi)]
            implied = norm.cdf((19.5 - b["pred"]) / sd).mean()
            line.append(f"{label}: pred {b['pred'].mean():.1f} real {b['points'].mean():.1f}, "
                        f"under 20 implied {implied:.2f} real {(b['points'] < 20).mean():.2f}")
        print(" | ".join(line))
        if base is None:
            base = games.set_index("game_id")
        else:
            other = games.set_index("game_id").reindex(base.index)
            diff = (other["pred"] - other["total"]).abs() - (base["pred"] - base["total"]).abs()
            seasons = diff.groupby(base["season"]).mean()
            print(f"    vs live: MAE change {diff.mean():+.3f}, better in {(seasons < 0).sum()} of {len(seasons)} seasons")


if __name__ == "__main__":
    main()
