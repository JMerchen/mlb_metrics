"""Should the points model chase a team on a scoring run?

Why (2026-11-27, user: "Why can't our model adjust to how bad they are?"):
through 11 games of 2026 the Cowboys' games averaged 62.5 points against
our 51 and Vegas's 52. If a team that keeps beating our total keeps doing
it, the model is missing something it could learn.

The test: the walk-forward predictions of scripts/backtest_nfl_totals.py
(each season's model fit on earlier seasons), then for every team-game two
more numbers - how many points a game the team has scored above its
predictions so far this season, and how many its opponent's defense has
allowed above its predictions - each pulled toward zero with k games. A
second model, fit on earlier seasons only, turns those into an adjustment.

RESULT (2026-11-27; 1,615 games, 2020-2025):
  MAE without 10.360; with, k=2 10.386, k=4 10.394, k=6 10.398, k=10
  10.402 (the market 10.283). Worse at every k (by 0.03-0.04, se 0.04) and
  better in only 1 to 3 of 6 seasons. It over-corrects: the 157 games it
  moved up 2+ points went from 48.4 to 51.3 and came in at 49.7 (market
  49.6); the 176 it moved down went from 44.5 to 41.7 and came in at
  43.7. Over/under leans 5+ points from the market: right 63.7% without,
  61.6% with. So the base model is a little slow on a team on a run (1.3
  points) and a learned correction is worse than none - a run is mostly
  not going to continue. NOT ADOPTED.

  Background from data/backtest/nfl_game_misses_replay.csv (2,127 games,
  2018-2025): the totals are not too compressed in general (actual total
  on ours has a slope of 0.82, on the market's 1.00); games with a top-6
  scoring offense and bottom-6 scoring defense came in at 48.5 against
  our 47.9 and the market's 49.1 (52 games); a team 10+ points a game
  over our totals through 11 games happens in 2% of team-seasons (2018
  Chiefs +14.4 the most); and teams running 6+ over the MARKET's totals
  went over it a little under half the time afterwards.

Usage:
    python scripts/backtest_nfl_totals_team_run.py
"""

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest_nfl_totals as base  # noqa: E402
from mlb_metrics import nfl_totals  # noqa: E402

FIRST_SCORED = 2020


def main():
    team_games, qb_games, schedules = base.load()
    columns = nfl_totals.TOTALS_FEATURE_COLUMNS
    features = base.weekly_features(team_games, qb_games, schedules).dropna(subset=columns + ["points"])
    rows = []
    for season in base.SCORED_SEASONS:
        model = base.fit(features[features["season"] < season])
        test = features[features["season"] == season].copy()
        test["predicted"] = model.predict(sm.add_constant(test[columns], has_constant="add"))
        rows.append(test)
    d = pd.concat(rows).merge(schedules[["game_id", "week"]], on="game_id").sort_values(["season", "week"])
    d["miss"] = d["points"] - d["predicted"]
    for k in (2, 4, 6, 10):
        for key, name in (("team", "offense_run"), ("opp", "defense_run")):
            by = d.groupby(["season", key])
            d[name] = (by["miss"].cumsum() - d["miss"]) / (by.cumcount() + k)
        scored = []
        for season in range(FIRST_SCORED, base.SCORED_SEASONS.stop):
            earlier, test = d[d["season"] < season], d[d["season"] == season].copy()
            test["adjust"] = sm.OLS(earlier["miss"], earlier[["offense_run", "defense_run"]]).fit().predict(test[["offense_run", "defense_run"]])
            scored.append(test)
        g = pd.concat(scored).groupby("game_id").agg(before=("predicted", "sum"), adjust=("adjust", "sum"), total=("points", "sum"),
                                                      sides=("points", "size"), season=("season", "first"))
        g = g[g["sides"] == 2].join(schedules.set_index("game_id")["total_line"]).dropna(subset=["total_line"])
        g["after"] = g["before"] + g["adjust"]
        gain = (g["total"] - g["before"]).abs() - (g["total"] - g["after"]).abs()
        better = int((gain.groupby(g["season"]).mean() > 0).sum())
        print(f"k={k}: {len(g)} games  MAE without {(g['total'] - g['before']).abs().mean():.3f}  with {(g['total'] - g['after']).abs().mean():.3f}  "
              f"market {(g['total'] - g['total_line']).abs().mean():.3f}  gain {gain.mean():+.3f} (se {gain.std() / np.sqrt(len(g)):.3f})  better in {better} of {g['season'].nunique()} seasons")
        if k == 4:
            for label, moved in (("up", g[g["adjust"] >= 2]), ("down", g[g["adjust"] <= -2])):
                print(f"   moved {label} 2+ points: {len(moved)} games  without {moved['before'].mean():.1f}  with {moved['after'].mean():.1f}  "
                      f"market {moved['total_line'].mean():.1f}  actual {moved['total'].mean():.1f}")
            decided = g[g["total"] != g["total_line"]]
            for column in ("before", "after"):
                lean = decided[column] - decided["total_line"]
                far = decided[lean.abs() >= 5]
                print(f"   leans 5+ from the market, {column}: {len(far)} games, right {((lean[far.index] > 0) == (far['total'] > far['total_line'])).mean():.3f}")


if __name__ == "__main__":
    main()
