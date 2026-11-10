"""Backtest: judge each game by how good the opponent is, measured from
ALL the opponent's other games so far - before and after the game.

User idea (2026-11-08): judge teams "not only how they played against who
they played, but by how their opponent did the week following playing
them" - the Cardinals put 33 on Dallas, then looked flat against Seattle,
which says more about Dallas's defense than the 33 alone.

An earlier test (scripts/backtest_nfl_defense_features.py) adjusted only
for the opponent's form BEFORE the game, and only in the win model. Here,
for the points model (nfl_totals.py), every game in the history window is
re-scored against the opponent as rated from its other games in that same
window (leave-one-out, so the game being judged doesn't vouch for itself),
SRS-style over two passes; the opponent's rating is shrunk toward average
by SHRINK_PLAYS pseudo-plays and the adjustment scaled by STRENGTH. Each
side's adjusted EPA then goes through the usual recency windows
(nfl_totals.wave_ratings). Walk-forward exactly as
backtest_nfl_totals.py (history = prior season + this season's earlier
weeks; the regression refit on earlier seasons).

RESULT (2026-11-10; walk-forward, 8 seasons; live MAE 10.475, Vegas 10.39):
    adjusted at half strength  MAE 10.472 (-0.003, 95% CI -0.014 to +0.006), better in 4 of 8 seasons
    adjusted at full strength  MAE 10.469 (-0.006, 95% CI -0.025 to +0.011), better in 4 of 8 seasons
  Over/under leans (5+ points from Vegas): live right 62.0% of 237, half
  61.7% of 240, full 61.3% of 235. Dallas's defense, the case that
  prompted this, is 31st of 32 every way (+0.196, +0.199, +0.200 EPA per
  play allowed). No gain worth a change: the points model is left as it is.

Usage:
    python scripts/backtest_nfl_opponent_adjusted.py
"""

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import backtest_nfl_totals as bt  # noqa: E402
from mlb_metrics import config, nfl_totals  # noqa: E402

SHRINK_PLAYS = 300.0
VARIANTS = {"live": None, "adjusted x0.5": 0.5, "adjusted x1.0": 1.0}


def adjusted_games(tg: pd.DataFrame, strength: float, passes: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(offense-adjusted, defense-adjusted) copies of team_games: each row's
    EPA re-scored against its opponent's rating from the opponent's OTHER
    games in `tg`."""
    g = tg.copy()
    league = g["epa_sum"].sum() / g["plays"].sum()
    rate = g["epa_sum"] / g["plays"] - league            # each offense's EPA per play vs the league

    def team_ratings(values, key):
        """Plays-weighted rating per team over its rows, with the row's own
        contribution removed (leave-one-out), shrunk toward 0."""
        num = (values * g["plays"]).groupby(g[key]).transform("sum") - values * g["plays"]
        den = g["plays"].groupby(g[key]).transform("sum") - g["plays"]
        return num / (den + SHRINK_PLAYS)

    off_adj, def_adj = rate.copy(), rate.copy()
    for _ in range(passes):
        # A defense's rating, seen by this row's offense, from its other
        # games (allowed EPA there, adjusted for those offenses); and an
        # offense's rating, seen by this row's defense, likewise.
        opp_def = team_ratings(def_adj, "opp")      # rows keyed by the defending team
        own_off = team_ratings(off_adj, "team")     # rows keyed by the attacking team
        off_adj = rate - strength * opp_def
        def_adj = rate - strength * own_off
    off_games = g.assign(epa_sum=(off_adj + league) * g["plays"])
    def_games = g.assign(epa_sum=(def_adj + league) * g["plays"])
    return off_games, def_games


def ratings(tg: pd.DataFrame, strength) -> pd.DataFrame:
    if strength is None or tg.empty:
        return nfl_totals.wave_ratings(tg)
    off_games, def_games = adjusted_games(tg, strength)
    off = nfl_totals.wave_ratings(off_games)[["team", "off_epa", "off_plays"]]
    dfn = nfl_totals.wave_ratings(def_games)[["team", "def_epa", "def_plays"]]
    return off.merge(dfn, on="team", how="outer")


def weekly_features(team_games, qb_games, schedules, strength) -> pd.DataFrame:
    def before(df, season, week):
        return df[(df["season"] == season - 1) | ((df["season"] == season) & (df["week"] < week))]

    frames = []
    for (season, week), games in schedules[schedules["season"] >= bt.FIRST_FIT_SEASON].groupby(["season", "week"]):
        features = nfl_totals.matchup_features(
            ratings(before(team_games, season, week), strength), games,
            before(qb_games, season, week), before(schedules, season, week),
        )
        features["season"] = season
        frames.append(features)
    features = pd.concat(frames, ignore_index=True)
    return features.merge(team_games[["game_id", "team", "points"]], on=["game_id", "team"], how="inner")


def evaluate(features: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    features = features.dropna(subset=nfl_totals.TOTALS_FEATURE_COLUMNS + ["points"])
    scored = []
    for season in bt.SCORED_SEASONS:
        model = bt.fit(features[features["season"] < season])
        test = features[features["season"] == season].copy()
        test["predicted"] = model.predict(sm.add_constant(test[nfl_totals.TOTALS_FEATURE_COLUMNS], has_constant="add"))
        scored.append(test)
    scored = pd.concat(scored)
    games = scored.groupby("game_id").agg(predicted_total=("predicted", "sum"), total=("points", "sum"),
                                          sides=("points", "size")).reset_index()
    games = games[games["sides"] == 2].merge(schedules[["game_id", "season", "total_line"]], on="game_id")
    return games.dropna(subset=["total_line"])


def main():
    team_games, qb_games, schedules = bt.load()
    results = {}
    for name, strength in VARIANTS.items():
        games = evaluate(weekly_features(team_games, qb_games, schedules, strength), schedules)
        results[name] = games.set_index("game_id")
        miss = (games["predicted_total"] - games["total"]).abs()
        both = sm.OLS(games["total"], sm.add_constant(games[["total_line", "predicted_total"]])).fit()
        lean = games[(games["predicted_total"] - games["total_line"]).abs() >= config.NFL_TOTAL_LEAN_MIN_GAP]
        lean = lean[lean["total"] != lean["total_line"]]
        right = ((lean["predicted_total"] > lean["total_line"]) == (lean["total"] > lean["total_line"])).mean()
        print(f"{name:14s} MAE {miss.mean():.3f}  adds to market: coef {both.params['predicted_total']:.2f}"
              f" p {both.pvalues['predicted_total']:.3f}  leans 5+: {right:.3f} of {len(lean)}")
    live = results["live"]
    for name in VARIANTS:
        if name == "live":
            continue
        other = results[name].reindex(live.index)
        diff = (other["predicted_total"] - other["total"]).abs() - (live["predicted_total"] - live["total"]).abs()
        rng = np.random.default_rng(0)
        boots = [diff.sample(frac=1, replace=True, random_state=int(rng.integers(1e9))).mean() for _ in range(1000)]
        by_season = diff.groupby(live["season"]).mean()
        print(f"  {name} vs live: MAE change {diff.mean():+.3f} (95% CI {np.percentile(boots, 2.5):+.3f} to "
              f"{np.percentile(boots, 97.5):+.3f}), better in {(by_season < 0).sum()} of {len(by_season)} seasons")

    # Dallas now: raw vs adjusted defense, from every game through the latest week.
    season = int(team_games["season"].max())
    week = int(team_games.loc[team_games["season"] == season, "week"].max()) + 1
    history = team_games[(team_games["season"] == season - 1) | ((team_games["season"] == season) & (team_games["week"] < week))]
    for name, strength in VARIANTS.items():
        r = ratings(history, strength).set_index("team")
        r["def_rank"] = r["def_epa"].rank(ascending=True)
        print(f"  {name:14s} DAL defense {r.loc['DAL', 'def_epa']:+.3f} EPA/play allowed, rank {int(r.loc['DAL', 'def_rank'])} of 32")


if __name__ == "__main__":
    main()
