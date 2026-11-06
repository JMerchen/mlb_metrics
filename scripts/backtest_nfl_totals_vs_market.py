"""Backtest: predict how far the real total lands from Vegas's line,
instead of the total itself.

User request (2026-11-06): "Our goal shouldn't be to capture total, it
should be to just be better than Vegas." The points model (nfl_totals.py)
is fit to the score; a lean (config.NFL_TOTAL_LEAN_MIN_GAP) is wherever
its total happens to sit far from the line. A model of the market's miss -
target = actual total minus the closing total_line - learns directly when
the line tends to be wrong.

Walk-forward throughout. The points model is refit season by season on
earlier seasons (backtest_nfl_totals.py), so every game's predicted total
is out of sample; each market-miss model is then fit on the earlier
seasons' games and scored on the next season. Scored: 2019-2025 (2018 is
the first season with out-of-sample totals to train on).

Compared, on the same games:
    A  gap         our predicted total minus the line (the live lean)
    B  linear      OLS of (actual - line) on the gap, the weather terms,
                   roof, the line itself, the point spread's size, division
                   games and Thursday games
    C  boosted     gradient-boosted trees on the same features
Each bets the same share of games per season - its most confident 5, 10,
15, 20 or 30% by the size of its predicted edge - so win rates compare at
equal volume, plus A at its live rule (|gap| >= 5). Units use the game's
real over/under odds (nflverse's over_odds/under_odds; -110 when missing),
one unit risked per bet; pushes are refunds.

RESULT (2026-11-06, 1,871 games 2019-2025): the model of the market's
miss does NOT beat the gap. Right / bets, units at the real odds:
    share     gap             linear          boosted
    top 5%    59.6% / 94 +14  66.3% / 95 +26  52.1% / 96  +1
    top 10%   60.5% /185 +31  58.1% /186 +22  50.8% /187  -4
    top 15%   58.3% /278 +34  54.6% /280 +16  50.2% /281  -9
    top 20%   56.9% /369 +36  53.4% /371 +11  51.5% /371  -2
    top 30%   55.2% /558 +35  52.6% /561  +9  52.7% /560 +10
The live rule (|gap| >= 5): 62.5% of 184, +37.2 units, p 0.004; the
linear model at the same volume 58.2% (+22.1), boosted 50.8% (-3.8).
Linear only edges the gap at the top 5% (95 bets, within noise); the
boosted model is a coin flip. Why: the real total minus the line is mostly
noise (SD ~13) with little signal to learn from, while the points model
learns team strength from every game's scoring and the line comparison
does the rest. One thing the line still misses after the gap: rain (-4.3
points beyond our gap, p 0.005; wind -0.17 a mph, p 0.075). Kept: the
gap rule (config.NFL_TOTAL_LEAN_MIN_GAP).

Usage:
    python scripts/backtest_nfl_totals_vs_market.py
"""

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import backtest_nfl_totals as totals_backtest  # noqa: E402
from mlb_metrics import config, nfl_totals, nfl_weather  # noqa: E402

FEATURES = ["gap", "wx_wind", "wx_cold", "wx_rain", "dome", "total_line", "abs_spread", "div_game", "thursday"]
SHARES = (0.05, 0.10, 0.15, 0.20, 0.30)


def out_of_sample_games() -> pd.DataFrame:
    """Every 2018-2025 game with a closing total: the walk-forward
    predicted total, the market's line and odds, and the game's features."""
    team_games, qb_games, schedules = totals_backtest.load()
    features = totals_backtest.weekly_features(team_games, qb_games, schedules).dropna(
        subset=nfl_totals.TOTALS_FEATURE_COLUMNS + ["points"])
    scored = []
    for season in totals_backtest.SCORED_SEASONS:
        model = totals_backtest.fit(features[features["season"] < season])
        test = features[features["season"] == season].copy()
        test["predicted"] = model.predict(sm.add_constant(test[nfl_totals.TOTALS_FEATURE_COLUMNS], has_constant="add"))
        scored.append(test)
    rows = pd.concat(scored)
    games = rows.groupby("game_id").agg(
        predicted_total=("predicted", "sum"), total=("points", "sum"), sides=("points", "size"),
        wx_wind=("wx_wind", "first"), wx_cold=("wx_cold", "first"), wx_rain=("wx_rain", "first"),
        dome=("dome", "first"),
    ).reset_index()
    games = games[games["sides"] == 2]
    market = schedules[["game_id", "season", "week", "total_line", "spread_line", "div_game", "weekday",
                        "over_odds", "under_odds"]]
    games = games.merge(market, on="game_id").dropna(subset=["total_line"])
    games["gap"] = games["predicted_total"] - games["total_line"]
    games["miss"] = games["total"] - games["total_line"]
    games["abs_spread"] = games["spread_line"].abs()
    games["thursday"] = (games["weekday"] == "Thursday").astype(float)
    games["div_game"] = games["div_game"].astype(float)
    return games


def predicted_edges(games: pd.DataFrame) -> pd.DataFrame:
    """Each model's predicted edge (points the real total lands over the
    line; negative = under) for 2019-2025, fit on earlier seasons only."""
    out = []
    for season in sorted(games["season"].unique())[1:]:
        train, test = games[games["season"] < season], games[games["season"] == season].copy()
        test["edge_gap"] = test["gap"]
        linear = sm.OLS(train["miss"], sm.add_constant(train[FEATURES])).fit()
        test["edge_linear"] = linear.predict(sm.add_constant(test[FEATURES], has_constant="add"))
        boosted = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200,
                                                min_samples_leaf=40, l2_regularization=1.0, random_state=0)
        boosted.fit(train[FEATURES], train["miss"])
        test["edge_boosted"] = boosted.predict(test[FEATURES])
        out.append(test)
    return pd.concat(out)


def payout(odds: pd.Series) -> pd.Series:
    """Profit on one unit risked at American odds (-110 when missing)."""
    odds = pd.to_numeric(odds, errors="coerce").fillna(-110)
    return np.where(odds < 0, 100 / -odds, odds / 100)


def grade(bets: pd.DataFrame, edge: str) -> dict:
    side_over = bets[edge] > 0
    decided = bets[bets["miss"] != 0]
    right = ((decided[edge] > 0) == (decided["miss"] > 0))
    odds = np.where(side_over, bets["over_odds"], bets["under_odds"])
    won = (side_over == (bets["miss"] > 0)) & (bets["miss"] != 0)
    lost = (side_over != (bets["miss"] > 0)) & (bets["miss"] != 0)
    units = float((payout(pd.Series(odds, index=bets.index)) * won).sum() - lost.sum())
    n, wins = len(decided), int(right.sum())
    p = stats.binomtest(wins, n, 0.524, alternative="greater").pvalue if n else float("nan")
    return {"n": n, "right": wins / n if n else float("nan"), "units": units, "p": p}


def top_share(games: pd.DataFrame, edge: str, share: float) -> pd.DataFrame:
    """Each season's most confident `share` of games by |edge|."""
    picked = []
    for _, season in games.groupby("season"):
        k = max(1, int(round(len(season) * share)))
        picked.append(season.reindex(season[edge].abs().sort_values(ascending=False).index[:k]))
    return pd.concat(picked)


def main():
    games = predicted_edges(out_of_sample_games())
    print(f"{len(games)} games 2019-2025 with a closing total; real total minus line: mean {games['miss'].mean():+.2f}")
    for name in ("gap", "linear", "boosted"):
        corr = np.corrcoef(games[f"edge_{name}"], games["miss"])[0, 1]
        print(f"  {name:8s} edge vs real miss: r {corr:+.3f}")
    print("\nSame volume per season (top share by |edge|): right / n, units, p vs 52.4%")
    for share in SHARES:
        line = [f"  top {int(share * 100):>2}%:"]
        for name in ("gap", "linear", "boosted"):
            g = grade(top_share(games, f"edge_{name}", share), f"edge_{name}")
            line.append(f"{name} {g['right']:.3f}/{g['n']} {g['units']:+6.1f}u p{g['p']:.3f}")
        print("  ".join(line))
    live = games[games["gap"].abs() >= config.NFL_TOTAL_LEAN_MIN_GAP]
    g = grade(live, "edge_gap")
    print(f"\nlive rule |gap| >= {config.NFL_TOTAL_LEAN_MIN_GAP}: {g['right']:.3f} of {g['n']}, {g['units']:+.1f} units, p {g['p']:.3f}")
    for name in ("linear", "boosted"):
        same = top_share(games, f"edge_{name}", len(live) / len(games))
        g = grade(same, f"edge_{name}")
        print(f"  {name} at the same volume: {g['right']:.3f} of {g['n']}, {g['units']:+.1f} units, p {g['p']:.3f}")
    final = sm.OLS(games["miss"], sm.add_constant(games[FEATURES])).fit()
    print("\nlinear model fit on every scored season (coefficient, p):")
    print({k: (round(float(v), 3), round(float(final.pvalues[k]), 3)) for k, v in final.params.items()})
    for season, part in games.groupby("season"):
        row = [f"  {season}:"]
        for name in ("gap", "linear", "boosted"):
            g = grade(top_share(part, f"edge_{name}", 0.10), f"edge_{name}")
            row.append(f"{name} {g['right']:.2f}/{g['n']}")
        print(" ".join(row))


if __name__ == "__main__":
    main()
