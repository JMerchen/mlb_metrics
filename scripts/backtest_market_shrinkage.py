"""Replays the MLB moneyline bet rule on the real game-picks log with the
model's win probability pulled part of the way toward the market's, to
answer: does any amount of trust in the model's disagreements with the
market make money?

Motivation (2026-10-04): advised bets won 32.8% (40-82) at an average
odds-implied 39.6%, for -21% ROI. The model gave those bets a 48.7%
average chance, and the larger its claimed edge, the worse they did -
the signature of disagreements with the market being mostly noise.

    blended_home = w * model_home + (1 - w) * market_home (de-vigged)
    edge(side)   = blended(side) - vigged implied(side)
    bet when edge >= min_edge, flat 1 unit, paid at the vigged price

w = 1 with min_edge = KELLY_MIN_EDGE is the live rule, and the replay
checks it recovers the bets the live rule actually made. Flat stakes keep
the comparison about bet SELECTION; the live Kelly sizing and caps only
rescale what is selected.

The log stores only the bet side's moneyline, so both sides' vigged
prices are rebuilt from the de-vigged home probability p:
  - a game we bet: the bet side's implied i is known exactly, so the
    other side follows exactly from p = h / (h + a);
  - any other game: the median overround measured on the bet games is
    applied, h = p * O and a = (1 - p) * O.

This is in-sample on the same games that exposed the problem, and the
sample is small, so read it as "which direction" rather than as a tuned
setting.

RESULT (2026-10-04, 493 resolved games, median overround 1.045): no
setting makes money. The live rule (w = 1, min_edge 0.05) replays at
-21% ROI, the same in August (-20%) and September on (-22%), so it is
not one bad month. Pulling toward the market only thins the bets; every
cell with bets is between -11% and -43%, and within each weight a larger
min_edge does worse, not better. A logistic regression of home wins on
both log-odds explains why: the market is strongly predictive (coef 1.60,
p < 0.001) and the model adds nothing once the market is known (coef
-0.26, p = 0.65). The model's home probabilities are also compressed
(SD 0.050 against the market's 0.087), which rates every underdog about
4.8 points above the market - hence 95% of advised bets being underdogs.

Usage:
    python scripts/backtest_market_shrinkage.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import config

LOG_PATH = "data/predictions/game_predictions.csv"
WEIGHTS = [1.0, 0.75, 0.5, 0.25, 0.0]
MIN_EDGES = [0.02, 0.05, 0.08]


def _implied(moneyline):
    moneyline = np.asarray(moneyline, dtype=float)
    # np.where evaluates both branches; a -100 line divides by zero in the
    # branch it does not use.
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(moneyline < 0, -moneyline / (-moneyline + 100), 100 / (moneyline + 100))


def rebuild_market(log: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Resolved v1 games with a market price, carrying the model's home
    probability, the de-vigged market home probability, and both sides'
    vigged implied probabilities. Returns the frame and the median
    overround it used for games without a stored moneyline."""
    games = log[
        (log["model_version"] == "v1")
        & log["market_home_win_probability"].notna()
        & log["actual_winner"].notna()
        & (pd.to_numeric(log["game_played"], errors="coerce") == 1)
    ].copy()
    home_favored = games["predicted_winner"] == games["home_team"]
    games["model_home"] = np.where(home_favored, games["predicted_probability"], 1 - games["predicted_probability"])
    p = games["market_home_win_probability"].astype(float)

    has_line = games["bet_moneyline"].notna() & (games["bet_units"].fillna(0) > 0)
    side_implied = pd.Series(_implied(games["bet_moneyline"].fillna(100)), index=games.index)
    bet_home = games["bet_team"] == games["home_team"]
    home = np.where(bet_home, side_implied, side_implied * p / (1 - p))
    away = np.where(bet_home, side_implied * (1 - p) / p, side_implied)
    overround = float(np.median((home + away)[has_line])) if has_line.any() else 1.045

    games["home_implied"] = np.where(has_line, home, p * overround)
    games["away_implied"] = np.where(has_line, away, (1 - p) * overround)
    games["home_won"] = games["actual_winner"] == games["home_team"]
    return games, overround


def replay(games: pd.DataFrame, weight: float, min_edge: float) -> pd.DataFrame:
    """One row per bet the rule would place under (weight, min_edge)."""
    blended_home = weight * games["model_home"] + (1 - weight) * games["market_home_win_probability"]
    edge_home = blended_home - games["home_implied"]
    edge_away = (1 - blended_home) - games["away_implied"]
    bets = []
    for side, edge, implied, won in (
        ("home", edge_home, games["home_implied"], games["home_won"]),
        ("away", edge_away, games["away_implied"], ~games["home_won"]),
    ):
        chosen = edge >= min_edge
        bets.append(pd.DataFrame({
            "date": games.loc[chosen, "date"],
            "game_pk": games.loc[chosen, "game_pk"],
            "side": side,
            "implied": implied[chosen],
            "won": won[chosen],
            "profit": np.where(won[chosen], 1 / implied[chosen] - 1, -1.0),
        }))
    return pd.concat(bets, ignore_index=True)


def summarize(bets: pd.DataFrame) -> dict:
    if bets.empty:
        return {"bets": 0, "win_rate": np.nan, "odds_implied": np.nan, "roi": np.nan, "units": 0.0}
    return {
        "bets": len(bets),
        "win_rate": bets["won"].mean(),
        "odds_implied": bets["implied"].mean(),
        "roi": bets["profit"].mean(),
        "units": bets["profit"].sum(),
    }


def main() -> None:
    log = pd.read_csv(LOG_PATH)
    games, overround = rebuild_market(log)
    print(f"{len(games)} resolved games with a market price; median overround {overround:.4f}")

    live = log[(log["bet_units"].fillna(0) > 0) & log["actual_winner"].notna()]
    replayed = replay(games, 1.0, config.KELLY_MIN_EDGE)
    overlap = len(set(replayed["game_pk"]) & set(live["game_pk"]))
    print(f"Live rule check: replay picks {len(replayed)} bets, {overlap} of the {len(live)} live bets\n")

    rows = []
    for weight in WEIGHTS:
        for min_edge in MIN_EDGES:
            bets = replay(games, weight, min_edge)
            row = {"model_weight": weight, "min_edge": min_edge, **summarize(bets)}
            for half, part in (("aug", bets[bets["date"] < "2026-09-01"]), ("sep+", bets[bets["date"] >= "2026-09-01"])):
                row[f"roi_{half}"] = summarize(part)["roi"]
            rows.append(row)
    table = pd.DataFrame(rows)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 140):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
