"""Grades the MLB game model as a bettor against historical closing lines.

Until now the bet rule could only be replayed on the 488 games with
odds logged by the live pipeline (August 2026 on). With 2022-2025
closing moneylines backfilled from ESPN (scripts/backfill_historical_odds.py),
this replays it on every game the walk-forward backtest predicted
(scripts/backtest_mlb_game_model.py: each month predicted by a model fit
only on earlier games).

Games are joined on (date, home team, away team). Doubleheaders, where
that key matches two games, are dropped rather than guessed at.

Reported:
  1. Brier for the market's de-vigged closing line and for the model, on
     the same games, by season.
  2. Whether the model adds information beyond the market: a logistic
     regression of home wins on both log-odds.
  3. The bet rule (bet a side when the model's probability beats the
     vigged implied probability by min_edge; flat 1 unit, paid at the
     closing price), by season and overall, at several thresholds.
  4. An honest out-of-sample check: choose the threshold on 2022-2023,
     then report 2024-2025 at that threshold only.

Closing lines are the hardest prices to beat; bets placed at morning
prices could do somewhat better or worse.

RESULT (2026-10-11): 9,211 of 9,552 predicted 2022-2025 games joined to a
closing line (96%). The market's de-vigged close beats the model every
season (Brier 0.2404 vs 0.2422 overall), and the model adds nothing
beyond it (logit coefficient 0.02, p = 0.88; the market's is 0.96). The
bet rule loses at every threshold: -3.6% ROI on 5,991 bets at a zero
threshold (95% CI -6.4% to -0.7%), about the bookmaker's margin, with
68-93% of bets on underdogs. Choosing the threshold on 2022-2023 and
applying it to 2024-2025 gives +1.5% on 133 bets (95% CI -20% to +23%):
indistinguishable from zero. The model is a reasonable forecaster with
no betting edge against closing lines.

Usage:
    python scripts/backtest_bets_historical_odds.py
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from mlb_metrics import historical_odds, market_odds, mlb_game_model  # noqa: E402
import backtest_mlb_game_model as walk  # noqa: E402

MIN_EDGES = [0.0, 0.02, 0.04, 0.06, 0.08]
TUNE_SEASONS = (2022, 2023)
TEST_SEASONS = (2024, 2025)


def load_closing_lines(raw_dir: str = "data/raw") -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(raw_dir, "market_odds", "espn_mlb_odds_*.parquet")))
    if not paths:
        raise SystemExit("No data/raw/market_odds/espn_mlb_odds_*.parquet - run the backfill first.")
    rows = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    lines = historical_odds.consensus_closing_lines(rows)
    # ESPN codes -> the model's codes (ARI/CHW via the live fixups, then
    # the model's own normalization, e.g. AZ -> ARI, ATH -> OAK).
    for side in ("home_team", "away_team"):
        lines[side] = mlb_game_model.normalize_team(lines[side].map(market_odds._apply_abbrev_fixup))
    lines["game_date"] = pd.to_datetime(lines["date"])
    return lines


def join(predictions: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    key = ["game_date", "home_team", "away_team"]
    preds = predictions[predictions.groupby(key)["game_pk"].transform("size") == 1]
    lines = lines[lines.groupby(key)["event_id"].transform("size") == 1]
    joined = preds.merge(lines, on=key, how="inner")
    joined["home_implied"] = historical_odds.implied(joined["home_ml"])
    joined["away_implied"] = historical_odds.implied(joined["away_ml"])
    joined["season"] = joined["game_date"].dt.year
    return joined


def replay(joined: pd.DataFrame, min_edge: float) -> pd.DataFrame:
    """Flat 1-unit bets wherever the model's probability for a side beats
    that side's vigged implied probability by at least min_edge."""
    bets = []
    for side, model_p, implied, won in (
        ("home", joined["p"], joined["home_implied"], joined["home_win"] == 1),
        ("away", 1 - joined["p"], joined["away_implied"], joined["home_win"] == 0),
    ):
        chosen = (model_p - implied) >= min_edge
        bets.append(pd.DataFrame({
            "season": joined.loc[chosen, "season"],
            "side": side,
            "implied": implied[chosen],
            "won": won[chosen],
            "profit": np.where(won[chosen], 1 / implied[chosen] - 1, -1.0),
        }))
    return pd.concat(bets, ignore_index=True)


def summarize(bets: pd.DataFrame) -> dict:
    if bets.empty:
        return {"bets": 0, "win_rate": np.nan, "odds_implied": np.nan, "roi": np.nan, "units": 0.0,
                "roi_ci_low": np.nan, "roi_ci_high": np.nan, "underdogs": np.nan}
    se = bets["profit"].std(ddof=1) / np.sqrt(len(bets)) if len(bets) > 1 else np.nan
    return {
        "bets": len(bets),
        "win_rate": bets["won"].mean(),
        "odds_implied": bets["implied"].mean(),
        "roi": bets["profit"].mean(),
        "units": bets["profit"].sum(),
        "roi_ci_low": bets["profit"].mean() - 1.96 * se,
        "roi_ci_high": bets["profit"].mean() + 1.96 * se,
        "underdogs": (bets["implied"] < 0.5).mean(),
    }


def main() -> None:
    games, pa = mlb_game_model.prepare_history(walk.load_statcast())
    predictions = walk.walk_forward(games, pa)
    predictions = predictions[predictions["game_date"].dt.year.isin(range(2022, 2026))]
    lines = load_closing_lines()
    joined = join(predictions, lines)
    print(f"Predicted 2022-2025 games: {len(predictions)}; with a closing line: {len(joined)} "
          f"({len(joined) / len(predictions):.1%}); books per game median {joined['n_books'].median():.0f}")

    y = joined["home_win"]
    market_p = joined["market_home_win_probability"]
    print("\nForecast accuracy on the same games (Brier, lower is better):")
    for season, g in joined.groupby("season"):
        print(f"  {season}: n={len(g)}  market {brier_score_loss(g['home_win'], g['market_home_win_probability']):.4f}"
              f"  model {brier_score_loss(g['home_win'], g['p']):.4f}")
    print(f"  all:  n={len(joined)}  market {brier_score_loss(y, market_p):.4f}  model {brier_score_loss(y, joined['p']):.4f}")

    logit = lambda x: np.log(x / (1 - x))  # noqa: E731
    fit = sm.Logit(y, sm.add_constant(pd.DataFrame({"market": logit(market_p), "model": logit(joined["p"])}))).fit(disp=0)
    print(f"\nBeyond the market: model coef {fit.params['model']:.3f} (p = {fit.pvalues['model']:.3g}), "
          f"market coef {fit.params['market']:.3f}")

    print("\nBet rule replay at closing prices (flat 1 unit):")
    rows = []
    for min_edge in MIN_EDGES:
        bets = replay(joined, min_edge)
        rows.append({"seasons": "all", "min_edge": min_edge, **summarize(bets)})
        for season in sorted(joined["season"].unique()):
            rows.append({"seasons": str(season), "min_edge": min_edge, **summarize(bets[bets["season"] == season])})
    table = pd.DataFrame(rows)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 160):
        print(table.to_string(index=False))

    tune = joined[joined["season"].isin(TUNE_SEASONS)]
    test = joined[joined["season"].isin(TEST_SEASONS)]
    tuned = max(MIN_EDGES, key=lambda e: summarize(replay(tune, e))["roi"] if len(replay(tune, e)) >= 100 else -np.inf)
    result = summarize(replay(test, tuned))
    print(f"\nOut of sample: min_edge chosen on {TUNE_SEASONS} = {tuned}; on {TEST_SEASONS}: "
          f"{result['bets']} bets, ROI {result['roi']:.3f} (95% CI {result['roi_ci_low']:.3f} to {result['roi_ci_high']:.3f}), "
          f"win rate {result['win_rate']:.3f} vs {result['odds_implied']:.3f} implied")


if __name__ == "__main__":
    main()
