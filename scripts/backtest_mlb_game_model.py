"""Walk-forward backtest of mlb_game_model against the old composite ratio
and the betting market.

Each calendar month from April 2022 on is predicted by a model fit only
on games before that month, with history from 2021 (2021-2024 from the
compact backfill under data/raw/game_model/). Reported three ways:
  1. Brier score and log loss on every predicted game, and on games before
     2026-08-01 (the period the recency/shrinkage settings were tuned on).
  2. On the games with logged market odds (August 2026 on, held back from
     tuning): Brier for the market, the old ratio and this model, plus a
     logistic regression of home wins on market and model log-odds - a
     model coefficient indistinguishable from zero means the model adds
     nothing the market does not already know.
  3. The bet rule replayed with this model's probabilities, flat 1 unit
     per bet at the rebuilt vigged prices (scripts/backtest_market_shrinkage.py).

Usage:
    python scripts/backtest_mlb_game_model.py
"""

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from mlb_metrics import data, mlb_game_model  # noqa: E402
from backtest_market_shrinkage import LOG_PATH, rebuild_market, replay, summarize  # noqa: E402

SEASONS = tuple(range(2021, 2027))
FIRST_PREDICTED_MONTH = "2022-04"
TUNING_CUTOFF = "2026-08-01"


def load_statcast(raw_dir: str = "data/raw") -> pd.DataFrame:
    return data.load_game_model_history(raw_dir, SEASONS)


def walk_forward(games: pd.DataFrame, pa: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Every game from FIRST_PREDICTED_MONTH on, with `p` from a model fit
    only on games before its month."""
    features = mlb_game_model.build_features(games, games, pa, params)
    features["home_win"] = games["home_win"].to_numpy()
    features["game_pk"] = games["game_pk"].to_numpy()
    features["month"] = features["game_date"].dt.to_period("M")
    predicted = []
    for month in sorted(features["month"].unique()):
        if month < pd.Period(FIRST_PREDICTED_MONTH, "M"):
            continue
        train = features[features["game_date"] < month.start_time]
        test = features[features["month"] == month]
        model = LogisticRegression(C=1.0).fit(train[mlb_game_model.FEATURE_COLUMNS], train["home_win"])
        predicted.append(test.assign(p=model.predict_proba(test[mlb_game_model.FEATURE_COLUMNS])[:, 1]))
    return pd.concat(predicted, ignore_index=True)


def _scores(y, p) -> str:
    return f"Brier {brier_score_loss(y, p):.4f}, log loss {log_loss(y, p):.4f}"


def main() -> None:
    games, pa = mlb_game_model.prepare_history(load_statcast())
    predictions = walk_forward(games, pa)
    tuning = predictions[predictions["game_date"] < TUNING_CUTOFF]
    home_rate = np.full(len(predictions), predictions["home_win"].mean())
    print(f"All predicted games (n={len(predictions)}): {_scores(predictions['home_win'], predictions['p'])}; "
          f"home-rate-only Brier {brier_score_loss(predictions['home_win'], home_rate):.4f}")
    print(f"Tuning period (n={len(tuning)}): {_scores(tuning['home_win'], tuning['p'])}")

    market, _ = rebuild_market(pd.read_csv(LOG_PATH))
    joined = market.merge(predictions[["game_pk", "p"]], on="game_pk", how="inner")
    y = joined["home_won"].astype(int)
    market_p = joined["market_home_win_probability"].astype(float)
    print(f"\nHeld-back games with market odds (n={len(joined)}):")
    print(f"  market    {_scores(y, market_p)}")
    print(f"  old ratio {_scores(y, joined['model_home'])}")
    print(f"  new model {_scores(y, joined['p'])}")

    logit = lambda x: np.log(x / (1 - x))  # noqa: E731
    fit = sm.Logit(y, sm.add_constant(pd.DataFrame({"market": logit(market_p), "model": logit(joined["p"])}))).fit(disp=0)
    print(f"  beyond the market: model coef {fit.params['model']:.2f} (p = {fit.pvalues['model']:.3f}), "
          f"market coef {fit.params['market']:.2f}")
    print(f"  spread of probabilities: model SD {joined['p'].std():.3f}, market SD {market_p.std():.3f}")

    print("\nBet rule replay with the new model (flat 1 unit):")
    rows = []
    for weight in (1.0, 0.5):
        for min_edge in (0.0, 0.02, 0.04, 0.06):
            bets = replay(joined.assign(model_home=joined["p"]), weight, min_edge)
            rows.append({"model_weight": weight, "min_edge": min_edge, **summarize(bets),
                         "underdog_share": (bets["implied"] < 0.5).mean() if len(bets) else np.nan})
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 140):
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
