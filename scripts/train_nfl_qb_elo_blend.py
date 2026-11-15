"""Builds and checks the v5 NFL win model: v4 averaged with the
quarterback-adjusted Elo (src/mlb_metrics/nfl_qb_elo.py, which has the
why and the method).

Nothing is fit here. The quarterback-adjusted Elo is walk-forward by
construction (every game is predicted from the ratings as they stood
before it), and v4's walk-forward predictions are the ones
scripts/train_nfl_elo_margin_model.py saved. This script:
  1. replays the Elo over every schedule and weekly-stats file on disk,
  2. scores v4, the Elo alone, their plain average (v5) and Vegas on the
     same games, by season,
  3. writes the predictions back to the walk-forward file (columns
     qb_elo and v5) and the track record's backtest counts for v5,
  4. saves the live model: ml_models.BlendedWinModel of the v4 model and
     ml_models.EloGapModel, reading nfl_game_picks.V5_FEATURE_COLUMNS.

RESULT (2026-11-15, 1,999 games 2019-2026):
                      log loss   winners
    v4                 0.6307     1,284
    QB-adjusted Elo    0.6242     1,299
    v5 (the average)   0.6242     1,306
    Vegas              0.6094     1,332
v5 beats v4 by 0.0065 (t 3.7) and in all 8 seasons - about a third of the
distance v4 was from Vegas. It still does not beat the closing line (see
scripts/research_nfl_bets.py): this makes the picks more accurate, it does
not make winners a bet.

Usage:
    PYTHONPATH=src python3 scripts/train_nfl_qb_elo_blend.py [--no-save]
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mlb_metrics import config, ml_models, nfl_game_picks, nfl_qb_elo  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
WALK_FORWARD = os.path.join(REPO, "data", "backtest", "nfl_win_model_v4_walk_forward.csv")


def log_loss(p, y):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def replay() -> pd.DataFrame:
    """[game_id, qb_elo] - the Elo's home win chance for every completed
    game on disk, as it stood before the game."""
    raw = os.path.join(REPO, config.NFL_RAW_DATA_DIR) if not os.path.isabs(config.NFL_RAW_DATA_DIR) else config.NFL_RAW_DATA_DIR
    schedules = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(raw, "schedules_20*.parquet")))])
    weekly = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(raw, "weekly_20*.parquet")))])
    _, gaps = nfl_qb_elo.rate(schedules, weekly, record=True)
    return gaps.assign(qb_elo=nfl_qb_elo.gap_to_probability(gaps["qb_elo_gap"]))[["game_id", "qb_elo"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    scored = pd.read_csv(WALK_FORWARD).drop(columns=["qb_elo", "v5"], errors="ignore").merge(replay(), on="game_id", how="left")
    scored["v5"] = (scored["new"] + scored["qb_elo"]) / 2
    games = scored[scored["home_won"].isin([0, 1]) & scored["market"].notna() & scored["v5"].notna()]
    y = games["home_won"]
    print(f"{len(games)} games {int(games['season'].min())}-{int(games['season'].max())}")
    for label, column in (("v4", "new"), ("QB-adjusted Elo", "qb_elo"), ("v5 (the average)", "v5"), ("Vegas", "market")):
        print(f"  {label:18s} log loss {log_loss(games[column], y).mean():.4f}   winners {int(((games[column] > 0.5) == (y == 1)).sum())}")
    gain = log_loss(games["new"], y) - log_loss(games["v5"], y)
    by_season = gain.groupby(games["season"]).mean()
    print(f"  v5 over v4: {gain.mean():+.4f} (t {gain.mean() / (gain.std() / np.sqrt(len(gain))):.1f}), "
          f"better in {int((by_season > 0).sum())} of {len(by_season)} seasons")
    table = games.groupby("season").apply(lambda s: pd.Series(
        {c: log_loss(s[c], s["home_won"]).mean() for c in ("new", "qb_elo", "v5", "market")}))
    print(table.round(4).T.to_string())
    if args.no_save:
        return

    scored.to_csv(WALK_FORWARD, index=False)
    import train_nfl_elo_margin_model as v4_script
    v4_script.write_confidence_backtest(scored.assign(new=scored["v5"]).dropna(subset=["new"]))

    artifact = ml_models.load_model(config.NFL_GAME_PICK_WIN_PROBABILITY_MODEL_PATH)
    if artifact is None:
        raise SystemExit("No saved win model to build on - run scripts/train_nfl_elo_margin_model.py first.")
    v4_model = artifact["model"]
    if artifact["feature_columns"] == nfl_game_picks.V5_FEATURE_COLUMNS:      # already v5: rebuild around its v4 member
        v4_model = artifact["model"].members[0][0]
    elif artifact["feature_columns"] != nfl_game_picks.BLEND_FEATURE_COLUMNS:
        raise SystemExit("The saved win model is not v4's blend.")
    model = ml_models.BlendedWinModel([
        (v4_model, nfl_game_picks.BLEND_FEATURE_COLUMNS),
        (ml_models.EloGapModel(), nfl_game_picks.QB_ELO_FEATURE_COLUMNS),
    ])
    ml_models.save_model({**artifact, "model": model, "feature_columns": nfl_game_picks.V5_FEATURE_COLUMNS},
                         config.NFL_GAME_PICK_WIN_PROBABILITY_MODEL_PATH)
    print(f"saved {config.NFL_GAME_PICK_WIN_PROBABILITY_MODEL_PATH}")


if __name__ == "__main__":
    main()
