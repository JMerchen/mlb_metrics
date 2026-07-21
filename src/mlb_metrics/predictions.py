"""Daily pick logging and outcome resolution - the core of Phase B.

Nothing in the original script ever checked whether WAVE/Game_Hit_Probability
actually predicted hits. This module maintains an append-only log of
(date, player, predicted probability, realized outcome) that evaluation.py
scores: `select_picks` turns a computed hitters table into that day's
ranked, qualified picks; `append_predictions` logs them *before* the game is
played; `resolve_predictions` fills in whether the pick actually got a hit
once that date's outcome data is available.
"""

import os

import pandas as pd

from mlb_metrics import config, helpers

PREDICTION_COLUMNS = ["date", "key_mlbam", "name", "rank", "predicted_probability", "metric", "actual_hit"]


def select_picks(
    hitters: pd.DataFrame,
    date,
    top_n: int = config.BACKTEST_TOP_N,
    min_plate_appearances: int = config.BACKTEST_MIN_PLATE_APPEARANCES,
    metric: str = "Game_Hit_Probability",
) -> pd.DataFrame:
    """Rank a computed hitters table (the wave.csv-equivalent output of
    hitters.assemble_hitters) by `metric` and return the top `top_n`
    qualified picks for `date`, in PREDICTION_COLUMNS shape with `actual_hit`
    left null (unresolved)."""
    qualified = hitters[(hitters["PA_L"] + hitters["PA_R"]) >= min_plate_appearances].copy()
    picks = qualified.sort_values(metric, ascending=False).head(top_n).reset_index(drop=True)

    picks["rank"] = picks.index + 1
    picks["date"] = pd.Timestamp(date)
    picks["name"] = picks["name_first"].fillna("").astype(str) + " " + picks["name_last"].fillna("").astype(str)
    picks["predicted_probability"] = picks[metric]
    picks["metric"] = metric
    picks["actual_hit"] = pd.NA

    return picks[PREDICTION_COLUMNS]


def append_predictions(picks: pd.DataFrame, log_path: str) -> pd.DataFrame:
    """Append `picks` to the predictions log at `log_path`, deduping on
    (date, key_mlbam, metric) so re-running a day's pipeline doesn't create
    duplicate log entries. Existing rows (including already-resolved
    actual_hit values) always win over a re-logged pick for the same key."""
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    if os.path.exists(log_path):
        existing = pd.read_csv(log_path, parse_dates=["date"])
        combined = pd.concat([picks, existing], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date", "key_mlbam", "metric"], keep="last")
    else:
        combined = picks

    combined = combined.sort_values(["date", "rank"]).reset_index(drop=True)
    combined.to_csv(log_path, index=False)
    return combined


def resolve_predictions(log_path: str, completed_events_by_date: pd.DataFrame) -> pd.DataFrame:
    """Fill in `actual_hit` for any still-pending rows in the predictions log
    whose (date, key_mlbam) appears in `completed_events_by_date` (columns:
    game_date, batter, events - e.g. the persisted raw Statcast data).
    Already-resolved rows are left untouched."""
    if not os.path.exists(log_path):
        return pd.DataFrame(columns=PREDICTION_COLUMNS)

    log = pd.read_csv(log_path, parse_dates=["date"])
    if log.empty:
        return log

    events = completed_events_by_date.copy()
    events["had_hit"] = helpers.is_hit(events["events"])
    game_hit = (
        events.groupby(["game_date", "batter"], as_index=False)["had_hit"]
        .max()
        .rename(columns={"game_date": "date", "batter": "key_mlbam", "had_hit": "resolved_hit"})
    )

    log = log.merge(game_hit, on=["date", "key_mlbam"], how="left")
    still_pending = log["actual_hit"].isna()
    log.loc[still_pending, "actual_hit"] = log.loc[still_pending, "resolved_hit"]
    log = log.drop(columns=["resolved_hit"])

    log.to_csv(log_path, index=False)
    return log
