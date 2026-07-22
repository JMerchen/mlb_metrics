"""Daily game-pick logging and outcome resolution - the game-level analog
of predictions.py. Kept as a separate module rather than extending
predictions.py: the entity (a game, keyed on game_pk) and its resolution
data source (final scores via schedule.fetch_game_results, not Statcast)
both differ enough from the batter-pick case that threading two entity
types through one module would hurt readability more than sharing code
would help.
"""

import os

import pandas as pd

from mlb_metrics import config

GAME_PREDICTION_COLUMNS = [
    "date", "game_pk", "home_team", "away_team", "predicted_winner",
    "predicted_probability", "metric", "actual_winner", "game_played", "model_version",
]

# Same purpose as predictions.LEGACY_MODEL_VERSION - tagged onto any row
# logged before model_version existed, or reconstructed by a historical
# replay (see game_picks_backtest.py).
LEGACY_MODEL_VERSION = "legacy"


def select_game_picks(
    win_probabilities: pd.DataFrame,
    date,
    min_probability: float = config.GAME_PICK_MIN_PROBABILITY,
    metric: str = "GamePick_Win_Probability",
    model_version: str = config.GAME_PICK_MODEL_VERSION,
) -> pd.DataFrame:
    """Turn game_picks.compute_game_win_probabilities' output into the
    day's picked games: only games where the favored side's win probability
    clears `min_probability` ("real separation between the two teams") - a
    day can have 0 or more picks, never padded to a fixed count.
    `predicted_winner` is whichever side (home_team/away_team) has the
    higher probability; `predicted_probability` is that side's probability
    (always >= 0.5). `model_version` (see config.GAME_PICK_MODEL_VERSION) is
    stamped onto every row so game_evaluation.py/the dashboard can segment
    accuracy by which win-probability logic actually produced a pick - same
    reasoning as predictions.select_picks' own model_version."""
    df = win_probabilities.copy()
    favors_home = df["home_win_probability"] >= 0.5
    df["predicted_winner"] = df["home_team"].where(favors_home, df["away_team"])
    df["predicted_probability"] = df["home_win_probability"].where(favors_home, 1 - df["home_win_probability"])

    picks = df[df["predicted_probability"] >= min_probability].copy()
    picks["date"] = pd.Timestamp(date)
    picks["metric"] = metric
    picks["actual_winner"] = pd.NA
    picks["game_played"] = pd.NA
    picks["model_version"] = model_version

    return picks[GAME_PREDICTION_COLUMNS]


def append_game_predictions(picks: pd.DataFrame, log_path: str) -> pd.DataFrame:
    """Append `picks` to the game-predictions log at `log_path`, deduping
    on (date, game_pk, metric) so a re-run doesn't create duplicate log
    entries. Existing rows (including already-resolved outcomes) always win
    over a re-logged pick for the same key."""
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    if os.path.exists(log_path):
        existing = pd.read_csv(log_path, parse_dates=["date"])
        if "model_version" not in existing.columns:
            existing["model_version"] = LEGACY_MODEL_VERSION  # migrate a log written before model_version existed
        combined = pd.concat([picks, existing], ignore_index=True)
    else:
        combined = picks

    combined = combined.drop_duplicates(subset=["date", "game_pk", "metric"], keep="last")
    combined = combined.sort_values(["date", "game_pk"]).reset_index(drop=True)
    combined.to_csv(log_path, index=False)
    return combined


def resolve_game_predictions(log_path: str, fetch_results_fn, as_of_date) -> pd.DataFrame:
    """Fill in `actual_winner`/`game_played` for any still-pending rows
    (`game_played` is null) with a date strictly before `as_of_date`, by
    calling `fetch_results_fn(date)` (i.e. schedule.fetch_game_results) once
    per distinct pending date. `as_of_date` is explicit rather than reading
    the wall clock - same no-implicit-"now" philosophy as pipeline.run()
    itself (see its module docstring), and it keeps this function's
    behavior fully determined by its inputs, not by when it happens to run.
    Unlike Statcast (one bulk range fetch), statsapi has no bulk date-range
    endpoint, so this is necessarily N fetches for N distinct pending dates
    - each one is try/except-guarded so one bad/unreachable date can't
    block resolving the others or raise out of this function.

    Only a `status == "Final"` game is resolved (winner = whichever side
    scored more). Any other status (Scheduled, In Progress, postponed,
    etc.) is left pending - the exact status strings MLB Stats API uses for
    a postponed/cancelled game aren't confirmed in this codebase, so rather
    than guess at them and risk mis-resolving a game, those picks simply
    stay pending indefinitely in this first pass (a safe failure mode, not
    a wrong-data one)."""
    if not os.path.exists(log_path):
        return pd.DataFrame(columns=GAME_PREDICTION_COLUMNS)

    log = pd.read_csv(log_path, parse_dates=["date"])
    if log.empty:
        return log
    if "game_played" not in log.columns:
        log["game_played"] = pd.NA  # migrate a log written before game_played existed
    if "model_version" not in log.columns:
        log["model_version"] = LEGACY_MODEL_VERSION  # migrate a log written before model_version existed
    # A log with no resolved games yet round-trips actual_winner as an
    # all-null float64 column (empty strings -> NaN on read) - cast back to
    # object so assigning a team abbreviation string into it doesn't raise.
    log["actual_winner"] = log["actual_winner"].astype("object")

    pending_dates = log.loc[
        log["game_played"].isna() & (log["date"] < pd.Timestamp(as_of_date)), "date"
    ].unique()

    for date in pending_dates:
        date = pd.Timestamp(date)
        try:
            results = fetch_results_fn(date.date())
        except Exception as exc:
            print(
                f"WARNING: failed to fetch game results for {date.date()} ({exc}); "
                f"leaving that date's game picks pending."
            )
            continue
        if results.empty:
            continue

        results = results[["game_pk", "status", "home_score", "away_score"]].rename(
            columns={"status": "_status", "home_score": "_home_score", "away_score": "_away_score"}
        )
        merged = log.merge(results, on="game_pk", how="left")

        day_mask = (log["date"] == date) & log["game_played"].isna()
        final = day_mask & (merged["_status"] == "Final")
        home_won = final & (merged["_home_score"] > merged["_away_score"])
        away_won = final & (merged["_away_score"] > merged["_home_score"])

        log.loc[final, "game_played"] = 1
        log.loc[home_won, "actual_winner"] = log.loc[home_won, "home_team"]
        log.loc[away_won, "actual_winner"] = log.loc[away_won, "away_team"]

    log.to_csv(log_path, index=False)
    return log
