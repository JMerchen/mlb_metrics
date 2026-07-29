"""Game-pick backtest scoring - the game-level analog of evaluation.py's
Beat the Streak Tracker export (build_beat_the_streak_export). See
game_picks.py/game_predictions.py.
"""

import pandas as pd

from mlb_metrics import config, evaluation


def _classify_outcome(df: pd.DataFrame) -> pd.Series:
    """Per-pick outcome: "pending" (game_played unknown yet), "not_played"
    (confirmed postponed/cancelled - game_played=0), "win" (predicted_winner
    matched actual_winner), or "loss" (it didn't)."""
    game_played = pd.to_numeric(df["game_played"], errors="coerce")

    outcome = pd.Series("pending", index=df.index)
    outcome[game_played == 0] = "not_played"
    played = game_played == 1
    correct = df["predicted_winner"] == df["actual_winner"]
    outcome[played & correct] = "win"
    outcome[played & ~correct] = "loss"
    return outcome


def build_game_picks_export(
    predictions: pd.DataFrame,
    metric: str = "GamePick_Win_Probability",
    min_probability: float = config.GAME_PICK_MIN_PROBABILITY,
    model_version: str | None = None,
):
    """Build the two tables the dashboard's Automated Game Picks section
    reads: (picks_table, summary_row). picks_table is EVERY logged game for
    `metric` (no longer filtered by predicted_probability - see
    game_predictions.select_game_picks, which now logs every scheduled game
    with an `above_threshold` flag instead of dropping sub-threshold ones)
    with a win/loss/not_played/pending status, most recent day first, plus
    that `above_threshold` flag so the dashboard can highlight the confident
    ones without hiding the rest. summary_row's n_games_resolved/accuracy/
    brier_score/log_loss/current_streak/best_streak are computed from the
    `above_threshold` subset ONLY - scoring stays scoped to confident picks
    exactly as before this change, a below-threshold game's real outcome
    must never move these numbers. best_streak is a plain counter of
    consecutive correct picks, not Beat the Streak's reset-on-any-miss
    multi-pick mechanic (that's MLB's specific hitter-streak game rule and
    doesn't apply to independent game-by-game win/loss picks).

    `model_version` (default None, i.e. every version blended together -
    unchanged behavior) restricts to picks tagged with a specific
    game_predictions.select_game_picks model_version (see
    config.GAME_PICK_MODEL_VERSION) - same reasoning as
    evaluation.summarize's own model_version filter."""
    picks = predictions[predictions["metric"] == metric].copy()
    if model_version is not None:
        picks = picks[picks["model_version"] == model_version] if "model_version" in picks.columns else picks.iloc[0:0]
    if "above_threshold" not in picks.columns:
        # Migrate a log written before this column existed - every row in
        # it was already filtered to predicted_probability >= min_probability
        # at logging time (see game_predictions.select_game_picks), so
        # recomputing the flag from that same comparison is exact, not a guess.
        picks["above_threshold"] = picks["predicted_probability"] >= min_probability
    picks["status"] = _classify_outcome(picks)
    picks["actual_correct"] = pd.NA
    picks.loc[picks["status"] == "win", "actual_correct"] = 1.0
    picks.loc[picks["status"] == "loss", "actual_correct"] = 0.0

    recommended = picks[picks["above_threshold"]]
    resolved = evaluation.resolved_only(recommended, outcome_col="actual_correct")
    n_resolved = len(resolved)
    accuracy = float(resolved["actual_correct"].mean()) if n_resolved else float("nan")
    brier = evaluation.brier_score(recommended, outcome_col="actual_correct")
    ll = evaluation.log_loss(recommended, outcome_col="actual_correct")

    current_streak = 0
    best_streak = 0
    for correct in resolved.sort_values("date")["actual_correct"]:
        current_streak = current_streak + 1 if correct == 1 else 0
        best_streak = max(best_streak, current_streak)

    picks_out = picks[
        [
            "date", "game_pk", "home_team", "away_team", "predicted_winner",
            "predicted_probability", "above_threshold", "status",
        ]
    ].sort_values("date", ascending=False).reset_index(drop=True)

    summary = pd.DataFrame(
        [
            {
                "model_version": model_version if model_version is not None else "all_time",
                "metric": metric,
                "min_probability": min_probability,
                "n_games_resolved": n_resolved,
                "accuracy": accuracy,
                "brier_score": brier,
                "log_loss": ll,
                "current_streak": current_streak,
                "best_streak": best_streak,
            }
        ]
    )
    return picks_out, summary
