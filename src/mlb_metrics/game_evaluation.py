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
):
    """Build the two tables the dashboard's Automated Game Picks section
    reads: (picks_table, summary_row). picks_table is every picked game
    (metric matches, predicted_probability cleared min_probability - same
    re-filter-at-export-time approach as Beat the Streak's
    _recommended_picks, so this stays correct even if a second candidate
    formula is ever logged under a different `metric`) with a
    win/loss/not_played/pending status, most recent day first. summary_row
    has n_games_resolved/accuracy/brier_score/log_loss plus a simple
    current_streak/best_streak of consecutive correct picks - a plain
    counter, not Beat the Streak's reset-on-any-miss multi-pick mechanic
    (that's MLB's specific hitter-streak game rule and doesn't apply to
    independent game-by-game win/loss picks)."""
    picks = predictions[
        (predictions["metric"] == metric) & (predictions["predicted_probability"] >= min_probability)
    ].copy()
    picks["status"] = _classify_outcome(picks)
    picks["actual_correct"] = pd.NA
    picks.loc[picks["status"] == "win", "actual_correct"] = 1.0
    picks.loc[picks["status"] == "loss", "actual_correct"] = 0.0

    resolved = evaluation.resolved_only(picks, outcome_col="actual_correct")
    n_resolved = len(resolved)
    accuracy = float(resolved["actual_correct"].mean()) if n_resolved else float("nan")
    brier = evaluation.brier_score(picks, outcome_col="actual_correct")
    ll = evaluation.log_loss(picks, outcome_col="actual_correct")

    current_streak = 0
    best_streak = 0
    for correct in resolved.sort_values("date")["actual_correct"]:
        current_streak = current_streak + 1 if correct == 1 else 0
        best_streak = max(best_streak, current_streak)

    picks_out = picks[
        ["date", "game_pk", "home_team", "away_team", "predicted_winner", "predicted_probability", "status"]
    ].sort_values("date", ascending=False).reset_index(drop=True)

    summary = pd.DataFrame(
        [
            {
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
