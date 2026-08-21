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
    if "market_home_win_probability" not in picks.columns:
        # Migrate a log written before slice 2's market column existed -
        # genuinely no real market data for those rows, NaN not a guess.
        picks["market_home_win_probability"] = pd.NA
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

    market_accuracy, market_brier, market_ll, n_market_resolved = _market_comparison_metrics(recommended)
    beat_closing_line_rate, n_beat_closing_line_compared = _beat_closing_line_rate(recommended)

    picks_out = picks[
        [
            "date", "game_pk", "home_team", "away_team", "predicted_winner",
            "predicted_probability", "above_threshold", "status", "market_home_win_probability",
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
                "n_market_resolved": n_market_resolved,
                "market_accuracy": market_accuracy,
                "market_brier_score": market_brier,
                "market_log_loss": market_ll,
                "n_beat_closing_line_compared": n_beat_closing_line_compared,
                "beat_closing_line_rate": beat_closing_line_rate,
            }
        ]
    )
    return picks_out, summary


def _market_comparison_metrics(recommended: pd.DataFrame):
    """The market's own accuracy/Brier/log-loss on the same
    above_threshold-scoped picks, computed the same way as the model's own
    (game_evaluation.build_game_picks_export above): reuses
    evaluation.resolved_only/brier_score/log_loss by building a local frame
    with the market's probability of ITS OWN predicted winner renamed to
    "predicted_probability" - not a reimplementation, the same reuse
    pattern already used for the model side. Restricted to rows that
    actually have real market data (market_home_win_probability not null);
    real games slice 1/2 haven't backfilled yet correctly return NaN/0
    here, not a fabricated number. Quant-analytics item #6, slice 2."""
    if "market_home_win_probability" not in recommended.columns:
        return float("nan"), float("nan"), float("nan"), 0

    with_market = recommended[recommended["market_home_win_probability"].notna()].copy()
    if with_market.empty:
        return float("nan"), float("nan"), float("nan"), 0

    market_home_prob = pd.to_numeric(with_market["market_home_win_probability"], errors="coerce")
    favors_home = market_home_prob >= 0.5
    with_market["market_predicted_winner"] = with_market["home_team"].where(favors_home, with_market["away_team"])
    with_market["predicted_probability"] = market_home_prob.where(favors_home, 1 - market_home_prob)
    with_market["market_correct"] = pd.NA
    played = with_market["market_predicted_winner"].notna() & with_market["actual_winner"].notna()
    correct = with_market["market_predicted_winner"] == with_market["actual_winner"]
    with_market.loc[played & correct, "market_correct"] = 1.0
    with_market.loc[played & ~correct, "market_correct"] = 0.0

    resolved = evaluation.resolved_only(with_market, outcome_col="market_correct")
    n_resolved = len(resolved)
    accuracy = float(resolved["market_correct"].mean()) if n_resolved else float("nan")
    brier = evaluation.brier_score(with_market, outcome_col="market_correct")
    ll = evaluation.log_loss(with_market, outcome_col="market_correct")
    return accuracy, brier, ll, n_resolved


def _beat_closing_line_rate(recommended: pd.DataFrame):
    """The item's literal stated goal: "we beat the closing line," not
    just "we beat our own heuristic." Puts both the model's and the
    market's probabilities on the SAME basis (probability the HOME team
    wins - not each side's own predicted-winner probability, which would
    silently flip basis whenever the model and market favor different
    teams) and compares each side's squared error against the real
    actual-home-win outcome, per game. Reports the fraction of resolved,
    market-available games where the model's squared error is strictly
    lower than the market's - ties (equal squared error) are excluded
    from both the numerator and the denominator, and the real comparison
    base is reported separately as n_beat_closing_line_compared so a rate
    can never hide a tiny n."""
    if "market_home_win_probability" not in recommended.columns:
        return float("nan"), 0

    scoped = recommended[
        recommended["market_home_win_probability"].notna() & recommended["actual_winner"].notna()
    ].copy()
    if scoped.empty:
        return float("nan"), 0

    actual_home_win = (scoped["actual_winner"] == scoped["home_team"]).astype(float)
    model_favors_home = scoped["predicted_winner"] == scoped["home_team"]
    model_home_probability = scoped["predicted_probability"].where(
        model_favors_home, 1 - scoped["predicted_probability"]
    )
    market_home_probability = pd.to_numeric(scoped["market_home_win_probability"], errors="coerce")

    model_error = (model_home_probability - actual_home_win) ** 2
    market_error = (market_home_probability - actual_home_win) ** 2

    compared = model_error != market_error
    n_compared = int(compared.sum())
    if n_compared == 0:
        return float("nan"), 0

    beat = (model_error < market_error) & compared
    return float(beat.sum() / n_compared), n_compared
