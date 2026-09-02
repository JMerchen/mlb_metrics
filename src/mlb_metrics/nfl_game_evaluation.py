"""NFL game-pick backtest scoring - direct structural mirror of
game_evaluation.py (see that module's own docstring), keyed on real
`game_id` (not `game_pk`) and reusing evaluation.py's generic scoring
functions exactly the same way. Unlike game_evaluation.py, this module
carries no legacy-schema migration branches (`above_threshold`/
`market_home_win_probability`/`bet_units` backfills) - nfl_game_predictions.py's
own log has always written every one of those columns from day one (see
the approved plan's "no historical backfill of the NFL prediction log
itself" non-goal), so there is no pre-existing schema to migrate from.
"""

import pandas as pd

from mlb_metrics import evaluation


def _classify_outcome(df: pd.DataFrame) -> pd.Series:
    """Per-pick outcome: "pending" (game_played unknown yet), "not_played"
    (confirmed postponed/cancelled - game_played=0), "win" (predicted_winner
    matched actual_winner), or "loss" (it didn't). Direct mirror of
    game_evaluation._classify_outcome."""
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
    metric: str = "NFL_GamePick_Win_Probability",
    model_version: str | None = None,
):
    """Build the two tables the dashboard's NFL Automated Game Picks
    section reads: (picks_table, summary_row). Direct structural mirror
    of game_evaluation.build_game_picks_export - see that function's own
    docstring for the full reasoning (above_threshold flags every logged
    game, summary_row's real bet P&L tracking is scoped to bet_units > 0
    not above_threshold, predicted_loser/market_predicted_winner_probability
    are derived display columns for the History table's curated column set)."""
    picks = predictions[predictions["metric"] == metric].copy()
    if model_version is not None:
        picks = picks[picks["model_version"] == model_version] if "model_version" in picks.columns else picks.iloc[0:0]

    picks["status"] = _classify_outcome(picks)
    picks["actual_correct"] = pd.NA
    picks.loc[picks["status"] == "win", "actual_correct"] = 1.0
    picks.loc[picks["status"] == "loss", "actual_correct"] = 0.0

    recommended = picks[picks["above_threshold"]]
    (
        market_accuracy, market_brier, market_ll, n_market_resolved,
        market_accuracy_ci_low, market_accuracy_ci_high,
    ) = _market_comparison_metrics(recommended)
    (
        beat_closing_line_rate, n_beat_closing_line_compared,
        beat_closing_line_rate_ci_low, beat_closing_line_rate_ci_high, beat_closing_line_rate_p_value,
    ) = _beat_closing_line_rate(recommended)
    (
        n_bets_advised, bets_won, bets_lost, win_rate_on_advised_bets,
        total_staked_units, total_profit_units, roi, current_bet_streak, best_bet_streak,
        win_rate_on_advised_bets_ci_low, win_rate_on_advised_bets_ci_high, roi_p_value,
    ) = _bet_pnl_metrics(picks)

    home_favored = picks["predicted_winner"] == picks["home_team"]
    picks["predicted_loser"] = picks["away_team"].where(home_favored, picks["home_team"])
    picks["market_predicted_winner_probability"] = picks["market_home_win_probability"].where(
        home_favored, 1 - picks["market_home_win_probability"]
    )

    picks_out = picks[
        [
            "date", "season", "week", "game_id", "home_team", "away_team", "predicted_winner", "predicted_loser",
            "predicted_probability", "above_threshold", "status", "market_home_win_probability",
            "market_predicted_winner_probability",
            "bet_units", "bet_side", "bet_team", "bet_moneyline", "bet_profit_units",
        ]
    ].sort_values("date", ascending=False).reset_index(drop=True)

    summary = pd.DataFrame(
        [
            {
                "model_version": model_version if model_version is not None else "all_time",
                "metric": metric,
                "n_bets_advised": n_bets_advised,
                "bets_won": bets_won,
                "bets_lost": bets_lost,
                "win_rate_on_advised_bets": win_rate_on_advised_bets,
                "win_rate_on_advised_bets_ci_low": win_rate_on_advised_bets_ci_low,
                "win_rate_on_advised_bets_ci_high": win_rate_on_advised_bets_ci_high,
                "total_staked_units": total_staked_units,
                "total_profit_units": total_profit_units,
                "roi": roi,
                "roi_p_value": roi_p_value,
                "current_bet_streak": current_bet_streak,
                "best_bet_streak": best_bet_streak,
                "n_market_resolved": n_market_resolved,
                "market_accuracy": market_accuracy,
                "market_accuracy_ci_low": market_accuracy_ci_low,
                "market_accuracy_ci_high": market_accuracy_ci_high,
                "market_brier_score": market_brier,
                "market_log_loss": market_ll,
                "n_beat_closing_line_compared": n_beat_closing_line_compared,
                "beat_closing_line_rate": beat_closing_line_rate,
                "beat_closing_line_rate_ci_low": beat_closing_line_rate_ci_low,
                "beat_closing_line_rate_ci_high": beat_closing_line_rate_ci_high,
                "beat_closing_line_rate_p_value": beat_closing_line_rate_p_value,
            }
        ]
    )
    return picks_out, summary


def _bet_pnl_metrics(picks: pd.DataFrame):
    """Real units won/lost, scoped to games where a bet was actually
    ADVISED (bet_units > 0) - direct mirror of
    game_evaluation._bet_pnl_metrics (see its own docstring for the full
    reasoning: day streaks not per-bet streaks, ROI significance via a
    real one-sample t-test on bet_profit_units)."""
    resolved = picks[picks["bet_profit_units"].notna()].copy()
    n_bets_advised = len(resolved)
    if n_bets_advised == 0:
        return 0, 0, 0, float("nan"), 0.0, 0.0, float("nan"), 0, 0, 0.0, 1.0, float("nan")

    resolved["bet_profit_units"] = resolved["bet_profit_units"].astype(float)
    resolved["bet_units"] = resolved["bet_units"].astype(float)

    bets_won = int((resolved["bet_profit_units"] > 0).sum())
    bets_lost = int((resolved["bet_profit_units"] < 0).sum())
    win_rate = bets_won / n_bets_advised
    win_rate_ci_low, win_rate_ci_high = evaluation.wilson_confidence_interval(bets_won, n_bets_advised)
    total_staked = float(resolved["bet_units"].sum())
    total_profit = float(resolved["bet_profit_units"].sum())
    roi = total_profit / total_staked if total_staked else float("nan")
    roi_p_value = evaluation.mean_significance(resolved["bet_profit_units"], null_value=0.0)

    daily_profit = resolved.groupby("date")["bet_profit_units"].sum().sort_index()
    current_streak = 0
    best_streak = 0
    for day_profit in daily_profit:
        current_streak = current_streak + 1 if day_profit > 0 else 0
        best_streak = max(best_streak, current_streak)

    return (
        n_bets_advised, bets_won, bets_lost, win_rate, total_staked, total_profit, roi, current_streak, best_streak,
        win_rate_ci_low, win_rate_ci_high, roi_p_value,
    )


def _market_comparison_metrics(recommended: pd.DataFrame):
    """The market's own accuracy/Brier/log-loss on the same
    above_threshold-scoped picks - direct mirror of
    game_evaluation._market_comparison_metrics."""
    if "market_home_win_probability" not in recommended.columns:
        return float("nan"), float("nan"), float("nan"), 0, float("nan"), float("nan")

    with_market = recommended[recommended["market_home_win_probability"].notna()].copy()
    if with_market.empty:
        return float("nan"), float("nan"), float("nan"), 0, float("nan"), float("nan")

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
    ci_low, ci_high = evaluation.wilson_confidence_interval(int(resolved["market_correct"].sum()), n_resolved) if n_resolved else (float("nan"), float("nan"))
    return accuracy, brier, ll, n_resolved, ci_low, ci_high


def _beat_closing_line_rate(recommended: pd.DataFrame):
    """Direct mirror of game_evaluation._beat_closing_line_rate - see its
    own docstring for the full reasoning (same basis, same ties-excluded
    comparison, same Wilson CI + binomial p-value against a genuinely
    well-posed 0.5 null)."""
    if "market_home_win_probability" not in recommended.columns:
        return float("nan"), 0, float("nan"), float("nan"), float("nan")

    scoped = recommended[
        recommended["market_home_win_probability"].notna() & recommended["actual_winner"].notna()
    ].copy()
    if scoped.empty:
        return float("nan"), 0, float("nan"), float("nan"), float("nan")

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
        return float("nan"), 0, float("nan"), float("nan"), float("nan")

    beat = (model_error < market_error) & compared
    n_beat = int(beat.sum())
    rate = float(n_beat / n_compared)
    ci_low, ci_high = evaluation.wilson_confidence_interval(n_beat, n_compared)
    p_value = evaluation.binomial_significance(n_beat, n_compared, null_probability=0.5)
    return rate, n_compared, ci_low, ci_high, p_value
