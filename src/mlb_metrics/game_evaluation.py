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
    that `above_threshold` flag so the dashboard can highlight the model's
    own confident picks alongside the rest. summary_row's real tracking
    (`n_bets_advised`/`total_profit_units`/etc., see `_bet_pnl_metrics`
    below) is scoped to `bet_units > 0` - NOT `above_threshold` - since the
    real question worth tracking is "did the bets the market actually
    disagreed with make money," not "was the model's own favorite right."
    (`above_threshold` still flags/publishes every game, it just no longer
    drives the headline scoring - see `_market_comparison_metrics`/
    `_beat_closing_line_rate` below for the genuinely separate "are we
    better forecasters than the market" question, which IS still scoped to
    `above_threshold` and untouched by this.)

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
    if "bet_units" not in picks.columns:
        # A row logged before bet advice existed genuinely never had a bet
        # advised - 0.0 units is the factually correct backfill, not a guess.
        picks["bet_units"] = 0.0
        for col in ("bet_side", "bet_team", "bet_moneyline", "bet_stake_fraction", "bet_profit_units"):
            picks[col] = pd.NA
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

    # Real follow-up (2026-08-28 - "we're dumping almost everything data
    # wise into the [History] table"): predicted_loser and
    # market_predicted_winner_probability are new, DERIVED display
    # columns - the dashboard's History table shows only these plus a
    # small, curated set of the existing ones (see docs/app.js's
    # renderGamePickHistory), while renderTodaysGamePicks keeps reading
    # the full row for its own cards (home_team/away_team/bet_moneyline/
    # etc. are still real, needed data - not removed here, just not all
    # surfaced in the History table).
    home_favored = picks["predicted_winner"] == picks["home_team"]
    picks["predicted_loser"] = picks["away_team"].where(home_favored, picks["home_team"])
    # The market's own real probability for the SAME side the model
    # favored - de-vigged market_home_win_probability is always the HOME
    # team's probability, so when the model favors the away team this
    # needs flipping (1 - p) to stay an apples-to-apples "model prob vs.
    # market prob for the predicted winner" comparison, not a home-vs-
    # picked-side mismatch. NaN-safe: 1 - NaN stays NaN, same "no real
    # market data" signal market_home_win_probability's own NaN already
    # carries.
    picks["market_predicted_winner_probability"] = picks["market_home_win_probability"].where(
        home_favored, 1 - picks["market_home_win_probability"]
    )

    picks_out = picks[
        [
            "date", "game_pk", "home_team", "away_team", "predicted_winner", "predicted_loser",
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
    ADVISED (game_predictions.advise_bets' real Kelly-edge gate cleared,
    i.e. bet_units > 0) - NOT the model's own above_threshold confidence
    gate. This project's real quant-facing question is "did the advised
    bets make money," not "was the model's favorite pick accurate" - see
    build_game_picks_export's own docstring. Scoped to
    bet_profit_units.notna() - real, resolved, advised bets only; a still-
    pending advised bet doesn't count yet, and a non-advised game
    (bet_units == 0) never gets a bet_profit_units value at all (see
    game_predictions.resolve_game_predictions).

    `current_bet_streak`/`best_bet_streak` are DAY streaks (2026-08-25 -
    direct follow-up to the uncertainty-scaled Kelly change above), not
    per-bet streaks: a day extends the streak by exactly 1 if that day's
    real advised bets, summed together, made money overall - and resets
    it to 0 otherwise - regardless of how many individual bets were
    advised that day or how they each did. A day with two winners and one
    bigger loser is a losing day for the streak; a single-bet day and a
    five-bet day both count for at most one real streak step.

    Also returns two quant-analytics item #5 ("backtest scope and
    statistical significance") additions:
    - `win_rate_ci_low`/`win_rate_ci_high`: a Wilson CI on win_rate,
      informational only (see evaluation.binomial_significance's own
      docstring for why a win-rate p-value against 0.5 would be
      statistically wrong here - moneylines vary bet to bet, so a raw
      win/loss count alone can't tell a good -150 favorite bet apart
      from a bad one the way real profit can).
    - `roi_p_value`: the real, correctly-posed test - a one-sample
      t-test (evaluation.mean_significance) on each advised bet's real
      bet_profit_units against a null of 0 ("breaking even"). This is
      the honest answer to "is this edge distinguishable from noise
      yet," not just reporting a P&L number that could flip sign with
      the next handful of bets."""
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

    # Streak is DAYS, not bets (2026-08-25 - "the streak should be days...
    # if the cumulative bets made money that day"): a day can carry several
    # advised bets, but it counts for at most +1 (or a reset to 0) toward
    # the streak, scored on that day's TOTAL real profit, not on any one
    # bet in isolation - a day with a winner and a bigger loser is a losing
    # day, not a streak-extending one. Only days that actually had a
    # resolved advised bet enter this at all (grouping by date on
    # `resolved`), so a day with no advice neither extends nor breaks it.
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
    # Quant-analytics item #5: CI only, deliberately no p-value here - real
    # MLB home teams win somewhat more than half their games, so 0.5 isn't
    # a genuine "no skill" null for an unconditional accuracy rate the way
    # it is for _beat_closing_line_rate's symmetric win/loss-per-game
    # comparison below (see evaluation.binomial_significance's docstring).
    ci_low, ci_high = evaluation.wilson_confidence_interval(int(resolved["market_correct"].sum()), n_resolved) if n_resolved else (float("nan"), float("nan"))
    return accuracy, brier, ll, n_resolved, ci_low, ci_high


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
    can never hide a tiny n.

    Also returns a real Wilson CI and, unlike the other rate metrics in
    this module, a real binomial_significance p-value against a null of
    0.5 (quant-analytics item #5, "backtest scope and statistical
    significance") - this IS a well-posed 0.5 null, unlike a raw
    accuracy rate: "whose squared error is lower on this game" is a
    genuinely symmetric coin flip under "no real skill difference
    between the model and the market," so a small n like the 12-game
    read this project started with can be honestly flagged as not yet
    distinguishable from chance instead of read as real evidence of an
    edge."""
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
