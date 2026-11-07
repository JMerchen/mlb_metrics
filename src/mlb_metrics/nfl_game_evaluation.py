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
from scipy import stats

from mlb_metrics import config, evaluation, nfl_totals, nfl_weather

POINTS_COLUMNS = ["predicted_home_points", "predicted_away_points", "predicted_total", "market_total", "actual_total"]
DISPLAY_COLUMNS = POINTS_COLUMNS + nfl_totals.EPA_DISPLAY_COLUMNS + nfl_weather.WEATHER_DISPLAY_COLUMNS


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

    picks, report_compare = _latest_pick_per_game(picks)

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
        odds_implied_win_rate,
    ) = _bet_pnl_metrics(picks)

    home_favored = picks["predicted_winner"] == picks["home_team"]
    picks["predicted_loser"] = picks["away_team"].where(home_favored, picks["home_team"])
    picks["market_predicted_winner_probability"] = picks["market_home_win_probability"].where(
        home_favored, 1 - picks["market_home_win_probability"]
    )

    # Logs written before predicted points existed lack those columns.
    picks = picks.reindex(columns=picks.columns.union(DISPLAY_COLUMNS, sort=False))
    picks["predicted_total_low"], picks["predicted_total_high"] = total_band(picks["predicted_total"])
    picks = picks.join(total_leans(picks))
    # Tuesday's lean (the pick before the injury report, against Tuesday's
    # line), graded the same way - its closing-line value is the fairest
    # read, with the most time for the line to move.
    tuesday = total_leans(picks.assign(predicted_total=picks["before_report_predicted_total"],
                                       market_total=picks["before_report_market_total"]))
    for column in ("total_lean", "total_lean_line", "total_lean_result", "total_lean_clv"):
        picks[f"tuesday_{column}"] = tuesday[column]
    picks_out = picks[
        [
            "date", "season", "week", "game_id", "home_team", "away_team", "predicted_winner", "predicted_loser",
            "predicted_probability", "above_threshold", "status", "market_home_win_probability",
            "market_predicted_winner_probability",
            "bet_units", "bet_side", "bet_team", "bet_moneyline", "bet_profit_units",
            *DISPLAY_COLUMNS, "predicted_total_low", "predicted_total_high",
            "total_range_result", "total_range_miss", "total_lean", "total_lean_line", "total_lean_result",
            "total_lean_close", "total_lean_clv",
            "tuesday_total_lean", "tuesday_total_lean_line", "tuesday_total_lean_result", "tuesday_total_lean_clv",
            "injury_report", "injury_note", "before_report_winner", "before_report_probability",
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
                "odds_implied_win_rate": odds_implied_win_rate,
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
                **evaluation.model_vs_market(picks),
                **points_total_metrics(picks),
                **total_lean_metrics(picks),
                **{f"tuesday_{k}": v for k, v in total_lean_metrics(
                    picks.rename(columns={"total_lean_result": "_shown_result", "total_lean_clv": "_shown_clv",
                                          "tuesday_total_lean_result": "total_lean_result",
                                          "tuesday_total_lean_clv": "total_lean_clv"})).items()},
                **report_compare,
            }
        ]
    )
    return picks_out, summary


def _latest_pick_per_game(picks: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One row per game: the pick after the final injury report where there
    is one (nfl_game_predictions logs both), else the one before it - with
    the before-report pick alongside (before_report_winner/_probability).
    Also, over finished games picked both ways, how often each was right
    (n_report_compared, before_report_accuracy, after_report_accuracy)."""
    picks = picks.copy()
    if "injury_report" not in picks.columns:
        picks["injury_report"] = "before"
    if "injury_note" not in picks.columns:
        picks["injury_note"] = pd.NA
    picks["injury_report"] = picks["injury_report"].fillna("before")
    before = picks[picks["injury_report"] != "after"].drop_duplicates("game_id", keep="first").set_index("game_id")
    after = picks[picks["injury_report"] == "after"].drop_duplicates("game_id", keep="last")
    latest = pd.concat([after, picks[(picks["injury_report"] != "after") & ~picks["game_id"].isin(after["game_id"])]],
                       ignore_index=True)
    latest["before_report_winner"] = latest["game_id"].map(before["predicted_winner"])
    latest["before_report_probability"] = latest["game_id"].map(before["predicted_probability"])
    # Tuesday's total and the line it was compared with, for grading
    # Tuesday's over/under lean on its own (the earliest line to bet).
    for column in ("predicted_total", "market_total"):
        latest[f"before_report_{column}"] = latest["game_id"].map(before[column]) if column in before.columns else pd.NA

    both = after.set_index("game_id").join(before[["predicted_winner"]].rename(columns={"predicted_winner": "before_winner"}),
                                           how="inner")
    done = both[both["actual_winner"].notna() & (both["game_played"] == 1)]
    compare = {"n_report_compared": len(done)}
    if len(done):
        compare["before_report_accuracy"] = float((done["before_winner"] == done["actual_winner"]).mean())
        compare["after_report_accuracy"] = float((done["predicted_winner"] == done["actual_winner"]).mean())
    return latest, compare


def points_total_metrics(picks: pd.DataFrame) -> dict:
    """How far the predicted point total landed from the real one, next to
    the market's total, on the same played games (those with a predicted,
    a market and an actual total): the average miss in points, and how
    often the real total fell in the range shown on the site (ours, and
    the same width around Vegas's - config.NFL_TOTAL_BAND_HALF_WIDTH).
    NaN (and n 0) when no game qualifies."""
    columns = ["predicted_total", "market_total", "actual_total"]
    empty = {"points_total_n": 0, "points_total_mae": float("nan"), "market_points_total_mae": float("nan"),
             "points_total_in_range": float("nan"), "market_points_total_in_range": float("nan")}
    if picks.empty or not set(columns).issubset(picks.columns):
        return empty
    totals = picks[columns].apply(pd.to_numeric, errors="coerce").dropna()
    if totals.empty:
        return empty
    ours = totals["predicted_total"] - totals["actual_total"]
    vegas = totals["market_total"] - totals["actual_total"]
    low, high = total_band(totals["predicted_total"])
    return {
        "points_total_n": int(len(totals)),
        "points_total_mae": float(ours.abs().mean()),
        "market_points_total_mae": float(vegas.abs().mean()),
        "points_total_in_range": float(((totals["actual_total"] >= low) & (totals["actual_total"] <= high)).mean()),
        "market_points_total_in_range": float((vegas.abs() <= config.NFL_TOTAL_BAND_HALF_WIDTH).mean()),
    }


def total_leans(picks: pd.DataFrame) -> pd.DataFrame:
    """For each pick: whether the real total landed in our range
    (total_range_result "in", "over" or "under" with total_range_miss, the
    points outside it), and the over/under lean when our total is
    config.NFL_TOTAL_LEAN_MIN_GAP or more from Vegas's (total_lean "over"
    or "under" at total_lean_line, settled as total_lean_result "win",
    "loss" or "push"; blank until the game is played)."""
    num = lambda c: pd.to_numeric(picks[c], errors="coerce") if c in picks.columns else pd.Series(float("nan"), index=picks.index)
    predicted, market, actual = num("predicted_total"), num("market_total"), num("actual_total")
    low, high = total_band(predicted)
    out = pd.DataFrame(index=picks.index)
    out["total_range_result"] = pd.Series(pd.NA, index=picks.index, dtype=object)
    played = actual.notna() & predicted.notna()
    out.loc[played & (actual > high), "total_range_result"] = "over"
    out.loc[played & (actual < low), "total_range_result"] = "under"
    out.loc[played & (actual >= low) & (actual <= high), "total_range_result"] = "in"
    out["total_range_miss"] = (actual - high).where(actual > high, (low - actual).where(actual < low, 0.0)).where(played)
    gap = predicted - market
    lean = pd.Series(pd.NA, index=picks.index, dtype=object)
    lean[gap >= config.NFL_TOTAL_LEAN_MIN_GAP] = "over"
    lean[gap <= -config.NFL_TOTAL_LEAN_MIN_GAP] = "under"
    out["total_lean"] = lean
    out["total_lean_line"] = market.where(lean.notna())
    result = pd.Series(pd.NA, index=picks.index, dtype=object)
    settled = lean.notna() & actual.notna()
    went_over = actual > market
    result[settled & (actual == market)] = "push"
    result[settled & (actual != market) & ((lean == "over") == went_over)] = "win"
    result[settled & (actual != market) & ((lean == "over") != went_over)] = "loss"
    out["total_lean_result"] = result
    # Closing-line value: how many points the line moved toward the lean
    # between when it was posted (total_lean_line, the line in the pick)
    # and kickoff (closing_total). Positive = we got a better number than
    # the market's final one.
    closing = num("closing_total")
    toward = (closing - market).where(lean == "over", (market - closing).where(lean == "under"))
    out["total_lean_close"] = closing.where(lean.notna())
    out["total_lean_clv"] = toward.where(lean.notna() & closing.notna())
    return out


def total_lean_metrics(picks: pd.DataFrame) -> dict:
    """The over/under leans' record: won, lost, pushed, and units at the
    standard -110 (risk 1.1 to win 1, reported per unit risked)."""
    result = picks["total_lean_result"] if "total_lean_result" in picks.columns else pd.Series(dtype=object)
    won, lost, pushed = int((result == "win").sum()), int((result == "loss").sum()), int((result == "push").sum())
    clv = pd.to_numeric(picks["total_lean_clv"], errors="coerce").dropna() if "total_lean_clv" in picks.columns \
        else pd.Series(dtype=float)
    n = len(clv)
    # One-sided t-test that the average move is toward us (needs 3+ leans
    # and some spread).
    p = float(stats.ttest_1samp(clv, 0.0, alternative="greater").pvalue) if n >= 3 and clv.std() > 0 else float("nan")
    return {"total_leans_won": won, "total_leans_lost": lost, "total_leans_pushed": pushed,
            "total_lean_units": round(won * 100 / 110 - lost, 2),
            "total_lean_clv_n": n, "total_lean_clv_mean": float(clv.mean()) if n else float("nan"),
            "total_lean_clv_beat": int((clv > 0).sum()), "total_lean_clv_worse": int((clv < 0).sum()),
            "total_lean_clv_p": p}


def total_band(predicted_total: pd.Series) -> tuple[pd.Series, pd.Series]:
    """The range shown for a predicted total: whole points,
    config.NFL_TOTAL_BAND_HALF_WIDTH either side (never below 0)."""
    total = pd.to_numeric(predicted_total, errors="coerce")
    half = config.NFL_TOTAL_BAND_HALF_WIDTH
    return (total - half).round().clip(lower=0), (total + half).round()


def _bet_pnl_metrics(picks: pd.DataFrame):
    """Real units won/lost, scoped to games where a bet was actually
    ADVISED (bet_units > 0) - direct mirror of
    game_evaluation._bet_pnl_metrics (see its own docstring for the full
    reasoning: day streaks not per-bet streaks, ROI significance via a
    real one-sample t-test on bet_profit_units)."""
    resolved = picks[picks["bet_profit_units"].notna()].copy()
    n_bets_advised = len(resolved)
    if n_bets_advised == 0:
        return 0, 0, 0, float("nan"), 0.0, 0.0, float("nan"), 0, 0, 0.0, 1.0, float("nan"), float("nan")

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
    # What the same bets were priced to win - the yardstick win_rate is
    # read against (see evaluation.odds_implied_win_rate).
    odds_implied = evaluation.odds_implied_win_rate(resolved["bet_moneyline"])

    daily_profit = resolved.groupby("date")["bet_profit_units"].sum().sort_index()
    current_streak = 0
    best_streak = 0
    for day_profit in daily_profit:
        current_streak = current_streak + 1 if day_profit > 0 else 0
        best_streak = max(best_streak, current_streak)

    return (
        n_bets_advised, bets_won, bets_lost, win_rate, total_staked, total_profit, roi, current_streak, best_streak,
        win_rate_ci_low, win_rate_ci_high, roi_p_value, odds_implied,
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
