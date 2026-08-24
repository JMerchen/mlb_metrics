"""Backtest scoring: turns a resolved predictions log (from predictions.py)
into the numbers that actually answer "does this beat a coin flip, let
alone Beat the Streak" - hit rate by pick rank, calibration, Brier score,
and log loss.
"""

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_1samp

from mlb_metrics import helpers


def resolved_only(predictions: pd.DataFrame, outcome_col: str = "actual_hit") -> pd.DataFrame:
    """Rows with a known outcome (0/1) in `outcome_col`, i.e. the game has
    been played. `outcome_col` defaults to "actual_hit" (hitter picks); pass
    "actual_correct" for game picks (see game_evaluation.py)."""
    resolved = predictions[predictions[outcome_col].notna()].copy()
    resolved[outcome_col] = resolved[outcome_col].astype(float)
    return resolved


def pick_accuracy_by_rank(predictions: pd.DataFrame, outcome_col: str = "actual_hit") -> pd.DataFrame:
    """Hit rate for each individual pick rank (1st-ranked pick, 2nd-ranked,
    ...), independent of the others. If the model has any skill, this should
    decrease as rank increases; if it's flat, the ranking isn't doing anything."""
    resolved = resolved_only(predictions, outcome_col)
    if resolved.empty:
        return pd.DataFrame(columns=["rank", "hit_rate", "n"])
    grouped = resolved.groupby("rank")[outcome_col].agg(hit_rate="mean", n="count").reset_index()
    return grouped.sort_values("rank").reset_index(drop=True)


def top_k_hit_rate(predictions: pd.DataFrame, k: int, require_all: bool = False, outcome_col: str = "actual_hit") -> float:
    """Per-day rate of success using the top `k` picks. require_all=False
    (default) scores a day as a "hit" if *any* of the top-k picks got a hit
    (a "pick k, need one" strategy); require_all=True scores a day as a hit
    only if *all* k did (Beat the Streak's actual multi-pick mode, where
    every pick must land to extend the streak)."""
    resolved = resolved_only(predictions, outcome_col)
    picks = resolved[resolved["rank"] <= k]
    if picks.empty:
        return float("nan")
    per_day = picks.groupby("date")[outcome_col]
    outcome = per_day.min() if require_all else per_day.max()
    return float(outcome.mean())


def brier_score(predictions: pd.DataFrame, outcome_col: str = "actual_hit") -> float:
    """Mean squared error between predicted probability and actual (0/1)
    outcome - lower is better, 0 is perfect, 0.25 is what an uninformative
    always-predict-0.5 model scores."""
    resolved = resolved_only(predictions, outcome_col)
    if resolved.empty:
        return float("nan")
    return float(np.mean((resolved["predicted_probability"] - resolved[outcome_col]) ** 2))


def log_loss(predictions: pd.DataFrame, eps: float = 1e-6, outcome_col: str = "actual_hit") -> float:
    resolved = resolved_only(predictions, outcome_col)
    if resolved.empty:
        return float("nan")
    p = resolved["predicted_probability"].clip(eps, 1 - eps)
    y = resolved[outcome_col]
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def wilson_confidence_interval(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Scalar Wilson score confidence interval for one aggregate backtest
    rate (accuracy, beat_closing_line_rate, win_rate_on_advised_bets,
    day_survival_rate, ...) - quant-analytics item #5 ("backtest scope
    and statistical significance"): the honest answer to "is n big
    enough to trust this rate," not just reporting the rate itself. A
    12-game 33% "beat the market" rate and a 1200-game 33% rate are NOT
    the same amount of evidence, and a bare percentage on a dashboard
    can't tell them apart.

    Reuses helpers.wilson_ci (quant-analytics item #3's per-row
    vectorized version, already proven against
    statsmodels.stats.proportion.proportion_confint) by wrapping the
    scalar successes/n in a length-1 Series and unwrapping the result -
    not a second implementation of the same formula. n=0 returns
    (0.0, 1.0), the same "no information, could be anywhere" contract
    helpers.wilson_ci already establishes."""
    low, high = helpers.wilson_ci(pd.Series([successes]), pd.Series([n]), alpha=alpha)
    return float(low.iloc[0]), float(high.iloc[0])


def binomial_significance(successes: int, n: int, null_probability: float = 0.5) -> float:
    """Two-sided exact binomial test p-value: given only `n` real
    trials, is the observed rate distinguishable from `null_probability`
    (default 0.5 - a coin flip, i.e. "no real skill difference")? A
    small p-value means the observed rate would be unlikely if the null
    were actually true. Uses scipy.stats.binomtest - the exact test, not
    a normal approximation (unreliable at the small n this project's
    real backtests actually have, e.g. n=12) - not hand-derived.

    0.5 is only a well-posed null for a genuinely symmetric comparison
    (e.g. beat_closing_line_rate's "whose squared error was lower on
    this game," where under "no skill difference" either side is
    equally likely to win) - NOT for an unconditional accuracy rate
    (home teams win somewhat more than half of real MLB games, so 0.5
    isn't actually "no skill" there). Callers are responsible for only
    using this where the null is real, not just convenient - see
    game_evaluation.py's own callers for which metrics get a p-value at
    all versus a confidence interval only. n=0 returns NaN, not a
    fabricated 1.0 (no data is no evidence either way, not "certainly
    the null")."""
    if n == 0:
        return float("nan")
    return float(binomtest(successes, n, null_probability).pvalue)


def mean_significance(values: pd.Series, null_value: float = 0.0) -> float:
    """One-sample two-sided t-test p-value: is the real mean of `values`
    (e.g. each advised bet's real bet_profit_units) distinguishable from
    `null_value` (default 0.0 - "breaking even")? This is the honest
    test for "did the advised bets actually make money, or is this
    within noise" - deliberately NOT binomial_significance on
    win_rate_on_advised_bets, which would silently throw away each bet's
    real price (a -150 favorite winning 55% of the time and a +150
    underdog winning 55% of the time are very different real outcomes
    that a win-rate-only test can't tell apart; the real profit each bet
    produced already prices that in). Needs at least 2 real resolved
    values to estimate a variance; returns NaN otherwise, not a
    fabricated p-value off a single data point."""
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 2:
        return float("nan")
    return float(ttest_1samp(clean, null_value).pvalue)


def calibration_table(predictions: pd.DataFrame, n_bins: int = 10, outcome_col: str = "actual_hit") -> pd.DataFrame:
    """Bins picks by predicted probability and compares each bin's mean
    predicted probability to its actual hit rate - a well-calibrated metric
    should have predicted_mean ~= actual_rate in every bin."""
    resolved = resolved_only(predictions, outcome_col)
    if resolved.empty:
        return pd.DataFrame(columns=["bin", "predicted_mean", "actual_rate", "n"])

    bins = pd.cut(resolved["predicted_probability"], bins=n_bins, include_lowest=True)
    grouped = (
        resolved.groupby(bins, observed=True)
        .agg(predicted_mean=("predicted_probability", "mean"), actual_rate=(outcome_col, "mean"), n=(outcome_col, "size"))
        .reset_index(names="bin")
    )
    grouped["bin"] = grouped["bin"].astype(str)
    return grouped


def _filter_metric(predictions: pd.DataFrame, metric: str | None) -> pd.DataFrame:
    return predictions if metric is None else predictions[predictions["metric"] == metric]


def _filter_model_version(predictions: pd.DataFrame, model_version: str | None) -> pd.DataFrame:
    """Restricts to rows tagged with `model_version` (see
    predictions.select_picks/config.HITTER_MODEL_VERSION) - None (the
    default everywhere below) means "all versions, unfiltered", preserving
    every existing caller's behavior byte-for-byte. Passing a specific
    version is what lets a real recalibration's effect actually show up in
    these stats instead of being diluted by pre-change history forever."""
    if model_version is None:
        return predictions
    if "model_version" not in predictions.columns:
        return predictions.iloc[0:0]
    return predictions[predictions["model_version"] == model_version]


def _combined_probability(df: pd.DataFrame) -> pd.Series:
    """A blended recommendation score from whichever of
    predicted_probability (Game_Hit_Probability, always present),
    probability (the WAVE-based binomial estimate), and
    Matchup_Hit_Probability (opposing-pitcher-adjusted, only present on
    days schedule/matchup data was available - see predictions.select_picks)
    exist as columns - a row-wise mean, skipping any that are missing for
    that row rather than requiring all three.

    This is the same three signals select_picks already jointly requires to
    clear HITTER_MIN_PROBABILITY at selection time (see
    JOINT_PROBABILITY_GATE_COLUMNS) - gating "recommended" on
    Game_Hit_Probability alone ignored the other two entirely, and produced
    a string of zero-pick days whenever GHP landed just under the bar
    despite `probability`/`Matchup_Hit_Probability` being strong (see
    config.DAILY_PICK_MIN_PROBABILITY)."""
    columns = [c for c in ("predicted_probability", "probability", "Matchup_Hit_Probability") if c in df.columns]
    return df[columns].astype(float).mean(axis=1, skipna=True)


def _recommended_picks(
    predictions: pd.DataFrame,
    metric: str | None,
    max_picks: int,
    min_probability: float,
    model_version: str | None = None,
) -> pd.DataFrame:
    """The subset of logged picks that actually count toward the tracked
    streak/day_survival_rate for a given day: top-ranked, capped at
    `max_picks`, and only those whose _combined_probability clears
    `min_probability` ("a good matchup"). A day can have 0, 1, or
    `max_picks` rows here depending on how many clear the bar - it's never
    padded out to a fixed count. This is the STREAK-COUNTING definition of
    "recommended" only - see graded_daily_picks for what the dashboard
    actually displays, which is a superset of this (every logged
    candidate, not just the ones that clear the bar)."""
    df = _filter_metric(predictions, metric)
    df = _filter_model_version(df, model_version)
    df = df[(df["rank"] <= max_picks) & (_combined_probability(df) >= min_probability)]
    return df


def graded_daily_picks(
    predictions: pd.DataFrame,
    metric: str | None,
    max_picks: int,
    min_probability: float,
    model_version: str | None = None,
) -> pd.DataFrame:
    """Every day's top `max_picks` candidates by rank, ALWAYS returned -
    unlike _recommended_picks, a day is never empty just because nobody
    cleared `min_probability`. Each row gets its own real `combined_probability`
    and a `grade` ("recommended" if that clears `min_probability` - the
    exact same bar/columns _recommended_picks gates on, so a "recommended"
    grade here is precisely what counts toward the tracked streak/
    day_survival_rate, see streak_progression - else "speculative"). A
    "speculative" pick is still a real, already-qualified candidate
    (predictions.select_picks already gated it on HITTER_MIN_PROBABILITY/
    the model shortlist before it was ever logged) - just below the
    backtested confidence bar, shown for visibility rather than hidden.

    Added because DAILY_PICK_MIN_PROBABILITY (0.77) was validated on a
    42-day historical replay where Matchup_Hit_Probability was always NaN
    (never persisted to git history at the time - see that constant's own
    docstring) - once live runs started actually carrying real
    Matchup_Hit_Probability values most days, the blended mean runs lower
    on an ordinary day than that replay ever exercised, and real live data
    hit a 5-day-straight stretch (2026-08-11 through 2026-08-15) where the
    top-ranked candidate's real combined probability landed at 0.71-0.77 -
    just under the bar every single day, producing a blank dashboard
    despite real, qualified candidates existing every one of those days.
    Rather than re-chase a moving threshold, the dashboard now always shows
    its real top candidates, graded honestly, instead of going blank."""
    df = _filter_metric(predictions, metric)
    df = _filter_model_version(df, model_version)
    df = df[df["rank"] <= max_picks].copy()
    df["combined_probability"] = _combined_probability(df)
    df["grade"] = np.where(df["combined_probability"] >= min_probability, "recommended", "speculative")
    return df


def _classify_outcome(df: pd.DataFrame) -> pd.Series:
    """Per-pick outcome: "pending" (at_bats unknown yet), "no_game"
    (confirmed zero at-bats - a rainout, DNP, etc.), "hit", or "miss"."""
    at_bats = pd.to_numeric(df["at_bats"], errors="coerce")
    actual_hit = pd.to_numeric(df["actual_hit"], errors="coerce")

    outcome = pd.Series("pending", index=df.index)
    outcome[at_bats == 0] = "no_game"
    outcome[(at_bats > 0) & (actual_hit == 1)] = "hit"
    outcome[(at_bats > 0) & (actual_hit == 0)] = "miss"
    return outcome


def streak_progression(
    predictions: pd.DataFrame,
    metric: str = "Game_Hit_Probability",
    max_picks: int = 2,
    min_probability: float = 0.0,
    model_version: str | None = None,
) -> pd.DataFrame:
    """Day-by-day Beat the Streak simulation using the real game's actual
    rules, not a simplified win/loss-per-day model:

    - A pick with >=1 at-bat and no hit ("miss") resets the streak to 0,
      no matter what the other pick (if any) did that day.
    - Otherwise the streak increases by however many picks got a hit that
      day (0, 1, or up to max_picks) - a pick with 0 at-bats ("no_game")
      contributes nothing, positive or negative.
    - A day isn't processed at all until every pick logged for it is
      resolved (no "pending" outcomes) - it's simply skipped, not counted
      as a break, until the data catches up.
    - A day with zero recommended picks (no good matchup - see
      _recommended_picks) never appears in the log in the first place, so
      it's implicitly skipped too, which is exactly the desired no-op.

    Returns one row per resolved day, oldest first: date, the running
    streak value after that day, and whether that day reset it.
    """
    df = _recommended_picks(predictions, metric, max_picks, min_probability, model_version).copy()
    df["outcome"] = _classify_outcome(df)

    rows = []
    streak = 0
    for date, day in df.sort_values("date").groupby("date", sort=True):
        if (day["outcome"] == "pending").any():
            continue
        reset = bool((day["outcome"] == "miss").any())
        if reset:
            streak = 0
        else:
            streak += int((day["outcome"] == "hit").sum())
        rows.append({"date": date, "streak": streak, "reset": reset})

    return pd.DataFrame(rows, columns=["date", "streak", "reset"])


def longest_streak(
    predictions: pd.DataFrame,
    metric: str = "Game_Hit_Probability",
    max_picks: int = 2,
    min_probability: float = 0.0,
    model_version: str | None = None,
) -> int:
    progression = streak_progression(predictions, metric, max_picks, min_probability, model_version)
    return int(progression["streak"].max()) if len(progression) else 0


def current_streak(
    predictions: pd.DataFrame,
    metric: str = "Game_Hit_Probability",
    max_picks: int = 2,
    min_probability: float = 0.0,
    model_version: str | None = None,
) -> int:
    """Streak value as of the most recently *resolved* day (a trailing
    run of still-pending or no-pick days doesn't change this)."""
    progression = streak_progression(predictions, metric, max_picks, min_probability, model_version)
    return int(progression["streak"].iloc[-1]) if len(progression) else 0


def build_beat_the_streak_export(
    predictions: pd.DataFrame,
    metric: str = "Game_Hit_Probability",
    max_picks: int = 2,
    min_probability: float = 0.0,
    model_version: str | None = None,
):
    """Build the two tables the dashboard's Beat the Streak section reads:
    (picks_table, summary_row). picks_table is every day's top `max_picks`
    candidates by rank (see graded_daily_picks) with a hit/miss/no_game/
    pending status AND a "recommended"/"speculative" grade, most recent day
    first - unlike before, a day with real logged candidates is NEVER blank
    just because none cleared `min_probability`; it always shows its real
    best options, honestly graded. summary_row has longest_streak/
    current_streak plus a day_survival_rate (fraction of resolved days that
    didn't reset the streak - a looser sanity metric than the streak count
    itself, since a single miss zeroes a long streak) - these are still
    computed from ONLY "recommended"-grade picks (see streak_progression/
    _recommended_picks, unchanged), so a "speculative" day is still a
    real no-op for the tracked streak, exactly like a no_pick day was
    before this function's picks_table stopped going blank on those days.

    A date is only absent from picks_table's real rows (and gets the
    explicit "no_pick" placeholder row instead) when NOTHING was logged for
    it at all - a genuine off day (All-Star break), a rainout across the
    whole slate, or a pipeline gap - not a weak-slate day, which now gets a
    real "speculative" row instead of vanishing.

    `model_version` (default None, i.e. every version blended together -
    unchanged behavior) restricts to picks tagged with a specific
    predictions.select_picks model_version (see config.HITTER_MODEL_VERSION)
    - the summary row's own "model_version" column is set to whatever was
    passed (or "all_time" when None), so pipeline.py can build one small
    CSV covering both views without ambiguity about which row is which."""
    picks = graded_daily_picks(predictions, metric, max_picks, min_probability, model_version).copy()
    picks["status"] = _classify_outcome(picks)
    picks = picks[["date", "rank", "name", "predicted_probability", "combined_probability", "actual_hit", "status", "grade"]]

    # graded_daily_picks already includes every rank<=max_picks candidate
    # regardless of grade, so a date can only be missing from `picks` here
    # when NOTHING was logged for it at all (see docstring above) - surface
    # that explicitly as its own row rather than leaving the date silently
    # absent, which a reader (or the dashboard) would otherwise misread as
    # "the most recent day was some earlier date."
    filtered = _filter_model_version(_filter_metric(predictions, metric), model_version)
    no_pick_dates = sorted(set(filtered["date"]) - set(picks["date"]))
    if no_pick_dates:
        no_pick_rows = pd.DataFrame(
            {
                "date": no_pick_dates,
                "rank": pd.NA,
                "name": pd.NA,
                "predicted_probability": pd.NA,
                "combined_probability": pd.NA,
                "actual_hit": pd.NA,
                "status": "no_pick",
                "grade": pd.NA,
            }
        )
        picks = pd.concat([picks, no_pick_rows], ignore_index=True)

    picks = picks.sort_values(["date", "rank"], ascending=[False, True]).reset_index(drop=True)

    progression = streak_progression(predictions, metric, max_picks, min_probability, model_version)
    n_days = len(progression)
    survival_successes = int((~progression["reset"]).sum()) if n_days else 0
    survival_rate = float(survival_successes / n_days) if n_days else float("nan")
    # Quant-analytics item #5 ("backtest scope and statistical
    # significance"): CI only here, deliberately no binomial_significance
    # p-value - "recommended" picks are already gated on a high
    # min_probability bar, so there's no real symmetric "no skill" null
    # to test survival against the way beat_closing_line_rate has one
    # (see game_evaluation.py's own choice of which metrics get a
    # p-value). The CI still answers the real question a small n_days
    # raises: how much could this rate move with more data.
    survival_ci_low, survival_ci_high = wilson_confidence_interval(survival_successes, n_days)

    summary = pd.DataFrame(
        [
            {
                "model_version": model_version if model_version is not None else "all_time",
                "metric": metric,
                "max_picks": max_picks,
                "min_probability": min_probability,
                "n_days_resolved": n_days,
                "day_survival_rate": survival_rate,
                "day_survival_rate_ci_low": survival_ci_low,
                "day_survival_rate_ci_high": survival_ci_high,
                "longest_streak": int(progression["streak"].max()) if n_days else 0,
                "current_streak": int(progression["streak"].iloc[-1]) if n_days else 0,
            }
        ]
    )
    return picks, summary


def summarize(predictions: pd.DataFrame, top_k_values=(1, 2, 5), model_version: str | None = None) -> pd.DataFrame:
    """One-row-per-metric summary table, split by the `metric` column so
    multiple candidate metrics (e.g. "probability" vs "Game_Hit_Probability")
    logged into the same predictions file can be compared directly.

    `model_version` (default None, i.e. every version blended together -
    unchanged behavior) restricts to picks tagged with a specific
    predictions.select_picks model_version (see config.HITTER_MODEL_VERSION) -
    this is what lets a real recalibration's effect on accuracy/Brier/etc.
    actually show up, instead of being diluted by pre-change history."""
    predictions = _filter_model_version(predictions, model_version)
    rows = []
    for metric_name, group in predictions.groupby("metric"):
        resolved = resolved_only(group)
        row = {
            "metric": metric_name,
            "n_resolved": len(resolved),
            "brier_score": brier_score(group),
            "log_loss": log_loss(group),
        }
        for k in top_k_values:
            row[f"any_of_top_{k}_hit_rate"] = top_k_hit_rate(group, k, require_all=False)
            row[f"all_of_top_{k}_hit_rate"] = top_k_hit_rate(group, k, require_all=True)
        rows.append(row)
    return pd.DataFrame(rows)
