"""Backtest scoring: turns a resolved predictions log (from predictions.py)
into the numbers that actually answer "does this beat a coin flip, let
alone Beat the Streak" - hit rate by pick rank, calibration, Brier score,
and log loss.
"""

import numpy as np
import pandas as pd


def resolved_only(predictions: pd.DataFrame) -> pd.DataFrame:
    """Rows with a known actual_hit outcome (0/1), i.e. the game has been played."""
    resolved = predictions[predictions["actual_hit"].notna()].copy()
    resolved["actual_hit"] = resolved["actual_hit"].astype(float)
    return resolved


def pick_accuracy_by_rank(predictions: pd.DataFrame) -> pd.DataFrame:
    """Hit rate for each individual pick rank (1st-ranked pick, 2nd-ranked,
    ...), independent of the others. If the model has any skill, this should
    decrease as rank increases; if it's flat, the ranking isn't doing anything."""
    resolved = resolved_only(predictions)
    if resolved.empty:
        return pd.DataFrame(columns=["rank", "hit_rate", "n"])
    grouped = resolved.groupby("rank")["actual_hit"].agg(hit_rate="mean", n="count").reset_index()
    return grouped.sort_values("rank").reset_index(drop=True)


def top_k_hit_rate(predictions: pd.DataFrame, k: int, require_all: bool = False) -> float:
    """Per-day rate of success using the top `k` picks. require_all=False
    (default) scores a day as a "hit" if *any* of the top-k picks got a hit
    (a "pick k, need one" strategy); require_all=True scores a day as a hit
    only if *all* k did (Beat the Streak's actual multi-pick mode, where
    every pick must land to extend the streak)."""
    resolved = resolved_only(predictions)
    picks = resolved[resolved["rank"] <= k]
    if picks.empty:
        return float("nan")
    per_day = picks.groupby("date")["actual_hit"]
    outcome = per_day.min() if require_all else per_day.max()
    return float(outcome.mean())


def brier_score(predictions: pd.DataFrame) -> float:
    """Mean squared error between predicted probability and actual (0/1)
    outcome - lower is better, 0 is perfect, 0.25 is what an uninformative
    always-predict-0.5 model scores."""
    resolved = resolved_only(predictions)
    if resolved.empty:
        return float("nan")
    return float(np.mean((resolved["predicted_probability"] - resolved["actual_hit"]) ** 2))


def log_loss(predictions: pd.DataFrame, eps: float = 1e-6) -> float:
    resolved = resolved_only(predictions)
    if resolved.empty:
        return float("nan")
    p = resolved["predicted_probability"].clip(eps, 1 - eps)
    y = resolved["actual_hit"]
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_table(predictions: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Bins picks by predicted probability and compares each bin's mean
    predicted probability to its actual hit rate - a well-calibrated metric
    should have predicted_mean ~= actual_rate in every bin."""
    resolved = resolved_only(predictions)
    if resolved.empty:
        return pd.DataFrame(columns=["bin", "predicted_mean", "actual_rate", "n"])

    bins = pd.cut(resolved["predicted_probability"], bins=n_bins, include_lowest=True)
    grouped = (
        resolved.groupby(bins, observed=True)
        .agg(predicted_mean=("predicted_probability", "mean"), actual_rate=("actual_hit", "mean"), n=("actual_hit", "size"))
        .reset_index(names="bin")
    )
    grouped["bin"] = grouped["bin"].astype(str)
    return grouped


def _filter_metric(predictions: pd.DataFrame, metric: str | None) -> pd.DataFrame:
    return predictions if metric is None else predictions[predictions["metric"] == metric]


def streak_series(predictions: pd.DataFrame, k: int, require_all: bool = True, metric: str | None = None) -> pd.DataFrame:
    """One row per day that has `k` *resolved* picks at rank <= k for
    `metric`, sorted oldest-first, with whether that day's picks satisfy the
    streak condition (all k hit if require_all, else any of the k). Days
    with fewer than k resolved picks (a pending day, or a day too early in
    the season to have k qualified batters) are left out rather than counted
    as a break - they simply haven't happened/resolved yet."""
    top_k = _filter_metric(predictions, metric)
    top_k = top_k[top_k["rank"] <= k]
    resolved = resolved_only(top_k)

    complete_days = resolved.groupby("date").size()
    complete_days = complete_days[complete_days >= k].index
    resolved = resolved[resolved["date"].isin(complete_days)]
    if resolved.empty:
        return pd.DataFrame(columns=["date", "streak_continues"])

    per_day = resolved.groupby("date")["actual_hit"]
    outcome = (per_day.min() if require_all else per_day.max()).astype(bool)
    return outcome.rename("streak_continues").reset_index().sort_values("date").reset_index(drop=True)


def longest_streak(predictions: pd.DataFrame, k: int, require_all: bool = True, metric: str | None = None) -> int:
    """Longest run of consecutive resolved pick-days (not necessarily
    consecutive calendar dates - off-days don't break a streak) satisfying
    the streak condition."""
    series = streak_series(predictions, k, require_all, metric)
    longest = current = 0
    for continues in series["streak_continues"]:
        current = current + 1 if continues else 0
        longest = max(longest, current)
    return longest


def current_streak(predictions: pd.DataFrame, k: int, require_all: bool = True, metric: str | None = None) -> int:
    """Streak length as of the most recent resolved pick-day (0 if that day broke it)."""
    series = streak_series(predictions, k, require_all, metric)
    current = 0
    for continues in series["streak_continues"]:
        current = current + 1 if continues else 0
    return current


def build_beat_the_streak_export(
    predictions: pd.DataFrame,
    k: int = 2,
    require_all: bool = True,
    metric: str = "Game_Hit_Probability",
):
    """Build the two tables the dashboard's Beat the Streak section reads:
    (picks_table, summary_row). picks_table is every logged rank<=k pick
    (most recent day first) with a hit/miss/pending status; summary_row is
    one row of streak_success_rate/longest_streak/current_streak."""
    picks = _filter_metric(predictions, metric)
    picks = picks[picks["rank"] <= k].copy()
    picks["status"] = picks["actual_hit"].map({1.0: "hit", 0.0: "miss"}).fillna("pending")
    picks = picks[["date", "rank", "name", "predicted_probability", "actual_hit", "status"]]
    picks = picks.sort_values(["date", "rank"], ascending=[False, True]).reset_index(drop=True)

    series = streak_series(predictions, k, require_all, metric)
    n_days = len(series)
    success_rate = float(series["streak_continues"].mean()) if n_days else float("nan")

    summary = pd.DataFrame(
        [
            {
                "k": k,
                "require_all": require_all,
                "metric": metric,
                "n_days_resolved": n_days,
                "streak_success_rate": success_rate,
                "longest_streak": longest_streak(predictions, k, require_all, metric),
                "current_streak": current_streak(predictions, k, require_all, metric),
            }
        ]
    )
    return picks, summary


def summarize(predictions: pd.DataFrame, top_k_values=(1, 2, 5)) -> pd.DataFrame:
    """One-row-per-metric summary table, split by the `metric` column so
    multiple candidate metrics (e.g. "probability" vs "Game_Hit_Probability")
    logged into the same predictions file can be compared directly."""
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
