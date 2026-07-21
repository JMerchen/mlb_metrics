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
