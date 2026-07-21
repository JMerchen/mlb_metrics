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


def _recommended_picks(
    predictions: pd.DataFrame,
    metric: str | None,
    max_picks: int,
    min_probability: float,
) -> pd.DataFrame:
    """The subset of logged picks that actually count as "recommended" for
    a given day: top-ranked, capped at `max_picks`, and only those that
    clear `min_probability` ("a good matchup"). A day can have 0, 1, or
    `max_picks` rows here depending on how many clear the bar - it's never
    padded out to a fixed count."""
    df = _filter_metric(predictions, metric)
    df = df[(df["rank"] <= max_picks) & (df["predicted_probability"] >= min_probability)]
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
    df = _recommended_picks(predictions, metric, max_picks, min_probability).copy()
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
) -> int:
    progression = streak_progression(predictions, metric, max_picks, min_probability)
    return int(progression["streak"].max()) if len(progression) else 0


def current_streak(
    predictions: pd.DataFrame,
    metric: str = "Game_Hit_Probability",
    max_picks: int = 2,
    min_probability: float = 0.0,
) -> int:
    """Streak value as of the most recently *resolved* day (a trailing
    run of still-pending or no-pick days doesn't change this)."""
    progression = streak_progression(predictions, metric, max_picks, min_probability)
    return int(progression["streak"].iloc[-1]) if len(progression) else 0


def build_beat_the_streak_export(
    predictions: pd.DataFrame,
    metric: str = "Game_Hit_Probability",
    max_picks: int = 2,
    min_probability: float = 0.0,
):
    """Build the two tables the dashboard's Beat the Streak section reads:
    (picks_table, summary_row). picks_table is every *recommended* pick
    (see _recommended_picks - not necessarily a fixed count per day) with a
    hit/miss/no_game/pending status, most recent day first; summary_row has
    longest_streak/current_streak plus a day_survival_rate (fraction of
    resolved days that didn't reset the streak - a looser sanity metric than
    the streak count itself, since a single miss zeroes a long streak)."""
    picks = _recommended_picks(predictions, metric, max_picks, min_probability).copy()
    picks["status"] = _classify_outcome(picks)
    picks = picks[["date", "rank", "name", "predicted_probability", "actual_hit", "status"]]
    picks = picks.sort_values(["date", "rank"], ascending=[False, True]).reset_index(drop=True)

    progression = streak_progression(predictions, metric, max_picks, min_probability)
    n_days = len(progression)
    survival_rate = float((~progression["reset"]).mean()) if n_days else float("nan")

    summary = pd.DataFrame(
        [
            {
                "metric": metric,
                "max_picks": max_picks,
                "min_probability": min_probability,
                "n_days_resolved": n_days,
                "day_survival_rate": survival_rate,
                "longest_streak": int(progression["streak"].max()) if n_days else 0,
                "current_streak": int(progression["streak"].iloc[-1]) if n_days else 0,
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
