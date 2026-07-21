import math

import pandas as pd
import pytest

from mlb_metrics import evaluation


def _predictions():
    return pd.DataFrame(
        [
            {"date": "2026-06-18", "rank": 1, "predicted_probability": 0.9, "metric": "m", "actual_hit": 1},
            {"date": "2026-06-18", "rank": 2, "predicted_probability": 0.6, "metric": "m", "actual_hit": 0},
            {"date": "2026-06-19", "rank": 1, "predicted_probability": 0.8, "metric": "m", "actual_hit": 1},
            {"date": "2026-06-19", "rank": 2, "predicted_probability": 0.5, "metric": "m", "actual_hit": 1},
            {"date": "2026-06-20", "rank": 1, "predicted_probability": 0.7, "metric": "m", "actual_hit": 0},
            {"date": "2026-06-20", "rank": 2, "predicted_probability": 0.4, "metric": "m", "actual_hit": 0},
        ]
    )


def test_pick_accuracy_by_rank():
    table = evaluation.pick_accuracy_by_rank(_predictions()).set_index("rank")
    assert table.loc[1, "hit_rate"] == pytest.approx(2 / 3)
    assert table.loc[1, "n"] == 3
    assert table.loc[2, "hit_rate"] == pytest.approx(1 / 3)


def test_top_k_hit_rate_any_vs_all():
    preds = _predictions()
    assert evaluation.top_k_hit_rate(preds, 1) == pytest.approx(2 / 3)
    assert evaluation.top_k_hit_rate(preds, 2, require_all=False) == pytest.approx(2 / 3)
    assert evaluation.top_k_hit_rate(preds, 2, require_all=True) == pytest.approx(1 / 3)


def test_brier_score():
    preds = _predictions()
    expected = sum(
        (p - y) ** 2 for p, y in zip(preds["predicted_probability"], preds["actual_hit"])
    ) / len(preds)
    assert evaluation.brier_score(preds) == pytest.approx(expected)


def test_log_loss():
    preds = _predictions()
    expected = -sum(
        y * math.log(p) + (1 - y) * math.log(1 - p)
        for p, y in zip(preds["predicted_probability"], preds["actual_hit"])
    ) / len(preds)
    assert evaluation.log_loss(preds) == pytest.approx(expected)


def test_unresolved_rows_are_excluded_from_every_metric():
    preds = _predictions()
    pending = pd.DataFrame(
        [{"date": "2026-06-21", "rank": 1, "predicted_probability": 0.99, "metric": "m", "actual_hit": None}]
    )
    with_pending = pd.concat([preds, pending], ignore_index=True)

    assert evaluation.brier_score(with_pending) == pytest.approx(evaluation.brier_score(preds))
    assert evaluation.log_loss(with_pending) == pytest.approx(evaluation.log_loss(preds))
    assert evaluation.top_k_hit_rate(with_pending, 1) == pytest.approx(evaluation.top_k_hit_rate(preds, 1))


def test_calibration_table_covers_every_resolved_row():
    table = evaluation.calibration_table(_predictions(), n_bins=2)
    assert table["n"].sum() == 6


def test_summarize_splits_by_metric():
    preds = _predictions()
    other_metric = preds.copy()
    other_metric["metric"] = "other"

    summary = evaluation.summarize(pd.concat([preds, other_metric], ignore_index=True)).set_index("metric")

    assert set(summary.index) == {"m", "other"}
    assert summary.loc["m", "n_resolved"] == 6
    assert summary.loc["m", "any_of_top_1_hit_rate"] == pytest.approx(2 / 3)


def _pick_pair(date, hit1, hit2, metric="Game_Hit_Probability"):
    return [
        {"date": date, "rank": 1, "name": "A", "predicted_probability": 0.9, "metric": metric, "actual_hit": hit1},
        {"date": date, "rank": 2, "name": "B", "predicted_probability": 0.8, "metric": metric, "actual_hit": hit2},
    ]


def _streak_predictions():
    rows = (
        _pick_pair("2026-06-18", 1, 1)  # both hit
        + _pick_pair("2026-06-19", 1, 1)  # both hit -> streak of 2
        + _pick_pair("2026-06-20", 1, 0)  # one misses -> breaks it
        + _pick_pair("2026-06-21", 1, 1)  # both hit -> new streak of 1
        + _pick_pair("2026-06-22", None, None)  # pending -> excluded, not a break
    )
    # An incomplete day (only rank 1 resolved) - should also be excluded, not
    # counted as a miss.
    rows.append(
        {"date": "2026-06-23", "rank": 1, "predicted_probability": 0.9, "metric": "Game_Hit_Probability", "actual_hit": 1}
    )
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_streak_series_excludes_pending_and_incomplete_days():
    series = evaluation.streak_series(_streak_predictions(), k=2, require_all=True)
    assert list(series["date"].dt.strftime("%Y-%m-%d")) == ["2026-06-18", "2026-06-19", "2026-06-20", "2026-06-21"]
    assert list(series["streak_continues"]) == [True, True, False, True]


def test_longest_and_current_streak():
    preds = _streak_predictions()
    assert evaluation.longest_streak(preds, k=2, require_all=True) == 2
    assert evaluation.current_streak(preds, k=2, require_all=True) == 1


def test_current_streak_is_zero_after_a_break():
    preds = pd.DataFrame(_pick_pair("2026-06-18", 1, 1) + _pick_pair("2026-06-19", 1, 0))
    assert evaluation.current_streak(preds, k=2, require_all=True) == 0
    assert evaluation.longest_streak(preds, k=2, require_all=True) == 1


def test_require_all_false_only_needs_one_pick_to_hit():
    # Day where rank 2 hits but rank 1 doesn't: fails require_all=True but
    # should count under require_all=False ("any of the k").
    preds = pd.DataFrame(_pick_pair("2026-06-18", 0, 1))
    assert evaluation.current_streak(preds, k=2, require_all=True) == 0
    assert evaluation.current_streak(preds, k=2, require_all=False) == 1


def test_build_beat_the_streak_export_picks_table_and_summary():
    preds = _streak_predictions()
    picks, summary = evaluation.build_beat_the_streak_export(preds, k=2, require_all=True)

    # Most recent day first, includes the still-pending and incomplete days.
    assert picks["date"].iloc[0] == picks["date"].max()
    assert set(picks["date"].dt.strftime("%Y-%m-%d")) == {
        "2026-06-18", "2026-06-19", "2026-06-20", "2026-06-21", "2026-06-22", "2026-06-23"
    }
    pending_rows = picks[picks["date"] == "2026-06-22"]
    assert (pending_rows["status"] == "pending").all()
    miss_row = picks[(picks["date"] == "2026-06-20") & (picks["rank"] == 2)].iloc[0]
    assert miss_row["status"] == "miss"

    assert summary.loc[0, "n_days_resolved"] == 4
    assert summary.loc[0, "longest_streak"] == 2
    assert summary.loc[0, "current_streak"] == 1
    assert summary.loc[0, "streak_success_rate"] == pytest.approx(3 / 4)
