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
