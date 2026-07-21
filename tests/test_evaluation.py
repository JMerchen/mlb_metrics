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


def _pick(date, rank, predicted_probability, at_bats, actual_hit, metric="Game_Hit_Probability", name="Player"):
    return {
        "date": date, "rank": rank, "name": name, "predicted_probability": predicted_probability,
        "metric": metric, "actual_hit": actual_hit, "at_bats": at_bats,
    }


def _streak_predictions():
    """Exercises every real Beat the Streak rule from the miss/hit/no_game/
    pending model, with min_probability disabled (0.0) so every logged row
    counts - this fixture is about the cumulative-count/reset logic itself,
    not the recommendation-threshold gating (see the dedicated test below)."""
    rows = [
        _pick("2026-06-18", 1, 0.9, 4, 1),   # hit
        _pick("2026-06-18", 2, 0.85, 3, 1),  # hit -> day adds 2 -> streak=2
        _pick("2026-06-19", 1, 0.88, 3, 1),  # hit
        _pick("2026-06-19", 2, 0.82, 0, None),  # no_game (0 at-bats) -> neutral -> day adds 1 -> streak=3
        _pick("2026-06-20", 1, 0.87, 3, 1),  # hit
        _pick("2026-06-20", 2, 0.81, 2, 0),  # miss -> resets the whole day -> streak=0
        _pick("2026-06-21", 1, 0.90, 3, 1),  # single pick, hit -> streak=1
        _pick("2026-06-22", 1, 0.90, None, None),  # pending
        _pick("2026-06-22", 2, 0.85, None, None),  # pending -> whole day skipped, not a break
        _pick("2026-06-23", 1, 0.90, 1, 1),  # single pick, hit -> continues from 06-21 (06-22 skipped) -> streak=2
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_streak_progression_follows_beat_the_streak_rules():
    progression = evaluation.streak_progression(_streak_predictions(), min_probability=0.0)

    assert list(progression["date"].dt.strftime("%Y-%m-%d")) == [
        "2026-06-18", "2026-06-19", "2026-06-20", "2026-06-21", "2026-06-23",
    ]  # 06-22 (all pending) never appears
    assert list(progression["streak"]) == [2, 3, 0, 1, 2]
    assert list(progression["reset"]) == [False, False, True, False, False]


def test_longest_and_current_streak():
    preds = _streak_predictions()
    assert evaluation.longest_streak(preds, min_probability=0.0) == 3
    assert evaluation.current_streak(preds, min_probability=0.0) == 2


def test_a_miss_resets_the_streak_regardless_of_the_other_pick():
    preds = pd.DataFrame(
        [
            _pick("2026-06-18", 1, 0.9, 3, 1),
            _pick("2026-06-18", 2, 0.85, 3, 1),
            _pick("2026-06-19", 1, 0.9, 3, 1),   # hit
            _pick("2026-06-19", 2, 0.85, 2, 0),  # miss -> resets despite the hit
        ]
    )
    assert evaluation.current_streak(preds, min_probability=0.0) == 0
    assert evaluation.longest_streak(preds, min_probability=0.0) == 2


def test_zero_at_bats_pick_is_neutral_not_a_break():
    preds = pd.DataFrame(
        [
            _pick("2026-06-18", 1, 0.9, 3, 1),
            _pick("2026-06-18", 2, 0.85, 0, None),  # no_game
        ]
    )
    # Should behave exactly like a single-pick day that hit: +1, no reset.
    assert evaluation.current_streak(preds, min_probability=0.0) == 1
    assert evaluation.longest_streak(preds, min_probability=0.0) == 1


def test_recommended_picks_gated_by_threshold_can_be_zero_one_or_two():
    preds = pd.DataFrame(
        [
            # Day A: both clear the bar -> 2 recommended.
            _pick("2026-06-18", 1, 0.90, 3, 1),
            _pick("2026-06-18", 2, 0.85, 3, 1),
            # Day B: only rank 1 clears it -> 1 recommended.
            _pick("2026-06-19", 1, 0.90, 3, 1),
            _pick("2026-06-19", 2, 0.60, 3, 0),
            # Day C: neither clears it -> 0 recommended, day is a no-op.
            _pick("2026-06-20", 1, 0.70, 3, 0),
            _pick("2026-06-20", 2, 0.65, 3, 1),
        ]
    )

    picks, summary = evaluation.build_beat_the_streak_export(preds, max_picks=2, min_probability=0.80)

    assert set(picks[picks["date"] == "2026-06-18"]["rank"]) == {1, 2}
    assert set(picks[picks["date"] == "2026-06-19"]["rank"]) == {1}
    assert picks[picks["date"] == "2026-06-20"].empty

    # Day C never enters the streak at all (not even as a no-op skip in the
    # progression table), and day A+B both hit -> 2 + 1 = 3.
    assert summary.loc[0, "current_streak"] == 3
    assert summary.loc[0, "longest_streak"] == 3
    assert summary.loc[0, "n_days_resolved"] == 2


def test_build_beat_the_streak_export_picks_table_status_and_summary():
    preds = _streak_predictions()
    picks, summary = evaluation.build_beat_the_streak_export(preds, min_probability=0.0)

    # Most recent day first, includes pending rows.
    assert picks["date"].iloc[0] == picks["date"].max()
    pending_rows = picks[picks["date"] == "2026-06-22"]
    assert (pending_rows["status"] == "pending").all()
    no_game_row = picks[(picks["date"] == "2026-06-19") & (picks["rank"] == 2)].iloc[0]
    assert no_game_row["status"] == "no_game"
    miss_row = picks[(picks["date"] == "2026-06-20") & (picks["rank"] == 2)].iloc[0]
    assert miss_row["status"] == "miss"
    hit_row = picks[(picks["date"] == "2026-06-18") & (picks["rank"] == 1)].iloc[0]
    assert hit_row["status"] == "hit"

    assert summary.loc[0, "n_days_resolved"] == 5
    assert summary.loc[0, "longest_streak"] == 3
    assert summary.loc[0, "current_streak"] == 2
    assert summary.loc[0, "day_survival_rate"] == pytest.approx(4 / 5)  # 4 of 5 resolved days didn't reset
