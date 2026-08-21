import pandas as pd
import pytest

from mlb_metrics import decision_theory


def test_solve_reservation_thresholds_one_day_horizon_hand_computed():
    # 1-day horizon, streak=0 (k=0), gain=1, two equally-likely real p
    # draws: 0.3 and 0.9. Hand-computed:
    #   V_0(0)=0, V_0(1)=1 (terminal utility = streak length itself)
    #   threshold[1][0] = (V_0(0)-V_0(0)) / (V_0(1)-V_0(0)) = 0/1 = 0
    #   p=0.3: play_value = 0.3*1 + 0.7*0 = 0.3; max(sit=0, 0.3) = 0.3
    #   p=0.9: play_value = 0.9*1 + 0.1*0 = 0.9; max(sit=0, 0.9) = 0.9
    #   V_1(0) = (0.3 + 0.9) / 2 = 0.6
    solved = decision_theory.solve_reservation_thresholds([0.3, 0.9], horizon=1, gain=1.0)

    assert solved["V"][1][0] == pytest.approx(0.6)
    assert solved["thresholds"][1][0] == pytest.approx(0.0)
    # Both real p draws clear threshold=0, so PLAY is optimal at k=0 for
    # either one - a real, non-trivial confirmation, not a tautology.
    assert decision_theory.should_play(streak=0, days_remaining=1, todays_probability=0.3, solved=solved) is True
    assert decision_theory.should_play(streak=0, days_remaining=1, todays_probability=0.9, solved=solved) is True


def test_should_play_protects_an_established_streak_against_a_mediocre_pick():
    # Streak=5, 1 day remaining, gain=1, only a mediocre p=0.3 available.
    # Hand-computed (solved with horizon=5 so k=5 is within the queryable
    # range, but we only read the r=1 slice - terminal V_0 doesn't depend
    # on horizon size):
    #   V_0(5)=5, V_0(6)=6, V_0(0)=0
    #   threshold[1][5] = (5-0)/(6-0) = 5/6 = 0.8333...
    #   play_value at p=0.3 = 0.3*6 + 0.7*0 = 1.8 < sit_value=5 -> SIT wins
    solved = decision_theory.solve_reservation_thresholds([0.3], horizon=5, gain=1.0)

    assert solved["thresholds"][1][5] == pytest.approx(5 / 6)
    assert solved["V"][1][5] == pytest.approx(5.0)  # SIT's value carries through - real protective behavior
    # A real, meaningful result: even though p=0.3 > 0 (there IS some real
    # chance of extending the streak), it's not good enough to risk 5
    # already-banked picks with only one day left to recover.
    assert decision_theory.should_play(streak=5, days_remaining=1, todays_probability=0.3, solved=solved) is False


def test_reservation_threshold_is_non_decreasing_in_streak():
    solved = decision_theory.solve_reservation_thresholds([0.3], horizon=5, gain=1.0)
    thresholds_at_r1 = [solved["thresholds"][1][k] for k in range(6)]
    # Hand-computable: threshold[1][k] = k / (k+1) for this V_0(s)=s terminal
    # utility - strictly increasing. A longer streak needs a strictly
    # better pick to justify risking it.
    for k, threshold in enumerate(thresholds_at_r1):
        assert threshold == pytest.approx(k / (k + 1))
    assert thresholds_at_r1 == sorted(thresholds_at_r1)


def test_solve_reservation_thresholds_rejects_empty_p_samples():
    with pytest.raises(ValueError):
        decision_theory.solve_reservation_thresholds([], horizon=5, gain=1.0)


def test_solve_reservation_thresholds_rejects_non_positive_gain():
    with pytest.raises(ValueError):
        decision_theory.solve_reservation_thresholds([0.5], horizon=5, gain=0.0)


def test_should_play_out_of_range_streak_raises_rather_than_silently_wrong():
    solved = decision_theory.solve_reservation_thresholds([0.5], horizon=3, gain=1.0)
    with pytest.raises(ValueError):
        decision_theory.should_play(streak=100, days_remaining=1, todays_probability=0.9, solved=solved)


def test_estimate_gain_averages_only_non_reset_days_increment():
    # Day 1: streak 0->2 (a real hit day, gain of 2 that day).
    # Day 2: reset to 0 (a miss - contributes nothing, not a negative gain).
    # Day 3: streak 0->1 (a real hit day, gain of 1 that day).
    progression = pd.DataFrame([
        {"date": "2026-06-01", "streak": 2, "reset": False},
        {"date": "2026-06-02", "streak": 0, "reset": True},
        {"date": "2026-06-03", "streak": 1, "reset": False},
    ])
    # Real average of the two non-reset increments: (2 + 1) / 2 = 1.5
    assert decision_theory.estimate_gain(progression) == pytest.approx(1.5)


def test_estimate_gain_no_history_returns_zero_not_a_crash():
    assert decision_theory.estimate_gain(pd.DataFrame(columns=["date", "streak", "reset"])) == 0.0
