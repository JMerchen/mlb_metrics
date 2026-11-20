"""Bayesian online changepoint detection (src/mlb_metrics/changepoint.py)."""

import numpy as np
import pytest

from mlb_metrics import changepoint


def test_a_steady_rate_uses_all_of_its_history_and_flags_nothing():
    rng = np.random.default_rng(0)
    n = rng.integers(20, 34, 16)
    k = rng.binomial(n, 0.55)
    out = changepoint.rate_runs(k, n, hazard=0.1, temper=0.7)
    assert out["mean"][-1] == pytest.approx(k.sum() / n.sum(), abs=0.04)
    assert out["changed"].max() < 0.5
    assert out["run"][-1] >= 12


def test_a_real_change_is_caught_within_a_few_games_and_the_old_level_is_dropped():
    rng = np.random.default_rng(1)
    n = np.full(14, 28)
    k = np.r_[rng.binomial(28, 0.62, 8), rng.binomial(28, 0.2, 6)]
    out = changepoint.rate_runs(k, n, hazard=0.1, temper=0.7)
    assert out["mean"][7] == pytest.approx(0.62, abs=0.08)
    # Three games after the break the estimate has left the old level
    # behind; an all-history average would still be near 0.5.
    assert out["mean"][10] < 0.32
    assert out["changed"][8:11].max() > 0.5
    assert out["mean"][-1] == pytest.approx(0.2, abs=0.07)
    assert out["run"][-1] == 6


def test_tempering_makes_the_detector_slower_to_call_a_change():
    """A team's carries are not independent coin flips; counting each as a
    full chance would see a new role in every odd game."""
    n = np.full(10, 30)
    k = np.array([15, 16, 14, 15, 15, 21, 15, 14, 16, 15])     # one big day
    eager = changepoint.rate_runs(k, n, hazard=0.1, temper=1.5)
    calm = changepoint.rate_runs(k, n, hazard=0.1, temper=0.2)
    assert eager["mean"][5] - eager["mean"][4] > calm["mean"][5] - calm["mean"][4]
    assert calm["changed"].max() < eager["changed"].max()


def test_a_new_run_starts_from_the_players_own_average():
    # Before any games the estimate is the prior; after one it has moved
    # toward what he did, more with a weaker prior.
    weak = changepoint.rate_runs([3], [10], hazard=0.05, prior_mean=0.25, prior_strength=2)
    strong = changepoint.rate_runs([3], [10], hazard=0.05, prior_mean=0.25, prior_strength=400)
    assert 0.25 < strong["mean"][0] < weak["mean"][0] < 0.3
    assert changepoint.rate_runs([], [], hazard=0.1)["mean"].size == 0


def test_a_measured_level_follows_a_drop_and_ignores_one_odd_game():
    rng = np.random.default_rng(2)
    speed = np.r_[rng.normal(95.2, 0.4, 9), rng.normal(93.4, 0.4, 6)]
    out = changepoint.level_runs(speed, hazard=0.2)
    assert out["mean"][8] == pytest.approx(95.2, abs=0.3)
    assert out["mean"][11] < 94.0                    # three games after the drop
    assert out["changed"][9:12].max() > 0.5
    # One slow night is not a change: the next estimate barely moves.
    odd = np.r_[rng.normal(95.2, 0.3, 9), [93.5], rng.normal(95.2, 0.3, 4)]
    back = changepoint.level_runs(odd, hazard=0.2)
    assert back["mean"][-1] == pytest.approx(95.2, abs=0.35)
    assert changepoint.level_runs([], hazard=0.2)["mean"].size == 0
