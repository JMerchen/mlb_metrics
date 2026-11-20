"""Anytime-valid evidence for bets (src/mlb_metrics/anytime_valid.py)."""

import numpy as np
import pytest

from mlb_metrics import anytime_valid as av


def test_the_bankroll_grows_on_winners_and_shrinks_on_coin_flips():
    winners = av.at_110([1, 1, 0, 1, 1, 1, 0, 1] * 40)          # 75% at -110
    assert av.wealth_path(winners)[-1] > 1000
    flips = av.at_110([1, 0] * 200)                             # 50% at -110 loses the vig
    path = av.wealth_path(flips)
    assert path[-1] < 0.2
    # A push changes nothing; nothing can lose more than the stake.
    assert av.wealth_path([0.0, 0.0])[-1] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        av.wealth_path([-1.5])
    assert av.evidence([]) == {"bets": 0, "units": 0.0, "wealth": 1.0, "peak": 1.0, "peak_at": None}


def test_with_no_edge_the_bankroll_rarely_reaches_the_bar_however_often_we_look():
    """The whole point: at break-even (52.4% at -110) the chance the
    bankroll EVER reaches 20 over 850 bets is at most 1 in 20. A fixed-size
    test checked after every bet would cry wolf far more often."""
    rng = np.random.default_rng(7)
    ever, peeking = 0, 0
    for _ in range(600):
        won = (rng.random(850) < 110 / 210).astype(float)
        ever += av.wealth_path(av.at_110(won)).max() >= av.EVIDENCE_BAR
        n = np.arange(1, 851)
        z = (np.cumsum(won) - n * 110 / 210) / np.sqrt(n * (110 / 210) * (100 / 210))
        peeking += (z[20:] > 1.645).any()
    assert ever / 600 <= 0.05
    assert peeking / 600 > 0.2


def test_evidence_reports_the_peak_and_when():
    out = av.evidence(av.at_110([1] * 30 + [0] * 30))
    assert out["bets"] == 60 and out["peak_at"] == 29
    assert out["peak"] > out["wealth"]
    assert out["units"] == pytest.approx(30 * av.WIN_AT_110 - 30)


def test_the_range_of_win_rates_holds_the_truth_and_narrows():
    rng = np.random.default_rng(3)
    few = av.confidence_sequence((rng.random(60) < 0.57).astype(float))
    many = av.confidence_sequence((rng.random(3000) < 0.57).astype(float))
    assert few["low"] < 0.57 < few["high"] and many["low"] < 0.57 < many["high"]
    assert many["high"] - many["low"] < 0.5 * (few["high"] - few["low"])
    assert av.confidence_sequence([]) is None and av.confidence_sequence([np.nan]) is None
    # It is honest about being read at every bet: wider than the usual interval.
    usual = 2 * 1.96 * np.sqrt(0.57 * 0.43 / 3000)
    assert many["high"] - many["low"] > usual


def test_the_fade_alarm_trips_after_a_drop_whenever_it_started_and_not_before():
    rng = np.random.default_rng(11)
    good = (rng.random(600) < 0.58).astype(float)
    steady = av.fade_detector(good, 110 / 210)
    assert steady["tripped_at"] is None
    faded = np.concatenate([good, (rng.random(700) < 0.42).astype(float)])
    alarm = av.fade_detector(faded, 110 / 210)
    assert alarm["tripped_at"] is not None and alarm["tripped_at"] >= 600
    assert av.fade_detector([], 0.5)["tripped_at"] is None
