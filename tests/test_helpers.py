import pandas as pd
import pytest

from mlb_metrics import helpers

EVENTS = pd.Series(
    [
        "single", "double", "triple", "home_run",
        "walk", "hit_by_pitch", "strikeout", "strikeout_double_play",
        "field_out", "force_out", "grounded_into_double_play",
    ]
)

OUTS_EVENTS = pd.Series(
    [
        "field_out", "force_out", "strikeout", "fielders_choice_out",
        "grounded_into_double_play", "double_play", "strikeout_double_play",
        "fielders_choice", "field_error", "single", "walk",
    ]
)


def test_is_hit():
    assert helpers.is_hit(EVENTS).tolist() == [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]


def test_total_bases():
    assert helpers.total_bases(EVENTS).tolist() == [1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0]


def test_is_strikeout_walk_hbp():
    assert helpers.is_strikeout_walk_hbp(EVENTS).tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]


def test_is_home_run():
    assert helpers.is_home_run(EVENTS).tolist() == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]


def test_is_on_base():
    assert helpers.is_on_base(EVENTS).tolist() == [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]


def test_is_official_at_bat():
    assert helpers.is_official_at_bat(EVENTS).tolist() == [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1]


def test_is_strikeout():
    assert helpers.is_strikeout(EVENTS).tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]


def test_is_walk():
    assert helpers.is_walk(EVENTS).tolist() == [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]


def test_is_hit_by_pitch():
    assert helpers.is_hit_by_pitch(EVENTS).tolist() == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]


def test_is_walk_for_dk_scoring_counts_intentional_walks_too():
    events = pd.Series(["walk", "intent_walk", "single", "strikeout"])
    assert helpers.is_walk_for_dk_scoring(events).tolist() == [1, 1, 0, 0]


def test_estimate_rbi_exact_score_delta_clipped_at_zero():
    df = pd.DataFrame({
        "bat_score": [0, 2, 5],
        "post_bat_score": [2, 2, 4],  # last row: a data artifact, must clip to 0 not go negative
    })
    assert helpers.estimate_rbi(df).tolist() == [2, 0, 0]


def test_outs_recorded():
    assert helpers.outs_recorded(OUTS_EVENTS).tolist() == [1, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0]


def test_shrink_rate_zero_strength_is_exact_unshrunk_rate():
    count = pd.Series([3, 30, 0])
    n = pd.Series([10, 100, 5])
    result = helpers.shrink_rate(count, n, prior_rate=0.25, prior_strength=0.0)
    assert result.tolist() == [0.3, 0.3, 0.0]


def test_shrink_rate_zero_over_zero_is_nan_not_a_crash():
    result = helpers.shrink_rate(pd.Series([0]), pd.Series([0]), prior_rate=0.25, prior_strength=0.0)
    assert pd.isna(result.iloc[0])


def test_shrink_rate_pulls_low_n_hard_toward_prior_and_high_n_barely():
    # Same raw rate (0.50) at two very different sample sizes; same prior (0.25), same strength.
    low_n = helpers.shrink_rate(pd.Series([5]), pd.Series([10]), prior_rate=0.25, prior_strength=20.0)
    high_n = helpers.shrink_rate(pd.Series([250]), pd.Series([500]), prior_rate=0.25, prior_strength=20.0)
    # Hand-computed: low_n = (5 + 20*0.25) / (10 + 20) = 10/30 = 0.3333...
    #                high_n = (250 + 20*0.25) / (500 + 20) = 255/520 = 0.4904...
    assert low_n.iloc[0] == pytest.approx(10 / 30)
    assert high_n.iloc[0] == pytest.approx(255 / 520)
    # Both started at the same raw 0.50 rate; the low-n player got pulled much
    # further toward the 0.25 prior than the high-n player did.
    assert abs(low_n.iloc[0] - 0.25) < abs(high_n.iloc[0] - 0.25)
    assert (0.50 - low_n.iloc[0]) > (0.50 - high_n.iloc[0])
