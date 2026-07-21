import pandas as pd

from mlb_metrics import helpers

EVENTS = pd.Series(
    [
        "single", "double", "triple", "home_run",
        "walk", "hit_by_pitch", "strikeout", "strikeout_double_play",
        "field_out", "force_out", "grounded_into_double_play",
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
