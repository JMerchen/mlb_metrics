import pandas as pd
import pytest

from mlb_metrics import traditional_stats


def _events(batter, events):
    return [{"batter": batter, "game_date": pd.Timestamp("2026-06-01"), "events": e, "p_throws": "R"} for e in events]


def test_compute_traditional_batting_stats_exact_arithmetic():
    # 5 AB (single, double, strikeout, field_out, field_out), 1 walk.
    # H=2, AB=5, TB = 1(single) + 2(double) = 3, OB = 2 hits + 1 walk = 3, PA = 6.
    rows = _events(1, ["single", "double", "strikeout", "field_out", "field_out", "walk"])
    dt = pd.DataFrame(rows)

    result = traditional_stats.compute_traditional_batting_stats(dt).set_index("key_mlbam")

    assert result.loc[1, "PA"] == 6
    assert result.loc[1, "AB"] == 5
    assert result.loc[1, "AVG"] == pytest.approx(2 / 5)
    assert result.loc[1, "OBP"] == pytest.approx(3 / 6)
    assert result.loc[1, "SLG"] == pytest.approx(3 / 5)
    assert result.loc[1, "OPS"] == pytest.approx(3 / 6 + 3 / 5)


def test_compute_traditional_batting_stats_min_at_bats_filters():
    rows = _events(1, ["single"]) + _events(2, ["single"] * 5)
    dt = pd.DataFrame(rows)

    result = traditional_stats.compute_traditional_batting_stats(dt, min_at_bats=3)

    assert set(result["key_mlbam"]) == {2}


def test_compute_traditional_batting_stats_zero_at_bats_does_not_divide_by_zero():
    rows = _events(1, ["walk", "hit_by_pitch"])  # 0 official at-bats
    dt = pd.DataFrame(rows)

    result = traditional_stats.compute_traditional_batting_stats(dt).set_index("key_mlbam")

    assert result.loc[1, "AB"] == 0
    assert result.loc[1, "AVG"] == 0
    assert result.loc[1, "SLG"] == 0
    assert result.loc[1, "OBP"] == pytest.approx(2 / 2)  # 2 on-base events / 2 PA
