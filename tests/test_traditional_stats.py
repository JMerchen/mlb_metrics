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


def _pitcher_events(pitcher, events):
    return [{"pitcher": pitcher, "game_date": pd.Timestamp("2026-06-01"), "events": e} for e in events]


def test_compute_traditional_pitching_stats_exact_arithmetic():
    # Outs: strikeout(1) + strikeout(1) + field_out(1) + grounded_into_double_play(2) = 5 -> IP = 5/3.
    # SO=2, BB=1, HBP=1, HR=1.
    rows = _pitcher_events(1, [
        "strikeout", "strikeout", "field_out", "grounded_into_double_play",
        "walk", "home_run", "hit_by_pitch",
    ])
    pdf = pd.DataFrame(rows)

    result = traditional_stats.compute_traditional_pitching_stats(pdf).set_index("key_mlbam")

    assert result.loc[1, "IP"] == pytest.approx(5 / 3)
    assert result.loc[1, "K9"] == pytest.approx(2 * 9 / (5 / 3))
    assert result.loc[1, "BB9"] == pytest.approx(1 * 9 / (5 / 3))
    assert result.loc[1, "HR9"] == pytest.approx(1 * 9 / (5 / 3))
    # FIP = (13*1 + 3*(1+1) - 2*2) / (5/3) + 3.10 = (13+6-4)/(5/3) + 3.10 = 9.0 + 3.10.
    assert result.loc[1, "FIP"] == pytest.approx(9.0 + 3.10)


def test_compute_traditional_pitching_stats_min_ip_filters():
    rows = _pitcher_events(1, ["field_out"]) + _pitcher_events(2, ["field_out"] * 60)
    pdf = pd.DataFrame(rows)

    result = traditional_stats.compute_traditional_pitching_stats(pdf, min_ip=10)

    assert set(result["key_mlbam"]) == {2}


def test_compute_traditional_pitching_stats_zero_ip_does_not_divide_by_zero():
    rows = _pitcher_events(1, ["single", "walk"])  # 0 outs recorded
    pdf = pd.DataFrame(rows)

    result = traditional_stats.compute_traditional_pitching_stats(pdf).set_index("key_mlbam")

    assert result.loc[1, "IP"] == 0
    assert result.loc[1, "K9"] == 0
    assert result.loc[1, "FIP"] == 0
