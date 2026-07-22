import pandas as pd
import pytest

from mlb_metrics import hitters

LATEST = "2026-06-20"


def _at_bats(batter, throws, rows):
    """rows: list of (game_date, events) tuples."""
    return [
        {"batter": batter, "game_date": pd.Timestamp(date), "events": events, "p_throws": throws}
        for date, events in rows
    ]


def test_compute_wave_blends_windows_and_converts_to_probability():
    # Batter 1 faces only RHP. Counts land in each of the four WAVE windows
    # (full/81d/30d/10d relative to 2026-06-20) with hand-computable rates:
    # full=4/6, 81d=4/5, 30d=3/3, 10d=1/1 -> WAVE_R = 0.9 exactly.
    rows = _at_bats(1, "R", [
        ("2026-03-12", "field_out"),   # only in the full-season window
        ("2026-05-01", "single"),      # in 81d, not 30d
        ("2026-05-01", "field_out"),
        ("2026-05-31", "single"),      # in 30d, not 10d
        ("2026-05-31", "double"),
        ("2026-06-15", "home_run"),    # in 10d
    ])
    # Batter 2 is the mirror image, facing mostly LHP, to check the L side
    # blends the same way independent of R. WAVE is anchored on batters with
    # a full-season at-bat vs RHP (see the dedicated test below), so give
    # batter 2 one token R at-bat, outside every window but "full", so it
    # isn't dropped from the output while still keeping WAVE_R at 0.
    rows += _at_bats(2, "R", [("2026-03-01", "field_out")])
    rows += _at_bats(2, "L", [
        ("2026-03-12", "field_out"),
        ("2026-05-01", "single"),
        ("2026-05-01", "field_out"),
        ("2026-05-31", "single"),
        ("2026-05-31", "double"),
        ("2026-06-15", "home_run"),
    ])
    # Batter 3 has an even split, all-hit vs R and all-out vs L, to isolate
    # the abtl/abtr platoon-share blend from the window blend.
    rows += _at_bats(3, "R", [("2026-06-15", "single"), ("2026-06-15", "double")])
    rows += _at_bats(3, "L", [("2026-06-15", "field_out"), ("2026-06-15", "field_out")])

    dt = pd.DataFrame(rows)
    wave = hitters.compute_wave(dt).set_index("key_mlbam")

    assert wave.loc[1, "WAVE_R"] == pytest.approx(0.9)
    assert wave.loc[1, "WAVE_L"] == 0
    assert wave.loc[1, "WAVE"] == pytest.approx(0.9)
    assert wave.loc[1, "r_at_bat"] == 6
    assert wave.loc[1, "l_at_bat"] == 0
    assert wave.loc[1, "probability"] == pytest.approx(1 - 0.1**3.5)

    assert wave.loc[2, "WAVE_L"] == pytest.approx(0.9)
    assert wave.loc[2, "WAVE_R"] == 0

    # batter 3: WAVE_R = 1.0 (all hits, all windows), WAVE_L = 0.0 (all outs),
    # 2 AB each side -> WAVE = 1.0*0.5 + 0.0*0.5 = 0.5
    assert wave.loc[3, "WAVE_R"] == pytest.approx(1.0)
    assert wave.loc[3, "WAVE_L"] == pytest.approx(0.0)
    assert wave.loc[3, "WAVE"] == pytest.approx(0.5)


def test_wave_excludes_batters_with_no_full_season_at_bats_vs_rhp():
    # Matches the original script's behavior: WAVE is anchored on batters who
    # have at least one full-season at-bat against a right-handed pitcher.
    dt = pd.DataFrame(_at_bats(9, "L", [("2026-06-15", "single")]))
    wave = hitters.compute_wave(dt)
    assert wave.empty


def test_compute_game_hit_probability_blends_game_level_hit_rate():
    rows = [
        {"batter": 1, "game_id": 1, "game_date": pd.Timestamp("2026-03-12"), "events": "field_out"},
        {"batter": 1, "game_id": 2, "game_date": pd.Timestamp("2026-05-01"), "events": "single"},
        {"batter": 1, "game_id": 3, "game_date": pd.Timestamp("2026-05-31"), "events": "single"},
        {"batter": 1, "game_id": 4, "game_date": pd.Timestamp("2026-06-15"), "events": "single"},
    ]
    data_with_game_id = pd.DataFrame(rows)

    result = hitters.compute_game_hit_probability(data_with_game_id).set_index("key_mlbam")

    # full=3/4, 81d=3/3, 30d=2/2, 10d=1/1
    expected = (3 / 4) * 0.175 + 1.0 * 0.225 + 1.0 * 0.275 + 1.0 * 0.325
    assert result.loc[1, "Game_Hit_Probability"] == pytest.approx(expected)


def test_assemble_hitters_output_columns_and_derived_fields():
    dt = pd.DataFrame(_at_bats(1, "R", [("2026-06-15", "single"), ("2026-06-16", "field_out")]))
    data_with_game_id = pd.DataFrame([
        {"batter": 1, "game_id": 1, "game_date": pd.Timestamp("2026-06-15"), "events": "single"},
        {"batter": 1, "game_id": 2, "game_date": pd.Timestamp("2026-06-16"), "events": "field_out"},
    ])
    names = pd.DataFrame([{"key_mlbam": 1, "name_first": "Test", "name_last": "Player"}])
    latest_team = pd.DataFrame([{"key_mlbam": 1, "team": "NYY"}])

    result = hitters.assemble_hitters(dt, data_with_game_id, names, latest_team)

    assert list(result.columns) == [
        "key_mlbam", "name_first", "name_last", "team", "PA_L", "PA_R",
        "WAVE", "probability_L", "probability_R", "probability",
        "Game_Hit_Probability", "Consistency", "Approach", "Expected_Bases",
    ]
    row = result.iloc[0]
    assert row["name_last"] == "Player"
    assert row["Consistency"] == pytest.approx(row["Game_Hit_Probability"] - row["probability"])
    assert row["Approach"] == pytest.approx(row["Game_Hit_Probability"] * row["probability"])


def test_assemble_hitters_merges_lineup_consistency_when_provided():
    dt = pd.DataFrame(_at_bats(1, "R", [("2026-06-15", "single"), ("2026-06-16", "field_out")]))
    data_with_game_id = pd.DataFrame([
        {"batter": 1, "game_id": 1, "game_date": pd.Timestamp("2026-06-15"), "events": "single"},
        {"batter": 1, "game_id": 2, "game_date": pd.Timestamp("2026-06-16"), "events": "field_out"},
    ])
    names = pd.DataFrame([{"key_mlbam": 1, "name_first": "Test", "name_last": "Player"}])
    latest_team = pd.DataFrame([{"key_mlbam": 1, "team": "NYY"}])
    # Batter 1 has a lineup-consistency row; the population itself is
    # untouched by omitting one (a defensive merge, not expected in practice
    # since lineup_consistency is normally computed for the same population).
    lineup_consistency = pd.DataFrame([{"key_mlbam": 1, "avg_batting_order": 2.5, "start_rate": 0.8}])

    result = hitters.assemble_hitters(dt, data_with_game_id, names, latest_team, lineup_consistency)

    assert "avg_batting_order" in result.columns and "start_rate" in result.columns
    row = result.iloc[0]
    assert row["avg_batting_order"] == pytest.approx(2.5)
    assert row["start_rate"] == pytest.approx(0.8)


def test_assemble_hitters_lineup_consistency_missing_batter_is_null_not_zero():
    dt = pd.DataFrame(_at_bats(1, "R", [("2026-06-15", "single"), ("2026-06-16", "field_out")]))
    data_with_game_id = pd.DataFrame([
        {"batter": 1, "game_id": 1, "game_date": pd.Timestamp("2026-06-15"), "events": "single"},
        {"batter": 1, "game_id": 2, "game_date": pd.Timestamp("2026-06-16"), "events": "field_out"},
    ])
    names = pd.DataFrame([{"key_mlbam": 1, "name_first": "Test", "name_last": "Player"}])
    latest_team = pd.DataFrame([{"key_mlbam": 1, "team": "NYY"}])
    lineup_consistency = pd.DataFrame(columns=["key_mlbam", "avg_batting_order", "start_rate"])

    result = hitters.assemble_hitters(dt, data_with_game_id, names, latest_team, lineup_consistency)

    row = result.iloc[0]
    assert pd.isna(row["avg_batting_order"])  # never filled to 0 - that would look like batting 1st
    assert row["start_rate"] == 0
