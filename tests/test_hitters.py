import pandas as pd
import pytest

from mlb_metrics import hitters

LATEST = "2026-06-20"


def _at_bats(batter, throws, rows):
    """rows: list of (game_date, events) tuples, or (game_date, events,
    bat_score, post_bat_score) 4-tuples when a row needs a real RBI delta -
    bat_score/post_bat_score default to (0, 0) (Expected_RBI == 0) when
    omitted, so existing 2-tuple call sites are unaffected."""
    result = []
    for row in rows:
        if len(row) == 4:
            date, events, bat_score, post_bat_score = row
        else:
            date, events = row
            bat_score, post_bat_score = 0, 0
        result.append({
            "batter": batter, "game_date": pd.Timestamp(date), "events": events, "p_throws": throws,
            "bat_score": bat_score, "post_bat_score": post_bat_score,
        })
    return result


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


def test_compute_extended_dk_rates_exact_arithmetic_all_windows():
    # All 5 PAs land within 7 days of LATEST (2026-06-20), so they're
    # included in every one of config.WHOPS_WTB_WINDOWS (7d/15d/30d/full) -
    # weights sum to 1.0, so the blended rate equals the raw rate exactly,
    # same technique the WAVE window test above uses.
    rows = _at_bats(1, "R", [
        ("2026-06-15", "walk"),
        ("2026-06-16", "hit_by_pitch"),
        ("2026-06-17", "single", 2, 3),   # scores 1 run -> 1 RBI
        ("2026-06-18", "field_out"),
        ("2026-06-19", "field_out"),
    ])
    dt = pd.DataFrame(rows)

    result = hitters.compute_extended_dk_rates(dt).set_index("key_mlbam")

    from mlb_metrics import config
    expected_rate = 1 / 5
    assert result.loc[1, "Expected_BB"] == pytest.approx(expected_rate * config.WTB_TRIALS_PER_GAME)
    assert result.loc[1, "Expected_HBP"] == pytest.approx(expected_rate * config.WTB_TRIALS_PER_GAME)
    assert result.loc[1, "Expected_RBI"] == pytest.approx(expected_rate * config.WTB_TRIALS_PER_GAME)


def test_compute_extended_dk_rates_blends_platoon_share():
    # Batter faces both sides with different BB rates - isolate the
    # abtl/abtr platoon blend the same way test_compute_wave's batter 3 does.
    rows = _at_bats(1, "R", [("2026-06-19", "walk"), ("2026-06-19", "field_out")])  # 1/2 BB vs R
    rows += _at_bats(1, "L", [("2026-06-19", "field_out"), ("2026-06-19", "field_out")])  # 0/2 BB vs L
    dt = pd.DataFrame(rows)

    from mlb_metrics import config
    result = hitters.compute_extended_dk_rates(dt).set_index("key_mlbam")

    # 2 PA each side -> abtr = abtl = 0.5; blended BB rate = 0.5*0.5 + 0*0.5 = 0.25
    assert result.loc[1, "Expected_BB"] == pytest.approx(0.25 * config.WTB_TRIALS_PER_GAME)


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


def test_compute_current_hit_streaks_counts_trailing_hits_only():
    # hit, miss, hit, hit (most recent first when read backward) - only the
    # trailing 2 straight hits count; the miss two games back breaks it, so
    # the earlier hit before that miss must NOT be added back in.
    rows = [
        {"batter": 1, "game_id": 1, "game_date": pd.Timestamp("2026-06-10"), "events": "single"},
        {"batter": 1, "game_id": 2, "game_date": pd.Timestamp("2026-06-11"), "events": "field_out"},
        {"batter": 1, "game_id": 3, "game_date": pd.Timestamp("2026-06-12"), "events": "single"},
        {"batter": 1, "game_id": 4, "game_date": pd.Timestamp("2026-06-13"), "events": "double"},
    ]
    result = hitters.compute_current_hit_streaks(pd.DataFrame(rows)).set_index("key_mlbam")
    assert result.loc[1, "Current_Hit_Streak"] == 2


def test_compute_current_hit_streaks_doubleheader_counts_as_two_games():
    # Two real games (distinct game_id) sharing the SAME game_date (a
    # doubleheader) must count as two separate games toward the streak,
    # not get conflated by date-only grouping (data.assign_game_ids's
    # documented failure mode).
    rows = [
        {"batter": 1, "game_id": 1, "game_date": pd.Timestamp("2026-06-01"), "events": "single"},
        {"batter": 1, "game_id": 2, "game_date": pd.Timestamp("2026-06-05"), "events": "single"},
        {"batter": 1, "game_id": 3, "game_date": pd.Timestamp("2026-06-05"), "events": "double"},
    ]
    result = hitters.compute_current_hit_streaks(pd.DataFrame(rows)).set_index("key_mlbam")
    assert result.loc[1, "Current_Hit_Streak"] == 3


def test_compute_current_hit_streaks_excludes_stale_batters():
    rows = [
        # Batter 1: last game is the global latest date - stays in the result.
        {"batter": 1, "game_id": 1, "game_date": pd.Timestamp("2026-06-15"), "events": "single"},
        # Batter 2: last game is 30 days before the global latest - excluded
        # entirely (default HIT_STREAK_RECENT_DAYS=5), not shown with a
        # stale frozen streak.
        {"batter": 2, "game_id": 2, "game_date": pd.Timestamp("2026-05-16"), "events": "single"},
    ]
    result = hitters.compute_current_hit_streaks(pd.DataFrame(rows))
    assert set(result["key_mlbam"]) == {1}


def test_compute_current_hit_streaks_zero_streak_still_included():
    # Recently active but the most recent game was a miss - streak is 0,
    # but the batter is still returned (caller decides what's "on a streak").
    rows = [{"batter": 1, "game_id": 1, "game_date": pd.Timestamp("2026-06-15"), "events": "field_out"}]
    result = hitters.compute_current_hit_streaks(pd.DataFrame(rows)).set_index("key_mlbam")
    assert result.loc[1, "Current_Hit_Streak"] == 0


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
        "WAVE", "WAVE_L", "WAVE_R", "probability_L", "probability_R", "probability",
        "Game_Hit_Probability", "Consistency", "Approach", "Expected_Bases",
        "Expected_BB", "Expected_HBP", "Expected_RBI",
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
