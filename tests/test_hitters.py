import pandas as pd
import pytest

from mlb_metrics import config, hitters

LATEST = "2026-06-20"


def _at_bats(batter, throws, rows):
    """rows: list of (game_date, events) tuples, or (game_date, events,
    bat_score, post_bat_score) 4-tuples when a row needs a real RBI delta -
    bat_score/post_bat_score default to (0, 0) (Expected_RBI == 0) when
    omitted, so existing 2-tuple call sites are unaffected.

    Every row also gets the batted-ball-quality columns
    (compute_quality_of_contact) defaulted to null/not-a-batted-ball (type
    != "X") - none of these existing call sites specify real Statcast
    exit-velocity/xBA data, so none of their at-bats should be counted as
    a real batted ball (see _batted_balls below for that dedicated case) -
    and pitch_type defaulted to null too (compute_pitch_family_rates), so
    none of these existing call sites' at-bats count toward any pitch
    family either (see _family_at_bats below for that dedicated case)."""
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
            "type": pd.NA, "launch_speed": pd.NA, "estimated_ba_using_speedangle": pd.NA,
            "estimated_woba_using_speedangle": pd.NA, "launch_speed_angle": pd.NA,
            "pitch_type": pd.NA,
        })
    return result


def _batted_balls(batter, throws, rows):
    """rows: list of (game_date, launch_speed, launch_speed_angle, xba, xwoba)
    5-tuples - each becomes a real ball-in-play row (type="X") for
    hitters.compute_quality_of_contact. events is a fixed "field_out" (an
    arbitrary real batted-ball outcome - compute_quality_of_contact never
    reads `events` at all, only the batted-ball-quality columns)."""
    result = []
    for date, launch_speed, launch_speed_angle, xba, xwoba in rows:
        result.append({
            "batter": batter, "game_date": pd.Timestamp(date), "events": "field_out", "p_throws": throws,
            "bat_score": 0, "post_bat_score": 0,
            "type": "X", "launch_speed": launch_speed, "estimated_ba_using_speedangle": xba,
            "estimated_woba_using_speedangle": xwoba, "launch_speed_angle": launch_speed_angle,
            "pitch_type": pd.NA,
        })
    return result


def _family_at_bats(batter, throws, rows):
    """rows: list of (game_date, pitch_type, events) tuples - each becomes
    a real completed-PA row tagged with a real Statcast pitch_type code for
    hitters.compute_pitch_family_rates (see helpers.PITCH_TYPE_FAMILY for
    which family each code maps to)."""
    result = []
    for date, pitch_type, events in rows:
        result.append({
            "batter": batter, "game_date": pd.Timestamp(date), "events": events, "p_throws": throws,
            "bat_score": 0, "post_bat_score": 0,
            "type": pd.NA, "launch_speed": pd.NA, "estimated_ba_using_speedangle": pd.NA,
            "estimated_woba_using_speedangle": pd.NA, "launch_speed_angle": pd.NA,
            "pitch_type": pitch_type,
        })
    return result


def _pitches_seen(batter, rows):
    """rows: list of (game_date, description, zone) tuples - each becomes
    a real pitch row for hitters.compute_plate_discipline
    (pipeline.build_all_pitch_events's own shape - no PA-ending filter,
    every real pitch the batter saw)."""
    return [
        {"batter": batter, "game_date": pd.Timestamp(date), "description": description, "zone": zone}
        for date, description, zone in rows
    ]


def test_compute_plate_discipline_exact_arithmetic_all_windows():
    # Windows relative to latest=2026-06-20 (config.WAVE_WINDOWS: full/81d/
    # 30d/10d, cutoffs 2026-03-31/05-21/06-10).
    rows = _pitches_seen(1, [
        ("2026-03-15", "swinging_strike", 12),  # only full: swing+whiff, out-of-zone+chase
        ("2026-05-01", "foul", 5),               # in 81d: swing, contact, in-zone (not chase-eligible)
        ("2026-05-25", "ball", 13),               # in 30d: take, out-of-zone (not a chase, no swing)
        ("2026-06-15", "swinging_strike", 11),    # in 10d: swing+whiff, out-of-zone+chase
    ])
    all_pitches = pd.DataFrame(rows)

    result = hitters.compute_plate_discipline(all_pitches).set_index("key_mlbam")

    # full (4 pitches): swings=3 (03-15,05-01,06-15), whiffs=2 (03-15,06-15) -> 2/3
    # 81d (05-01,05-25,06-15): swings=2 (05-01,06-15), whiffs=1 (06-15) -> 1/2
    # 30d (05-25,06-15): swings=1 (06-15), whiffs=1 -> 1.0
    # 10d (06-15): swings=1, whiffs=1 -> 1.0
    expected_whiff = (2 / 3) * 0.150 + (1 / 2) * 0.250 + 1.0 * 0.275 + 1.0 * 0.325
    assert result.loc[1, "Whiff_Rate"] == pytest.approx(expected_whiff)

    # full: out_of_zone=3 (03-15,05-25,06-15), chases=2 (03-15,06-15) -> 2/3
    # 81d: out_of_zone=2 (05-25,06-15), chases=1 (06-15) -> 1/2
    # 30d: out_of_zone=2 (05-25,06-15), chases=1 (06-15) -> 1/2
    # 10d: out_of_zone=1 (06-15), chases=1 -> 1.0
    expected_chase = (2 / 3) * 0.150 + (1 / 2) * 0.250 + (1 / 2) * 0.275 + 1.0 * 0.325
    assert result.loc[1, "Chase_Rate"] == pytest.approx(expected_chase)


def test_compute_plate_discipline_foul_tip_is_contact_not_a_whiff():
    # A foul tip is real Statcast contact (the bat touched the ball) - must
    # count as a swing, but never as a whiff.
    rows = _pitches_seen(1, [("2026-06-15", "foul_tip", 5)])
    all_pitches = pd.DataFrame(rows)

    result = hitters.compute_plate_discipline(all_pitches).set_index("key_mlbam")

    assert result.loc[1, "Whiff_Rate"] == 0


def test_compute_plate_discipline_take_in_zone_is_neither_swing_nor_chase():
    # A called strike inside the real zone: not a swing, not out-of-zone -
    # contributes to neither Whiff_Rate's nor Chase_Rate's denominator.
    rows = _pitches_seen(1, [("2026-06-15", "called_strike", 5)])
    all_pitches = pd.DataFrame(rows)

    result = hitters.compute_plate_discipline(all_pitches).set_index("key_mlbam")

    assert result.loc[1, "Whiff_Rate"] == 0
    assert result.loc[1, "Chase_Rate"] == 0


def test_compute_plate_discipline_absent_batter_not_in_output():
    # A batter who never appears in all_pitches at all has no real
    # underlying data - absent from the output entirely (same "absence,
    # not a fabricated zero" precedent every other blended rate in this
    # file establishes), not present with a 0.
    all_pitches = pd.DataFrame(_pitches_seen(1, [("2026-06-15", "ball", 5)]))

    result = hitters.compute_plate_discipline(all_pitches)

    assert 2 not in set(result["key_mlbam"])


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
    # Pinned to the exact-null-hypothesis shrinkage_strength=0.0 - this test
    # is about the window-blend/platoon-share math, not shrinkage (which has
    # its own dedicated test below), and must stay independent of whatever
    # config.WAVE_SHRINKAGE_STRENGTH's live default happens to be.
    wave = hitters.compute_wave(dt, shrinkage_strength=0.0).set_index("key_mlbam")

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


def test_compute_wave_shrinkage_pulls_low_n_toward_league_rate_more_than_high_n():
    # Every row lands on the same date (2026-06-15, within 10 days of
    # LATEST), so all four WAVE windows (full/81/30/10) see the exact same
    # at-bats and their weights sum to 1.0 - the blended output equals
    # helpers.shrink_rate's own per-window result exactly, making this
    # hand-computable end to end through compute_wave, not just at the
    # helpers.shrink_rate unit-test level.
    rows = _at_bats(1, "R", [("2026-06-15", "single")] * 2)  # 2 AB, 2 hits: raw rate 1.0
    rows += _at_bats(2, "R", [("2026-06-15", "single")] * 10 + [("2026-06-15", "field_out")] * 10)  # 20 AB, 10 hits: raw rate 0.5
    dt = pd.DataFrame(rows)

    # league_rate = (2 hits + 10 hits) / (2 AB + 20 AB) = 12/22
    wave = hitters.compute_wave(dt, shrinkage_strength=10.0).set_index("key_mlbam")

    # Batter 1: (2 + 10*12/22) / (2 + 10) = (2 + 60/11) / 12
    expected_1 = (2 + 10 * (12 / 22)) / (2 + 10)
    # Batter 2: (10 + 10*12/22) / (20 + 10)
    expected_2 = (10 + 10 * (12 / 22)) / (20 + 10)
    assert wave.loc[1, "WAVE_R"] == pytest.approx(expected_1)
    assert wave.loc[2, "WAVE_R"] == pytest.approx(expected_2)

    # Batter 1 (n=2, raw rate 1.0) got pulled much further from its raw rate
    # than batter 2 (n=20, raw rate 0.5) - the real small-sample-shrinkage
    # behavior, not just "some number changed."
    assert abs(1.0 - wave.loc[1, "WAVE_R"]) > abs(0.5 - wave.loc[2, "WAVE_R"])


def test_compute_wave_ci_is_the_real_wilson_interval_on_raw_counts_not_shrinkage_dependent():
    # Same fixture as the shrinkage test above (single-date trick, all
    # windows collapse to the raw rate) - batter 1: 2 AB/2 hits, batter 2:
    # 20 AB/10 hits.
    rows = _at_bats(1, "R", [("2026-06-15", "single")] * 2)
    rows += _at_bats(2, "R", [("2026-06-15", "single")] * 10 + [("2026-06-15", "field_out")] * 10)
    dt = pd.DataFrame(rows)

    # Real Wilson score interval, hand-derived independently (not via
    # helpers.wilson_ci itself) for count=2,n=2 and count=10,n=20,
    # z=1.959963984540054 (alpha=0.05):
    #   batter 1 (2/2): (0.34238022750665303, 1.0)
    #   batter 2 (10/20): (0.2992980081982123, 0.7007019918017877)
    for strength in (0.0, 10.0, 50.0):
        wave = hitters.compute_wave(dt, shrinkage_strength=strength).set_index("key_mlbam")
        # The CI is computed on the RAW hit/AB counts, never the shrunk
        # WAVE point estimate - it must be IDENTICAL regardless of
        # shrinkage_strength, the core design decision this slice makes.
        assert wave.loc[1, "WAVE_CI_Low"] == pytest.approx(0.34238022750665303)
        assert wave.loc[1, "WAVE_CI_High"] == pytest.approx(1.0)
        assert wave.loc[2, "WAVE_CI_Low"] == pytest.approx(0.2992980081982123)
        assert wave.loc[2, "WAVE_CI_High"] == pytest.approx(0.7007019918017877)

        # probability_CI_* is the exact monotonic transform of WAVE_CI_*,
        # not an independently (re-)computed interval.
        trials = config.WAVE_TRIALS_PER_GAME
        assert wave.loc[1, "probability_CI_Low"] == pytest.approx(1 - (1 - wave.loc[1, "WAVE_CI_Low"]) ** trials)
        assert wave.loc[1, "probability_CI_High"] == pytest.approx(1 - (1 - wave.loc[1, "WAVE_CI_High"]) ** trials)


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


def test_compute_quality_of_contact_exact_arithmetic_all_windows():
    # All 5 batted balls land within 7 days of the latest date (2026-06-20),
    # so they're included in every one of config.WHOPS_WTB_WINDOWS
    # (7d/15d/30d/full) - weights sum to 1.0, so the blended rate equals
    # the raw mean/rate exactly, same technique
    # test_compute_extended_dk_rates_exact_arithmetic_all_windows uses.
    rows = _batted_balls(1, "R", [
        ("2026-06-16", 90.0, 3, 0.20, 0.25),
        ("2026-06-17", 95.0, 4, 0.30, 0.35),
        ("2026-06-18", 100.0, 6, 0.40, 0.45),  # real barrel (launch_speed_angle == 6)
        ("2026-06-19", 105.0, 5, 0.50, 0.55),
        ("2026-06-20", 110.0, 2, 0.60, 0.65),
    ])
    dt = pd.DataFrame(rows)

    result = hitters.compute_quality_of_contact(dt).set_index("key_mlbam")

    assert result.loc[1, "Exit_Velo"] == pytest.approx(100.0)  # mean(90,95,100,105,110)
    assert result.loc[1, "Barrel_Rate"] == pytest.approx(0.2)  # 1 barrel / 5 batted balls
    assert result.loc[1, "xBA"] == pytest.approx(0.4)  # mean(0.2..0.6)
    assert result.loc[1, "xwOBA"] == pytest.approx(0.45)  # mean(0.25..0.65)


def test_compute_quality_of_contact_excludes_non_batted_ball_events():
    # A strikeout/walk has no real batted-ball data (type != "X") - must
    # not crash on the missing launch columns. A batter with real at-bats
    # but ZERO real batted balls on either side has no anchoring row on
    # either side (same "anchored on batters with a real full-season row
    # on that side" precedent WAVE/WTB/compute_extended_dk_rates already
    # establish) - absent from this function's own output entirely, not
    # present-with-a-zero. assemble_hitters's own left-join + fillna(0) is
    # what actually turns that absence into a real 0 for such a batter in
    # the real output table (see test_assemble_hitters_output_columns_and_derived_fields).
    rows = _at_bats(1, "R", [("2026-06-15", "strikeout"), ("2026-06-16", "walk")])
    dt = pd.DataFrame(rows)

    result = hitters.compute_quality_of_contact(dt)

    assert result.empty


def test_compute_quality_of_contact_blends_platoon_share():
    # Batter faces both sides with very different exit velo - isolate the
    # bbtl/bbtr platoon-share blend the same way
    # test_compute_extended_dk_rates_blends_platoon_share does.
    rows = _batted_balls(1, "R", [("2026-06-19", 100.0, 6, 0.5, 0.5), ("2026-06-19", 100.0, 6, 0.5, 0.5)])
    rows += _batted_balls(1, "L", [("2026-06-19", 50.0, 0, 0.1, 0.1), ("2026-06-19", 50.0, 0, 0.1, 0.1)])
    dt = pd.DataFrame(rows)

    result = hitters.compute_quality_of_contact(dt).set_index("key_mlbam")

    # 2 batted balls each side -> bbtr = bbtl = 0.5; blended Exit_Velo = 100*0.5 + 50*0.5 = 75
    assert result.loc[1, "Exit_Velo"] == pytest.approx(75.0)


def test_compute_pitch_family_rates_exact_arithmetic_all_windows():
    # Windows relative to latest=2026-06-20 (config.WAVE_WINDOWS: full/81d/
    # 30d/10d, cutoffs 2026-03-31/05-21/06-10).
    rows = _family_at_bats(1, "R", [
        ("2026-03-15", "FF", "single"),      # only full (before 81d cutoff)
        ("2026-05-01", "FF", "field_out"),   # in 81d, not 30d/10d
        ("2026-05-25", "FF", "single"),      # in 30d, not 10d
        ("2026-06-15", "FF", "single"),      # in 10d
    ])
    dt = pd.DataFrame(rows)

    result = hitters.compute_pitch_family_rates(dt).set_index("key_mlbam")

    # full: n=4, hits=3 -> 0.75; 81d: n=3, hits=2 -> 2/3; 30d: n=2, hits=2 -> 1.0; 10d: n=1, hits=1 -> 1.0
    expected = 0.75 * 0.150 + (2 / 3) * 0.250 + 1.0 * 0.275 + 1.0 * 0.325
    assert result.loc[1, "Fastball_WAVE"] == pytest.approx(expected)
    assert result.loc[1, "Breaking_WAVE"] == 0
    assert result.loc[1, "Offspeed_WAVE"] == 0


def test_compute_pitch_family_rates_excludes_unclassifiable_pitch_types():
    # A pitchout ("PO") and a real null pitch_type have no real family -
    # same "absent from this function's own output entirely" precedent
    # compute_quality_of_contact's non-batted-ball test establishes.
    rows = _family_at_bats(1, "R", [("2026-06-15", "PO", "single"), ("2026-06-16", None, "single")])
    dt = pd.DataFrame(rows)

    result = hitters.compute_pitch_family_rates(dt)

    assert result.empty


def test_compute_pitch_family_rates_family_absent_reads_zero_not_dropped():
    # A batter with real data in ONE family only (breaking) must still
    # appear in the output, with the other two families at a real 0 - not
    # silently dropped from the whole table just because one family has no
    # data (the union-merge across all three families, not an anchor on
    # any single one).
    rows = _family_at_bats(1, "R", [("2026-06-18", "SL", "single")])
    dt = pd.DataFrame(rows)

    result = hitters.compute_pitch_family_rates(dt).set_index("key_mlbam")

    assert result.loc[1, "Breaking_WAVE"] > 0
    assert result.loc[1, "Fastball_WAVE"] == 0
    assert result.loc[1, "Offspeed_WAVE"] == 0


def test_compute_game_hit_probability_blends_game_level_hit_rate():
    rows = [
        {"batter": 1, "game_id": 1, "game_date": pd.Timestamp("2026-03-12"), "events": "field_out"},
        {"batter": 1, "game_id": 2, "game_date": pd.Timestamp("2026-05-01"), "events": "single"},
        {"batter": 1, "game_id": 3, "game_date": pd.Timestamp("2026-05-31"), "events": "single"},
        {"batter": 1, "game_id": 4, "game_date": pd.Timestamp("2026-06-15"), "events": "single"},
    ]
    data_with_game_id = pd.DataFrame(rows)

    # Pinned to the exact-null-hypothesis shrinkage_strength=0.0 - this test
    # is about the window-blend math, not shrinkage (which has its own
    # dedicated test below), and must stay independent of whatever
    # config.GAME_HIT_PROB_SHRINKAGE_STRENGTH's live default happens to be.
    result = hitters.compute_game_hit_probability(data_with_game_id, shrinkage_strength=0.0).set_index("key_mlbam")

    # full=3/4, 81d=3/3, 30d=2/2, 10d=1/1
    expected = (3 / 4) * 0.175 + 1.0 * 0.225 + 1.0 * 0.275 + 1.0 * 0.325
    assert result.loc[1, "Game_Hit_Probability"] == pytest.approx(expected)


def test_compute_game_hit_probability_shrinkage_pulls_low_n_toward_league_rate_more_than_high_n():
    # All games on the same recent date so every window (full/81/30/10)
    # sees the identical game set and weights sum to 1.0 - same trick as
    # the WAVE shrinkage test above, makes this hand-computable exactly.
    rows = [
        {"batter": 1, "game_id": i, "game_date": pd.Timestamp("2026-06-15"), "events": "single"}
        for i in range(2)
    ]  # 2 games, 2 hits: raw rate 1.0
    rows += [
        {"batter": 2, "game_id": 100 + i, "game_date": pd.Timestamp("2026-06-15"), "events": "single"}
        for i in range(10)
    ] + [
        {"batter": 2, "game_id": 200 + i, "game_date": pd.Timestamp("2026-06-15"), "events": "field_out"}
        for i in range(10)
    ]  # 20 games, 10 hits: raw rate 0.5
    data_with_game_id = pd.DataFrame(rows)

    # league_rate = (2 + 10) / (2 + 20) = 12/22
    result = hitters.compute_game_hit_probability(data_with_game_id, shrinkage_strength=10.0).set_index("key_mlbam")

    expected_1 = (2 + 10 * (12 / 22)) / (2 + 10)
    expected_2 = (10 + 10 * (12 / 22)) / (20 + 10)
    assert result.loc[1, "Game_Hit_Probability"] == pytest.approx(expected_1)
    assert result.loc[2, "Game_Hit_Probability"] == pytest.approx(expected_2)
    assert abs(1.0 - result.loc[1, "Game_Hit_Probability"]) > abs(0.5 - result.loc[2, "Game_Hit_Probability"])


def test_compute_game_hit_probability_ci_is_the_real_wilson_interval_on_raw_counts_not_shrinkage_dependent():
    rows = [
        {"batter": 1, "game_id": i, "game_date": pd.Timestamp("2026-06-15"), "events": "single"}
        for i in range(2)
    ]  # 2 games, 2 hits
    rows += [
        {"batter": 2, "game_id": 100 + i, "game_date": pd.Timestamp("2026-06-15"), "events": "single"}
        for i in range(10)
    ] + [
        {"batter": 2, "game_id": 200 + i, "game_date": pd.Timestamp("2026-06-15"), "events": "field_out"}
        for i in range(10)
    ]  # 20 games, 10 hits
    data_with_game_id = pd.DataFrame(rows)

    # Same hand-derived Wilson bounds as the compute_wave CI test above
    # (identical count/n pairs: 2/2 and 10/20).
    for strength in (0.0, 10.0, 50.0):
        result = hitters.compute_game_hit_probability(
            data_with_game_id, shrinkage_strength=strength
        ).set_index("key_mlbam")
        assert result.loc[1, "Game_Hit_Probability_CI_Low"] == pytest.approx(0.34238022750665303)
        assert result.loc[1, "Game_Hit_Probability_CI_High"] == pytest.approx(1.0)
        assert result.loc[2, "Game_Hit_Probability_CI_Low"] == pytest.approx(0.2992980081982123)
        assert result.loc[2, "Game_Hit_Probability_CI_High"] == pytest.approx(0.7007019918017877)


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
        "WAVE_CI_Low", "WAVE_CI_High", "probability_CI_Low", "probability_CI_High",
        "Game_Hit_Probability", "Game_Hit_Probability_CI_Low", "Game_Hit_Probability_CI_High",
        "Consistency", "Approach", "Expected_Bases",
        "Expected_BB", "Expected_HBP", "Expected_RBI",
        "Exit_Velo", "Barrel_Rate", "xBA", "xwOBA",
        "Fastball_WAVE", "Breaking_WAVE", "Offspeed_WAVE", "Last_Game_Date",
    ]
    row = result.iloc[0]
    assert row["name_last"] == "Player"
    assert row["Consistency"] == pytest.approx(row["Game_Hit_Probability"] - row["probability"])
    assert row["Approach"] == pytest.approx(row["Game_Hit_Probability"] * row["probability"])
    assert row["Last_Game_Date"] == pd.Timestamp("2026-06-16")
    # Neither at-bat in this fixture is a real batted ball (see _at_bats's
    # docstring) - quality-of-contact should read 0 (via assemble_hitters's
    # final fillna(0), since compute_quality_of_contact itself returns no
    # row at all for a batter with zero batted balls - see the dedicated
    # test for that), not a fabricated value.
    assert row["Exit_Velo"] == 0
    assert row["Barrel_Rate"] == 0
    assert row["xBA"] == 0
    assert row["xwOBA"] == 0
    # Neither at-bat has a real pitch_type either - same "0, not a
    # fabricated value" convention for pitch-family rates.
    assert row["Fastball_WAVE"] == 0
    assert row["Breaking_WAVE"] == 0
    assert row["Offspeed_WAVE"] == 0


def test_compute_last_game_dates_most_recent_completed_event():
    rows = [
        {"batter": 1, "game_date": pd.Timestamp("2026-06-10"), "events": "single"},
        {"batter": 1, "game_date": pd.Timestamp("2026-06-15"), "events": "field_out"},
        {"batter": 2, "game_date": pd.Timestamp("2026-06-12"), "events": "double"},
    ]
    result = hitters.compute_last_game_dates(pd.DataFrame(rows)).set_index("key_mlbam")
    assert result.loc[1, "Last_Game_Date"] == pd.Timestamp("2026-06-15")
    assert result.loc[2, "Last_Game_Date"] == pd.Timestamp("2026-06-12")


def test_assemble_hitters_last_game_date_is_nat_not_dropped_for_zero_events():
    # A batter present in `dt` (has WAVE at-bats) but with no completed
    # event at all in data_with_game_id (an edge case, e.g. a
    # not-yet-completed at-bat) must get Last_Game_Date=NaT, not be
    # silently dropped from the output or coerced to a real-looking date.
    dt = pd.DataFrame(_at_bats(1, "R", [("2026-06-15", "single"), ("2026-06-16", "field_out")]))
    data_with_game_id = pd.DataFrame(columns=["batter", "game_id", "game_date", "events"])
    names = pd.DataFrame([{"key_mlbam": 1, "name_first": "Test", "name_last": "Player"}])
    latest_team = pd.DataFrame([{"key_mlbam": 1, "team": "NYY"}])

    result = hitters.assemble_hitters(dt, data_with_game_id, names, latest_team)

    assert len(result) == 1
    assert pd.isna(result.iloc[0]["Last_Game_Date"])


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


def test_assemble_hitters_merges_plate_discipline_when_all_pitches_provided():
    dt = pd.DataFrame(_at_bats(1, "R", [("2026-06-15", "single"), ("2026-06-16", "field_out")]))
    data_with_game_id = pd.DataFrame([
        {"batter": 1, "game_id": 1, "game_date": pd.Timestamp("2026-06-15"), "events": "single"},
        {"batter": 1, "game_id": 2, "game_date": pd.Timestamp("2026-06-16"), "events": "field_out"},
    ])
    names = pd.DataFrame([{"key_mlbam": 1, "name_first": "Test", "name_last": "Player"}])
    latest_team = pd.DataFrame([{"key_mlbam": 1, "team": "NYY"}])
    all_pitches = pd.DataFrame(_pitches_seen(1, [("2026-06-15", "swinging_strike", 12)]))

    result = hitters.assemble_hitters(dt, data_with_game_id, names, latest_team, all_pitches=all_pitches)

    assert "Whiff_Rate" in result.columns and "Chase_Rate" in result.columns
    row = result.iloc[0]
    assert row["Whiff_Rate"] == pytest.approx(1.0)
    assert row["Chase_Rate"] == pytest.approx(1.0)


def test_assemble_hitters_omits_plate_discipline_columns_when_all_pitches_not_provided():
    dt = pd.DataFrame(_at_bats(1, "R", [("2026-06-15", "single"), ("2026-06-16", "field_out")]))
    data_with_game_id = pd.DataFrame([
        {"batter": 1, "game_id": 1, "game_date": pd.Timestamp("2026-06-15"), "events": "single"},
        {"batter": 1, "game_id": 2, "game_date": pd.Timestamp("2026-06-16"), "events": "field_out"},
    ])
    names = pd.DataFrame([{"key_mlbam": 1, "name_first": "Test", "name_last": "Player"}])
    latest_team = pd.DataFrame([{"key_mlbam": 1, "team": "NYY"}])

    result = hitters.assemble_hitters(dt, data_with_game_id, names, latest_team)

    assert "Whiff_Rate" not in result.columns and "Chase_Rate" not in result.columns


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
