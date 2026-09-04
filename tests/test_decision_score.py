import pandas as pd
import pytest

from mlb_metrics import config, decision_score


def _pa_events(batter, rows):
    """rows: list of (game_date, events, zone) tuples -
    pipeline.build_pitch_events's shape (one row per completed PA, the
    PA-ending pitch's own real zone)."""
    return pd.DataFrame([
        {"batter": batter, "game_date": pd.Timestamp(date), "events": events, "zone": zone}
        for date, events, zone in rows
    ])


def test_classify_count_context_buckets():
    balls = pd.Series([3, 3, 2, 0, 1, 1, 0, 3, 2, 2])
    strikes = pd.Series([0, 1, 0, 2, 2, 1, 0, 2, 1, 2])

    result = decision_score.classify_count_context(balls, strikes)

    # 3-0/3-1/2-0 hitter, 0-2/1-2 pitcher, everything else (including the
    # symmetric 3-2 full count) neutral.
    expected = [
        "hitter", "hitter", "hitter", "pitcher", "pitcher",
        "neutral", "neutral", "neutral", "neutral", "neutral",
    ]
    assert list(result) == expected


def test_classify_count_context_missing_count_is_neutral():
    balls = pd.Series([3, None])
    strikes = pd.Series([None, 2])

    result = decision_score.classify_count_context(balls, strikes)

    assert list(result) == ["neutral", "neutral"]


def test_compute_batter_overall_ops_exact_arithmetic():
    windows = [(None, 1.0)]  # single full-season window - no blending to hand-compute through
    pa_events = _pa_events(1, [
        ("2026-03-01", "single", 3),
        ("2026-03-02", "single", 3),
        ("2026-03-03", "single", 7),
        ("2026-03-04", "walk", 11),
        ("2026-03-05", "strikeout", 5),
    ])

    result = decision_score.compute_batter_overall_ops(pa_events, windows).set_index("key_mlbam")

    # PA=5, AB=4 (walk excluded), OB=4 (3 singles + 1 walk), TB=3 (3 singles).
    assert result.loc[1, "OBP"] == pytest.approx(4 / 5)
    assert result.loc[1, "SLG"] == pytest.approx(3 / 4)
    assert result.loc[1, "OPS"] == pytest.approx(4 / 5 + 3 / 4)


def test_compute_zone_ops_shrinks_toward_batters_own_overall_average():
    """Real point of the shrinkage: zone 5's own raw (unshrunk) rate would
    be OPS=0 (its lone PA there was a strikeout) - shrinkage pulls it
    toward this batter's own strong 1.55 overall OPS instead of leaving
    it at a noisy 0."""
    windows = [(None, 1.0)]
    pa_events = _pa_events(1, [
        ("2026-03-01", "single", 3),
        ("2026-03-02", "single", 3),
        ("2026-03-03", "single", 7),
        ("2026-03-04", "walk", 11),
        ("2026-03-05", "strikeout", 5),
    ])

    overall = decision_score.compute_batter_overall_ops(pa_events, windows)
    assert overall.loc[0, "OPS"] == pytest.approx(1.55)  # OBP 0.8 + SLG 0.75

    zone_ops = decision_score.compute_zone_ops(pa_events, overall, windows).set_index(["key_mlbam", "zone"])

    strength = config.DECISION_SCORE_ZONE_SHRINKAGE_STRENGTH
    expected_zone5_obp = (0 + strength * 0.8) / (1 + strength)  # ob=0, n=1
    expected_zone5_slg = (0 + strength * 0.75) / (1 + strength)  # tb=0, ab=1
    assert zone_ops.loc[(1, 5), "Zone_OPS"] == pytest.approx(expected_zone5_obp + expected_zone5_slg)
    assert zone_ops.loc[(1, 5), "Zone_OPS"] > 0  # pulled well above the raw 0, toward the batter's own average

    # Zone 3 (2 real singles, real signal) shrinks far less - still clearly above its own zone's naive rate would suggest less pull.
    expected_zone3_obp = (2 + strength * 0.8) / (2 + strength)
    expected_zone3_slg = (2 + strength * 0.75) / (2 + strength)
    assert zone_ops.loc[(1, 3), "Zone_OPS"] == pytest.approx(expected_zone3_obp + expected_zone3_slg)

    # Zone 11's own PA was a walk (0 official at-bats) - shrink_rate's
    # `ab + strength` denominator handles ab=0 without dividing by zero.
    expected_zone11_obp = (1 + strength * 0.8) / (1 + strength)  # ob=1 (the walk), n=1
    expected_zone11_slg = (0 + strength * 0.75) / (0 + strength)  # tb=0, ab=0
    assert zone_ops.loc[(1, 11), "Zone_OPS"] == pytest.approx(expected_zone11_obp + expected_zone11_slg)


def test_compute_zone_ops_absent_zone_not_in_output():
    windows = [(None, 1.0)]
    pa_events = _pa_events(1, [("2026-03-01", "single", 3)])
    overall = decision_score.compute_batter_overall_ops(pa_events, windows)

    zone_ops = decision_score.compute_zone_ops(pa_events, overall, windows)

    assert set(zone_ops["zone"]) == {3}  # only the one real zone this batter ever ended a PA in


def _pitch(batter, zone, balls, strikes, inning=3, bat_score=0, fld_score=0, on_2b=None, on_3b=None):
    return {
        "batter": batter, "zone": zone, "balls": balls, "strikes": strikes,
        "inning": inning, "bat_score": bat_score, "fld_score": fld_score,
        "on_2b": on_2b, "on_3b": on_3b,
    }


def test_compute_decision_advice_count_and_situation_multipliers():
    """Verifies the multiplier MECHANISM itself (via explicit overrides,
    not the live config defaults - the real backtest found the shipped
    defaults are 1.0/no-op, see config.py's DECISION_SCORE_* comments -
    this test still needs to prove the machinery works correctly for
    scripts/backtest_decision_score.py's own sweep to mean anything)."""
    overall_ops = pd.DataFrame([{"key_mlbam": 1, "OPS": 1.0}])
    zone_ops = pd.DataFrame([
        {"key_mlbam": 1, "zone": 5, "Zone_OPS": 0.90},  # below neutral(1.0)/hitter(1.15), above pitcher(0.85)
        {"key_mlbam": 1, "zone": 7, "Zone_OPS": 0.97},  # below neutral(1.0), above high-leverage(0.95)
    ])
    all_pitches = pd.DataFrame([
        _pitch(1, zone=5, balls=1, strikes=1),               # neutral count -> take (0.90 < 1.0)
        _pitch(1, zone=5, balls=3, strikes=0),                # hitter's count -> take (0.90 < 1.15)
        _pitch(1, zone=5, balls=0, strikes=2),                # pitcher's count -> swing (0.90 >= 0.85)
        _pitch(1, zone=7, balls=1, strikes=1),                # neutral, low leverage -> take (0.97 < 1.0)
        _pitch(1, zone=7, balls=1, strikes=1, inning=8, bat_score=3, fld_score=2, on_2b=12345),  # high leverage -> swing (0.97 >= 0.95)
    ])

    advice = decision_score.compute_decision_advice(
        all_pitches, overall_ops, zone_ops,
        hitter_multiplier=1.15, pitcher_multiplier=0.85, leverage_multiplier=0.95,
    )

    assert list(advice) == ["take", "take", "swing", "take", "swing"]


def test_compute_decision_advice_default_multipliers_are_all_neutral():
    """The shipped config defaults are 1.0/no-op (see config.py's
    DECISION_SCORE_* comments for the real backtest finding this was
    validated against) - a pitch's advice with no overrides passed should
    depend ONLY on zone vs. overall OPS, with count/situation making no
    difference at all."""
    overall_ops = pd.DataFrame([{"key_mlbam": 1, "OPS": 1.0}])
    zone_ops = pd.DataFrame([{"key_mlbam": 1, "zone": 5, "Zone_OPS": 0.90}])
    all_pitches = pd.DataFrame([
        _pitch(1, zone=5, balls=1, strikes=1),   # neutral count
        _pitch(1, zone=5, balls=3, strikes=0),   # hitter's count - no adjustment by default
        _pitch(1, zone=5, balls=0, strikes=2),   # pitcher's count - no adjustment by default
        _pitch(1, zone=5, balls=1, strikes=1, inning=8, bat_score=3, fld_score=2, on_2b=12345),  # high leverage - no adjustment by default
    ])

    advice = decision_score.compute_decision_advice(all_pitches, overall_ops, zone_ops)

    assert list(advice) == ["take", "take", "take", "take"]  # 0.90 < 1.0 in every case


def test_compute_decision_advice_unknown_zone_defaults_to_take():
    overall_ops = pd.DataFrame([{"key_mlbam": 1, "OPS": 1.0}])
    zone_ops = pd.DataFrame([{"key_mlbam": 1, "zone": 5, "Zone_OPS": 1.2}])
    all_pitches = pd.DataFrame([_pitch(1, zone=99, balls=1, strikes=1)])  # a zone this batter has no real data for

    advice = decision_score.compute_decision_advice(all_pitches, overall_ops, zone_ops)

    assert list(advice) == ["take"]  # Zone_OPS fills to 0, never clears a positive threshold


def test_compute_decision_score_exact_arithmetic_end_to_end():
    """PA1: a taken ball, then a single in zone 3 (a swing). PA2: a single
    swinging-strike strikeout in zone 5. All neutral counts/situations
    (multiplier 1.0 throughout) so the threshold is just this batter's own
    overall OPS (1.0) - hand-computed below."""
    windows = [(None, 1.0)]
    pa_events = _pa_events(1, [
        ("2026-04-01", "single", 3),
        ("2026-04-02", "strikeout", 5),
    ])
    all_pitches = pd.DataFrame([
        {**_pitch(1, zone=12, balls=0, strikes=0), "game_date": pd.Timestamp("2026-04-01"), "description": "ball"},
        {**_pitch(1, zone=3, balls=1, strikes=0), "game_date": pd.Timestamp("2026-04-01"), "description": "hit_into_play"},
        {**_pitch(1, zone=5, balls=0, strikes=0), "game_date": pd.Timestamp("2026-04-02"), "description": "swinging_strike"},
    ])

    result = decision_score.compute_decision_score(all_pitches, pa_events, windows).set_index("key_mlbam")

    # Overall OPS = 1.0 (OBP 0.5 + SLG 0.5, from PA=2/AB=2/OB=1/TB=1).
    # Pitch 1 (zone 12, unseen -> Zone_OPS 0): threshold 1.0, advice "take" - actual take -> matched.
    # Pitch 2 (zone 3, Zone_OPS ~1.048 shrunk toward 1.0): advice "swing" - actual swing -> matched.
    # Pitch 3 (zone 5, Zone_OPS ~0.952 shrunk toward 1.0): advice "take" - actual swing -> NOT matched.
    assert result.loc[1, "Decision_Score"] == pytest.approx(2 / 3 * 100)
    assert result.loc[1, "Decision_Score_N"] == 3


def test_compute_decision_score_absent_batter_not_in_output():
    windows = [(None, 1.0)]
    pa_events = _pa_events(1, [("2026-04-01", "single", 3)])
    all_pitches = pd.DataFrame([
        {**_pitch(1, zone=3, balls=1, strikes=0), "game_date": pd.Timestamp("2026-04-01"), "description": "hit_into_play"},
    ])

    result = decision_score.compute_decision_score(all_pitches, pa_events, windows)

    assert set(result["key_mlbam"]) == {1}  # only a batter with real pitches seen gets a row
