import pandas as pd
import pytest

from mlb_metrics import matchup


def _wave(rows):
    """rows: list of (key_mlbam, team, game_hit_probability)."""
    return pd.DataFrame(
        [{"key_mlbam": k, "team": t, "Game_Hit_Probability": g} for k, t, g in rows]
    )


def test_compute_matchup_hit_probability_exact_arithmetic():
    wave = _wave([(1, "NYY", 0.8)])
    pave = pd.DataFrame([{"key_mlbam": 999, "PAVE_PLUS": 0.9}])  # opposing probable starter
    confidence = pd.DataFrame([{"team": "BOS", "Bullpen_PAVE_PLUS": 1.1}])
    schedule_df = pd.DataFrame([{"team": "NYY", "opponent": "BOS", "probable_pitcher_key_mlbam": 999}])

    result = matchup.compute_matchup_hit_probability(wave, pave, confidence, schedule_df).set_index("key_mlbam")

    # opponent_quality = .6*.9 + .4*1.1 = .98 -> .8*.98 = .784
    assert result.loc[1, "Matchup_Hit_Probability"] == pytest.approx(0.784)


def test_pave_plus_is_clipped_before_blending_not_after():
    wave = _wave([(1, "NYY", 0.3)])
    pave = pd.DataFrame([{"key_mlbam": 999, "PAVE_PLUS": 5.0}])  # extreme outlier
    confidence = pd.DataFrame([{"team": "BOS", "Bullpen_PAVE_PLUS": 0.1}])  # extreme outlier
    schedule_df = pd.DataFrame([{"team": "NYY", "opponent": "BOS", "probable_pitcher_key_mlbam": 999}])

    result = matchup.compute_matchup_hit_probability(wave, pave, confidence, schedule_df).set_index("key_mlbam")

    # Clipped to (0.5, 1.75) BEFORE blending: .6*1.75 + .4*.5 = 1.25 -> .3*1.25 = .375
    # (blending the raw 5.0/0.1 first and clipping only the final result
    # would give .3*(.6*5.0+.4*.1) = .912 instead - this test would catch that bug.)
    assert result.loc[1, "Matchup_Hit_Probability"] == pytest.approx(0.375)


def test_missing_probable_starter_uses_neutral_multiplier():
    wave = _wave([(1, "NYY", 0.5)])
    pave = pd.DataFrame(columns=["key_mlbam", "PAVE_PLUS"])  # no one announced yet
    confidence = pd.DataFrame([{"team": "BOS", "Bullpen_PAVE_PLUS": 1.2}])
    schedule_df = pd.DataFrame([{"team": "NYY", "opponent": "BOS", "probable_pitcher_key_mlbam": None}])

    result = matchup.compute_matchup_hit_probability(wave, pave, confidence, schedule_df).set_index("key_mlbam")

    # starter component neutral (1.0): .6*1.0 + .4*1.2 = 1.08 -> .5*1.08 = .54
    assert result.loc[1, "Matchup_Hit_Probability"] == pytest.approx(0.54)


def test_batter_with_no_game_today_is_excluded():
    wave = _wave([(1, "NYY", 0.8), (2, "SEA", 0.7)])
    pave = pd.DataFrame(columns=["key_mlbam", "PAVE_PLUS"])
    confidence = pd.DataFrame([{"team": "BOS", "Bullpen_PAVE_PLUS": 1.0}])
    # Only NYY has a game today; SEA does not appear in the schedule at all.
    schedule_df = pd.DataFrame([{"team": "NYY", "opponent": "BOS", "probable_pitcher_key_mlbam": None}])

    result = matchup.compute_matchup_hit_probability(wave, pave, confidence, schedule_df)

    assert list(result["key_mlbam"]) == [1]


def test_final_probability_is_clipped_to_zero_one():
    wave = _wave([(1, "NYY", 0.99)])
    pave = pd.DataFrame([{"key_mlbam": 999, "PAVE_PLUS": 1.75}])
    confidence = pd.DataFrame([{"team": "BOS", "Bullpen_PAVE_PLUS": 1.75}])
    schedule_df = pd.DataFrame([{"team": "NYY", "opponent": "BOS", "probable_pitcher_key_mlbam": 999}])

    result = matchup.compute_matchup_hit_probability(wave, pave, confidence, schedule_df).set_index("key_mlbam")

    assert result.loc[1, "Matchup_Hit_Probability"] == 1.0
