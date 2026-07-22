import pandas as pd
import pytest

from mlb_metrics import game_picks


def _confidence(rows):
    """rows: list of dicts with team, pyth_Strength, pyth_Confidence,
    suppression_resistance, true_power, Bullpen_PAVE_PLUS."""
    return pd.DataFrame(rows)


def test_team_composite_exact_arithmetic():
    confidence = _confidence([
        {"team": "NYY", "pyth_Strength": 1.1, "pyth_Confidence": 1.05,
         "suppression_resistance": 1.0, "true_power": 1.0, "Bullpen_PAVE_PLUS": 0.9},
    ])
    result = game_picks._team_composite(confidence).set_index("team")
    # .25*1.1 + .25*1.05 + .25*1.0 + .25*1.0 = .25*4.15 = 1.0375
    assert result.loc["NYY", "composite"] == pytest.approx(1.0375)


def _schedule_games(home_team="NYY", away_team="BOS", home_pitcher=1, away_pitcher=2):
    return pd.DataFrame([{
        "game_pk": 100, "date": pd.Timestamp("2026-07-22"),
        "home_team": home_team, "away_team": away_team,
        "home_probable_pitcher_key_mlbam": home_pitcher, "away_probable_pitcher_key_mlbam": away_pitcher,
        "status": "Scheduled", "home_score": None, "away_score": None,
    }])


def test_compute_game_win_probabilities_exact_arithmetic(monkeypatch):
    # Pinned to pure PAVE_PLUS (weight=0.0) regardless of the live
    # GAME_PICK_SUSCEPTIBILITY_WEIGHT default - this test specifically
    # exercises the PAVE_PLUS arithmetic path; the blended-weight behavior
    # has its own dedicated tests below.
    monkeypatch.setattr(game_picks.config, "GAME_PICK_SUSCEPTIBILITY_WEIGHT", 0.0)
    confidence = _confidence([
        {"team": "NYY", "pyth_Strength": 1.1, "pyth_Confidence": 1.05,
         "suppression_resistance": 1.0, "true_power": 1.0, "Bullpen_PAVE_PLUS": 0.9},
        {"team": "BOS", "pyth_Strength": 0.9, "pyth_Confidence": 0.95,
         "suppression_resistance": 1.0, "true_power": 1.0, "Bullpen_PAVE_PLUS": 1.1},
    ])
    pave = pd.DataFrame([
        {"key_mlbam": 1, "PAVE_PLUS": 0.8},  # NYY probable starter (tough)
        {"key_mlbam": 2, "PAVE_PLUS": 1.2},  # BOS probable starter (easy)
    ])
    schedule_games = _schedule_games(home_pitcher=1, away_pitcher=2)

    result = game_picks.compute_game_win_probabilities(confidence, pave, schedule_games)

    # home_composite = .25*(1.1+1.05+1.0+1.0) = 1.0375; away_composite = .25*(0.9+0.95+1.0+1.0) = .9625
    # home faces away's pitching: .6*1.2 + .4*1.1 = 1.16 -> home_rating = 1.0375*1.16 = 1.2035
    # away faces home's pitching: .6*0.8 + .4*0.9 = 0.84 -> away_rating = .9625*0.84 = 0.8085
    # home_win_probability = 1.2035 / (1.2035 + 0.8085)
    expected = 1.2035 / (1.2035 + 0.8085)
    assert result.iloc[0]["home_win_probability"] == pytest.approx(expected)
    assert result.iloc[0]["game_pk"] == 100


def test_missing_probable_starter_uses_neutral_multiplier():
    confidence = _confidence([
        {"team": "NYY", "pyth_Strength": 1.0, "pyth_Confidence": 1.0,
         "suppression_resistance": 1.0, "true_power": 1.0, "Bullpen_PAVE_PLUS": 1.0},
        {"team": "BOS", "pyth_Strength": 1.0, "pyth_Confidence": 1.0,
         "suppression_resistance": 1.0, "true_power": 1.0, "Bullpen_PAVE_PLUS": 1.0},
    ])
    pave = pd.DataFrame(columns=["key_mlbam", "PAVE_PLUS"])  # no one announced yet
    schedule_games = _schedule_games(home_pitcher=None, away_pitcher=None)

    result = game_picks.compute_game_win_probabilities(confidence, pave, schedule_games)

    # Every input is neutral (1.0) - a perfect coin flip.
    assert result.iloc[0]["home_win_probability"] == pytest.approx(0.5)


def test_missing_bullpen_pave_plus_column_uses_neutral_multiplier():
    # confidence.csv snapshots from before Bullpen_PAVE_PLUS existed (see
    # game_picks_backtest.py, which replays confidence.csv git history
    # going back further than that column) lack the column ENTIRELY, not
    # just have missing values in it - a real KeyError this test guards
    # against regressing.
    confidence = pd.DataFrame([
        {"team": "NYY", "pyth_Strength": 1.0, "pyth_Confidence": 1.0,
         "suppression_resistance": 1.0, "true_power": 1.0},
        {"team": "BOS", "pyth_Strength": 1.0, "pyth_Confidence": 1.0,
         "suppression_resistance": 1.0, "true_power": 1.0},
    ])
    pave = pd.DataFrame(columns=["key_mlbam", "PAVE_PLUS"])
    schedule_games = _schedule_games(home_pitcher=None, away_pitcher=None)

    result = game_picks.compute_game_win_probabilities(confidence, pave, schedule_games)

    assert result.iloc[0]["home_win_probability"] == pytest.approx(0.5)


def test_power_a_plus_blends_into_pitching_quality_when_weight_nonzero(monkeypatch):
    monkeypatch.setattr(game_picks.config, "GAME_PICK_SUSCEPTIBILITY_WEIGHT", 0.5)
    confidence = _confidence([
        {"team": "NYY", "pyth_Strength": 1.1, "pyth_Confidence": 1.05,
         "suppression_resistance": 1.0, "true_power": 1.0,
         "Bullpen_PAVE_PLUS": 0.9, "Bullpen_Power_A_PLUS": 0.95},
        {"team": "BOS", "pyth_Strength": 0.9, "pyth_Confidence": 0.95,
         "suppression_resistance": 1.0, "true_power": 1.0,
         "Bullpen_PAVE_PLUS": 1.1, "Bullpen_Power_A_PLUS": 1.05},
    ])
    pave = pd.DataFrame([
        {"key_mlbam": 1, "PAVE_PLUS": 0.8, "Power_A_PLUS": 0.85},  # NYY probable starter
        {"key_mlbam": 2, "PAVE_PLUS": 1.2, "Power_A_PLUS": 1.15},  # BOS probable starter
    ])
    schedule_games = _schedule_games(home_pitcher=1, away_pitcher=2)

    result = game_picks.compute_game_win_probabilities(confidence, pave, schedule_games)

    # home_composite = 1.0375; away_composite = .9625 (same as the PAVE_PLUS-only test)
    # home faces away's (BOS's) pitching:
    #   pave_quality = .6*1.2 + .4*1.1 = 1.16
    #   power_a_quality = .6*1.15 + .4*1.05 = 1.11
    #   blended (weight .5) = .5*1.16 + .5*1.11 = 1.135 -> home_rating = 1.0375*1.135 = 1.1775625
    # away faces home's (NYY's) pitching:
    #   pave_quality = .6*0.8 + .4*0.9 = .84
    #   power_a_quality = .6*0.85 + .4*0.95 = .89
    #   blended = .5*.84 + .5*.89 = .865 -> away_rating = .9625*.865 = .8325625
    expected = 1.1775625 / (1.1775625 + 0.8325625)
    assert result.iloc[0]["home_win_probability"] == pytest.approx(expected)


def test_missing_power_a_plus_columns_uses_neutral_multiplier_even_with_nonzero_weight(monkeypatch):
    # confidence.csv/pave.csv snapshots from before Power_A_PLUS existed
    # lack the columns ENTIRELY - must degrade gracefully to a neutral
    # multiplier even when GAME_PICK_SUSCEPTIBILITY_WEIGHT is active, not crash.
    monkeypatch.setattr(game_picks.config, "GAME_PICK_SUSCEPTIBILITY_WEIGHT", 0.5)
    confidence = pd.DataFrame([
        {"team": "NYY", "pyth_Strength": 1.0, "pyth_Confidence": 1.0,
         "suppression_resistance": 1.0, "true_power": 1.0, "Bullpen_PAVE_PLUS": 1.0},
        {"team": "BOS", "pyth_Strength": 1.0, "pyth_Confidence": 1.0,
         "suppression_resistance": 1.0, "true_power": 1.0, "Bullpen_PAVE_PLUS": 1.0},
    ])
    pave = pd.DataFrame([
        {"key_mlbam": 1, "PAVE_PLUS": 1.0},
        {"key_mlbam": 2, "PAVE_PLUS": 1.0},
    ])
    schedule_games = _schedule_games(home_pitcher=1, away_pitcher=2)

    result = game_picks.compute_game_win_probabilities(confidence, pave, schedule_games)

    assert result.iloc[0]["home_win_probability"] == pytest.approx(0.5)


def test_rating_floor_prevents_division_blowup():
    confidence = _confidence([
        {"team": "NYY", "pyth_Strength": 0.0, "pyth_Confidence": 0.0,
         "suppression_resistance": 0.0, "true_power": 0.0, "Bullpen_PAVE_PLUS": 1.0},
        {"team": "BOS", "pyth_Strength": 1.0, "pyth_Confidence": 1.0,
         "suppression_resistance": 1.0, "true_power": 1.0, "Bullpen_PAVE_PLUS": 1.0},
    ])
    pave = pd.DataFrame(columns=["key_mlbam", "PAVE_PLUS"])
    schedule_games = _schedule_games(home_pitcher=None, away_pitcher=None)

    result = game_picks.compute_game_win_probabilities(confidence, pave, schedule_games)

    # NYY's composite is 0 -> its rating floors at GAME_PICK_RATING_FLOOR
    # (0.05) rather than 0, so the division stays well-defined and NYY still
    # gets a real (very low, non-crashing) win probability.
    assert 0.0 < result.iloc[0]["home_win_probability"] < 0.1
