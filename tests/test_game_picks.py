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


def test_compute_game_win_probabilities_exact_arithmetic():
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
