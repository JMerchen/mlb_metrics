import pandas as pd
import pytest

from mlb_metrics import config, nfl_game_picks


def _master(rows):
    """rows: list of dicts with team, pyth_Strength, pyth_Confidence,
    defensive_edge, true_power, turnover_margin, points_per_drive."""
    return pd.DataFrame(rows)


def test_team_composite_exact_arithmetic():
    master = _master([
        {"team": "KC", "pyth_Strength": 1.1, "pyth_Confidence": 1.05, "defensive_edge": 1.0, "true_power": 1.0,
         "turnover_margin": 1.0, "points_per_drive": 1.0},
    ])
    result = nfl_game_picks._team_composite(master).set_index("team")
    # (1.1+1.05+1.0+1.0+1.0+1.0)/6 = 6.15/6 = 1.025
    assert result.loc["KC", "composite"] == pytest.approx(1.025)


def _schedule_games(home_team="KC", away_team="DEN", home_qb="qb_home", away_qb="qb_away"):
    return pd.DataFrame([{
        "game_id": "2025_08_DEN_KC", "season": 2025, "week": 8,
        "home_team": home_team, "away_team": away_team,
        "home_qb_id": home_qb, "away_qb_id": away_qb,
    }])


def _weekly_epa(player_id, epa):
    return {
        "player_id": player_id, "position": "QB", "season": 2025, "week": 1,
        "game_id": f"2025_01_{player_id}",
        "attempts": 30, "completions": 20, "passing_yards": 200, "passing_tds": 1,
        "passing_interceptions": 0, "carries": 2, "rushing_yards": 5, "rushing_tds": 0,
        "passing_epa": epa,
    }


def test_build_game_features_qb_adjustment_is_zero_when_starter_matches_recent_primary():
    # Both teams' confirmed starter IS their own identified recent-primary
    # QB - same real epa lookup both ways, so the adjustment must be
    # exactly 0 regardless of the population's mean/std.
    master = _master([
        {"team": "KC", "pyth_Strength": 1.1, "pyth_Confidence": 1.05, "defensive_edge": 1.0, "true_power": 1.0,
         "turnover_margin": 1.0, "points_per_drive": 1.0},
        {"team": "DEN", "pyth_Strength": 0.9, "pyth_Confidence": 0.95, "defensive_edge": 1.0, "true_power": 1.0,
         "turnover_margin": 1.0, "points_per_drive": 1.0},
    ])
    qb_continuity = pd.DataFrame([
        {"team": "KC", "recent_primary_qb_id": "qb_home", "recent_primary_qb_epa": 5.0, "recent_primary_qb_games": 8},
        {"team": "DEN", "recent_primary_qb_id": "qb_away", "recent_primary_qb_epa": -2.0, "recent_primary_qb_games": 8},
    ])
    weekly = pd.DataFrame([_weekly_epa("qb_home", 5.0), _weekly_epa("qb_away", -2.0)])
    schedule_games = _schedule_games()

    features = nfl_game_picks.build_game_features(master, qb_continuity, weekly, schedule_games)

    row = features.iloc[0]
    assert row["home_qb_adjustment"] == pytest.approx(0.0)
    assert row["away_qb_adjustment"] == pytest.approx(0.0)
    # home_composite = (1.1+1.05+1.0+1.0+1.0+1.0)/6 = 1.025; away_composite = (0.9+0.95+1.0+1.0+1.0+1.0)/6 = 0.975
    assert row["home_composite"] == pytest.approx(1.025)
    assert row["away_composite"] == pytest.approx(0.975)


def test_build_game_features_qb_adjustment_favors_confirmed_starter_over_backup():
    # KC's real recent-primary QB (the one who's actually been playing) is
    # a struggling backup, but the CONFIRMED starter for this specific game
    # is the real, better incumbent returning from injury - the adjustment
    # must be POSITIVE (a real quality upgrade over what recent snaps show).
    master = _master([
        {"team": "KC", "pyth_Strength": 1.0, "pyth_Confidence": 1.0, "defensive_edge": 1.0, "true_power": 1.0,
         "turnover_margin": 1.0, "points_per_drive": 1.0},
        {"team": "DEN", "pyth_Strength": 1.0, "pyth_Confidence": 1.0, "defensive_edge": 1.0, "true_power": 1.0,
         "turnover_margin": 1.0, "points_per_drive": 1.0},
    ])
    qb_continuity = pd.DataFrame([
        {"team": "KC", "recent_primary_qb_id": "kc_backup", "recent_primary_qb_epa": -5.0, "recent_primary_qb_games": 3},
        {"team": "DEN", "recent_primary_qb_id": "den_starter", "recent_primary_qb_epa": 2.0, "recent_primary_qb_games": 8},
    ])
    weekly = pd.DataFrame([
        _weekly_epa("kc_backup", -5.0), _weekly_epa("kc_starter", 6.0), _weekly_epa("den_starter", 2.0),
    ])
    schedule_games = _schedule_games(home_qb="kc_starter", away_qb="den_starter")

    features = nfl_game_picks.build_game_features(master, qb_continuity, weekly, schedule_games)

    row = features.iloc[0]
    assert row["home_qb_adjustment"] > 0
    assert row["away_qb_adjustment"] == pytest.approx(0.0)


def test_compute_game_win_probabilities_exact_arithmetic(monkeypatch):
    monkeypatch.setattr(nfl_game_picks.config, "NFL_QB_CONTINUITY_WEIGHT", 0.0)
    master = _master([
        {"team": "KC", "pyth_Strength": 1.1, "pyth_Confidence": 1.05, "defensive_edge": 1.0, "true_power": 1.0,
         "turnover_margin": 1.0, "points_per_drive": 1.0},
        {"team": "DEN", "pyth_Strength": 0.9, "pyth_Confidence": 0.95, "defensive_edge": 1.0, "true_power": 1.0,
         "turnover_margin": 1.0, "points_per_drive": 1.0},
    ])
    qb_continuity = pd.DataFrame([
        {"team": "KC", "recent_primary_qb_id": "qb_home", "recent_primary_qb_epa": 5.0, "recent_primary_qb_games": 8},
        {"team": "DEN", "recent_primary_qb_id": "qb_away", "recent_primary_qb_epa": -2.0, "recent_primary_qb_games": 8},
    ])
    weekly = pd.DataFrame([_weekly_epa("qb_home", 5.0), _weekly_epa("qb_away", -2.0)])
    schedule_games = _schedule_games()

    result = nfl_game_picks.compute_game_win_probabilities(master, qb_continuity, weekly, schedule_games)

    # QB continuity weight pinned to 0, so this reduces to a pure composite ratio:
    # home_composite = 1.025, away_composite = 0.975
    expected = 1.025 / (1.025 + 0.975)
    assert result.iloc[0]["home_win_probability"] == pytest.approx(expected)
    assert result.iloc[0]["game_id"] == "2025_08_DEN_KC"


def test_apply_calibration_is_a_noop_with_no_saved_model(monkeypatch, tmp_path):
    monkeypatch.setattr(
        nfl_game_picks.config, "NFL_GAME_PICK_CALIBRATION_MODEL_PATH", str(tmp_path / "does_not_exist.joblib")
    )
    win_probabilities = pd.DataFrame([{"game_id": "g1", "home_win_probability": 0.6}])

    result = nfl_game_picks.apply_calibration(win_probabilities)

    assert result.equals(win_probabilities)
