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

    result = nfl_game_picks.compute_game_win_probabilities(
        master, qb_continuity, weekly, schedule_games, home_field_weight=0.0
    )

    # QB continuity weight pinned to 0, home-field weight pinned to 0, so
    # this reduces to a pure composite ratio:
    # home_composite = 1.025, away_composite = 0.975
    expected = 1.025 / (1.025 + 0.975)
    assert result.iloc[0]["home_win_probability"] == pytest.approx(expected)
    assert result.iloc[0]["game_id"] == "2025_08_DEN_KC"


def test_compute_game_win_probabilities_home_field_weight_pushes_home_side(monkeypatch):
    # Real follow-up (2026-09-04 - "a little push or pull from home/
    # away"): two otherwise-IDENTICAL teams should split exactly 50/50 at
    # home_field_weight=0.0, and the home side should gain ground as the
    # weight increases - a real regression guard for the new term.
    monkeypatch.setattr(nfl_game_picks.config, "NFL_QB_CONTINUITY_WEIGHT", 0.0)
    master = _master([
        {"team": "KC", "pyth_Strength": 1.0, "pyth_Confidence": 1.0, "defensive_edge": 1.0, "true_power": 1.0,
         "turnover_margin": 1.0, "points_per_drive": 1.0},
        {"team": "DEN", "pyth_Strength": 1.0, "pyth_Confidence": 1.0, "defensive_edge": 1.0, "true_power": 1.0,
         "turnover_margin": 1.0, "points_per_drive": 1.0},
    ])
    qb_continuity = pd.DataFrame([
        {"team": "KC", "recent_primary_qb_id": "qb_home", "recent_primary_qb_epa": 0.0, "recent_primary_qb_games": 8},
        {"team": "DEN", "recent_primary_qb_id": "qb_away", "recent_primary_qb_epa": 0.0, "recent_primary_qb_games": 8},
    ])
    weekly = pd.DataFrame([_weekly_epa("qb_home", 0.0), _weekly_epa("qb_away", 0.0)])
    schedule_games = _schedule_games()

    no_home_field = nfl_game_picks.compute_game_win_probabilities(
        master, qb_continuity, weekly, schedule_games, home_field_weight=0.0
    )
    with_home_field = nfl_game_picks.compute_game_win_probabilities(
        master, qb_continuity, weekly, schedule_games, home_field_weight=0.05
    )

    assert no_home_field.iloc[0]["home_win_probability"] == pytest.approx(0.5)
    assert with_home_field.iloc[0]["home_win_probability"] > 0.5


def test_apply_calibration_is_a_noop_with_no_saved_model(monkeypatch, tmp_path):
    monkeypatch.setattr(
        nfl_game_picks.config, "NFL_GAME_PICK_CALIBRATION_MODEL_PATH", str(tmp_path / "does_not_exist.joblib")
    )
    win_probabilities = pd.DataFrame([{"game_id": "g1", "home_win_probability": 0.6}])

    result = nfl_game_picks.apply_calibration(win_probabilities)

    assert result.equals(win_probabilities)


# --- Real fix for the ratio formula's structural "can't exceed ~59%" ceiling (2026-09-04) ---


def _master_disaggregated(rows):
    """Same shape as `_master`, plus `offensive_edge` -
    build_game_features_disaggregated's own DISAGGREGATED_SIGNAL_COLUMNS
    needs it too, unlike the plain composite tests above."""
    return pd.DataFrame(rows)


def test_build_game_features_disaggregated_returns_the_right_columns():
    master = _master_disaggregated([
        {"team": "KC", "pyth_Strength": 1.1, "pyth_Confidence": 1.05, "offensive_edge": 1.02,
         "defensive_edge": 1.0, "turnover_margin": 1.0, "points_per_drive": 1.0},
        {"team": "DEN", "pyth_Strength": 0.9, "pyth_Confidence": 0.95, "offensive_edge": 0.98,
         "defensive_edge": 1.0, "turnover_margin": 1.0, "points_per_drive": 1.0},
    ])
    qb_continuity = pd.DataFrame([
        {"team": "KC", "recent_primary_qb_id": "qb_home", "recent_primary_qb_epa": 5.0, "recent_primary_qb_games": 8},
        {"team": "DEN", "recent_primary_qb_id": "qb_away", "recent_primary_qb_epa": -2.0, "recent_primary_qb_games": 8},
    ])
    weekly = pd.DataFrame([_weekly_epa("qb_home", 5.0), _weekly_epa("qb_away", -2.0)])
    schedule_games = _schedule_games()

    features = nfl_game_picks.build_game_features_disaggregated(master, qb_continuity, weekly, schedule_games)

    assert list(features.columns) == (
        ["game_id", "season", "week", "home_team", "away_team"] + nfl_game_picks.DISAGGREGATED_FEATURE_COLUMNS
    )
    row = features.iloc[0]
    assert row["home_pyth_Strength"] == pytest.approx(1.1)
    assert row["away_offensive_edge"] == pytest.approx(0.98)
    assert row["home_qb_adjustment"] == pytest.approx(0.0)  # confirmed starter IS the recent-primary QB


def test_apply_ml_model_is_a_noop_with_no_saved_model(monkeypatch, tmp_path):
    monkeypatch.setattr(
        nfl_game_picks.config, "NFL_GAME_PICK_WIN_PROBABILITY_MODEL_PATH", str(tmp_path / "does_not_exist.joblib")
    )
    win_probabilities = pd.DataFrame([{"game_id": "2025_08_DEN_KC", "home_win_probability": 0.6}])
    master = _master([
        {"team": "KC", "pyth_Strength": 1.0, "pyth_Confidence": 1.0, "defensive_edge": 1.0, "true_power": 1.0,
         "turnover_margin": 1.0, "points_per_drive": 1.0},
        {"team": "DEN", "pyth_Strength": 1.0, "pyth_Confidence": 1.0, "defensive_edge": 1.0, "true_power": 1.0,
         "turnover_margin": 1.0, "points_per_drive": 1.0},
    ])
    qb_continuity = pd.DataFrame([
        {"team": "KC", "recent_primary_qb_id": "qb_home", "recent_primary_qb_epa": 0.0, "recent_primary_qb_games": 8},
        {"team": "DEN", "recent_primary_qb_id": "qb_away", "recent_primary_qb_epa": 0.0, "recent_primary_qb_games": 8},
    ])
    weekly = pd.DataFrame([_weekly_epa("qb_home", 0.0), _weekly_epa("qb_away", 0.0)])

    result = nfl_game_picks.apply_ml_model(win_probabilities, master, qb_continuity, weekly, _schedule_games())

    assert result.equals(win_probabilities)


def test_apply_ml_model_overwrites_home_win_probability_with_a_real_artifact(monkeypatch, tmp_path):
    # A real, tiny sklearn LogisticRegression - fit on synthetic data
    # engineered so it's guaranteed to predict something far from 0.5 on
    # the real test game below (a genuinely different number than the
    # ratio heuristic's own compressed range would ever produce) - a real
    # regression guard that apply_ml_model actually uses the model's OWN
    # prediction, not just something that happens to look plausible.
    from sklearn.linear_model import LogisticRegression

    X_fit = pd.DataFrame({
        "home_composite": [1.2, 0.8, 1.3, 0.7],
        "away_composite": [0.8, 1.2, 0.7, 1.3],
        "home_qb_adjustment": [0.0, 0.0, 0.0, 0.0],
        "away_qb_adjustment": [0.0, 0.0, 0.0, 0.0],
    })
    y_fit = [1, 0, 1, 0]
    model = LogisticRegression().fit(X_fit, y_fit)

    model_path = tmp_path / "nfl_game_pick_win_probability_model.joblib"
    nfl_game_picks.ml_models.save_model(
        {"model": model, "feature_columns": nfl_game_picks.GAME_PICK_FEATURE_COLUMNS}, str(model_path)
    )
    monkeypatch.setattr(nfl_game_picks.config, "NFL_GAME_PICK_WIN_PROBABILITY_MODEL_PATH", str(model_path))

    win_probabilities = pd.DataFrame([{"game_id": "2025_08_DEN_KC", "home_win_probability": 0.55}])
    master = _master([
        {"team": "KC", "pyth_Strength": 1.3, "pyth_Confidence": 1.3, "defensive_edge": 1.3, "true_power": 1.3,
         "turnover_margin": 1.3, "points_per_drive": 1.3},
        {"team": "DEN", "pyth_Strength": 0.7, "pyth_Confidence": 0.7, "defensive_edge": 0.7, "true_power": 0.7,
         "turnover_margin": 0.7, "points_per_drive": 0.7},
    ])
    qb_continuity = pd.DataFrame([
        {"team": "KC", "recent_primary_qb_id": "qb_home", "recent_primary_qb_epa": 0.0, "recent_primary_qb_games": 8},
        {"team": "DEN", "recent_primary_qb_id": "qb_away", "recent_primary_qb_epa": 0.0, "recent_primary_qb_games": 8},
    ])
    weekly = pd.DataFrame([_weekly_epa("qb_home", 0.0), _weekly_epa("qb_away", 0.0)])

    result = nfl_game_picks.apply_ml_model(win_probabilities, master, qb_continuity, weekly, _schedule_games())

    assert result.iloc[0]["home_win_probability"] != pytest.approx(0.55)  # real overwrite, not the original heuristic
    assert result.iloc[0]["home_win_probability"] == pytest.approx(model.predict_proba(
        pd.DataFrame([{"home_composite": 1.3, "away_composite": 0.7, "home_qb_adjustment": 0.0, "away_qb_adjustment": 0.0}])
    )[0, 1])


def test_apply_ml_model_falls_back_per_row_for_a_team_missing_from_master(monkeypatch, tmp_path):
    # A real degenerate-input guard: a game whose team isn't in `master`
    # at all (not expected in practice) keeps its ORIGINAL heuristic
    # probability rather than a fabricated ML prediction built from a
    # filled-in-zero feature row.
    from sklearn.linear_model import LogisticRegression

    X_fit = pd.DataFrame({
        "home_composite": [1.2, 0.8], "away_composite": [0.8, 1.2],
        "home_qb_adjustment": [0.0, 0.0], "away_qb_adjustment": [0.0, 0.0],
    })
    model = LogisticRegression().fit(X_fit, [1, 0])
    model_path = tmp_path / "model.joblib"
    nfl_game_picks.ml_models.save_model(
        {"model": model, "feature_columns": nfl_game_picks.GAME_PICK_FEATURE_COLUMNS}, str(model_path)
    )
    monkeypatch.setattr(nfl_game_picks.config, "NFL_GAME_PICK_WIN_PROBABILITY_MODEL_PATH", str(model_path))

    win_probabilities = pd.DataFrame([{"game_id": "2025_08_DEN_KC", "home_win_probability": 0.55}])
    master = _master([
        {"team": "KC", "pyth_Strength": 1.0, "pyth_Confidence": 1.0, "defensive_edge": 1.0, "true_power": 1.0,
         "turnover_margin": 1.0, "points_per_drive": 1.0},
        # DEN deliberately absent from master - a real missing-team case.
    ])
    qb_continuity = pd.DataFrame([
        {"team": "KC", "recent_primary_qb_id": "qb_home", "recent_primary_qb_epa": 0.0, "recent_primary_qb_games": 8},
    ])
    weekly = pd.DataFrame([_weekly_epa("qb_home", 0.0)])

    result = nfl_game_picks.apply_ml_model(win_probabilities, master, qb_continuity, weekly, _schedule_games())

    assert result.iloc[0]["home_win_probability"] == pytest.approx(0.55)  # unchanged - real fallback


# --- Real follow-up: defer to market on large disagreement (2026-09-05) ---


def _win_probs(rows):
    return pd.DataFrame(rows)


def _market(rows):
    return pd.DataFrame(rows)


def test_apply_market_tiebreak_no_op_below_the_threshold():
    win_probabilities = _win_probs([
        {"game_id": "g1", "home_team": "KC", "away_team": "DEN", "home_win_probability": 0.60},
    ])
    market = _market([{"home_team": "KC", "away_team": "DEN", "market_home_win_probability": 0.55}])

    result = nfl_game_picks.apply_market_tiebreak(win_probabilities, market, disagreement_threshold=0.20)

    assert result.iloc[0]["home_win_probability"] == pytest.approx(0.60)  # 0.05 disagreement - below threshold


def test_apply_market_tiebreak_defers_to_market_above_the_threshold():
    win_probabilities = _win_probs([
        {"game_id": "g1", "home_team": "KC", "away_team": "DEN", "home_win_probability": 0.70},
    ])
    market = _market([{"home_team": "KC", "away_team": "DEN", "market_home_win_probability": 0.40}])

    result = nfl_game_picks.apply_market_tiebreak(win_probabilities, market, disagreement_threshold=0.20)

    # 0.30 real disagreement clears the 0.20 threshold - overwritten with
    # the real market's own devigged probability, not just a flipped side.
    assert result.iloc[0]["home_win_probability"] == pytest.approx(0.40)


def test_apply_market_tiebreak_is_a_noop_with_no_market_data_for_a_game():
    win_probabilities = _win_probs([
        {"game_id": "g1", "home_team": "KC", "away_team": "DEN", "home_win_probability": 0.90},
    ])
    market = _market([{"home_team": "SF", "away_team": "LA", "market_home_win_probability": 0.10}])  # a different game

    result = nfl_game_picks.apply_market_tiebreak(win_probabilities, market, disagreement_threshold=0.20)

    assert result.iloc[0]["home_win_probability"] == pytest.approx(0.90)  # can't measure disagreement - real no-op


def test_apply_market_tiebreak_default_threshold_matches_config():
    win_probabilities = _win_probs([
        {"game_id": "g1", "home_team": "KC", "away_team": "DEN", "home_win_probability": 0.85},
    ])
    market = _market([{"home_team": "KC", "away_team": "DEN", "market_home_win_probability": 0.60}])

    result = nfl_game_picks.apply_market_tiebreak(win_probabilities, market)  # no override - real config default

    # 0.25 disagreement clears config.NFL_GAME_PICK_MARKET_DISAGREEMENT_THRESHOLD (0.20).
    assert result.iloc[0]["home_win_probability"] == pytest.approx(0.60)
