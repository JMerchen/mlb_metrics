import pandas as pd
import pytest

from mlb_metrics import config, dfs_ml, ml_models


def _wave_row(key_mlbam, team, **overrides):
    row = {
        "key_mlbam": key_mlbam, "team": team,
        "WAVE": 0.28, "WAVE_L": 0.30, "WAVE_R": 0.27, "PA_L": 20, "PA_R": 40,
        "probability": 0.65, "Game_Hit_Probability": 0.70, "Consistency": 0.05,
        "Approach": 0.45, "Expected_Bases": 1.5,
        "Expected_BB": 0.3, "Expected_HBP": 0.1, "Expected_RBI": 0.4,
        "Exit_Velo": 90.0, "Barrel_Rate": 0.08, "xBA": 0.25, "xwOBA": 0.32,
        "Fastball_WAVE": 0.30, "Breaking_WAVE": 0.20, "Offspeed_WAVE": 0.15,
        "Whiff_Rate": 0.22, "Chase_Rate": 0.28,
        "WAVE_Home": 0.32, "WAVE_Away": 0.24,
    }
    row.update(overrides)
    return row


def test_build_hitter_features_joins_matchup_ingredients():
    # is_home=True means the game's venue is BOS's own park (see
    # matchup.py's identical venue_team logic) - Park_Factor must be
    # looked up under "BOS", not the opponent "NYY".
    wave = pd.DataFrame([_wave_row(1, "BOS")])
    pave = pd.DataFrame([{
        "key_mlbam": 99, "PAVE": 0.24,
        "Fastball_Rate": 0.60, "Breaking_Rate": 0.25, "Offspeed_Rate": 0.15,
    }])
    confidence = pd.DataFrame([
        {"team": "NYY", "Bullpen_PAVE": 0.26, "Park_Factor": 0.95},
        {"team": "BOS", "Bullpen_PAVE": 0.25, "Park_Factor": 1.05},
    ])
    schedule_df = pd.DataFrame([{"team": "BOS", "opponent": "NYY", "is_home": True, "probable_pitcher_key_mlbam": 99}])
    matchup_probability = pd.DataFrame([{"key_mlbam": 1, "Matchup_Hit_Probability": 0.72}])

    result = dfs_ml.build_hitter_features(wave, pave, confidence, schedule_df, matchup_probability).set_index("key_mlbam")

    assert result.loc[1, "starter_PAVE"] == pytest.approx(0.24)
    assert result.loc[1, "Bullpen_PAVE"] == pytest.approx(0.26)
    assert result.loc[1, "Park_Factor"] == pytest.approx(1.05)
    assert result.loc[1, "is_home"] == True  # noqa: E712
    assert result.loc[1, "Matchup_Hit_Probability"] == pytest.approx(0.72)
    assert result.loc[1, "Whiff_Rate"] == pytest.approx(0.22)
    assert result.loc[1, "Chase_Rate"] == pytest.approx(0.28)
    assert result.loc[1, "Expected_Bases"] == pytest.approx(1.5)
    assert result.loc[1, "Fastball_WAVE"] == pytest.approx(0.30)
    assert result.loc[1, "WAVE_Home"] == pytest.approx(0.32)
    assert result.loc[1, "WAVE_Away"] == pytest.approx(0.24)
    assert result.loc[1, "starter_fastball_rate"] == pytest.approx(0.60)
    assert result.loc[1, "starter_breaking_rate"] == pytest.approx(0.25)
    assert result.loc[1, "starter_offspeed_rate"] == pytest.approx(0.15)
    assert list(result.reset_index().columns) == ["key_mlbam"] + dfs_ml.HITTER_FEATURE_COLUMNS


def test_build_hitter_features_falls_back_to_null_when_pitch_family_columns_missing():
    # An older wave.csv/pave.csv snapshot predating pitch-family signals
    # has neither the batter's own Fastball_WAVE/etc. nor the starter's
    # own Fastball_Rate/etc. - degrades to null (not a KeyError), same
    # convention WAVE_L/WAVE_R's own fallback test establishes.
    wave = pd.DataFrame([{
        "key_mlbam": 1, "team": "BOS", "WAVE": 0.28, "WAVE_L": 0.30, "WAVE_R": 0.27,
        "PA_L": 20, "PA_R": 40, "probability": 0.65, "Game_Hit_Probability": 0.70,
        "Consistency": 0.05, "Approach": 0.45, "Expected_Bases": 1.5,
        "Expected_BB": 0.3, "Expected_HBP": 0.1, "Expected_RBI": 0.4,
        "Exit_Velo": 90.0, "Barrel_Rate": 0.08, "xBA": 0.25, "xwOBA": 0.32,
    }])
    pave = pd.DataFrame([{"key_mlbam": 99, "PAVE": 0.24}])
    confidence = pd.DataFrame([{"team": "NYY", "Bullpen_PAVE": 0.26, "Park_Factor": 0.95}])
    schedule_df = pd.DataFrame([{"team": "BOS", "opponent": "NYY", "is_home": False, "probable_pitcher_key_mlbam": 99}])
    matchup_probability = pd.DataFrame([{"key_mlbam": 1, "Matchup_Hit_Probability": 0.72}])

    result = dfs_ml.build_hitter_features(wave, pave, confidence, schedule_df, matchup_probability).set_index("key_mlbam")

    assert pd.isna(result.loc[1, "Fastball_WAVE"])
    assert pd.isna(result.loc[1, "starter_fastball_rate"])
    assert pd.isna(result.loc[1, "Whiff_Rate"])
    assert pd.isna(result.loc[1, "WAVE_Home"])
    assert pd.isna(result.loc[1, "WAVE_Away"])


def test_build_hitter_features_falls_back_to_wave_when_wave_l_r_missing():
    # Regression test: an old wave.csv snapshot predating platoon-awareness
    # has no WAVE_L/WAVE_R columns at all - build_hitter_features must fall
    # back to WAVE rather than KeyError, same as matchup.py's _platoon_wave.
    wave = pd.DataFrame([{
        "key_mlbam": 1, "team": "BOS", "WAVE": 0.28, "PA_L": 20, "PA_R": 40,
        "probability": 0.65, "Game_Hit_Probability": 0.70, "Consistency": 0.05,
        "Approach": 0.45, "Expected_Bases": 1.5,
        "Expected_BB": 0.3, "Expected_HBP": 0.1, "Expected_RBI": 0.4,
        "Exit_Velo": 90.0, "Barrel_Rate": 0.08, "xBA": 0.25, "xwOBA": 0.32,
    }])
    pave = pd.DataFrame([{"key_mlbam": 99, "PAVE": 0.24}])
    confidence = pd.DataFrame([{"team": "NYY", "Bullpen_PAVE": 0.26, "Park_Factor": 0.95}])
    schedule_df = pd.DataFrame([{"team": "BOS", "opponent": "NYY", "is_home": False, "probable_pitcher_key_mlbam": 99}])
    matchup_probability = pd.DataFrame([{"key_mlbam": 1, "Matchup_Hit_Probability": 0.72}])

    result = dfs_ml.build_hitter_features(wave, pave, confidence, schedule_df, matchup_probability).set_index("key_mlbam")

    assert result.loc[1, "WAVE_L"] == pytest.approx(0.28)
    assert result.loc[1, "WAVE_R"] == pytest.approx(0.28)


def test_build_hitter_features_excludes_batter_with_no_resolvable_matchup():
    wave = pd.DataFrame([_wave_row(1, "BOS")])
    pave = pd.DataFrame([{"key_mlbam": 99, "PAVE": 0.24}])
    confidence = pd.DataFrame([{"team": "NYY", "Bullpen_PAVE": 0.26, "Park_Factor": 1.05}])
    schedule_df = pd.DataFrame([{"team": "BOS", "opponent": "NYY", "is_home": True, "probable_pitcher_key_mlbam": 99}])
    matchup_probability = pd.DataFrame(columns=["key_mlbam", "Matchup_Hit_Probability"])  # no match for batter 1

    result = dfs_ml.build_hitter_features(wave, pave, confidence, schedule_df, matchup_probability)

    assert result.empty


def test_hitter_feature_matrix_fills_missing_and_coerces_is_home():
    features = pd.DataFrame([{
        "key_mlbam": 1, "WAVE": 0.28, "WAVE_L": 0.30, "WAVE_R": 0.27, "PA_L": 20, "PA_R": 40,
        "probability": 0.65, "Game_Hit_Probability": 0.70, "Consistency": 0.05, "Approach": 0.45,
        "Expected_Bases": 1.5, "Exit_Velo": 90.0, "Barrel_Rate": 0.08, "xBA": pd.NA, "xwOBA": 0.32,
        "starter_PAVE": pd.NA, "Bullpen_PAVE": 0.26, "Park_Factor": pd.NA,
        "is_home": True, "Matchup_Hit_Probability": 0.72,
        "Fastball_WAVE": pd.NA, "starter_fastball_rate": pd.NA,
        "Whiff_Rate": pd.NA,
    }])

    X = dfs_ml.hitter_feature_matrix(features)

    assert list(X.columns) == dfs_ml.HITTER_FEATURE_COLUMNS
    assert X.loc[0, "starter_PAVE"] == 0
    assert X.loc[0, "Park_Factor"] == 0
    assert X.loc[0, "xBA"] == 0
    assert X.loc[0, "is_home"] == 1.0
    assert X.loc[0, "Fastball_WAVE"] == 0
    assert X.loc[0, "starter_fastball_rate"] == 0
    assert X.loc[0, "Whiff_Rate"] == 0


class _ConstantProbaModel:
    """Minimal stand-in for a fitted sklearn classifier - always predicts a
    fixed positive-class probability, in predict_proba's [n_samples, 2]
    shape."""

    def __init__(self, positive_proba):
        self.positive_proba = positive_proba

    def predict_proba(self, X):
        import numpy as np
        return np.column_stack([1 - np.full(len(X), self.positive_proba), np.full(len(X), self.positive_proba)])


def test_predict_hitter_hit_probability_no_model_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HITTER_HIT_PROBABILITY_MODEL_PATH", str(tmp_path / "missing.joblib"))

    hitter_features = dfs_ml.build_hitter_features(
        pd.DataFrame([_wave_row(1, "BOS")]),
        pd.DataFrame([{"key_mlbam": 99, "PAVE": 0.24}]),
        pd.DataFrame([{"team": "NYY", "Bullpen_PAVE": 0.26, "Park_Factor": 1.05}]),
        pd.DataFrame([{"team": "BOS", "opponent": "NYY", "is_home": True, "probable_pitcher_key_mlbam": 99}]),
        pd.DataFrame([{"key_mlbam": 1, "Matchup_Hit_Probability": 0.72}]),
    )

    result = dfs_ml.predict_hitter_hit_probability(hitter_features)

    assert result.empty
    assert list(result.columns) == ["key_mlbam", "Model_Hit_Probability"]


def test_predict_hitter_hit_probability_with_model_returns_probability(tmp_path, monkeypatch):
    model_path = str(tmp_path / "hit_probability.joblib")
    ml_models.save_model(_ConstantProbaModel(0.63), model_path)
    monkeypatch.setattr(config, "HITTER_HIT_PROBABILITY_MODEL_PATH", model_path)

    hitter_features = dfs_ml.build_hitter_features(
        pd.DataFrame([_wave_row(1, "BOS")]),
        pd.DataFrame([{"key_mlbam": 99, "PAVE": 0.24}]),
        pd.DataFrame([{"team": "NYY", "Bullpen_PAVE": 0.26, "Park_Factor": 1.05}]),
        pd.DataFrame([{"team": "BOS", "opponent": "NYY", "is_home": True, "probable_pitcher_key_mlbam": 99}]),
        pd.DataFrame([{"key_mlbam": 1, "Matchup_Hit_Probability": 0.72}]),
    )

    result = dfs_ml.predict_hitter_hit_probability(hitter_features).set_index("key_mlbam")

    assert result.loc[1, "Model_Hit_Probability"] == pytest.approx(0.63)
