"""End-to-end smoke test for scripts/build_hitter_hit_predictions.py's
wiring: synthetic wave.csv/pave.csv/confidence.csv + a monkeypatched
schedule fetch + a saved fake model artifact, checked that
hitter_hit_predictions.csv gets written correctly, and that a failed/empty
schedule fetch or a missing model artifact leaves prior output untouched
(resilience), mirroring test_build_dfs_rankings_script.py's exact pattern."""

import datetime
import importlib.util
import sys
from pathlib import Path

import pandas as pd

from mlb_metrics import ml_models

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_hitter_hit_predictions.py"


def _load_module():
    sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location("build_hitter_hit_predictions", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ConstantProbaModel:
    def __init__(self, positive_proba):
        self.positive_proba = positive_proba

    def predict_proba(self, X):
        import numpy as np
        return np.column_stack([1 - np.full(len(X), self.positive_proba), np.full(len(X), self.positive_proba)])


def _write_daily_csvs(data_dir):
    wave = pd.DataFrame([{
        "key_mlbam": 1, "name_first": "Test", "name_last": "Hitter", "team": "BOS",
        "PA_L": 20, "PA_R": 20, "WAVE": 0.28, "WAVE_L": 0.28, "WAVE_R": 0.28,
        "probability_L": 0.6, "probability_R": 0.6, "probability": 0.6,
        "Game_Hit_Probability": 0.70, "Consistency": 0.1, "Approach": 0.4, "Expected_Bases": 1.5,
        "Expected_BB": 0.3, "Expected_HBP": 0.1, "Expected_RBI": 0.4,
    }])
    pave = pd.DataFrame([{
        "key_mlbam": 99, "name_first": "Test", "name_last": "Pitcher", "team": "NYY",
        "at_bats": 100, "Throws": "R", "PAVE": 0.24, "PAVE_PLUS": 1.0, "Power_A_PLUS": 1.0,
        "Expected_Hits": 1.0, "Expected_Bases": 1.5, "Expected_HRs": 0.1,
    }])
    confidence = pd.DataFrame([
        {"team": "BOS", "Bullpen_PAVE": 0.25, "Park_Factor": 1.0},
        {"team": "NYY", "Bullpen_PAVE": 0.25, "Park_Factor": 1.0},
    ])
    wave.to_csv(data_dir / "wave.csv", index=False)
    pave.to_csv(data_dir / "pave.csv", index=False)
    confidence.to_csv(data_dir / "confidence.csv", index=False)


def _schedule_df():
    return pd.DataFrame([
        {"date": "2026-06-20", "team": "BOS", "opponent": "NYY", "probable_pitcher_key_mlbam": 99, "game_pk": 1, "is_home": False},
        {"date": "2026-06-20", "team": "NYY", "opponent": "BOS", "probable_pitcher_key_mlbam": pd.NA, "game_pk": 1, "is_home": True},
    ])


def test_build_hitter_hit_predictions_writes_csv(tmp_path, monkeypatch):
    module = _load_module()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_daily_csvs(data_dir)

    model_path = str(tmp_path / "model.joblib")
    ml_models.save_model(_ConstantProbaModel(0.61), model_path)
    monkeypatch.setattr(module.config, "HITTER_HIT_PROBABILITY_MODEL_PATH", model_path)
    monkeypatch.setattr(module.schedule, "fetch_probable_pitchers", lambda date: _schedule_df())

    sys.argv = ["build_hitter_hit_predictions.py", "--data-dir", str(data_dir), "--as-of-date", "2026-06-20"]
    module.main()

    result = pd.read_csv(data_dir / "hitter_hit_predictions.csv")
    assert len(result) == 1
    assert result.iloc[0]["key_mlbam"] == 1
    assert result.iloc[0]["Model_Hit_Probability"] == 0.61
    assert result.iloc[0]["opponent"] == "NYY"


def test_build_hitter_hit_predictions_missing_daily_csvs_writes_nothing(tmp_path, monkeypatch):
    module = _load_module()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    sys.argv = ["build_hitter_hit_predictions.py", "--data-dir", str(data_dir), "--as-of-date", "2026-06-20"]
    module.main()

    assert not (data_dir / "hitter_hit_predictions.csv").exists()


def test_build_hitter_hit_predictions_failed_schedule_leaves_existing_file_untouched(tmp_path, monkeypatch):
    module = _load_module()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_daily_csvs(data_dir)
    (data_dir / "hitter_hit_predictions.csv").write_text("stale,data\n1,2\n")

    def _boom(date):
        raise RuntimeError("statsapi is down")

    monkeypatch.setattr(module.schedule, "fetch_probable_pitchers", _boom)

    sys.argv = ["build_hitter_hit_predictions.py", "--data-dir", str(data_dir), "--as-of-date", "2026-06-20"]
    module.main()

    assert (data_dir / "hitter_hit_predictions.csv").read_text() == "stale,data\n1,2\n"


def test_build_hitter_hit_predictions_missing_model_leaves_existing_file_untouched(tmp_path, monkeypatch):
    module = _load_module()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_daily_csvs(data_dir)
    (data_dir / "hitter_hit_predictions.csv").write_text("stale,data\n1,2\n")

    monkeypatch.setattr(module.config, "HITTER_HIT_PROBABILITY_MODEL_PATH", str(tmp_path / "missing.joblib"))
    monkeypatch.setattr(module.schedule, "fetch_probable_pitchers", lambda date: _schedule_df())

    sys.argv = ["build_hitter_hit_predictions.py", "--data-dir", str(data_dir), "--as-of-date", "2026-06-20"]
    module.main()

    assert (data_dir / "hitter_hit_predictions.csv").read_text() == "stale,data\n1,2\n"
