"""Unit tests for scripts/recommend_bets.py's own remaining logic -
_load_target_date_picks' hard-refuse behavior. The real edge/Kelly
decision logic itself (formerly this script's own build_bet_recommendations)
now lives in game_predictions.advise_bets (shared with pipeline.run()) -
see tests/test_game_predictions.py for its full test coverage. Mirrors
tests/test_backfill_market_odds.py's established importlib.util
module-loading pattern for scripts/ files."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "recommend_bets.py"


def _load_module():
    sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location("recommend_bets", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_target_date_picks_refuses_when_log_missing(tmp_path):
    module = _load_module()
    log_path = str(tmp_path / "game_predictions.csv")

    with pytest.raises(SystemExit):
        module._load_target_date_picks(log_path, pd.Timestamp("2026-08-24"))


def test_load_target_date_picks_refuses_when_no_rows_for_date(tmp_path):
    module = _load_module()
    log_path = str(tmp_path / "game_predictions.csv")
    pd.DataFrame([{
        "date": pd.Timestamp("2026-08-20"), "game_pk": 1, "home_team": "NYY", "away_team": "BOS",
        "predicted_winner": "NYY", "predicted_probability": 0.6, "metric": "GamePick_Win_Probability",
        "game_played": pd.NA,
    }]).to_csv(log_path, index=False)

    with pytest.raises(SystemExit):
        module._load_target_date_picks(log_path, pd.Timestamp("2026-08-24"))


def test_load_target_date_picks_refuses_when_all_resolved(tmp_path):
    module = _load_module()
    log_path = str(tmp_path / "game_predictions.csv")
    pd.DataFrame([{
        "date": pd.Timestamp("2026-08-24"), "game_pk": 1, "home_team": "NYY", "away_team": "BOS",
        "predicted_winner": "NYY", "predicted_probability": 0.6, "metric": "GamePick_Win_Probability",
        "game_played": 1,
    }]).to_csv(log_path, index=False)

    with pytest.raises(SystemExit):
        module._load_target_date_picks(log_path, pd.Timestamp("2026-08-24"))


def test_load_target_date_picks_returns_only_pending_rows(tmp_path):
    module = _load_module()
    log_path = str(tmp_path / "game_predictions.csv")
    pd.DataFrame([
        {
            "date": pd.Timestamp("2026-08-24"), "game_pk": 1, "home_team": "NYY", "away_team": "BOS",
            "predicted_winner": "NYY", "predicted_probability": 0.6, "metric": "GamePick_Win_Probability",
            "game_played": pd.NA,
        },
        {
            "date": pd.Timestamp("2026-08-24"), "game_pk": 2, "home_team": "LAD", "away_team": "SF",
            "predicted_winner": "LAD", "predicted_probability": 0.6, "metric": "GamePick_Win_Probability",
            "game_played": 1,
        },
    ]).to_csv(log_path, index=False)

    pending = module._load_target_date_picks(log_path, pd.Timestamp("2026-08-24"))

    assert list(pending["game_pk"]) == [1]
