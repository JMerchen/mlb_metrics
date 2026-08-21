"""Unit tests for scripts/backfill_market_odds.py's core
backfill_market_probabilities logic - injected fake fetch_fn, no real
network, mirroring tests/test_backtest_same_game_diversification.py's
established importlib.util module-loading pattern for scripts/ files."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "backfill_market_odds.py"


def _load_module():
    sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location("backfill_market_odds", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _log_row(date, game_pk, home_team, away_team, market_home_win_probability=None):
    return {
        "date": pd.Timestamp(date), "game_pk": game_pk, "home_team": home_team, "away_team": away_team,
        "predicted_winner": home_team, "predicted_probability": 0.6, "above_threshold": True,
        "metric": "GamePick_Win_Probability", "actual_winner": home_team, "game_played": 1,
        "model_version": "v1", "market_home_win_probability": market_home_win_probability,
    }


def test_backfill_fills_only_null_rows_for_the_given_dates(tmp_path):
    module = _load_module()
    log_path = str(tmp_path / "game_predictions.csv")
    pd.DataFrame([
        _log_row("2026-08-16", 1, "NYY", "BOS"),
        _log_row("2026-08-16", 2, "LAD", "SF"),
    ]).to_csv(log_path, index=False)

    def fake_fetch(date):
        return pd.DataFrame([
            {"home_team": "NYY", "away_team": "BOS", "market_home_win_probability": 0.62, "market_provider": "DraftKings"},
            {"home_team": "LAD", "away_team": "SF", "market_home_win_probability": 0.55, "market_provider": "DraftKings"},
        ])

    result = module.backfill_market_probabilities(log_path, fake_fetch, [pd.Timestamp("2026-08-16")])

    by_pk = result.set_index("game_pk")
    assert by_pk.loc[1, "market_home_win_probability"] == 0.62
    assert by_pk.loc[2, "market_home_win_probability"] == 0.55


def test_backfill_never_overwrites_an_existing_real_value(tmp_path):
    module = _load_module()
    log_path = str(tmp_path / "game_predictions.csv")
    # A value already backfilled by an earlier run - must survive untouched
    # even though fake_fetch would return a different number for it.
    pd.DataFrame([_log_row("2026-08-16", 1, "NYY", "BOS", market_home_win_probability=0.70)]).to_csv(
        log_path, index=False
    )

    def fake_fetch(date):
        return pd.DataFrame([
            {"home_team": "NYY", "away_team": "BOS", "market_home_win_probability": 0.11, "market_provider": "DraftKings"},
        ])

    result = module.backfill_market_probabilities(log_path, fake_fetch, [pd.Timestamp("2026-08-16")])

    assert result.iloc[0]["market_home_win_probability"] == 0.70


def test_backfill_leaves_an_unmatched_row_null(tmp_path):
    module = _load_module()
    log_path = str(tmp_path / "game_predictions.csv")
    pd.DataFrame([_log_row("2026-08-16", 1, "NYY", "BOS")]).to_csv(log_path, index=False)

    def fake_fetch(date):
        # Real ESPN simply never had this matchup that day.
        return pd.DataFrame([
            {"home_team": "LAD", "away_team": "SF", "market_home_win_probability": 0.55, "market_provider": "DraftKings"},
        ])

    result = module.backfill_market_probabilities(log_path, fake_fetch, [pd.Timestamp("2026-08-16")])

    assert pd.isna(result.iloc[0]["market_home_win_probability"])


def test_backfill_one_bad_date_does_not_block_the_others(tmp_path):
    module = _load_module()
    log_path = str(tmp_path / "game_predictions.csv")
    pd.DataFrame([
        _log_row("2026-08-15", 1, "NYY", "BOS"),
        _log_row("2026-08-16", 2, "LAD", "SF"),
    ]).to_csv(log_path, index=False)

    def fake_fetch(date):
        if pd.Timestamp(date) == pd.Timestamp("2026-08-15"):
            raise RuntimeError("ESPN is unreachable")
        return pd.DataFrame([
            {"home_team": "LAD", "away_team": "SF", "market_home_win_probability": 0.55, "market_provider": "DraftKings"},
        ])

    result = module.backfill_market_probabilities(
        log_path, fake_fetch, [pd.Timestamp("2026-08-15"), pd.Timestamp("2026-08-16")]
    )

    by_pk = result.set_index("game_pk")
    assert pd.isna(by_pk.loc[1, "market_home_win_probability"])
    assert by_pk.loc[2, "market_home_win_probability"] == 0.55
