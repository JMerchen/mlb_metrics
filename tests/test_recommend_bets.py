"""Unit tests for scripts/recommend_bets.py - build_bet_recommendations
(pure, no network) and _load_target_date_picks' hard-refuse behavior,
mirroring tests/test_backfill_market_odds.py's established
importlib.util module-loading pattern for scripts/ files."""

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


def _pick_row(game_pk, home_team, away_team, predicted_winner, predicted_probability):
    return {
        "date": pd.Timestamp("2026-08-24"), "game_pk": game_pk, "home_team": home_team,
        "away_team": away_team, "predicted_winner": predicted_winner,
        "predicted_probability": predicted_probability,
    }


def _market_row(home_team, away_team, home_moneyline, away_moneyline):
    return {
        "home_team": home_team, "away_team": away_team,
        "market_home_win_probability": None, "market_provider": "DraftKings",
        "home_moneyline": home_moneyline, "away_moneyline": away_moneyline,
    }


def test_build_bet_recommendations_finds_a_real_positive_home_edge():
    module = _load_module()
    picks = pd.DataFrame([_pick_row(1, "NYY", "TOR", "NYY", 0.70)])
    # home implied = 150/250 = 0.6 -> edge = 0.70 - 0.6 = 0.10
    # away model prob = 0.30, away implied = 100/230 = 0.4348 -> edge < 0
    market = pd.DataFrame([_market_row("NYY", "TOR", -150, 130)])

    recs = module.build_bet_recommendations(picks, market, kelly_fraction_multiplier=1.0, min_edge=0.02)

    assert len(recs) == 1
    row = recs.iloc[0]
    assert row["side"] == "home"
    assert row["team"] == "NYY"
    assert row["edge"] == pytest.approx(0.10, abs=1e-6)
    assert row["kelly_stake_fraction"] > 0


def test_build_bet_recommendations_below_min_edge_recommends_nothing():
    module = _load_module()
    picks = pd.DataFrame([_pick_row(1, "NYY", "TOR", "NYY", 0.52)])
    market = pd.DataFrame([_market_row("NYY", "TOR", -115, 105)])

    recs = module.build_bet_recommendations(picks, market, kelly_fraction_multiplier=1.0, min_edge=0.02)

    assert recs.empty


def test_build_bet_recommendations_skips_a_game_missing_from_market(capsys):
    module = _load_module()
    picks = pd.DataFrame([
        _pick_row(1, "NYY", "TOR", "NYY", 0.70),
        _pick_row(2, "LAD", "SF", "LAD", 0.65),
    ])
    # Only game_pk=1's matchup has real market data.
    market = pd.DataFrame([_market_row("NYY", "TOR", -150, 130)])

    recs = module.build_bet_recommendations(picks, market, kelly_fraction_multiplier=1.0, min_edge=0.02)

    assert list(recs["game_pk"]) == [1]


def test_build_bet_recommendations_drops_both_sides_on_a_data_anomaly(capsys):
    module = _load_module()
    # An unrealistic (negative-vig) market row - both implied probabilities
    # sum to < 1, so both sides can clear min_edge at once. Real data never
    # does this (see market_odds.devig's own test asserting real vig > 0)
    # - this exercises the defensive guard, not a real scenario.
    picks = pd.DataFrame([_pick_row(1, "NYY", "TOR", "NYY", 0.50)])
    market = pd.DataFrame([_market_row("NYY", "TOR", 120, 120)])

    recs = module.build_bet_recommendations(picks, market, kelly_fraction_multiplier=1.0, min_edge=0.02)

    assert recs.empty
    assert "data-quality anomaly" in capsys.readouterr().out


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
