import pandas as pd
import pytest

from mlb_metrics import config, game_predictions


def _win_probabilities(rows):
    """rows: list of dicts with game_pk, date, home_team, away_team, home_win_probability."""
    return pd.DataFrame(rows)


def test_select_game_picks_logs_every_game_and_flags_above_threshold():
    win_probs = _win_probabilities([
        {"game_pk": 1, "date": pd.Timestamp("2026-07-22"), "home_team": "NYY", "away_team": "BOS",
         "home_win_probability": 0.65},  # clears threshold, home favored
        {"game_pk": 2, "date": pd.Timestamp("2026-07-22"), "home_team": "LAD", "away_team": "SF",
         "home_win_probability": 0.55},  # below threshold - still logged, just flagged False
    ])

    picks = game_predictions.select_game_picks(win_probs, pd.Timestamp("2026-07-22"))

    # Every scheduled game is logged now, not just the ones clearing the
    # threshold - the dashboard publishes the complete slate and highlights
    # the flagged ones instead of hiding the rest.
    assert len(picks) == 2
    assert set(picks["game_pk"]) == {1, 2}

    game1 = picks[picks["game_pk"] == 1].iloc[0]
    assert game1["predicted_winner"] == "NYY"
    assert game1["predicted_probability"] == 0.65
    assert game1["above_threshold"] == True  # noqa: E712
    assert game1["metric"] == "GamePick_Win_Probability"
    assert pd.isna(game1["actual_winner"])
    assert pd.isna(game1["game_played"])
    assert game1["model_version"] == config.GAME_PICK_MODEL_VERSION

    game2 = picks[picks["game_pk"] == 2].iloc[0]
    assert game2["above_threshold"] == False  # noqa: E712


def test_append_game_predictions_migrates_a_log_written_before_model_version_existed(tmp_path):
    log_path = str(tmp_path / "game_predictions.csv")
    legacy_log = pd.DataFrame([{
        "date": pd.Timestamp("2026-07-19"), "game_pk": 1, "home_team": "NYY", "away_team": "BOS",
        "predicted_winner": "NYY", "predicted_probability": 0.65, "metric": "GamePick_Win_Probability",
        "actual_winner": "NYY", "game_played": 1,
    }])
    legacy_log.to_csv(log_path, index=False)

    new_pick = game_predictions.select_game_picks(
        _win_probabilities([{"game_pk": 2, "date": pd.Timestamp("2026-07-20"), "home_team": "LAD",
                              "away_team": "SF", "home_win_probability": 0.65}]),
        pd.Timestamp("2026-07-20"),
    )
    combined = game_predictions.append_game_predictions(new_pick, log_path)

    row1 = combined[combined["game_pk"] == 1].iloc[0]
    assert row1["model_version"] == game_predictions.LEGACY_MODEL_VERSION
    row2 = combined[combined["game_pk"] == 2].iloc[0]
    assert row2["model_version"] == config.GAME_PICK_MODEL_VERSION


def test_append_game_predictions_migrates_a_log_written_before_above_threshold_existed(tmp_path):
    log_path = str(tmp_path / "game_predictions.csv")
    legacy_log = pd.DataFrame([{
        "date": pd.Timestamp("2026-07-19"), "game_pk": 1, "home_team": "NYY", "away_team": "BOS",
        "predicted_winner": "NYY", "predicted_probability": 0.65, "metric": "GamePick_Win_Probability",
        "actual_winner": "NYY", "game_played": 1, "model_version": "v1",
    }])
    legacy_log.to_csv(log_path, index=False)
    assert "above_threshold" not in legacy_log.columns

    new_pick = game_predictions.select_game_picks(
        _win_probabilities([{"game_pk": 2, "date": pd.Timestamp("2026-07-20"), "home_team": "LAD",
                              "away_team": "SF", "home_win_probability": 0.52}]),
        pd.Timestamp("2026-07-20"),
    )
    combined = game_predictions.append_game_predictions(new_pick, log_path)

    # The legacy row already cleared the old hard filter by definition
    # (it's a real logged row from before above_threshold existed) - True
    # is the factually correct backfill, not an arbitrary default.
    row1 = combined[combined["game_pk"] == 1].iloc[0]
    assert row1["above_threshold"] == True  # noqa: E712
    row2 = combined[combined["game_pk"] == 2].iloc[0]
    assert row2["above_threshold"] == False  # noqa: E712


def test_select_game_picks_away_favored():
    win_probs = _win_probabilities([
        {"game_pk": 1, "date": pd.Timestamp("2026-07-22"), "home_team": "NYY", "away_team": "BOS",
         "home_win_probability": 0.3},  # away (BOS) favored at .7
    ])

    picks = game_predictions.select_game_picks(win_probs, pd.Timestamp("2026-07-22"))

    assert picks.iloc[0]["predicted_winner"] == "BOS"
    assert picks.iloc[0]["predicted_probability"] == 0.7


def test_select_game_picks_still_logs_a_game_that_clears_no_threshold():
    win_probs = _win_probabilities([
        {"game_pk": 1, "date": pd.Timestamp("2026-07-22"), "home_team": "NYY", "away_team": "BOS",
         "home_win_probability": 0.52},
    ])

    picks = game_predictions.select_game_picks(win_probs, pd.Timestamp("2026-07-22"))

    assert len(picks) == 1
    assert picks.iloc[0]["above_threshold"] == False  # noqa: E712


def test_append_game_predictions_dedupes_keeping_existing_resolved_row(tmp_path):
    log_path = str(tmp_path / "game_predictions.csv")

    first = pd.DataFrame([{
        "date": pd.Timestamp("2026-07-20"), "game_pk": 1, "home_team": "NYY", "away_team": "BOS",
        "predicted_winner": "NYY", "predicted_probability": 0.65, "metric": "GamePick_Win_Probability",
        "actual_winner": "NYY", "game_played": 1,
    }])
    game_predictions.append_game_predictions(first, log_path)

    # Re-logging the same (date, game_pk, metric) with a fresh/unresolved
    # row must not clobber the already-resolved outcome.
    relogged = pd.DataFrame([{
        "date": pd.Timestamp("2026-07-20"), "game_pk": 1, "home_team": "NYY", "away_team": "BOS",
        "predicted_winner": "NYY", "predicted_probability": 0.65, "metric": "GamePick_Win_Probability",
        "actual_winner": pd.NA, "game_played": pd.NA,
    }])
    result = game_predictions.append_game_predictions(relogged, log_path)

    assert len(result) == 1
    assert result.iloc[0]["actual_winner"] == "NYY"
    assert result.iloc[0]["game_played"] == 1


def test_resolve_game_predictions_final_game_sets_winner(tmp_path):
    log_path = str(tmp_path / "game_predictions.csv")
    picks = pd.DataFrame([{
        "date": pd.Timestamp("2026-07-20"), "game_pk": 1, "home_team": "NYY", "away_team": "BOS",
        "predicted_winner": "NYY", "predicted_probability": 0.65, "metric": "GamePick_Win_Probability",
        "actual_winner": pd.NA, "game_played": pd.NA,
    }])
    game_predictions.append_game_predictions(picks, log_path)

    def fetch_results(date):
        return pd.DataFrame([{"game_pk": 1, "status": "Final", "home_score": 5, "away_score": 3}])

    resolved = game_predictions.resolve_game_predictions(log_path, fetch_results, pd.Timestamp("2026-07-21"))

    assert resolved.iloc[0]["game_played"] == 1
    assert resolved.iloc[0]["actual_winner"] == "NYY"


def test_resolve_game_predictions_away_team_wins(tmp_path):
    log_path = str(tmp_path / "game_predictions.csv")
    picks = pd.DataFrame([{
        "date": pd.Timestamp("2026-07-20"), "game_pk": 1, "home_team": "NYY", "away_team": "BOS",
        "predicted_winner": "NYY", "predicted_probability": 0.65, "metric": "GamePick_Win_Probability",
        "actual_winner": pd.NA, "game_played": pd.NA,
    }])
    game_predictions.append_game_predictions(picks, log_path)

    def fetch_results(date):
        return pd.DataFrame([{"game_pk": 1, "status": "Final", "home_score": 2, "away_score": 6}])

    resolved = game_predictions.resolve_game_predictions(log_path, fetch_results, pd.Timestamp("2026-07-21"))

    assert resolved.iloc[0]["actual_winner"] == "BOS"


def test_resolve_game_predictions_leaves_non_final_games_pending(tmp_path):
    log_path = str(tmp_path / "game_predictions.csv")
    picks = pd.DataFrame([{
        "date": pd.Timestamp("2026-07-20"), "game_pk": 1, "home_team": "NYY", "away_team": "BOS",
        "predicted_winner": "NYY", "predicted_probability": 0.65, "metric": "GamePick_Win_Probability",
        "actual_winner": pd.NA, "game_played": pd.NA,
    }])
    game_predictions.append_game_predictions(picks, log_path)

    def fetch_results(date):
        return pd.DataFrame([{"game_pk": 1, "status": "Postponed", "home_score": None, "away_score": None}])

    resolved = game_predictions.resolve_game_predictions(log_path, fetch_results, pd.Timestamp("2026-07-21"))

    assert pd.isna(resolved.iloc[0]["game_played"])
    assert pd.isna(resolved.iloc[0]["actual_winner"])


def test_resolve_game_predictions_one_bad_date_does_not_block_others(tmp_path):
    log_path = str(tmp_path / "game_predictions.csv")
    picks = pd.DataFrame([
        {
            "date": pd.Timestamp("2026-07-19"), "game_pk": 1, "home_team": "NYY", "away_team": "BOS",
            "predicted_winner": "NYY", "predicted_probability": 0.65, "metric": "GamePick_Win_Probability",
            "actual_winner": pd.NA, "game_played": pd.NA,
        },
        {
            "date": pd.Timestamp("2026-07-20"), "game_pk": 2, "home_team": "LAD", "away_team": "SF",
            "predicted_winner": "LAD", "predicted_probability": 0.6, "metric": "GamePick_Win_Probability",
            "actual_winner": pd.NA, "game_played": pd.NA,
        },
    ])
    game_predictions.append_game_predictions(picks, log_path)

    def fetch_results(date):
        if date == pd.Timestamp("2026-07-19").date():
            raise RuntimeError("statsapi is down for this date")
        return pd.DataFrame([{"game_pk": 2, "status": "Final", "home_score": 4, "away_score": 1}])

    resolved = game_predictions.resolve_game_predictions(log_path, fetch_results, pd.Timestamp("2026-07-21"))

    row1 = resolved[resolved["game_pk"] == 1].iloc[0]
    row2 = resolved[resolved["game_pk"] == 2].iloc[0]
    assert pd.isna(row1["game_played"])  # left pending, no crash
    assert row2["game_played"] == 1
    assert row2["actual_winner"] == "LAD"


def test_resolve_game_predictions_migrates_a_log_written_before_model_version_existed(tmp_path):
    log_path = str(tmp_path / "game_predictions.csv")
    legacy_log = pd.DataFrame([{
        "date": pd.Timestamp("2026-07-20"), "game_pk": 1, "home_team": "NYY", "away_team": "BOS",
        "predicted_winner": "NYY", "predicted_probability": 0.65, "metric": "GamePick_Win_Probability",
        "actual_winner": pd.NA, "game_played": pd.NA,
    }])
    legacy_log.to_csv(log_path, index=False)

    def fetch_results(date):
        return pd.DataFrame([{"game_pk": 1, "status": "Final", "home_score": 5, "away_score": 3}])

    resolved = game_predictions.resolve_game_predictions(log_path, fetch_results, pd.Timestamp("2026-07-21"))

    assert resolved.iloc[0]["model_version"] == game_predictions.LEGACY_MODEL_VERSION


def _market_probabilities(rows):
    """rows: list of dicts with home_team, away_team, market_home_win_probability."""
    return pd.DataFrame(rows)


def test_select_game_picks_merges_a_matching_market_probability():
    win_probs = _win_probabilities([
        {"game_pk": 1, "date": pd.Timestamp("2026-07-22"), "home_team": "NYY", "away_team": "BOS",
         "home_win_probability": 0.65},
    ])
    market = _market_probabilities([
        {"home_team": "NYY", "away_team": "BOS", "market_home_win_probability": 0.62, "market_provider": "DraftKings"},
    ])

    picks = game_predictions.select_game_picks(win_probs, pd.Timestamp("2026-07-22"), market_probabilities=market)

    assert picks.iloc[0]["market_home_win_probability"] == pytest.approx(0.62)


def test_select_game_picks_market_probability_is_nan_when_no_match():
    win_probs = _win_probabilities([
        {"game_pk": 1, "date": pd.Timestamp("2026-07-22"), "home_team": "NYY", "away_team": "BOS",
         "home_win_probability": 0.65},
    ])
    # Market data exists for a different game entirely - a real "ESPN
    # doesn't have this matchup" case, not a crash.
    market = _market_probabilities([
        {"home_team": "LAD", "away_team": "SF", "market_home_win_probability": 0.55, "market_provider": "DraftKings"},
    ])

    picks = game_predictions.select_game_picks(win_probs, pd.Timestamp("2026-07-22"), market_probabilities=market)

    assert pd.isna(picks.iloc[0]["market_home_win_probability"])


def test_select_game_picks_market_probability_is_nan_when_none_given():
    win_probs = _win_probabilities([
        {"game_pk": 1, "date": pd.Timestamp("2026-07-22"), "home_team": "NYY", "away_team": "BOS",
         "home_win_probability": 0.65},
    ])

    picks = game_predictions.select_game_picks(win_probs, pd.Timestamp("2026-07-22"))

    assert "market_home_win_probability" in picks.columns
    assert pd.isna(picks.iloc[0]["market_home_win_probability"])


def test_append_game_predictions_migrates_a_log_written_before_market_column_existed(tmp_path):
    log_path = str(tmp_path / "game_predictions.csv")
    legacy_log = pd.DataFrame([{
        "date": pd.Timestamp("2026-07-19"), "game_pk": 1, "home_team": "NYY", "away_team": "BOS",
        "predicted_winner": "NYY", "predicted_probability": 0.65, "metric": "GamePick_Win_Probability",
        "actual_winner": "NYY", "game_played": 1, "model_version": "v1", "above_threshold": True,
    }])
    legacy_log.to_csv(log_path, index=False)
    assert "market_home_win_probability" not in legacy_log.columns

    new_pick = game_predictions.select_game_picks(
        _win_probabilities([{"game_pk": 2, "date": pd.Timestamp("2026-07-20"), "home_team": "LAD",
                              "away_team": "SF", "home_win_probability": 0.65}]),
        pd.Timestamp("2026-07-20"),
    )
    combined = game_predictions.append_game_predictions(new_pick, log_path)

    row1 = combined[combined["game_pk"] == 1].iloc[0]
    assert pd.isna(row1["market_home_win_probability"])


def test_resolve_game_predictions_migrates_a_log_written_before_market_column_existed(tmp_path):
    log_path = str(tmp_path / "game_predictions.csv")
    legacy_log = pd.DataFrame([{
        "date": pd.Timestamp("2026-07-20"), "game_pk": 1, "home_team": "NYY", "away_team": "BOS",
        "predicted_winner": "NYY", "predicted_probability": 0.65, "metric": "GamePick_Win_Probability",
        "actual_winner": pd.NA, "game_played": pd.NA,
    }])
    legacy_log.to_csv(log_path, index=False)

    def fetch_results(date):
        return pd.DataFrame([{"game_pk": 1, "status": "Final", "home_score": 5, "away_score": 3}])

    resolved = game_predictions.resolve_game_predictions(log_path, fetch_results, pd.Timestamp("2026-07-21"))

    assert pd.isna(resolved.iloc[0]["market_home_win_probability"])
