import pandas as pd
import pytest

from mlb_metrics import evaluation, nfl_game_evaluation


def _pick(date, game_id, home_team, away_team, predicted_winner, predicted_probability, actual_winner, game_played,
          metric="NFL_GamePick_Win_Probability", above_threshold=True, market_home_win_probability=None,
          bet_units=0.0, bet_side=None, bet_team=None, bet_moneyline=None, bet_profit_units=None,
          season=2025, week=8):
    return {
        "date": date, "season": season, "week": week, "game_id": game_id,
        "home_team": home_team, "away_team": away_team,
        "predicted_winner": predicted_winner, "predicted_probability": predicted_probability,
        "above_threshold": above_threshold,
        "metric": metric, "actual_winner": actual_winner, "game_played": game_played,
        "market_home_win_probability": market_home_win_probability,
        "bet_units": bet_units, "bet_side": bet_side, "bet_team": bet_team, "bet_moneyline": bet_moneyline,
        "bet_profit_units": bet_profit_units,
    }


def _predictions():
    rows = [
        _pick("2025-10-19", "g1", "KC", "DEN", "KC", 0.65, "KC", 1),  # win
        _pick("2025-10-20", "g2", "SF", "LA", "SF", 0.6, "LA", 1),  # loss
        _pick("2025-10-21", "g3", "BUF", "MIA", "BUF", 0.7, "BUF", 1),  # win
        _pick("2025-10-22", "g4", "PHI", "DAL", "PHI", 0.62, None, None),  # pending
        _pick("2025-10-18", "g5", "GB", "CHI", "GB", 0.6, None, 0),  # not_played (postponed)
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_classify_outcome():
    df = _predictions()
    outcome = nfl_game_evaluation._classify_outcome(df)
    assert list(outcome) == ["win", "loss", "win", "pending", "not_played"]


def test_build_game_picks_export_picks_table_status_and_order():
    picks, summary = nfl_game_evaluation.build_game_picks_export(_predictions())

    assert picks["date"].iloc[0] == picks["date"].max()

    by_id = picks.set_index("game_id")
    assert by_id.loc["g1", "status"] == "win"
    assert by_id.loc["g2", "status"] == "loss"
    assert by_id.loc["g4", "status"] == "pending"
    assert by_id.loc["g5", "status"] == "not_played"


def test_build_game_picks_export_predicted_loser_and_market_predicted_winner_probability():
    rows = [
        _pick("2025-10-19", "g1", "KC", "DEN", "KC", 0.65, "KC", 1, market_home_win_probability=0.6),
        _pick("2025-10-20", "g2", "SF", "LA", "LA", 0.55, "LA", 1, market_home_win_probability=0.7),
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    picks, summary = nfl_game_evaluation.build_game_picks_export(df)

    by_id = picks.set_index("game_id")
    assert by_id.loc["g1", "predicted_loser"] == "DEN"
    assert by_id.loc["g1", "market_predicted_winner_probability"] == pytest.approx(0.6)
    # g2's model favors the AWAY team, so the market's home probability
    # must be flipped (1 - p) to stay comparing "market prob for the
    # SAME side the model favored."
    assert by_id.loc["g2", "predicted_loser"] == "SF"
    assert by_id.loc["g2", "market_predicted_winner_probability"] == pytest.approx(1 - 0.7)


def test_build_game_picks_export_above_threshold_flags_but_does_not_drop():
    rows = [
        _pick("2025-10-19", "g1", "KC", "DEN", "KC", 0.65, "KC", 1, above_threshold=True),
        _pick("2025-10-20", "g2", "SF", "LA", "SF", 0.55, "LA", 1, above_threshold=False),
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    picks, summary = nfl_game_evaluation.build_game_picks_export(df)

    assert set(picks["game_id"]) == {"g1", "g2"}
    by_id = picks.set_index("game_id")
    assert by_id.loc["g1", "above_threshold"] == True  # noqa: E712
    assert by_id.loc["g2", "above_threshold"] == False  # noqa: E712


def test_build_game_picks_export_only_bet_advised_games_affect_pnl_scoring():
    rows = [
        _pick("2025-10-19", "g1", "KC", "DEN", "KC", 0.90, "KC", 1,
              bet_units=3.0, bet_side="home", bet_team="KC", bet_moneyline=-150,
              bet_profit_units=3 * (100 / 150)),  # advised, real win
        _pick("2025-10-20", "g2", "SF", "LA", "SF", 0.50, "LA", 1),  # NOT advised, real loss - must not count
        _pick("2025-10-21", "g3", "BUF", "MIA", "BUF", 0.85, "BUF", 1),  # NOT advised, real win - must not count
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    picks, summary = nfl_game_evaluation.build_game_picks_export(df)

    assert set(picks["game_id"]) == {"g1", "g2", "g3"}  # published regardless
    assert summary.loc[0, "n_bets_advised"] == 1
    assert summary.loc[0, "bets_won"] == 1
    assert summary.loc[0, "win_rate_on_advised_bets"] == pytest.approx(1.0)
    assert summary.loc[0, "total_profit_units"] == pytest.approx(3 * (100 / 150))


def test_build_game_picks_export_ignores_other_metrics():
    preds = _predictions()
    other = preds.copy()
    other["metric"] = "SomeOtherFormula"
    other["game_id"] = other["game_id"] + "_other"
    combined = pd.concat([preds, other], ignore_index=True)

    picks, summary = nfl_game_evaluation.build_game_picks_export(combined, metric="NFL_GamePick_Win_Probability")

    assert set(picks["game_id"]) == {"g1", "g2", "g3", "g4", "g5"}


def test_build_game_picks_export_model_version_filters_and_labels_summary():
    preds = _predictions()
    preds["model_version"] = "v1"
    v2_row = pd.DataFrame([_pick("2025-10-23", "g6", "SEA", "ARI", "SEA", 0.66, "SEA", 1)])
    v2_row["date"] = pd.to_datetime(v2_row["date"])
    v2_row["model_version"] = "v2"
    combined = pd.concat([preds, v2_row], ignore_index=True)

    all_time_picks, all_time_summary = nfl_game_evaluation.build_game_picks_export(combined)
    assert all_time_summary.loc[0, "model_version"] == "all_time"
    assert set(all_time_picks["game_id"]) == {"g1", "g2", "g3", "g4", "g5", "g6"}

    v1_picks, v1_summary = nfl_game_evaluation.build_game_picks_export(combined, model_version="v1")
    assert v1_summary.loc[0, "model_version"] == "v1"
    assert "g6" not in set(v1_picks["game_id"])


def test_build_game_picks_export_no_resolved_games_yet():
    preds = pd.DataFrame([_pick("2025-10-22", "g1", "PHI", "DAL", "PHI", 0.62, None, None)])
    preds["date"] = pd.to_datetime(preds["date"])

    picks, summary = nfl_game_evaluation.build_game_picks_export(preds)

    assert summary.loc[0, "n_bets_advised"] == 0
    assert pd.isna(summary.loc[0, "win_rate_on_advised_bets"])
    assert summary.loc[0, "current_bet_streak"] == 0
    assert summary.loc[0, "best_bet_streak"] == 0


def test_build_game_picks_export_bet_pnl_metrics_real_win_and_loss():
    rows = [
        _pick("2025-10-19", "g1", "KC", "DEN", "KC", 0.65, "KC", 1,
              bet_units=3.0, bet_side="home", bet_team="KC", bet_moneyline=-150,
              bet_profit_units=3 * (100 / 150)),  # win
        _pick("2025-10-20", "g2", "SF", "LA", "SF", 0.6, "LA", 1,
              bet_units=1.5, bet_side="home", bet_team="SF", bet_moneyline=-120,
              bet_profit_units=-1.5),  # loss
        _pick("2025-10-21", "g3", "BUF", "MIA", "BUF", 0.7, "BUF", 1),  # never advised - excluded entirely
        _pick("2025-10-22", "g4", "PHI", "DAL", "PHI", 0.62, None, None,
              bet_units=2.25, bet_side="home", bet_team="PHI", bet_moneyline=-130,
              bet_profit_units=None),  # advised but still pending - excluded
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    picks, summary = nfl_game_evaluation.build_game_picks_export(df)

    assert summary.loc[0, "n_bets_advised"] == 2
    assert summary.loc[0, "bets_won"] == 1
    assert summary.loc[0, "bets_lost"] == 1
    assert summary.loc[0, "win_rate_on_advised_bets"] == pytest.approx(0.5)
    assert summary.loc[0, "total_staked_units"] == pytest.approx(4.5)
    expected_profit = 3 * (100 / 150) - 1.5
    assert summary.loc[0, "total_profit_units"] == pytest.approx(expected_profit)
    assert summary.loc[0, "roi"] == pytest.approx(expected_profit / 4.5)

    expected_wr_low, expected_wr_high = evaluation.wilson_confidence_interval(1, 2)
    assert summary.loc[0, "win_rate_on_advised_bets_ci_low"] == pytest.approx(expected_wr_low)
    assert summary.loc[0, "win_rate_on_advised_bets_ci_high"] == pytest.approx(expected_wr_high)
    expected_roi_p = evaluation.mean_significance(pd.Series([3 * (100 / 150), -1.5]), null_value=0.0)
    assert summary.loc[0, "roi_p_value"] == pytest.approx(expected_roi_p)


def test_build_game_picks_export_bet_streak_is_a_day_streak_not_a_bet_streak():
    rows = [
        _pick("2025-10-19", "g1", "KC", "DEN", "KC", 0.65, "KC", 1,
              bet_units=3.0, bet_team="KC", bet_moneyline=-150, bet_profit_units=2.0),
        _pick("2025-10-20", "g2", "SF", "LA", "SF", 0.6, "LA", 1,
              bet_units=1.5, bet_team="SF", bet_moneyline=-120, bet_profit_units=-1.5),
        _pick("2025-10-21", "g3", "BUF", "MIA", "BUF", 0.7, "BUF", 1,
              bet_units=2.25, bet_team="BUF", bet_moneyline=-130, bet_profit_units=1.73),
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    picks, summary = nfl_game_evaluation.build_game_picks_export(df)

    assert summary.loc[0, "current_bet_streak"] == 1
    assert summary.loc[0, "best_bet_streak"] == 1


def test_build_game_picks_export_bet_streak_scores_a_day_by_its_total_profit():
    rows = [
        _pick("2025-10-19", "g1", "KC", "DEN", "KC", 0.65, "KC", 1,
              bet_units=3.0, bet_team="KC", bet_moneyline=-150, bet_profit_units=2.0),
        _pick("2025-10-19", "g2", "SF", "LA", "SF", 0.6, "LA", 1,
              bet_units=1.0, bet_team="SF", bet_moneyline=-120, bet_profit_units=-0.5),
        _pick("2025-10-20", "g3", "BUF", "MIA", "BUF", 0.7, "BUF", 1,
              bet_units=1.0, bet_team="BUF", bet_moneyline=-130, bet_profit_units=0.5),
        _pick("2025-10-20", "g4", "PHI", "DAL", "PHI", 0.66, None, 1,
              bet_units=2.0, bet_team="PHI", bet_moneyline=-140, bet_profit_units=-1.5),
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    picks, summary = nfl_game_evaluation.build_game_picks_export(df)

    assert summary.loc[0, "current_bet_streak"] == 0
    assert summary.loc[0, "best_bet_streak"] == 1


def test_build_game_picks_export_market_metrics_are_nan_with_no_real_market_data():
    preds = _predictions()  # no market_home_win_probability values

    picks, summary = nfl_game_evaluation.build_game_picks_export(preds)

    assert summary.loc[0, "n_market_resolved"] == 0
    assert pd.isna(summary.loc[0, "market_accuracy"])
    assert pd.isna(summary.loc[0, "beat_closing_line_rate"])


def test_build_game_picks_export_market_accuracy_brier_log_loss():
    rows = [
        _pick("2025-10-19", "g1", "KC", "DEN", "KC", 0.65, "KC", 1, market_home_win_probability=0.6),  # market correct
        _pick("2025-10-20", "g2", "SF", "LA", "SF", 0.6, "LA", 1, market_home_win_probability=0.55),  # market wrong
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    picks, summary = nfl_game_evaluation.build_game_picks_export(df)

    assert summary.loc[0, "n_market_resolved"] == 2
    assert summary.loc[0, "market_accuracy"] == pytest.approx(0.5)


def test_build_game_picks_export_beat_closing_line_rate_excludes_ties():
    rows = [
        # home won; model 0.9 (err .01) vs market 0.6 (err .16) - model beats market.
        _pick("2025-10-19", "g1", "KC", "DEN", "KC", 0.9, "KC", 1, market_home_win_probability=0.6),
        # exact tie - excluded.
        _pick("2025-10-20", "g2", "SF", "LA", "SF", 0.5, "SF", 1, market_home_win_probability=0.5),
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    picks, summary = nfl_game_evaluation.build_game_picks_export(df)

    assert summary.loc[0, "n_beat_closing_line_compared"] == 1
    assert summary.loc[0, "beat_closing_line_rate"] == pytest.approx(1.0)


def test_build_game_picks_export_picks_out_includes_market_column():
    preds = _predictions()

    picks, summary = nfl_game_evaluation.build_game_picks_export(preds)

    assert "market_home_win_probability" in picks.columns
    assert "market_predicted_winner_probability" in picks.columns
