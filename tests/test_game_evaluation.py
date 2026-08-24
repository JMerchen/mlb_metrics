import math

import pandas as pd
import pytest

from mlb_metrics import game_evaluation


def _pick(date, game_pk, home_team, away_team, predicted_winner, predicted_probability, actual_winner, game_played,
          metric="GamePick_Win_Probability", market_home_win_probability=None,
          bet_advised=False, bet_side=None, bet_team=None, bet_moneyline=None,
          bet_stake_dollars=None, bet_profit_dollars=None):
    return {
        "date": date, "game_pk": game_pk, "home_team": home_team, "away_team": away_team,
        "predicted_winner": predicted_winner, "predicted_probability": predicted_probability,
        "metric": metric, "actual_winner": actual_winner, "game_played": game_played,
        "market_home_win_probability": market_home_win_probability,
        "bet_advised": bet_advised, "bet_side": bet_side, "bet_team": bet_team, "bet_moneyline": bet_moneyline,
        "bet_stake_dollars": bet_stake_dollars, "bet_profit_dollars": bet_profit_dollars,
    }


def _predictions():
    rows = [
        _pick("2026-07-18", 1, "NYY", "BOS", "NYY", 0.65, "NYY", 1),  # win
        _pick("2026-07-19", 2, "LAD", "SF", "LAD", 0.6, "SF", 1),  # loss
        _pick("2026-07-20", 3, "HOU", "SEA", "HOU", 0.7, "HOU", 1),  # win
        _pick("2026-07-21", 4, "ATL", "PHI", "ATL", 0.62, None, None),  # pending
        _pick("2026-07-17", 5, "TB", "TOR", "TB", 0.6, None, 0),  # not_played (postponed)
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_classify_outcome():
    df = _predictions()
    outcome = game_evaluation._classify_outcome(df)
    assert list(outcome) == ["win", "loss", "win", "pending", "not_played"]


def test_build_game_picks_export_picks_table_status_and_order():
    picks, summary = game_evaluation.build_game_picks_export(_predictions())

    # Most recent date first.
    assert picks["date"].iloc[0] == picks["date"].max()

    by_pk = picks.set_index("game_pk")
    assert by_pk.loc[1, "status"] == "win"
    assert by_pk.loc[2, "status"] == "loss"
    assert by_pk.loc[4, "status"] == "pending"
    assert by_pk.loc[5, "status"] == "not_played"


def test_build_game_picks_export_min_probability_flags_but_does_not_drop():
    preds = _predictions()
    picks, summary = game_evaluation.build_game_picks_export(preds, min_probability=0.63)

    # Every game is still published, not just the ones clearing .63 - only
    # 0.65 (pk 1) and 0.7 (pk 3) are flagged above_threshold. above_threshold
    # is still published for the dashboard, it just no longer drives the
    # headline scoring below (bet_advised does - see the bet-P&L tests).
    assert set(picks["game_pk"]) == {1, 2, 3, 4, 5}
    by_pk = picks.set_index("game_pk")
    assert by_pk.loc[1, "above_threshold"] == True  # noqa: E712
    assert by_pk.loc[3, "above_threshold"] == True  # noqa: E712
    assert by_pk.loc[2, "above_threshold"] == False  # noqa: E712
    assert by_pk.loc[4, "above_threshold"] == False  # noqa: E712
    assert by_pk.loc[5, "above_threshold"] == False  # noqa: E712


def test_build_game_picks_export_only_bet_advised_games_affect_pnl_scoring():
    # A real win on a game where NO bet was advised must not move the P&L
    # numbers at all - above_threshold is irrelevant here too; only
    # bet_advised (and a real resolved bet_profit_dollars) counts.
    rows = [
        _pick("2026-07-18", 1, "NYY", "BOS", "NYY", 0.90, "NYY", 1,
              bet_advised=True, bet_side="home", bet_team="NYY", bet_moneyline=-150,
              bet_stake_dollars=100.0, bet_profit_dollars=100 * (100 / 150)),  # advised, real win
        _pick("2026-07-19", 2, "LAD", "SF", "LAD", 0.50, "SF", 1),  # NOT advised, real loss - must not count
        _pick("2026-07-20", 3, "HOU", "SEA", "HOU", 0.85, "HOU", 1),  # NOT advised, real win - must not count
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    picks, summary = game_evaluation.build_game_picks_export(df)

    assert set(picks["game_pk"]) == {1, 2, 3}  # published regardless
    assert summary.loc[0, "n_bets_advised"] == 1
    assert summary.loc[0, "bets_won"] == 1
    assert summary.loc[0, "win_rate_on_advised_bets"] == pytest.approx(1.0)
    assert summary.loc[0, "total_profit_dollars"] == pytest.approx(100 * (100 / 150))


def test_build_game_picks_export_ignores_other_metrics():
    preds = _predictions()
    other = preds.copy()
    other["metric"] = "SomeOtherFormula"
    other["game_pk"] = other["game_pk"] + 100
    combined = pd.concat([preds, other], ignore_index=True)

    picks, summary = game_evaluation.build_game_picks_export(combined, metric="GamePick_Win_Probability")

    assert set(picks["game_pk"]) == {1, 2, 3, 4, 5}


def test_build_game_picks_export_model_version_filters_and_labels_summary():
    preds = _predictions()
    preds["model_version"] = "v1"
    v2_row = pd.DataFrame([_pick("2026-07-22", 6, "SD", "COL", "SD", 0.66, "SD", 1)])
    v2_row["date"] = pd.to_datetime(v2_row["date"])
    v2_row["model_version"] = "v2"
    combined = pd.concat([preds, v2_row], ignore_index=True)

    all_time_picks, all_time_summary = game_evaluation.build_game_picks_export(combined)
    assert all_time_summary.loc[0, "model_version"] == "all_time"
    assert set(all_time_picks["game_pk"]) == {1, 2, 3, 4, 5, 6}

    v1_picks, v1_summary = game_evaluation.build_game_picks_export(combined, model_version="v1")
    assert v1_summary.loc[0, "model_version"] == "v1"
    assert 6 not in set(v1_picks["game_pk"])


def test_build_game_picks_export_no_resolved_games_yet():
    preds = pd.DataFrame([_pick("2026-07-21", 1, "ATL", "PHI", "ATL", 0.62, None, None)])
    preds["date"] = pd.to_datetime(preds["date"])

    picks, summary = game_evaluation.build_game_picks_export(preds)

    assert summary.loc[0, "n_bets_advised"] == 0
    assert pd.isna(summary.loc[0, "win_rate_on_advised_bets"])
    assert summary.loc[0, "current_bet_streak"] == 0
    assert summary.loc[0, "best_bet_streak"] == 0


def test_build_game_picks_export_bet_pnl_metrics_real_win_and_loss():
    rows = [
        _pick("2026-07-18", 1, "NYY", "BOS", "NYY", 0.65, "NYY", 1,
              bet_advised=True, bet_side="home", bet_team="NYY", bet_moneyline=-150,
              bet_stake_dollars=100.0, bet_profit_dollars=100 * (100 / 150)),  # win
        _pick("2026-07-19", 2, "LAD", "SF", "LAD", 0.6, "SF", 1,
              bet_advised=True, bet_side="home", bet_team="LAD", bet_moneyline=-120,
              bet_stake_dollars=50.0, bet_profit_dollars=-50.0),  # loss
        _pick("2026-07-20", 3, "HOU", "SEA", "HOU", 0.7, "HOU", 1),  # never advised - excluded entirely
        _pick("2026-07-21", 4, "ATL", "PHI", "ATL", 0.62, None, None,
              bet_advised=True, bet_side="home", bet_team="ATL", bet_moneyline=-130,
              bet_stake_dollars=75.0, bet_profit_dollars=None),  # advised but still pending - excluded
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    picks, summary = game_evaluation.build_game_picks_export(df)

    assert summary.loc[0, "n_bets_advised"] == 2
    assert summary.loc[0, "bets_won"] == 1
    assert summary.loc[0, "bets_lost"] == 1
    assert summary.loc[0, "win_rate_on_advised_bets"] == pytest.approx(0.5)
    assert summary.loc[0, "total_staked_dollars"] == pytest.approx(150.0)
    expected_profit = 100 * (100 / 150) - 50.0
    assert summary.loc[0, "total_profit_dollars"] == pytest.approx(expected_profit)
    assert summary.loc[0, "roi"] == pytest.approx(expected_profit / 150.0)


def test_build_game_picks_export_bet_streak_is_plain_consecutive_wins():
    # Chronological order: win, loss, win -> current streak resets on the
    # loss then rebuilds to 1 on the next win - a plain consecutive-win
    # counter over resolved advised bets, same style the old accuracy-based
    # streak used, just scored on profit>0 instead of correctness.
    rows = [
        _pick("2026-07-18", 1, "NYY", "BOS", "NYY", 0.65, "NYY", 1,
              bet_advised=True, bet_team="NYY", bet_moneyline=-150, bet_stake_dollars=100.0,
              bet_profit_dollars=66.67),
        _pick("2026-07-19", 2, "LAD", "SF", "LAD", 0.6, "SF", 1,
              bet_advised=True, bet_team="LAD", bet_moneyline=-120, bet_stake_dollars=50.0,
              bet_profit_dollars=-50.0),
        _pick("2026-07-20", 3, "HOU", "SEA", "HOU", 0.7, "HOU", 1,
              bet_advised=True, bet_team="HOU", bet_moneyline=-130, bet_stake_dollars=75.0,
              bet_profit_dollars=57.69),
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    picks, summary = game_evaluation.build_game_picks_export(df)

    assert summary.loc[0, "current_bet_streak"] == 1
    assert summary.loc[0, "best_bet_streak"] == 1


def test_build_game_picks_export_bet_columns_missing_migration():
    # A log written before bet advice existed at all (not just null values)
    # must not crash - same migration convention as market_home_win_probability's
    # own missing-column case.
    rows = [
        {"date": "2026-07-18", "game_pk": 1, "home_team": "NYY", "away_team": "BOS", "predicted_winner": "NYY",
         "predicted_probability": 0.65, "metric": "GamePick_Win_Probability", "actual_winner": "NYY", "game_played": 1},
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    assert "bet_advised" not in df.columns

    picks, summary = game_evaluation.build_game_picks_export(df)

    assert summary.loc[0, "n_bets_advised"] == 0
    assert "bet_advised" in picks.columns


def test_build_game_picks_export_market_metrics_are_nan_with_no_real_market_data():
    # _predictions() logs no market_home_win_probability values at all -
    # every game predates quant-analytics item #6 slice 2's real wiring.
    # market_accuracy/etc must be honestly NaN, never a fabricated number.
    picks, summary = game_evaluation.build_game_picks_export(_predictions())

    assert summary.loc[0, "n_market_resolved"] == 0
    assert pd.isna(summary.loc[0, "market_accuracy"])
    assert pd.isna(summary.loc[0, "market_brier_score"])
    assert pd.isna(summary.loc[0, "market_log_loss"])
    assert summary.loc[0, "n_beat_closing_line_compared"] == 0
    assert pd.isna(summary.loc[0, "beat_closing_line_rate"])
    assert "market_home_win_probability" in picks.columns


def test_build_game_picks_export_market_metrics_missing_column_migration():
    # A log written before slice 2's column existed at all (not just null
    # values) must not crash - same migration convention as
    # above_threshold's own missing-column case.
    rows = [
        {"date": "2026-07-18", "game_pk": 1, "home_team": "NYY", "away_team": "BOS", "predicted_winner": "NYY",
         "predicted_probability": 0.65, "metric": "GamePick_Win_Probability", "actual_winner": "NYY", "game_played": 1},
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    assert "market_home_win_probability" not in df.columns

    picks, summary = game_evaluation.build_game_picks_export(df)

    assert summary.loc[0, "n_market_resolved"] == 0
    assert pd.isna(summary.loc[0, "market_accuracy"])


def _market_comparison_rows():
    # Three above-threshold, resolved games with real market data:
    #  - pk 1: model home prob 0.90, market 0.60 - model's squared error
    #    (0.01) beats the market's (0.16) -> a real model win.
    #  - pk 2: model favors home at 0.60 (home prob 0.60), market favors
    #    home much less (0.30); the away team (SF) actually wins - model's
    #    squared error (0.36) is worse than the market's (0.09) -> a real
    #    market win.
    #  - pk 3: model home prob 0.75, market ALSO exactly 0.75 - a real
    #    exact tie, excluded from both sides of the rate.
    rows = [
        _pick("2026-07-18", 1, "NYY", "BOS", "NYY", 0.90, "NYY", 1, market_home_win_probability=0.60),
        _pick("2026-07-19", 2, "LAD", "SF", "LAD", 0.60, "SF", 1, market_home_win_probability=0.30),
        _pick("2026-07-20", 3, "HOU", "SEA", "HOU", 0.75, "HOU", 1, market_home_win_probability=0.75),
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_build_game_picks_export_market_accuracy_brier_log_loss():
    picks, summary = game_evaluation.build_game_picks_export(_market_comparison_rows())

    # All three market picks (NYY, SF, HOU) match the real actual winners.
    assert summary.loc[0, "n_market_resolved"] == 3
    assert summary.loc[0, "market_accuracy"] == pytest.approx(1.0)

    expected_market_brier = ((0.60 - 1) ** 2 + (0.70 - 1) ** 2 + (0.75 - 1) ** 2) / 3
    assert summary.loc[0, "market_brier_score"] == pytest.approx(expected_market_brier)

    expected_market_log_loss = -(math.log(0.60) + math.log(0.70) + math.log(0.75)) / 3
    assert summary.loc[0, "market_log_loss"] == pytest.approx(expected_market_log_loss)


def test_build_game_picks_export_beat_closing_line_rate_excludes_ties():
    picks, summary = game_evaluation.build_game_picks_export(_market_comparison_rows())

    # pk 3's exact tie is excluded from the comparison base entirely -
    # only pk 1 (model win) and pk 2 (market win) count.
    assert summary.loc[0, "n_beat_closing_line_compared"] == 2
    assert summary.loc[0, "beat_closing_line_rate"] == pytest.approx(0.5)


def test_build_game_picks_export_picks_out_includes_market_column():
    picks, summary = game_evaluation.build_game_picks_export(_market_comparison_rows())

    assert "market_home_win_probability" in picks.columns
    by_pk = picks.set_index("game_pk")
    assert by_pk.loc[1, "market_home_win_probability"] == pytest.approx(0.60)
