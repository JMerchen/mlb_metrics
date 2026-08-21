import math

import pandas as pd
import pytest

from mlb_metrics import game_evaluation


def _pick(date, game_pk, home_team, away_team, predicted_winner, predicted_probability, actual_winner, game_played,
          metric="GamePick_Win_Probability", market_home_win_probability=None):
    return {
        "date": date, "game_pk": game_pk, "home_team": home_team, "away_team": away_team,
        "predicted_winner": predicted_winner, "predicted_probability": predicted_probability,
        "metric": metric, "actual_winner": actual_winner, "game_played": game_played,
        "market_home_win_probability": market_home_win_probability,
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


def test_build_game_picks_export_summary_numbers():
    picks, summary = game_evaluation.build_game_picks_export(_predictions())

    assert summary.loc[0, "n_games_resolved"] == 3  # win/loss/win only, not pending or not_played
    assert summary.loc[0, "accuracy"] == pytest.approx(2 / 3)

    # brier_score: (.65-1)^2 + (.6-0)^2 + (.7-1)^2, mean over 3
    expected_brier = ((0.65 - 1) ** 2 + (0.6 - 0) ** 2 + (0.7 - 1) ** 2) / 3
    assert summary.loc[0, "brier_score"] == pytest.approx(expected_brier)


def test_build_game_picks_export_picks_table_status_and_order():
    picks, summary = game_evaluation.build_game_picks_export(_predictions())

    # Most recent date first.
    assert picks["date"].iloc[0] == picks["date"].max()

    by_pk = picks.set_index("game_pk")
    assert by_pk.loc[1, "status"] == "win"
    assert by_pk.loc[2, "status"] == "loss"
    assert by_pk.loc[4, "status"] == "pending"
    assert by_pk.loc[5, "status"] == "not_played"


def test_build_game_picks_export_streak_is_plain_consecutive_count():
    # Chronological order: win(07-18), loss(07-19), win(07-20) -> current
    # streak resets on the loss, then rebuilds to 1 on the next win. No
    # Beat-the-Streak-style "reset the whole day if any pick misses" rule -
    # each game is independent.
    picks, summary = game_evaluation.build_game_picks_export(_predictions())

    assert summary.loc[0, "current_streak"] == 1
    assert summary.loc[0, "best_streak"] == 1


def test_build_game_picks_export_min_probability_flags_but_does_not_drop():
    preds = _predictions()
    picks, summary = game_evaluation.build_game_picks_export(preds, min_probability=0.63)

    # Every game is still published, not just the ones clearing .63 - only
    # 0.65 (pk 1) and 0.7 (pk 3) are flagged above_threshold.
    assert set(picks["game_pk"]) == {1, 2, 3, 4, 5}
    by_pk = picks.set_index("game_pk")
    assert by_pk.loc[1, "above_threshold"] == True  # noqa: E712
    assert by_pk.loc[3, "above_threshold"] == True  # noqa: E712
    assert by_pk.loc[2, "above_threshold"] == False  # noqa: E712
    assert by_pk.loc[4, "above_threshold"] == False  # noqa: E712
    assert by_pk.loc[5, "above_threshold"] == False  # noqa: E712

    # Scoring stays scoped to the above_threshold subset only: pk 1 (win)
    # and pk 3 (win) are both resolved and both correct - pk 2's real loss
    # must not drag the numbers down since it's below threshold.
    assert summary.loc[0, "n_games_resolved"] == 2
    assert summary.loc[0, "accuracy"] == pytest.approx(1.0)


def test_build_game_picks_export_below_threshold_outcome_does_not_affect_scoring():
    # A below-threshold game that actually LOSES must not move accuracy or
    # break the streak - only above_threshold games count toward scoring.
    rows = [
        _pick("2026-07-18", 1, "NYY", "BOS", "NYY", 0.90, "NYY", 1),  # above threshold, win
        _pick("2026-07-19", 2, "LAD", "SF", "LAD", 0.50, "SF", 1),  # below threshold, loss
        _pick("2026-07-20", 3, "HOU", "SEA", "HOU", 0.85, "HOU", 1),  # above threshold, win
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    picks, summary = game_evaluation.build_game_picks_export(df, min_probability=0.58)

    assert set(picks["game_pk"]) == {1, 2, 3}  # published regardless
    assert summary.loc[0, "n_games_resolved"] == 2  # pk 2 excluded from scoring
    assert summary.loc[0, "accuracy"] == pytest.approx(1.0)
    assert summary.loc[0, "current_streak"] == 2
    assert summary.loc[0, "best_streak"] == 2


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
    assert v1_summary.loc[0, "n_games_resolved"] == 3


def test_build_game_picks_export_no_resolved_games_yet():
    preds = pd.DataFrame([_pick("2026-07-21", 1, "ATL", "PHI", "ATL", 0.62, None, None)])
    preds["date"] = pd.to_datetime(preds["date"])

    picks, summary = game_evaluation.build_game_picks_export(preds)

    assert summary.loc[0, "n_games_resolved"] == 0
    assert pd.isna(summary.loc[0, "accuracy"])
    assert summary.loc[0, "current_streak"] == 0
    assert summary.loc[0, "best_streak"] == 0


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
