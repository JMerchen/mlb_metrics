import pandas as pd
import pytest

from mlb_metrics import game_evaluation


def _pick(date, game_pk, home_team, away_team, predicted_winner, predicted_probability, actual_winner, game_played,
          metric="GamePick_Win_Probability"):
    return {
        "date": date, "game_pk": game_pk, "home_team": home_team, "away_team": away_team,
        "predicted_winner": predicted_winner, "predicted_probability": predicted_probability,
        "metric": metric, "actual_winner": actual_winner, "game_played": game_played,
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


def test_build_game_picks_export_min_probability_filters_at_export_time():
    preds = _predictions()
    picks, summary = game_evaluation.build_game_picks_export(preds, min_probability=0.63)

    # Only games with predicted_probability >= .63 survive: 0.65 and 0.7.
    assert set(picks["game_pk"]) == {1, 3}
    assert summary.loc[0, "n_games_resolved"] == 2


def test_build_game_picks_export_ignores_other_metrics():
    preds = _predictions()
    other = preds.copy()
    other["metric"] = "SomeOtherFormula"
    other["game_pk"] = other["game_pk"] + 100
    combined = pd.concat([preds, other], ignore_index=True)

    picks, summary = game_evaluation.build_game_picks_export(combined, metric="GamePick_Win_Probability")

    assert set(picks["game_pk"]) == {1, 2, 3, 4, 5}


def test_build_game_picks_export_no_resolved_games_yet():
    preds = pd.DataFrame([_pick("2026-07-21", 1, "ATL", "PHI", "ATL", 0.62, None, None)])
    preds["date"] = pd.to_datetime(preds["date"])

    picks, summary = game_evaluation.build_game_picks_export(preds)

    assert summary.loc[0, "n_games_resolved"] == 0
    assert pd.isna(summary.loc[0, "accuracy"])
    assert summary.loc[0, "current_streak"] == 0
    assert summary.loc[0, "best_streak"] == 0
