import pandas as pd
import pytest

from mlb_metrics import config, evaluation, nfl_game_evaluation


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
    assert pd.isna(summary.loc[0, "odds_implied_win_rate"])
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
    # Priced to win: -150 -> 0.6, -120 -> 120/220. The pending -130 bet
    # is not resolved, so it is not part of the yardstick either.
    assert summary.loc[0, "odds_implied_win_rate"] == pytest.approx((0.6 + 120 / 220) / 2)

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


def test_build_game_picks_export_points_columns_and_total_miss_next_to_the_market():
    predictions = _predictions()
    points = {
        "g1": (24.0, 20.0, 44.0, 47.5, 41.0),   # we miss by 3, market by 6.5
        "g2": (21.0, 22.0, 43.0, 44.0, 50.0),   # we miss by 7, market by 6
        "g3": (25.0, 18.0, 43.0, None, 40.0),   # no market total: left out of the comparison
        "g4": (20.0, 20.0, 40.0, 41.0, None),   # pending: left out
    }
    for column, i in (("predicted_home_points", 0), ("predicted_away_points", 1), ("predicted_total", 2),
                      ("market_total", 3), ("actual_total", 4)):
        predictions[column] = predictions["game_id"].map(lambda g: points.get(g, (None,) * 5)[i])

    picks, summary = nfl_game_evaluation.build_game_picks_export(predictions)

    row = summary.iloc[0]
    assert row["points_total_n"] == 2
    assert row["points_total_mae"] == pytest.approx(5.0)
    assert row["market_points_total_mae"] == pytest.approx(6.25)
    g1 = picks.set_index("game_id").loc["g1"]
    assert (g1["predicted_home_points"], g1["predicted_away_points"], g1["market_total"]) == (24.0, 20.0, 47.5)
    # The total is shown as a range 9 points either side (config.NFL_TOTAL_BAND_HALF_WIDTH):
    # g1's 41 is inside 35-53, g2's 50 inside 34-52; Vegas's 47.5 +/- 9 holds 41, 44 +/- 9 holds 50.
    assert (g1["predicted_total_low"], g1["predicted_total_high"]) == (35.0, 53.0)
    assert row["points_total_in_range"] == pytest.approx(1.0)
    assert row["market_points_total_in_range"] == pytest.approx(1.0)


def test_the_total_range_is_whole_points_and_never_below_zero():
    low, high = nfl_game_evaluation.total_band(pd.Series([44.4, 44.6, 5.0, None]))
    assert low.tolist()[:3] == [35.0, 36.0, 0.0] and high.tolist()[:3] == [53.0, 54.0, 14.0]
    assert pd.isna(low.iloc[3])
    # A real total outside the range counts as a miss.
    picks = pd.DataFrame({"predicted_total": [44.0, 44.0], "market_total": [44.0, 60.0], "actual_total": [54.0, 50.0]})
    metrics = nfl_game_evaluation.points_total_metrics(picks)
    assert metrics["points_total_in_range"] == pytest.approx(0.5)
    assert metrics["market_points_total_in_range"] == pytest.approx(0.0)


def test_build_game_picks_export_log_without_points_columns_still_exports():
    picks, summary = nfl_game_evaluation.build_game_picks_export(_predictions())
    assert picks["predicted_total"].isna().all()
    assert summary.iloc[0]["points_total_n"] == 0


def test_build_game_picks_export_carries_team_epa_columns():
    predictions = _predictions()
    predictions["home_off_epa"] = 0.1
    predictions["away_def_rank"] = 4.0
    picks, _ = nfl_game_evaluation.build_game_picks_export(predictions)
    assert (picks["home_off_epa"] == 0.1).all() and (picks["away_def_rank"] == 4.0).all()
    assert picks["home_def_epa"].isna().all()


def test_build_game_picks_export_shows_the_pick_after_the_injury_report_with_the_one_before():
    rows = [
        {**_pick("2025-10-19", "g1", "KC", "DEN", "KC", 0.62, "DEN", 1), "injury_report": "before"},
        {**_pick("2025-10-19", "g1", "KC", "DEN", "DEN", 0.55, "DEN", 1), "injury_report": "after",
         "injury_note": "Patrick Mahomes (QB; Gardner Minshew starts)"},
        {**_pick("2025-10-19", "g2", "BUF", "MIA", "BUF", 0.70, "BUF", 1), "injury_report": "before"},
    ]
    picks, summary = nfl_game_evaluation.build_game_picks_export(pd.DataFrame(rows))

    assert len(picks) == 2
    g1 = picks.set_index("game_id").loc["g1"]
    assert g1["injury_report"] == "after"
    assert g1["predicted_winner"] == "DEN"
    assert g1["before_report_winner"] == "KC"
    assert g1["before_report_probability"] == 0.62
    assert "Mahomes" in g1["injury_note"]
    assert picks.set_index("game_id").loc["g2", "injury_report"] == "before"
    # The record counts each game once, with its latest pick: both right.
    s = summary.iloc[0]
    assert s["n_report_compared"] == 1
    assert s["before_report_accuracy"] == 0.0
    assert s["after_report_accuracy"] == 1.0


def test_total_leans_and_games_outside_our_range():
    picks = pd.DataFrame({
        "predicted_total": [44.6, 30.0, 44.0, 44.0, 44.0],
        "market_total":    [37.5, 37.0, 42.0, 38.0, 38.0],
        "actual_total":    [61.0, 31.0, 40.0, None, 38.0],
    })
    out = nfl_game_evaluation.total_leans(picks)
    # 44.6 vs 37.5: lean over (7.1 >= 5), and 61 beat it; 61 is 7 above our 36-54 range.
    assert out.loc[0, ["total_lean", "total_lean_line", "total_lean_result"]].tolist() == ["over", 37.5, "win"]
    assert (out.loc[0, "total_range_result"], out.loc[0, "total_range_miss"]) == ("over", 7.0)
    # 30 vs 37: lean under; 31 stayed under. Inside 21-39.
    assert out.loc[1, ["total_lean", "total_lean_result", "total_range_result"]].tolist() == ["under", "win", "in"]
    # 44 vs 42: only 2 apart, no lean.
    assert pd.isna(out.loc[2, "total_lean"]) and pd.isna(out.loc[2, "total_lean_result"])
    # Not played yet: a lean, no result.
    assert out.loc[3, "total_lean"] == "over" and pd.isna(out.loc[3, "total_lean_result"])
    # Landing on Vegas's number is a push; 38 is inside our 35-53.
    assert out.loc[4, "total_lean_result"] == "push"
    assert (out.loc[4, "total_range_result"], out.loc[4, "total_range_miss"]) == ("in", 0.0)
    metrics = nfl_game_evaluation.total_lean_metrics(out)
    assert {k: metrics[k] for k in ("total_leans_won", "total_leans_lost", "total_leans_pushed", "total_lean_units")} == {
        "total_leans_won": 2, "total_leans_lost": 0, "total_leans_pushed": 1, "total_lean_units": round(2 * 100 / 110, 2)}
    # No closing totals in these picks: no closing-line value yet.
    assert metrics["total_lean_clv_n"] == 0


def test_closing_line_value_of_each_lean():
    picks = pd.DataFrame({
        "predicted_total": [44.6, 40.0, 44.6, 44.6],
        "market_total":    [37.5, 47.5, 37.5, 37.5],
        "actual_total":    [61.0, 30.0, 41.0, None],
        "closing_total":   [39.5, 45.5, 36.5, None],
    })
    out = nfl_game_evaluation.total_leans(picks)
    # Over 37.5 that closed 39.5: 2 points our way. Under 47.5 that closed 45.5: 2 our way.
    # Over 37.5 that closed 36.5: 1 against us. Not played: no closing line yet.
    assert out["total_lean_clv"].tolist()[:3] == [2.0, 2.0, -1.0]
    assert pd.isna(out.loc[3, "total_lean_clv"])
    assert out.loc[1, "total_lean_close"] == 45.5
    metrics = nfl_game_evaluation.total_lean_metrics(out)
    assert metrics["total_lean_clv_n"] == 3
    assert metrics["total_lean_clv_mean"] == pytest.approx(1.0)
    assert (metrics["total_lean_clv_beat"], metrics["total_lean_clv_worse"]) == (2, 1)
    assert 0 < metrics["total_lean_clv_p"] < 1


def test_tuesday_leans_are_graded_against_the_close_on_their_own():
    predictions = _predictions()
    g = predictions["game_id"].iloc[0]
    first = predictions[predictions["game_id"] == g].iloc[[0]].copy()
    # Tuesday: we say 40 against 47.5 (lean under); Saturday, after the report: 44 against 46.5 (no lean).
    before = first.assign(injury_report="before", predicted_total=40.0, market_total=47.5,
                          actual_total=38.0, closing_total=45.5)
    after = first.assign(injury_report="after", predicted_total=44.0, market_total=46.5,
                         actual_total=38.0, closing_total=45.5)
    picks, summary = nfl_game_evaluation.build_game_picks_export(pd.concat([before, after], ignore_index=True))
    row = picks.set_index("game_id").loc[g]
    assert pd.isna(row["total_lean"])
    assert (row["tuesday_total_lean"], row["tuesday_total_lean_line"]) == ("under", 47.5)
    assert row["tuesday_total_lean_result"] == "win" and row["tuesday_total_lean_clv"] == 2.0
    s = summary.iloc[0]
    assert s["tuesday_total_lean_clv_n"] == 1 and s["tuesday_total_lean_clv_mean"] == 2.0
    assert s["tuesday_total_leans_won"] == 1 and s["total_leans_won"] == 0


def test_rain_and_wind_in_the_forecast_add_to_the_lean():
    base = {"predicted_total": 44.0, "market_total": 47.5, "actual_total": None}
    picks = pd.DataFrame([
        {**base},                                                                   # no forecast: plain gap 3.5, no lean
        {**base, "weather_indoor": 0, "weather_rain": 0.9, "weather_wind": 12.0},   # rain and wind: under
        {**base, "weather_indoor": 1, "weather_rain": 0.9, "weather_wind": 12.0},   # indoors: weather doesn't count
    ])
    out = nfl_game_evaluation.total_leans(picks)
    assert pd.isna(out.loc[0, "total_lean"]) and pd.isna(out.loc[2, "total_lean"])
    weather = config.NFL_TOTAL_LEAN_WEATHER["wx_rain"] * 0.9 + config.NFL_TOTAL_LEAN_WEATHER["wx_wind"] * 7
    assert out.loc[1, "total_lean"] == "under"
    assert out.loc[1, "total_lean_weather"] == pytest.approx(round(weather, 2))
    assert out.loc[1, "total_lean_edge"] == pytest.approx(round(-3.5 + weather, 2))


def test_a_tuesday_lean_is_held_when_starters_are_in_doubt_and_uses_tuesdays_forecast():
    predictions = _predictions()
    g1, g2 = predictions["game_id"].iloc[0], predictions["game_id"].iloc[1]
    row = lambda g: predictions[predictions["game_id"] == g].iloc[[0]].copy()
    # g1: Tuesday under 47.5 (our 40), a QB in doubt -> held, on the card too (no Saturday pick yet).
    held = row(g1).assign(injury_report="before", predicted_total=40.0, market_total=47.5,
                          lean_hold="KC QB Patrick Mahomes (questionable on the last report)")
    # g2: Tuesday 44 vs 47.5 (gap 3.5) with an early forecast of rain -> the weather makes it a lean.
    rainy = row(g2).assign(injury_report="before", predicted_total=44.0, market_total=47.5,
                           weather_indoor=0, weather_rain=0.9, weather_wind=10.0)
    picks, summary = nfl_game_evaluation.build_game_picks_export(pd.concat([held, rainy], ignore_index=True))
    picks = picks.set_index("game_id")
    assert pd.isna(picks.loc[g1, "total_lean"]) and "Mahomes" in picks.loc[g1, "total_lean_held"]
    assert pd.isna(picks.loc[g1, "tuesday_total_lean"]) and "Mahomes" in picks.loc[g1, "tuesday_total_lean_held"]
    assert picks.loc[g2, "tuesday_total_lean"] == "under" and picks.loc[g2, "total_lean"] == "under"
    assert pd.isna(picks.loc[g2, "tuesday_total_lean_held"])


def test_spread_sides_put_our_margin_next_to_the_spread_and_grade_the_side():
    picks = pd.DataFrame([
        # We have KC by ~3.3 (60%); Vegas has KC by 6.5 -> our side is DEN +6.5; KC won by 3 -> covered.
        {"home_team": "KC", "away_team": "DEN", "predicted_winner": "KC", "predicted_probability": 0.60,
         "market_spread": 6.5, "actual_margin": 3},
        # We have BUF (away) by ~6.8 (70%); Vegas has BUF by 3 (-3 for the home team) -> our side is BUF -3; BUF won by 3 -> push.
        {"home_team": "MIA", "away_team": "BUF", "predicted_winner": "BUF", "predicted_probability": 0.70,
         "market_spread": -3.0, "actual_margin": -3},
        # No spread at pick time: the closing one stands in. We have SF by ~1.6 (55%), Vegas has LA by 2 -> SF +2; SF lost by 7 -> didn't cover.
        {"home_team": "SF", "away_team": "LA", "predicted_winner": "SF", "predicted_probability": 0.55,
         "market_spread": None, "closing_spread": -2.0, "actual_margin": -7},
        # Not played yet, and one with no spread anywhere.
        {"home_team": "DAL", "away_team": "NYG", "predicted_winner": "DAL", "predicted_probability": 0.80,
         "market_spread": 4.0, "actual_margin": None},
        {"home_team": "SEA", "away_team": "ARI", "predicted_winner": "SEA", "predicted_probability": 0.65,
         "market_spread": None, "closing_spread": None, "actual_margin": None},
    ])
    out = nfl_game_evaluation.spread_sides(picks)
    assert out["predicted_margin"].tolist() == pytest.approx([3.3, -6.9, 1.6, 11.0, 5.0], abs=0.11)
    assert out["spread_line"].tolist()[:4] == [6.5, -3.0, -2.0, 4.0] and pd.isna(out["spread_line"][4])
    assert out["spread_side"].tolist()[:4] == ["DEN", "BUF", "SF", "DAL"] and pd.isna(out["spread_side"][4])
    assert out["spread_result"].tolist()[:3] == ["win", "push", "loss"]
    assert pd.isna(out["spread_result"][3]) and pd.isna(out["spread_result"][4])
    metrics = nfl_game_evaluation.spread_side_metrics(out)
    assert (metrics["spread_sides_won"], metrics["spread_sides_lost"], metrics["spread_sides_pushed"]) == (1, 1, 1)
    assert metrics["spread_backtest_games"] == config.NFL_SPREAD_BACKTEST["games"]


def test_confidence_counts_by_how_sure_and_by_siding_with_vegas():
    #            home chance, home won, Vegas home chance
    games = [(0.52, 1, 0.60), (0.58, 0, 0.45), (0.30, 0, 0.40), (0.80, 1, None), (0.75, 0, 0.70), (0.55, 1, 0.50)]
    counts = nfl_game_evaluation.confidence_counts(*map(list, zip(*games))).set_index(["group", "label"])
    assert counts.loc[("confidence", "50–55%"), ["games", "right"]].tolist() == [1, 1]
    assert counts.loc[("confidence", "55–60%"), ["games", "right"]].tolist() == [2, 1]   # 0.58 wrong, 0.55 right
    assert counts.loc[("confidence", "70–75%"), ["games", "right"]].tolist() == [1, 1]   # the 30% home team lost
    assert counts.loc[("confidence", "75%+"), ["games", "right"]].tolist() == [2, 1]
    # Five games have a Vegas number; we went against it once (0.58 vs 0.45) and lost.
    assert counts.loc[("vegas", "with"), ["games", "right"]].tolist() == [4, 3]
    assert counts.loc[("vegas", "against"), ["games", "right"]].tolist() == [1, 0]
    assert counts["games"].loc["confidence"].sum() == 6


def test_confidence_table_sets_this_seasons_finished_picks_next_to_the_backtest():
    picks = pd.DataFrame([
        {"season": 2026, "status": "win", "home_team": "KC", "predicted_winner": "KC", "predicted_probability": 0.62,
         "market_home_win_probability": 0.70},
        {"season": 2026, "status": "loss", "home_team": "MIA", "predicted_winner": "BUF", "predicted_probability": 0.64,
         "market_home_win_probability": 0.55},
        {"season": 2026, "status": "pending", "home_team": "SF", "predicted_winner": "SF", "predicted_probability": 0.61,
         "market_home_win_probability": 0.60},
        {"season": 2025, "status": "win", "home_team": "SF", "predicted_winner": "SF", "predicted_probability": 0.61,
         "market_home_win_probability": 0.60},
    ])
    backtest = pd.DataFrame([{"group": "confidence", "label": "60–65%", "low": 0.6, "high": 0.65, "games": 300, "right": 186},
                             {"group": "vegas", "label": "against", "low": None, "high": None, "games": 200, "right": 90}])
    table = nfl_game_evaluation.confidence_table(picks, backtest).set_index(["group", "label"])
    assert table.loc[("confidence", "60–65%"), ["season_games", "season_right", "backtest_games", "backtest_right"]].tolist() == [2, 1, 300, 186]
    assert table.loc[("vegas", "against"), ["season_games", "season_right", "backtest_games"]].tolist() == [1, 0, 200]
    assert table.loc[("vegas", "with"), "season_games"] == 1 and pd.isna(table.loc[("vegas", "with"), "backtest_games"])
    # No backtest file yet: this season's counts alone.
    assert nfl_game_evaluation.confidence_table(picks)["backtest_games"].isna().all()


def test_build_game_picks_export_carries_the_spread_and_card_columns():
    predictions = _predictions()
    predictions["market_spread"] = 3.0
    predictions["home_qb"] = "Patrick Mahomes"
    predictions["lean_hold"] = "KC QB Patrick Mahomes (questionable on the last report)"
    picks, summary = nfl_game_evaluation.build_game_picks_export(predictions)
    for column in [*nfl_game_evaluation.SPREAD_COLUMNS, *nfl_game_evaluation.CARD_COLUMNS, "starters_in_doubt"]:
        assert column in picks.columns
    assert (picks["spread_line"] == 3.0).all() and picks["predicted_margin"].notna().all()
    assert (picks["starters_in_doubt"] == "KC QB Patrick Mahomes (questionable on the last report)").all()
    assert {"spread_sides_won", "spread_backtest_won_rate", "total_lean_backtest_games"} <= set(summary.columns)


def test_qb_ranks_order_a_weeks_starters_once_enough_of_them_are_rated():
    teams = [f"T{i}" for i in range(16)]
    week = pd.DataFrame({"season": 2026, "week": 10, "home_team": teams[:8], "away_team": teams[8:],
                         "home_qb_plus": [0.30, 0.10, 0.05, 0.0, -0.05, -0.10, -0.15, -0.20],
                         "away_qb_plus": [0.25, 0.12, 0.04, 0.01, -0.04, -0.11, -0.16, -0.30]})
    home, away = nfl_game_evaluation.qb_ranks(week)
    assert home.tolist()[:2] == [1, 4] and away.tolist()[:2] == [2, 3] and away.tolist()[-1] == 16
    # Late in a week only the games still to play are rated: too few to rank.
    late = week.assign(week=11).head(2)
    home, away = nfl_game_evaluation.qb_ranks(pd.concat([week, late], ignore_index=True))
    assert home.iloc[:8].notna().all() and home.iloc[8:].isna().all() and away.iloc[8:].isna().all()
    # A log from before the ratings were kept has no ranks.
    home, _ = nfl_game_evaluation.qb_ranks(week.drop(columns=["home_qb_plus", "away_qb_plus"]))
    assert home.isna().all()
    home, away = nfl_game_evaluation.qb_ranks(week.iloc[0:0])
    assert home.empty and away.empty
