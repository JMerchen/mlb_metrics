import pandas as pd
import pytest

from mlb_metrics import config, kelly, nfl_game_predictions


def _win_probabilities(rows):
    """rows: list of dicts with game_id, season, week, home_team, away_team, home_win_probability."""
    return pd.DataFrame(rows)


def _schedule(rows):
    """rows: list of dicts with game_id, gameday (and optionally home_score/away_score)."""
    return pd.DataFrame(rows)


def test_select_game_picks_logs_every_game_and_flags_above_threshold():
    win_probs = _win_probabilities([
        {"game_id": "2025_08_A", "season": 2025, "week": 8, "home_team": "KC", "away_team": "DEN",
         "home_win_probability": 0.70},  # clears threshold, home favored
        {"game_id": "2025_08_B", "season": 2025, "week": 8, "home_team": "SF", "away_team": "LA",
         "home_win_probability": 0.55},  # below threshold - still logged, just flagged False
    ])
    schedule = _schedule([
        {"game_id": "2025_08_A", "gameday": "2025-10-26"},
        {"game_id": "2025_08_B", "gameday": "2025-10-26"},
    ])

    picks = nfl_game_predictions.select_game_picks(win_probs, schedule)

    assert len(picks) == 2
    assert set(picks["game_id"]) == {"2025_08_A", "2025_08_B"}

    game1 = picks[picks["game_id"] == "2025_08_A"].iloc[0]
    assert game1["predicted_winner"] == "KC"
    assert game1["predicted_probability"] == 0.70
    assert game1["above_threshold"] == True  # noqa: E712
    assert game1["metric"] == "NFL_GamePick_Win_Probability"
    assert pd.isna(game1["actual_winner"])
    assert pd.isna(game1["game_played"])
    assert game1["model_version"] == config.NFL_GAME_PICK_MODEL_VERSION
    assert game1["date"] == pd.Timestamp("2025-10-26")

    game2 = picks[picks["game_id"] == "2025_08_B"].iloc[0]
    assert game2["above_threshold"] == False  # noqa: E712


def test_select_game_picks_away_favored():
    win_probs = _win_probabilities([
        {"game_id": "2025_08_A", "season": 2025, "week": 8, "home_team": "KC", "away_team": "DEN",
         "home_win_probability": 0.40},
    ])
    schedule = _schedule([{"game_id": "2025_08_A", "gameday": "2025-10-26"}])

    picks = nfl_game_predictions.select_game_picks(win_probs, schedule)

    row = picks.iloc[0]
    assert row["predicted_winner"] == "DEN"
    assert row["predicted_probability"] == pytest.approx(0.60)


def test_select_game_picks_no_bet_columns_without_moneylines_backward_compat():
    win_probs = _win_probabilities([
        {"game_id": "2025_08_A", "season": 2025, "week": 8, "home_team": "KC", "away_team": "DEN",
         "home_win_probability": 0.70},
    ])
    schedule = _schedule([{"game_id": "2025_08_A", "gameday": "2025-10-26"}])

    picks = nfl_game_predictions.select_game_picks(win_probs, schedule)

    assert (picks["bet_units"] == 0.0).all()
    assert picks["bet_team"].isna().all()


def test_select_game_picks_merges_a_matching_market_probability():
    win_probs = _win_probabilities([
        {"game_id": "2025_08_A", "season": 2025, "week": 8, "home_team": "KC", "away_team": "DEN",
         "home_win_probability": 0.70},
    ])
    schedule = _schedule([{"game_id": "2025_08_A", "gameday": "2025-10-26"}])
    market = pd.DataFrame([{"home_team": "KC", "away_team": "DEN", "market_home_win_probability": 0.6}])

    picks = nfl_game_predictions.select_game_picks(win_probs, schedule, market_probabilities=market)

    assert picks.iloc[0]["market_home_win_probability"] == pytest.approx(0.6)


# ---------------------------------------------------------------------
# advise_bets - direct mirror of test_game_predictions.py's own tests,
# with game_id in place of game_pk.
# ---------------------------------------------------------------------

def _bet_pick_row(game_id, home_team, away_team, predicted_winner, predicted_probability):
    return {
        "date": pd.Timestamp("2025-10-26"), "game_id": game_id, "home_team": home_team,
        "away_team": away_team, "predicted_winner": predicted_winner,
        "predicted_probability": predicted_probability,
    }


def _bet_market_row(home_team, away_team, home_moneyline, away_moneyline):
    return {
        "home_team": home_team, "away_team": away_team,
        "home_moneyline": home_moneyline, "away_moneyline": away_moneyline,
    }


def test_advise_bets_finds_a_real_positive_home_edge():
    picks = pd.DataFrame([_bet_pick_row("g1", "KC", "DEN", "KC", 0.70)])
    # home implied = 150/250 = 0.6 -> edge = 0.70 - 0.6 = 0.10
    market = pd.DataFrame([_bet_market_row("KC", "DEN", -150, 130)])

    recs = nfl_game_predictions.advise_bets(picks, market, kelly_fraction_multiplier=1.0, min_edge=0.02)

    assert len(recs) == 1
    row = recs.iloc[0]
    assert row["side"] == "home"
    assert row["team"] == "KC"
    assert row["edge"] == pytest.approx(0.10, abs=1e-6)
    assert row["kelly_stake_fraction"] > 0


def test_advise_bets_below_min_edge_recommends_nothing():
    picks = pd.DataFrame([_bet_pick_row("g1", "KC", "DEN", "KC", 0.52)])
    market = pd.DataFrame([_bet_market_row("KC", "DEN", -115, 105)])

    recs = nfl_game_predictions.advise_bets(picks, market, kelly_fraction_multiplier=1.0, min_edge=0.02)

    assert recs.empty


def test_advise_bets_skips_a_game_missing_from_market():
    picks = pd.DataFrame([
        _bet_pick_row("g1", "KC", "DEN", "KC", 0.70),
        _bet_pick_row("g2", "SF", "LA", "SF", 0.65),
    ])
    market = pd.DataFrame([_bet_market_row("KC", "DEN", -150, 130)])

    recs = nfl_game_predictions.advise_bets(picks, market, kelly_fraction_multiplier=1.0, min_edge=0.02)

    assert list(recs["game_id"]) == ["g1"]


def test_advise_bets_drops_both_sides_on_a_data_anomaly(capsys):
    picks = pd.DataFrame([_bet_pick_row("g1", "KC", "DEN", "KC", 0.50)])
    market = pd.DataFrame([_bet_market_row("KC", "DEN", 120, 120)])

    recs = nfl_game_predictions.advise_bets(picks, market, kelly_fraction_multiplier=1.0, min_edge=0.02)

    assert recs.empty
    assert "data-quality anomaly" in capsys.readouterr().out


def test_advise_bets_sizes_off_pessimistic_probability_when_given():
    picks = pd.DataFrame([{
        **_bet_pick_row("g1", "KC", "DEN", "KC", 0.70),
        "home_win_probability_pessimistic": 0.61,
        "away_win_probability_pessimistic": 0.05,
    }])
    market = pd.DataFrame([_bet_market_row("KC", "DEN", -150, 130)])

    recs = nfl_game_predictions.advise_bets(picks, market, kelly_fraction_multiplier=0.5, min_edge=0.02)

    assert len(recs) == 1
    row = recs.iloc[0]
    assert row["edge"] == pytest.approx(0.70 - (150 / 250), abs=1e-6)
    expected_stake = kelly.kelly_fraction(0.61, -150, 0.5)
    assert expected_stake > 0
    assert row["kelly_stake_fraction"] == pytest.approx(expected_stake, abs=1e-9)


def test_advise_bets_clips_a_single_bet_to_the_max_single_bet_cap():
    picks = pd.DataFrame([_bet_pick_row("g1", "KC", "DEN", "KC", 0.80)])
    market = pd.DataFrame([_bet_market_row("KC", "DEN", -150, 130)])

    recs = nfl_game_predictions.advise_bets(picks, market, kelly_fraction_multiplier=1.0, min_edge=0.02)

    uncapped = kelly.kelly_fraction(0.80, -150, 1.0)
    single_bet_cap = config.KELLY_MAX_SINGLE_BET_UNIT_CAP * config.UNIT_SIZE_FRACTION
    assert uncapped > single_bet_cap  # sanity - the cap is genuinely binding here
    assert recs.iloc[0]["kelly_stake_fraction"] == pytest.approx(single_bet_cap, abs=1e-9)


def test_advise_bets_daily_cap_layers_on_top_of_the_single_bet_cap():
    # Two huge edges that each clip to the per-bet cap, plus one modest
    # edge that stays under it on its own - their post-per-bet-cap sum
    # still exceeds the daily cap, so the daily cap scales all three down
    # further, proportionally (same numbers as
    # test_game_predictions.py's own identical test).
    picks = pd.DataFrame([
        _bet_pick_row("g1", "KC", "DEN", "KC", 0.80),
        _bet_pick_row("g2", "SF", "LA", "SF", 0.80),
        _bet_pick_row("g3", "BUF", "MIA", "BUF", 0.346),
    ])
    market = pd.DataFrame([
        _bet_market_row("KC", "DEN", -150, 130),
        _bet_market_row("SF", "LA", -140, 120),
        _bet_market_row("BUF", "MIA", 200, -260),
    ])

    recs = nfl_game_predictions.advise_bets(picks, market, kelly_fraction_multiplier=1.0, min_edge=0.01)

    single_bet_cap = config.KELLY_MAX_SINGLE_BET_UNIT_CAP * config.UNIT_SIZE_FRACTION
    daily_cap = config.KELLY_DAILY_UNIT_CAP * config.UNIT_SIZE_FRACTION
    buf_pre_daily_cap = kelly.kelly_fraction(0.346, 200, 1.0)
    assert buf_pre_daily_cap < single_bet_cap  # sanity - BUF isn't hitting the per-bet cap on its own
    pre_daily_total = 2 * single_bet_cap + buf_pre_daily_cap
    assert pre_daily_total > daily_cap  # sanity - the daily cap is genuinely binding

    assert recs["kelly_stake_fraction"].sum() == pytest.approx(daily_cap, abs=1e-9)


# ---------------------------------------------------------------------
# append_game_predictions / resolve_game_predictions
# ---------------------------------------------------------------------

def _resolved_pick(game_id, home_team, away_team, home_score=None, away_score=None, game_played=pd.NA):
    return {
        "date": pd.Timestamp("2025-10-26"), "season": 2025, "week": 8,
        "game_id": game_id, "home_team": home_team, "away_team": away_team,
        "predicted_winner": home_team, "predicted_probability": 0.65, "above_threshold": True,
        "metric": "NFL_GamePick_Win_Probability", "actual_winner": pd.NA, "game_played": game_played,
        "model_version": "v1", "market_home_win_probability": pd.NA,
        "bet_units": 0.0, "bet_side": pd.NA, "bet_team": pd.NA, "bet_moneyline": pd.NA,
        "bet_stake_fraction": pd.NA, "bet_profit_units": pd.NA,
        "home_win_probability_pessimistic": pd.NA, "away_win_probability_pessimistic": pd.NA,
    }


def test_append_game_predictions_dedupes_keeping_existing_resolved_row(tmp_path):
    log_path = str(tmp_path / "nfl_game_predictions.csv")
    resolved = pd.DataFrame([{**_resolved_pick("g1", "KC", "DEN"), "actual_winner": "KC", "game_played": 1}])
    nfl_game_predictions.append_game_predictions(resolved, log_path)

    # A re-run of select_game_picks for the same game logs it again as pending.
    fresh = pd.DataFrame([_resolved_pick("g1", "KC", "DEN")])
    combined = nfl_game_predictions.append_game_predictions(fresh, log_path)

    assert len(combined) == 1
    assert combined.iloc[0]["actual_winner"] == "KC"
    assert combined.iloc[0]["game_played"] == 1


def test_resolve_game_predictions_final_game_sets_winner(tmp_path):
    log_path = str(tmp_path / "nfl_game_predictions.csv")
    nfl_game_predictions.append_game_predictions(pd.DataFrame([_resolved_pick("g1", "KC", "DEN")]), log_path)
    schedule = _schedule([{"game_id": "g1", "gameday": "2025-10-26", "home_score": 24, "away_score": 17}])

    resolved = nfl_game_predictions.resolve_game_predictions(log_path, schedule)

    assert resolved.iloc[0]["game_played"] == 1
    assert resolved.iloc[0]["actual_winner"] == "KC"


def test_resolve_game_predictions_away_team_wins(tmp_path):
    log_path = str(tmp_path / "nfl_game_predictions.csv")
    nfl_game_predictions.append_game_predictions(pd.DataFrame([_resolved_pick("g1", "KC", "DEN")]), log_path)
    schedule = _schedule([{"game_id": "g1", "gameday": "2025-10-26", "home_score": 17, "away_score": 24}])

    resolved = nfl_game_predictions.resolve_game_predictions(log_path, schedule)

    assert resolved.iloc[0]["actual_winner"] == "DEN"


def test_resolve_game_predictions_leaves_unplayed_games_pending(tmp_path):
    log_path = str(tmp_path / "nfl_game_predictions.csv")
    nfl_game_predictions.append_game_predictions(pd.DataFrame([_resolved_pick("g1", "KC", "DEN")]), log_path)
    schedule = _schedule([{"game_id": "g1", "gameday": "2025-10-26", "home_score": None, "away_score": None}])

    resolved = nfl_game_predictions.resolve_game_predictions(log_path, schedule)

    assert pd.isna(resolved.iloc[0]["game_played"])
    assert pd.isna(resolved.iloc[0]["actual_winner"])


def test_resolve_game_predictions_computes_real_profit_on_a_winning_advised_bet(tmp_path):
    log_path = str(tmp_path / "nfl_game_predictions.csv")
    pick = {
        **_resolved_pick("g1", "KC", "DEN"),
        "bet_units": 2.0, "bet_side": "home", "bet_team": "KC", "bet_moneyline": 130.0,
    }
    nfl_game_predictions.append_game_predictions(pd.DataFrame([pick]), log_path)
    schedule = _schedule([{"game_id": "g1", "gameday": "2025-10-26", "home_score": 24, "away_score": 17}])

    resolved = nfl_game_predictions.resolve_game_predictions(log_path, schedule)

    expected_profit = 2.0 * kelly.moneyline_to_net_odds(130.0)
    assert resolved.iloc[0]["bet_profit_units"] == pytest.approx(expected_profit)


def test_resolve_game_predictions_computes_real_loss_on_a_losing_advised_bet(tmp_path):
    log_path = str(tmp_path / "nfl_game_predictions.csv")
    pick = {
        **_resolved_pick("g1", "KC", "DEN"),
        "bet_units": 2.0, "bet_side": "home", "bet_team": "KC", "bet_moneyline": 130.0,
    }
    nfl_game_predictions.append_game_predictions(pd.DataFrame([pick]), log_path)
    schedule = _schedule([{"game_id": "g1", "gameday": "2025-10-26", "home_score": 10, "away_score": 24}])

    resolved = nfl_game_predictions.resolve_game_predictions(log_path, schedule)

    assert resolved.iloc[0]["bet_profit_units"] == pytest.approx(-2.0)


def test_resolve_game_predictions_leaves_profit_null_for_a_non_advised_resolved_game(tmp_path):
    log_path = str(tmp_path / "nfl_game_predictions.csv")
    nfl_game_predictions.append_game_predictions(pd.DataFrame([_resolved_pick("g1", "KC", "DEN")]), log_path)
    schedule = _schedule([{"game_id": "g1", "gameday": "2025-10-26", "home_score": 24, "away_score": 17}])

    resolved = nfl_game_predictions.resolve_game_predictions(log_path, schedule)

    assert pd.isna(resolved.iloc[0]["bet_profit_units"])
