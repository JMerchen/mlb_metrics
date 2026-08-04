import pandas as pd
import pytest

from mlb_metrics import config, nfl_matchup


def test_compute_opponent_adjustment_ratio_weight_zero_is_always_exactly_one():
    opponent_rate = pd.Series([50.0, 500.0, 0.0])
    result = nfl_matchup.compute_opponent_adjustment_ratio(opponent_rate, league_rate=200.0, weight=0.0)
    assert (result == 1.0).all()


def test_compute_opponent_adjustment_ratio_weight_one_is_full_raw_ratio_within_clip():
    # 220 / 200 = 1.10, inside config.NFL_MATCHUP_OFFENSE_CLIP's (0.85, 1.15) range.
    result = nfl_matchup.compute_opponent_adjustment_ratio(pd.Series([220.0]), league_rate=200.0, weight=1.0)
    assert result.iloc[0] == pytest.approx(1.10)


def test_compute_opponent_adjustment_ratio_clips_extreme_outliers():
    lo, hi = config.NFL_MATCHUP_OFFENSE_CLIP
    # An absurdly generous defense (allowing 10x league average) would
    # otherwise blow the ratio far past the clip range.
    result = nfl_matchup.compute_opponent_adjustment_ratio(pd.Series([2000.0]), league_rate=200.0, weight=1.0)
    assert result.iloc[0] == pytest.approx(hi)


def test_attach_matchup_adjustment_uses_schedule_opponent_not_defense_rates_order():
    # Two teams (SF, KC) each play a DIFFERENT real opponent this week
    # (SEA, DEN respectively) per current_week_schedule_df. defense_rates_df
    # is deliberately ordered/shaped so that if this module ever looked up
    # "the other team in defense_rates_df" instead of the real schedule, it
    # would pick the wrong opponent - this proves it doesn't.
    players_df = pd.DataFrame([{"player_id": "p1", "team": "SF"}, {"player_id": "p2", "team": "KC"}])
    defense_rates_df = pd.DataFrame(
        [
            {"team": "SEA", "pass_yards_allowed_per_game": 220.0, "rush_yards_allowed_per_game": 110.0, "receptions_allowed_per_game": 22.0},
            {"team": "DEN", "pass_yards_allowed_per_game": 180.0, "rush_yards_allowed_per_game": 90.0, "receptions_allowed_per_game": 18.0},
        ]
    )
    schedule_df = pd.DataFrame(
        [
            {"home_team": "SF", "away_team": "SEA"},
            {"home_team": "KC", "away_team": "DEN"},
        ]
    )

    result = nfl_matchup.attach_matchup_adjustment(players_df, defense_rates_df, schedule_df, weight=1.0).set_index("player_id")

    league_pass = defense_rates_df["pass_yards_allowed_per_game"].mean()  # 200.0
    assert result.loc["p1", "pass_yards_allowed_per_game"] == 220.0
    assert result.loc["p1", "Opponent_Pass_Yards_Allowed_Ratio"] == pytest.approx(220.0 / league_pass)
    assert result.loc["p2", "pass_yards_allowed_per_game"] == 180.0
    assert result.loc["p2", "Opponent_Pass_Yards_Allowed_Ratio"] == pytest.approx(180.0 / league_pass)


def test_attach_matchup_adjustment_falls_back_to_league_average_for_missing_opponent():
    players_df = pd.DataFrame([{"player_id": "p1", "team": "SF"}])
    defense_rates_df = pd.DataFrame(
        [
            {"team": "DEN", "pass_yards_allowed_per_game": 300.0, "rush_yards_allowed_per_game": 100.0, "receptions_allowed_per_game": 20.0},
            {"team": "KC", "pass_yards_allowed_per_game": 100.0, "rush_yards_allowed_per_game": 50.0, "receptions_allowed_per_game": 10.0},
        ]
    )
    # SF's real opponent this week (SEA) is missing from defense_rates_df.
    schedule_df = pd.DataFrame([{"home_team": "SF", "away_team": "SEA"}])

    result = nfl_matchup.attach_matchup_adjustment(players_df, defense_rates_df, schedule_df, weight=1.0).set_index("player_id")

    league_pass = defense_rates_df["pass_yards_allowed_per_game"].mean()
    assert result.loc["p1", "pass_yards_allowed_per_game"] == pytest.approx(league_pass)
    assert result.loc["p1", "Opponent_Pass_Yards_Allowed_Ratio"] == pytest.approx(1.0)


def test_attach_matchup_adjustment_weight_zero_is_neutral_for_every_player():
    players_df = pd.DataFrame([{"player_id": "p1", "team": "SF"}])
    defense_rates_df = pd.DataFrame(
        [{"team": "SEA", "pass_yards_allowed_per_game": 999.0, "rush_yards_allowed_per_game": 999.0, "receptions_allowed_per_game": 999.0}]
    )
    schedule_df = pd.DataFrame([{"home_team": "SF", "away_team": "SEA"}])

    result = nfl_matchup.attach_matchup_adjustment(players_df, defense_rates_df, schedule_df, weight=0.0)

    assert result["Opponent_Pass_Yards_Allowed_Ratio"].iloc[0] == 1.0
    assert result["Opponent_Rush_Yards_Allowed_Ratio"].iloc[0] == 1.0
    assert result["Opponent_Receptions_Allowed_Ratio"].iloc[0] == 1.0
