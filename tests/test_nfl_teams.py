import pandas as pd
import pytest

from mlb_metrics import config, nfl_teams


def _weekly_row(team, opponent_team, season, week, passing_yards=0, rushing_yards=0, receptions=0):
    return {
        "team": team,
        "opponent_team": opponent_team,
        "season": season,
        "week": week,
        "passing_yards": passing_yards,
        "rushing_yards": rushing_yards,
        "receptions": receptions,
    }


def test_compute_team_week_allowed_sums_across_players_facing_the_same_defense():
    # Two offensive players from the same team faced defense "SEA" in
    # week 1 - their combined output is what SEA allowed that week.
    rows = [
        _weekly_row("SF", "SEA", 2025, 1, passing_yards=200, rushing_yards=0, receptions=0),
        _weekly_row("SF", "SEA", 2025, 1, passing_yards=0, rushing_yards=80, receptions=5),
        # A different game entirely, must not bleed into SEA's row.
        _weekly_row("KC", "DEN", 2025, 1, passing_yards=300, rushing_yards=100, receptions=8),
    ]
    weekly_df = pd.DataFrame(rows)

    result = nfl_teams.compute_team_week_allowed(weekly_df).set_index("team")

    assert result.loc["SEA", "pass_yards_allowed"] == 200
    assert result.loc["SEA", "rush_yards_allowed"] == 80
    assert result.loc["SEA", "receptions_allowed"] == 5
    assert result.loc["DEN", "pass_yards_allowed"] == 300


def test_nfl_defense_windows_sum_to_one():
    assert sum(weight for _, weight in config.NFL_DEFENSE_WINDOWS) == pytest.approx(1.0)


def test_compute_defense_rolling_rates_blends_windows():
    rows = [_weekly_row("OPP", "SEA", 2025, week, passing_yards=week * 10) for week in range(1, 11)]
    weekly_df = pd.DataFrame(rows)

    result = nfl_teams.compute_defense_rolling_rates(weekly_df).set_index("team")

    assert result.loc["SEA", "games"] == 10

    full_mean, w8_mean, w4_mean = 55.0, 65.0, 85.0
    windows = dict(config.NFL_DEFENSE_WINDOWS)
    expected = full_mean * windows[None] + w8_mean * windows[8] + w4_mean * windows[4]
    assert result.loc["SEA", "pass_yards_allowed_per_game"] == pytest.approx(expected)


def test_compute_defense_rolling_rates_bye_week_excluded_not_zero_filled():
    weeks_and_yards = [(1, 100), (2, 200), (3, 300), (5, 500), (6, 600), (7, 700)]
    rows = [_weekly_row("OPP", "SEA", 2025, week, passing_yards=yards) for week, yards in weeks_and_yards]
    weekly_df = pd.DataFrame(rows)

    result = nfl_teams.compute_defense_rolling_rates(weekly_df).set_index("team")

    assert result.loc["SEA", "games"] == 6

    full_mean, w4_mean = 400.0, 525.0
    windows = dict(config.NFL_DEFENSE_WINDOWS)
    expected = full_mean * windows[None] + full_mean * windows[8] + w4_mean * windows[4]
    assert result.loc["SEA", "pass_yards_allowed_per_game"] == pytest.approx(expected)
