import pandas as pd
import pytest

from mlb_metrics import config, nfl_rush_rec


def _skill_row(player_id, position, season, week, carries=0, rush_yards=0, rush_tds=0, targets=0, receptions=0, rec_yards=0, rec_tds=0, rush_fum_lost=0, rec_fum_lost=0):
    return {
        "player_id": player_id,
        "position": position,
        "season": season,
        "week": week,
        "game_id": f"{season}_{week:02d}_{player_id}",
        "carries": carries,
        "rushing_yards": rush_yards,
        "rushing_tds": rush_tds,
        "targets": targets,
        "receptions": receptions,
        "receiving_yards": rec_yards,
        "receiving_tds": rec_tds,
        "rushing_fumbles_lost": rush_fum_lost,
        "receiving_fumbles_lost": rec_fum_lost,
    }


def test_nfl_skill_windows_sum_to_one():
    assert sum(weight for _, weight in config.NFL_SKILL_WINDOWS) == pytest.approx(1.0)


def test_compute_skill_rolling_stats_blends_windows():
    rows = [_skill_row("wr1", "WR", 2025, week, targets=8, receptions=5, rec_yards=week * 10) for week in range(1, 11)]
    weekly_df = pd.DataFrame(rows)

    result = nfl_rush_rec.compute_skill_rolling_stats(weekly_df).set_index("player_id")

    assert result.loc["wr1", "games"] == 10
    assert result.loc["wr1", "position"] == "WR"

    full_mean, w8_mean, w4_mean = 55.0, 65.0, 85.0
    windows = dict(config.NFL_SKILL_WINDOWS)
    expected = full_mean * windows[None] + w8_mean * windows[8] + w4_mean * windows[4]
    assert result.loc["wr1", "receiving_yards_per_game"] == pytest.approx(expected)
    assert result.loc["wr1", "targets_per_game"] == pytest.approx(8.0)


def test_compute_skill_rolling_stats_bye_week_excluded_not_zero_filled():
    rows = [
        _skill_row("rb1", "RB", 2025, 1, carries=10, rush_yards=100),
        _skill_row("rb1", "RB", 2025, 2, carries=10, rush_yards=200),
        _skill_row("rb1", "RB", 2025, 3, carries=10, rush_yards=300),
        _skill_row("rb1", "RB", 2025, 5, carries=10, rush_yards=500),
        _skill_row("rb1", "RB", 2025, 6, carries=10, rush_yards=600),
        _skill_row("rb1", "RB", 2025, 7, carries=10, rush_yards=700),
    ]
    weekly_df = pd.DataFrame(rows)

    result = nfl_rush_rec.compute_skill_rolling_stats(weekly_df).set_index("player_id")

    assert result.loc["rb1", "games"] == 6

    full_mean, w4_mean = 400.0, 525.0
    windows = dict(config.NFL_SKILL_WINDOWS)
    expected = full_mean * windows[None] + full_mean * windows[8] + w4_mean * windows[4]
    assert result.loc["rb1", "rushing_yards_per_game"] == pytest.approx(expected)


def test_compute_skill_rolling_stats_fumbles_lost_combines_rush_and_receiving_only():
    # sack_fumbles_lost is deliberately excluded (a QB-being-sacked stat,
    # not a skill-player one - see module docstring) - it isn't even a
    # column here, so this also proves the module doesn't require it.
    rows = [_skill_row("rb1", "RB", 2025, 1, carries=10, rush_yards=50, rush_fum_lost=1, rec_fum_lost=1)]
    weekly_df = pd.DataFrame(rows)

    result = nfl_rush_rec.compute_skill_rolling_stats(weekly_df).set_index("player_id")

    assert result.loc["rb1", "fumbles_lost_per_game"] == pytest.approx(2.0)


def test_compute_skill_rolling_stats_filters_to_skill_positions_only():
    rows = [
        _skill_row("rb1", "RB", 2025, 1, carries=10, rush_yards=50),
        {**_skill_row("qb1", "QB", 2025, 1), "position": "QB"},
        {**_skill_row("k1", "K", 2025, 1), "position": "K"},
    ]
    weekly_df = pd.DataFrame(rows)

    result = nfl_rush_rec.compute_skill_rolling_stats(weekly_df)

    assert list(result["player_id"]) == ["rb1"]


def test_compute_skill_rolling_stats_min_games_qualifier_count_is_exposed():
    rows = [_skill_row("rb1", "RB", 2025, 1, carries=10, rush_yards=50)]
    weekly_df = pd.DataFrame(rows)

    result = nfl_rush_rec.compute_skill_rolling_stats(weekly_df).set_index("player_id")

    assert result.loc["rb1", "games"] == 1
    assert result.loc["rb1", "games"] < config.NFL_SKILL_MIN_GAMES
