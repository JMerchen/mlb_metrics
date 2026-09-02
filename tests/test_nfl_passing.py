import pandas as pd
import pytest

from mlb_metrics import config, nfl_passing


def _qb_row(player_id, season, week, attempts, completions, passing_yards, passing_tds, ints, carries, rush_yards, rush_tds, passing_epa=0.0):
    return {
        "player_id": player_id,
        "position": "QB",
        "season": season,
        "week": week,
        "game_id": f"{season}_{week:02d}_{player_id}",
        "attempts": attempts,
        "completions": completions,
        "passing_yards": passing_yards,
        "passing_tds": passing_tds,
        "passing_interceptions": ints,
        "carries": carries,
        "rushing_yards": rush_yards,
        "rushing_tds": rush_tds,
        "passing_epa": passing_epa,
    }


def test_nfl_qb_windows_sum_to_one():
    assert sum(weight for _, weight in config.NFL_QB_WINDOWS) == pytest.approx(1.0)


def test_compute_qb_rolling_stats_blends_windows():
    # Weeks 1-10, one game per week, no bye - most-recent-4 window is
    # weeks 7-10, full window is weeks 1-10 (config.NFL_QB_WINDOWS has no
    # 8-game intermediate window collision issue here since we use 10
    # games total, matching the (None, 8, 4) windows real shape).
    rows = [_qb_row("qb1", 2025, week, attempts=30, completions=20, passing_yards=week * 10, passing_tds=1, ints=0, carries=2, rush_yards=5, rush_tds=0) for week in range(1, 11)]
    weekly_df = pd.DataFrame(rows)

    result = nfl_passing.compute_qb_rolling_stats(weekly_df).set_index("player_id")

    assert result.loc["qb1", "games"] == 10

    # passing_yards_per_game: full window mean of 10..100 (step 10) = 55;
    # 8-game window (weeks 3-10) mean of 30..100 = 65; 4-game window
    # (weeks 7-10) mean of 70..100 = 85.
    full_mean, w8_mean, w4_mean = 55.0, 65.0, 85.0
    windows = dict(config.NFL_QB_WINDOWS)
    expected = full_mean * windows[None] + w8_mean * windows[8] + w4_mean * windows[4]
    assert result.loc["qb1", "passing_yards_per_game"] == pytest.approx(expected)

    # attempts_per_game is constant (30) across every window regardless of blend.
    assert result.loc["qb1", "attempts_per_game"] == pytest.approx(30.0)


def test_compute_qb_rolling_stats_bye_week_excluded_not_zero_filled():
    # Weeks 1, 2, 3, 5, 6, 7 (bye at week 4 - simply absent, matching
    # nflreadpy's real behavior). If the bye were wrongly treated as a
    # real game with zero stats, the most-recent-4-games window would
    # wrongly include a phantom zero row between week 3 and week 5.
    rows = [
        _qb_row("qb1", 2025, 1, 20, 10, 100, 0, 0, 0, 0, 0),
        _qb_row("qb1", 2025, 2, 20, 10, 200, 0, 0, 0, 0, 0),
        _qb_row("qb1", 2025, 3, 20, 10, 300, 0, 0, 0, 0, 0),
        _qb_row("qb1", 2025, 5, 20, 10, 500, 0, 0, 0, 0, 0),
        _qb_row("qb1", 2025, 6, 20, 10, 600, 0, 0, 0, 0, 0),
        _qb_row("qb1", 2025, 7, 20, 10, 700, 0, 0, 0, 0, 0),
    ]
    weekly_df = pd.DataFrame(rows)

    result = nfl_passing.compute_qb_rolling_stats(weekly_df).set_index("player_id")

    assert result.loc["qb1", "games"] == 6

    # Full/8-game windows both cover all 6 real games: mean = 2400/6 = 400.
    # The 4-game window covers the 4 REAL most-recent games (weeks 7/6/5/3,
    # correctly skipping the bye rather than zero-padding it):
    # (700+600+500+300)/4 = 525, not (700+600+500+0)/4 = 450.
    full_mean, w4_mean = 400.0, 525.0
    windows = dict(config.NFL_QB_WINDOWS)
    expected = full_mean * windows[None] + full_mean * windows[8] + w4_mean * windows[4]
    assert result.loc["qb1", "passing_yards_per_game"] == pytest.approx(expected)


def test_compute_qb_rolling_stats_filters_to_qb_position_only():
    rows = [
        _qb_row("qb1", 2025, 1, 20, 10, 100, 0, 0, 0, 0, 0),
        {**_qb_row("rb1", 2025, 1, 0, 0, 0, 0, 0, 15, 80, 1), "position": "RB"},
    ]
    weekly_df = pd.DataFrame(rows)

    result = nfl_passing.compute_qb_rolling_stats(weekly_df)

    assert list(result["player_id"]) == ["qb1"]


def test_compute_qb_rolling_stats_blends_passing_epa():
    # passing_epa is a real per-game QB efficiency total (nfl_team_strength's
    # QB-continuity adjustment relies on it) - blended across NFL_QB_WINDOWS
    # the same way every other STAT_COLS entry is.
    rows = [_qb_row("qb1", 2025, week, attempts=30, completions=20, passing_yards=200, passing_tds=1, ints=0,
                     carries=2, rush_yards=5, rush_tds=0, passing_epa=float(week)) for week in range(1, 11)]
    weekly_df = pd.DataFrame(rows)

    result = nfl_passing.compute_qb_rolling_stats(weekly_df).set_index("player_id")

    # Same window shape as the passing_yards blend test: full mean of
    # weeks 1-10 = 5.5; 8-game window (weeks 3-10) mean = 6.5; 4-game
    # window (weeks 7-10) mean = 8.5.
    full_mean, w8_mean, w4_mean = 5.5, 6.5, 8.5
    windows = dict(config.NFL_QB_WINDOWS)
    expected = full_mean * windows[None] + w8_mean * windows[8] + w4_mean * windows[4]
    assert result.loc["qb1", "passing_epa_per_game"] == pytest.approx(expected)


def test_compute_qb_rolling_stats_min_games_qualifier_count_is_exposed():
    # A QB with only 1 game of history - fewer than config.NFL_QB_MIN_GAMES.
    # This module doesn't gate on it (that's the DK-scoring consumer's
    # job, Phase 4) but must expose an accurate `games` count so that
    # gate can be applied downstream.
    rows = [_qb_row("qb1", 2025, 1, 20, 10, 100, 0, 0, 0, 0, 0)]
    weekly_df = pd.DataFrame(rows)

    result = nfl_passing.compute_qb_rolling_stats(weekly_df).set_index("player_id")

    assert result.loc["qb1", "games"] == 1
    assert result.loc["qb1", "games"] < config.NFL_QB_MIN_GAMES
