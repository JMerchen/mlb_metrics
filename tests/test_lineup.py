import pandas as pd
import pytest

from mlb_metrics import config, lineup


def _order_rows(team, batter, game_ids, batting_order):
    return [{"team": team, "batter": batter, "game_id": g, "batting_order": batting_order} for g in game_ids]


def _pa_rows(batter, game_ids, pa_count):
    """data_with_game_id-shaped rows: `pa_count` distinct at_bat_number
    values for `batter` in each of `game_ids` - compute_expected_plate_appearances
    counts real PA via at_bat_number.nunique()."""
    return [
        {"game_id": g, "batter": batter, "at_bat_number": ab}
        for g in game_ids
        for ab in range(1, pa_count + 1)
    ]


def test_compute_lineup_consistency_windows_to_most_recent_games(monkeypatch):
    monkeypatch.setattr(config, "LINEUP_WINDOW_GAMES", 5)

    rows = (
        # Filler regular so team X registers 8 total games (1-8).
        _order_rows("X", 502, range(1, 9), 1)
        # Batter 501 started games 1-3 (outside the 5-game window: 4-8) and
        # 6-7 (inside it), always batting 3rd.
        + _order_rows("X", 501, [1, 2, 3, 6, 7], 3)
    )
    batting_order = pd.DataFrame(rows)
    latest_team = pd.DataFrame([{"key_mlbam": 501, "team": "X"}, {"key_mlbam": 502, "team": "X"}])

    result = lineup.compute_lineup_consistency(batting_order, latest_team).set_index("key_mlbam")

    assert result.loc[501, "avg_batting_order"] == 3
    assert result.loc[501, "start_rate"] == 2 / 5  # only games 6,7 fall in the 5-game window


def test_compute_lineup_consistency_uses_actual_games_played_early_in_season(monkeypatch):
    monkeypatch.setattr(config, "LINEUP_WINDOW_GAMES", 5)

    # Team Y has only played 3 games all season - the denominator must be 3,
    # not the full window of 5, so a batter who started every game so far
    # isn't penalized for the season being young.
    rows = _order_rows("Y", 601, [1, 2, 3], 2)
    batting_order = pd.DataFrame(rows)
    latest_team = pd.DataFrame([{"key_mlbam": 601, "team": "Y"}])

    result = lineup.compute_lineup_consistency(batting_order, latest_team).set_index("key_mlbam")

    assert result.loc[601, "start_rate"] == 1.0
    assert result.loc[601, "avg_batting_order"] == 2


def test_compute_lineup_consistency_resets_window_on_trade(monkeypatch):
    monkeypatch.setattr(config, "LINEUP_WINDOW_GAMES", 5)

    rows = (
        # Batter 701 started every game for OLD before being traded.
        _order_rows("OLD", 701, [101, 102, 103, 104, 105], 2)
        # Then started for NEW afterward, batting 4th.
        + _order_rows("NEW", 701, [201, 202, 203], 4)
    )
    batting_order = pd.DataFrame(rows)
    # latest_team reflects the trade: 701 is now on NEW.
    latest_team = pd.DataFrame([{"key_mlbam": 701, "team": "NEW"}])

    result = lineup.compute_lineup_consistency(batting_order, latest_team).set_index("key_mlbam")

    # Only NEW's 3 games count - OLD's history (a different slot, more
    # games) must not leak into the current-team signal.
    assert result.loc[701, "avg_batting_order"] == 4
    assert result.loc[701, "start_rate"] == 1.0


def test_compute_lineup_consistency_null_avg_for_never_started():
    # A batter with no starts at all for their current team must get a null
    # avg_batting_order (not a misleading 0) and a start_rate of 0.
    batting_order = pd.DataFrame(_order_rows("Z", 801, [1, 2, 3], 1))
    latest_team = pd.DataFrame([{"key_mlbam": 999, "team": "Z"}])  # never appears in batting_order

    result = lineup.compute_lineup_consistency(batting_order, latest_team).set_index("key_mlbam")

    assert pd.isna(result.loc[999, "avg_batting_order"])
    assert result.loc[999, "start_rate"] == 0


def test_compute_lineup_consistency_accepts_a_custom_window(monkeypatch):
    # config.LINEUP_WINDOW_GAMES is NOT monkeypatched here - passing
    # window= explicitly must override it, since
    # compute_expected_plate_appearances relies on this to use a shorter
    # window than compute_lineup_consistency's other callers.
    monkeypatch.setattr(config, "LINEUP_WINDOW_GAMES", 20)

    rows = (
        _order_rows("X", 502, range(1, 9), 1)
        + _order_rows("X", 501, [1, 2, 3, 6, 7], 3)
    )
    batting_order = pd.DataFrame(rows)
    latest_team = pd.DataFrame([{"key_mlbam": 501, "team": "X"}, {"key_mlbam": 502, "team": "X"}])

    result = lineup.compute_lineup_consistency(batting_order, latest_team, window=5).set_index("key_mlbam")

    assert result.loc[501, "start_rate"] == 2 / 5  # only games 6,7 fall in the 5-game window


def test_compute_expected_plate_appearances_uses_real_per_slot_pa_average():
    # Two leadoff (slot 1) starts averaging 5 real PA, two #9 (slot 9)
    # starts averaging 3 real PA - a real, empirically-derived table, not
    # an assumed one.
    batting_order = pd.DataFrame(
        _order_rows("X", 10, [1, 2], 1) + _order_rows("X", 20, [1, 2], 9)
    )
    data_with_game_id = pd.DataFrame(_pa_rows(10, [1, 2], 5) + _pa_rows(20, [1, 2], 3))
    latest_team = pd.DataFrame([{"key_mlbam": 10, "team": "X"}, {"key_mlbam": 20, "team": "X"}])

    result = lineup.compute_expected_plate_appearances(
        data_with_game_id, batting_order, latest_team, window=5
    ).set_index("key_mlbam")

    assert result.loc[10, "Recent_Avg_Batting_Order"] == 1
    assert result.loc[10, "Expected_PA"] == 5
    assert result.loc[20, "Recent_Avg_Batting_Order"] == 9
    assert result.loc[20, "Expected_PA"] == 3


def test_compute_expected_plate_appearances_interpolates_fractional_slots():
    # Batter 10 splits time between slot 1 (5 real PA/game) and slot 3
    # (4 real PA/game) - a recent average slot of 2.0 should land exactly
    # halfway between those two real per-slot averages (4.5), not round
    # to the nearest whole slot and discard the fractional precision.
    batting_order = pd.DataFrame(
        _order_rows("X", 10, [1], 1) + _order_rows("X", 10, [2], 3) + _order_rows("X", 20, [1, 2], 9)
    )
    data_with_game_id = pd.DataFrame(
        _pa_rows(10, [1], 5) + _pa_rows(10, [2], 4) + _pa_rows(20, [1, 2], 3)
    )
    latest_team = pd.DataFrame([{"key_mlbam": 10, "team": "X"}])

    result = lineup.compute_expected_plate_appearances(
        data_with_game_id, batting_order, latest_team, window=5
    ).set_index("key_mlbam")

    assert result.loc[10, "Recent_Avg_Batting_Order"] == 2.0
    assert result.loc[10, "Expected_PA"] == pytest.approx(4.5)


def test_compute_expected_plate_appearances_falls_back_to_league_average():
    # Batter 999 never started for their current team in the window - no
    # slot to interpolate from - must fall back to the real league-wide
    # average PA/game across all real starters, not a fabricated value.
    batting_order = pd.DataFrame(
        _order_rows("X", 10, [1, 2], 1) + _order_rows("X", 20, [1, 2], 9)
    )
    data_with_game_id = pd.DataFrame(_pa_rows(10, [1, 2], 5) + _pa_rows(20, [1, 2], 3))
    latest_team = pd.DataFrame([{"key_mlbam": 999, "team": "X"}])

    result = lineup.compute_expected_plate_appearances(
        data_with_game_id, batting_order, latest_team, window=5
    ).set_index("key_mlbam")

    assert pd.isna(result.loc[999, "Recent_Avg_Batting_Order"])
    assert result.loc[999, "Expected_PA"] == pytest.approx(4.0)  # mean of 5,5,3,3


def test_compute_expected_plate_appearances_uses_its_own_shorter_default_window(monkeypatch):
    # LINEUP_RECENT_WINDOW_GAMES defaults to a real, shorter window than
    # LINEUP_WINDOW_GAMES - confirmed by monkeypatching only the recent
    # one and checking it actually takes effect without an explicit
    # window= argument.
    monkeypatch.setattr(config, "LINEUP_RECENT_WINDOW_GAMES", 2)

    # Batter 10 batted leadoff for games 1-3, then dropped to #9 for game
    # 4 (most recent). Only the last 2 games (3, 4) should count.
    batting_order = pd.DataFrame(
        _order_rows("X", 10, [1, 2, 3], 1) + _order_rows("X", 10, [4], 9)
        + _order_rows("X", 20, [1, 2, 3, 4], 5)  # filler so team X has 4 real games
    )
    data_with_game_id = pd.DataFrame(
        _pa_rows(10, [1, 2, 3], 5) + _pa_rows(10, [4], 3) + _pa_rows(20, [1, 2, 3, 4], 4)
    )
    latest_team = pd.DataFrame([{"key_mlbam": 10, "team": "X"}])

    result = lineup.compute_expected_plate_appearances(
        data_with_game_id, batting_order, latest_team
    ).set_index("key_mlbam")

    # Last 2 games: slot 1 (game 3) and slot 9 (game 4) -> average slot 5.
    assert result.loc[10, "Recent_Avg_Batting_Order"] == 5
