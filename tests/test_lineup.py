import pandas as pd

from mlb_metrics import config, lineup


def _order_rows(team, batter, game_ids, batting_order):
    return [{"team": team, "batter": batter, "game_id": g, "batting_order": batting_order} for g in game_ids]


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
