import numpy as np
import pandas as pd
import pytest

from mlb_metrics import config, nfl_prop_streaks


def _week(player_id, season, week, receptions=0, receiving_yards=0, rushing_yards=0, passing_yards=0):
    return {
        "player_id": player_id, "season": season, "week": week, "season_type": "REG",
        "receptions": receptions, "receiving_yards": receiving_yards,
        "rushing_yards": rushing_yards, "passing_yards": passing_yards,
    }


def _edge(player_id, category, projection, player_name="Player One"):
    return {
        "player_id": player_id, "player_name": player_name, "team": "SF",
        "opponent": "SEA", "game": "SEA @ SF", "category": category, "projection": projection,
    }


def test_highest_line_always_cleared_for_a_whole_number_stat():
    """A back whose worst game was exactly 4 receptions carries 3.5 - he
    cleared it - not 4.5, which he did not."""
    assert nfl_prop_streaks.highest_line_always_cleared(4.0, 1.0) == pytest.approx(3.5)
    assert nfl_prop_streaks.highest_line_always_cleared(3.0, 1.0) == pytest.approx(2.5)
    assert nfl_prop_streaks.highest_line_always_cleared(1.0, 1.0) == pytest.approx(0.5)


def test_highest_line_always_cleared_for_a_five_yard_step():
    """Yardage props are posted in 5-yard steps, so a receiver whose
    worst week was 12 yards carries 9.5."""
    assert nfl_prop_streaks.highest_line_always_cleared(12.0, 5.0) == pytest.approx(9.5)
    assert nfl_prop_streaks.highest_line_always_cleared(20.0, 5.0) == pytest.approx(19.5)
    assert nfl_prop_streaks.highest_line_always_cleared(45.0, 5.0) == pytest.approx(44.5)
    # Always strictly below the floor - never a line the player failed.
    for low in (7.0, 13.0, 28.0, 101.0):
        assert nfl_prop_streaks.highest_line_always_cleared(low, 5.0) < low


def test_highest_line_always_cleared_returns_zero_for_a_scoreless_game():
    """A player held scoreless in any game has no streak at any line."""
    assert nfl_prop_streaks.highest_line_always_cleared(0.0, 1.0) == 0.0
    assert nfl_prop_streaks.highest_line_always_cleared(0.0, 5.0) == 0.0
    assert nfl_prop_streaks.highest_line_always_cleared(float("nan"), 1.0) == 0.0


def test_probability_to_american_odds_both_sides_of_even_money():
    assert nfl_prop_streaks.probability_to_american_odds(0.75) == -300
    assert nfl_prop_streaks.probability_to_american_odds(0.50) == -100
    assert nfl_prop_streaks.probability_to_american_odds(0.40) == 150
    # The price the request called out as not worth showing.
    assert nfl_prop_streaks.probability_to_american_odds(0.9167) == pytest.approx(-1100, abs=5)


def test_season_streaks_require_a_player_to_have_played_every_week():
    """A streak with a gap in it is not a streak - literal to the request
    ("hit every week of that season to that point")."""
    rows = [
        _week("ever_present", 2026, 1, receptions=5),
        _week("ever_present", 2026, 2, receptions=4),
        # Missed week 1 entirely, however good week 2 was.
        _week("missed_a_week", 2026, 2, receptions=9),
    ]
    streaks = nfl_prop_streaks.compute_season_streaks(pd.DataFrame(rows), 2026)
    receptions = streaks[streaks["category"] == "Receptions"]

    assert list(receptions["player_id"]) == ["ever_present"]
    # Measured at his FLOOR (4), not his average.
    assert receptions.iloc[0]["line"] == pytest.approx(3.5)
    assert receptions.iloc[0]["season_low"] == pytest.approx(4.0)


def test_season_streaks_use_the_floor_not_the_average():
    """One bad week must sink the line, which is what lets the board
    claim it has never failed."""
    rows = [
        _week("steady", 2026, 1, receiving_yards=60),
        _week("steady", 2026, 2, receiving_yards=58),
        _week("spiky", 2026, 1, receiving_yards=115),
        _week("spiky", 2026, 2, receiving_yards=6),
    ]
    streaks = nfl_prop_streaks.compute_season_streaks(pd.DataFrame(rows), 2026)
    lines = streaks[streaks["category"] == "Receiving Yards"].set_index("player_id")["line"]

    # Both average around 59, but the spiky player's floor is 6.
    assert lines["steady"] == pytest.approx(54.5)
    assert lines["spiky"] == pytest.approx(4.5)


def test_season_streaks_ignore_other_seasons():
    rows = [
        _week("wr1", 2025, 1, receptions=9),
        _week("wr1", 2025, 2, receptions=9),
        _week("wr1", 2026, 1, receptions=3),
        _week("wr1", 2026, 2, receptions=2),
    ]
    streaks = nfl_prop_streaks.compute_season_streaks(pd.DataFrame(rows), 2026)
    receptions = streaks[streaks["category"] == "Receptions"].set_index("player_id")

    # Last season's monster games must not lift this season's floor.
    assert receptions.loc["wr1", "line"] == pytest.approx(1.5)


def test_streak_board_ranks_on_probability_not_on_streak_length():
    """The heart of the design. A perfect streak is a weak signal - with
    two weeks played, a player whose true rate is 70% runs one 49% of the
    time - so the streak decides eligibility and the model's projection
    decides the order.
    """
    rows = []
    # Both players have an identical, perfect two-week streak at the same
    # line. Only the projection separates them.
    for player in ("modest", "strong"):
        rows += [_week(player, 2026, 1, receptions=4), _week(player, 2026, 2, receptions=4)]
    edges = pd.DataFrame([
        _edge("modest", "Receptions", 4.1, "Modest Projection"),
        _edge("strong", "Receptions", 5.6, "Strong Projection"),
    ])

    board = nfl_prop_streaks.build_streak_board(edges, pd.DataFrame(rows), 2026, probability_band=(0.0, 1.0))

    assert list(board["player_name"]) == ["Strong Projection", "Modest Projection"]
    assert board.iloc[0]["hit_probability"] > board.iloc[1]["hit_probability"]


def test_streak_board_excludes_prices_that_are_not_worth_taking():
    """The user's own constraint: "something with odds of -1100 probably
    shouldn't show, because winning means very little and losing means
    losing money"."""
    rows = [
        # A near-certainty at its line - the -1100 case.
        _week("lock", 2026, 1, receptions=4), _week("lock", 2026, 2, receptions=4),
        # A genuine coin flip - too thin to call a repeat likely.
        _week("toss_up", 2026, 1, receptions=4), _week("toss_up", 2026, 2, receptions=4),
    ]
    edges = pd.DataFrame([
        _edge("lock", "Receptions", 12.0, "Near Lock"),
        _edge("toss_up", "Receptions", 3.5, "Toss Up"),
    ])

    board = nfl_prop_streaks.build_streak_board(edges, pd.DataFrame(rows), 2026)

    names = list(board["player_name"])
    assert "Near Lock" not in names      # too short to be worth taking
    assert "Toss Up" not in names        # too thin to call a repeat


def test_streak_board_reports_fair_odds_inside_the_configured_band():
    rows = []
    for index in range(6):
        player = f"wr{index}"
        rows += [
            _week(player, 2026, 1, receiving_yards=40 + index * 6),
            _week(player, 2026, 2, receiving_yards=42 + index * 6),
        ]
    edges = pd.DataFrame([
        _edge(f"wr{index}", "Receiving Yards", 55 + index * 8, f"WR {index}")
        for index in range(6)
    ])

    board = nfl_prop_streaks.build_streak_board(edges, pd.DataFrame(rows), 2026)
    low, high = config.NFL_STREAK_PROBABILITY_BAND

    assert not board.empty
    assert board["hit_probability"].between(low, high).all()
    # Fair odds and probability must agree - the odds column is derived,
    # not independently computed.
    for _, row in board.iterrows():
        assert row["fair_odds"] == nfl_prop_streaks.probability_to_american_odds(row["hit_probability"])


def test_streak_board_excludes_sacks():
    """Sacks has no fitted outcome spread (no projection to take
    residuals against), so a probability for it would be backed by
    nothing."""
    assert "Sacks" not in nfl_prop_streaks.STREAK_CATEGORY_LINES


def test_streak_board_survives_empty_inputs():
    empty_weekly = pd.DataFrame(columns=["player_id", "season", "week", "season_type", "receptions"])
    empty_edges = pd.DataFrame(columns=["player_id", "category", "projection"])

    assert nfl_prop_streaks.compute_season_streaks(empty_weekly, 2026).empty
    assert nfl_prop_streaks.build_streak_board(empty_edges, empty_weekly, 2026).empty
    # And a season nobody has played yet.
    rows = pd.DataFrame([_week("wr1", 2025, 1, receptions=5)])
    assert nfl_prop_streaks.compute_season_streaks(rows, 2026).empty


def test_write_streak_board_leaves_a_prior_file_alone_when_nothing_qualifies(tmp_path):
    path = str(tmp_path / "board" / "nfl_prop_streaks.csv")
    empty = pd.DataFrame(columns=["player_id", "category", "projection"])

    result = nfl_prop_streaks.write_streak_board_csv(
        empty, pd.DataFrame(columns=["player_id", "season", "week"]), 2026, path
    )

    assert result.empty
    import os
    assert not os.path.exists(path)
