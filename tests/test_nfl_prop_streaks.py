import pandas as pd
import pytest

from mlb_metrics import config, nfl_prop_streaks


def _week(player_id, team, week, position="WR", position_group="WR", season=2026, name=None, **stats):
    row = {
        "player_id": player_id, "player_display_name": name or player_id,
        "team": team, "position": position, "position_group": position_group,
        "season": season, "week": week, "season_type": "REG",
    }
    row.update(stats)
    return row


def _floors(rows, stat):
    floors = nfl_prop_streaks.compute_season_floors(pd.DataFrame(rows), 2026)
    return floors[floors["stat"] == stat].set_index("player_id")


def test_floor_is_the_minimum_so_it_drops_when_a_game_dips():
    """The user's own example: "5+ every week but then the next week drops
    to 4, the screen should now show that he's had 4+ every week"."""
    rows = [
        _week("wr1", "SF", 1, receptions=5),
        _week("wr1", "SF", 2, receptions=6),
    ]
    assert _floors(rows, "Receptions").loc["wr1", "floor"] == 5

    rows.append(_week("wr1", "SF", 3, receptions=4))
    floors = _floors(rows, "Receptions")
    assert floors.loc["wr1", "floor"] == 4
    # The game log shows every game in week order, so the dip is visible.
    assert floors.loc["wr1", "game_log"] == "5, 6, 4"


def test_floor_is_the_actual_number_not_a_betting_line():
    """This is a record, not a price - no rounding to 5-yard increments."""
    rows = [_week("wr1", "SF", 1, receiving_yards=47), _week("wr1", "SF", 2, receiving_yards=52)]
    assert _floors(rows, "Receiving Yards").loc["wr1", "floor"] == 47


def test_combined_stats_sum_per_game_before_taking_the_floor():
    """An anytime touchdown is rushing OR receiving - one of each in
    different games is still a touchdown every week."""
    rows = [
        _week("rb1", "SF", 1, position="RB", position_group="RB", rushing_tds=1, receiving_tds=0),
        _week("rb1", "SF", 2, position="RB", position_group="RB", rushing_tds=0, receiving_tds=1),
    ]
    assert _floors(rows, "Anytime TD (Rush + Rec)").loc["rb1", "floor"] == 1


def test_a_bye_week_is_not_a_missed_week():
    """The team did not play either, so the streak is unbroken."""
    rows = [
        # SF played weeks 1 and 3 (bye in 2); its receiver played both.
        _week("sf_wr", "SF", 1, receptions=5),
        _week("sf_wr", "SF", 3, receptions=5),
        # KC played all three, so the league has three weeks of data.
        _week("kc_wr", "KC", 1, receptions=5),
        _week("kc_wr", "KC", 2, receptions=5),
        _week("kc_wr", "KC", 3, receptions=5),
    ]
    floors = _floors(rows, "Receptions")
    assert "sf_wr" in floors.index
    assert floors.loc["sf_wr", "games"] == 2


def test_a_game_the_player_sat_out_breaks_the_streak():
    """Literal to the request - "every week" - so a player who missed a
    game his team played is not listed."""
    rows = [
        _week("starter", "SF", 1, receptions=5),
        _week("starter", "SF", 2, receptions=5),
        _week("starter", "SF", 3, receptions=5),
        # Missed week 2 while SF played.
        _week("hurt", "SF", 1, receptions=9),
        _week("hurt", "SF", 3, receptions=9),
    ]
    floors = _floors(rows, "Receptions")
    assert "starter" in floors.index
    assert "hurt" not in floors.index


def test_other_seasons_do_not_leak_into_this_seasons_floor():
    rows = [
        _week("wr1", "SF", 1, season=2025, receptions=12),
        _week("wr1", "SF", 1, receptions=3),
        _week("wr1", "SF", 2, receptions=4),
    ]
    assert _floors(rows, "Receptions").loc["wr1", "floor"] == 3


def _peer_pool(stat_column, values, position="WR", group="WR", team_prefix="T"):
    rows = []
    for index, value in enumerate(values):
        for week in (1, 2):
            rows.append(_week(
                f"p{index}", f"{team_prefix}{index}", week,
                position=position, position_group=group, **{stat_column: value},
            ))
    return rows


def test_rarity_counts_peers_who_matched_or_beat_the_floor():
    # Twelve receivers: one caught 8 every week, two caught 5, nine caught 1.
    rows = _peer_pool("receptions", [8, 5, 5] + [1] * 9)
    board = nfl_prop_streaks.build_streak_board(pd.DataFrame(rows), 2026)
    receptions = board[board["stat"] == "Receptions"].set_index("player_id")

    assert receptions.loc["p0", "players_at_or_above"] == 1
    # 3 of 12 = exactly the 25% cap, so still listed ...
    assert receptions.loc["p1", "players_at_or_above"] == 3
    assert receptions.loc["p0", "pool_size"] == 12
    # ... while "1+ reception, like most receivers" is common - not listed.
    assert "p3" not in receptions.index


def test_rarity_cap_excludes_a_floor_most_peers_share():
    # 3 of 10 at or above 5 is 30%, over the 25% cap.
    rows = _peer_pool("receptions", [8, 5, 5] + [1] * 7)
    board = nfl_prop_streaks.build_streak_board(pd.DataFrame(rows), 2026)
    receptions = board[board["stat"] == "Receptions"].set_index("player_id")

    assert "p0" in receptions.index
    assert "p1" not in receptions.index


def test_rarity_is_measured_only_where_the_stat_is_part_of_the_job():
    """The correctness case that forced per-stat position pools: on the
    live board, pooling every position made "4+ rushing yards every week"
    look rare for a WR, only because receivers seldom carry the ball."""
    rows = _peer_pool("rushing_yards", [4] + [0] * 9, position="WR", group="WR")
    board = nfl_prop_streaks.build_streak_board(pd.DataFrame(rows), 2026)

    assert board[board["stat"] == "Rushing Yards"].empty


def test_kickers_are_pooled_by_position_not_by_special_teams_group():
    """The SPEC group holds punters and long snappers, who never attempt a
    field goal - pooling them in turned "2 of 32 kickers" into "2 of 62"."""
    rows = []
    for index in range(10):
        for week in (1, 2):
            rows.append(_week(f"k{index}", f"K{index}", week, position="K", position_group="SPEC",
                              fg_made=3 if index == 0 else 1))
            rows.append(_week(f"p{index}", f"K{index}", week, position="P", position_group="SPEC", fg_made=0))
    board = nfl_prop_streaks.build_streak_board(pd.DataFrame(rows), 2026)
    fgs = board[board["stat"] == "Field Goals Made"].set_index("player_id")

    assert fgs.loc["k0", "pool_size"] == 10       # kickers only, no punters
    assert fgs.loc["k0", "pool_group"] == "K"


def test_small_pools_are_skipped():
    """"1 of 3" says nothing."""
    rows = _peer_pool("receptions", [8, 1, 1])
    board = nfl_prop_streaks.build_streak_board(pd.DataFrame(rows), 2026)
    assert board[board["stat"] == "Receptions"].empty
    assert config.NFL_STREAK_MIN_POOL > 3


def test_a_zero_floor_is_never_shown():
    rows = _peer_pool("receiving_tds", [0] * 10)
    board = nfl_prop_streaks.build_streak_board(pd.DataFrame(rows), 2026)
    assert board[board["stat"] == "Anytime TD (Rush + Rec)"].empty


def test_touchdowns_come_first_in_the_board_order():
    """The request singled touchdowns out ("touchdowns (especially)")."""
    rows = []
    for index in range(10):
        for week in (1, 2):
            rows.append(_week(
                f"p{index}", f"T{index}", week,
                receptions=9 if index == 0 else 1,
                receiving_tds=1 if index == 1 else 0,
            ))
    board = nfl_prop_streaks.build_streak_board(pd.DataFrame(rows), 2026)
    assert board.iloc[0]["stat"] == "Anytime TD (Rush + Rec)"


def test_next_game_is_attached_and_a_bye_stays_on_the_board():
    """History stands whether or not the player plays this week."""
    rows = _peer_pool("receptions", [8] + [1] * 9)
    schedule = pd.DataFrame([{"home_team": "T1", "away_team": "T2"}])
    board = nfl_prop_streaks.build_streak_board(pd.DataFrame(rows), 2026, schedule)
    receptions = board[board["stat"] == "Receptions"].set_index("player_id")

    # T0 is not on this week's schedule - a bye - and is still listed.
    assert "p0" in receptions.index
    assert pd.isna(receptions.loc["p0", "game"])


def test_the_board_makes_no_prediction():
    """A history section, per the request - no probability, projection or
    odds column should survive into the output."""
    for column in ("hit_probability", "projection", "fair_odds"):
        assert column not in nfl_prop_streaks.STREAK_COLUMNS


def test_empty_inputs_produce_an_empty_board():
    empty = pd.DataFrame(columns=["player_id", "team", "season", "week", "season_type"])
    assert nfl_prop_streaks.compute_season_floors(empty, 2026).empty
    assert nfl_prop_streaks.build_streak_board(empty, 2026).empty


def test_write_leaves_a_prior_file_alone_when_nothing_qualifies(tmp_path):
    import os

    path = str(tmp_path / "board" / "nfl_prop_streaks.csv")
    empty = pd.DataFrame(columns=["player_id", "team", "season", "week", "season_type"])
    result = nfl_prop_streaks.write_streak_board_csv(empty, 2026, None, path)

    assert result.empty
    assert not os.path.exists(path)
