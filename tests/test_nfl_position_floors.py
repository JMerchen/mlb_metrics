import os

import pandas as pd

from mlb_metrics import config, nfl_position_floors

SEASON = 2026


def _snap(pfr, team, week, position="WR", snaps=50, pct=0.8, name=None, season=SEASON):
    offense = position not in ("DE", "DT", "NT", "DL", "CB", "S", "FS", "SS", "DB")
    return {
        "season": season, "game_type": "REG", "week": week,
        "pfr_player_id": pfr, "player": name or pfr, "position": position, "team": team,
        "offense_snaps": snaps if offense else 0, "offense_pct": pct if offense else 0.0,
        "defense_snaps": 0 if offense else snaps, "defense_pct": 0.0 if offense else pct,
    }


def _roster(pfr, week, status="ACT", season=SEASON):
    return {"season": season, "week": week, "pfr_id": pfr, "gsis_id": f"g_{pfr}", "status": status}


def _stat(pfr, week, season=SEASON, **stats):
    row = {"player_id": f"g_{pfr}", "season": season, "week": week, "season_type": "REG"}
    row.update(stats)
    return row


def _board(snaps, rosters, weekly, schedule=None):
    weekly_df = pd.DataFrame(weekly) if weekly else pd.DataFrame(columns=["player_id", "season", "week", "season_type"])
    return nfl_position_floors.compute_position_floors(
        pd.DataFrame(snaps), pd.DataFrame(rosters), weekly_df, SEASON, schedule
    ).set_index("player_id")


def test_a_game_with_snaps_and_no_stats_row_is_a_zero_floor():
    """The stats table skips a game where the player recorded nothing, so
    the game list has to come from snap counts - otherwise this floor would
    read 5 when the player really had a zero week."""
    snaps = [_snap("wr1", "SF", w) for w in (1, 2, 3)]
    rosters = [_roster("wr1", w) for w in (1, 2, 3, 4)]
    weekly = [_stat("wr1", 1, receptions=5), _stat("wr1", 3, receptions=6)]
    board = _board(snaps, rosters, weekly)

    assert board.loc["g_wr1", "receptions"] == 0
    assert board.loc["g_wr1", "receptions_log"] == "5, 0, 6"


def test_floor_is_the_minimum_over_counted_games():
    snaps = [_snap("wr1", "SF", w) for w in (1, 2, 3)]
    rosters = [_roster("wr1", w) for w in (1, 2, 3)]
    weekly = [
        _stat("wr1", 1, receptions=5, receiving_yards=70),
        _stat("wr1", 2, receptions=6, receiving_yards=41),
        _stat("wr1", 3, receptions=4, receiving_yards=88),
    ]
    board = _board(snaps, rosters, weekly)
    assert board.loc["g_wr1", "receptions"] == 4
    assert board.loc["g_wr1", "receiving_yards"] == 41


def test_anytime_td_sums_rushing_and_receiving():
    snaps = [_snap("rb1", "SF", w, position="RB") for w in (1, 2)]
    rosters = [_roster("rb1", w) for w in (1, 2)]
    weekly = [_stat("rb1", 1, rushing_tds=1), _stat("rb1", 2, receiving_tds=1)]
    assert _board(snaps, rosters, weekly).loc["g_rb1", "anytime_td"] == 1


def _injury_case(next_status):
    """wr1 plays ~80% of snaps, drops to 10% in week 2 with a 0-catch
    game, and his team plays again in week 3 while he has `next_status`."""
    snaps = [
        _snap("wr1", "SF", 1, pct=0.8),
        _snap("wr1", "SF", 2, snaps=6, pct=0.1),
        _snap("wr1", "SF", 4, pct=0.8),
        _snap("wr1", "SF", 5, pct=0.8),
        # A teammate proves SF played week 3.
        _snap("wr2", "SF", 3), _snap("wr2", "SF", 5),
    ]
    rosters = [_roster("wr1", w) for w in (1, 2, 4, 5)] + [_roster("wr1", 3, next_status)]
    rosters += [_roster("wr2", w) for w in (3, 5)]
    weekly = [
        _stat("wr1", 1, receptions=5), _stat("wr1", 4, receptions=6), _stat("wr1", 5, receptions=7),
    ]
    return _board(snaps, rosters, weekly)


def test_a_confirmed_injury_exit_does_not_count_toward_the_floor():
    board = _injury_case("INA")
    assert board.loc["g_wr1", "receptions"] == 5
    assert board.loc["g_wr1", "excluded_weeks"] == "2"
    assert board.loc["g_wr1", "games_played"] == 4
    assert board.loc["g_wr1", "games_counted"] == 3
    # Still visible in the log, bracketed, so nothing is hidden.
    assert board.loc["g_wr1", "receptions_log"] == "5, [0], 6, 7"


def test_reserve_also_confirms_an_injury():
    assert _injury_case("RES").loc["g_wr1", "receptions"] == 5


def test_a_collapse_while_still_active_counts_as_a_real_low_game():
    """Blowouts, benchings and a backup's mop-up duty look like an injury
    in the snap data; only the roster status tells them apart."""
    board = _injury_case("ACT")
    assert board.loc["g_wr1", "receptions"] == 0
    assert board.loc["g_wr1", "excluded_weeks"] == ""


def test_confirmation_skips_the_bye_to_the_teams_next_game():
    """SF is on bye in week 3; the injury is confirmed by the week-4
    roster, not by the empty bye week."""
    snaps = [
        _snap("wr1", "SF", 1, pct=0.8),
        _snap("wr1", "SF", 2, snaps=6, pct=0.1),
        _snap("wr1", "SF", 5, pct=0.8),
        _snap("wr2", "SF", 4), _snap("wr2", "SF", 5),
    ]
    rosters = [_roster("wr1", w) for w in (1, 2, 5)] + [_roster("wr1", 4, "INA")]
    rosters += [_roster("wr2", w) for w in (4, 5)]
    weekly = [_stat("wr1", 1, receptions=5), _stat("wr1", 5, receptions=6)]
    board = _board(snaps, rosters, weekly)
    assert board.loc["g_wr1", "excluded_weeks"] == "2"
    assert board.loc["g_wr1", "receptions"] == 5


def test_the_latest_game_is_counted_and_flagged_not_excluded():
    """Nobody can know yet whether he misses the next game, and most
    early exits are not injuries - so it counts, with a flag."""
    snaps = [_snap("wr1", "SF", 1, pct=0.8), _snap("wr1", "SF", 2, pct=0.8), _snap("wr1", "SF", 3, snaps=5, pct=0.1)]
    rosters = [_roster("wr1", w) for w in (1, 2, 3)]
    weekly = [_stat("wr1", 1, receptions=5), _stat("wr1", 2, receptions=6)]
    board = _board(snaps, rosters, weekly)
    assert board.loc["g_wr1", "receptions"] == 0
    assert board.loc["g_wr1", "flagged_week"] == "3"
    assert board.loc["g_wr1", "receptions_log"] == "5, 6, 0?"


def test_top_n_by_season_snaps_per_tab(monkeypatch):
    monkeypatch.setattr(config, "NFL_FLOOR_TOP_N", 2)
    snaps = [
        _snap("wr_a", "SF", 1, snaps=60), _snap("wr_b", "KC", 1, snaps=50), _snap("wr_c", "LV", 1, snaps=10),
        _snap("te_a", "SF", 1, position="TE", snaps=5),
    ]
    rosters = [_roster(p, 1) for p in ("wr_a", "wr_b", "wr_c", "te_a")]
    board = _board(snaps, rosters, [])
    wr = board[board["tab"] == "WR"]
    assert list(wr.index) == ["g_wr_a", "g_wr_b"]
    assert list(wr["rank"]) == [1, 2]
    # A small tab still lists whoever it has.
    assert list(board[board["tab"] == "TE"].index) == ["g_te_a"]


def test_defenders_use_defensive_snaps_and_defensive_stats():
    snaps = [_snap("cb1", "SF", w, position="CB") for w in (1, 2)] + [_snap("de1", "SF", 1, position="DE")]
    rosters = [_roster("cb1", w) for w in (1, 2)] + [_roster("de1", 1)]
    weekly = [
        _stat("cb1", 1, def_tackles_solo=3, def_tackle_assists=2, def_interceptions=1),
        _stat("cb1", 2, def_tackles_solo=4, def_tackle_assists=0),
        _stat("de1", 1, def_sacks=1.5),
    ]
    board = _board(snaps, rosters, weekly)
    assert board.loc["g_cb1", "tab"] == "DB"
    assert board.loc["g_cb1", "tackles"] == 4
    assert board.loc["g_cb1", "def_interceptions"] == 0
    assert board.loc["g_de1", "tab"] == "DL"
    assert board.loc["g_de1", "sacks_log"] == "1.5"
    # Offensive stats are not columns for a defender.
    assert pd.isna(board.loc["g_cb1", "receptions"])


def test_next_game_is_attached_and_a_bye_is_blank():
    snaps = [_snap("wr1", "SF", 1), _snap("wr2", "KC", 1)]
    rosters = [_roster("wr1", 1), _roster("wr2", 1)]
    schedule = pd.DataFrame([{"home_team": "SF", "away_team": "LA"}])
    board = _board(snaps, rosters, [], schedule)
    assert board.loc["g_wr1", "game"] == "LA @ SF"
    assert pd.isna(board.loc["g_wr2", "game"])


def test_other_seasons_and_non_positions_are_ignored():
    snaps = [
        _snap("wr1", "SF", 1), _snap("wr1", "SF", 1, season=SEASON - 1),
        _snap("k1", "SF", 1, position="K"),
    ]
    rosters = [_roster("wr1", 1), _roster("k1", 1)]
    weekly = [_stat("wr1", 1, receptions=3), _stat("wr1", 1, season=SEASON - 1, receptions=12)]
    board = _board(snaps, rosters, weekly)
    assert list(board.index) == ["g_wr1"]
    assert board.loc["g_wr1", "receptions_log"] == "3"


def test_empty_inputs_write_nothing(tmp_path):
    path = str(tmp_path / "out" / "nfl_position_floors.csv")
    result = nfl_position_floors.write_position_floors_csv(
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), SEASON, None, path
    )
    assert result.empty
    assert not os.path.exists(path)
