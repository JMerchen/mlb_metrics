import pandas as pd
import pytest

from mlb_metrics import config, dfs_backtest


def _statcast_game(game_pk, date, home_team, away_team, home_pitcher, away_pitcher, home_score, away_score):
    """A minimal 4-row single-game Statcast fixture - same shape
    test_game_picks_backtest.py's fixture uses, since
    derive_historical_team_schedule reuses derive_historical_schedule_games
    directly."""
    return [
        {"game_pk": game_pk, "game_date": date, "home_team": home_team, "away_team": away_team,
         "pitcher": home_pitcher, "inning_topbot": "Top", "at_bat_number": 1,
         "post_home_score": 0, "post_away_score": 0},
        {"game_pk": game_pk, "game_date": date, "home_team": home_team, "away_team": away_team,
         "pitcher": away_pitcher, "inning_topbot": "Bot", "at_bat_number": 2,
         "post_home_score": min(1, home_score), "post_away_score": 0},
        {"game_pk": game_pk, "game_date": date, "home_team": home_team, "away_team": away_team,
         "pitcher": home_pitcher, "inning_topbot": "Top", "at_bat_number": 3,
         "post_home_score": min(1, home_score), "post_away_score": min(1, away_score)},
        {"game_pk": game_pk, "game_date": date, "home_team": home_team, "away_team": away_team,
         "pitcher": away_pitcher, "inning_topbot": "Bot", "at_bat_number": 4,
         "post_home_score": home_score, "post_away_score": away_score},
    ]


def test_derive_historical_team_schedule_widens_to_one_row_per_team():
    rows = _statcast_game(100, pd.Timestamp("2026-06-01"), "NYY", "BOS", 101, 201, 5, 2)
    result = dfs_backtest.derive_historical_team_schedule(pd.DataFrame(rows))

    assert len(result) == 2
    nyy = result[result["team"] == "NYY"].iloc[0]
    bos = result[result["team"] == "BOS"].iloc[0]

    assert nyy["opponent"] == "BOS"
    assert nyy["is_home"] == True  # noqa: E712
    assert nyy["probable_pitcher_key_mlbam"] == 101
    assert bos["opponent"] == "NYY"
    assert bos["is_home"] == False  # noqa: E712
    assert bos["probable_pitcher_key_mlbam"] == 201


def _batter_events(batter, events, date="2026-06-01"):
    return [{"game_date": pd.Timestamp(date), "batter": batter, "events": e} for e in events]


def test_compute_actual_hitter_dk_points_hit_types_only():
    rows = _batter_events(1, ["single", "double", "triple", "home_run", "walk", "field_out"])
    result = dfs_backtest.compute_actual_hitter_dk_points(pd.DataFrame(rows)).set_index("key_mlbam")

    expected = (
        config.DFS_DK_HITTER_SINGLE_POINTS + config.DFS_DK_HITTER_DOUBLE_POINTS
        + config.DFS_DK_HITTER_TRIPLE_POINTS + config.DFS_DK_HITTER_HR_POINTS
    )
    # walk/field_out contribute 0 - not modeled, confirming they're excluded.
    assert result.loc[1, "Actual_DK_Points_Modeled"] == expected


def _pitcher_events(pitcher, events, date="2026-06-01"):
    return [{"game_date": pd.Timestamp(date), "pitcher": pitcher, "events": e} for e in events]


def test_compute_actual_pitcher_dk_points_exact_arithmetic():
    # 2 K (2 outs), 1 field_out (1 out) = 3 outs = 1 IP. 1 BB, 1 single, 1 HR.
    rows = _pitcher_events(99, ["strikeout", "strikeout", "field_out", "walk", "single", "home_run"])
    result = dfs_backtest.compute_actual_pitcher_dk_points(pd.DataFrame(rows)).set_index("key_mlbam")

    assert result.loc[99, "Actual_IP"] == pytest.approx(1.0)
    assert result.loc[99, "Actual_K"] == 2
    assert result.loc[99, "Actual_BB"] == 1
    assert result.loc[99, "Actual_H"] == 2  # single + home_run both count as hits

    fip = (13 * 1 + 3 * 1 - 2 * 2) / 1.0 + config.FIP_CONSTANT
    er = max(fip * 1.0 / 9, 0)
    expected_points = (
        1.0 * config.DFS_DK_PITCHER_IP_POINTS
        + 2 * config.DFS_DK_PITCHER_K_POINTS
        + 1 * config.DFS_DK_PITCHER_BB_POINTS
        + 2 * config.DFS_DK_PITCHER_H_POINTS
        + er * config.DFS_DK_PITCHER_ER_POINTS
    )
    assert result.loc[99, "Actual_DK_Points_Modeled"] == pytest.approx(expected_points)


def test_compute_actual_pitcher_dk_points_zero_ip_does_not_divide_by_zero():
    # 0 outs recorded, and no BB/H/K/HR either - isolates the zero-IP FIP
    # division from any other scoring component (a walk/hit allowed with 0
    # outs would correctly still score negative via the BB/H penalty terms,
    # which is not what this test is checking).
    rows = _pitcher_events(99, ["hit_by_pitch"])  # not modeled as a hit/BB/K, 0 outs
    result = dfs_backtest.compute_actual_pitcher_dk_points(pd.DataFrame(rows)).set_index("key_mlbam")

    assert result.loc[99, "Actual_IP"] == 0
    assert result.loc[99, "Actual_DK_Points_Modeled"] == 0
