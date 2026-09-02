import pandas as pd
import pytest

from mlb_metrics import config, nfl_dfs_backtest


def _qb_week(player_id, season, week, passing_yards=200, passing_tds=1, ints=0, rush_yards=10, rush_tds=0, two_pt=0):
    return {
        "player_id": player_id, "position": "QB", "season": season, "week": week,
        "game_id": f"{season}_{week:02d}_{player_id}",
        "passing_yards": passing_yards, "passing_tds": passing_tds, "passing_interceptions": ints,
        "rushing_yards": rush_yards, "rushing_tds": rush_tds, "passing_2pt_conversions": two_pt,
        "carries": 3, "attempts": 30, "completions": 20,
        # QB rows get filtered out of nfl_rush_rec.compute_skill_rolling_stats,
        # but the columns it touches must still exist on the frame even
        # when 0 rows survive the filter - pandas requires the column,
        # not just a nonzero row count, to do column arithmetic.
        "targets": 0, "receptions": 0, "receiving_yards": 0, "receiving_tds": 0,
        "rushing_fumbles_lost": 0, "receiving_fumbles_lost": 0, "rushing_2pt_conversions": 0, "receiving_2pt_conversions": 0,
        "passing_epa": 0.0,
    }


def _skill_week(player_id, position, season, week, rush_yards=0, rec_yards=0, receptions=0, rush_tds=0, rec_tds=0, rush_fum_lost=0, rec_fum_lost=0, rush_2pt=0, rec_2pt=0):
    return {
        "player_id": player_id, "position": position, "season": season, "week": week,
        "game_id": f"{season}_{week:02d}_{player_id}",
        "carries": 10, "rushing_yards": rush_yards, "rushing_tds": rush_tds,
        "targets": receptions + 1, "receptions": receptions, "receiving_yards": rec_yards, "receiving_tds": rec_tds,
        "rushing_fumbles_lost": rush_fum_lost, "receiving_fumbles_lost": rec_fum_lost,
        "rushing_2pt_conversions": rush_2pt, "receiving_2pt_conversions": rec_2pt,
    }


def test_compute_actual_qb_dk_points_matches_hand_computed_formula():
    week_df = pd.DataFrame([_qb_week("qb1", 2025, 1, passing_yards=320, passing_tds=3, ints=1, rush_yards=25, rush_tds=1, two_pt=1)])

    result = nfl_dfs_backtest.compute_actual_qb_dk_points(week_df).set_index("player_id")

    expected = (
        320 * config.NFL_DK_PASS_YARD_POINTS
        + 3 * config.NFL_DK_PASS_TD_POINTS
        + 1 * config.NFL_DK_INTERCEPTION_POINTS
        + 25 * config.NFL_DK_RUSH_YARD_POINTS
        + 1 * config.NFL_DK_RUSH_TD_POINTS
        + 1 * config.NFL_DK_300_PASS_YARD_BONUS  # real 320 >= 300, discrete not EV
        + 1 * config.NFL_DK_2PT_POINTS
    )
    assert result.loc["qb1", "Actual_DK_Points_QB"] == pytest.approx(expected)


def test_compute_actual_qb_dk_points_below_300_gets_no_bonus():
    week_df = pd.DataFrame([_qb_week("qb1", 2025, 1, passing_yards=299)])
    result = nfl_dfs_backtest.compute_actual_qb_dk_points(week_df).set_index("player_id")
    # _qb_week defaults rush_yards=10 -> +1.0 rushing point included below.
    expected = 299 * config.NFL_DK_PASS_YARD_POINTS + 1 * config.NFL_DK_PASS_TD_POINTS + 10 * config.NFL_DK_RUSH_YARD_POINTS
    assert result.loc["qb1", "Actual_DK_Points_QB"] == pytest.approx(expected)


def test_compute_actual_skill_dk_points_matches_hand_computed_formula():
    week_df = pd.DataFrame([_skill_week("rb1", "RB", 2025, 1, rush_yards=110, rec_yards=40, receptions=3, rush_tds=1, rush_fum_lost=1)])

    result = nfl_dfs_backtest.compute_actual_skill_dk_points(week_df).set_index("player_id")

    expected = (
        110 * config.NFL_DK_RUSH_YARD_POINTS
        + 1 * config.NFL_DK_RUSH_TD_POINTS
        + 40 * config.NFL_DK_RECEIVING_YARD_POINTS
        + 3 * config.NFL_DK_RECEPTION_POINTS
        + 1 * config.NFL_DK_FUMBLE_LOST_POINTS
        + 1 * config.NFL_DK_100_YARD_BONUS  # real 110 >= 100 rush
    )
    assert result.loc["rb1", "Actual_DK_Points_Skill"] == pytest.approx(expected)


def test_compute_actual_dst_dk_points_uses_real_single_game_points_allowed():
    team_stats_df = pd.DataFrame(
        [
            {
                "team": "SEA", "def_sacks": 3, "def_interceptions": 1, "fumble_recovery_opp": 1,
                "def_safeties": 0, "fg_blocked": 0, "pt_blocked": 0, "pat_blocked": 0,
                "def_tds": 0, "fumble_recovery_tds": 0, "special_teams_tds": 0,
            }
        ]
    )
    schedule_df = pd.DataFrame([{"home_team": "SF", "away_team": "SEA", "home_score": 24, "away_score": 20, "season": 2025, "week": 1}])

    result = nfl_dfs_backtest.compute_actual_dst_dk_points(team_stats_df, schedule_df).set_index("team")

    # SEA allowed 24 (SF's score) - real single-game value, falls in the 21-27 bucket = 0 bonus.
    expected = 3 * config.NFL_DK_DST_SACK_POINTS + 1 * config.NFL_DK_DST_INT_POINTS + 1 * config.NFL_DK_DST_FUMBLE_REC_POINTS + 0
    assert result.loc["SEA", "Actual_DK_Points_DST"] == pytest.approx(expected)


def test_backtest_nfl_dfs_projections_is_no_lookahead():
    # Week 1: QB has no prior history at all - must be skipped entirely
    # (no projection possible with zero history). Week 2's projection for
    # that QB must be built ONLY from week 1's real stats, never week 2's
    # own (which would be lookahead / data leakage).
    weekly_rows = [
        _qb_week("qb1", 2025, 1, passing_yards=200),
        _qb_week("qb1", 2025, 2, passing_yards=999),  # week 2's real outcome - must NOT leak into its own projection
    ]
    weekly_df = pd.DataFrame(weekly_rows)
    team_stats_df = pd.DataFrame(columns=["team", "season", "week", "def_sacks", "def_interceptions", "fumble_recovery_opp", "def_safeties", "fg_blocked", "pt_blocked", "pat_blocked", "def_tds", "fumble_recovery_tds", "special_teams_tds"])
    schedules_df = pd.DataFrame(columns=["home_team", "away_team", "home_score", "away_score", "season", "week"])

    result = nfl_dfs_backtest.backtest_nfl_dfs_projections(weekly_df, team_stats_df, schedules_df)

    assert len(result["qb"]) == 1  # only week 2 is scored (week 1 had no prior history)
    row = result["qb"].iloc[0]
    assert row["season"] == 2025
    assert row["week"] == 2
    # Week 2's PROJECTION is built purely from week 1's 200 yards, not week 2's real 999.
    assert row["passing_yards_per_game"] == pytest.approx(200.0)
    # Week 2's ACTUAL correctly reflects the real week-2 box score (999
    # yards + the default rush_yards=10 -> +1.0 rushing point).
    assert row["Actual_DK_Points_QB"] == pytest.approx(
        999 * config.NFL_DK_PASS_YARD_POINTS + 1 * config.NFL_DK_PASS_TD_POINTS
        + 10 * config.NFL_DK_RUSH_YARD_POINTS + config.NFL_DK_300_PASS_YARD_BONUS
    )


def test_backtest_nfl_dfs_projections_weeks_caps_to_most_recent():
    weekly_rows = [_qb_week("qb1", 2025, week, passing_yards=200) for week in range(1, 6)]
    weekly_df = pd.DataFrame(weekly_rows)
    team_stats_df = pd.DataFrame(columns=["team", "season", "week", "def_sacks", "def_interceptions", "fumble_recovery_opp", "def_safeties", "fg_blocked", "pt_blocked", "pat_blocked", "def_tds", "fumble_recovery_tds", "special_teams_tds"])
    schedules_df = pd.DataFrame(columns=["home_team", "away_team", "home_score", "away_score", "season", "week"])

    result = nfl_dfs_backtest.backtest_nfl_dfs_projections(weekly_df, team_stats_df, schedules_df, weeks=2)

    assert sorted(result["qb"]["week"].tolist()) == [4, 5]  # weeks 2/3 skipped by the cap, week 1 has no history anyway
