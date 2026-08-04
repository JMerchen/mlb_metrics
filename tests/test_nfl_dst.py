import pandas as pd
import pytest

from mlb_metrics import config, nfl_dst


def _team_stats_row(team, season, week, def_sacks=0, def_interceptions=0, fumble_recovery_opp=0, def_safeties=0, fg_blocked=0, pt_blocked=0, pat_blocked=0, def_tds=0, fumble_recovery_tds=0, special_teams_tds=0):
    return {
        "team": team,
        "season": season,
        "week": week,
        "def_sacks": def_sacks,
        "def_interceptions": def_interceptions,
        "fumble_recovery_opp": fumble_recovery_opp,
        "def_safeties": def_safeties,
        "fg_blocked": fg_blocked,
        "pt_blocked": pt_blocked,
        "pat_blocked": pat_blocked,
        "def_tds": def_tds,
        "fumble_recovery_tds": fumble_recovery_tds,
        "special_teams_tds": special_teams_tds,
    }


def test_compute_dst_box_score_rates_sums_td_columns_without_double_counting():
    # A single week with one of EACH TD category - real nflreadpy data
    # confirms these three columns don't overlap (see module docstring),
    # so the total must be the sum of all three, not just one.
    rows = [_team_stats_row("SEA", 2025, 1, def_tds=1, fumble_recovery_tds=1, special_teams_tds=1)]
    team_stats_df = pd.DataFrame(rows)

    result = nfl_dst.compute_dst_box_score_rates(team_stats_df).set_index("team")

    assert result.loc["SEA", "defensive_tds_per_game"] == pytest.approx(3.0)


def test_compute_dst_box_score_rates_blocked_kicks_sums_fg_pt_pat():
    rows = [_team_stats_row("SEA", 2025, 1, fg_blocked=1, pt_blocked=1, pat_blocked=1)]
    team_stats_df = pd.DataFrame(rows)

    result = nfl_dst.compute_dst_box_score_rates(team_stats_df).set_index("team")

    assert result.loc["SEA", "blocked_kicks_per_game"] == pytest.approx(3.0)


def test_nfl_defense_windows_used_by_dst_sum_to_one():
    assert sum(weight for _, weight in config.NFL_DEFENSE_WINDOWS) == pytest.approx(1.0)


def test_compute_points_allowed_derives_from_opponent_score_not_own():
    schedule_df = pd.DataFrame([{"home_team": "SF", "away_team": "SEA", "home_score": 24, "away_score": 17, "season": 2025, "week": 1}])

    result = nfl_dst.compute_points_allowed(schedule_df).set_index("team")

    # SF's DEFENSE allowed what SEA (the opponent) scored, not SF's own score.
    assert result.loc["SF", "points_allowed"] == 17
    assert result.loc["SEA", "points_allowed"] == 24


def test_compute_points_allowed_bonus_maps_buckets_correctly():
    points_allowed = pd.Series([0.0, 6.0, 13.0, 20.0, 27.0, 34.0, 100.0])
    result = nfl_dst.compute_points_allowed_bonus(points_allowed)

    assert result.tolist() == [10.0, 7.0, 4.0, 1.0, 0.0, -1.0, -4.0]


def test_compute_dst_dk_points_combines_box_score_and_points_allowed_bonus():
    box_score_rates = pd.DataFrame(
        [
            {
                "team": "SEA", "games": 10,
                "sacks_per_game": 2.0, "interceptions_per_game": 1.0, "fumbles_recovered_per_game": 0.5,
                "safeties_per_game": 0.0, "blocked_kicks_per_game": 0.0, "defensive_tds_per_game": 0.1,
            }
        ]
    )
    points_allowed_rolling = pd.DataFrame([{"team": "SEA", "games": 10, "points_allowed_per_game": 6.0}])

    result = nfl_dst.compute_dst_dk_points(box_score_rates, points_allowed_rolling).set_index("team")

    expected = (
        2.0 * config.NFL_DK_DST_SACK_POINTS
        + 1.0 * config.NFL_DK_DST_INT_POINTS
        + 0.5 * config.NFL_DK_DST_FUMBLE_REC_POINTS
        + 0.1 * config.NFL_DK_DST_TD_POINTS
        + 7.0  # points_allowed_per_game=6.0 falls in the 1-6 bucket = +7
    )
    assert result.loc["SEA", "DK_Points_DST"] == pytest.approx(expected)


def test_compute_dst_dk_points_missing_points_allowed_falls_back_to_neutral_bucket():
    box_score_rates = pd.DataFrame(
        [{"team": "SEA", "games": 1, "sacks_per_game": 0.0, "interceptions_per_game": 0.0, "fumbles_recovered_per_game": 0.0, "safeties_per_game": 0.0, "blocked_kicks_per_game": 0.0, "defensive_tds_per_game": 0.0}]
    )
    points_allowed_rolling = pd.DataFrame(columns=["team", "games", "points_allowed_per_game"])

    result = nfl_dst.compute_dst_dk_points(box_score_rates, points_allowed_rolling).set_index("team")

    assert result.loc["SEA", "Points_Allowed_Bonus"] == 0.0
