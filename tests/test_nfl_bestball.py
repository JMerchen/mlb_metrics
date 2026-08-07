import pandas as pd
import pytest

from mlb_metrics import config, nfl_bestball


def _qb_week(player_id, season, week, name="QB One", passing_yards=200, passing_tds=1, ints=0, rush_yards=10, rush_tds=0, two_pt=0, team="NYJ", season_type="REG"):
    return {
        "player_id": player_id, "player_display_name": name, "position": "QB", "team": team,
        "season": season, "week": week, "season_type": season_type,
        "passing_yards": passing_yards, "passing_tds": passing_tds, "passing_interceptions": ints,
        "rushing_yards": rush_yards, "rushing_tds": rush_tds, "passing_2pt_conversions": two_pt,
        "receptions": 0, "receiving_yards": 0, "receiving_tds": 0,
        "rushing_fumbles_lost": 0, "receiving_fumbles_lost": 0, "rushing_2pt_conversions": 0, "receiving_2pt_conversions": 0,
    }


def _skill_week(player_id, position, season, week, name="Skill One", rush_yards=0, rec_yards=0, receptions=0, rush_tds=0, rec_tds=0, team="NYJ", season_type="REG"):
    return {
        "player_id": player_id, "player_display_name": name, "position": position, "team": team,
        "season": season, "week": week, "season_type": season_type,
        "rushing_yards": rush_yards, "rushing_tds": rush_tds,
        "receptions": receptions, "receiving_yards": rec_yards, "receiving_tds": rec_tds,
        "rushing_fumbles_lost": 0, "receiving_fumbles_lost": 0, "rushing_2pt_conversions": 0, "receiving_2pt_conversions": 0,
        "passing_yards": 0, "passing_tds": 0, "passing_interceptions": 0, "passing_2pt_conversions": 0,
    }


def _schedule_row(season, week, home, away, game_type="REG"):
    return {"season": season, "week": week, "game_type": game_type, "home_team": home, "away_team": away}


def test_compute_player_games_played_counts_real_weeks_and_missed_games():
    # Player played weeks 1-2 out of a real 3-game team schedule (weeks
    # 1-3) - a real week absent from weekly_df means they didn't play it.
    weekly = pd.DataFrame([
        _qb_week("qb1", 2025, 1, team="NYJ"),
        _qb_week("qb1", 2025, 2, team="NYJ"),
    ])
    schedules = pd.DataFrame([
        _schedule_row(2025, 1, "NYJ", "BUF"),
        _schedule_row(2025, 2, "MIA", "NYJ"),
        _schedule_row(2025, 3, "NYJ", "NE"),
    ])

    result = nfl_bestball.compute_player_games_played(weekly, schedules, 2025).set_index("player_id")

    assert result.loc["qb1", "games_played"] == 2
    assert result.loc["qb1", "possible_games"] == 3
    assert result.loc["qb1", "games_missed"] == 1
    assert result.loc["qb1", "team"] == "NYJ"


def test_compute_player_games_played_derives_possible_games_from_real_schedule_not_hardcoded():
    # A 16-game-season team schedule must yield possible_games=16, not a
    # hardcoded 17 - confirms the denominator is always read from real
    # schedules_df, never assumed.
    weekly = pd.DataFrame([_qb_week("qb1", 2019, 1, team="NYJ")])
    schedules = pd.DataFrame([_schedule_row(2019, w, "NYJ" if w % 2 else "BUF", "BUF" if w % 2 else "NYJ") for w in range(1, 17)])

    result = nfl_bestball.compute_player_games_played(weekly, schedules, 2019).set_index("player_id")

    assert result.loc["qb1", "possible_games"] == 16


def test_compute_player_games_played_ignores_postseason():
    weekly = pd.DataFrame([
        _qb_week("qb1", 2025, 1, team="NYJ", season_type="REG"),
        _qb_week("qb1", 2025, 19, team="NYJ", season_type="POST"),
    ])
    schedules = pd.DataFrame([
        _schedule_row(2025, 1, "NYJ", "BUF", game_type="REG"),
        _schedule_row(2025, 19, "NYJ", "BUF", game_type="WC"),
    ])

    result = nfl_bestball.compute_player_games_played(weekly, schedules, 2025).set_index("player_id")

    assert result.loc["qb1", "games_played"] == 1
    assert result.loc["qb1", "possible_games"] == 1


def test_compute_season_realized_dk_points_qb_sums_real_weeks():
    weekly = pd.DataFrame([
        _qb_week("qb1", 2025, 1, passing_yards=320, passing_tds=3, ints=1, rush_yards=25, rush_tds=1, two_pt=1),
        _qb_week("qb1", 2025, 2, passing_yards=200, passing_tds=1, ints=0, rush_yards=10, rush_tds=0),
    ])

    result = nfl_bestball.compute_season_realized_dk_points(weekly, "QB", 2025).set_index("player_id")

    week1 = (
        320 * config.NFL_DK_PASS_YARD_POINTS + 3 * config.NFL_DK_PASS_TD_POINTS
        + 1 * config.NFL_DK_INTERCEPTION_POINTS + 25 * config.NFL_DK_RUSH_YARD_POINTS
        + 1 * config.NFL_DK_RUSH_TD_POINTS + 1 * config.NFL_DK_300_PASS_YARD_BONUS + 1 * config.NFL_DK_2PT_POINTS
    )
    week2 = 200 * config.NFL_DK_PASS_YARD_POINTS + 1 * config.NFL_DK_PASS_TD_POINTS + 10 * config.NFL_DK_RUSH_YARD_POINTS
    assert result.loc["qb1", "dk_points_total"] == pytest.approx(week1 + week2)


def test_compute_season_realized_dk_points_skill_sums_real_weeks():
    weekly = pd.DataFrame([
        _skill_week("rb1", "RB", 2025, 1, rush_yards=110, rec_yards=40, receptions=3, rush_tds=1),
        _skill_week("rb1", "RB", 2025, 2, rush_yards=50, rec_yards=0, receptions=0),
    ])

    result = nfl_bestball.compute_season_realized_dk_points(weekly, "SKILL", 2025).set_index("player_id")

    week1 = (
        110 * config.NFL_DK_RUSH_YARD_POINTS + 1 * config.NFL_DK_RUSH_TD_POINTS
        + 40 * config.NFL_DK_RECEIVING_YARD_POINTS + 3 * config.NFL_DK_RECEPTION_POINTS
        + 1 * config.NFL_DK_100_YARD_BONUS
    )
    week2 = 50 * config.NFL_DK_RUSH_YARD_POINTS
    assert result.loc["rb1", "dk_points_total"] == pytest.approx(week1 + week2)


def test_compute_season_realized_dk_points_rejects_unknown_position_group():
    weekly = pd.DataFrame([_qb_week("qb1", 2025, 1)])
    with pytest.raises(ValueError):
        nfl_bestball.compute_season_realized_dk_points(weekly, "K", 2025)


def test_build_bestball_rankings_ranks_by_total_points_descending():
    weekly = pd.DataFrame([
        _qb_week("qb1", 2025, 1, name="Big Arm", passing_yards=350, passing_tds=4, team="NYJ"),
        _skill_week("rb1", "RB", 2025, 1, name="Workhorse", rush_yards=60, rush_tds=1, team="NYJ"),
        _skill_week("rb1", "RB", 2025, 2, name="Workhorse", rush_yards=40, team="NYJ"),
    ])
    schedules = pd.DataFrame([
        _schedule_row(2025, 1, "NYJ", "BUF"),
        _schedule_row(2025, 2, "MIA", "NYJ"),
    ])

    result = nfl_bestball.build_bestball_rankings(weekly, schedules, 2025)

    assert list(result["player_id"]) == ["qb1", "rb1"]  # qb1's single big week outscores rb1's two games
    rb1 = result.set_index("player_id").loc["rb1"]
    assert rb1["games_played"] == 2
    assert rb1["dk_points_per_game"] == pytest.approx(rb1["dk_points_total"] / 2)


def test_build_bestball_rankings_excludes_non_qb_skill_positions():
    weekly = pd.DataFrame([
        _qb_week("qb1", 2025, 1, team="NYJ"),
        {**_skill_week("lb1", "RB", 2025, 1, team="NYJ"), "position": "LB"},
    ])
    schedules = pd.DataFrame([_schedule_row(2025, 1, "NYJ", "BUF")])

    result = nfl_bestball.build_bestball_rankings(weekly, schedules, 2025)

    assert "lb1" not in set(result["player_id"])


def test_build_bestball_rankings_prior_season_column_added_only_when_given():
    weekly = pd.DataFrame([_qb_week("qb1", 2025, 1, team="NYJ")])
    schedules = pd.DataFrame([_schedule_row(2025, 1, "NYJ", "BUF")])

    without_prior = nfl_bestball.build_bestball_rankings(weekly, schedules, 2025)
    assert "games_missed_prior_season" not in without_prior.columns

    prior_weekly = pd.DataFrame([_qb_week("qb1", 2024, 1, team="NYJ")])
    prior_schedules = pd.DataFrame([
        _schedule_row(2024, 1, "NYJ", "BUF"),
        _schedule_row(2024, 2, "NYJ", "NE"),  # a real second game qb1 missed in 2024
    ])
    with_prior = nfl_bestball.build_bestball_rankings(
        weekly, schedules, 2025, prior_season=2024, prior_weekly_df=prior_weekly, prior_schedules_df=prior_schedules
    )
    assert with_prior.set_index("player_id").loc["qb1", "games_missed_prior_season"] == 1
