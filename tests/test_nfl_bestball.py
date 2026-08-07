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


def _rankings_row(player_id, position, games_played, dk_points_total):
    return {"player_id": player_id, "position": position, "games_played": games_played, "dk_points_total": dk_points_total}


def test_compute_position_scarcity_counts_total_vs_qualified_players():
    # 3 real WRs total, but only 2 clear the games threshold.
    rankings = pd.DataFrame([
        _rankings_row("wr1", "WR", 17, 200),
        _rankings_row("wr2", "WR", 15, 150),
        _rankings_row("wr3", "WR", 2, 30),  # below threshold - counted in total, not qualified
    ])

    result = nfl_bestball.compute_position_scarcity(rankings, min_games=8).set_index("position")

    assert result.loc["WR", "total_players"] == 3
    assert result.loc["WR", "qualified_players"] == 2


def test_compute_position_scarcity_mean_and_std_match_hand_computation():
    # Qualified points: 100, 200, 300 -> mean 200, population std (ddof=0)
    # is sqrt(((100)^2+(0)^2+(100)^2)/3) = sqrt(20000/3).
    rankings = pd.DataFrame([
        _rankings_row("rb1", "RB", 17, 100),
        _rankings_row("rb2", "RB", 17, 200),
        _rankings_row("rb3", "RB", 17, 300),
    ])

    result = nfl_bestball.compute_position_scarcity(rankings, min_games=8).set_index("position")

    assert result.loc["RB", "mean_dk_points"] == pytest.approx(200.0)
    assert result.loc["RB", "std_dk_points"] == pytest.approx((20000 / 3) ** 0.5)


def test_compute_position_scarcity_buckets_players_by_standard_deviation():
    # mean=200, std=100 (population) with these 5 values -> z-scores of
    # roughly -1.41, -0.71, 0, 0.71, 1.41, all landing "within_1sd" except
    # the two extremes which fall just short of +/-2sd, still within_1sd
    # too (|z|<1.5). Use a wider, cleaner spread instead so buckets are
    # unambiguous by construction: exact z-scores of -3, -1, 0, 1, 3 via a
    # constructed mean/std.
    values = [0, 100, 150, 200, 300]  # mean=150, population std=100 (by construction below)
    rankings = pd.DataFrame([_rankings_row(f"qb{i}", "QB", 17, v) for i, v in enumerate(values)])
    mean = sum(values) / len(values)
    std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
    assert mean == pytest.approx(150.0)
    assert std == pytest.approx(100.0)
    # z-scores: 0 -> -1.5, 100 -> -0.5, 150 -> 0, 200 -> 0.5, 300 -> 1.5

    result = nfl_bestball.compute_position_scarcity(rankings, min_games=8).set_index("position")

    row = result.loc["QB"]
    assert row["-2sd_to_-1sd"] == 1  # z=-1.5
    assert row["within_1sd"] == 3  # z=-0.5, 0, 0.5
    assert row["1sd_to_2sd"] == 1  # z=1.5
    assert sum(row[label] for label in nfl_bestball.SCARCITY_BUCKET_LABELS) == 5


def test_compute_position_scarcity_handles_fewer_than_two_qualified_players():
    rankings = pd.DataFrame([_rankings_row("te1", "TE", 17, 100)])

    result = nfl_bestball.compute_position_scarcity(rankings, min_games=8).set_index("position")

    row = result.loc["TE"]
    assert row["qualified_players"] == 1
    assert pd.isna(row["mean_dk_points"]) is False  # a single qualified player still has a real mean
    assert pd.isna(row["std_dk_points"])  # but no real std with only one data point
    assert sum(row[label] for label in nfl_bestball.SCARCITY_BUCKET_LABELS) == 0


def test_compute_position_scarcity_handles_zero_variance_without_crashing():
    # Every qualified player tied on points - std is exactly 0, so a
    # z-score is undefined; must not divide by zero.
    rankings = pd.DataFrame([
        _rankings_row("wr1", "WR", 17, 100),
        _rankings_row("wr2", "WR", 17, 100),
    ])

    result = nfl_bestball.compute_position_scarcity(rankings, min_games=8).set_index("position")

    row = result.loc["WR"]
    assert row["mean_dk_points"] == pytest.approx(100.0)
    assert row["std_dk_points"] == pytest.approx(0.0)
    assert sum(row[label] for label in nfl_bestball.SCARCITY_BUCKET_LABELS) == 0
