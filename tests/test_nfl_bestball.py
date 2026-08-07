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


def _snap_row(pfr_player_id, season, week, offense_pct, game_type="REG"):
    return {"pfr_player_id": pfr_player_id, "season": season, "week": week, "game_type": game_type, "offense_pct": offense_pct}


def _roster_row(gsis_id, pfr_id, season):
    return {"gsis_id": gsis_id, "pfr_id": pfr_id, "season": season}


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


def test_compute_player_snap_share_averages_real_offense_pct_across_weeks():
    snaps = pd.DataFrame([
        _snap_row("MahoPa00", 2025, 1, 0.95),
        _snap_row("MahoPa00", 2025, 2, 0.85),
    ])
    rosters = pd.DataFrame([_roster_row("00-0033873", "MahoPa00", 2025)])

    result = nfl_bestball.compute_player_snap_share(snaps, rosters, 2025).set_index("player_id")

    assert result.loc["00-0033873", "avg_offense_pct"] == pytest.approx(0.9)


def test_compute_player_snap_share_ignores_postseason():
    snaps = pd.DataFrame([
        _snap_row("MahoPa00", 2025, 1, 1.0, game_type="REG"),
        _snap_row("MahoPa00", 2025, 19, 0.1, game_type="WC"),  # a real low-snap mop-up playoff game
    ])
    rosters = pd.DataFrame([_roster_row("00-0033873", "MahoPa00", 2025)])

    result = nfl_bestball.compute_player_snap_share(snaps, rosters, 2025).set_index("player_id")

    assert result.loc["00-0033873", "avg_offense_pct"] == pytest.approx(1.0)


def test_compute_player_snap_share_excludes_player_missing_from_real_crosswalk():
    # A real snap-count row with no matching pfr_id on any real roster row
    # that season (e.g. the ~0.3% real gap confirmed against 2025 data) is
    # simply absent from the result, not given a fabricated share.
    snaps = pd.DataFrame([_snap_row("NoCrosswalk00", 2025, 1, 0.9)])
    rosters = pd.DataFrame([_roster_row("00-0099999", "SomeoneElse00", 2025)])

    result = nfl_bestball.compute_player_snap_share(snaps, rosters, 2025)

    assert result.empty


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


def test_build_bestball_rankings_snap_share_column_added_only_when_given():
    weekly = pd.DataFrame([_qb_week("qb1", 2025, 1, team="NYJ")])
    schedules = pd.DataFrame([_schedule_row(2025, 1, "NYJ", "BUF")])

    without_snaps = nfl_bestball.build_bestball_rankings(weekly, schedules, 2025)
    assert "avg_offense_pct" not in without_snaps.columns

    snap_counts = pd.DataFrame([_snap_row("QbOnePfr", 2025, 1, 0.88)])
    rosters = pd.DataFrame([_roster_row("qb1", "QbOnePfr", 2025)])
    with_snaps = nfl_bestball.build_bestball_rankings(
        weekly, schedules, 2025, snap_counts_df=snap_counts, rosters_df=rosters
    )
    assert with_snaps.set_index("player_id").loc["qb1", "avg_offense_pct"] == pytest.approx(0.88)


def _rankings_row(player_id, position, dk_points_total, avg_offense_pct=0.6):
    # avg_offense_pct defaults above config.NFL_BESTBALL_SCARCITY_MIN_SNAP_SHARE
    # (0.5) so a test can omit it entirely when only exercising unrelated
    # behavior; tests exercising the qualifier itself override it directly.
    return {"player_id": player_id, "position": position, "dk_points_total": dk_points_total, "avg_offense_pct": avg_offense_pct}


def test_compute_position_scarcity_counts_total_vs_qualified_players():
    # 3 real WRs total, but only 2 clear the snap-share threshold.
    rankings = pd.DataFrame([
        _rankings_row("wr1", "WR", 200, avg_offense_pct=0.9),
        _rankings_row("wr2", "WR", 150, avg_offense_pct=0.55),
        _rankings_row("wr3", "WR", 30, avg_offense_pct=0.02),  # below threshold - counted in total, not qualified
    ])

    result = nfl_bestball.compute_position_scarcity(rankings, min_snap_share=0.5).set_index("position")

    assert result.loc["WR", "total_players"] == 3
    assert result.loc["WR", "qualified_players"] == 2


def test_compute_position_scarcity_excludes_players_missing_snap_share_entirely():
    # A real NaN avg_offense_pct (no snap-count crosswalk match) never
    # satisfies >=, so it must be excluded, not treated as a fabricated 0.
    rankings = pd.DataFrame([
        _rankings_row("wr1", "WR", 200, avg_offense_pct=0.9),
        _rankings_row("wr2", "WR", 150, avg_offense_pct=float("nan")),
    ])

    result = nfl_bestball.compute_position_scarcity(rankings, min_snap_share=0.5).set_index("position")

    assert result.loc["WR", "total_players"] == 2
    assert result.loc["WR", "qualified_players"] == 1


def test_compute_position_scarcity_handles_missing_avg_offense_pct_column():
    # rankings_df built without snap_counts_df/rosters_df has no
    # avg_offense_pct column at all - must not crash (KeyError), just
    # report "nothing real to qualify."
    rankings = pd.DataFrame([{"player_id": "wr1", "position": "WR", "dk_points_total": 200}])

    result = nfl_bestball.compute_position_scarcity(rankings, min_snap_share=0.5).set_index("position")

    assert result.loc["WR", "total_players"] == 1
    assert result.loc["WR", "qualified_players"] == 0


def test_compute_position_scarcity_mean_and_std_match_hand_computation():
    # Qualified points: 100, 200, 300 -> mean 200, population std (ddof=0)
    # is sqrt(((100)^2+(0)^2+(100)^2)/3) = sqrt(20000/3).
    rankings = pd.DataFrame([
        _rankings_row("rb1", "RB", 100),
        _rankings_row("rb2", "RB", 200),
        _rankings_row("rb3", "RB", 300),
    ])

    result = nfl_bestball.compute_position_scarcity(rankings, min_snap_share=0.5).set_index("position")

    assert result.loc["RB", "mean_dk_points"] == pytest.approx(200.0)
    assert result.loc["RB", "std_dk_points"] == pytest.approx((20000 / 3) ** 0.5)


def test_compute_position_scarcity_buckets_players_into_quarter_sd_subbuckets():
    # A clean, no-outlier population (no value clears the IQR fence) with
    # exact z-scores of -1.5, -0.5, 0, 0.5, 1.5 by construction, hitting
    # both the outer bands and the four within-1sd quarter-SD subbuckets.
    values = [0, 100, 150, 200, 300]  # mean=150, population std=100
    rankings = pd.DataFrame([_rankings_row(f"qb{i}", "QB", v) for i, v in enumerate(values)])
    mean = sum(values) / len(values)
    std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
    assert mean == pytest.approx(150.0)
    assert std == pytest.approx(100.0)
    # z-scores: 0 -> -1.5, 100 -> -0.5, 150 -> 0, 200 -> 0.5, 300 -> 1.5

    result = nfl_bestball.compute_position_scarcity(rankings, min_snap_share=0.5).set_index("position")

    # pd.cut bins are right-inclusive - (-1,-0.5] contains -0.5, (-0.5,0]
    # contains 0, (0,0.5] contains 0.5, so an exact-boundary z-score lands
    # in the bucket below it, not above.
    row = result.loc["QB"]
    assert row["outliers_removed"] == 0
    assert row["-2sd_to_-1sd"] == 1  # z=-1.5
    assert row["-1sd_to_-0.5sd"] == 1  # z=-0.5
    assert row["-0.5sd_to_0sd"] == 1  # z=0
    assert row["0sd_to_0.5sd"] == 1  # z=0.5
    assert row["1sd_to_2sd"] == 1  # z=1.5
    assert sum(row[label] for label in nfl_bestball.SCARCITY_BUCKET_LABELS) == 5


def test_compute_position_scarcity_excludes_real_outliers_from_mean_std_but_buckets_them():
    # 8 tightly-clustered "core" values plus one extreme outlier that a
    # standard 1.5x IQR fence flags. The outlier must not pull the
    # mean/std, but must still be counted somewhere in the bell curve.
    core_values = [95, 100, 100, 105, 100, 95, 105, 100]  # mean 100, tight spread
    rankings = pd.DataFrame(
        [_rankings_row(f"wr{i}", "WR", v) for i, v in enumerate(core_values)]
        + [_rankings_row("wr_outlier", "WR", 1000)]  # far outside any real IQR fence
    )

    result = nfl_bestball.compute_position_scarcity(rankings, min_snap_share=0.5).set_index("position")
    row = result.loc["WR"]

    assert row["qualified_players"] == 9
    assert row["outliers_removed"] == 1
    assert row["mean_dk_points"] == pytest.approx(sum(core_values) / len(core_values))
    assert row["mean_dk_points"] < 200  # nowhere near being pulled toward 1000
    assert sum(row[label] for label in nfl_bestball.SCARCITY_BUCKET_LABELS) == 9  # all 9, outlier included
    assert row["above_3sd"] == 1  # the real outlier lands in the extreme band


def test_compute_position_scarcity_handles_fewer_than_two_qualified_players():
    rankings = pd.DataFrame([_rankings_row("te1", "TE", 100)])

    result = nfl_bestball.compute_position_scarcity(rankings, min_snap_share=0.5).set_index("position")

    row = result.loc["TE"]
    assert row["qualified_players"] == 1
    assert pd.isna(row["mean_dk_points"]) is False  # a single qualified player still has a real mean
    assert pd.isna(row["std_dk_points"])  # but no real std with only one data point
    assert pd.isna(row["coefficient_of_variation"])
    assert sum(row[label] for label in nfl_bestball.SCARCITY_BUCKET_LABELS) == 0


def test_compute_position_scarcity_handles_zero_variance_without_crashing():
    # Every qualified player tied on points - std is exactly 0, so a
    # z-score is undefined; must not divide by zero.
    rankings = pd.DataFrame([
        _rankings_row("wr1", "WR", 100),
        _rankings_row("wr2", "WR", 100),
    ])

    result = nfl_bestball.compute_position_scarcity(rankings, min_snap_share=0.5).set_index("position")

    row = result.loc["WR"]
    assert row["mean_dk_points"] == pytest.approx(100.0)
    assert row["std_dk_points"] == pytest.approx(0.0)
    assert pd.isna(row["coefficient_of_variation"])
    assert sum(row[label] for label in nfl_bestball.SCARCITY_BUCKET_LABELS) == 0


def test_compute_draft_strategy_takeaways_ranks_by_relative_dispersion():
    # QB is flat (CV low), WR is dispersed (CV high) - QB should rank
    # below-median (flatter/deeper), WR above-median (scarcer).
    scarcity = pd.DataFrame([
        {"position": "QB", "mean_dk_points": 200.0, "std_dk_points": 20.0, "coefficient_of_variation": 0.1},
        {"position": "RB", "mean_dk_points": 150.0, "std_dk_points": 60.0, "coefficient_of_variation": 0.4},
        {"position": "WR", "mean_dk_points": 100.0, "std_dk_points": 80.0, "coefficient_of_variation": 0.8},
        {"position": "TE", "mean_dk_points": 80.0, "std_dk_points": 24.0, "coefficient_of_variation": 0.3},
    ])

    result = nfl_bestball.compute_draft_strategy_takeaways(scarcity).set_index("position")

    assert result.loc["WR", "dispersion_rank"] == 1  # most dispersed
    assert result.loc["QB", "dispersion_rank"] == 4  # least dispersed
    assert "flatter" in result.loc["QB", "takeaway"].lower()
    assert "scarcer" in result.loc["WR", "takeaway"].lower() or "top-heavy" in result.loc["WR", "takeaway"].lower()


def test_compute_draft_strategy_takeaways_handles_missing_coefficient_of_variation():
    scarcity = pd.DataFrame([
        {"position": "QB", "mean_dk_points": 200.0, "std_dk_points": 20.0, "coefficient_of_variation": 0.1},
        {"position": "RB", "mean_dk_points": 150.0, "std_dk_points": 60.0, "coefficient_of_variation": 0.4},
        {"position": "TE", "mean_dk_points": float("nan"), "std_dk_points": float("nan"), "coefficient_of_variation": float("nan")},
    ])

    result = nfl_bestball.compute_draft_strategy_takeaways(scarcity).set_index("position")

    assert pd.isna(result.loc["TE", "dispersion_rank"])
    assert "not enough" in result.loc["TE", "takeaway"].lower()
    assert result.loc["QB", "dispersion_rank"] == 2  # still comparable against RB
