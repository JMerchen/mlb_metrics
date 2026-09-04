import pandas as pd
import pytest

from mlb_metrics import nfl_game_picks_backtest as bt


def _game(game_id, season, week, home_team, away_team, home_score, away_score,
          home_ml=-150, away_ml=130, game_type="REG"):
    return {
        "game_id": game_id, "season": season, "week": week, "game_type": game_type,
        "home_team": home_team, "away_team": away_team,
        "home_score": home_score, "away_score": away_score,
        "home_qb_id": f"{home_team}_qb", "away_qb_id": f"{away_team}_qb",
        "home_moneyline": home_ml, "away_moneyline": away_ml,
    }


# Each team has its own real EPA quality level, and it varies by week too
# (not just by team) - an all-teams-identical-every-week EPA fixture gives
# offensive_edge/defensive_edge a real zero-std population (see
# test_nfl_team_strength.py's own identical fix), and even a per-team-but-
# week-CONSTANT EPA turns out to give this specific symmetric round-robin
# schedule (every team's week-1/2 opponent pair sums to the exact same
# real total) an identical real mean `defensive_edge` for every team -
# a genuine structural property of a full round-robin with fixed
# opponent-independent EPA, not a bug, but still degenerate for a test
# that needs real cross-team variance to exercise z-normalization.
PASSING_EPA_BY_TEAM = {"A": 2.0, "B": 1.0, "C": 0.5, "D": -0.5}
# Real per-team turnover profile (lost, forced) - same "needs real
# cross-team variance, not identical for every team" reasoning as
# PASSING_EPA_BY_TEAM above (turnover_margin z-normalizes across teams too).
TURNOVERS_BY_TEAM = {"A": (0, 2), "B": (1, 1), "C": (1, 0), "D": (2, 0)}
# Real per-team fixed points-per-drive profile - same cross-team-variance
# reasoning as PASSING_EPA_BY_TEAM/TURNOVERS_BY_TEAM above.
POINTS_BY_TEAM = {"A": 7, "B": 3, "C": 0, "D": 3}


def _ts_row(team, opp, season, week, gid):
    lost, forced = TURNOVERS_BY_TEAM[team]
    return {
        "team": team, "opponent_team": opp, "season": season, "week": week, "game_id": gid,
        "season_type": "REG",
        "passing_epa": PASSING_EPA_BY_TEAM[team] + week * 0.1, "rushing_epa": 0.5, "receiving_epa": 0.0,
        "passing_interceptions": lost, "fumbles_lost_total": 0,
        "def_interceptions": forced, "fumble_recovery_opp": 0,
    }


def _tiny_season(num_weeks=8, season=2025):
    """A real (small, synthetic) round-robin among 4 teams across
    `num_weeks` weeks - reused across replay_season tests. Home team
    always wins 24-17, so real results are deterministic and hand-checkable."""
    weekly_pairings = [
        [("A", "B"), ("C", "D")],
        [("A", "C"), ("B", "D")],
        [("A", "D"), ("B", "C")],
    ]
    sched_rows, ts_rows, weekly_rows, snap_rows, pbp_rows = [], [], [], [], []
    for week in range(1, num_weeks + 1):
        pairings = weekly_pairings[(week - 1) % len(weekly_pairings)]
        for game_num, (home, away) in enumerate(pairings, start=1):
            gid = f"{season}_{week:02d}_{game_num}"
            sched_rows.append(_game(gid, season, week, home, away, 24, 17))
            for team, opp in [(home, away), (away, home)]:
                ts_rows.append(_ts_row(team, opp, season, week, gid))
                weekly_rows.append({
                    "player_id": f"{team}_qb", "position": "QB", "season": season, "week": week,
                    "season_type": "REG", "game_id": gid,
                    "attempts": 30, "completions": 20, "passing_yards": 200, "passing_tds": 1,
                    "passing_interceptions": 0, "carries": 2, "rushing_yards": 5, "rushing_tds": 0,
                    "passing_epa": 1.0,
                })
                snap_rows.append({
                    "game_id": gid, "season": season, "week": week, "game_type": "REG",
                    "team": team, "position": "QB", "pfr_player_id": f"{team}_pfr",
                    "offense_snaps": 60, "offense_pct": 0.95,
                })
                points = POINTS_BY_TEAM[team]
                for play_id, (score, score_post) in enumerate([(0, 0), (0, points)], start=1):
                    pbp_rows.append({
                        "game_id": gid, "season": season, "week": week, "season_type": "REG",
                        "play_id": play_id, "posteam": team, "fixed_drive": 1,
                        "posteam_score": score, "posteam_score_post": score_post,
                    })
    rosters = pd.DataFrame([
        {"season": season, "gsis_id": f"{t}_qb", "pfr_id": f"{t}_pfr"} for t in ["A", "B", "C", "D"]
    ])
    return (
        pd.DataFrame(sched_rows), pd.DataFrame(ts_rows), pd.DataFrame(weekly_rows),
        pd.DataFrame(snap_rows), rosters, pd.DataFrame(pbp_rows),
    )


def _tiny_two_seasons(num_weeks=8):
    """Two consecutive real (small, synthetic) seasons (2024, 2025), same
    shape as `_tiny_season`, concatenated - the minimum real fixture that
    exercises build_multi_season_history/replay_multi_season's own
    cross-season carryover (a single-season fixture, like `_tiny_season`
    on its own, has no real prior season to carry over from at all)."""
    prior = _tiny_season(num_weeks=num_weeks, season=2024)
    current = _tiny_season(num_weeks=num_weeks, season=2025)
    return tuple(
        pd.concat([prior[i], current[i]], ignore_index=True) for i in range(len(prior))
    )


def test_replay_season_default_weeks_excludes_week_1_and_2():
    schedules, team_stats, weekly, snap_counts, rosters, pbp = _tiny_season(num_weeks=8)

    replay = bt.replay_season(schedules, team_stats, weekly, snap_counts, rosters, pbp, season=2025)

    assert replay["week"].min() >= 3
    assert not replay["home_win_probability"].isna().any()


def test_replay_season_no_lookahead_future_weeks_dont_change_past_predictions():
    schedules, team_stats, weekly, snap_counts, rosters, pbp = _tiny_season(num_weeks=4)
    replay_short = bt.replay_season(schedules, team_stats, weekly, snap_counts, rosters, pbp, season=2025, weeks=[3])

    # Extend the same season with 4 more real weeks (different scorelines
    # each week) - week 3's own replayed prediction must be byte-identical,
    # since replay_season restricts history to games strictly before the
    # requested week.
    schedules_long, team_stats_long, weekly_long, snap_counts_long, rosters_long, pbp_long = _tiny_season(num_weeks=8)
    replay_long = bt.replay_season(
        schedules_long, team_stats_long, weekly_long, snap_counts_long, rosters_long, pbp_long,
        season=2025, weeks=[3],
    )

    pd.testing.assert_frame_equal(
        replay_short.reset_index(drop=True), replay_long.reset_index(drop=True)
    )


def test_score_predictions_exact_arithmetic():
    replay = pd.DataFrame([
        {"home_win_probability": 0.8, "home_won": 1.0},  # correct, favored home won
        {"home_win_probability": 0.7, "home_won": 0.0},  # incorrect
        {"home_win_probability": 0.3, "home_won": 0.0},  # correct, favored away won
    ])

    result = bt.score_predictions(replay, "home_win_probability")

    assert result["n"] == 3
    assert result["accuracy"] == pytest.approx(2 / 3)
    # Brier: mean((p - y)^2) = ((0.2)^2 + (0.7)^2 + (0.3)^2) / 3
    expected_brier = ((0.8 - 1.0) ** 2 + (0.7 - 0.0) ** 2 + (0.3 - 0.0) ** 2) / 3
    assert result["brier_score"] == pytest.approx(expected_brier)


def test_score_predictions_excludes_nan_predictions_from_n_and_accuracy():
    # A real degenerate-week NaN prediction (see module docstring) must
    # never be silently compared as if it were a real "predicted away"
    # via `NaN >= 0.5 == False`.
    replay = pd.DataFrame([
        {"home_win_probability": 0.8, "home_won": 1.0},
        {"home_win_probability": float("nan"), "home_won": 0.0},  # would look "correct" if NaN weren't excluded
    ])

    result = bt.score_predictions(replay, "home_win_probability")

    assert result["n"] == 1
    assert result["accuracy"] == pytest.approx(1.0)


def test_score_predictions_no_resolved_games_returns_nan():
    replay = pd.DataFrame([{"home_win_probability": 0.8, "home_won": float("nan")}])

    result = bt.score_predictions(replay, "home_win_probability")

    assert result["n"] == 0
    assert pd.isna(result["accuracy"])
    assert pd.isna(result["brier_score"])


def test_beat_closing_line_rate_counts_strictly_lower_squared_error():
    replay = pd.DataFrame([
        # home won (1.0); model says 0.9 (error .01), market says 0.6 (error .16) - model beats market.
        {"home_win_probability": 0.9, "market_home_win_probability": 0.6, "home_won": 1.0},
        # home won (1.0); model says 0.55 (error .2025), market says 0.8 (error .04) - market beats model.
        {"home_win_probability": 0.55, "market_home_win_probability": 0.8, "home_won": 1.0},
        # exact tie - excluded from both numerator and denominator.
        {"home_win_probability": 0.5, "market_home_win_probability": 0.5, "home_won": 1.0},
    ])

    result = bt.beat_closing_line_rate(replay)

    assert result["n_compared"] == 2
    assert result["rate"] == pytest.approx(0.5)


def test_beat_closing_line_rate_no_market_data_returns_nan():
    replay = pd.DataFrame([{"home_win_probability": 0.8, "market_home_win_probability": float("nan"), "home_won": 1.0}])

    result = bt.beat_closing_line_rate(replay)

    assert result["n_compared"] == 0
    assert pd.isna(result["rate"])


def test_replay_multi_season_excludes_the_first_season_and_includes_weeks_1_and_2():
    # Real follow-up (2026-09-04): unlike replay_season, a real prior
    # season's history means weeks 1-2 of the SECOND season have a real
    # prediction to make - and the first season in the list is never
    # itself replayed (no real prior season exists for it in this data).
    schedules, team_stats, weekly, snap_counts, rosters, pbp = _tiny_two_seasons(num_weeks=4)

    replay = bt.replay_multi_season(
        schedules, team_stats, weekly, snap_counts, rosters, pbp, seasons=[2024, 2025]
    )

    assert set(replay["season"]) == {2025}
    assert replay["week"].min() == 1
    assert not replay["home_win_probability"].isna().any()


def test_build_multi_season_history_and_score_snapshots_match_replay_multi_season():
    # score_multi_season_snapshots re-scoring already-built snapshots must
    # produce byte-identical output to the one-call replay_multi_season
    # wrapper - the whole point of splitting them is a cheap re-score, not
    # a different real result.
    schedules, team_stats, weekly, snap_counts, rosters, pbp = _tiny_two_seasons(num_weeks=4)

    snapshots = bt.build_multi_season_history(schedules, team_stats, weekly, snap_counts, rosters, pbp, seasons=[2024, 2025])
    from_snapshots = bt.score_multi_season_snapshots(snapshots)
    direct = bt.replay_multi_season(schedules, team_stats, weekly, snap_counts, rosters, pbp, seasons=[2024, 2025])

    pd.testing.assert_frame_equal(
        from_snapshots.reset_index(drop=True), direct.reset_index(drop=True)
    )


def test_build_backtest_report_splits_by_train_max_week():
    replay = pd.DataFrame([
        {"week": 3, "home_win_probability": 0.7, "market_home_win_probability": 0.6, "home_won": 1.0},
        {"week": bt.TRAIN_MAX_WEEK, "home_win_probability": 0.6, "market_home_win_probability": 0.55, "home_won": 0.0},
        {"week": bt.TRAIN_MAX_WEEK + 1, "home_win_probability": 0.55, "market_home_win_probability": 0.6, "home_won": 1.0},
        {"week": 18, "home_win_probability": 0.4, "market_home_win_probability": 0.45, "home_won": 0.0},
    ])

    report = bt.build_backtest_report(replay)

    train_rows = report[report["split"] == f"train (wk3-{bt.TRAIN_MAX_WEEK})"]
    test_rows = report[report["split"] == f"test (wk{bt.TRAIN_MAX_WEEK + 1}-18)"]
    assert set(train_rows["source"]) == {"model", "market", "beat_closing_line"}
    assert set(test_rows["source"]) == {"model", "market", "beat_closing_line"}
    # 2 real games in each split.
    assert (train_rows[train_rows["source"] == "model"]["n"] == 2).all()
    assert (test_rows[test_rows["source"] == "model"]["n"] == 2).all()
