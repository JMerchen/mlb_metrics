import numpy as np
import pandas as pd
import pytest

from mlb_metrics import nfl_pipeline


def _game(game_id, season, week, home_team, away_team, home_score=np.nan, away_score=np.nan,
          home_ml=-150, away_ml=130, game_type="REG"):
    # Real nflreadpy schedules use float64 (NaN, not object/None) for
    # home_score/away_score even on a fully-unplayed season - confirmed
    # live against the real 2026 schedule - so this fixture matches that
    # dtype rather than Python's `None`, which would silently upcast an
    # otherwise-int64 column to `object` on concat and break
    # nfl_team_strength.py's own .cumsum() (a real dtype hazard this
    # fixture exists specifically to avoid reintroducing).
    return {
        "game_id": game_id, "season": season, "week": week, "game_type": game_type,
        "home_team": home_team, "away_team": away_team,
        "home_score": float(home_score) if home_score is not None else np.nan,
        "away_score": float(away_score) if away_score is not None else np.nan,
        "home_qb_id": f"{home_team}_qb", "away_qb_id": f"{away_team}_qb",
        "gameday": "2026-09-10",
        "home_moneyline": home_ml, "away_moneyline": away_ml,
    }


def test_determine_predictable_week_returns_earliest_unplayed_week():
    schedules = pd.DataFrame([
        _game("g1", 2026, 1, "A", "B", 24, 17),  # played
        _game("g2", 2026, 2, "A", "C", None, None),  # unplayed
        _game("g3", 2026, 3, "A", "D", None, None),  # unplayed
    ])

    assert nfl_pipeline.determine_predictable_week(schedules, 2026) == 2


def test_determine_predictable_week_no_schedule_posted_yet():
    schedules = pd.DataFrame(columns=["season", "game_type", "week", "home_score", "away_score"])

    assert nfl_pipeline.determine_predictable_week(schedules, 2026) is None


def test_determine_predictable_week_season_already_complete():
    schedules = pd.DataFrame([_game("g1", 2026, 1, "A", "B", 24, 17)])

    assert nfl_pipeline.determine_predictable_week(schedules, 2026) is None


def test_determine_predictable_week_ignores_playoff_games():
    schedules = pd.DataFrame([
        _game("g1", 2026, 1, "A", "B", 24, 17),
        _game("g2", 2026, 19, "A", "B", None, None, game_type="WC"),  # unplayed but a playoff game
    ])

    assert nfl_pipeline.determine_predictable_week(schedules, 2026) is None


def test_build_market_probabilities_devigs_real_moneylines():
    this_week_games = pd.DataFrame([_game("g1", 2026, 2, "A", "B", home_ml=-150, away_ml=130)])

    market = nfl_pipeline._build_market_probabilities(this_week_games)

    assert market.iloc[0]["market_home_win_probability"] > 0.5  # -150 is the real favorite


def test_write_nfl_game_picks_export_no_log_is_a_noop(tmp_path):
    output_dir = tmp_path / "docs_data"
    output_dir.mkdir()

    nfl_pipeline.write_nfl_game_picks_export(str(tmp_path / "does_not_exist.csv"), str(output_dir))

    assert list(output_dir.iterdir()) == []


def test_write_nfl_game_picks_export_writes_three_csvs(tmp_path):
    log_path = tmp_path / "nfl_game_predictions.csv"
    output_dir = tmp_path / "docs_data"
    row = {
        "date": "2026-09-10", "season": 2026, "week": 1, "game_id": "g1",
        "home_team": "A", "away_team": "B", "predicted_winner": "A", "predicted_probability": 0.6,
        "above_threshold": False, "metric": "NFL_GamePick_Win_Probability",
        "actual_winner": None, "game_played": None, "model_version": "v1",
        "market_home_win_probability": None,
        "bet_units": 0.0, "bet_side": None, "bet_team": None, "bet_moneyline": None, "bet_profit_units": None,
        "home_win_probability_pessimistic": None, "away_win_probability_pessimistic": None,
    }
    pd.DataFrame([row]).to_csv(log_path, index=False)

    nfl_pipeline.write_nfl_game_picks_export(str(log_path), str(output_dir))

    assert (output_dir / "nfl_game_picks_picks.csv").exists()
    assert (output_dir / "nfl_game_picks_summary.csv").exists()
    assert (output_dir / "nfl_game_picks_summary_by_version.csv").exists()


def test_run_end_to_end_with_synthetic_fetchers(tmp_path, monkeypatch):
    # A real, small 4-team round-robin across two seasons (mirrors
    # test_nfl_team_strength.py's own fixture pattern) - 2025 fully
    # played (real history), 2026 week 1 unplayed (the week to predict).
    # Monkeypatches TABLE_FETCHERS so no live network fetch happens.
    def _team_stats_row(team, opp, season, week, gid):
        return {
            "team": team, "opponent_team": opp, "season": season, "week": week, "game_id": gid,
            "season_type": "REG", "passing_epa": {"A": 2.0, "B": 1.0, "C": 0.5, "D": -0.5}[team] + week * 0.1,
            "rushing_epa": 0.5, "receiving_epa": 0.0,
        }

    def _weekly_row(team, season, week):
        return {
            "player_id": f"{team}_qb", "position": "QB", "season": season, "week": week, "season_type": "REG",
            "game_id": f"{season}_{week:02d}_{team}",
            "attempts": 30, "completions": 20, "passing_yards": 200, "passing_tds": 1,
            "passing_interceptions": 0, "carries": 2, "rushing_yards": 5, "rushing_tds": 0, "passing_epa": 1.0,
        }

    def _snap_row(team, season, week, gid):
        return {
            "game_id": gid, "season": season, "week": week, "game_type": "REG",
            "team": team, "position": "QB", "pfr_player_id": f"{team}_pfr",
            "offense_snaps": 60, "offense_pct": 0.95,
        }

    weekly_pairings = [[("A", "B"), ("C", "D")], [("A", "C"), ("B", "D")], [("A", "D"), ("B", "C")]]
    sched_2025, ts_2025, weekly_2025, snaps_2025 = [], [], [], []
    for week in range(1, 4):
        for game_num, (home, away) in enumerate(weekly_pairings[(week - 1) % 3], start=1):
            gid = f"2025_{week:02d}_{game_num}"
            sched_2025.append(_game(gid, 2025, week, home, away, 24, 17))
            for team, opp in [(home, away), (away, home)]:
                ts_2025.append(_team_stats_row(team, opp, 2025, week, gid))
                weekly_2025.append(_weekly_row(team, 2025, week))
                snaps_2025.append(_snap_row(team, 2025, week, gid))

    sched_2026 = pd.DataFrame([_game("2026_01_A_B", 2026, 1, "A", "B", None, None)])
    rosters = pd.DataFrame([{"season": s, "gsis_id": f"{t}_qb", "pfr_id": f"{t}_pfr"} for s in (2025, 2026) for t in "ABCD"])

    fake_tables = {
        "schedules": sched_2026,
        "team_stats": pd.DataFrame(columns=["team", "opponent_team", "season", "week", "game_id", "season_type", "passing_epa", "rushing_epa", "receiving_epa"]),
        "weekly": pd.DataFrame(columns=["player_id", "position", "season", "week", "season_type", "game_id"]),
        "snap_counts": pd.DataFrame(columns=["game_id", "season", "week", "game_type", "team", "position", "pfr_player_id", "offense_snaps", "offense_pct"]),
        "rosters_weekly": rosters[rosters["season"] == 2026],
    }
    monkeypatch.setattr(nfl_pipeline, "TABLE_FETCHERS", [(name, (lambda df: (lambda seasons: df))(df)) for name, df in fake_tables.items()])

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pd.DataFrame(sched_2025).to_parquet(raw_dir / "schedules_2025.parquet")
    pd.DataFrame(ts_2025).to_parquet(raw_dir / "team_stats_2025.parquet")
    pd.DataFrame(weekly_2025).to_parquet(raw_dir / "weekly_2025.parquet")
    pd.DataFrame(snaps_2025).to_parquet(raw_dir / "snap_counts_2025.parquet")
    rosters[rosters["season"] == 2025].to_parquet(raw_dir / "rosters_weekly_2025.parquet")

    predictions_log = tmp_path / "predictions" / "nfl_game_predictions.csv"
    output_dir = tmp_path / "docs_data"

    nfl_pipeline.run(
        season=2026, raw_dir=str(raw_dir), output_dir=str(output_dir), predictions_log_path=str(predictions_log)
    )

    assert predictions_log.exists()
    log = pd.read_csv(predictions_log)
    assert list(log["game_id"]) == ["2026_01_A_B"]
    assert (output_dir / "nfl_game_picks_picks.csv").exists()


def test_run_no_predictable_week_still_writes_export(tmp_path, monkeypatch):
    # Every real 2026 game already has a final score - nothing new to
    # predict, but the dashboard export still refreshes from whatever's
    # already logged.
    sched_2026 = pd.DataFrame([_game("2026_01_A_B", 2026, 1, "A", "B", 24, 17)])
    fake_tables = {
        "schedules": sched_2026,
        "team_stats": pd.DataFrame(),
        "weekly": pd.DataFrame(),
        "snap_counts": pd.DataFrame(),
        "rosters_weekly": pd.DataFrame(),
    }
    monkeypatch.setattr(nfl_pipeline, "TABLE_FETCHERS", [(name, (lambda df: (lambda seasons: df))(df)) for name, df in fake_tables.items()])

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    predictions_log = tmp_path / "predictions" / "nfl_game_predictions.csv"
    output_dir = tmp_path / "docs_data"

    nfl_pipeline.run(
        season=2026, raw_dir=str(raw_dir), output_dir=str(output_dir), predictions_log_path=str(predictions_log)
    )

    assert not predictions_log.exists()  # nothing was ever logged
    assert not output_dir.exists()  # write_nfl_game_picks_export is a no-op with no log - never even creates the dir


def test_run_resilient_to_a_failed_table_fetch(tmp_path, monkeypatch, capsys):
    sched_2026 = pd.DataFrame([_game("2026_01_A_B", 2026, 1, "A", "B", None, None)])

    def _raise(seasons):
        raise RuntimeError("nflverse hasn't published this season's file yet")

    monkeypatch.setattr(nfl_pipeline, "TABLE_FETCHERS", [
        ("schedules", lambda seasons: sched_2026),
        ("team_stats", _raise),
        ("weekly", _raise),
        ("snap_counts", _raise),
        ("rosters_weekly", _raise),
    ])

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    predictions_log = tmp_path / "predictions" / "nfl_game_predictions.csv"
    output_dir = tmp_path / "docs_data"

    # No prior-season history persisted at all - real "brand-new" case,
    # must not crash.
    nfl_pipeline.run(
        season=2026, raw_dir=str(raw_dir), output_dir=str(output_dir), predictions_log_path=str(predictions_log)
    )

    assert "falling back to the last persisted copy" in capsys.readouterr().out
    assert not predictions_log.exists()  # no real history to predict off of - skipped, not crashed
