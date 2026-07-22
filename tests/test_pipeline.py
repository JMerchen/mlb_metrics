import datetime
import os

import pandas as pd
import pytest

from mlb_metrics import pipeline


def test_run_excludes_games_on_or_after_as_of_date(monkeypatch, tmp_path):
    """The pipeline must only ever compute metrics from games strictly
    before --as-of-date, regardless of what the fetch/persist layer hands
    back - this is the fix for the original script's implicit same-day
    leakage risk (see mlb_metrics.pipeline module docstring)."""
    raw = pd.DataFrame({
        "game_date": pd.to_datetime(["2026-06-18", "2026-06-19", "2026-06-20"]),
        "value": [1, 2, 3],
    })

    monkeypatch.setattr(pipeline.data, "fetch_statcast_range", lambda start, end: raw)
    monkeypatch.setattr(pipeline.data, "persist_raw_statcast", lambda df, raw_dir, season: df)

    captured = {}

    def fake_compute_outputs(df):
        captured["dates"] = sorted(df["game_date"].dt.strftime("%Y-%m-%d").tolist())
        return {"wave": pd.DataFrame(), "pave": pd.DataFrame(), "confidence": pd.DataFrame()}

    monkeypatch.setattr(pipeline, "compute_outputs", fake_compute_outputs)

    pipeline.run(
        datetime.date(2026, 6, 20),
        raw_dir=str(tmp_path / "raw"),
        output_dir=str(tmp_path / "out"),
        log_predictions=False,
    )

    assert captured["dates"] == ["2026-06-18", "2026-06-19"]


def test_run_rejects_as_of_date_before_season_start(tmp_path):
    try:
        pipeline.run(
            datetime.date(2026, 1, 1),
            raw_dir=str(tmp_path / "raw"),
            output_dir=str(tmp_path / "out"),
            log_predictions=False,
        )
        assert False, "expected ValueError"
    except ValueError:
        pass


def _no_schedule(date):
    raise RuntimeError("no network in tests")


def test_run_logs_and_resolves_predictions(monkeypatch, tmp_path):
    """End-to-end check that run() both logs today's picks and resolves any
    pick from a previous run whose target date has now happened, using
    exactly the data fetched for this run (no extra data source)."""
    raw = pd.DataFrame({
        "game_date": pd.to_datetime(["2026-06-18", "2026-06-18", "2026-06-19", "2026-06-19"]),
        "batter": [1, 2, 1, 2],
        # Both of 06-19's picks get a hit -> the Beat the Streak day should succeed.
        "events": ["field_out", "field_out", "single", "double"],
    })
    monkeypatch.setattr(pipeline.data, "fetch_statcast_range", lambda start, end: raw)
    monkeypatch.setattr(pipeline.data, "persist_raw_statcast", lambda df, raw_dir, season: df)
    # This test only cares about Game_Hit_Probability picks/streak wiring,
    # not matchup blending or game picks - and must not depend on live
    # network access (statsapi is reachable in CI but not in every sandbox,
    # which would make this test's outcome nondeterministic depending on
    # where it runs).
    monkeypatch.setattr(pipeline.schedule, "fetch_probable_pitchers", _no_schedule)
    monkeypatch.setattr(pipeline.schedule, "fetch_todays_games", _no_schedule)

    wave = pd.DataFrame([
        {
            "key_mlbam": 1, "name_first": "Test", "name_last": "PlayerOne", "team": "NYY",
            "PA_L": 0, "PA_R": 40, "probability_L": 0, "probability_R": 0.9, "probability": 0.9,
            "Game_Hit_Probability": 0.8, "Consistency": -0.1, "Approach": 0.72, "Expected_Bases": 1.5,
        },
        {
            "key_mlbam": 2, "name_first": "Test", "name_last": "PlayerTwo", "team": "BOS",
            "PA_L": 0, "PA_R": 35, "probability_L": 0, "probability_R": 0.8, "probability": 0.8,
            # Above DAILY_PICK_MIN_PROBABILITY (0.80) too, so both picks
            # qualify as "recommended" and the streak wiring below covers a
            # genuine 2-hit day, not just a 1-pick day.
            "Game_Hit_Probability": 0.85, "Consistency": -0.1, "Approach": 0.56, "Expected_Bases": 1.2,
        },
    ])
    monkeypatch.setattr(
        pipeline, "compute_outputs",
        lambda df: {"wave": wave, "pave": pd.DataFrame(), "confidence": pd.DataFrame()},
    )

    predictions_dir = str(tmp_path / "predictions")
    log_path = f"{predictions_dir}/predictions.csv"

    # First run: as_of_date=2026-06-19 logs a pending pick for that date.
    pipeline.run(
        datetime.date(2026, 6, 19),
        raw_dir=str(tmp_path / "raw"),
        output_dir=str(tmp_path / "out"),
        predictions_dir=predictions_dir,
        persist_raw=False,
        log_predictions=True,
    )
    logged = pd.read_csv(log_path, parse_dates=["date"])
    assert len(logged) == 2  # both ranked picks for 06-19
    assert logged["actual_hit"].isna().all()

    # Second run: as_of_date=2026-06-20 fetches through 06-19, which resolves
    # both pending 06-19 picks (both batters got a hit that day) and logs new
    # pending picks for 06-20.
    pipeline.run(
        datetime.date(2026, 6, 20),
        raw_dir=str(tmp_path / "raw"),
        output_dir=str(tmp_path / "out"),
        predictions_dir=predictions_dir,
        persist_raw=False,
        log_predictions=True,
    )
    logged = pd.read_csv(log_path, parse_dates=["date"])
    assert len(logged) == 4
    resolved_rows = logged[logged["date"] == "2026-06-19"]
    assert (resolved_rows["actual_hit"] == 1.0).all()
    pending_rows = logged[logged["date"] == "2026-06-20"]
    assert pending_rows["actual_hit"].isna().all()

    # write_beat_the_streak_export() should have run after each pass and
    # reflect that 06-19's day (both picks hit) added 2 to the streak.
    picks_export = pd.read_csv(f"{tmp_path}/out/beat_the_streak_picks.csv", parse_dates=["date"])
    assert set(picks_export["date"].dt.strftime("%Y-%m-%d")) == {"2026-06-19", "2026-06-20"}
    resolved_export_rows = picks_export[picks_export["date"] == "2026-06-19"]
    assert len(resolved_export_rows) == 2  # both picks cleared the recommendation threshold
    assert (resolved_export_rows["status"] == "hit").all()

    summary_export = pd.read_csv(f"{tmp_path}/out/beat_the_streak_summary.csv")
    assert summary_export.loc[0, "n_days_resolved"] == 1
    assert summary_export.loc[0, "current_streak"] == 2
    assert summary_export.loc[0, "longest_streak"] == 2

    # Every pick here was logged by this run's own select_picks() call (no
    # legacy rows mixed in), so the current-model-version row matches the
    # all_time one exactly.
    by_version = pd.read_csv(f"{tmp_path}/out/beat_the_streak_summary_by_version.csv")
    assert set(by_version["model_version"]) == {"all_time", pipeline.config.HITTER_MODEL_VERSION}
    current_row = by_version[by_version["model_version"] == pipeline.config.HITTER_MODEL_VERSION].iloc[0]
    assert current_row["n_days_resolved"] == 1
    assert current_row["current_streak"] == 2


def _minimal_outputs():
    wave = pd.DataFrame([
        {
            "key_mlbam": 1, "name_first": "Test", "name_last": "PlayerOne", "team": "NYY",
            "PA_L": 0, "PA_R": 40, "WAVE": 0.33, "probability_L": 0, "probability_R": 0.9, "probability": 0.9,
            "Game_Hit_Probability": 0.85, "Consistency": -0.1, "Approach": 0.72, "Expected_Bases": 1.5,
        },
    ])
    # PAVE/PAVE_PLUS chosen so this fixture's Matchup_Hit_Probability clears
    # HITTER_MIN_PROBABILITY (see test_run_logs_matchup_probability_when_schedule_fetch_succeeds).
    pave = pd.DataFrame([{
        "key_mlbam": 999, "name_first": "Probable", "name_last": "Starter",
        "PAVE": 0.25, "PAVE_PLUS": 0.9, "Power_A_PLUS": 0.9,
    }])
    confidence = pd.DataFrame([{"team": "BOS", "Bullpen_PAVE": 0.28, "Bullpen_PAVE_PLUS": 1.0}])
    return {"wave": wave, "pave": pave, "confidence": confidence}


def test_run_logs_matchup_probability_when_schedule_fetch_succeeds(monkeypatch, tmp_path):
    """Matchup_Hit_Probability is no longer a separate parallel metric row -
    it's merged into the same Game_Hit_Probability-tagged pick pool and
    factored into both the joint qualifier gate and the ranking (see
    pipeline.run). This batter must still qualify: probability=0.9,
    Game_Hit_Probability=0.85, and (per _minimal_outputs' comment)
    Matchup_Hit_Probability all clear HITTER_MIN_PROBABILITY."""
    monkeypatch.setattr(pipeline.data, "fetch_statcast_range", lambda start, end: pd.DataFrame({
        "game_date": pd.to_datetime([]), "batter": [], "events": [],
    }))
    monkeypatch.setattr(pipeline.data, "persist_raw_statcast", lambda df, raw_dir, season: df)
    monkeypatch.setattr(pipeline, "compute_outputs", lambda df: _minimal_outputs())

    schedule_df = pd.DataFrame([{
        "date": pd.Timestamp("2026-06-20"), "team": "NYY", "opponent": "BOS",
        "probable_pitcher_key_mlbam": 999, "is_home": True,
    }])
    monkeypatch.setattr(pipeline.schedule, "fetch_probable_pitchers", lambda date: schedule_df)
    # Scoped to hitter-pick metrics only - game picks get their own tests below.
    monkeypatch.setattr(pipeline.schedule, "fetch_todays_games", lambda date: pd.DataFrame())

    predictions_dir = str(tmp_path / "predictions")
    pipeline.run(
        datetime.date(2026, 6, 20),
        raw_dir=str(tmp_path / "raw"),
        output_dir=str(tmp_path / "out"),
        predictions_dir=predictions_dir,
        persist_raw=False,
    )

    logged = pd.read_csv(f"{predictions_dir}/predictions.csv")
    assert set(logged["metric"]) == {"Game_Hit_Probability"}
    assert (logged["key_mlbam"] == 1).all()
    assert logged.loc[0, "predicted_probability"] == pytest.approx(0.85)  # still Game_Hit_Probability, not the matchup blend

    probable_pitchers = pd.read_csv(f"{tmp_path}/out/probable_pitchers.csv")
    assert len(probable_pitchers) == 1
    assert probable_pitchers.loc[0, "team"] == "NYY"
    assert probable_pitchers.loc[0, "opponent"] == "BOS"
    assert probable_pitchers.loc[0, "pitcher_name"] == "Probable Starter"
    assert probable_pitchers.loc[0, "PAVE_PLUS"] == pytest.approx(0.9)


def test_run_excludes_pick_with_a_bad_matchup_even_with_strong_probability_and_ghp(monkeypatch, tmp_path):
    """A batter with strong probability/Game_Hit_Probability but a tough
    matchup (a dominant opposing starter+bullpen) must be excluded once
    schedule/matchup data is available - a good matchup is now required just
    as much as the other two signals (see predictions.select_picks)."""
    monkeypatch.setattr(pipeline.data, "fetch_statcast_range", lambda start, end: pd.DataFrame({
        "game_date": pd.to_datetime([]), "batter": [], "events": [],
    }))
    monkeypatch.setattr(pipeline.data, "persist_raw_statcast", lambda df, raw_dir, season: df)

    def _outputs_with_tough_opponent(df):
        outputs = _minimal_outputs()
        # A dominant probable starter + bullpen (both well below league
        # PAVE) should drag this batter's Matchup_Hit_Probability under
        # HITTER_MIN_PROBABILITY despite their strong standalone numbers.
        outputs["pave"] = pd.DataFrame([{
            "key_mlbam": 999, "name_first": "Tough", "name_last": "Opponent",
            "PAVE": 0.08, "PAVE_PLUS": 0.08 / 0.25,
        }])
        outputs["confidence"] = pd.DataFrame([{"team": "BOS", "Bullpen_PAVE": 0.08, "Bullpen_PAVE_PLUS": 1.0}])
        return outputs

    monkeypatch.setattr(pipeline, "compute_outputs", _outputs_with_tough_opponent)

    schedule_df = pd.DataFrame([{
        "date": pd.Timestamp("2026-06-20"), "team": "NYY", "opponent": "BOS",
        "probable_pitcher_key_mlbam": 999, "is_home": True,
    }])
    monkeypatch.setattr(pipeline.schedule, "fetch_probable_pitchers", lambda date: schedule_df)
    monkeypatch.setattr(pipeline.schedule, "fetch_todays_games", lambda date: pd.DataFrame())

    predictions_dir = str(tmp_path / "predictions")
    pipeline.run(
        datetime.date(2026, 6, 20),
        raw_dir=str(tmp_path / "raw"),
        output_dir=str(tmp_path / "out"),
        predictions_dir=predictions_dir,
        persist_raw=False,
    )

    logged = pd.read_csv(f"{predictions_dir}/predictions.csv")
    assert logged.empty


def test_run_continues_without_matchup_when_schedule_fetch_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline.data, "fetch_statcast_range", lambda start, end: pd.DataFrame({
        "game_date": pd.to_datetime([]), "batter": [], "events": [],
    }))
    monkeypatch.setattr(pipeline.data, "persist_raw_statcast", lambda df, raw_dir, season: df)
    monkeypatch.setattr(pipeline, "compute_outputs", lambda df: _minimal_outputs())

    def raise_error(date):
        raise RuntimeError("statsapi is down")

    monkeypatch.setattr(pipeline.schedule, "fetch_probable_pitchers", raise_error)
    monkeypatch.setattr(pipeline.schedule, "fetch_todays_games", raise_error)

    predictions_dir = str(tmp_path / "predictions")
    # Must not raise - a new external dependency can't be allowed to break
    # the whole daily update.
    pipeline.run(
        datetime.date(2026, 6, 20),
        raw_dir=str(tmp_path / "raw"),
        output_dir=str(tmp_path / "out"),
        predictions_dir=predictions_dir,
        persist_raw=False,
    )

    logged = pd.read_csv(f"{predictions_dir}/predictions.csv")
    assert set(logged["metric"]) == {"Game_Hit_Probability"}  # no Matchup_Hit_Probability logged
    assert not os.path.exists(f"{predictions_dir}/game_predictions.csv")  # no game picks logged either
    assert not os.path.exists(f"{tmp_path}/out/probable_pitchers.csv")  # no probable-pitchers list either


def _minimal_outputs_with_confidence():
    """Like _minimal_outputs, but with the confidence.csv columns
    game_picks.compute_game_win_probabilities needs for two teams."""
    outputs = _minimal_outputs()
    # Bullpen_Power_A_PLUS/Power_A_PLUS mirror the PAVE_PLUS-side values (a
    # reasonable synthetic stand-in) so GAME_PICK_SUSCEPTIBILITY_WEIGHT's
    # blend doesn't dilute this fixture's intended separation toward neutral.
    outputs["confidence"] = pd.DataFrame([
        {
            "team": "NYY", "pyth_Strength": 1.1, "pyth_Confidence": 1.05,
            "suppression_resistance": 1.0, "true_power": 1.0,
            "Bullpen_PAVE_PLUS": 0.9, "Bullpen_Power_A_PLUS": 0.9,
        },
        {
            "team": "BOS", "pyth_Strength": 0.9, "pyth_Confidence": 0.95,
            "suppression_resistance": 1.0, "true_power": 1.0,
            "Bullpen_PAVE_PLUS": 1.1, "Bullpen_Power_A_PLUS": 1.1,
        },
    ])
    outputs["pave"] = pd.DataFrame([
        {"key_mlbam": 501, "PAVE_PLUS": 0.8, "Power_A_PLUS": 0.8},  # NYY probable starter (tough)
        {"key_mlbam": 502, "PAVE_PLUS": 1.2, "Power_A_PLUS": 1.2},  # BOS probable starter (easy)
    ])
    return outputs


def test_run_logs_game_picks_when_schedule_fetch_succeeds(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline.data, "fetch_statcast_range", lambda start, end: pd.DataFrame({
        "game_date": pd.to_datetime([]), "batter": [], "events": [],
    }))
    monkeypatch.setattr(pipeline.data, "persist_raw_statcast", lambda df, raw_dir, season: df)
    monkeypatch.setattr(pipeline, "compute_outputs", lambda df: _minimal_outputs_with_confidence())
    monkeypatch.setattr(pipeline.schedule, "fetch_probable_pitchers", _no_schedule)

    schedule_games_df = pd.DataFrame([{
        "game_pk": 100, "date": pd.Timestamp("2026-06-20"), "home_team": "NYY", "away_team": "BOS",
        "home_probable_pitcher_key_mlbam": 501, "away_probable_pitcher_key_mlbam": 502,
        "status": "Scheduled", "home_score": None, "away_score": None,
    }])
    monkeypatch.setattr(pipeline.schedule, "fetch_todays_games", lambda date: schedule_games_df)

    predictions_dir = str(tmp_path / "predictions")
    pipeline.run(
        datetime.date(2026, 6, 20),
        raw_dir=str(tmp_path / "raw"),
        output_dir=str(tmp_path / "out"),
        predictions_dir=predictions_dir,
        persist_raw=False,
    )

    logged = pd.read_csv(f"{predictions_dir}/game_predictions.csv", parse_dates=["date"])
    assert len(logged) == 1
    assert logged.loc[0, "game_pk"] == 100
    assert logged.loc[0, "predicted_winner"] == "NYY"  # NYY has the better composite + faces the weaker pitching
    assert logged.loc[0, "metric"] == "GamePick_Win_Probability"
    assert pd.isna(logged.loc[0, "game_played"])

    summary = pd.read_csv(f"{tmp_path}/out/game_picks_summary.csv")
    assert summary.loc[0, "n_games_resolved"] == 0


def test_run_continues_without_game_picks_when_schedule_fetch_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline.data, "fetch_statcast_range", lambda start, end: pd.DataFrame({
        "game_date": pd.to_datetime([]), "batter": [], "events": [],
    }))
    monkeypatch.setattr(pipeline.data, "persist_raw_statcast", lambda df, raw_dir, season: df)
    monkeypatch.setattr(pipeline, "compute_outputs", lambda df: _minimal_outputs_with_confidence())
    monkeypatch.setattr(pipeline.schedule, "fetch_probable_pitchers", _no_schedule)
    monkeypatch.setattr(pipeline.schedule, "fetch_todays_games", _no_schedule)

    predictions_dir = str(tmp_path / "predictions")
    # Must not raise - a new external dependency can't be allowed to break
    # the whole daily update.
    pipeline.run(
        datetime.date(2026, 6, 20),
        raw_dir=str(tmp_path / "raw"),
        output_dir=str(tmp_path / "out"),
        predictions_dir=predictions_dir,
        persist_raw=False,
    )

    assert not os.path.exists(f"{predictions_dir}/game_predictions.csv")


def test_run_resolves_game_picks_across_two_runs(monkeypatch, tmp_path):
    """A game pick logged on day N as pending gets resolved on day N+1 once
    schedule.fetch_game_results reports it Final, and the exported
    game_picks_picks.csv/game_picks_summary.csv reflect the win."""
    monkeypatch.setattr(pipeline.data, "fetch_statcast_range", lambda start, end: pd.DataFrame({
        "game_date": pd.to_datetime([]), "batter": [], "events": [],
    }))
    monkeypatch.setattr(pipeline.data, "persist_raw_statcast", lambda df, raw_dir, season: df)
    monkeypatch.setattr(pipeline, "compute_outputs", lambda df: _minimal_outputs_with_confidence())
    monkeypatch.setattr(pipeline.schedule, "fetch_probable_pitchers", _no_schedule)

    schedule_games_day1 = pd.DataFrame([{
        "game_pk": 100, "date": pd.Timestamp("2026-06-19"), "home_team": "NYY", "away_team": "BOS",
        "home_probable_pitcher_key_mlbam": 501, "away_probable_pitcher_key_mlbam": 502,
        "status": "Scheduled", "home_score": None, "away_score": None,
    }])
    monkeypatch.setattr(pipeline.schedule, "fetch_todays_games", lambda date: schedule_games_day1)

    predictions_dir = str(tmp_path / "predictions")
    pipeline.run(
        datetime.date(2026, 6, 19),
        raw_dir=str(tmp_path / "raw"),
        output_dir=str(tmp_path / "out"),
        predictions_dir=predictions_dir,
        persist_raw=False,
    )

    # Day 2: no new games today, but day 1's pick resolves - NYY (predicted
    # winner) actually won 5-3.
    monkeypatch.setattr(pipeline.schedule, "fetch_todays_games", lambda date: pd.DataFrame())

    def fake_fetch_results(date):
        assert date == datetime.date(2026, 6, 19)
        return pd.DataFrame([{"game_pk": 100, "status": "Final", "home_score": 5, "away_score": 3}])

    monkeypatch.setattr(pipeline.schedule, "fetch_game_results", fake_fetch_results)

    pipeline.run(
        datetime.date(2026, 6, 20),
        raw_dir=str(tmp_path / "raw"),
        output_dir=str(tmp_path / "out"),
        predictions_dir=predictions_dir,
        persist_raw=False,
    )

    logged = pd.read_csv(f"{predictions_dir}/game_predictions.csv")
    assert logged.loc[0, "actual_winner"] == "NYY"
    assert logged.loc[0, "game_played"] == 1

    picks_export = pd.read_csv(f"{tmp_path}/out/game_picks_picks.csv")
    assert picks_export.loc[0, "status"] == "win"

    summary_export = pd.read_csv(f"{tmp_path}/out/game_picks_summary.csv")
    assert summary_export.loc[0, "n_games_resolved"] == 1
    assert summary_export.loc[0, "accuracy"] == 1.0

    by_version = pd.read_csv(f"{tmp_path}/out/game_picks_summary_by_version.csv")
    assert set(by_version["model_version"]) == {"all_time", pipeline.config.GAME_PICK_MODEL_VERSION}
    current_row = by_version[by_version["model_version"] == pipeline.config.GAME_PICK_MODEL_VERSION].iloc[0]
    assert current_row["n_games_resolved"] == 1
    assert current_row["accuracy"] == 1.0
