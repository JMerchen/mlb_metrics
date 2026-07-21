import datetime

import pandas as pd

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


def _minimal_outputs():
    wave = pd.DataFrame([
        {
            "key_mlbam": 1, "name_first": "Test", "name_last": "PlayerOne", "team": "NYY",
            "PA_L": 0, "PA_R": 40, "probability_L": 0, "probability_R": 0.9, "probability": 0.9,
            "Game_Hit_Probability": 0.85, "Consistency": -0.1, "Approach": 0.72, "Expected_Bases": 1.5,
        },
    ])
    pave = pd.DataFrame([{"key_mlbam": 999, "PAVE_PLUS": 0.9}])
    confidence = pd.DataFrame([{"team": "BOS", "Bullpen_PAVE_PLUS": 1.0}])
    return {"wave": wave, "pave": pave, "confidence": confidence}


def test_run_logs_matchup_probability_when_schedule_fetch_succeeds(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline.data, "fetch_statcast_range", lambda start, end: pd.DataFrame({
        "game_date": pd.to_datetime([]), "batter": [], "events": [],
    }))
    monkeypatch.setattr(pipeline.data, "persist_raw_statcast", lambda df, raw_dir, season: df)
    monkeypatch.setattr(pipeline, "compute_outputs", lambda df: _minimal_outputs())

    schedule_df = pd.DataFrame([{"team": "NYY", "opponent": "BOS", "probable_pitcher_key_mlbam": 999}])
    monkeypatch.setattr(pipeline.schedule, "fetch_probable_pitchers", lambda date: schedule_df)

    predictions_dir = str(tmp_path / "predictions")
    pipeline.run(
        datetime.date(2026, 6, 20),
        raw_dir=str(tmp_path / "raw"),
        output_dir=str(tmp_path / "out"),
        predictions_dir=predictions_dir,
        persist_raw=False,
    )

    logged = pd.read_csv(f"{predictions_dir}/predictions.csv")
    assert set(logged["metric"]) == {"Game_Hit_Probability", "Matchup_Hit_Probability"}
    assert (logged["key_mlbam"] == 1).all()


def test_run_continues_without_matchup_when_schedule_fetch_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline.data, "fetch_statcast_range", lambda start, end: pd.DataFrame({
        "game_date": pd.to_datetime([]), "batter": [], "events": [],
    }))
    monkeypatch.setattr(pipeline.data, "persist_raw_statcast", lambda df, raw_dir, season: df)
    monkeypatch.setattr(pipeline, "compute_outputs", lambda df: _minimal_outputs())

    def raise_error(date):
        raise RuntimeError("statsapi is down")

    monkeypatch.setattr(pipeline.schedule, "fetch_probable_pitchers", raise_error)

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
