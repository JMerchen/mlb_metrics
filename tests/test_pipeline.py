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
        "game_date": pd.to_datetime(["2026-06-18", "2026-06-19"]),
        "batter": [1, 1],
        "events": ["field_out", "single"],  # batter 1 gets a hit on 06-19, the date that gets picked/resolved
    })
    monkeypatch.setattr(pipeline.data, "fetch_statcast_range", lambda start, end: raw)
    monkeypatch.setattr(pipeline.data, "persist_raw_statcast", lambda df, raw_dir, season: df)

    wave = pd.DataFrame([
        {
            "key_mlbam": 1, "name_first": "Test", "name_last": "Player", "team": "NYY",
            "PA_L": 0, "PA_R": 40, "probability_L": 0, "probability_R": 0.9, "probability": 0.9,
            "Game_Hit_Probability": 0.8, "Consistency": -0.1, "Approach": 0.72, "Expected_Bases": 1.5,
        }
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
    assert len(logged) == 1
    assert pd.isna(logged.loc[0, "actual_hit"])

    # Second run: as_of_date=2026-06-20 fetches through 06-19, which resolves
    # the pending 06-19 pick (batter 1 had a single that day -> hit) and logs
    # a new pending pick for 06-20.
    pipeline.run(
        datetime.date(2026, 6, 20),
        raw_dir=str(tmp_path / "raw"),
        output_dir=str(tmp_path / "out"),
        predictions_dir=predictions_dir,
        persist_raw=False,
        log_predictions=True,
    )
    logged = pd.read_csv(log_path, parse_dates=["date"])
    assert len(logged) == 2
    resolved_row = logged[logged["date"] == "2026-06-19"].iloc[0]
    assert resolved_row["actual_hit"] == 1.0
    pending_row = logged[logged["date"] == "2026-06-20"].iloc[0]
    assert pd.isna(pending_row["actual_hit"])
