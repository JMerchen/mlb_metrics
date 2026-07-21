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
    )

    assert captured["dates"] == ["2026-06-18", "2026-06-19"]


def test_run_rejects_as_of_date_before_season_start(tmp_path):
    try:
        pipeline.run(datetime.date(2026, 1, 1), raw_dir=str(tmp_path / "raw"), output_dir=str(tmp_path / "out"))
        assert False, "expected ValueError"
    except ValueError:
        pass
