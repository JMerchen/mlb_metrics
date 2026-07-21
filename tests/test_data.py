import pandas as pd

from mlb_metrics import data


def test_persist_raw_statcast_appends_and_dedupes_by_pitch(tmp_path):
    raw_dir = str(tmp_path / "raw")
    season = 2026

    day1 = pd.DataFrame({
        "game_pk": [1, 1],
        "at_bat_number": [1, 2],
        "pitch_number": [1, 1],
        "game_date": pd.to_datetime(["2026-04-01", "2026-04-01"]),
        "events": ["single", "field_out"],
    })
    combined1 = data.persist_raw_statcast(day1, raw_dir, season)
    assert len(combined1) == 2

    day2 = pd.DataFrame({
        "game_pk": [1, 2],
        "at_bat_number": [2, 1],  # first row shares day1's pitch key
        "pitch_number": [1, 1],
        "game_date": pd.to_datetime(["2026-04-01", "2026-04-02"]),
        "events": ["walk", "single"],
    })
    combined2 = data.persist_raw_statcast(day2, raw_dir, season)

    assert len(combined2) == 3
    dup_row = combined2[(combined2["game_pk"] == 1) & (combined2["at_bat_number"] == 2)]
    assert dup_row.iloc[0]["events"] == "walk"  # keep="last" -> day2's value wins on conflict

    reloaded = data.load_persisted_statcast(raw_dir, season)
    assert len(reloaded) == 3


def test_load_persisted_statcast_returns_none_when_absent(tmp_path):
    assert data.load_persisted_statcast(str(tmp_path / "raw"), 2026) is None
