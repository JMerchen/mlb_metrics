import pandas as pd

from mlb_metrics import nfl_data


def test_table_path_includes_table_and_season():
    path = nfl_data.table_path("data/raw/nfl", "weekly", 2025)
    assert path == "data/raw/nfl/weekly_2025.parquet"


def test_load_persisted_table_returns_none_when_absent(tmp_path):
    assert nfl_data.load_persisted_table(str(tmp_path), "weekly", 2025) is None


def test_persist_and_load_table_round_trips(tmp_path):
    df = pd.DataFrame([{"player_id": "00-0023459", "season": 2025, "week": 1}])
    raw_dir = str(tmp_path / "raw")

    nfl_data.persist_table(df, raw_dir, "weekly", 2025)
    loaded = nfl_data.load_persisted_table(raw_dir, "weekly", 2025)

    assert loaded is not None
    assert loaded.iloc[0]["player_id"] == "00-0023459"


def test_persist_table_overwrites_not_appends(tmp_path):
    # Real behavior this project depends on: nflverse retroactively
    # corrects published stats, so a fresh persist must fully replace the
    # prior file, not merge into it (unlike data.py's Statcast pattern) -
    # see nfl_data.py's module docstring for the full reasoning.
    raw_dir = str(tmp_path / "raw")
    first = pd.DataFrame([{"player_id": "a", "passing_yards": 200}, {"player_id": "b", "passing_yards": 100}])
    nfl_data.persist_table(first, raw_dir, "weekly", 2025)

    corrected = pd.DataFrame([{"player_id": "a", "passing_yards": 250}])
    nfl_data.persist_table(corrected, raw_dir, "weekly", 2025)

    loaded = nfl_data.load_persisted_table(raw_dir, "weekly", 2025)
    assert len(loaded) == 1
    assert loaded.iloc[0]["passing_yards"] == 250
