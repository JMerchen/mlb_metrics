"""End-to-end smoke test for scripts/fetch_nfl_historical.py's wiring:
confirms the skip-if-already-persisted / fetch-and-persist-if-absent
orchestration, without needing real network access to nflreadpy (blocked
in this sandbox - see nfl_data.py's module docstring)."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd

from mlb_metrics import nfl_data

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "fetch_nfl_historical.py"


def _load_fetch_nfl_historical_module():
    sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location("fetch_nfl_historical", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fetches_and_persists_missing_seasons(tmp_path, monkeypatch):
    module = _load_fetch_nfl_historical_module()
    raw_dir = tmp_path / "raw"

    calls = []

    def fake_fetch(name):
        def _fetch(seasons):
            calls.append((name, seasons))
            return pd.DataFrame([{"season": seasons[0], "col": name}])
        return _fetch

    module.TABLES = [(name, fake_fetch(name)) for name, _ in module.TABLES]

    monkeypatch.setattr(sys, "argv", ["fetch_nfl_historical.py", "--raw-dir", str(raw_dir), "--seasons", "2024"])
    module.main()

    assert len(calls) == len(module.TABLES)
    for name, _ in module.TABLES:
        loaded = nfl_data.load_persisted_table(str(raw_dir), name, 2024)
        assert loaded is not None
        assert loaded.iloc[0]["col"] == name


def test_skips_already_persisted_seasons(tmp_path, monkeypatch):
    module = _load_fetch_nfl_historical_module()
    raw_dir = tmp_path / "raw"

    # Pre-persist every table for 2024 so the script should skip all of them.
    for name, _ in module.TABLES:
        nfl_data.persist_table(pd.DataFrame([{"season": 2024, "col": "already-there"}]), str(raw_dir), name, 2024)

    calls = []

    def fake_fetch(seasons):
        calls.append(seasons)
        return pd.DataFrame([{"season": seasons[0]}])

    module.TABLES = [(name, fake_fetch) for name, _ in module.TABLES]

    monkeypatch.setattr(sys, "argv", ["fetch_nfl_historical.py", "--raw-dir", str(raw_dir), "--seasons", "2024"])
    module.main()

    assert calls == []
    for name, _ in module.TABLES:
        loaded = nfl_data.load_persisted_table(str(raw_dir), name, 2024)
        assert loaded.iloc[0]["col"] == "already-there"
