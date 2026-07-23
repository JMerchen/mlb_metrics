"""End-to-end smoke test for scripts/build_age_curves.py's wiring: a
synthetic current-season Statcast pull + synthetic Lahman tables, checked
that build_current_player_pool correctly threads the key_mlbam -> Lahman
playerID crosswalk and age computation, without needing real network
access to pybaseball's Lahman/chadwick_register downloads (blocked in this
sandbox - see scripts/fetch_lahman.py's docstring)."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_age_curves.py"


def _load_build_age_curves_module():
    sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location("build_age_curves", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _statcast_at_bats(batter, n, hit_events="single"):
    return [
        {"game_pk": 1, "game_date": pd.Timestamp("2026-06-01"), "batter": batter, "events": hit_events}
        for _ in range(n)
    ]


def test_build_current_player_pool_threads_crosswalk_and_age(tmp_path, monkeypatch):
    module = _load_build_age_curves_module()

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = _statcast_at_bats(1, 10)
    pd.DataFrame(rows).to_parquet(raw_dir / "statcast_2026.parquet", index=False)

    chadwick_register = pd.DataFrame([
        {"key_mlbam": 1, "key_bbref": "playerx01", "name_first": "Test", "name_last": "Player"},
    ])
    monkeypatch.setattr(module.data, "get_name_register", lambda: chadwick_register)

    people = pd.DataFrame([
        {"playerID": "playerx01", "bbrefID": "playerx01", "birthYear": 1996, "birthMonth": 1, "birthDay": 1},
    ])
    monkeypatch.setattr(
        module.lahman_data,
        "load_persisted_lahman_table",
        lambda raw_dir, name: people if name == "people" else None,
    )

    pool = module.build_current_player_pool(str(raw_dir), season=2026, min_at_bats=5)

    assert len(pool) == 1
    row = pool.iloc[0]
    assert row["key_mlbam"] == 1
    assert row["name_first"] == "Test"
    assert row["age"] == 30  # born 1996-01-01, as of 2026-06-30
    assert row["OPS"] > 0


def test_build_current_player_pool_excludes_players_below_min_at_bats(tmp_path, monkeypatch):
    module = _load_build_age_curves_module()

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = _statcast_at_bats(1, 3)  # below the min_at_bats threshold used below
    pd.DataFrame(rows).to_parquet(raw_dir / "statcast_2026.parquet", index=False)

    chadwick_register = pd.DataFrame([{"key_mlbam": 1, "key_bbref": "playerx01", "name_first": "T", "name_last": "P"}])
    monkeypatch.setattr(module.data, "get_name_register", lambda: chadwick_register)
    people = pd.DataFrame([{"playerID": "playerx01", "bbrefID": "playerx01", "birthYear": 1996, "birthMonth": 1, "birthDay": 1}])
    monkeypatch.setattr(
        module.lahman_data, "load_persisted_lahman_table", lambda raw_dir, name: people if name == "people" else None
    )

    pool = module.build_current_player_pool(str(raw_dir), season=2026, min_at_bats=5)

    assert pool.empty


def test_describe_comparables_joins_name_and_next_season(tmp_path):
    module = _load_build_age_curves_module()

    comparables = pd.DataFrame([{"playerID": "p1", "yearID": 2000, "age": 27, "AB": 400, "OPS": 0.750}])
    historical_seasons = pd.DataFrame([
        {"playerID": "p1", "yearID": 2001, "age": 28, "AB": 400, "OPS": 0.800},  # p1's actual next season
    ])
    people = pd.DataFrame([{"playerID": "p1", "nameFirst": "Test", "nameLast": "Player"}])

    result = module.describe_comparables(999, comparables, historical_seasons, people)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["key_mlbam"] == 999
    assert row["name"] == "Test Player"
    assert row["next_OPS"] == pytest.approx(0.800)


def test_describe_comparables_null_next_ops_when_no_next_season(tmp_path):
    module = _load_build_age_curves_module()

    comparables = pd.DataFrame([{"playerID": "p1", "yearID": 2000, "age": 27, "AB": 400, "OPS": 0.750}])
    historical_seasons = pd.DataFrame(columns=["playerID", "yearID", "age", "AB", "OPS"])
    people = pd.DataFrame([{"playerID": "p1", "nameFirst": "Test", "nameLast": "Player"}])

    result = module.describe_comparables(999, comparables, historical_seasons, people)

    assert pd.isna(result.iloc[0]["next_OPS"])


def test_build_current_player_pool_excludes_players_with_no_lahman_match(tmp_path, monkeypatch):
    module = _load_build_age_curves_module()

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = _statcast_at_bats(1, 10)
    pd.DataFrame(rows).to_parquet(raw_dir / "statcast_2026.parquet", index=False)

    chadwick_register = pd.DataFrame([{"key_mlbam": 1, "key_bbref": "unmatched01", "name_first": "T", "name_last": "P"}])
    monkeypatch.setattr(module.data, "get_name_register", lambda: chadwick_register)
    # People has no row with a matching bbrefID.
    people = pd.DataFrame([{"playerID": "other01", "bbrefID": "other01", "birthYear": 1996, "birthMonth": 1, "birthDay": 1}])
    monkeypatch.setattr(
        module.lahman_data, "load_persisted_lahman_table", lambda raw_dir, name: people if name == "people" else None
    )

    pool = module.build_current_player_pool(str(raw_dir), season=2026, min_at_bats=5)

    assert pool.empty
