"""Unit tests for scripts/backfill_statcast_season.py's core chunking/
resilience logic - injected fake fetch_fn/persist_fn/load_fn, no real
network or disk, mirroring tests/test_backfill_market_odds.py's own
established importlib.util module-loading pattern for scripts/ files."""

import datetime
import importlib.util
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "backfill_statcast_season.py"


def _load_module():
    sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location("backfill_statcast_season", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_month_chunks_splits_a_range_spanning_three_months():
    module = _load_module()
    chunks = list(module.month_chunks(datetime.date(2025, 3, 18), datetime.date(2025, 5, 10)))

    assert chunks == [
        (datetime.date(2025, 3, 18), datetime.date(2025, 3, 31)),
        (datetime.date(2025, 4, 1), datetime.date(2025, 4, 30)),
        (datetime.date(2025, 5, 1), datetime.date(2025, 5, 10)),
    ]


def test_month_chunks_single_month_range_is_one_chunk():
    module = _load_module()
    chunks = list(module.month_chunks(datetime.date(2025, 9, 5), datetime.date(2025, 9, 28)))
    assert chunks == [(datetime.date(2025, 9, 5), datetime.date(2025, 9, 28))]


def test_chunk_already_covered_true_only_when_every_day_has_a_real_row():
    module = _load_module()
    chunk_start, chunk_end = datetime.date(2025, 4, 1), datetime.date(2025, 4, 3)

    full_coverage = pd.DataFrame({"game_date": pd.to_datetime(["2025-04-01", "2025-04-02", "2025-04-03"])})
    assert module.chunk_already_covered(full_coverage, chunk_start, chunk_end) is True

    # A real off day (no games, no rows) on 2025-04-02 means this chunk is
    # NOT fully covered - must be re-fetched, not silently skipped.
    partial_coverage = pd.DataFrame({"game_date": pd.to_datetime(["2025-04-01", "2025-04-03"])})
    assert module.chunk_already_covered(partial_coverage, chunk_start, chunk_end) is False

    assert module.chunk_already_covered(None, chunk_start, chunk_end) is False
    assert module.chunk_already_covered(pd.DataFrame(columns=["game_date"]), chunk_start, chunk_end) is False


def test_backfill_statcast_season_skips_a_month_already_fully_persisted():
    module = _load_module()
    fetch_calls = []

    def fake_fetch(start, end):
        fetch_calls.append((start, end))
        return pd.DataFrame({"game_date": pd.to_datetime([str(start)]), "game_pk": [1]})

    # March is already fully covered by "persisted" data; April is not.
    persisted_state = {"df": pd.DataFrame({
        "game_date": pd.to_datetime([f"2025-03-{d:02d}" for d in range(18, 32)]),
        "game_pk": list(range(14)),
    })}

    def fake_load(raw_dir, season):
        return persisted_state["df"]

    def fake_persist(df, raw_dir, season):
        persisted_state["df"] = pd.concat([persisted_state["df"], df], ignore_index=True)
        return persisted_state["df"]

    module.backfill_statcast_season(
        "unused", 2025, datetime.date(2025, 3, 18), datetime.date(2025, 4, 30),
        fetch_fn=fake_fetch, persist_fn=fake_persist, load_fn=fake_load,
    )

    # Only April was actually fetched - March's real coverage was
    # correctly detected and skipped, saving real network time.
    assert fetch_calls == [(datetime.date(2025, 4, 1), datetime.date(2025, 4, 30))]


def test_backfill_statcast_season_one_bad_chunk_does_not_block_the_others():
    module = _load_module()
    fetch_calls = []
    persisted_state = {"df": None}

    def fake_fetch(start, end):
        fetch_calls.append((start, end))
        if start.month == 4:
            raise RuntimeError("Statcast is unreachable")
        return pd.DataFrame({"game_date": pd.to_datetime([str(start)]), "game_pk": [1]})

    def fake_load(raw_dir, season):
        return persisted_state["df"]

    def fake_persist(df, raw_dir, season):
        existing = persisted_state["df"]
        persisted_state["df"] = pd.concat([existing, df], ignore_index=True) if existing is not None else df
        return persisted_state["df"]

    result = module.backfill_statcast_season(
        "unused", 2025, datetime.date(2025, 3, 18), datetime.date(2025, 5, 10),
        fetch_fn=fake_fetch, persist_fn=fake_persist, load_fn=fake_load,
    )

    # March and May both attempted despite April's real failure - the one
    # bad chunk didn't stop the rest.
    assert len(fetch_calls) == 3
    assert result is not None
    assert set(pd.to_datetime(result["game_date"]).dt.month) == {3, 5}


def test_backfill_statcast_season_empty_fetch_result_is_not_persisted():
    module = _load_module()
    persist_calls = []

    def fake_fetch(start, end):
        return pd.DataFrame(columns=["game_date", "game_pk"])  # a real all-off-days month

    def fake_persist(df, raw_dir, season):
        persist_calls.append(df)
        return df

    module.backfill_statcast_season(
        "unused", 2025, datetime.date(2025, 3, 18), datetime.date(2025, 3, 31),
        fetch_fn=fake_fetch, persist_fn=fake_persist, load_fn=lambda raw_dir, season: None,
    )

    assert persist_calls == []
