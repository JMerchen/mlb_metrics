"""End-to-end smoke test for scripts/build_dfs_rankings.py's wiring: a
synthetic wave.csv/pave.csv/confidence.csv (as scripts/wave.py would have
just written) + synthetic persisted Statcast + a monkeypatched schedule
fetch, checked that dfs_hitters.csv/dfs_pitchers.csv get written
correctly, and that a failed/empty schedule fetch leaves no new file
(resilience), mirroring test_build_age_curves_script.py's exact pattern."""

import datetime
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_dfs_rankings.py"


def _load_build_dfs_rankings_module():
    sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location("build_dfs_rankings", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_daily_csvs(data_dir):
    wave = pd.DataFrame([{
        "key_mlbam": 1, "name_first": "Test", "name_last": "Hitter", "team": "BOS",
        "PA_L": 20, "PA_R": 20, "WAVE": 0.28, "WAVE_L": 0.28, "WAVE_R": 0.28,
        "probability_L": 0.6, "probability_R": 0.6, "probability": 0.6,
        "Game_Hit_Probability": 0.70, "Consistency": 0.1, "Approach": 0.4, "Expected_Bases": 1.5,
        "Expected_BB": 0.3, "Expected_HBP": 0.1, "Expected_RBI": 0.4,
        "Exit_Velo": 90.0, "Barrel_Rate": 0.08, "xBA": 0.25, "xwOBA": 0.32,
    }])
    pave = pd.DataFrame([{
        "key_mlbam": 99, "name_first": "Test", "name_last": "Pitcher", "team": "NYY",
        "at_bats": 100, "Throws": "R", "PAVE": 0.24, "PAVE_PLUS": 1.0, "Power_A_PLUS": 1.0,
        "Expected_Hits": 1.0, "Expected_Bases": 1.5, "Expected_HRs": 0.1,
    }])
    confidence = pd.DataFrame([
        {"team": "BOS", "Bullpen_PAVE": 0.25, "Park_Factor": 1.0, "team_bases_pg": 5.0},
        {"team": "NYY", "Bullpen_PAVE": 0.25, "Park_Factor": 1.0, "team_bases_pg": 5.0},
    ])
    wave.to_csv(data_dir / "wave.csv", index=False)
    pave.to_csv(data_dir / "pave.csv", index=False)
    confidence.to_csv(data_dir / "confidence.csv", index=False)


def _write_persisted_statcast(raw_dir, n_starts=1):
    rows = []
    for start in range(n_starts):
        events = ["strikeout"] * 20 + ["field_out"] * 10 + ["walk"] * 5 + ["single"] * 10
        for i, e in enumerate(events):
            rows.append({
                "game_pk": start + 1, "game_date": pd.Timestamp("2026-06-01") + pd.Timedelta(days=start * 5),
                "pitcher": 99, "batter": 1, "events": e, "p_throws": "R",
                "inning_topbot": "Top", "home_team": "NYY", "away_team": "BOS",
                "at_bat_number": i + 1, "pitch_number": 1,
                "bat_score": 0, "post_bat_score": 0,
            })
    pd.DataFrame(rows).to_parquet(raw_dir / "statcast_2026.parquet", index=False)


def _schedule_df():
    return pd.DataFrame([
        {"date": "2026-06-20", "team": "BOS", "opponent": "NYY", "probable_pitcher_key_mlbam": 99, "game_pk": 1, "is_home": False},
        {"date": "2026-06-20", "team": "NYY", "opponent": "BOS", "probable_pitcher_key_mlbam": pd.NA, "game_pk": 1, "is_home": True},
    ])


def test_build_dfs_rankings_writes_both_csvs(tmp_path, monkeypatch):
    module = _load_build_dfs_rankings_module()

    raw_dir = tmp_path / "raw"
    data_dir = tmp_path / "data"
    raw_dir.mkdir()
    data_dir.mkdir()
    _write_daily_csvs(data_dir)
    _write_persisted_statcast(raw_dir, n_starts=4)  # clears DFS_PITCHER_MIN_STARTS

    monkeypatch.setattr(module.schedule, "fetch_probable_pitchers", lambda date: _schedule_df())

    sys.argv = [
        "build_dfs_rankings.py", "--raw-dir", str(raw_dir), "--data-dir", str(data_dir),
        "--season", "2026", "--as-of-date", "2026-06-20",
    ]
    module.main()

    hitters = pd.read_csv(data_dir / "dfs_hitters.csv")
    pitchers = pd.read_csv(data_dir / "dfs_pitchers.csv")
    assert len(hitters) == 1
    assert hitters.iloc[0]["key_mlbam"] == 1
    assert hitters.iloc[0]["DK_Points_Hitter"] > 0
    assert len(pitchers) == 1
    assert pitchers.iloc[0]["key_mlbam"] == 99
    assert pitchers.iloc[0]["DK_Points_Pitcher"] != 0


def test_build_dfs_rankings_missing_daily_csvs_writes_nothing(tmp_path, monkeypatch):
    module = _load_build_dfs_rankings_module()

    raw_dir = tmp_path / "raw"
    data_dir = tmp_path / "data"
    raw_dir.mkdir()
    data_dir.mkdir()
    # No wave.csv/pave.csv/confidence.csv written.

    sys.argv = [
        "build_dfs_rankings.py", "--raw-dir", str(raw_dir), "--data-dir", str(data_dir),
        "--season", "2026", "--as-of-date", "2026-06-20",
    ]
    module.main()

    assert not (data_dir / "dfs_hitters.csv").exists()
    assert not (data_dir / "dfs_pitchers.csv").exists()


def test_build_dfs_rankings_failed_schedule_fetch_leaves_existing_files_untouched(tmp_path, monkeypatch):
    module = _load_build_dfs_rankings_module()

    raw_dir = tmp_path / "raw"
    data_dir = tmp_path / "data"
    raw_dir.mkdir()
    data_dir.mkdir()
    _write_daily_csvs(data_dir)
    _write_persisted_statcast(raw_dir, n_starts=4)

    # Simulate a stale file from yesterday's run that must survive a failed fetch.
    (data_dir / "dfs_hitters.csv").write_text("stale,data\n1,2\n")

    def _boom(date):
        raise RuntimeError("statsapi is down")

    monkeypatch.setattr(module.schedule, "fetch_probable_pitchers", _boom)

    sys.argv = [
        "build_dfs_rankings.py", "--raw-dir", str(raw_dir), "--data-dir", str(data_dir),
        "--season", "2026", "--as-of-date", "2026-06-20",
    ]
    module.main()

    assert (data_dir / "dfs_hitters.csv").read_text() == "stale,data\n1,2\n"
    assert not (data_dir / "dfs_pitchers.csv").exists()


def test_build_dfs_rankings_empty_schedule_leaves_existing_files_untouched(tmp_path, monkeypatch):
    module = _load_build_dfs_rankings_module()

    raw_dir = tmp_path / "raw"
    data_dir = tmp_path / "data"
    raw_dir.mkdir()
    data_dir.mkdir()
    _write_daily_csvs(data_dir)
    _write_persisted_statcast(raw_dir, n_starts=4)
    (data_dir / "dfs_hitters.csv").write_text("stale,data\n1,2\n")

    monkeypatch.setattr(
        module.schedule, "fetch_probable_pitchers",
        lambda date: pd.DataFrame(columns=["date", "team", "opponent", "probable_pitcher_key_mlbam", "game_pk", "is_home"]),
    )

    sys.argv = [
        "build_dfs_rankings.py", "--raw-dir", str(raw_dir), "--data-dir", str(data_dir),
        "--season", "2026", "--as-of-date", "2026-06-20",
    ]
    module.main()

    assert (data_dir / "dfs_hitters.csv").read_text() == "stale,data\n1,2\n"


def test_build_dfs_rankings_merges_dk_slot_from_position_eligibility(tmp_path, monkeypatch):
    module = _load_build_dfs_rankings_module()

    raw_dir = tmp_path / "raw"
    data_dir = tmp_path / "data"
    raw_dir.mkdir()
    data_dir.mkdir()
    _write_daily_csvs(data_dir)
    _write_persisted_statcast(raw_dir, n_starts=4)

    monkeypatch.setattr(module.schedule, "fetch_probable_pitchers", lambda date: _schedule_df())
    monkeypatch.setattr(
        module.roster_positions, "fetch_position_eligibility",
        lambda key_mlbams: pd.DataFrame([{"key_mlbam": 1, "primary_position": "SS", "dk_slot": "SS"}]),
    )

    sys.argv = [
        "build_dfs_rankings.py", "--raw-dir", str(raw_dir), "--data-dir", str(data_dir),
        "--season", "2026", "--as-of-date", "2026-06-20",
    ]
    module.main()

    hitters = pd.read_csv(data_dir / "dfs_hitters.csv")
    assert hitters.iloc[0]["dk_slot"] == "SS"


def test_build_dfs_rankings_failed_position_fetch_still_writes_hitters(tmp_path, monkeypatch):
    # Unlike a failed schedule fetch (which blocks the whole file), a
    # failed position-eligibility fetch is purely informational - today's
    # rankings should still publish, just without a usable dk_slot column.
    module = _load_build_dfs_rankings_module()

    raw_dir = tmp_path / "raw"
    data_dir = tmp_path / "data"
    raw_dir.mkdir()
    data_dir.mkdir()
    _write_daily_csvs(data_dir)
    _write_persisted_statcast(raw_dir, n_starts=4)

    monkeypatch.setattr(module.schedule, "fetch_probable_pitchers", lambda date: _schedule_df())

    def _boom(key_mlbams):
        raise RuntimeError("statsapi is down")

    monkeypatch.setattr(module.roster_positions, "fetch_position_eligibility", _boom)

    sys.argv = [
        "build_dfs_rankings.py", "--raw-dir", str(raw_dir), "--data-dir", str(data_dir),
        "--season", "2026", "--as-of-date", "2026-06-20",
    ]
    module.main()

    hitters = pd.read_csv(data_dir / "dfs_hitters.csv")
    assert len(hitters) == 1
    assert pd.isna(hitters.iloc[0]["dk_slot"])


def test_build_dfs_rankings_excludes_hitter_who_has_not_played_recently(tmp_path, monkeypatch):
    # A season-long injured-list stay (e.g. Dansby Swanson) leaves
    # season-to-date PA/rates looking qualified, but the hitter hasn't
    # actually played in months - must be excluded from dfs_hitters.csv
    # (and therefore from the optimizer, which reads this file directly).
    module = _load_build_dfs_rankings_module()

    raw_dir = tmp_path / "raw"
    data_dir = tmp_path / "data"
    raw_dir.mkdir()
    data_dir.mkdir()
    _write_daily_csvs(data_dir)
    # Only 1 "start" - last (and only) game_date is 2026-06-01, 19 days
    # before the 2026-06-20 as-of-date used below - well past
    # HITTER_MAX_DAYS_SINCE_LAST_GAME (5).
    _write_persisted_statcast(raw_dir, n_starts=1)

    monkeypatch.setattr(module.schedule, "fetch_probable_pitchers", lambda date: _schedule_df())

    sys.argv = [
        "build_dfs_rankings.py", "--raw-dir", str(raw_dir), "--data-dir", str(data_dir),
        "--season", "2026", "--as-of-date", "2026-06-20",
    ]
    module.main()

    hitters = pd.read_csv(data_dir / "dfs_hitters.csv")
    assert hitters.empty


def test_build_dfs_rankings_keeps_hitter_within_recency_window(tmp_path, monkeypatch):
    # Boundary check: exactly at the 5-day cutoff must still be included
    # (predictions.select_picks's own gate uses the same <= semantics).
    module = _load_build_dfs_rankings_module()

    raw_dir = tmp_path / "raw"
    data_dir = tmp_path / "data"
    raw_dir.mkdir()
    data_dir.mkdir()
    _write_daily_csvs(data_dir)

    rows = []
    events = ["strikeout"] * 20 + ["field_out"] * 10 + ["walk"] * 5 + ["single"] * 10
    for i, e in enumerate(events):
        rows.append({
            "game_pk": 1, "game_date": pd.Timestamp("2026-06-15"),  # exactly 5 days before as-of-date
            "pitcher": 99, "batter": 1, "events": e, "p_throws": "R",
            "inning_topbot": "Top", "home_team": "NYY", "away_team": "BOS",
            "at_bat_number": i + 1, "pitch_number": 1,
            "bat_score": 0, "post_bat_score": 0,
        })
    pd.DataFrame(rows).to_parquet(raw_dir / "statcast_2026.parquet", index=False)

    monkeypatch.setattr(module.schedule, "fetch_probable_pitchers", lambda date: _schedule_df())

    sys.argv = [
        "build_dfs_rankings.py", "--raw-dir", str(raw_dir), "--data-dir", str(data_dir),
        "--season", "2026", "--as-of-date", "2026-06-20",
    ]
    module.main()

    hitters = pd.read_csv(data_dir / "dfs_hitters.csv")
    assert len(hitters) == 1


def test_build_dfs_rankings_below_min_starts_excludes_pitcher(tmp_path, monkeypatch):
    module = _load_build_dfs_rankings_module()

    raw_dir = tmp_path / "raw"
    data_dir = tmp_path / "data"
    raw_dir.mkdir()
    data_dir.mkdir()
    _write_daily_csvs(data_dir)
    _write_persisted_statcast(raw_dir, n_starts=1)  # below DFS_PITCHER_MIN_STARTS

    monkeypatch.setattr(module.schedule, "fetch_probable_pitchers", lambda date: _schedule_df())

    sys.argv = [
        "build_dfs_rankings.py", "--raw-dir", str(raw_dir), "--data-dir", str(data_dir),
        "--season", "2026", "--as-of-date", "2026-06-20",
    ]
    module.main()

    pitchers = pd.read_csv(data_dir / "dfs_pitchers.csv")
    assert pitchers.empty
