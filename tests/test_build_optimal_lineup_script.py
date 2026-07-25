"""End-to-end smoke test for scripts/build_optimal_lineup.py's wiring:
synthetic dfs_hitters.csv/dfs_pitchers.csv (as scripts/build_dfs_rankings.py
would have just written) + a monkeypatched position-eligibility fetch,
checked that optimal_lineup.csv/dfs_salary_pool.csv get written correctly,
and that a failed/empty fetch or an infeasible solve leaves a prior file
untouched (resilience), mirroring test_build_dfs_rankings_script.py's
exact pattern."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_optimal_lineup.py"


def _load_build_optimal_lineup_module():
    sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location("build_optimal_lineup", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _full_slate():
    slot_layout = [("C", 2), ("1B", 2), ("2B", 2), ("3B", 2), ("SS", 2), ("OF", 5)]
    hitter_rows = []
    slot_by_key = {}
    key = 1
    for slot, count in slot_layout:
        for i in range(count):
            hitter_rows.append({
                "key_mlbam": key, "name_first": "H", "name_last": str(key), "team": "BOS", "opponent": "NYY",
                "is_home": True, "PA_L": 20, "PA_R": 20, "Expected_Bases": 1.5, "Game_Hit_Probability": 0.7,
                "Matchup_Hit_Probability": 0.7, "Matchup_Ratio": 1.0, "Adjusted_Expected_Bases": 1.5,
                "DK_Points_Hitter": 3.0 + i * 0.5,
            })
            slot_by_key[key] = slot
            key += 1

    pitcher_rows = []
    for i in range(3):
        pitcher_rows.append({
            "key_mlbam": 900 + i, "name_first": "P", "name_last": str(i), "team": "NYY", "opponent": "BOS",
            "is_home": False, "starts": 5, "K9": 9.0, "BB9": 3.0, "HR9": 1.0, "IP_per_start": 6.0,
            "Expected_IP": 6.0, "Expected_K": 6.0, "Expected_BB": 2.0, "Expected_H_Allowed": 5.0,
            "FIP_Windowed": 4.0, "Expected_ER": 2.0, "DK_Points_Pitcher": 10.0 + i,
        })

    return pd.DataFrame(hitter_rows), pd.DataFrame(pitcher_rows), slot_by_key


def _write_daily_csvs(data_dir, hitters, pitchers):
    hitters.to_csv(data_dir / "dfs_hitters.csv", index=False)
    pitchers.to_csv(data_dir / "dfs_pitchers.csv", index=False)


def _eligibility_fetcher(slot_by_key):
    def fetch(key_mlbams):
        return pd.DataFrame([
            {"key_mlbam": k, "primary_position": slot_by_key[k], "dk_slot": slot_by_key[k]}
            for k in key_mlbams if k in slot_by_key
        ])
    return fetch


def test_build_optimal_lineup_writes_lineup_and_pool(tmp_path, monkeypatch):
    module = _load_build_optimal_lineup_module()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    hitters, pitchers, slot_by_key = _full_slate()
    _write_daily_csvs(data_dir, hitters, pitchers)

    monkeypatch.setattr(module.roster_positions, "fetch_position_eligibility", _eligibility_fetcher(slot_by_key))

    sys.argv = ["build_optimal_lineup.py", "--data-dir", str(data_dir)]
    module.main()

    lineup = pd.read_csv(data_dir / "optimal_lineup.csv")
    pool = pd.read_csv(data_dir / "dfs_salary_pool.csv")
    assert len(lineup) == 10
    assert lineup["Estimated_Salary"].sum() <= module.config.DFS_SALARY_CAP
    assert not pool.empty
    assert "Estimated_Salary" in lineup.columns and "Salary" not in lineup.columns


def test_build_optimal_lineup_missing_daily_csvs_writes_nothing(tmp_path):
    module = _load_build_optimal_lineup_module()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    sys.argv = ["build_optimal_lineup.py", "--data-dir", str(data_dir)]
    module.main()

    assert not (data_dir / "optimal_lineup.csv").exists()


def test_build_optimal_lineup_failed_eligibility_fetch_leaves_existing_file_untouched(tmp_path, monkeypatch):
    module = _load_build_optimal_lineup_module()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    hitters, pitchers, slot_by_key = _full_slate()
    _write_daily_csvs(data_dir, hitters, pitchers)
    (data_dir / "optimal_lineup.csv").write_text("stale,data\n1,2\n")

    def _boom(key_mlbams):
        raise RuntimeError("statsapi is down")

    monkeypatch.setattr(module.roster_positions, "fetch_position_eligibility", _boom)

    sys.argv = ["build_optimal_lineup.py", "--data-dir", str(data_dir)]
    module.main()

    assert (data_dir / "optimal_lineup.csv").read_text() == "stale,data\n1,2\n"


def test_build_optimal_lineup_infeasible_solve_leaves_existing_file_untouched(tmp_path, monkeypatch):
    module = _load_build_optimal_lineup_module()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # Only 1 OF-eligible hitter total - needs 3, infeasible.
    hitters = pd.DataFrame([{
        "key_mlbam": 1, "name_first": "H", "name_last": "1", "team": "BOS", "opponent": "NYY",
        "is_home": True, "PA_L": 20, "PA_R": 20, "Expected_Bases": 1.5, "Game_Hit_Probability": 0.7,
        "Matchup_Hit_Probability": 0.7, "Matchup_Ratio": 1.0, "Adjusted_Expected_Bases": 1.5,
        "DK_Points_Hitter": 3.0,
    }])
    pitchers = pd.DataFrame(columns=[
        "key_mlbam", "name_first", "name_last", "team", "opponent", "is_home", "starts", "K9", "BB9", "HR9",
        "IP_per_start", "Expected_IP", "Expected_K", "Expected_BB", "Expected_H_Allowed", "FIP_Windowed",
        "Expected_ER", "DK_Points_Pitcher",
    ])
    _write_daily_csvs(data_dir, hitters, pitchers)
    (data_dir / "optimal_lineup.csv").write_text("stale,data\n1,2\n")

    monkeypatch.setattr(module.roster_positions, "fetch_position_eligibility", _eligibility_fetcher({1: "OF"}))

    sys.argv = ["build_optimal_lineup.py", "--data-dir", str(data_dir)]
    module.main()

    assert (data_dir / "optimal_lineup.csv").read_text() == "stale,data\n1,2\n"
