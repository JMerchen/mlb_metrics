"""End-to-end smoke test for scripts/build_nfl_optimal_lineup.py's wiring:
synthetic nfl_qb.csv/nfl_skill.csv/nfl_dst.csv (as nfl_pipeline.py, Phase
8, would eventually write), checked that nfl_optimal_lineup.csv/
nfl_dfs_salary_pool.csv get written correctly, and that missing/empty
input or an infeasible solve leaves a prior file untouched (resilience),
mirroring test_build_optimal_lineup_script.py's exact pattern."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_nfl_optimal_lineup.py"


def _load_build_nfl_optimal_lineup_module():
    sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location("build_nfl_optimal_lineup", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _full_slate():
    qb_dk = pd.DataFrame([{"player_id": "qb1", "DK_Points_QB": 20.0}, {"player_id": "qb2", "DK_Points_QB": 18.0}])

    skill_rows = []
    layout = [("RB", 4), ("WR", 5), ("TE", 3)]
    for position, count in layout:
        for i in range(count):
            skill_rows.append({"player_id": f"{position.lower()}{i}", "position": position, "DK_Points_Skill": 5.0 + i})
    skill_dk = pd.DataFrame(skill_rows)

    dst_dk = pd.DataFrame([{"team": "SEA", "DK_Points_DST": 8.0}, {"team": "SF", "DK_Points_DST": 6.0}])

    return qb_dk, skill_dk, dst_dk


def _write_weekly_csvs(data_dir, qb_dk, skill_dk, dst_dk):
    qb_dk.to_csv(data_dir / "nfl_qb.csv", index=False)
    skill_dk.to_csv(data_dir / "nfl_skill.csv", index=False)
    dst_dk.to_csv(data_dir / "nfl_dst.csv", index=False)


def test_build_nfl_optimal_lineup_writes_lineup_and_pool(tmp_path, monkeypatch):
    module = _load_build_nfl_optimal_lineup_module()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    qb_dk, skill_dk, dst_dk = _full_slate()
    _write_weekly_csvs(data_dir, qb_dk, skill_dk, dst_dk)

    monkeypatch.setattr(sys, "argv", ["build_nfl_optimal_lineup.py", "--data-dir", str(data_dir)])
    module.main()

    lineup = pd.read_csv(data_dir / "nfl_optimal_lineup.csv")
    pool = pd.read_csv(data_dir / "nfl_dfs_salary_pool.csv")

    assert len(lineup) == 9
    counts = lineup["dk_slot"].value_counts().to_dict()
    assert counts == {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1, "DST": 1}
    assert not pool.empty


def test_build_nfl_optimal_lineup_missing_input_leaves_prior_output(tmp_path, monkeypatch, capsys):
    module = _load_build_nfl_optimal_lineup_module()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "nfl_optimal_lineup.csv").write_text("stale,data\n1,2\n")

    monkeypatch.setattr(sys, "argv", ["build_nfl_optimal_lineup.py", "--data-dir", str(data_dir)])
    module.main()

    assert (data_dir / "nfl_optimal_lineup.csv").read_text() == "stale,data\n1,2\n"
    assert "run the NFL weekly pipeline first" in capsys.readouterr().out


def test_build_nfl_optimal_lineup_empty_input_leaves_prior_output(tmp_path, monkeypatch, capsys):
    module = _load_build_nfl_optimal_lineup_module()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "nfl_optimal_lineup.csv").write_text("stale,data\n1,2\n")
    empty = pd.DataFrame(columns=["player_id", "position", "DK_Points_QB", "DK_Points_Skill"])
    _write_weekly_csvs(data_dir, empty.rename(columns={"DK_Points_Skill": "DK_Points_QB"})[["player_id", "DK_Points_QB"]], empty[["player_id", "position", "DK_Points_Skill"]], pd.DataFrame(columns=["team", "DK_Points_DST"]))

    monkeypatch.setattr(sys, "argv", ["build_nfl_optimal_lineup.py", "--data-dir", str(data_dir)])
    module.main()

    assert (data_dir / "nfl_optimal_lineup.csv").read_text() == "stale,data\n1,2\n"
    assert "No qualified QB/skill/DST rows" in capsys.readouterr().out


def test_build_nfl_optimal_lineup_infeasible_solve_leaves_prior_output(tmp_path, monkeypatch, capsys):
    module = _load_build_nfl_optimal_lineup_module()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "nfl_optimal_lineup.csv").write_text("stale,data\n1,2\n")

    # Only 1 RB candidate - RB:2 requirement can never be filled.
    qb_dk = pd.DataFrame([{"player_id": "qb1", "DK_Points_QB": 20.0}])
    skill_dk = pd.DataFrame([{"player_id": "rb1", "position": "RB", "DK_Points_Skill": 10.0}])
    dst_dk = pd.DataFrame([{"team": "SEA", "DK_Points_DST": 8.0}])
    _write_weekly_csvs(data_dir, qb_dk, skill_dk, dst_dk)

    monkeypatch.setattr(sys, "argv", ["build_nfl_optimal_lineup.py", "--data-dir", str(data_dir)])
    module.main()

    assert (data_dir / "nfl_optimal_lineup.csv").read_text() == "stale,data\n1,2\n"
    assert "could not fill a full lineup" in capsys.readouterr().out
