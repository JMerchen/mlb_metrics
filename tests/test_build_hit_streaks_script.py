"""End-to-end smoke test for scripts/build_hit_streaks.py's wiring:
synthetic persisted Statcast written to a tmp_path parquet, checked that
hit_streaks.csv gets written correctly, and that no persisted data leaves
prior output untouched (resilience), mirroring
test_build_dfs_rankings_script.py's shape - this script has no schedule
dependency, so it's simpler than that one."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from mlb_metrics import data

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_hit_streaks.py"


def _load_module():
    sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location("build_hit_streaks", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _no_network_name_register(monkeypatch):
    monkeypatch.setattr(
        data, "get_name_register",
        lambda: pd.DataFrame(columns=["key_mlbam", "key_bbref", "name_first", "name_last"]),
    )


def _game_rows(game_pk, date, batter, events, home_team="NYY", away_team="BOS"):
    rows = []
    for i, e in enumerate(events):
        rows.append({
            "game_pk": game_pk, "game_date": date, "pitcher": 99, "batter": batter,
            "events": e, "p_throws": "R", "inning_topbot": "Top",
            "home_team": home_team, "away_team": away_team,
            "at_bat_number": i + 1, "pitch_number": 1,
        })
    return rows


def test_build_hit_streaks_writes_csv(tmp_path):
    module = _load_module()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    # Batter 1: two recent games, both with a hit -> streak 2, and stays
    # inside the default recency window (game dates near "today").
    today = pd.Timestamp.today().normalize()
    rows = _game_rows(1, today - pd.Timedelta(days=2), 1, ["single", "field_out"])
    rows += _game_rows(2, today, 1, ["double", "field_out"])
    pd.DataFrame(rows).to_parquet(raw_dir / "statcast_2026.parquet", index=False)

    sys.argv = ["build_hit_streaks.py", "--raw-dir", str(raw_dir), "--data-dir", str(tmp_path / "data"), "--season", "2026"]
    module.main()

    result = pd.read_csv(tmp_path / "data" / "hit_streaks.csv")
    assert len(result) == 1
    assert result.iloc[0]["key_mlbam"] == 1
    assert result.iloc[0]["Current_Hit_Streak"] == 2
    assert result.iloc[0]["team"] == "BOS"


def test_build_hit_streaks_no_persisted_data_writes_nothing(tmp_path):
    module = _load_module()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    sys.argv = ["build_hit_streaks.py", "--raw-dir", str(raw_dir), "--data-dir", str(tmp_path / "data"), "--season", "2026"]
    module.main()

    assert not (tmp_path / "data" / "hit_streaks.csv").exists()
