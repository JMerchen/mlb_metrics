"""End-to-end smoke test for scripts/build_nfl_bestball_rankings.py's
wiring: synthetic weekly_<season>.parquet/schedules_<season>.parquet (as
scripts/fetch_nfl_historical.py would persist), checked that
nfl_bestball_rankings.csv gets written correctly, that a prior-season
pair (when present) adds games_missed_prior_season, and that missing
input leaves nothing written rather than crashing."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_nfl_bestball_rankings.py"


def _load_module():
    sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location("build_nfl_bestball_rankings", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _weekly_row(player_id, position, season, week, name="Player One", team="NYJ", passing_yards=0, passing_tds=0, rushing_yards=0, rushing_tds=0, receptions=0, receiving_yards=0, receiving_tds=0, season_type="REG"):
    return {
        "player_id": player_id, "player_display_name": name, "position": position, "team": team,
        "season": season, "week": week, "season_type": season_type,
        "passing_yards": passing_yards, "passing_tds": passing_tds, "passing_interceptions": 0,
        "rushing_yards": rushing_yards, "rushing_tds": rushing_tds, "passing_2pt_conversions": 0,
        "receptions": receptions, "receiving_yards": receiving_yards, "receiving_tds": receiving_tds,
        "rushing_fumbles_lost": 0, "receiving_fumbles_lost": 0, "rushing_2pt_conversions": 0, "receiving_2pt_conversions": 0,
    }


def _schedule_row(season, week, home, away, game_type="REG"):
    return {"season": season, "week": week, "game_type": game_type, "home_team": home, "away_team": away}


def _persist(raw_dir, table, season, df):
    df.to_parquet(raw_dir / f"{table}_{season}.parquet", index=False)


def test_build_nfl_bestball_rankings_writes_csv_with_prior_season(tmp_path):
    module = _load_module()
    raw_dir = tmp_path / "raw"
    data_dir = tmp_path / "data"
    raw_dir.mkdir()

    _persist(raw_dir, "weekly", 2025, pd.DataFrame([_weekly_row("qb1", "QB", 2025, 1, passing_yards=250, passing_tds=2)]))
    _persist(raw_dir, "schedules", 2025, pd.DataFrame([_schedule_row(2025, 1, "NYJ", "BUF")]))
    _persist(raw_dir, "weekly", 2024, pd.DataFrame([_weekly_row("qb1", "QB", 2024, 1, passing_yards=200, passing_tds=1)]))
    _persist(raw_dir, "schedules", 2024, pd.DataFrame([
        _schedule_row(2024, 1, "NYJ", "BUF"),
        _schedule_row(2024, 2, "NYJ", "NE"),  # a real second 2024 game qb1 missed
    ]))

    sys.argv = ["build_nfl_bestball_rankings.py", "--raw-dir", str(raw_dir), "--data-dir", str(data_dir), "--season", "2025"]
    module.main()

    result = pd.read_csv(data_dir / "nfl_bestball_rankings.csv")
    assert len(result) == 1
    assert result.iloc[0]["player_id"] == "qb1"
    assert result.iloc[0]["games_missed_prior_season"] == 1

    scarcity = pd.read_csv(data_dir / "nfl_position_scarcity.csv")
    assert set(scarcity["position"]) == {"QB", "RB", "WR", "TE"}
    qb_row = scarcity.set_index("position").loc["QB"]
    assert qb_row["total_players"] == 1
    assert qb_row["qualified_players"] == 0  # 1 game played, below the default min_games qualifier


def test_build_nfl_bestball_rankings_missing_season_data_writes_nothing(tmp_path):
    module = _load_module()
    raw_dir = tmp_path / "raw"
    data_dir = tmp_path / "data"
    raw_dir.mkdir()

    sys.argv = ["build_nfl_bestball_rankings.py", "--raw-dir", str(raw_dir), "--data-dir", str(data_dir), "--season", "2025"]
    module.main()

    assert not (data_dir / "nfl_bestball_rankings.csv").exists()
    assert not (data_dir / "nfl_position_scarcity.csv").exists()


def test_build_nfl_bestball_rankings_missing_prior_season_still_writes_without_that_column(tmp_path):
    module = _load_module()
    raw_dir = tmp_path / "raw"
    data_dir = tmp_path / "data"
    raw_dir.mkdir()

    _persist(raw_dir, "weekly", 2025, pd.DataFrame([_weekly_row("qb1", "QB", 2025, 1, passing_yards=250, passing_tds=2)]))
    _persist(raw_dir, "schedules", 2025, pd.DataFrame([_schedule_row(2025, 1, "NYJ", "BUF")]))
    # No 2024 files persisted at all.

    sys.argv = ["build_nfl_bestball_rankings.py", "--raw-dir", str(raw_dir), "--data-dir", str(data_dir), "--season", "2025"]
    module.main()

    result = pd.read_csv(data_dir / "nfl_bestball_rankings.csv")
    assert len(result) == 1
    assert "games_missed_prior_season" not in result.columns
