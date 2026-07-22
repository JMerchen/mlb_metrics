"""End-to-end smoke test for scripts/run_backtest.py: reconstruct picks from
a scratch git repo's wave.csv history, resolve them against a synthetic
persisted raw dataset, and confirm a sane summary comes out the other end."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_backtest.py"


def _load_run_backtest_module():
    sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location("run_backtest", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(args, cwd):
    subprocess.run(["git"] + args, cwd=cwd, check=True, capture_output=True, text=True)


def _wave_csv(rows):
    # probability mirrors Game_Hit_Probability so these fixtures clear
    # select_picks' joint HITTER_MIN_PROBABILITY gate on both columns.
    return pd.DataFrame(
        [
            {
                "key_mlbam": key, "name_first": f"F{key}", "name_last": f"L{key}", "team": "NYY",
                "PA_L": 0, "PA_R": pa_r,
                "probability_L": 0, "probability_R": 0, "probability": ghp,
                "Game_Hit_Probability": ghp, "Consistency": 0, "Approach": ghp * ghp, "Expected_Bases": 0,
            }
            for key, pa_r, ghp in rows
        ]
    )


def _init_repo_with_wave_history(tmp_path):
    repo = tmp_path / "repo"
    data_dir = repo / "docs" / "data"
    data_dir.mkdir(parents=True)
    wave_path = data_dir / "wave.csv"

    _git(["init", "-q"], repo)
    _git(["config", "user.email", "test@example.com"], repo)
    _git(["config", "user.name", "Test"], repo)

    # 06-18: player 1 picked, gets a hit. 06-19: player 1 picked again, no hit.
    _wave_csv([(1, 40, 0.80)]).to_csv(wave_path, index=False)
    _git(["add", "docs/data/wave.csv"], repo)
    _git(["commit", "-q", "-m", "day1", "--date=2026-06-18T12:00:00"], repo)

    _wave_csv([(1, 40, 0.70)]).to_csv(wave_path, index=False)
    _git(["add", "docs/data/wave.csv"], repo)
    _git(["commit", "-q", "-m", "day2", "--date=2026-06-19T12:00:00"], repo)

    return repo


def test_run_backtest_reconstructs_resolves_and_summarizes(tmp_path, monkeypatch):
    run_backtest = _load_run_backtest_module()

    repo = _init_repo_with_wave_history(tmp_path)
    raw_dir = tmp_path / "raw"
    predictions_log = tmp_path / "predictions.csv"
    summary_out = tmp_path / "backtest_summary.csv"

    # Persisted outcome data covering both picked dates: a hit on 06-18, no hit on 06-19.
    raw_dir.mkdir()
    outcomes = pd.DataFrame({
        "game_pk": [1, 2],
        "at_bat_number": [1, 1],
        "pitch_number": [1, 1],
        "game_date": pd.to_datetime(["2026-06-18", "2026-06-19"]),
        "batter": [1, 1],
        "events": ["single", "field_out"],
    })
    outcomes.to_parquet(raw_dir / f"statcast_{run_backtest.config.SEASON_START.year}.parquet", index=False)

    monkeypatch.setattr(
        sys, "argv",
        [
            "run_backtest.py",
            "--repo-dir", str(repo),
            "--raw-dir", str(raw_dir),
            "--predictions-log", str(predictions_log),
            "--summary-out", str(summary_out),
            "--docs-data-dir", str(tmp_path / "docs_data"),
            "--top-n", "1",
            "--min-plate-appearances", "30",
        ],
    )

    run_backtest.main()

    log = pd.read_csv(predictions_log, parse_dates=["date"])
    assert len(log) == 2
    assert log[log["date"] == "2026-06-18"].iloc[0]["actual_hit"] == 1
    assert log[log["date"] == "2026-06-19"].iloc[0]["actual_hit"] == 0

    summary = pd.read_csv(summary_out)
    assert summary.loc[0, "n_resolved"] == 2
    assert summary.loc[0, "any_of_top_1_hit_rate"] == pytest.approx(0.5)

    docs_data_dir = tmp_path / "docs_data"
    streak_picks = pd.read_csv(docs_data_dir / "beat_the_streak_picks.csv")
    # 06-18's pick (0.80 predicted) clears the default 0.80 "good matchup"
    # bar and gets a hit; 06-19's (0.70) doesn't clear the bar at all, so
    # that day is a no-op with zero recommended picks - surfaced as its own
    # explicit "no_pick" row rather than being silently absent.
    assert len(streak_picks) == 2
    hit_row = streak_picks[streak_picks["date"] == "2026-06-18"].iloc[0]
    assert hit_row["status"] == "hit"
    no_pick_row = streak_picks[streak_picks["date"] == "2026-06-19"].iloc[0]
    assert no_pick_row["status"] == "no_pick"
    streak_summary = pd.read_csv(docs_data_dir / "beat_the_streak_summary.csv")
    assert streak_summary.loc[0, "n_days_resolved"] == 1
    assert streak_summary.loc[0, "current_streak"] == 1
    assert streak_summary.loc[0, "longest_streak"] == 1
