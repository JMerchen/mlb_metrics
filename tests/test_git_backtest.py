import subprocess

import pandas as pd

from mlb_metrics import git_backtest


def _git(args, cwd):
    subprocess.run(["git"] + args, cwd=cwd, check=True, capture_output=True, text=True)


def _wave_csv(rows):
    return pd.DataFrame(
        [
            {
                "key_mlbam": key, "name_first": f"F{key}", "name_last": f"L{key}", "team": "NYY",
                "PA_L": 0, "PA_R": pa_r,
                "probability_L": 0, "probability_R": 0, "probability": 0,
                "Game_Hit_Probability": ghp, "Consistency": 0, "Approach": 0, "Expected_Bases": 0,
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

    # Day 1: player 2 has the higher probability but too few PA to qualify.
    _wave_csv([(1, 40, 0.80), (2, 5, 0.99)]).to_csv(wave_path, index=False)
    _git(["add", "docs/data/wave.csv"], repo)
    _git(["commit", "-q", "-m", "day1", "--date=2026-06-18T12:00:00"], repo)

    # Day 2: player 3 now leads, qualified.
    _wave_csv([(1, 40, 0.60), (3, 40, 0.85)]).to_csv(wave_path, index=False)
    _git(["add", "docs/data/wave.csv"], repo)
    _git(["commit", "-q", "-m", "day2", "--date=2026-06-19T12:00:00"], repo)

    return repo


def test_list_wave_csv_commits_returns_dates_oldest_first(tmp_path):
    repo = _init_repo_with_wave_history(tmp_path)
    commits = git_backtest.list_wave_csv_commits(repo_dir=str(repo))

    assert list(commits["date"]) == [pd.Timestamp("2026-06-18"), pd.Timestamp("2026-06-19")]


def test_reconstruct_historical_picks_replays_each_commit_through_select_picks(tmp_path):
    repo = _init_repo_with_wave_history(tmp_path)

    picks = git_backtest.reconstruct_historical_picks(
        repo_dir=str(repo), top_n=1, min_plate_appearances=30
    )

    assert len(picks) == 2
    day1_pick = picks[picks["date"] == "2026-06-18"].iloc[0]
    assert day1_pick["key_mlbam"] == 1  # player 2 excluded: only 5 PA
    day2_pick = picks[picks["date"] == "2026-06-19"].iloc[0]
    assert day2_pick["key_mlbam"] == 3
    assert (picks["actual_hit"].isna()).all()  # reconstruction only assigns picks, never outcomes
