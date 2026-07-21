"""Reconstruct historical picks from the git history of docs/data/wave.csv.

The daily workflow has been committing wave.csv since the project started,
which means ~40+ days of past picks already exist in git history even
though nothing was ever logged or scored at the time. This module replays
that history through predictions.select_picks() to backfill a predictions
log without waiting for new data to accumulate.

Actually *resolving* those picks (whether the batter got a hit that day)
needs real outcome data for each historical date, which this module doesn't
fetch itself - pass in whatever source of completed-event data you have
(persisted raw Statcast, or a fresh per-date pybaseball pull) to
predictions.resolve_predictions() afterward. See scripts/run_backtest.py.
"""

import io
import subprocess

import pandas as pd

from mlb_metrics import config, predictions

WAVE_CSV_PATH = "docs/data/wave.csv"


def list_wave_csv_commits(repo_dir: str = ".", path: str = WAVE_CSV_PATH) -> pd.DataFrame:
    """Returns [commit, date] for every commit that touched `path`, oldest first."""
    result = subprocess.run(
        ["git", "log", "--follow", "--format=%H %ad", "--date=short", "--", path],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    rows = [line.split(" ", 1) for line in result.stdout.strip().splitlines() if line.strip()]
    commits = pd.DataFrame(rows, columns=["commit", "date"])
    commits["date"] = pd.to_datetime(commits["date"])
    return commits.sort_values("date").reset_index(drop=True)


def read_csv_at_commit(commit: str, path: str, repo_dir: str = ".") -> pd.DataFrame:
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"], cwd=repo_dir, capture_output=True, text=True, check=True
    )
    return pd.read_csv(io.StringIO(result.stdout))


def reconstruct_historical_picks(
    repo_dir: str = ".",
    top_n: int = config.BACKTEST_TOP_N,
    min_plate_appearances: int = config.BACKTEST_MIN_PLATE_APPEARANCES,
    metric: str = "Game_Hit_Probability",
) -> pd.DataFrame:
    """Replay every historical wave.csv commit through select_picks(),
    skipping commits whose schema predates the columns select_picks needs
    (i.e. anything from before this project had PA_L/PA_R/Game_Hit_Probability)."""
    commits = list_wave_csv_commits(repo_dir)
    required_columns = {"PA_L", "PA_R", metric, "key_mlbam", "name_first", "name_last"}

    all_picks = []
    for _, row in commits.iterrows():
        try:
            wave = read_csv_at_commit(row["commit"], WAVE_CSV_PATH, repo_dir)
        except subprocess.CalledProcessError:
            continue
        if not required_columns.issubset(wave.columns) or wave.empty:
            continue
        picks = predictions.select_picks(
            wave, row["date"], top_n=top_n, min_plate_appearances=min_plate_appearances, metric=metric
        )
        all_picks.append(picks)

    if not all_picks:
        return pd.DataFrame(columns=predictions.PREDICTION_COLUMNS)
    return pd.concat(all_picks, ignore_index=True)
