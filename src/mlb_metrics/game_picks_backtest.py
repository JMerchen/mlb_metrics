"""Reconstruct historical Automated Game Picks (see game_picks.py) from the
git history of docs/data/confidence.csv/pave.csv, resolved immediately
against the ACTUAL games/starters/scores already contained in persisted
Statcast data (data/raw/) - not statsapi's "probable" pitcher announcements
or a live final-score fetch, neither of which was ever persisted to git
history (see schedule.py's module docstring, and the README's "no
historical backtest" limitation for the live system).

Using the actual starter in place of the announced "probable" one is a
reasonable stand-in for backtesting purposes - the two agree the vast
majority of the time, and this is retrospective analysis of games that
already happened, so there's no "probable vs. actual" distinction left to
make. The final score is the real final score either way.

Game identity here is Statcast's own `game_pk` column (MLB's real game id,
the same ID space as schedule.py's live `gamePk`) - deliberately NOT
data.assign_game_ids' custom reconstruction, which exists for other
metrics' own historical reasons and has its own at-bat-boundary quirks at
the edges of whatever date range gets passed to it (confirmed empirically:
it can fragment a single real game across two ids, or conflate two
different games under one). `game_pk` already correctly identifies real
games - including real doubleheaders - so reusing it directly is both
simpler and more reliable for this purely backward-looking reconstruction.

The daily workflow has been committing confidence.csv/pave.csv (in the
same commit as wave.csv) since the project started, each one already in
the exact "as-of-that-date" snapshot shape compute_game_win_probabilities
needs - this is the same git-history-replay technique git_backtest.py
already uses for hitter picks (see list_wave_csv_commits/read_csv_at_commit,
reused here directly), applied to the team-level side. Unlike live picks
(pending until a later run resolves them), a backtested pick's outcome is
already known at reconstruction time, so there's no pending state at all.
"""

import subprocess

import pandas as pd

from mlb_metrics import config, data, game_picks, game_predictions, git_backtest

CONFIDENCE_CSV_PATH = "docs/data/confidence.csv"
PAVE_CSV_PATH = "docs/data/pave.csv"

REQUIRED_CONFIDENCE_COLUMNS = {"team", "pyth_Strength", "pyth_Confidence", "suppression_resistance", "true_power"}
REQUIRED_PAVE_COLUMNS = {"key_mlbam", "PAVE_PLUS"}


def derive_historical_schedule_games(persisted_statcast: pd.DataFrame) -> pd.DataFrame:
    """One row per historical game, in the same shape as
    schedule.normalize_schedule_games's output: [game_pk, date, home_team,
    away_team, home_probable_pitcher_key_mlbam,
    away_probable_pitcher_key_mlbam, status, home_score, away_score] - but
    "probable" here means whoever actually started, and the score is the
    actual final score, both reconstructed from already-persisted Statcast
    keyed on its own real `game_pk` column (see module docstring)."""
    data_with_game_id = persisted_statcast.rename(columns={"game_pk": "game_id"})
    roles = data.label_pitcher_roles(data_with_game_id)
    starters = roles[roles["is_starter"]][["game_id", "team", "pitcher"]]

    results = data.extract_game_results(data_with_game_id)
    # MLB games cannot end 0-0 (no ties, extra innings continue until
    # someone scores) - a reconstructed "final" score of 0-0 would only
    # ever indicate a data artifact, never a real result. Cheap, strictly
    # correct insurance regardless of cause.
    results = results[(results["home_score"] > 0) | (results["away_score"] > 0)]

    home_starters = starters.rename(columns={"team": "home_team", "pitcher": "home_probable_pitcher_key_mlbam"})
    away_starters = starters.rename(columns={"team": "away_team", "pitcher": "away_probable_pitcher_key_mlbam"})

    games = results.merge(home_starters, on=["game_id", "home_team"], how="left")
    games = games.merge(away_starters, on=["game_id", "away_team"], how="left")
    games["status"] = "Final"
    games = games.rename(columns={"game_id": "game_pk", "game_date": "date"})

    return games[
        [
            "game_pk", "date", "home_team", "away_team",
            "home_probable_pitcher_key_mlbam", "away_probable_pitcher_key_mlbam",
            "status", "home_score", "away_score",
        ]
    ]


def reconstruct_historical_game_picks(
    repo_dir: str = ".",
    raw_dir: str = "data/raw",
    season: int | None = None,
    days: int = 40,
    model_version: str = game_predictions.LEGACY_MODEL_VERSION,
) -> pd.DataFrame:
    """Replay the last `days` daily commits of confidence.csv/pave.csv
    through game_picks.compute_game_win_probabilities +
    game_predictions.select_game_picks, resolved immediately against
    derive_historical_schedule_games's actual outcomes. Skips any commit
    whose confidence.csv/pave.csv predates the columns the model needs
    (e.g. Bullpen_PAVE_PLUS was added partway through this project's
    history - clip_and_blend_pitching_quality already treats a missing
    bullpen column as neutral, so those dates aren't skipped for that
    reason, only for missing the four REQUIRED_CONFIDENCE_COLUMNS).

    `model_version` defaults to game_predictions.LEGACY_MODEL_VERSION, not
    config.GAME_PICK_MODEL_VERSION - same reasoning as
    git_backtest.reconstruct_historical_picks: this reconstructs what old
    logic *would have* picked using old-era confidence.csv/pave.csv
    snapshots, so tagging it as the current live version would misrepresent
    it once both kinds of rows coexist in the same game_predictions.csv."""
    season = season or config.SEASON_START.year

    commits = git_backtest.list_wave_csv_commits(repo_dir, path=CONFIDENCE_CSV_PATH)
    if days:
        commits = commits.sort_values("date").tail(days)

    persisted = data.load_persisted_statcast(raw_dir, season)
    if persisted is None:
        return pd.DataFrame(columns=game_predictions.GAME_PREDICTION_COLUMNS)
    schedule_games = derive_historical_schedule_games(persisted)

    all_picks = []
    for _, row in commits.iterrows():
        date = row["date"]
        try:
            confidence = git_backtest.read_csv_at_commit(row["commit"], CONFIDENCE_CSV_PATH, repo_dir)
            pave = git_backtest.read_csv_at_commit(row["commit"], PAVE_CSV_PATH, repo_dir)
        except subprocess.CalledProcessError:
            continue
        if confidence.empty or not REQUIRED_CONFIDENCE_COLUMNS.issubset(confidence.columns):
            continue
        if not REQUIRED_PAVE_COLUMNS.issubset(pave.columns):
            continue

        todays_games = schedule_games[schedule_games["date"] == date]
        if todays_games.empty:
            continue

        win_probabilities = game_picks.compute_game_win_probabilities(confidence, pave, todays_games)
        picks = game_predictions.select_game_picks(win_probabilities, date, model_version=model_version)
        if picks.empty:
            continue

        results = todays_games[["game_pk", "home_team", "away_team", "home_score", "away_score"]]
        picks = picks.merge(results, on=["game_pk", "home_team", "away_team"], how="left")
        picks["game_played"] = 1
        picks["actual_winner"] = picks["home_team"].where(
            picks["home_score"] > picks["away_score"], picks["away_team"]
        )
        all_picks.append(picks[game_predictions.GAME_PREDICTION_COLUMNS])

    if not all_picks:
        return pd.DataFrame(columns=game_predictions.GAME_PREDICTION_COLUMNS)
    return pd.concat(all_picks, ignore_index=True)
