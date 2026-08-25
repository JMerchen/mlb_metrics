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

from mlb_metrics import config, data, game_picks, game_predictions, git_backtest, pipeline, pitchers

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


# All EXPLORATORY bullpen-fatigue candidate columns - NOT part of
# game_picks.GAME_PICK_FEATURE_COLUMNS - see
# scripts/train_game_pick_model.py's significance report for whether any
# clear a real bar before going anywhere near the live model.
# home_bullpen_recent_outs/away_bullpen_recent_outs (2026-08-24, "what
# about bullpen rest/readiness") tested a single fixed 2-day window and
# found no signal - real follow-up (2026-08-25, "I want to see if other
# applications of bullpen fatigue are significant... I don't care if
# they're cheap"): a sweep of additional window lengths
# (config.BULLPEN_FATIGUE_CANDIDATE_WINDOWS) plus two genuinely different
# hypotheses - pitchers.compute_bullpen_distinct_relievers (workload
# BREADTH: how many different arms got used, not how many total outs)
# and pitchers.compute_bullpen_back_to_back_relievers (a sharper "which
# SPECIFIC arms are on zero rest" signal, distinct from a team-wide
# workload total).
BULLPEN_FATIGUE_WINDOW_COLUMN_PAIRS = [
    (f"home_bullpen_recent_outs_{d}d", f"away_bullpen_recent_outs_{d}d")
    for d in config.BULLPEN_FATIGUE_CANDIDATE_WINDOWS
]
BULLPEN_FATIGUE_CANDIDATE_COLUMNS = (
    ["home_bullpen_recent_outs", "away_bullpen_recent_outs"]
    + [col for pair in BULLPEN_FATIGUE_WINDOW_COLUMN_PAIRS for col in pair]
    + ["home_bullpen_distinct_relievers", "away_bullpen_distinct_relievers"]
    + ["home_bullpen_back_to_back_relievers", "away_bullpen_back_to_back_relievers"]
)

GAME_PICK_LOG_COLUMNS = (
    ["game_pk", "date", "home_team", "away_team"]
    + game_picks.GAME_PICK_FEATURE_COLUMNS
    + BULLPEN_FATIGUE_CANDIDATE_COLUMNS
    + ["home_win_probability", "Home_Won"]
)


def _merge_team_metric(rows: pd.DataFrame, metric: pd.DataFrame, value_column: str, home_col: str, away_col: str) -> pd.DataFrame:
    """Left-merges a [team, value_column] frame onto `rows` twice - once
    keyed by home_team (as `home_col`), once by away_team (as `away_col`).
    Shared by every bullpen-fatigue candidate merge below so the same
    two-sided join isn't duplicated per metric."""
    rows = rows.merge(
        metric.rename(columns={"team": "home_team", value_column: home_col}), on="home_team", how="left",
    )
    rows = rows.merge(
        metric.rename(columns={"team": "away_team", value_column: away_col}), on="away_team", how="left",
    )
    return rows


def assemble_game_pick_log(
    raw_dir: str = "data/raw",
    season: int | None = None,
    days: int = 20,
) -> pd.DataFrame:
    """Training log for scripts/train_game_pick_model.py: one row per
    historical game with [game_pk, date, home_team, away_team] +
    game_picks.GAME_PICK_FEATURE_COLUMNS (the raw, unblended per-team
    ingredients) + home_win_probability (the existing heuristic's pick,
    carried through as a tracked comparison column - not fed to the model,
    used only to compare against on holdout) + Home_Won (the real outcome,
    derived from the actual final score).

    Same no-lookahead per-date replay shape as
    reconstruct_historical_game_picks_from_persisted (recomputes
    pipeline.compute_outputs off only the Statcast rows strictly before
    each replayed date), but calls game_picks.build_game_features/
    compute_game_win_probabilities directly instead of going through
    game_predictions.select_game_picks - this is a training log, not a
    picks log, so every scheduled game is a row regardless of whether the
    heuristic would have favored home or away. One row per game_pk - no
    doubleheader-dedup step needed (unlike assemble_hitter_hit_log), since
    games are already keyed one row per game_pk, not per team."""
    season = season or config.SEASON_START.year

    persisted = data.load_persisted_statcast(raw_dir, season)
    if persisted is None:
        return pd.DataFrame(columns=GAME_PICK_LOG_COLUMNS)
    schedule_games = derive_historical_schedule_games(persisted)

    dates = sorted(schedule_games["date"].unique())
    if days:
        dates = dates[-days:]

    all_rows = []
    for date in dates:
        history = persisted[persisted["game_date"] < date]
        if history.empty:
            continue

        todays_games = schedule_games[schedule_games["date"] == date]
        if todays_games.empty:
            continue

        outputs = pipeline.compute_outputs(history)
        features = game_picks.build_game_features(outputs["confidence"], outputs["pave"], todays_games)
        win_probabilities = game_picks.compute_game_win_probabilities(
            outputs["confidence"], outputs["pave"], todays_games
        )
        rows = features.merge(
            win_probabilities[["game_pk", "home_win_probability"]], on="game_pk", how="left"
        )

        # Exploratory candidate (see GAME_PICK_LOG_COLUMNS's own comment) -
        # game_pk is Statcast's own real game id here (this module's own
        # convention, not data.assign_game_ids - see module docstring).
        # `events` is required by build_pitcher_events_with_role/
        # completed_events - always present on real persisted Statcast, but
        # missing from this module's own minimal schedule/score-only
        # synthetic fixtures (derive_historical_schedule_games never needed
        # it) - degrades to null rather than a KeyError, same "missing
        # optional input becomes an honest null" precedent used elsewhere.
        if "events" in history.columns:
            data_with_game_id = history.rename(columns={"game_pk": "game_id"})
            roles = data.label_pitcher_roles(data_with_game_id)
            pdf_with_role = pipeline.build_pitcher_events_with_role(data_with_game_id, roles)

            bullpen_workload = pitchers.compute_bullpen_recent_workload(pdf_with_role)
            rows = _merge_team_metric(
                rows, bullpen_workload, "Bullpen_Recent_Outs", "home_bullpen_recent_outs", "away_bullpen_recent_outs"
            )

            for window, (home_col, away_col) in zip(
                config.BULLPEN_FATIGUE_CANDIDATE_WINDOWS, BULLPEN_FATIGUE_WINDOW_COLUMN_PAIRS
            ):
                workload_at_window = pitchers.compute_bullpen_recent_workload(pdf_with_role, recent_days=window)
                rows = _merge_team_metric(rows, workload_at_window, "Bullpen_Recent_Outs", home_col, away_col)

            distinct_relievers = pitchers.compute_bullpen_distinct_relievers(pdf_with_role)
            rows = _merge_team_metric(
                rows, distinct_relievers, "Bullpen_Distinct_Relievers",
                "home_bullpen_distinct_relievers", "away_bullpen_distinct_relievers",
            )

            back_to_back = pitchers.compute_bullpen_back_to_back_relievers(pdf_with_role)
            rows = _merge_team_metric(
                rows, back_to_back, "Bullpen_Back_To_Back_Relievers",
                "home_bullpen_back_to_back_relievers", "away_bullpen_back_to_back_relievers",
            )
        else:
            for column in BULLPEN_FATIGUE_CANDIDATE_COLUMNS:
                rows[column] = pd.NA

        results = todays_games[["game_pk", "home_score", "away_score"]]
        rows = rows.merge(results, on="game_pk", how="left")
        rows["Home_Won"] = (rows["home_score"] > rows["away_score"]).astype(int)

        all_rows.append(rows[GAME_PICK_LOG_COLUMNS])

    if not all_rows:
        return pd.DataFrame(columns=GAME_PICK_LOG_COLUMNS)
    return pd.concat(all_rows, ignore_index=True)


def reconstruct_historical_game_picks_from_persisted(
    raw_dir: str = "data/raw",
    season: int | None = None,
    days: int = 20,
    model_version: str = config.GAME_PICK_MODEL_VERSION,
) -> pd.DataFrame:
    """Like reconstruct_historical_game_picks, but recomputes confidence.csv/
    pave.csv fresh from persisted Statcast (pipeline.compute_outputs) for
    each replayed date, instead of replaying old git-committed snapshots.

    This answers a different question than the git-history version. Old
    commits' confidence.csv/pave.csv were written by whatever code was live
    on that day, so a column added since (e.g. Power_A_PLUS) simply isn't
    there and gets skipped over - that's the right tool for "how did the
    model as of some past commit perform." This function instead reruns
    pipeline.compute_outputs - the exact function pipeline.run() calls
    every day - against each date's as-of-that-date slice of persisted
    Statcast, so every replayed date reflects whatever signals are live in
    the code RIGHT NOW. That's the right tool for "how does the model I'd
    ship today perform," which is what's needed to evaluate a change before
    waiting for it to accumulate its own live picks (append-only logs mean
    today's already-logged picks can't be rewritten retroactively - see
    predictions.py/game_predictions.py module docstrings).

    Recomputes the full season-to-date pipeline once per replayed date, so
    this is meaningfully slower than the git-history version - keep `days`
    modest for interactive use.

    `model_version` defaults to config.GAME_PICK_MODEL_VERSION (not
    LEGACY_MODEL_VERSION) because every replayed date uses today's actual
    live logic end to end - there's no "old logic" here to distinguish from."""
    season = season or config.SEASON_START.year

    persisted = data.load_persisted_statcast(raw_dir, season)
    if persisted is None:
        return pd.DataFrame(columns=game_predictions.GAME_PREDICTION_COLUMNS)
    schedule_games = derive_historical_schedule_games(persisted)

    dates = sorted(schedule_games["date"].unique())
    if days:
        dates = dates[-days:]

    all_picks = []
    for date in dates:
        history = persisted[persisted["game_date"] < date]
        if history.empty:
            continue

        todays_games = schedule_games[schedule_games["date"] == date]
        if todays_games.empty:
            continue

        outputs = pipeline.compute_outputs(history)
        win_probabilities = game_picks.compute_game_win_probabilities(
            outputs["confidence"], outputs["pave"], todays_games
        )
        # This function's whole purpose is "what would today's live code
        # produce" (see its own docstring) - pipeline.run() applies this
        # same rescaling, so skipping it here would misrepresent what's
        # actually live.
        win_probabilities = game_picks.apply_calibration(win_probabilities)
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
