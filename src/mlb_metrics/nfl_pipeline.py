"""Weekly NFL Automated Game Picks pipeline entrypoint - the NFL analog of
pipeline.run()'s Automated Game Picks section (not the whole MLB daily
pipeline - NFL's DFS/bestball side has its own separate build scripts,
this module is scoped to game predictions only).

**Fetch strategy**: the CURRENT season (config.NFL_SEASON) is fetched
fresh and persisted as a whole-file overwrite every run (nfl_data.py's own
documented convention for "current season, re-fetched regularly" - see
its module docstring); the immediately PRIOR season is read from its
already-persisted parquet (scripts/fetch_nfl_historical.py, never
re-fetched here) purely to give nfl_team_strength.py's games-back windows
(largest is 8 games, config.NFL_TEAM_STRENGTH_WINDOWS) real history to
draw from at the very start of a new season, when the current season
alone has too few games. This is a genuine, deliberate difference from
nfl_game_picks_backtest.py's own single-season-only methodology (see that
module's own docstring) - the backtest validates the model against one
season in isolation on purpose; the LIVE pipeline should not cold-start
every single September the way the backtest's week-2 degeneracy shows a
single season does on its own (compute_strength_metrics/
compute_qb_continuity_adjustment don't reset at a season boundary -
confirmed by direct reading, see nfl_team_strength.py's own docstring).

**Which week gets predicted**: `determine_predictable_week` - the
earliest real REG week in the current season with at least one game whose
real final score isn't in yet. Weekly cadence (not daily - real NFL games
cluster Thu/Sun/Mon, and this project has already learned firsthand that
shared GitHub Actions minutes are a real, finite, easily-squeezed budget)
means this naturally advances by one week between scheduled runs.

**Real market odds**: `schedules_*.parquet`'s own real `home_moneyline`/
`away_moneyline` - unlike MLB, no separate market_odds.py scrape is
needed (see nfl_game_picks_backtest.py's own module docstring for the
same real advantage in the backtest). A pre-kickoff week whose real
moneylines haven't posted yet simply logs with null market data for
those games - the same real, graceful degradation
nfl_game_predictions.select_game_picks already handles for every other
missing-market case.
"""

import os

import pandas as pd

from mlb_metrics import config, market_odds, nfl_data, nfl_game_evaluation, nfl_game_picks, nfl_game_predictions, nfl_team_strength

NFL_GAME_PREDICTIONS_LOG_PATH = "data/predictions/nfl_game_predictions.csv"

# (table name, fetch function) - table names match scripts/fetch_nfl_historical.py's
# own TABLES list/persist_table's file-naming convention; fetch function
# names don't all match "fetch_<table>" (fetch_weekly_stats, not
# fetch_weekly), so this is an explicit mapping, not a derived getattr.
TABLE_FETCHERS = [
    ("schedules", nfl_data.fetch_schedules),
    ("team_stats", nfl_data.fetch_team_stats),
    ("weekly", nfl_data.fetch_weekly_stats),
    ("snap_counts", nfl_data.fetch_snap_counts),
    ("rosters_weekly", nfl_data.fetch_rosters_weekly),
    ("pbp", nfl_data.fetch_pbp),
]
TABLES = [table for table, _ in TABLE_FETCHERS]


def determine_predictable_week(schedules_df: pd.DataFrame, season: int) -> int | None:
    """The earliest real REG week in `season` with at least one game
    whose real final score isn't in yet - the week this run should
    predict. Returns None if the season's schedule hasn't posted yet (no
    real REG rows for `season` at all) or every real REG game already has
    a final score (the season is over)."""
    reg = schedules_df[(schedules_df["season"] == season) & (schedules_df["game_type"] == "REG")]
    if reg.empty:
        return None
    unplayed = reg[reg["home_score"].isna() | reg["away_score"].isna()]
    if unplayed.empty:
        return None
    return int(unplayed["week"].min())


def _build_market_probabilities(this_week_games: pd.DataFrame) -> pd.DataFrame:
    """[home_team, away_team, home_moneyline, away_moneyline,
    market_home_win_probability] straight from `this_week_games`'s own
    real schedules_*.parquet columns - no separate fetch (see module
    docstring)."""
    market = this_week_games[["home_team", "away_team", "home_moneyline", "away_moneyline"]].copy()
    home_implied = market["home_moneyline"].apply(market_odds.moneyline_to_implied_probability)
    away_implied = market["away_moneyline"].apply(market_odds.moneyline_to_implied_probability)
    market["market_home_win_probability"] = market_odds.devig(home_implied, away_implied)
    return market


def write_nfl_game_picks_export(predictions_log_path: str, output_dir: str) -> None:
    """Read the full NFL game-predictions log and (re)write the CSVs the
    dashboard's NFL Automated Game Picks section reads - direct mirror of
    pipeline.write_game_picks_export."""
    if not os.path.exists(predictions_log_path):
        return
    log = pd.read_csv(predictions_log_path, parse_dates=["date"])
    picks, summary = nfl_game_evaluation.build_game_picks_export(log)
    _, current_version_summary = nfl_game_evaluation.build_game_picks_export(
        log, model_version=config.NFL_GAME_PICK_MODEL_VERSION
    )
    by_version_summary = pd.concat([summary, current_version_summary], ignore_index=True)

    os.makedirs(output_dir, exist_ok=True)
    picks.to_csv(os.path.join(output_dir, "nfl_game_picks_picks.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "nfl_game_picks_summary.csv"), index=False)
    by_version_summary.to_csv(os.path.join(output_dir, "nfl_game_picks_summary_by_version.csv"), index=False)


def run(
    season: int | None = None,
    raw_dir: str = config.NFL_RAW_DATA_DIR,
    output_dir: str = "docs/data",
    predictions_log_path: str = NFL_GAME_PREDICTIONS_LOG_PATH,
) -> None:
    season = season or config.NFL_SEASON

    print(f"Fetching real {season} NFL data (whole-file overwrite)...")
    fresh = {}
    for table, fetch_fn in TABLE_FETCHERS:
        try:
            fresh[table] = fetch_fn([season])
        except Exception as exc:
            # Real, live-confirmed preseason case (not hypothetical): early
            # in a new season, nflverse hasn't published that season's
            # team_stats/weekly file yet (a real 404 - the file doesn't
            # exist upstream until real games start generating stats), and
            # snap_counts/rosters_weekly reject a season past their own
            # currently-valid upper bound outright (a real nflreadpy
            # ValueError, "Season must be between X and Y"). Only
            # `schedules` reliably posts ahead of kickoff. Falls back to
            # whatever was already persisted locally for this table/season
            # (a prior run that DID succeed), or an empty frame if nothing
            # has ever been fetched yet - never crashes the whole run over
            # one table not being live yet.
            print(f"WARNING: failed to fetch real {season} {table} ({exc}); falling back to the last persisted copy.")
            existing = nfl_data.load_persisted_table(raw_dir, table, season)
            fresh[table] = existing if existing is not None else pd.DataFrame()
            continue
        nfl_data.persist_table(fresh[table], raw_dir, table, season)

    schedules = fresh["schedules"]
    if schedules.empty:
        print(f"No real {season} schedule available at all - cannot proceed this run.")
        return

    # Resolve any still-pending picks against this season's freshly
    # fetched real final scores - a single bulk match, no per-date fetch
    # loop needed (see nfl_game_predictions.resolve_game_predictions's own
    # docstring).
    if os.path.exists(predictions_log_path):
        nfl_game_predictions.resolve_game_predictions(predictions_log_path, schedules)

    week = determine_predictable_week(schedules, season)
    if week is None:
        print(f"No predictable {season} week yet (schedule not posted, or season already complete) - skipping prediction.")
    else:
        print(f"Predicting {season} week {week}...")

        prior_season = {
            table: nfl_data.load_persisted_table(raw_dir, table, season - 1) for table in TABLES
        }
        # week-scoped tables (schedules/team_stats/weekly/snap_counts):
        # no-lookahead restricted to strictly-before-`week` current-season
        # rows. rosters_weekly has no such concern (a roster crosswalk
        # row, not a game result) - the whole current season's real
        # rosters plus last season's are always fair game.
        week_scoped_tables = [t for t in TABLES if t != "rosters_weekly"]

        def _current_season_before_week(table: str) -> pd.DataFrame:
            df = fresh[table]
            # A table whose live fetch failed entirely this run (see the
            # fetch loop above) degrades to a real, empty, column-less
            # frame - filtering by "week" on that is a real KeyError, not
            # a row to include, so it's skipped rather than crashing.
            if df.empty or "week" not in df.columns:
                return pd.DataFrame()
            return df[df["week"] < week]

        history = {
            table: pd.concat(
                [prior_season[table] if prior_season[table] is not None else pd.DataFrame(),
                 _current_season_before_week(table)],
                ignore_index=True,
            )
            for table in week_scoped_tables
        }
        history["rosters_weekly"] = pd.concat(
            [prior_season["rosters_weekly"] if prior_season["rosters_weekly"] is not None else pd.DataFrame(), fresh["rosters_weekly"]],
            ignore_index=True,
        )

        real_history_games = history["schedules"][
            (history["schedules"]["game_type"] == "REG")
            & history["schedules"]["home_score"].notna()
            & history["schedules"]["away_score"].notna()
        ]
        if real_history_games.empty:
            print("No real completed games in history yet (brand-new season with no prior-season data) - skipping prediction.")
        else:
            # current_season is passed EXPLICITLY, not left to
            # assemble_team_metrics' own default inference
            # (history["schedules"]["season"].max()) - at week 1 of a new
            # season, `history["schedules"]` contains ONLY the prior
            # season's real rows (this season's own games haven't been
            # played yet), so that inference would silently resolve to
            # the WRONG season - exactly the real cold-start case the
            # season-carryover feature exists to handle correctly (real
            # follow-up, 2026-09-04).
            master = nfl_team_strength.assemble_team_metrics(
                history["schedules"], history["team_stats"], history["pbp"], current_season=season
            )
            qb_continuity = nfl_team_strength.compute_qb_continuity_adjustment(
                history["snap_counts"], history["weekly"], history["rosters_weekly"]
            )

            this_week_games = schedules[
                (schedules["season"] == season) & (schedules["game_type"] == "REG") & (schedules["week"] == week)
            ][["game_id", "season", "week", "home_team", "away_team", "home_qb_id", "away_qb_id",
               "gameday", "home_moneyline", "away_moneyline"]]

            probs = nfl_game_picks.compute_game_win_probabilities(
                master, qb_continuity, history["weekly"], this_week_games
            )
            probs = nfl_game_picks.apply_calibration(probs)

            market = _build_market_probabilities(this_week_games)
            picks = nfl_game_predictions.select_game_picks(
                probs, this_week_games, market_probabilities=market, confidence=master
            )
            nfl_game_predictions.append_game_predictions(picks, predictions_log_path)

    write_nfl_game_picks_export(predictions_log_path, output_dir)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int, default=config.NFL_SEASON)
    parser.add_argument("--raw-dir", default=config.NFL_RAW_DATA_DIR)
    parser.add_argument("--output-dir", default="docs/data")
    parser.add_argument("--predictions-log", default=NFL_GAME_PREDICTIONS_LOG_PATH)
    args = parser.parse_args()

    run(
        season=args.season, raw_dir=args.raw_dir, output_dir=args.output_dir,
        predictions_log_path=args.predictions_log,
    )


if __name__ == "__main__":
    main()
