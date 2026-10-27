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

import datetime
import os
from zoneinfo import ZoneInfo

import pandas as pd

from mlb_metrics import (
    config, market_odds, nfl_data, nfl_game_evaluation, nfl_game_picks, nfl_game_predictions,
    nfl_player_props, nfl_position_floors, nfl_prop_predictions, nfl_team_strength, nfl_totals, nfl_units,
    nfl_whatif,
)

NFL_GAME_PREDICTIONS_LOG_PATH = "data/predictions/nfl_game_predictions.csv"
NFL_PROP_PREDICTIONS_LOG_PATH = "data/predictions/nfl_prop_predictions.csv"

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

# This week's injury report is fetched on its own rather than as a
# TABLE_FETCHERS table: those are history, filtered to weeks before the
# one being predicted, and the report that matters is the current week's.
INJURY_FETCHER = nfl_data.fetch_injuries


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


def _this_week_injury_report(raw_dir: str, season: int, week: int) -> pd.DataFrame:
    """This week's injury report (regular season), fetched fresh and
    persisted; the last persisted copy if the fetch fails, and an empty
    frame if there is none - props then simply run without it."""
    try:
        injuries = INJURY_FETCHER([season])
        nfl_data.persist_table(injuries, raw_dir, "injuries", season)
    except Exception as exc:
        print(f"WARNING: failed to fetch the {season} injury report ({exc}); using the last persisted copy.")
        injuries = nfl_data.load_persisted_table(raw_dir, "injuries", season)
    if injuries is None or injuries.empty or "week" not in injuries.columns:
        return pd.DataFrame()
    report = injuries[injuries["week"] == week]
    if "game_type" in report.columns:
        report = report[report["game_type"] == "REG"]
    return report


def _today_eastern() -> datetime.date:
    """Today on the US East Coast, where NFL game dates are set."""
    return datetime.datetime.now(ZoneInfo("America/New_York")).date()


def _report_final_games(games: pd.DataFrame, today: datetime.date) -> list:
    """The games in `games` (not yet played) whose final injury report is
    out by `today`'s morning run: those played today or tomorrow - Thursday
    games on the Thursday run, Sunday games on the Saturday run, Monday
    games on the Monday run."""
    if games.empty or "gameday" not in games.columns:
        return []
    day = pd.to_datetime(games["gameday"], errors="coerce").dt.date
    soon = (day == today) | (day == today + datetime.timedelta(days=1))
    return games.loc[soon, "game_id"].tolist()


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


def _history_before(fresh: dict, prior_season: dict, week: int, rosters_through: int | None = None) -> dict:
    """Last season's tables plus this season's strictly before `week` -
    no lookahead for the week-scoped tables (schedules/team_stats/weekly/
    snap_counts/pbp). rosters_weekly has no such concern for a live run (a
    roster crosswalk row, not a game result), so the whole current season's
    is used unless `rosters_through` caps it - a replay of a past week
    shouldn't know about later trades."""
    def _current_season_before_week(table: str) -> pd.DataFrame:
        df = fresh[table]
        # A table whose live fetch failed entirely this run (see the
        # fetch loop in `run`) degrades to a real, empty, column-less
        # frame - filtering by "week" on that is a real KeyError, not
        # a row to include, so it's skipped rather than crashing.
        if df.empty or "week" not in df.columns:
            return pd.DataFrame()
        return df[df["week"] < week]

    def _prior(table: str) -> pd.DataFrame:
        return prior_season[table] if prior_season[table] is not None else pd.DataFrame()

    history = {
        table: pd.concat([_prior(table), _current_season_before_week(table)], ignore_index=True)
        for table in TABLES if table != "rosters_weekly"
    }
    rosters = fresh["rosters_weekly"]
    if rosters_through is not None and not rosters.empty and "week" in rosters.columns:
        rosters = rosters[rosters["week"] <= rosters_through]
    history["rosters_weekly"] = pd.concat([_prior("rosters_weekly"), rosters], ignore_index=True)
    return history


def _write_last_week_whatif(fresh: dict, prior_season: dict, season: int, week: int, output_dir: str) -> None:
    """The "What if?" panel's last-week view: the week just finished,
    replayed with only what was known before it (history, rosters and the
    live model, market tie-break and calibration exactly as a forward run
    would have used them), plus each game's final score, the QBs who
    started and which listed players took no snaps - a test case for the
    panel. Skipped in week 1 (no week of this season to replay)."""
    path = os.path.join(output_dir, "nfl_whatif_last_week.json")
    schedules = fresh["schedules"]
    last = week - 1
    games = schedules[(schedules["season"] == season) & (schedules["game_type"] == "REG")
                      & (schedules["week"] == last) & schedules["home_score"].notna()]
    if last < 1 or games.empty:
        return
    history = _history_before(fresh, prior_season, last, rosters_through=last)
    master = nfl_team_strength.assemble_team_metrics(
        history["schedules"], history["team_stats"], history["pbp"], current_season=season
    )
    qb_continuity = nfl_team_strength.compute_qb_continuity_adjustment(
        history["snap_counts"], history["weekly"], history["rosters_weekly"]
    )
    columns = ["game_id", "season", "week", "home_team", "away_team", "home_qb_id", "away_qb_id",
               "gameday", "home_moneyline", "away_moneyline", "home_qb_name", "away_qb_name"]
    games = games[[c for c in columns if c in games.columns]]
    probs = nfl_game_picks.compute_game_win_probabilities(master, qb_continuity, history["weekly"], games)
    probs = nfl_game_picks.apply_ml_model(probs, master, qb_continuity, history["weekly"], games)
    probs = nfl_game_picks.apply_market_tiebreak(probs, _build_market_probabilities(games))
    probs = nfl_game_picks.apply_calibration(probs)
    data = nfl_whatif.build_whatif(
        probs, master, qb_continuity, history["weekly"], history["rosters_weekly"], history["snap_counts"], games,
        pbp=history.get("pbp"),
    )
    snaps = fresh.get("snap_counts", pd.DataFrame())
    week_snaps = snaps[snaps["week"] == last] if not snaps.empty and "week" in snaps.columns else snaps
    nfl_whatif.add_results(data, schedules[schedules["game_id"].isin(games["game_id"])], week_snaps)
    nfl_whatif.save_json(path, data)


def run(
    season: int | None = None,
    raw_dir: str = config.NFL_RAW_DATA_DIR,
    output_dir: str = "docs/data",
    predictions_log_path: str = NFL_GAME_PREDICTIONS_LOG_PATH,
    prop_predictions_log_path: str = NFL_PROP_PREDICTIONS_LOG_PATH,
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

    # Same bulk-resolve idea for real prop picks, against this season's
    # freshly fetched real weekly stats. Real, disclosed scope limit: this
    # frame only covers `season`, so a still-pending pick from a PRIOR
    # season simply stays pending rather than resolving - acceptable while
    # this log is days old and entirely within one real season, and
    # visible rather than hidden (nfl_prop_predictions.summarize_prop_results
    # reports n_pending alongside every real hit rate).
    if os.path.exists(prop_predictions_log_path):
        nfl_prop_predictions.resolve_prop_predictions(prop_predictions_log_path, fresh.get("weekly"))

    week = determine_predictable_week(schedules, season)
    if week is None:
        print(f"No predictable {season} week yet (schedule not posted, or season already complete) - skipping prediction.")
    else:
        print(f"Predicting {season} week {week}...")

        prior_season = {
            table: nfl_data.load_persisted_table(raw_dir, table, season - 1) for table in TABLES
        }
        history = _history_before(fresh, prior_season, week)

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
            for column in ("total_line", "roof", "stadium_id", "home_qb_name", "away_qb_name"):
                if column in schedules.columns:
                    this_week_games[column] = schedules.loc[this_week_games.index, column]

            # Real, direct user request (2026-09-16): matchup-based
            # player prop rankings - reuses this SAME real no-lookahead
            # weekly history and this-week schedule already assembled
            # above for the win-probability model, no separate real
            # fetch (this project has already learned that shared GitHub
            # Actions minutes are a real, finite budget - see module
            # docstring). Independent of whether the win-probability
            # model below succeeds.
            # Real current-season snap share (see
            # nfl_player_props.compute_current_season_snap_share): a
            # player's own rate is built from the PRIOR season plus this
            # one, so without this a departed player keeps a real rate, a
            # real latest_team, and therefore a real upcoming opponent -
            # confirmed live on the real week-2 board, which surfaced four
            # Arizona backs who had not played a single real 2026 snap.
            # Uses `history`, not `fresh`: history is already restricted
            # to real weeks strictly BEFORE the week being predicted, so
            # this can never be told who actually suited up for the game
            # it is predicting. `fresh` would carry that week's own real
            # snaps once the week has been played, which is harmless for
            # a live forward run but is real lookahead the moment this is
            # re-run or replayed - the exact distinction
            # nfl_game_picks_backtest.py exists to respect.
            snap_share = nfl_player_props.compute_current_season_snap_share(
                history.get("snap_counts", pd.DataFrame()), history["rosters_weekly"], season
            )
            # Props are built for games not played yet only, with this
            # week's injury report: each re-run later in the week (Thursday,
            # Saturday and Monday mornings) refreshes the board and the
            # pending picks for games still to come, and leaves a game
            # that's already been played alone.
            unplayed = schedules.loc[this_week_games.index, "home_score"].isna()
            props_games = this_week_games[unplayed]
            injury_report = _this_week_injury_report(raw_dir, season, week)
            props_edges = nfl_player_props.build_prop_edges(
                history["weekly"], props_games, snap_share=snap_share, injuries_df=injury_report
            )
            top_props = nfl_player_props.write_prop_bets_csv(
                props_edges, os.path.join(output_dir, "nfl_player_props.csv")
            )
            # Real, structural fix (2026-09-16): that CSV is OVERWRITTEN
            # every week, so without this log nothing ever recorded
            # whether a real prop pick came in - the feature's own real
            # hit rate was unknowable, and therefore unimprovable (see
            # nfl_prop_predictions.py's own module docstring). Logged
            # BEFORE the games are played, resolved on a later real run
            # once real weekly stats exist for that week - the same
            # log -> resolve loop every other pick in this project uses.
            # Season floors by position - a HISTORY board, not a
            # prediction one: the top 32 by snaps at each position tab,
            # with the lowest number each has posted in every game,
            # injury exits left out (see nfl_position_floors' module
            # docstring). Deliberately NOT fed into select_prop_picks
            # below - it makes no prediction, so there is nothing for the
            # pick log to grade.
            nfl_position_floors.write_position_floors_csv(
                history.get("snap_counts", pd.DataFrame()), history["rosters_weekly"],
                history["weekly"], season, this_week_games,
                os.path.join(output_dir, "nfl_position_floors.csv"),
            )

            prop_picks = nfl_prop_predictions.select_prop_picks(top_props, season, week)
            playing = set(props_games["home_team"]) | set(props_games["away_team"])
            nfl_prop_predictions.drop_pending_picks(prop_predictions_log_path, season, week, playing)
            nfl_prop_predictions.append_prop_predictions(prop_picks, prop_predictions_log_path)

            probs = nfl_game_picks.compute_game_win_probabilities(
                master, qb_continuity, history["weekly"], this_week_games
            )
            # Real follow-up (2026-09-04 - the ratio heuristic above
            # structurally can't exceed ~59% for any real matchup, see
            # config.NFL_GAME_PICK_WIN_PROBABILITY_MODEL_PATH's own
            # comments): a real, walk-forward-validated ML model
            # (scripts/train_nfl_game_pick_model.py), if one has cleared
            # its own real save-gate, replaces home_win_probability
            # entirely - a real no-op (unchanged heuristic) otherwise.
            probs = nfl_game_picks.apply_ml_model(probs, master, qb_continuity, history["weekly"], this_week_games)

            market = _build_market_probabilities(this_week_games)
            # Real follow-up (2026-09-05 - user-driven post-mortem: "can
            # you investigate what happened in our misses [at 65%+
            # confidence]"): when our own prediction disagrees sharply
            # with the real market, the market wins that disagreement
            # more often than we do (see config.py's own
            # NFL_GAME_PICK_MARKET_DISAGREEMENT_THRESHOLD comments for
            # the full real 1,482-game validation) - defer to market
            # specifically in that proven-worse zone, trust the model
            # everywhere else.
            probs = nfl_game_picks.apply_market_tiebreak(probs, market)
            probs = nfl_game_picks.apply_calibration(probs)

            # The NFL page's "What if?" panel: QB swaps and absences for the
            # games still to be played, starting from these probabilities.
            whatif = nfl_whatif.build_whatif(
                probs, master, qb_continuity, history["weekly"], history["rosters_weekly"],
                history.get("snap_counts", pd.DataFrame()), props_games, pbp=history.get("pbp"),
            )
            # Games whose final injury report is out - played today or
            # tomorrow (the Thursday, Saturday and Monday morning runs) -
            # are repriced with it: its Out/Doubtful regulars at their
            # position defaults, and a backup for a listed QB who's out,
            # exactly as the What-if panel would (and the panel opens on
            # that report for them).
            report_games = _report_final_games(props_games, today=_today_eastern())
            repriced = nfl_whatif.apply_injury_report(
                whatif, injury_report, fresh.get("rosters_weekly"), history.get("snap_counts", pd.DataFrame()),
                report_games,
            )
            nfl_whatif.save_json(os.path.join(output_dir, "nfl_whatif.json"), whatif)
            _write_last_week_whatif(fresh, prior_season, season, week, output_dir)

            # Predicted points per side and total (nfl_totals.py), from
            # the same no-lookahead history. Shown next to the market's
            # total; never bet on (see that module's docstring).
            history_pbp = history.get("pbp", pd.DataFrame())
            points = nfl_totals.predict_game_points(
                nfl_totals.team_game_epa(history_pbp, history["schedules"]),
                this_week_games,
                qb_games=nfl_totals.qb_game_epa(history_pbp),
                history_schedules=history["schedules"],
            )
            # Team unit ratings (QB+, Receiving+, ...) for the NFL page's
            # Team units tab, from the same no-lookahead history.
            units = nfl_units.team_unit_ratings(
                nfl_units.unit_game_epa(history_pbp),
                nfl_totals.qb_game_epa(history_pbp),
                nfl_units.listed_starters(this_week_games, history["schedules"]),
            )
            if not units.empty:
                units.insert(0, "week", week)
                units.insert(0, "season", season)
                units.to_csv(os.path.join(output_dir, "nfl_team_units.csv"), index=False)

            # Every game's pick before the injury report (the first one
            # logged stays), then, for the games repriced above, the pick
            # after it (the latest replaces an earlier one).
            picks = nfl_game_predictions.select_game_picks(
                probs, this_week_games, market_probabilities=market, confidence=master,
                point_predictions=points,
            )
            picks["injury_report"] = "before"
            nfl_game_predictions.append_game_predictions(picks, predictions_log_path)
            if repriced:
                after_probs = probs.copy()
                after_probs["home_win_probability"] = after_probs["game_id"].map(
                    {g: r["probability"] for g, r in repriced.items()}).fillna(after_probs["home_win_probability"])
                after = nfl_game_predictions.select_game_picks(
                    after_probs, this_week_games, market_probabilities=market, confidence=master,
                    point_predictions=points,
                )
                after = after[after["game_id"].isin(repriced)].copy()
                after["injury_report"] = "after"
                after["injury_note"] = after["game_id"].map({g: r["note"] for g, r in repriced.items()})
                nfl_game_predictions.append_game_predictions(after, predictions_log_path)

    write_nfl_game_picks_export(predictions_log_path, output_dir)

    # Real, honest scoreboard for the props feature itself - written every
    # run so a real hit rate (or a real "not enough resolved picks yet")
    # is always visible rather than requiring someone to re-aggregate the
    # raw log by hand.
    prop_summary = nfl_prop_predictions.summarize_prop_results(prop_predictions_log_path)
    if not prop_summary.empty:
        os.makedirs(output_dir, exist_ok=True)
        prop_summary.to_csv(os.path.join(output_dir, "nfl_player_props_results.csv"), index=False)


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
