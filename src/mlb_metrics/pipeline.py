"""End-to-end pipeline: fetch Statcast -> persist raw -> compute hitter,
pitcher, and team metrics -> write docs/data/*.csv.

This replaces the original monolithic scripts/wave.py; that file is now a
thin entrypoint that calls `main()` here.

`run()` takes an explicit `as_of_date` and only ever uses games strictly
before it, rather than implicitly relying on the pipeline being run early
enough in the day that Statcast doesn't have today's games yet. That makes
the same-day leakage cutoff explicit and testable, and - together with raw
data persistence - lets this function be re-run against any past date for
backtesting (Phase B).
"""

import argparse
import datetime
import os

import pandas as pd

from mlb_metrics import (
    config, data, evaluation, game_evaluation, game_picks, game_predictions,
    hitters, lineup, matchup, pitchers, predictions, schedule, teams,
)


def build_pitch_events(df: pd.DataFrame) -> pd.DataFrame:
    """Completed at-bat events with batter/p_throws, used by WAVE/WHOPS/WTB."""
    return data.completed_events(df, ["game_date", "batter", "events", "p_throws"])


def build_pitcher_events(df: pd.DataFrame) -> pd.DataFrame:
    """Completed at-bat events keyed by pitcher, used by PAVE."""
    return data.completed_events(df, ["game_date", "pitcher", "events"])


def build_pitcher_events_with_role(data_with_game_id: pd.DataFrame, roles: pd.DataFrame) -> pd.DataFrame:
    """Completed at-bat events keyed by pitcher, with the pitching `team` and
    `is_starter` for that appearance attached, used by compute_bullpen_pave."""
    completed = data.completed_events(
        data_with_game_id, ["game_date", "pitcher", "events", "game_id"]
    )
    return completed.merge(
        roles[["game_id", "pitcher", "team", "is_starter"]], on=["game_id", "pitcher"], how="left"
    )


def compute_outputs(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run all three metric families against an already as-of-date-filtered
    Statcast dataframe. Returns {"wave": ..., "pave": ..., "confidence": ...}."""
    names = data.get_name_register()[["key_mlbam", "name_first", "name_last"]]
    latest_batter_team = data.latest_team_for_batters(df)
    latest_pitcher_team = data.latest_team_for_pitchers(df)
    data_with_game_id = data.assign_game_ids(df)
    roles = data.label_pitcher_roles(data_with_game_id)

    dt = build_pitch_events(df)
    pdf = build_pitcher_events(df)
    pdf_with_role = build_pitcher_events_with_role(data_with_game_id, roles)
    bullpen_pave = pitchers.compute_bullpen_pave(pdf_with_role)

    batting_order = data.assign_batting_order(data_with_game_id)
    lineup_consistency = lineup.compute_lineup_consistency(batting_order, latest_batter_team)

    return {
        "wave": hitters.assemble_hitters(dt, data_with_game_id, names, latest_batter_team, lineup_consistency),
        "pave": pitchers.assemble_pitchers(pdf, names, latest_pitcher_team),
        "confidence": teams.assemble_team_metrics(data_with_game_id, bullpen_pave),
    }


def write_beat_the_streak_export(predictions_log_path: str, output_dir: str) -> None:
    """Read the full predictions log and (re)write the two CSVs the
    dashboard's Beat the Streak section reads: each day's recommended picks
    (0 to DAILY_PICK_MAX, gated by DAILY_PICK_MIN_PROBABILITY - "no good
    matchup" means zero) with hit/miss/no_game/pending status, and a
    longest_streak/current_streak summary following Beat the Streak's actual
    rules (see evaluation.streak_progression). No-op if nothing's logged yet."""
    if not os.path.exists(predictions_log_path):
        return
    log = pd.read_csv(predictions_log_path, parse_dates=["date"])
    picks, summary = evaluation.build_beat_the_streak_export(
        log, max_picks=config.DAILY_PICK_MAX, min_probability=config.DAILY_PICK_MIN_PROBABILITY
    )
    os.makedirs(output_dir, exist_ok=True)
    picks.to_csv(os.path.join(output_dir, "beat_the_streak_picks.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "beat_the_streak_summary.csv"), index=False)


def write_game_picks_export(game_predictions_log_path: str, output_dir: str) -> None:
    """Read the full game-predictions log and (re)write the two CSVs the
    dashboard's Automated Game Picks section reads: each picked game with a
    win/loss/not_played/pending status, and an accuracy/streak summary (see
    game_evaluation.build_game_picks_export). No-op if nothing's logged yet."""
    if not os.path.exists(game_predictions_log_path):
        return
    log = pd.read_csv(game_predictions_log_path, parse_dates=["date"])
    picks, summary = game_evaluation.build_game_picks_export(log)
    os.makedirs(output_dir, exist_ok=True)
    picks.to_csv(os.path.join(output_dir, "game_picks_picks.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "game_picks_summary.csv"), index=False)


def run(
    as_of_date: datetime.date,
    raw_dir: str = "data/raw",
    output_dir: str = "docs/data",
    predictions_dir: str = "data/predictions",
    persist_raw: bool = True,
    log_predictions: bool = True,
) -> dict[str, pd.DataFrame]:
    fetch_start = config.SEASON_START
    fetch_end = min(config.SEASON_END, as_of_date - datetime.timedelta(days=1))
    if fetch_end < fetch_start:
        raise ValueError(f"as_of_date {as_of_date} is before the season start {fetch_start}")

    fresh = data.fetch_statcast_range(fetch_start, fetch_end)
    df = data.persist_raw_statcast(fresh, raw_dir, season=fetch_start.year) if persist_raw else fresh

    # Belt-and-suspenders cutoff: even if persisted raw data (or a future
    # fetch_end miscalculation) contains rows on/after as_of_date, never let
    # them reach the metrics.
    df = df[df["game_date"] < pd.Timestamp(as_of_date)].copy()

    outputs = compute_outputs(df)

    os.makedirs(output_dir, exist_ok=True)
    outputs["wave"].to_csv(os.path.join(output_dir, "wave.csv"), index=False)
    outputs["pave"].to_csv(os.path.join(output_dir, "pave.csv"), index=False)
    outputs["confidence"].to_csv(os.path.join(output_dir, "confidence.csv"), index=False)

    if log_predictions:
        predictions_log_path = os.path.join(predictions_dir, "predictions.csv")
        game_predictions_log_path = os.path.join(predictions_dir, "game_predictions.csv")

        # `df` already covers every completed game strictly before as_of_date,
        # i.e. exactly what's needed to resolve any pick logged on an earlier
        # run whose target date has since happened.
        completed = data.completed_events(df, ["game_date", "batter", "events"])
        predictions.resolve_predictions(predictions_log_path, completed)
        # Resolving game picks needs final scores, not Statcast (see
        # schedule.fetch_game_results) - this call is internally resilient
        # per-date (see game_predictions.resolve_game_predictions), so it
        # doesn't need its own try/except here.
        game_predictions.resolve_game_predictions(game_predictions_log_path, schedule.fetch_game_results, as_of_date)

        # A new external dependency (statsapi) must not be able to break the
        # whole daily update - on failure, fall back to no schedule
        # awareness at all for this run rather than skipping Game_Hit_Probability too.
        schedule_df = None
        try:
            schedule_df = schedule.fetch_probable_pitchers(as_of_date)
        except Exception as exc:
            print(f"WARNING: failed to fetch today's schedule/probable pitchers ({exc}); "
                  f"skipping Matchup_Hit_Probability and the teams-playing-today qualifier for this run.")

        # Same resilience for the game-per-row shape Automated Game Picks
        # needs (see schedule.normalize_schedule_games) - a separate call
        # since it deliberately doesn't dedupe doubleheaders the way
        # schedule_df above does, so it can't be derived from schedule_df.
        schedule_games_df = None
        try:
            schedule_games_df = schedule.fetch_todays_games(as_of_date)
        except Exception as exc:
            print(f"WARNING: failed to fetch today's game schedule ({exc}); "
                  f"skipping Automated Game Picks for this run.")

        # None (fetch failed) means "unknown, don't filter"; an empty set
        # (fetch succeeded, zero games today) correctly excludes every pick.
        teams_playing_today = set(schedule_df["team"]) if schedule_df is not None else None

        # On a normal day (schedule fetch succeeded), pick from a table that
        # also carries Matchup_Hit_Probability - select_picks' joint gate
        # then requires a good matchup just as much as probability/
        # Game_Hit_Probability, and rank_metric="Matchup_Approach" (the
        # three-way product) ranks the qualified pool by all three signals
        # combined. Falls back to Approach-only ranking (no matchup
        # qualifier at all) when schedule/matchup data isn't available -
        # same resilience pattern as the schedule_df fetch itself.
        # predicted_probability/metric logged still reflect Game_Hit_Probability
        # either way, so DAILY_PICK_MIN_PROBABILITY's calibration is unaffected.
        pick_pool = outputs["wave"]
        rank_metric = "Approach"
        if schedule_df is not None and not schedule_df.empty:
            matchup_probability = matchup.compute_matchup_hit_probability(
                outputs["wave"], outputs["pave"], outputs["confidence"], schedule_df
            )
            pick_pool = outputs["wave"].merge(matchup_probability, on="key_mlbam", how="inner")
            pick_pool["Matchup_Approach"] = pick_pool["Approach"] * pick_pool["Matchup_Hit_Probability"]
            rank_metric = "Matchup_Approach"

        game_hit_picks = predictions.select_picks(
            pick_pool, as_of_date, rank_metric=rank_metric, teams_playing_today=teams_playing_today
        )
        predictions.append_predictions(game_hit_picks, predictions_log_path)

        write_beat_the_streak_export(predictions_log_path, output_dir)

        if schedule_games_df is not None and not schedule_games_df.empty:
            win_probabilities = game_picks.compute_game_win_probabilities(
                outputs["confidence"], outputs["pave"], schedule_games_df
            )
            todays_game_picks = game_predictions.select_game_picks(win_probabilities, as_of_date)
            game_predictions.append_game_predictions(todays_game_picks, game_predictions_log_path)

        write_game_picks_export(game_predictions_log_path, output_dir)

    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Run the daily hitter/pitcher/team metrics pipeline."
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help="YYYY-MM-DD; defaults to today. Only games strictly before this date are used.",
    )
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--output-dir", type=str, default="docs/data")
    parser.add_argument("--predictions-dir", type=str, default="data/predictions")
    parser.add_argument(
        "--no-persist-raw",
        action="store_true",
        help="Skip saving the Statcast pull to --raw-dir (useful for local/backtest runs).",
    )
    parser.add_argument(
        "--no-log-predictions",
        action="store_true",
        help="Skip logging today's picks / resolving past ones (useful for local/backtest runs).",
    )
    args = parser.parse_args()

    as_of_date = (
        datetime.date.fromisoformat(args.as_of_date) if args.as_of_date else datetime.date.today()
    )
    run(
        as_of_date,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        predictions_dir=args.predictions_dir,
        persist_raw=not args.no_persist_raw,
        log_predictions=not args.no_log_predictions,
    )


if __name__ == "__main__":
    main()
