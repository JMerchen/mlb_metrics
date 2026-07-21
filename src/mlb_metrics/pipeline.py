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

from mlb_metrics import config, data, hitters, pitchers, teams


def build_pitch_events(df: pd.DataFrame) -> pd.DataFrame:
    """Completed at-bat events with batter/p_throws, used by WAVE/WHOPS/WTB."""
    return data.completed_events(df, ["game_date", "batter", "events", "p_throws"])


def build_pitcher_events(df: pd.DataFrame) -> pd.DataFrame:
    """Completed at-bat events keyed by pitcher, used by PAVE."""
    return data.completed_events(df, ["game_date", "pitcher", "events"])


def compute_outputs(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run all three metric families against an already as-of-date-filtered
    Statcast dataframe. Returns {"wave": ..., "pave": ..., "confidence": ...}."""
    names = data.get_name_register()[["key_mlbam", "name_first", "name_last"]]
    latest_batter_team = data.latest_team_for_batters(df)
    latest_pitcher_team = data.latest_team_for_pitchers(df)
    data_with_game_id = data.assign_game_ids(df)

    dt = build_pitch_events(df)
    pdf = build_pitcher_events(df)

    return {
        "wave": hitters.assemble_hitters(dt, data_with_game_id, names, latest_batter_team),
        "pave": pitchers.assemble_pitchers(pdf, names, latest_pitcher_team),
        "confidence": teams.assemble_team_metrics(data_with_game_id),
    }


def run(
    as_of_date: datetime.date,
    raw_dir: str = "data/raw",
    output_dir: str = "docs/data",
    persist_raw: bool = True,
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
    parser.add_argument(
        "--no-persist-raw",
        action="store_true",
        help="Skip saving the Statcast pull to --raw-dir (useful for local/backtest runs).",
    )
    args = parser.parse_args()

    as_of_date = (
        datetime.date.fromisoformat(args.as_of_date) if args.as_of_date else datetime.date.today()
    )
    run(as_of_date, raw_dir=args.raw_dir, output_dir=args.output_dir, persist_raw=not args.no_persist_raw)


if __name__ == "__main__":
    main()
