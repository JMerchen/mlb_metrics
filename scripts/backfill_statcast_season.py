"""One-time (re-runnable) backfill: persists a full past MLB season's real
pitch-by-pitch Statcast data to data/raw/statcast_<season>.parquet, the
exact same format/location the daily pipeline already builds up
incrementally for the CURRENT season (see data.py's module docstring,
wave.py).

Why this exists: this project has only ever fetched/persisted Statcast
data for the current season (config.SEASON_START's year) - every
historical backtest (game_picks_backtest.assemble_game_pick_log,
scripts/train_game_pick_model.py, scripts/train_game_pick_calibration.py)
has therefore only ever trained/validated against a single, partial
season (n in the hundreds/low thousands of games). Real historical
Statcast (2015+) and this project's own hand-derived features are
independent of any market-odds history - data.fetch_statcast_range/
persist_raw_statcast already take an arbitrary season/date range, this
capability just hadn't been exercised beyond the current season yet.

Fetches in CHUNKS (one real pybaseball.statcast() call per calendar month
by default, not one call for the whole season) - a real, deliberate
resilience choice: pybaseball's own statcast() warns that a single very
large query risks losing all its progress on a network hiccup, and each
chunk is persisted (data.persist_raw_statcast, which already dedupes by
real pitch key) immediately after it succeeds, so a mid-run failure only
costs the one in-flight chunk, not the whole season - just re-run this
script and the already-persisted months are skipped over (dedup makes
re-fetching a chunk safe, but this script skips a chunk outright when its
real date range is already fully covered by what's persisted, to avoid
the wasted real network time). One bad chunk is reported and skipped
(never fatal), same "one bad date can't block the rest" discipline
scripts/backfill_market_odds.py already established. `fetch_fn` is
injected (same dependency-injection pattern backfill_market_odds.py's own
backfill_market_probabilities uses) so the chunking/skip/resilience logic
is testable without real network.

Needs real internet access to Statcast (blocked in this project's own dev
sandbox - dispatch via the paired backfill_statcast_season.yml workflow,
same as every other real-network task this project has needed).

Usage:
    python scripts/backfill_statcast_season.py --season 2025
    python scripts/backfill_statcast_season.py --season 2025 --start-date 2025-03-18 --end-date 2025-09-28
"""

import argparse
import calendar
import datetime
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from mlb_metrics import data


def month_chunks(start_date: datetime.date, end_date: datetime.date):
    """Yields (chunk_start, chunk_end) date pairs, one per real calendar
    month, clipped to [start_date, end_date] at both ends."""
    cursor = start_date.replace(day=1)
    while cursor <= end_date:
        _, last_day = calendar.monthrange(cursor.year, cursor.month)
        month_end = cursor.replace(day=last_day)
        chunk_start = max(cursor, start_date)
        chunk_end = min(month_end, end_date)
        yield chunk_start, chunk_end
        cursor = (month_end + datetime.timedelta(days=1)).replace(day=1)


def chunk_already_covered(persisted: pd.DataFrame | None, chunk_start: datetime.date, chunk_end: datetime.date) -> bool:
    """True if `persisted` already has at least one real row on EVERY
    calendar day in [chunk_start, chunk_end] - a real, conservative
    coverage check (not just "any row in the range somewhere"), so a
    re-run genuinely skips only chunks that don't need re-fetching."""
    if persisted is None or persisted.empty:
        return False
    persisted_days = set(pd.to_datetime(persisted["game_date"]).dt.date)
    chunk_days = {chunk_start + datetime.timedelta(days=d) for d in range((chunk_end - chunk_start).days + 1)}
    return chunk_days.issubset(persisted_days)


def backfill_statcast_season(
    raw_dir: str, season: int, start_date: datetime.date, end_date: datetime.date, fetch_fn, persist_fn=None,
    load_fn=None,
) -> pd.DataFrame | None:
    """Core chunked-backfill loop, real network/disk access injected via
    `fetch_fn(chunk_start, chunk_end) -> DataFrame` (defaults to
    data.fetch_statcast_range), `persist_fn(df, raw_dir, season) -> DataFrame`
    (defaults to data.persist_raw_statcast), and `load_fn(raw_dir, season)
    -> DataFrame | None` (defaults to data.load_persisted_statcast) - same
    injection shape backfill_market_odds.py's own core function uses, for
    real unit testing without network or disk. Returns the final persisted
    DataFrame for `season` (None if nothing was ever persisted)."""
    persist_fn = persist_fn or data.persist_raw_statcast
    load_fn = load_fn or data.load_persisted_statcast

    chunks = list(month_chunks(start_date, end_date))
    for i, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        persisted = load_fn(raw_dir, season)
        if chunk_already_covered(persisted, chunk_start, chunk_end):
            print(f"  [{i}/{len(chunks)}] {chunk_start} - {chunk_end}: already fully persisted, skipping.")
            continue

        print(f"  [{i}/{len(chunks)}] {chunk_start} - {chunk_end}: fetching real Statcast data...")
        try:
            fresh = fetch_fn(chunk_start, chunk_end)
        except Exception as exc:
            print(f"    WARNING: failed to fetch {chunk_start} - {chunk_end} ({exc}); skipping this chunk.")
            continue

        if fresh.empty:
            print(
                f"    No real rows returned for {chunk_start} - {chunk_end} (a real All-Star break / off day "
                f"stretch, or the real season boundary) - nothing to persist."
            )
            continue

        combined = persist_fn(fresh, raw_dir, season)
        print(f"    Persisted {len(fresh)} new real pitch rows ({len(combined)} total for season={season}).")

    return load_fn(raw_dir, season)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument(
        "--start-date", type=str, default=None,
        help="Defaults to March 18 of --season (real MLB Opening Day window).",
    )
    parser.add_argument(
        "--end-date", type=str, default=None,
        help="Defaults to September 28 of --season (real MLB regular-season end window).",
    )
    parser.add_argument("--raw-dir", default="data/raw")
    args = parser.parse_args()

    start_date = datetime.date.fromisoformat(args.start_date) if args.start_date else datetime.date(args.season, 3, 18)
    end_date = datetime.date.fromisoformat(args.end_date) if args.end_date else datetime.date(args.season, 9, 28)

    print(f"Backfilling real Statcast data for season={args.season}, {start_date} through {end_date}...")

    final = backfill_statcast_season(args.raw_dir, args.season, start_date, end_date, data.fetch_statcast_range)

    if final is None or final.empty:
        print(f"\nNo real data ended up persisted for season={args.season}.")
        return
    n_days = pd.to_datetime(final["game_date"]).dt.date.nunique()
    n_games = final["game_pk"].nunique() if "game_pk" in final.columns else float("nan")
    print(f"\nDone: {len(final)} real pitch rows across {n_days} real distinct dates, {n_games} real distinct games.")


if __name__ == "__main__":
    main()
