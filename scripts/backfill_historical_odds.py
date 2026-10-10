"""Backfills historical MLB moneylines from ESPN's core API for one season.

For every date in the range: ESPN's scoreboard gives the day's events,
then one core-API request per event gives every sportsbook's lines
(parsed by mlb_metrics.historical_odds). Rows are written to
data/raw/market_odds/espn_mlb_odds_<season>.parquet after every date, so
a re-run resumes where it stopped instead of starting over.

ESPN's API is public but undocumented: requests are paced (--pause) and
retried with backoff on errors, and a date that still fails is reported
and skipped rather than ending the run.

Needs real internet access to ESPN (blocked in this project's sandbox -
run through backfill_historical_odds.yml).

Usage:
    python scripts/backfill_historical_odds.py --season 2024
    python scripts/backfill_historical_odds.py --season 2022 --start 2022-04-07 --end 2022-10-05
"""

import argparse
import datetime
import os
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import historical_odds  # noqa: E402

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
CORE_ODDS_URL = "https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb/events/{event}/competitions/{event}/odds"

# Regular season windows, with a few days of slack; days without games
# simply return no events.
SEASON_WINDOWS = {
    2021: ("2021-03-30", "2021-10-04"),
    2022: ("2022-04-06", "2022-10-06"),
    2023: ("2023-03-29", "2023-10-02"),
    2024: ("2024-03-19", "2024-10-01"),
    2025: ("2025-03-17", "2025-09-29"),
}


def output_path(raw_dir: str, season: int) -> str:
    return os.path.join(raw_dir, "market_odds", f"espn_mlb_odds_{season}.parquet")


def _get(session: requests.Session, url: str, params=None, attempts: int = 4):
    for attempt in range(attempts):
        try:
            resp = session.get(url, params=params, timeout=30)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)


def fetch_date(session: requests.Session, date: datetime.date, pause: float) -> list:
    board = _get(session, SCOREBOARD_URL, params={"dates": date.strftime("%Y%m%d"), "limit": 50}) or {}
    rows = []
    for event in board.get("events", []):
        odds = _get(session, CORE_ODDS_URL.format(event=event.get("id")), params={"limit": 50})
        time.sleep(pause)
        if odds:
            rows.extend(historical_odds.parse_event_odds(date.isoformat(), event, odds))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--pause", type=float, default=0.2, help="Seconds between event requests.")
    args = parser.parse_args()

    default_start, default_end = SEASON_WINDOWS.get(args.season, (f"{args.season}-03-20", f"{args.season}-10-01"))
    start = datetime.date.fromisoformat(args.start or default_start)
    end = datetime.date.fromisoformat(args.end or default_end)
    path = output_path(args.raw_dir, args.season)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    existing = pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame(columns=historical_odds.ROW_COLUMNS)
    done = set(existing["date"].unique()) if not existing.empty else set()
    frames = [existing] if not existing.empty else []

    session = requests.Session()
    session.headers["User-Agent"] = "mlb_metrics historical odds backfill (personal research)"
    day = start
    failed = []
    while day <= end:
        if day.isoformat() not in done:
            try:
                rows = fetch_date(session, day, args.pause)
            except Exception as exc:
                print(f"{day}: failed ({exc}) - skipped")
                failed.append(day.isoformat())
                rows = []
            if rows:
                frames.append(pd.DataFrame(rows, columns=historical_odds.ROW_COLUMNS))
                combined = pd.concat(frames, ignore_index=True)
                combined.to_parquet(path, index=False)
                frames = [combined]
                n_games = combined.loc[combined["date"] == day.isoformat(), "event_id"].nunique()
                print(f"{day}: {n_games} games, {len(rows)} book lines")
        day += datetime.timedelta(days=1)

    final = pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame(columns=historical_odds.ROW_COLUMNS)
    consensus = historical_odds.consensus_closing_lines(final)
    print(f"\nDone: season {args.season}, {final['event_id'].nunique() if not final.empty else 0} events with odds, "
          f"{len(consensus)} finished games with a closing line; {len(failed)} dates failed {failed}")


if __name__ == "__main__":
    main()
