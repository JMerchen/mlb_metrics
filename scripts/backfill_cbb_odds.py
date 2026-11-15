"""Backfills college basketball betting lines from ESPN's core API for one
season: the spread, total and moneylines every sportsbook posted for each
game between two Division I teams (parsed by mlb_metrics.cbb_odds).

The games come from the season's hoopR schedule, whose game_id is ESPN's
event id: one request a game. Rows are written to
data/raw/market_odds/espn_cbb_odds_<season>.parquet every 200 games, and a
game with no odds gets one empty row, so a re-run resumes where it stopped
and asks for nothing twice.

ESPN's API is public but undocumented: requests are paced (--pause) and
retried with backoff, and a game that still fails is counted and skipped.
The first game with odds has its raw response printed, so the parse can
be checked against what ESPN actually sends.

Needs real internet access to ESPN (blocked in this project's sandbox -
run through backfill_cbb_odds.yml).

Usage:
    python scripts/backfill_cbb_odds.py --season 2026 [--limit 50]
"""

import argparse
import json
import os
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import cbb_data, cbb_odds  # noqa: E402

CORE_ODDS_URL = ("https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball"
                 "/events/{game}/competitions/{game}/odds")
SAVE_EVERY = 200


def output_path(raw_dir: str, season: int) -> str:
    return os.path.join(raw_dir, "market_odds", f"espn_cbb_odds_{season}.parquet")


def season_game_ids(season: int, cache_dir=None) -> list:
    """Completed games between two Division I teams (both have a conference), oldest first."""
    schedule = cbb_data.load_table("schedules", season, cache_dir=cache_dir)
    done = schedule[schedule["status_type_completed"].fillna(False).astype(bool)
                    & schedule["home_conference_id"].notna() & schedule["away_conference_id"].notna()]
    return [int(g) for g in done.sort_values(["game_date", "game_id"])["game_id"]]


def _get(session: requests.Session, url: str, attempts: int = 4):
    for attempt in range(attempts):
        try:
            resp = session.get(url, params={"limit": 50}, timeout=30)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int, required=True, help="The year the season ends in.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--pause", type=float, default=0.1, help="Seconds between requests.")
    parser.add_argument("--limit", type=int, default=None, help="Stop after this many new games (a probe).")
    args = parser.parse_args()

    path = output_path(args.raw_dir, args.season)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    existing = pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame(columns=cbb_odds.ROW_COLUMNS)
    done = set(existing["game_id"].astype(int)) if not existing.empty else set()
    todo = [g for g in season_game_ids(args.season) if g not in done]
    if args.limit:
        todo = todo[-args.limit:]   # the latest games: the likeliest to carry odds
    print(f"Season {args.season}: {len(done)} games already checked, {len(todo)} to fetch")

    session = requests.Session()
    session.headers["User-Agent"] = "sports research odds backfill (personal research)"
    frames, fresh, failed, shown = [existing] if not existing.empty else [], [], 0, False

    def save():
        nonlocal frames, fresh
        if fresh:
            frames.append(pd.DataFrame(fresh, columns=cbb_odds.ROW_COLUMNS))
            fresh = []
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined.to_parquet(path, index=False)
            frames = [combined]

    for count, game in enumerate(todo, start=1):
        try:
            odds = _get(session, CORE_ODDS_URL.format(game=game))
        except Exception as exc:
            failed += 1
            if failed <= 10:
                print(f"{game}: failed ({exc}) - skipped")
            continue
        rows = cbb_odds.parse_game_odds(game, odds)
        if rows and not shown:
            shown = True
            print(f"Raw response for game {game} (first 6,000 characters):")
            print(json.dumps(odds)[:6000])
            print("Parsed:", rows[:2])
        fresh.extend(rows or [{**{c: None for c in cbb_odds.ROW_COLUMNS}, "game_id": game, "provider": ""}])
        if count % SAVE_EVERY == 0:
            save()
            print(f"{count} of {len(todo)} games fetched")
        time.sleep(args.pause)
    save()

    final = pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame(columns=cbb_odds.ROW_COLUMNS)
    lines = cbb_odds.consensus_lines(final)
    print(f"\nDone: season {args.season}, {final['game_id'].nunique() if not final.empty else 0} games checked, "
          f"{len(lines)} with a line ({lines['spread'].notna().sum()} spreads, {lines['total'].notna().sum()} totals, "
          f"{lines['spread_open'].notna().sum()} opening spreads); {failed} requests failed")
    if not final.empty:
        print(final[final["provider"] != ""]["provider"].value_counts().head(15).to_string())


if __name__ == "__main__":
    main()
