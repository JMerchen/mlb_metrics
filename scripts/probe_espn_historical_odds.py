"""Diagnostic: does ESPN still serve MLB moneylines for past seasons?

The live pipeline reads odds from site.api.espn.com's summary `pickcenter`,
which was confirmed to keep odds only a few days after a game
(config.MARKET_ODDS_BACKFILL_DAYS_BACK). ESPN's separate "core" API
(sports.core.api.espn.com) is reported to keep per-sportsbook opening and
closing lines on each past game from 2024 on, and a timestamped
line-movement feed for earlier seasons. This prints what that API
actually returns for a few sample dates per season, so any backfill is
written against real response shapes rather than assumed ones.

Run from GitHub Actions (debug_espn_odds.yml with historical=true): this
project's sandbox cannot reach ESPN.

Usage:
    python scripts/probe_espn_historical_odds.py
    python scripts/probe_espn_historical_odds.py --dates 20230615 20250815 --games 3
"""

import argparse
import json

import requests

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
CORE_ODDS_URL = "https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb/events/{event}/competitions/{event}/odds"

DEFAULT_DATES = ["20220615", "20220915", "20230615", "20230915", "20240615", "20240915", "20250615", "20250915"]


def _get(url, **params):
    resp = requests.get(url, params=params or None, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _moneylines(side: dict) -> dict:
    """Every moneyline-looking value on one team's odds block: the current
    line plus any open/close/current snapshots nested under it."""
    found = {}
    if not isinstance(side, dict):
        return found
    if "moneyLine" in side:
        found["moneyLine"] = side.get("moneyLine")
    for snapshot in ("open", "close", "current"):
        block = side.get(snapshot)
        if isinstance(block, dict):
            ml = block.get("moneyLine")
            if isinstance(ml, dict):
                ml = ml.get("american") or ml.get("value") or ml.get("alternateDisplayValue")
            found[snapshot] = ml
    return found


def probe_date(date: str, n_games: int) -> None:
    board = _get(SCOREBOARD_URL, dates=date)
    events = board.get("events", [])
    print(f"\n=== {date}: {len(events)} events on the scoreboard")
    for event in events[:n_games]:
        event_id = event.get("id")
        print(f"\n--- event {event_id}: {event.get('shortName')} ({event.get('status', {}).get('type', {}).get('description')})")
        try:
            odds = _get(CORE_ODDS_URL.format(event=event_id))
        except Exception as exc:
            print(f"    core odds request failed: {exc}")
            continue
        items = odds.get("items", [])
        print(f"    core odds items: {len(items)}")
        for item in items:
            provider = (item.get("provider") or {}).get("name")
            print(f"    provider={provider!r} keys={sorted(item.keys())}")
            print(f"      home: {_moneylines(item.get('homeTeamOdds'))}")
            print(f"      away: {_moneylines(item.get('awayTeamOdds'))}")
            # Earlier seasons: open/close come from a line-movement feed.
            ref = item.get("$ref")
            if ref:
                history_url = ref.split("?")[0] + "/history/0/movement"
                try:
                    movement = _get(history_url, limit=200)
                    moves = movement.get("items", [])
                    print(f"      movement entries: {len(moves)}")
                    if moves:
                        print("      first:", json.dumps(moves[0])[:300])
                        print("      last: ", json.dumps(moves[-1])[:300])
                except Exception as exc:
                    print(f"      movement request failed: {exc}")
        if items:
            # The full first item once per date, so field names are visible.
            print("    first item raw:", json.dumps(items[0])[:1500])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dates", nargs="*", default=DEFAULT_DATES)
    parser.add_argument("--games", type=int, default=2)
    args = parser.parse_args()
    for date in args.dates:
        try:
            probe_date(date, args.games)
        except Exception as exc:
            print(f"\n=== {date}: failed ({exc})")


if __name__ == "__main__":
    main()
