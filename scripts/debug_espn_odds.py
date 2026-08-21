"""One-off diagnostic: print the real shape of ESPN's public, keyless
scoreboard/summary API responses for MLB - quant-analytics item #6 ("no
market benchmark"), slice 1 (confirmation-only, nothing wired into the
live pipeline yet - see README's "Market benchmark" section).

Real web research (not assumed) found `site.api.espn.com`'s scoreboard/
summary endpoints require no API key/auth and still serve real moneyline
odds via a `pickcenter` array on the summary response, per a currently-
referenced community project
(github.com/elkingarcia11/mlb-gameday-obp-odds). This project's own
sandbox blocks every odds-site/ESPN domain directly (confirmed by trying
several), so - same reasoning as scripts/debug_nfl_data.py/
scripts/debug_statsapi.py - this script exists to settle the REAL
response shape (pickcenter field names, how many sportsbook rows, real
ESPN team abbreviations vs. schedule.TEAM_ID_TO_ABBREV, and critically
whether odds survive past game day at all) via a GitHub Actions runner
with real internet access, rather than writing ingestion logic against
guessed shapes.

Usage:
    python scripts/debug_espn_odds.py                  # today's real MLB slate
    python scripts/debug_espn_odds.py --days-back 5     # a real date 5 days ago (historical-odds check)
"""

import argparse
import datetime
import json
import os
import sys

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics.schedule import TEAM_ID_TO_ABBREV

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/summary"


def moneyline_to_implied_probability(moneyline: float) -> float:
    """Standard American-odds-to-implied-probability conversion."""
    if moneyline < 0:
        return -moneyline / (-moneyline + 100)
    return 100 / (moneyline + 100)


def devig(home_implied: float, away_implied: float) -> float:
    """Standard proportional de-vig - normalizes both sides' implied
    probabilities to sum to 1.0, removing the bookmaker's overround. The
    simplest, most defensible de-vig method, not the only one that
    exists."""
    return home_implied / (home_implied + away_implied)


def fetch_scoreboard(date: str) -> dict:
    resp = requests.get(SCOREBOARD_URL, params={"dates": date}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_summary(event_id: str) -> dict:
    resp = requests.get(SUMMARY_URL, params={"event": event_id}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _real_espn_abbrevs(scoreboard: dict) -> set:
    abbrevs = set()
    for event in scoreboard.get("events", []):
        for competition in event.get("competitions", []):
            for competitor in competition.get("competitors", []):
                team = competitor.get("team") or {}
                abbrevs.add(team.get("abbreviation"))
    return abbrevs


def report_for_date(date: str) -> None:
    print(f"\n=== Real ESPN scoreboard for {date} ===")
    scoreboard = fetch_scoreboard(date)
    events = scoreboard.get("events", [])
    print(f"{len(events)} real events found.")

    real_abbrevs = _real_espn_abbrevs(scoreboard)
    our_abbrevs = set(TEAM_ID_TO_ABBREV.values())
    print(f"Real ESPN abbreviations seen: {sorted(a for a in real_abbrevs if a)}")
    print(f"schedule.TEAM_ID_TO_ABBREV's own values: {sorted(our_abbrevs)}")
    mismatches = (real_abbrevs - our_abbrevs) if real_abbrevs else set()
    print(f"Real abbreviations NOT already in TEAM_ID_TO_ABBREV: {sorted(a for a in mismatches if a)}")

    for event in events[:3]:  # a handful is enough to confirm the real shape
        event_id = event.get("id")
        name = event.get("name")
        print(f"\n--- Event {event_id}: {name} ---")
        summary = fetch_summary(event_id)
        pickcenter = summary.get("pickcenter")
        if not pickcenter:
            print("  No real 'pickcenter' key/data in this summary response.")
            continue
        print(f"  Real pickcenter row count: {len(pickcenter)}")
        print(f"  First real pickcenter row (raw): {json.dumps(pickcenter[0], indent=2)[:1500]}")
        for row in pickcenter:
            away = row.get("awayTeamOdds") or {}
            home = row.get("homeTeamOdds") or {}
            a_ml = away.get("moneyLine")
            h_ml = home.get("moneyLine")
            provider = (row.get("provider") or {}).get("name")
            if a_ml is not None and h_ml is not None:
                a_p = moneyline_to_implied_probability(a_ml)
                h_p = moneyline_to_implied_probability(h_ml)
                print(
                    f"  provider={provider!r} away_ml={a_ml} home_ml={h_ml} "
                    f"-> raw implied away={a_p:.4f} home={h_p:.4f} (sum={a_p + h_p:.4f}) "
                    f"-> de-vigged home={devig(h_p, a_p):.4f}"
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--days-back", type=int, default=None,
        help="Also check a real date this many days in the past, to test whether odds survive past game day.",
    )
    args = parser.parse_args()

    today = datetime.date.today().strftime("%Y%m%d")
    report_for_date(today)

    if args.days_back is not None:
        past_date = (datetime.date.today() - datetime.timedelta(days=args.days_back)).strftime("%Y%m%d")
        report_for_date(past_date)


if __name__ == "__main__":
    main()
