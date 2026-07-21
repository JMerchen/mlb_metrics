"""One-off diagnostic: print the real shape of statsapi's schedule response
for a given date, to settle whether the probable pitcher's numeric player ID
is available via the convenience `statsapi.schedule()` wrapper or only via
the lower-level `statsapi.get(...)` call. This sandbox that built schedule.py
can't reach statsapi.mlb.com at all, so this has to run somewhere with real
network access (see .github/workflows/debug_statsapi.yml) before schedule.py's
parsing logic can be finalized against the real field names.

Usage: python scripts/debug_statsapi.py [YYYY-MM-DD]  (defaults to today)
Delete this script (and its workflow) once schedule.py is implemented and
the real shape is confirmed - it's a bootstrapping tool, not part of the
pipeline.
"""

import json
import sys


def main():
    date = sys.argv[1] if len(sys.argv) > 1 else None

    import statsapi

    print(f"statsapi version: {getattr(statsapi, '__version__', 'unknown')}")

    print("\n=== statsapi.schedule() convenience wrapper ===")
    games = statsapi.schedule(date=date) if date else statsapi.schedule()
    print(f"{len(games)} games returned")
    if games:
        print("Keys in one game dict:", sorted(games[0].keys()))
        print("First game (pretty-printed):")
        print(json.dumps(games[0], indent=2, default=str))

    print("\n=== lower-level statsapi.get('schedule', ...) with probablePitcher hydrate ===")
    params = {"sportId": 1, "hydrate": "probablePitcher,team"}
    if date:
        params["date"] = date
    raw = statsapi.get("schedule", params)
    print("Top-level keys:", sorted(raw.keys()))
    dates = raw.get("dates", [])
    if dates:
        games_raw = dates[0].get("games", [])
        print(f"{len(games_raw)} raw games on {dates[0].get('date')}")
        if games_raw:
            print("First raw game (pretty-printed):")
            print(json.dumps(games_raw[0], indent=2, default=str))


if __name__ == "__main__":
    main()
