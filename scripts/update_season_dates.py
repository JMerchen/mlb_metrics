"""Refresh data/season_dates.json (read by config.py) from the MLB Stats
API's season calendar, so SEASON_START / REGULAR_SEASON_END / SEASON_END
roll over to the new season on its Opening Day without a code change.

The current season is this year's once its regular season has started,
otherwise last year's - so the off-season keeps the finished season (the
site shows it as over) until Opening Day.

Run by the Daily Update workflow before the model. A failed fetch keeps
the existing file.

Usage:
    python scripts/update_season_dates.py
"""

import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import config, schedule


def fetch_regular_season(year):
    """(start, end) dates of `year`'s MLB regular season, or None when the
    API doesn't have that season yet."""
    import statsapi

    seasons = statsapi.get("season", {"seasonId": year, "sportId": 1}).get("seasons") or []
    if not seasons:
        return None
    season = seasons[0]
    start, end = season.get("regularSeasonStartDate"), season.get("regularSeasonEndDate")
    if not start or not end:
        return None
    return datetime.date.fromisoformat(start), datetime.date.fromisoformat(end)


def current_season(today, fetch=fetch_regular_season):
    """(year, start, end) of the season the site should cover on `today`."""
    this_year = fetch(today.year)
    if this_year and today >= this_year[0]:
        return (today.year, *this_year)
    last_year = fetch(today.year - 1)
    if last_year is None:
        raise RuntimeError(f"No MLB season dates for {today.year - 1}")
    return (today.year - 1, *last_year)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", default=config.SEASON_DATES_PATH)
    args = parser.parse_args()

    year, start, end = current_season(schedule.today_local())
    payload = {"season": year, "regular_season_start": start.isoformat(), "regular_season_end": end.isoformat()}
    with open(args.path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"Season {year}: regular season {start} to {end}")


if __name__ == "__main__":
    main()
