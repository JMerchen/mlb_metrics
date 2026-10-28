"""Refresh data/payroll/team_payroll.csv's payrolls from The Baseball Cube
(payroll_sources): each requested season's opening-day payrolls replace
what's there, and a season not in the file yet is added with wins and
losses blank (build_payroll_value.py fills those in).

Or from the page's own "Payrolls by Team CSV" export, saved in a browser
(the site turns automated requests away; one file per season):

Usage:
    python scripts/fetch_payrolls.py                    # the current season
    python scripts/fetch_payrolls.py --seasons 2024-2025 --report changes.csv
    python scripts/fetch_payrolls.py --csv payroll_2026.csv
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from mlb_metrics import payroll_sources, payroll_value


def parse_seasons(text: str) -> list:
    seasons = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        seasons += list(range(int(lo), int(hi or lo) + 1))
    return seasons


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--payrolls", default="data/payroll/team_payroll.csv")
    parser.add_argument("--season-dates", default="data/season_dates.json")
    parser.add_argument("--seasons", default=None, help="e.g. 1988-2026 or 2025,2026 (default: the current season)")
    parser.add_argument("--capture", action="store_true", help="ask the Wayback Machine for a fresh capture when it has none")
    parser.add_argument("--report", default=None, help="write every payroll that changed to this CSV")
    parser.add_argument("--csv", nargs="+", default=None, help="The Baseball Cube's CSV export(s) to import instead of fetching")
    args = parser.parse_args()

    if args.csv:
        seasons = []
    elif args.seasons:
        seasons = parse_seasons(args.seasons)
    else:
        with open(args.season_dates) as f:
            seasons = [int(json.load(f)["season"])]

    payrolls = payroll_value.load_payrolls(args.payrolls)
    before = payrolls.set_index(["season", "franchise"])["payroll"]
    missing = []
    for path in args.csv or []:
        with open(path) as f:
            table = payroll_sources.parse_cube_csv(f.read())
        season = int(table["season"].iloc[0])
        if len(table) != payroll_sources.expected_teams(season) or table["franchise"].nunique() != len(table):
            sys.exit(f"{path}: {len(table)} teams for {season} - expected {payroll_sources.expected_teams(season)}")
        payrolls = payroll_sources.apply_payrolls(payrolls, table, source=f"thebaseballcube.com CSV export ({os.path.basename(path)})")
        print(f"{season}: {len(table)} teams from {path}")
    for season in seasons:
        table, snapshot = payroll_sources.fetch_season(season, capture=args.capture)
        if table is None:
            missing.append(season)
            print(f"{season}: no complete capture - kept as is")
            continue
        payrolls = payroll_sources.apply_payrolls(payrolls, table, snapshot)
        print(f"{season}: {len(table)} teams from the {snapshot} capture")

    after = payrolls.set_index(["season", "franchise"])["payroll"]
    changes = pd.DataFrame({"old": before.reindex(after.index), "new": after})
    changes = changes[changes["old"] != changes["new"]].reset_index()
    changes["change"] = changes["new"] - changes["old"]
    print(f"{len(changes)} payrolls changed or added across {changes['season'].nunique()} seasons"
          + (f"; no capture for {missing}" if missing else ""))
    if args.report:
        changes.to_csv(args.report, index=False)

    out = payrolls.copy()
    for col in ("wins", "losses"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    out.to_csv(args.payrolls, index=False)
    # A season asked for by name that couldn't be fetched is an error; the
    # weekly run's current season just may not be published yet.
    if args.seasons and seasons and len(missing) == len(seasons):
        sys.exit(1)


if __name__ == "__main__":
    main()
