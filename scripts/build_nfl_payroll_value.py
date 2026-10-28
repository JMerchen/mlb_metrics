"""Build the NFL page's Payroll tab data (nfl_payroll): team cash and cap
spending from OverTheCap's player contracts (nflverse), records from the
schedules, and payroll_value's metrics for each.

Writes data/nfl_payroll/team_payroll.csv (season, franchise, cash, cap,
players, wins, losses, ties, final) and docs/data/nfl_payroll_seasons.csv,
nfl_payroll_franchises.csv, nfl_payroll_league.csv (a `measure` column:
cash / cap).

Usage:
    python scripts/build_nfl_payroll_value.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from mlb_metrics import nfl_payroll

INT_COLUMNS = ["games", "payroll_rank", "wins_rank", "ties"]


def fetch_contracts() -> pd.DataFrame:
    import nflreadpy

    return nflreadpy.load_contracts().to_pandas()


def fetch_schedules(seasons) -> pd.DataFrame:
    import nflreadpy

    return nflreadpy.load_schedules(list(seasons)).to_pandas()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", default="data/nfl_payroll/team_payroll.csv")
    parser.add_argument("--data-dir", default="docs/data")
    parser.add_argument("--last-season", type=int, default=None, help="default: the latest season with a game played")
    args = parser.parse_args()

    schedules = fetch_schedules(range(nfl_payroll.FIRST_SEASON, pd.Timestamp.today().year + 1))
    last = args.last_season or nfl_payroll.last_season(schedules)
    table = nfl_payroll.team_seasons(nfl_payroll.team_spending(fetch_contracts()), nfl_payroll.team_records(schedules), last)
    counts = table.groupby("season")["franchise"].nunique()
    if (counts != len(nfl_payroll.TEAMS)).any():
        sys.exit(f"Expected {len(nfl_payroll.TEAMS)} teams every season, got {counts[counts != len(nfl_payroll.TEAMS)].to_dict()}")

    os.makedirs(os.path.dirname(args.table), exist_ok=True)
    out = table.copy()
    out["ties"] = out["ties"].astype("Int64")
    out.to_csv(args.table, index=False)

    metrics = nfl_payroll.season_metrics(table)
    franchises, league = nfl_payroll.summaries(metrics)
    metrics = metrics.astype({c: "Int64" for c in INT_COLUMNS})
    os.makedirs(args.data_dir, exist_ok=True)
    metrics.round(6).to_csv(os.path.join(args.data_dir, "nfl_payroll_seasons.csv"), index=False)
    franchises.round(6).to_csv(os.path.join(args.data_dir, "nfl_payroll_franchises.csv"), index=False)
    league.round(2).to_csv(os.path.join(args.data_dir, "nfl_payroll_league.csv"), index=False)
    status = metrics.groupby("season")["status"].first()
    print(f"Wrote NFL payroll value for {table['season'].min()}-{last} ({len(table)} team-seasons; "
          f"{status.iloc[-1]} {status.index[-1]}); expected wins miss by "
          f"{nfl_payroll.model_error_wins(metrics, 'cash'):.1f} (cash) / {nfl_payroll.model_error_wins(metrics, 'cap'):.1f} (cap) a season.")


if __name__ == "__main__":
    main()
