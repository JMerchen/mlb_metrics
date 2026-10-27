"""Build the MLB Payroll page's data: docs/data/payroll_seasons.csv,
payroll_franchises.csv and payroll_league.csv (payroll_value). The page
builds the downloadable workbook from these in the browser.

Payrolls are entered by hand in data/payroll/team_payroll.csv (season,
franchise, payroll, wins, losses - a new season's 30 rows with wins and
losses left blank make it the preseason forecast). This fills in the
current season's wins and losses (and any earlier season's left blank)
from the MLB standings - falling back to docs/data/team_schedule.csv, the
Stats page's game log - and writes them back to that file.

Usage:
    python scripts/build_payroll_value.py
"""

import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from mlb_metrics import payroll_value

INT_COLUMNS = ["wins", "losses", "games", "payroll_rank", "wins_rank"]


def fetch_standings(season: int) -> pd.DataFrame:
    import statsapi

    raw = statsapi.get("standings", {"leagueId": "103,104", "season": season, "standingsTypes": "regularSeason"})
    return payroll_value.standings_records(raw, season)


def season_records(season: int, schedule_path: str, payrolls: pd.DataFrame) -> pd.DataFrame:
    """Wins and losses for `season`: the MLB standings, or else the Stats
    page's game log (current season only) - which never replaces a record
    with fewer games (Statcast can miss a game the standings count)."""
    try:
        records = fetch_standings(season)
        if len(records) == 30:
            print(f"{season} records from the MLB standings.")
            return records
        print(f"MLB standings returned {len(records)} teams for {season}.")
    except Exception as e:  # network, API shape
        print(f"MLB standings unavailable for {season} ({e}).")
    if not os.path.exists(schedule_path):
        return pd.DataFrame(columns=["season", "franchise", "wins", "losses"])
    games = pd.read_csv(schedule_path)
    games = games[games["date"].astype(str).str.startswith(str(season))]
    if games.empty:
        return pd.DataFrame(columns=["season", "franchise", "wins", "losses"])
    print(f"{season} records from {schedule_path}.")
    records = payroll_value.schedule_records(games, season)
    have = payrolls[payrolls["season"] == season].set_index("franchise")
    played = (have["wins"] + have["losses"]).reindex(records["franchise"]).fillna(0).to_numpy()
    return records[(records["wins"] + records["losses"]).to_numpy() >= played]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--payrolls", default="data/payroll/team_payroll.csv")
    parser.add_argument("--season-dates", default="data/season_dates.json")
    parser.add_argument("--schedule", default="docs/data/team_schedule.csv")
    parser.add_argument("--data-dir", default="docs/data")
    parser.add_argument("--today", default=None, help="YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    today = datetime.date.fromisoformat(args.today) if args.today else datetime.date.today()
    with open(args.season_dates) as f:
        dates = json.load(f)
    season = int(dates["season"])
    started = today >= datetime.date.fromisoformat(dates["regular_season_start"])
    ended = today > datetime.date.fromisoformat(dates["regular_season_end"])

    payrolls = payroll_value.load_payrolls(args.payrolls)
    missing = payrolls.loc[payrolls["wins"].isna() & (payrolls["season"] < season), "season"].unique().tolist()
    to_fill = sorted(set(missing) | ({season} if started and (payrolls["season"] == season).any() else set()))
    for year in to_fill:
        payrolls = payroll_value.update_records(payrolls, season_records(int(year), args.schedule, payrolls))
    if to_fill:
        out = payrolls.copy()
        for col in ("wins", "losses"):
            out[col] = out[col].astype("Int64")
        out.to_csv(args.payrolls, index=False)

    final = [s for s in payrolls["season"].unique() if s < season or (s == season and ended)]
    seasons = payroll_value.season_metrics(payrolls, final_seasons=final)
    franchises = payroll_value.franchise_summary(seasons)
    league = payroll_value.league_seasons(seasons)

    os.makedirs(args.data_dir, exist_ok=True)
    seasons = seasons.astype({c: "Int64" for c in INT_COLUMNS})
    seasons.round(6).to_csv(os.path.join(args.data_dir, "payroll_seasons.csv"), index=False)
    franchises.round(6).to_csv(os.path.join(args.data_dir, "payroll_franchises.csv"), index=False)
    league.round(2).to_csv(os.path.join(args.data_dir, "payroll_league.csv"), index=False)
    status = seasons.groupby("season")["status"].first()
    print(f"Wrote payroll value for {seasons['season'].min()}-{seasons['season'].max()} "
          f"({len(seasons)} team-seasons; {status.iloc[-1]} {status.index[-1]}); "
          f"expected wins miss by {payroll_value.model_error_wins(seasons):.1f} a season on average.")


if __name__ == "__main__":
    main()
