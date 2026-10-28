"""NFL payroll value: the MLB Payroll page's numbers (payroll_value) for
NFL teams - the NFL page's Payroll tab.

Spending is summed from OverTheCap's player contracts (nflverse's
`load_contracts`, each player's season-by-season cap number and cash paid
with the team he was on), two ways:

- cash: the dollars a team actually paid its players that season,
  signing bonuses counted when paid - what varies most between teams;
- cap: the cap charges of those contracts. The cap is hard, so this sits
  close to it for everyone.

Summed player contracts leave out dead money (players since released),
and they're thin before 2014, so the tab starts there. Records come from
nflverse schedules (regular season; a tie is half a win and half a loss).
With only a dozen seasons, expected wins compare a season with other
teams' seasons from any earlier year (MLB's: 11 or more years earlier).
"""

import numpy as np
import pandas as pd

from mlb_metrics import payroll_value

FIRST_SEASON = 2014
MIN_YEARS_BACK = 1
FULL_SEASON_GAMES = 17
MEASURES = ("cash", "cap")

# OverTheCap's team names (nicknames, and Washington's earlier ones) to the
# site's codes.
OTC_TEAMS = {
    "49ers": "SF", "Bears": "CHI", "Bengals": "CIN", "Bills": "BUF", "Broncos": "DEN", "Browns": "CLE",
    "Buccaneers": "TB", "Cardinals": "ARI", "Chargers": "LAC", "Chiefs": "KC", "Colts": "IND", "Cowboys": "DAL",
    "Dolphins": "MIA", "Eagles": "PHI", "Falcons": "ATL", "Giants": "NYG", "Jaguars": "JAX", "Jets": "NYJ",
    "Lions": "DET", "Packers": "GB", "Panthers": "CAR", "Patriots": "NE", "Raiders": "LV", "Rams": "LA",
    "Ravens": "BAL", "Saints": "NO", "Seahawks": "SEA", "Steelers": "PIT", "Texans": "HOU", "Titans": "TEN",
    "Vikings": "MIN", "Commanders": "WAS", "Redskins": "WAS", "Washington": "WAS",
}
# nflverse's older codes for moved franchises.
SCHEDULE_ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
TEAMS = sorted(set(OTC_TEAMS.values()))


def team_code(name) -> str:
    """An OTC nickname or a site code -> the site code; raises on anything
    else so a typo can't become a 33rd team."""
    key = str(name).strip()
    if key in OTC_TEAMS:
        return OTC_TEAMS[key]
    key = SCHEDULE_ALIASES.get(key.upper(), key.upper())
    if key not in TEAMS:
        raise ValueError(f"Unknown NFL team {name!r}")
    return key


def team_spending(contracts: pd.DataFrame) -> pd.DataFrame:
    """nflverse contracts (one row per contract, `season_history` a list of
    per-season dicts) -> [season, franchise, cash, cap, players] in dollars.
    A player's season appears once per contract he signed, so seasons are
    counted once per player and team."""
    rows = []
    for otc_id, history in zip(contracts["otc_id"], contracts["season_history"]):
        if history is None:
            continue
        for s in history:
            year, team = str(s.get("year") or ""), s.get("team")
            if year.isdigit() and team in OTC_TEAMS:
                rows.append((otc_id, int(year), OTC_TEAMS[team], s.get("cash_paid") or 0.0, s.get("cap_number") or 0.0))
    seasons = pd.DataFrame(rows, columns=["otc_id", "season", "franchise", "cash", "cap"])
    seasons = seasons.drop_duplicates(["otc_id", "season", "franchise"])
    out = seasons.groupby(["season", "franchise"]).agg(
        cash=("cash", "sum"), cap=("cap", "sum"), players=("otc_id", "nunique")).reset_index()
    out[["cash", "cap"]] = (out[["cash", "cap"]] * 1e6).round()  # OTC lists millions
    return out


def team_records(schedules: pd.DataFrame) -> pd.DataFrame:
    """Regular-season games played -> [season, franchise, wins, losses,
    ties, final] (`final`: every game of the season has a score); wins and
    losses count a tie as half of each."""
    games = schedules[schedules["game_type"] == "REG"]
    played = games[games["home_score"].notna() & games["away_score"].notna()]
    sides = []
    for team, us, them in (("home_team", "home_score", "away_score"), ("away_team", "away_score", "home_score")):
        sides.append(pd.DataFrame({"season": played["season"], "franchise": played[team].replace(SCHEDULE_ALIASES),
                                   "w": played[us] > played[them], "l": played[us] < played[them],
                                   "t": played[us] == played[them]}))
    g = pd.concat(sides).groupby(["season", "franchise"])
    rec = g[["w", "l", "t"]].sum().astype(int).reset_index().rename(columns={"t": "ties"})
    rec["wins"] = rec["w"] + rec["ties"] / 2
    rec["losses"] = rec["l"] + rec["ties"] / 2
    complete = games.groupby("season")["home_score"].apply(lambda s: s.notna().all())
    rec["final"] = rec["season"].map(complete).fillna(False).astype(bool)
    return rec[["season", "franchise", "wins", "losses", "ties", "final"]]


def team_seasons(spending: pd.DataFrame, records: pd.DataFrame, last_season: int) -> pd.DataFrame:
    """Spending and records for every team, FIRST_SEASON to `last_season`
    (a season not started has blank records)."""
    spend = spending[(spending["season"] >= FIRST_SEASON) & (spending["season"] <= last_season)]
    out = spend.merge(records, on=["season", "franchise"], how="left")
    return out.sort_values(["season", "franchise"]).reset_index(drop=True)


def season_metrics(seasons: pd.DataFrame) -> pd.DataFrame:
    """payroll_value's metrics for each measure, stacked with a `measure`
    column ("cash" / "cap")."""
    final = seasons.loc[seasons["final"].fillna(False).astype(bool), "season"].unique()
    out = []
    for measure in MEASURES:
        payrolls = seasons[["season", "franchise", "wins", "losses"]].assign(payroll=seasons[measure])
        m = payroll_value.season_metrics(payrolls, final_seasons=final, min_years_back=MIN_YEARS_BACK,
                                         full_season_games=FULL_SEASON_GAMES)
        m = m.merge(seasons[["season", "franchise", "ties"]], on=["season", "franchise"], how="left")
        out.append(m.assign(measure=measure))
    return pd.concat(out, ignore_index=True)


def summaries(metrics: pd.DataFrame) -> tuple:
    """(franchises, league) per measure, each with a `measure` column."""
    franchises, league = [], []
    for measure, m in metrics.groupby("measure", sort=False):
        ties = m[m["status"] == "final"].groupby("franchise")["ties"].sum()
        f = payroll_value.franchise_summary(m)
        franchises.append(f.assign(ties=f["franchise"].map(ties).fillna(0).astype(int), measure=measure))
        league.append(payroll_value.league_seasons(m, FULL_SEASON_GAMES).assign(measure=measure))
    return pd.concat(franchises, ignore_index=True), pd.concat(league, ignore_index=True)


def model_error_wins(metrics: pd.DataFrame, measure: str = "cash") -> float:
    return payroll_value.model_error_wins(metrics[metrics["measure"] == measure], FULL_SEASON_GAMES)


def last_season(schedules: pd.DataFrame) -> int:
    """The latest season with a regular-season game played."""
    games = schedules[(schedules["game_type"] == "REG") & schedules["home_score"].notna()]
    return int(games["season"].max()) if not games.empty else int(np.nanmax(schedules["season"]))
