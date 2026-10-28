"""Payroll value: how much winning each team buys with its payroll.

The MLB Payroll page (docs/payroll.html) and its downloadable workbook,
rebuilt from data/payroll/team_payroll.csv (one row per team-season:
season, franchise, payroll, wins, losses - opening-day payrolls from The
Baseball Cube via scripts/fetch_payrolls.py, wins and losses filled in by
scripts/build_payroll_value.py).

Every number is relative to its own season, so 1988 and 2026 compare:

- Win/Average: wins over half the games played (1.0 = .500).
- Price/Win/Average: the team's payroll per win over the league's
  (total payroll over total wins that season).
- Payroll Effectiveness Index: Win/Average over Price/Win/Average. Above
  1 bought more winning per dollar than the league did.
- Relative payroll (Price/Game/Average): payroll per game over the
  league's - how much a team spent next to everyone else.
- Expected wins (the Front Office model): the Win/Average of the 10
  seasons that spent most like this one (closest relative payroll) - from
  other franchises, at least 11 seasons earlier, so the comparison never
  sees the future - plus the franchise's FO Score.
- Year FO Score: Win/Average over those 10 neighbours' - what the front
  office got beyond what the money usually buys. FO Score: the average of
  the franchise's previous 3 Year FO Scores (its recent knack).
- Cash flow: relative payroll minus the average of the previous 3 seasons'
  (spending up or down from its habit).
- Payroll rank 1 = highest payroll; wins rank 1 = most wins.

Seasons before 1999 have no seasons 11 years earlier to compare with, so
they have no expected wins (the original workbook fell back to the first
10 rows of its sheet there).
"""

import numpy as np
import pandas as pd

NEIGHBORS = 10
NEIGHBOR_MIN_YEARS_BACK = 11
HISTORY_SEASONS = 3
FULL_SEASON_GAMES = 162

# Every spelling the hand-entered sheet used, to the franchise code.
FRANCHISE_ALIASES = {
    "ATH": ["OAK", "Oakland Athletics", "Athletics"],
    "ATL": ["Atlanta Braves"], "AZ": ["ARI", "Arizona Diamondbacks"],
    "BAL": ["Baltimore Orioles"], "BOS": ["Boston Red Sox"], "CHC": ["Chicago Cubs"],
    "CIN": ["Cincinnati Reds"], "CLE": ["Cleveland Guardians", "Cleveland Indians"],
    "COL": ["Colorado Rockies"], "CWS": ["CHW", "Chicago White Sox"], "DET": ["Detroit Tigers"],
    "HOU": ["Houston Astros"], "KC": ["Kansas City Royals"],
    "LAA": ["Anaheim Angels", "California Angels", "Los Angeles Angels", "Los Angeles Angels of Anaheim"],
    "LAD": ["Los Angeles Dodgers"], "MIA": ["Florida Marlins", "Miami Marlins"],
    "MIL": ["Milwaukee Brewers"], "MIN": ["Minnesota Twins"], "NYM": ["New York Mets"],
    "NYY": ["New York Yankees"], "PHI": ["Philadelphia Phillies"], "PIT": ["Pittsburgh Pirates"],
    "SD": ["San Diego Padres"], "SEA": ["Seattle Mariners"], "SF": ["San Francisco Giants"],
    "STL": ["St. Louis Cardinals"], "TB": ["Tampa Bay Devil Rays", "Tampa Bay Rays"],
    "TEX": ["Texas Rangers"], "TOR": ["Toronto Blue Jays"], "WSH": ["WAS", "Montreal Expos", "Washington Nationals"],
}
_ALIAS = {name.upper(): code for code, names in FRANCHISE_ALIASES.items() for name in [code, *names]}

# MLB Stats API team ids (standings) to franchise codes.
MLB_TEAM_IDS = {
    108: "LAA", 109: "AZ", 110: "BAL", 111: "BOS", 112: "CHC", 113: "CIN", 114: "CLE", 115: "COL",
    116: "DET", 117: "HOU", 118: "KC", 119: "LAD", 120: "WSH", 121: "NYM", 133: "ATH", 134: "PIT",
    135: "SD", 136: "SEA", 137: "SF", 138: "STL", 139: "TB", 140: "TEX", 141: "TOR", 142: "MIN",
    143: "PHI", 144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
}


def franchise_code(name) -> str:
    """'Montreal Expos', ' OAK', 'CHW' -> 'WSH', 'ATH', 'CWS'. Raises on an
    unknown name so a typo can't become a 31st franchise."""
    key = str(name).replace("\xa0", " ").strip().upper()
    if key not in _ALIAS:
        raise ValueError(f"Unknown team name {name!r} - add it to payroll_value.FRANCHISE_ALIASES")
    return _ALIAS[key]


def load_payrolls(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["franchise"] = df["franchise"].map(franchise_code)
    for col in ("wins", "losses"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    dupes = df[df.duplicated(["season", "franchise"], keep=False)]
    if not dupes.empty:
        raise ValueError(f"Duplicate team-seasons in {path}:\n{dupes}")
    return df.sort_values(["season", "franchise"]).reset_index(drop=True)


def update_records(payrolls: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    """Fill wins/losses from `records` [season, franchise, wins, losses]
    (a season's standings), keeping every other row as entered."""
    out = payrolls.set_index(["season", "franchise"])
    out[["wins", "losses"]] = out[["wins", "losses"]].astype(float)
    rec = records.set_index(["season", "franchise"])[["wins", "losses"]]
    rec = rec[rec.index.isin(out.index)]
    out.loc[rec.index, ["wins", "losses"]] = rec.astype(float).values
    return out.reset_index()


def _rank_desc(values: pd.Series) -> pd.Series:
    """1 + how many are strictly higher (ties share the better rank)."""
    return values.rank(method="min", ascending=False)


def _minmax(values: pd.Series) -> pd.Series:
    lo, hi = values.min(), values.max()
    return 2 * (values - lo) / (hi - lo) - 1


def _recent_mean(df: pd.DataFrame, col: str) -> pd.Series:
    """Per row, the mean of `col` over the franchise's previous
    HISTORY_SEASONS seasons (those with a value); NaN with none."""
    out = pd.Series(np.nan, index=df.index)
    for _, g in df.sort_values("season").groupby("franchise"):
        prev = g[col].shift(1).rolling(HISTORY_SEASONS, min_periods=1).mean()
        out.loc[g.index] = prev
    return out


def league_seasons(seasons: pd.DataFrame) -> pd.DataFrame:
    """Per season: teams, total and average payroll, and the league's price
    per win and per game (unplayed seasons: per scheduled game)."""
    g = seasons.groupby("season")
    league = pd.DataFrame({
        "teams": g.size(), "total_payroll": g["payroll"].sum(), "average_payroll": g["payroll"].mean(),
        "wins": g["wins"].sum(min_count=1), "games": g["games"].sum(),
    })
    league["price_per_win"] = league["total_payroll"] / league["wins"]
    league["price_per_game"] = league["total_payroll"] / league["games"]
    return league.reset_index()


def season_metrics(payrolls: pd.DataFrame, final_seasons=None) -> pd.DataFrame:
    """Every metric per team-season. `final_seasons`: seasons whose records
    are complete (neighbour pool); default every season with records."""
    df = payrolls[["season", "franchise", "payroll", "wins", "losses"]].copy()
    played = df["wins"].notna() & df["losses"].notna()
    df["games"] = np.where(played, df["wins"] + df["losses"], FULL_SEASON_GAMES)
    df["wpct"] = df["wins"] / (df["wins"] + df["losses"])
    df["price_per_game"] = df["payroll"] / df["games"]
    df["price_per_win"] = df["payroll"] / df["wins"]
    league = league_seasons(df).set_index("season")
    df["league_price_per_win"] = df["season"].map(league["price_per_win"])
    df["price_per_win_avg"] = df["price_per_win"] / df["league_price_per_win"]
    df["win_avg"] = df["wins"] / (df["games"] / 2)
    df["pe_index"] = df["win_avg"] / df["price_per_win_avg"]
    df["relative_payroll"] = df["price_per_game"] / df["season"].map(league["price_per_game"])
    df["payroll_rank"] = df.groupby("season")["payroll"].transform(_rank_desc)
    df["wins_rank"] = df.groupby("season")["wins"].transform(_rank_desc)

    if final_seasons is None:
        final_seasons = df.loc[played, "season"].unique()
    pool = df[played & df["season"].isin(final_seasons)]
    df = df.join(_neighbors(df, pool))
    df["year_fo_score"] = df["win_avg"] - df["baseline_win_avg"]
    df["fo_score"] = _recent_mean(df, "year_fo_score").fillna(0).where(df["baseline_win_avg"].notna())
    df["expected_win_avg"] = df["baseline_win_avg"] + df["fo_score"]
    df["expected_wins"] = df["expected_win_avg"] * df["games"] / 2
    df["wins_above_expected"] = df["wins"] - df["expected_wins"]
    df["spending_habit"] = _recent_mean(df, "relative_payroll")
    df["cash_flow"] = df["relative_payroll"] - df["spending_habit"]

    done = played & df["season"].isin(final_seasons)
    df["normalized_effectiveness"] = _minmax(df["pe_index"].where(done))
    df["normalized_wpct"] = _minmax(df["wpct"].where(done))
    df["status"] = np.where(~played, "forecast", np.where(df["season"].isin(final_seasons), "final", "in progress"))
    return df.drop(columns=["league_price_per_win"])


def _neighbors(df: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """The NEIGHBORS pool seasons closest in relative payroll to each row,
    from other franchises at least NEIGHBOR_MIN_YEARS_BACK seasons earlier:
    their mean Win/Average and who they were ("2008 SEA;2001 KC;...")."""
    rel = pool["relative_payroll"].to_numpy()
    wins = pool["win_avg"].to_numpy()
    seasons = pool["season"].to_numpy()
    franchises = pool["franchise"].to_numpy()
    labels = (pool["season"].astype(str) + " " + pool["franchise"]).to_numpy()
    baseline, names = [], []
    for row in df.itertuples():
        ok = (franchises != row.franchise) & (row.season - seasons >= NEIGHBOR_MIN_YEARS_BACK)
        if ok.sum() < NEIGHBORS:
            baseline.append(np.nan)
            names.append("")
            continue
        idx = np.flatnonzero(ok)
        nearest = idx[np.argsort(np.abs(rel[idx] - row.relative_payroll), kind="stable")[:NEIGHBORS]]
        baseline.append(wins[nearest].mean())
        names.append(";".join(labels[nearest]))
    return pd.DataFrame({"baseline_win_avg": baseline, "neighbors": names}, index=df.index)


def franchise_summary(seasons: pd.DataFrame) -> pd.DataFrame:
    """Per franchise over its final seasons: average PE Index, payroll rank,
    wins rank and wins above expected, each also scaled to [-1, 1] across
    franchises (rank scales: -1 = the best average rank)."""
    final = seasons[seasons["status"] == "final"]
    g = final.groupby("franchise")
    out = pd.DataFrame({
        "seasons": g.size(), "first_season": g["season"].min(), "total_payroll": g["payroll"].sum(),
        "wins": g["wins"].sum(), "losses": g["losses"].sum(),
        "average_effectiveness": g["pe_index"].mean(), "average_payroll_rank": g["payroll_rank"].mean(),
        "average_wins_rank": g["wins_rank"].mean(), "average_relative_payroll": g["relative_payroll"].mean(),
        "average_wins_above_expected": g["wins_above_expected"].mean(),
        "average_fo_score": g["year_fo_score"].mean(),
    })
    out["normalized_effectiveness"] = _minmax(out["average_effectiveness"])
    out["normalized_payroll"] = _minmax(out["average_payroll_rank"])
    out["normalized_wins"] = _minmax(out["average_wins_rank"])
    return out.reset_index().sort_values("average_effectiveness", ascending=False)


def model_error_wins(seasons: pd.DataFrame) -> float:
    """Mean absolute miss of expected wins, per 162 games, over final
    seasons with a baseline."""
    s = seasons[(seasons["status"] == "final") & seasons["expected_wins"].notna()]
    if s.empty:
        return float("nan")
    return float((s["wins_above_expected"].abs() / s["games"].astype(float)).mean() * FULL_SEASON_GAMES)


def standings_records(raw: dict, season: int) -> pd.DataFrame:
    """MLB Stats API standings JSON -> [season, franchise, wins, losses]."""
    rows = []
    for division in raw.get("records", []):
        for team in division.get("teamRecords", []):
            code = MLB_TEAM_IDS.get(team["team"]["id"])
            if code:
                rows.append({"season": season, "franchise": code, "wins": team["wins"], "losses": team["losses"]})
    return pd.DataFrame(rows, columns=["season", "franchise", "wins", "losses"])


def schedule_records(team_games: pd.DataFrame, season: int) -> pd.DataFrame:
    """docs/data/team_schedule.csv (one row per team-game) -> records."""
    g = team_games.groupby("team")["result"]
    rec = pd.DataFrame({"wins": g.apply(lambda r: (r == "W").sum()), "losses": g.apply(lambda r: (r == "L").sum())})
    rec = rec.reset_index().rename(columns={"team": "franchise"})
    rec["franchise"] = rec["franchise"].map(franchise_code)
    rec["season"] = season
    return rec[["season", "franchise", "wins", "losses"]]
