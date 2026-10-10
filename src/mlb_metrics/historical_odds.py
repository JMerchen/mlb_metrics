"""Historical MLB moneylines from ESPN's core API, for backtesting bets.

The live pipeline logs odds each morning (market_odds.py), so before
August 2026 there were no odds to grade the model's bets against - only
488 games. ESPN's separate core API (sports.core.api.espn.com) keeps
moneylines on past games: an ESPN BET line with opening and closing
snapshots for 2024 on, an ESPN BET closing line plus 10-15 other books'
final lines for 2023, and several books' final lines for 2022 - confirmed
by scripts/probe_espn_historical_odds.py on 2026-10-10.

This module only parses responses into rows; scripts/backfill_historical_odds.py
does the fetching. One row per (game, sportsbook), "Live Odds" feeds
excluded (they hold in-game prices, e.g. -6000 late in a blowout).

Which number to bet against: `close` is the line just before first pitch,
the standard benchmark for a betting edge and the hardest one to beat. A
book with no `close` snapshot (2022-2023, except ESPN BET) still carries
`moneyLine`, its last pre-game price, which is the same thing in practice;
`consensus_closing_lines` uses `close` where present and falls back to it.
"""

import numpy as np
import pandas as pd

ROW_COLUMNS = [
    "date", "event_id", "home_team", "away_team", "start_time", "status",
    "provider", "home_ml", "away_ml", "home_ml_open", "away_ml_open", "home_ml_close", "away_ml_close",
]


def parse_american(value):
    """'+125' / '-115' / 'EVEN' / 125 -> 125 / -115 / 100 / 125. None for
    anything missing or unparseable, and for 0 (ESPN's placeholder)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip().upper()
        if text in ("EVEN", "EV", "PK"):
            return 100.0
        try:
            number = float(text.replace("+", ""))
        except ValueError:
            return None
    if number == 0 or abs(number) < 100:
        return None
    return number


def _snapshot_moneyline(side: dict, snapshot: str):
    block = (side or {}).get(snapshot)
    if not isinstance(block, dict):
        return None
    moneyline = block.get("moneyLine")
    if isinstance(moneyline, dict):
        return parse_american(moneyline.get("american") or moneyline.get("alternateDisplayValue"))
    return parse_american(moneyline)


def event_teams(event: dict):
    """(home, away) ESPN abbreviations for a scoreboard event."""
    home = away = None
    for competition in event.get("competitions", []):
        for competitor in competition.get("competitors", []):
            abbrev = (competitor.get("team") or {}).get("abbreviation")
            if competitor.get("homeAway") == "home":
                home = abbrev
            elif competitor.get("homeAway") == "away":
                away = abbrev
    return home, away


def parse_event_odds(date: str, event: dict, odds_json: dict) -> list:
    """Rows for one scoreboard event and its core-API odds response."""
    home, away = event_teams(event)
    status = ((event.get("status") or {}).get("type") or {}).get("name")
    rows = []
    for item in odds_json.get("items", []):
        provider = (item.get("provider") or {}).get("name") or ""
        if "live" in provider.lower():
            continue
        home_side = item.get("homeTeamOdds") or {}
        away_side = item.get("awayTeamOdds") or {}
        rows.append({
            "date": date,
            "event_id": str(event.get("id")),
            "home_team": home,
            "away_team": away,
            "start_time": event.get("date"),
            "status": status,
            "provider": provider,
            "home_ml": parse_american(home_side.get("moneyLine")),
            "away_ml": parse_american(away_side.get("moneyLine")),
            "home_ml_open": _snapshot_moneyline(home_side, "open"),
            "away_ml_open": _snapshot_moneyline(away_side, "open"),
            "home_ml_close": _snapshot_moneyline(home_side, "close"),
            "away_ml_close": _snapshot_moneyline(away_side, "close"),
        })
    return rows


def consensus_closing_lines(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per game: each book's closing moneyline (its `close`
    snapshot, else its last pre-game `moneyLine`), the median across books
    for each side, and the de-vigged home win probability from those
    medians. Only finished games, and only books quoting both sides.

    The median across books is the market's price, robust to one book's
    stale or mistyped line. With one book (2024 on, mostly ESPN BET only)
    it is simply that book's line."""
    columns = ["date", "event_id", "home_team", "away_team", "start_time", "n_books",
               "home_ml", "away_ml", "home_ml_open", "away_ml_open", "market_home_win_probability"]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    df = rows[rows["status"].fillna("").str.contains("FINAL")].copy()
    df["home_close"] = df["home_ml_close"].where(df["home_ml_close"].notna(), df["home_ml"])
    df["away_close"] = df["away_ml_close"].where(df["away_ml_close"].notna(), df["away_ml"])
    df = df[df["home_close"].notna() & df["away_close"].notna()]
    if df.empty:
        return pd.DataFrame(columns=columns)
    # Medians are taken on implied probability, not on American odds: the
    # American scale jumps from -100 to +100, so -105 and +105 are
    # neighbours in price but 210 apart as numbers.
    for col in ("home_close", "away_close", "home_ml_open", "away_ml_open"):
        df[col + "_p"] = implied(df[col])
    games = df.groupby(["date", "event_id", "home_team", "away_team", "start_time"], as_index=False).agg(
        n_books=("provider", "nunique"),
        home_p=("home_close_p", "median"),
        away_p=("away_close_p", "median"),
        home_open_p=("home_ml_open_p", "median"),
        away_open_p=("away_ml_open_p", "median"),
    )
    games["home_ml"] = to_american(games["home_p"])
    games["away_ml"] = to_american(games["away_p"])
    games["home_ml_open"] = to_american(games["home_open_p"])
    games["away_ml_open"] = to_american(games["away_open_p"])
    games["market_home_win_probability"] = games["home_p"] / (games["home_p"] + games["away_p"])
    return games[columns]


def to_american(probability):
    """Implied probability (with the vig) -> American odds; NaN stays NaN."""
    p = np.asarray(probability, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p >= 0.5, -100 * p / (1 - p), 100 * (1 - p) / p)


def implied(moneyline):
    """Vectorized American odds -> implied probability (with the vig)."""
    moneyline = np.asarray(moneyline, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(moneyline < 0, -moneyline / (-moneyline + 100), 100 / (moneyline + 100))
