"""College basketball betting lines from ESPN's core API: the point spread,
the over/under and both moneylines for a game, for testing the ratings
(cbb_ratings.py) against a real line.

Why (2026-11-15): the college model had only ever been scored against
results. Whether its disagreements with a sportsbook win is a different
question - the one a bettor needs answered - and it needs the lines.

The games come from the schedules the ratings already use (their game_id
is ESPN's event id), so each game is one request and the rows join
straight back. This module only parses; scripts/backfill_cbb_odds.py
fetches. One row per (game, sportsbook), "Live Odds" feeds excluded.

ESPN's field names are the same across sports (historical_odds.py reads
the MLB moneylines from the same response): `spread` is the home team's
(negative = home favored), `overUnder` the total, and where a book kept
snapshots, `open` and `close` hold the first and last pre-game numbers.
"""

import numpy as np
import pandas as pd

from mlb_metrics import historical_odds

ROW_COLUMNS = [
    "game_id", "provider", "spread", "total", "home_ml", "away_ml",
    "spread_open", "spread_close", "total_open", "total_close",
    "home_ml_open", "away_ml_open", "home_ml_close", "away_ml_close",
]


def parse_points(value):
    """'-7.5' / '+3' / 'PK' / 'o145.5' / 145.5 -> a float. None when missing
    or unparseable."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return None if pd.isna(value) else float(value)
    text = str(value).strip().upper()
    if text in ("PK", "PICK", "EVEN", "EV"):
        return 0.0
    text = text.lstrip("OU").replace("+", "")
    try:
        return float(text)
    except ValueError:
        return None


def _snapshot_spread(side: dict, snapshot: str):
    """One side's point spread in its `open` or `close` snapshot."""
    block = (side or {}).get(snapshot)
    if not isinstance(block, dict):
        return None
    spread = block.get("pointSpread")
    if isinstance(spread, dict):
        return parse_points(spread.get("american") or spread.get("alternateDisplayValue") or spread.get("value"))
    return parse_points(spread if spread is not None else block.get("spread"))


def _snapshot_total(item: dict, snapshot: str):
    block = item.get(snapshot)
    if not isinstance(block, dict):
        return None
    total = block.get("total")
    if isinstance(total, dict):
        return parse_points(total.get("american") or total.get("alternateDisplayValue") or total.get("value"))
    if total is not None:
        return parse_points(total)
    over = block.get("over")
    if isinstance(over, dict):
        return parse_points(over.get("american") or over.get("alternateDisplayValue") or over.get("value"))
    return None


def _home_spread(item: dict, home_side: dict, away_side: dict, snapshot: str):
    """The home team's spread in a snapshot, from either side's number."""
    home = _snapshot_spread(home_side, snapshot)
    if home is not None:
        return home
    away = _snapshot_spread(away_side, snapshot)
    return None if away is None else -away


def parse_game_odds(game_id, odds_json: dict) -> list:
    """Rows for one game's core-API odds response."""
    rows = []
    for item in (odds_json or {}).get("items", []):
        provider = (item.get("provider") or {}).get("name") or ""
        if "live" in provider.lower():
            continue
        home_side = item.get("homeTeamOdds") or {}
        away_side = item.get("awayTeamOdds") or {}
        rows.append({
            "game_id": int(game_id),
            "provider": provider,
            "spread": parse_points(item.get("spread")),
            "total": parse_points(item.get("overUnder")),
            "home_ml": historical_odds.parse_american(home_side.get("moneyLine")),
            "away_ml": historical_odds.parse_american(away_side.get("moneyLine")),
            "spread_open": _home_spread(item, home_side, away_side, "open"),
            "spread_close": _home_spread(item, home_side, away_side, "close"),
            "total_open": _snapshot_total(item, "open"),
            "total_close": _snapshot_total(item, "close"),
            "home_ml_open": historical_odds._snapshot_moneyline(home_side, "open"),
            "away_ml_open": historical_odds._snapshot_moneyline(away_side, "open"),
            "home_ml_close": historical_odds._snapshot_moneyline(home_side, "close"),
            "away_ml_close": historical_odds._snapshot_moneyline(away_side, "close"),
        })
    return rows


def consensus_lines(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per game: the median across books of the closing spread and
    total (a book's `close` snapshot, else its last pre-game number), the
    opening ones where any book kept them, and the closing moneylines.
    Columns: game_id, n_books, spread, total, spread_open, total_open,
    home_ml, away_ml."""
    columns = ["game_id", "n_books", "spread", "total", "spread_open", "total_open", "home_ml", "away_ml"]
    if rows is None or rows.empty:
        return pd.DataFrame(columns=columns)
    df = rows.copy()
    for name in ("spread", "total", "home_ml", "away_ml"):
        df[name + "_last"] = df[name + "_close"].where(df[name + "_close"].notna(), df[name])
    df = df[df["spread_last"].notna() | df["total_last"].notna()]
    if df.empty:
        return pd.DataFrame(columns=columns)
    for side in ("home", "away"):
        df[side + "_p"] = historical_odds.implied(df[side + "_ml_last"])
    games = df.groupby("game_id", as_index=False).agg(
        n_books=("provider", "nunique"),
        spread=("spread_last", "median"),
        total=("total_last", "median"),
        spread_open=("spread_open", "median"),
        total_open=("total_open", "median"),
        home_p=("home_p", "median"),
        away_p=("away_p", "median"),
    )
    games["home_ml"] = historical_odds.to_american(games["home_p"])
    games["away_ml"] = historical_odds.to_american(games["away_p"])
    return games[columns].replace({np.inf: np.nan, -np.inf: np.nan})
