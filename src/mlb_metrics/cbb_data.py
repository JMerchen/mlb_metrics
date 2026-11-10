"""College basketball data: hoopR's ESPN schedules and team box scores
(github.com/sportsdataverse/hoopR-mbb-data, updated daily in season), one
parquet file per season. A season is named for the year it ends in
(2026-27 is 2027)."""

import io
import os
import time

import pandas as pd
import requests

BASE_URL = "https://raw.githubusercontent.com/sportsdataverse/hoopR-mbb-data/main/mbb"
TABLES = {
    "schedules": "schedules/parquet/mbb_schedule_{season}.parquet",
    "team_box": "team_box/parquet/team_box_{season}.parquet",
}
# The columns the ratings and the site use (the schedule file carries 80+).
SCHEDULE_COLUMNS = [
    "game_id", "season", "season_type", "game_date", "neutral_site", "status_type_completed", "notes_headline",
    "home_id", "home_location", "home_short_display_name", "home_abbreviation", "home_conference_id", "home_color",
    "home_logo", "home_score", "home_current_rank",
    "away_id", "away_location", "away_short_display_name", "away_abbreviation", "away_conference_id", "away_color",
    "away_logo", "away_score", "away_current_rank",
]
BOX_COLUMNS = ["game_id", "team_id", "field_goals_attempted", "free_throws_attempted", "offensive_rebounds",
               "total_turnovers", "turnovers"]


def table_url(table: str, season: int) -> str:
    return f"{BASE_URL}/{TABLES[table].format(season=season)}"


def fetch_table(table: str, season: int, retries: int = 3, timeout: int = 120) -> pd.DataFrame:
    """One season's `table` ("schedules" or "team_box"), cut to the columns
    used. Raises after `retries` failed tries."""
    wanted = SCHEDULE_COLUMNS if table == "schedules" else BOX_COLUMNS
    last = None
    for attempt in range(retries):
        try:
            response = requests.get(table_url(table, season), timeout=timeout)
            response.raise_for_status()
            frame = pd.read_parquet(io.BytesIO(response.content))
            return frame[[c for c in wanted if c in frame.columns]]
        except Exception as exc:  # network or parse: try again, then give up
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"could not fetch {table} {season}: {last}")


def load_table(table: str, season: int, cache_dir: str | None = None, refresh: bool = False) -> pd.DataFrame:
    """fetch_table through a local cache: `cache_dir`/<table>_<season>.parquet
    is read when it's there (and `refresh` is false) and written after a
    fetch. No cache without `cache_dir`."""
    path = os.path.join(cache_dir, f"{table}_{season}.parquet") if cache_dir else None
    if path and os.path.exists(path) and not refresh:
        return pd.read_parquet(path)
    frame = fetch_table(table, season)
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        frame.to_parquet(path, index=False)
    return frame
