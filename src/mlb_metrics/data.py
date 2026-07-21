"""Data acquisition and persistence.

Historically the pipeline pulled the full season from Statcast fresh on every
run and discarded it after computing that day's metrics - nothing was ever
saved, so there was no way to re-run the pipeline against a past date or
build a backtest. `persist_raw_statcast`/`load_persisted_statcast` fix that by
appending each day's pull to a per-season parquet file that gets committed
alongside the output CSVs.
"""

import os

import pandas as pd

from mlb_metrics import config

# Columns that uniquely identify a single pitch, used to dedupe when merging
# a fresh Statcast pull into the persisted raw dataset.
PITCH_KEY_COLUMNS = ["game_pk", "at_bat_number", "pitch_number"]


def fetch_statcast_range(start_dt, end_dt) -> pd.DataFrame:
    """Pull pitch-by-pitch Statcast data for [start_dt, end_dt] and normalize game_date to datetime."""
    from pybaseball import statcast

    df = statcast(start_dt=str(start_dt), end_dt=str(end_dt))
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def raw_statcast_path(raw_dir: str, season: int) -> str:
    return os.path.join(raw_dir, f"statcast_{season}.parquet")


def load_persisted_statcast(raw_dir: str, season: int) -> pd.DataFrame | None:
    path = raw_statcast_path(raw_dir, season)
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def persist_raw_statcast(df: pd.DataFrame, raw_dir: str, season: int) -> pd.DataFrame:
    """Merge `df` into the persisted raw dataset for `season`, dedupe by pitch, and save.

    Returns the full merged dataset (existing history + new rows).
    """
    os.makedirs(raw_dir, exist_ok=True)
    path = raw_statcast_path(raw_dir, season)

    existing = load_persisted_statcast(raw_dir, season)
    combined = pd.concat([existing, df], ignore_index=True) if existing is not None else df.copy()

    key_columns = [c for c in PITCH_KEY_COLUMNS if c in combined.columns]
    if key_columns:
        combined = combined.drop_duplicates(subset=key_columns, keep="last")
    combined = combined.sort_values("game_date").reset_index(drop=True)

    combined.to_parquet(path, index=False)
    return combined


_name_register_cache: pd.DataFrame | None = None


def get_name_register() -> pd.DataFrame:
    """Cached wrapper around chadwick_register(); the original script called this
    four separate times per run for the same static player-ID lookup table."""
    global _name_register_cache
    if _name_register_cache is None:
        from pybaseball import chadwick_register

        _name_register_cache = chadwick_register()
    return _name_register_cache


def assign_game_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Assign a per-row `game_id` by detecting new games via a reset in at_bat_number.

    Mirrors the original `create_id` logic, which was duplicated verbatim in
    both the Hit_Prob and Strength sections of the monolithic script.
    """
    gam_id = (
        df[["game_date", "home_team", "away_team", "pitcher", "at_bat_number"]]
        .drop_duplicates()
        .iloc[::-1]
    )

    count = 0
    game_ids = []
    for val in gam_id["at_bat_number"]:
        if val == 1:
            count += 1
        game_ids.append(count)
    gam_id = gam_id.copy()
    gam_id["game_id"] = game_ids

    data = df.merge(
        gam_id, on=["game_date", "home_team", "away_team", "pitcher", "at_bat_number"]
    )
    data["ind"] = (
        data["game_id"].astype("str")
        + data["at_bat_number"].astype("str")
        + data["pitch_number"].astype("str")
    ).astype("int")
    return data.set_index("ind").sort_index()


def latest_team_for_batters(df: pd.DataFrame) -> pd.DataFrame:
    """Most recent team each batter appeared for, keyed by key_mlbam."""
    home = (
        df[df["inning_topbot"] == "Bot"][["batter", "home_team", "game_date"]]
        .drop_duplicates()
        .rename(columns={"home_team": "team"})
    )
    away = (
        df[df["inning_topbot"] == "Top"][["batter", "away_team", "game_date"]]
        .drop_duplicates()
        .rename(columns={"away_team": "team"})
    )
    team = pd.concat([home, away])

    latest_player = df[["game_date", "batter"]].groupby("batter", as_index=False).max()
    return (
        latest_player.merge(team, on=["batter", "game_date"], how="left")[["batter", "team"]]
        .rename(columns={"batter": "key_mlbam"})
    )


def latest_team_for_pitchers(df: pd.DataFrame) -> pd.DataFrame:
    """Most recent team each pitcher appeared for, keyed by key_mlbam."""
    homep = (
        df[df["inning_topbot"] == "Bot"][["pitcher", "away_team", "game_date"]]
        .drop_duplicates()
        .rename(columns={"away_team": "team"})
    )
    awayp = (
        df[df["inning_topbot"] == "Top"][["pitcher", "home_team", "game_date"]]
        .drop_duplicates()
        .rename(columns={"home_team": "team"})
    )
    teamp = pd.concat([homep, awayp])

    latest_pitcher = df[["game_date", "pitcher"]].groupby("pitcher", as_index=False).max()
    return (
        latest_pitcher.merge(teamp, on=["pitcher", "game_date"], how="left")[["pitcher", "team"]]
        .rename(columns={"pitcher": "key_mlbam"})
    )


def completed_events(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Filter pitch-by-pitch rows down to one row per completed plate-appearance event."""
    return df[df["events"].isin(config.COUNTED_EVENTS)][columns]
