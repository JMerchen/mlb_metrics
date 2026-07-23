"""Historical player data from Lahman's Baseball Database, read from CSVs
committed under `Lahman_Raw/` at the repo root (People.csv/Batting.csv/
Pitching.csv, sourced from SABR's box.com mirror of the database) - a
manually-refreshed source of truth, not a live fetch. Used by age_curve.py
to build KNN comparables for the Age Curves page.

Two automated fetch paths were tried first and both failed:
- pybaseball's own built-in `lahman` submodule (`pybaseball.lahman`)
  downloads a zip from
  `https://github.com/chadwickbureau/baseballdatabank/archive/master.zip`
  at call time, and that download started failing (`zipfile.BadZipFile`) -
  confirmed reproducing in real GitHub Actions CI with full internet
  access, so it's an upstream break, not a local network issue.
- The `lahman` PyPI package (no runtime fetch needed, data bundled in the
  wheel) turned out to be a single abandoned release from 2022 with data
  frozen at the 2020 season - stale by half a decade.

`Lahman_Raw/` sidesteps both: the CSVs are downloaded by hand from SABR's
mirror (https://sabr.app.box.com - Lahman's database is now hosted there)
and committed directly, alongside a `Lahman_Raw/Readme.md` documenting how
to refresh them. This data is historical and mostly static (it updates
roughly once a season, when a completed season is added), unlike this
project's own Statcast data which grows daily - so like the two fetch
paths before it, it's converted to data/raw/lahman/*.parquet by its own
separate, occasionally-run script (scripts/fetch_lahman.py), never the
daily pipeline. Persistence mirrors data.py's persist_raw_statcast/
load_persisted_statcast shape (parquet under data/raw/, load-if-exists)
but as a whole-table overwrite each refresh - Lahman doesn't stream
incrementally per-event the way Statcast does, so there's nothing to
merge/dedupe against.

`Lahman_Raw/People.csv` carries an extra leading `ID` column (a row
number, not `playerID`) that SABR's export includes and Lahman's own
distribution doesn't - harmless, just unused, since every read here
selects columns by name.

Bridging this project's own player identity (`key_mlbam`, from Statcast)
to Lahman's own `playerID` needs a two-hop crosswalk: chadwick_register()
(already used by data.get_name_register) carries `key_mlbam` alongside
`key_bbref`; Lahman's People table is keyed by `playerID` but carries a
`bbrefID` column. See build_crosswalk.
"""

import os

import pandas as pd


def fetch_people(lahman_raw_dir: str = "Lahman_Raw") -> pd.DataFrame:
    return pd.read_csv(os.path.join(lahman_raw_dir, "People.csv"))


def fetch_batting(lahman_raw_dir: str = "Lahman_Raw") -> pd.DataFrame:
    return pd.read_csv(os.path.join(lahman_raw_dir, "Batting.csv"))


def fetch_pitching(lahman_raw_dir: str = "Lahman_Raw") -> pd.DataFrame:
    return pd.read_csv(os.path.join(lahman_raw_dir, "Pitching.csv"))


def lahman_table_path(raw_dir: str, name: str) -> str:
    return os.path.join(raw_dir, "lahman", f"{name}.parquet")


def load_persisted_lahman_table(raw_dir: str, name: str) -> pd.DataFrame | None:
    path = lahman_table_path(raw_dir, name)
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def persist_lahman_table(df: pd.DataFrame, raw_dir: str, name: str) -> None:
    """Whole-table overwrite, not an incremental merge - see module
    docstring."""
    path = lahman_table_path(raw_dir, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


def build_crosswalk(chadwick_register: pd.DataFrame, lahman_people: pd.DataFrame) -> pd.DataFrame:
    """One row per player resolvable on both sides: [key_mlbam, playerID].
    key_mlbam -> key_bbref (chadwick_register) -> bbrefID (Lahman People)
    -> playerID (Lahman's own key, used to join Batting/Pitching). A
    player missing from either side (not yet in Lahman - a very recent
    call-up/international signee - or missing from the name register) is
    excluded, not defaulted - there's no meaningful neutral crosswalk for
    a missing identity join."""
    bridge = chadwick_register[["key_mlbam", "key_bbref"]].dropna()
    people = lahman_people[["playerID", "bbrefID"]].dropna()
    crosswalk = bridge.merge(people, left_on="key_bbref", right_on="bbrefID", how="inner")
    return crosswalk[["key_mlbam", "playerID"]].drop_duplicates()


def attach_age(seasons: pd.DataFrame, people: pd.DataFrame, year_column: str = "yearID") -> pd.DataFrame:
    """Inner-joins `seasons` (any table with a `playerID` column and a
    `year_column` - a full Lahman Batting table across 150+ years, or a
    single synthetic row for a current-season player) to `people` on
    `playerID`, adding a whole-years `age` column: age as of June 30 of
    that row's own `year_column` value (the standard baseball-reference/
    MLB convention for a player's "age that season"). Fully vectorized -
    no per-year looping, since each row supplies its own reference year.

    Rows for a player with no resolvable birth year are dropped (age is
    meaningless without one); a missing birth month/day defaults to July
    1st (mid-year - a small, bounded error) rather than dropping the row
    entirely over incomplete-but-still-informative birth data."""
    people = people.dropna(subset=["birthYear"])[["playerID", "birthYear", "birthMonth", "birthDay"]].copy()
    people["birthMonth"] = people["birthMonth"].fillna(7)
    people["birthDay"] = people["birthDay"].fillna(1)
    people["birth_date"] = pd.to_datetime(
        {
            "year": people["birthYear"].astype(int),
            "month": people["birthMonth"].astype(int),
            "day": people["birthDay"].astype(int),
        },
        errors="coerce",
    )
    people = people.dropna(subset=["birth_date"])

    merged = seasons.merge(people[["playerID", "birth_date"]], on="playerID", how="inner")
    reference = pd.to_datetime(merged[year_column].astype(int).astype(str) + "-06-30")
    merged["age"] = ((reference - merged["birth_date"]).dt.days / 365.25).astype(int)
    return merged.drop(columns=["birth_date"])
