"""NFL data acquisition and persistence, via `nflreadpy` (the actively-
maintained nflverse successor to the now-deprecated `nfl_data_py` - see
README.md's NFL DFS section for the real deprecation-notice citation).
Every real field name/shape referenced anywhere in this module is
confirmed via a live run of `scripts/debug_nfl_data.py`
(`.github/workflows/debug_nfl_data.yml`, run 30675199512, 2026-08-01) -
not assumed.

`nflreadpy` returns Polars DataFrames by default; every fetch function
here converts to pandas via `.to_pandas()` immediately, since the rest of
this codebase is pandas-only.

## Persistence deliberately diverges from data.py's Statcast pattern

`data.py`'s `persist_raw_statcast` appends each day's pull to a growing
per-season file, because Statcast is pitch-level (huge volume) and
re-fetching a whole season on every run would be wasteful. NFL weekly
data is the opposite shape on both axes:

- **Volume is tiny** (a few thousand rows/season across all tables
  combined - confirmed live: `load_player_stats` alone is ~19k rows for a
  full season), so re-fetching a whole season is cheap.
- **nflverse is known to retroactively correct published stats** (a
  scoring correction, a stat-tracking fix) after initial release. An
  append-and-dedupe pattern like Statcast's would let a since-corrected
  row linger forever unless the dedupe key happened to change too.

So `persist_table` here is a **whole-file overwrite**, not an
incremental merge - correct for both the "current season, re-fetched
regularly" case (a fresh pull always wins) and the "already-complete
historical season" case (fetched once, then left alone by the caller
simply not re-fetching it - see `scripts/fetch_nfl_historical.py`, which
skips any season already persisted rather than baking a "never overwrite
history" rule into this module itself).
"""

import os

import pandas as pd


def table_path(raw_dir: str, table: str, season: int) -> str:
    return os.path.join(raw_dir, f"{table}_{season}.parquet")


def load_persisted_table(raw_dir: str, table: str, season: int) -> pd.DataFrame | None:
    path = table_path(raw_dir, table, season)
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def persist_table(df: pd.DataFrame, raw_dir: str, table: str, season: int) -> None:
    """Whole-file overwrite - see module docstring for why this diverges
    from data.py's incremental-append Statcast pattern."""
    path = table_path(raw_dir, table, season)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_parquet(path, index=False)


def fetch_weekly_stats(seasons: list[int]) -> pd.DataFrame:
    """Per-player-per-week stat lines - confirmed columns include
    `player_id` (GSIS-format id, the stable cross-table join key),
    `position`, `team`, `opponent_team`, `season`/`week`/`season_type`
    (REG/POST), and every real DK-scoring-relevant stat:
    passing_yards/passing_tds/passing_interceptions,
    rushing_yards/rushing_tds, receiving_yards/receiving_tds/receptions/
    targets. Fumbles lost are split BY CATEGORY
    (rushing_fumbles_lost/receiving_fumbles_lost/sack_fumbles_lost), not
    one combined column - confirmed live, contrary to the Phase 0 plan's
    assumption of a single combined field. nflverse's own
    fantasy_points/fantasy_points_ppr are also present - useful as a
    cross-check, NOT a substitute for computing real DK Classic scoring
    (see nfl_dfs.py)."""
    import nflreadpy

    return nflreadpy.load_player_stats(seasons).to_pandas()


def fetch_schedules(seasons: list[int]) -> pd.DataFrame:
    """One row per game (regular season AND playoffs - filter `game_type`
    for regular-season-only use). Confirmed columns include `game_id`,
    `week`, `gameday`, `weekday` (real values confirmed:
    Thursday/Friday/Saturday/Sunday/Monday), `gametime`, `home_team`/
    `away_team`, `home_score`/`away_score`. Real bonus confirmed live:
    `home_qb_id`/`away_qb_id`/`home_qb_name`/`away_qb_name` are given
    directly on the schedule row itself - a starting-QB signal doesn't
    need injury-report inference. Bye weeks are simply absent rows (no
    explicit bye marker), confirmed by real season row counts."""
    import nflreadpy

    return nflreadpy.load_schedules(seasons).to_pandas()


def fetch_injuries(seasons: list[int]) -> pd.DataFrame:
    """One row per player per week with an actual injury report entry -
    a healthy player is simply ABSENT from this table, not given an
    explicit "Healthy" status (confirmed live). `report_status`'s real
    distinct values are exactly `Doubtful`/`Out`/`Questionable`. `gsis_id`
    is the same id space as `fetch_weekly_stats`'s `player_id`. No
    explicit day-of-week/report-timing column was found in the real
    columns - distinguishing a Wednesday snapshot from a Friday one (see
    nfl_schedule.py's future Friday-refresh design) will have to rely on
    WHEN this is fetched, not a field in the data itself."""
    import nflreadpy

    return nflreadpy.load_injuries(seasons).to_pandas()


def fetch_rosters_weekly(seasons: list[int]) -> pd.DataFrame:
    """One row per player per week (not per season) - confirmed real
    columns include `gsis_id`, `position`, `depth_chart_position`,
    `status`, `team`. Weekly (not season-level `fetch_rosters`/
    `load_rosters`) is used here specifically so a mid-season trade or
    signing resolves to the correct team for the week in question, not a
    single season-long snapshot."""
    import nflreadpy

    return nflreadpy.load_rosters_weekly(seasons).to_pandas()


def fetch_team_stats(seasons: list[int]) -> pd.DataFrame:
    """One row per team per week (per opponent) with real, already-
    aggregated team-level defense/special-teams box-score stats -
    confirmed live to include def_sacks/def_interceptions/def_tds/
    def_safeties/def_fumbles_forced/special_teams_tds/kickoff_return_yards
    etc. directly, with NO play-by-play aggregation needed for DST
    scoring (nfl_dst.py) - a real, positive simplification versus the
    Phase 0 plan's assumption that PBP aggregation would likely be
    required. Does NOT include points-allowed (a team's own row is its
    OWN stats, not what it allowed) - that comes from a simple join
    against `fetch_schedules`' opponent score instead."""
    import nflreadpy

    return nflreadpy.load_team_stats(seasons=seasons).to_pandas()
