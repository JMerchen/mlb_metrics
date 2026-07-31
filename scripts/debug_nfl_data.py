"""One-off diagnostic: print the real shape of nflreadpy's (nflverse)
responses, to settle field names/historical depth/DST-input availability
before nfl_data.py's ingestion logic is written against assumed shapes.
This sandbox has no network access at all - see
.github/workflows/debug_nfl_data.yml, same bootstrapping pattern as
scripts/debug_statsapi.py (that script's own docstring explains why this
convention exists).

nflreadpy, not nfl_data_py: nfl_data_py's own maintainers deprecated it
(no releases in over a year, official "switch immediately" notice) in
favor of nflreadpy, the actively-maintained successor from the same
nflverse project - confirmed via a live web search before writing this,
not assumed. nflreadpy returns Polars DataFrames by default; every
`load_*` call here is immediately converted via `.to_pandas()` so the
printed shapes match what the rest of this pandas-only codebase will
actually see.

Usage: python scripts/debug_nfl_data.py [SEASON]  (defaults to the most
recently completed season)
Delete this script (and its workflow) once nfl_data.py is implemented and
the real shape is confirmed - it's a bootstrapping tool, not part of the
pipeline.
"""

import datetime
import sys


def _print_frame(label, df, n_rows=1):
    print(f"\n=== {label} ===")
    print(f"shape: {df.shape}")
    print("columns:", sorted(df.columns))
    print("dtypes:\n", df.dtypes)
    if len(df):
        with __import__("pandas").option_context("display.max_columns", None, "display.width", 200):
            print(f"first {n_rows} row(s):\n", df.head(n_rows).to_string())
    else:
        print("(empty)")


def main():
    season = int(sys.argv[1]) if len(sys.argv) > 1 else datetime.date.today().year - 1

    import nflreadpy as nfl

    print(f"nflreadpy version: {getattr(nfl, '__version__', 'unknown')}")
    print(f"get_current_season(): {nfl.get_current_season() if hasattr(nfl, 'get_current_season') else 'n/a'}")
    print(f"get_current_week(): {nfl.get_current_week() if hasattr(nfl, 'get_current_week') else 'n/a'}")
    print(f"Querying season: {season}")

    weekly = nfl.load_player_stats([season]).to_pandas()
    _print_frame("load_player_stats", weekly)
    if "position" in weekly.columns:
        print("distinct positions:", sorted(weekly["position"].dropna().unique().tolist()))
    if "season_type" in weekly.columns:
        print("distinct season_type:", sorted(weekly["season_type"].dropna().unique().tolist()))

    schedules = nfl.load_schedules([season]).to_pandas()
    _print_frame("load_schedules", schedules)
    if "weekday" in schedules.columns:
        print("distinct weekday:", sorted(schedules["weekday"].dropna().unique().tolist()))
    if "game_type" in schedules.columns:
        print("distinct game_type:", sorted(schedules["game_type"].dropna().unique().tolist()))

    try:
        injuries = nfl.load_injuries([season]).to_pandas()
        _print_frame("load_injuries", injuries)
        if "report_status" in injuries.columns:
            print("distinct report_status:", sorted(injuries["report_status"].dropna().unique().tolist()))
    except Exception as exc:
        print(f"\n=== load_injuries === FAILED: {exc}")

    try:
        rosters = nfl.load_rosters([season]).to_pandas()
        _print_frame("load_rosters", rosters)
    except Exception as exc:
        print(f"\n=== load_rosters === FAILED: {exc}")

    try:
        rosters_weekly = nfl.load_rosters_weekly([season]).to_pandas()
        _print_frame("load_rosters_weekly", rosters_weekly)
    except Exception as exc:
        print(f"\n=== load_rosters_weekly === FAILED: {exc}")

    try:
        players = nfl.load_players().to_pandas()
        _print_frame("load_players", players, n_rows=1)
    except Exception as exc:
        print(f"\n=== load_players === FAILED: {exc}")

    # DST inputs: check for a ready-made team-stats table first (nflreadpy
    # advertises load_team_stats - if this already carries sacks/INTs/
    # points-allowed/defensive-TDs by team, it may remove the need for the
    # PBP-aggregation fallback below entirely).
    try:
        team_stats = nfl.load_team_stats(seasons=[season]).to_pandas()
        _print_frame("load_team_stats", team_stats)
    except Exception as exc:
        print(f"\n=== load_team_stats === FAILED: {exc}")

    # Fallback: play-by-play columns needed to aggregate DST stats by hand,
    # in case load_team_stats above doesn't carry everything needed.
    try:
        pbp = nfl.load_pbp([season]).to_pandas()
        defense_cols = [
            c for c in pbp.columns
            if c in {
                "defteam", "sack", "interception", "fumble_lost", "touchdown", "td_team",
                "return_touchdown", "safety", "week", "posteam", "home_team", "away_team",
            }
        ]
        print(f"\n=== load_pbp ({season}) - defense-relevant columns only ===")
        print(f"full shape: {pbp.shape}")
        print("defense-relevant columns present:", sorted(defense_cols))
        if defense_cols:
            print(pbp[defense_cols].head(5).to_string())
    except Exception as exc:
        print(f"\n=== load_pbp === FAILED: {exc}")

    # Historical depth probe: try much older seasons and see what happens.
    for probe_season in (2015, 2010, 1999):
        try:
            probe = nfl.load_player_stats([probe_season]).to_pandas()
            print(f"\nload_player_stats([{probe_season}]) -> {len(probe)} rows")
        except Exception as exc:
            print(f"\nload_player_stats([{probe_season}]) -> FAILED: {exc}")


if __name__ == "__main__":
    main()
