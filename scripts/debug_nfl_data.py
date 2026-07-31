"""One-off diagnostic: print the real shape of nfl_data_py's (nflverse)
responses, to settle field names/historical depth/DST-input availability
before nfl_data.py's ingestion logic is written against assumed shapes.
This sandbox has no network access at all - see
.github/workflows/debug_nfl_data.yml, same bootstrapping pattern as
scripts/debug_statsapi.py (that script's own docstring explains why this
convention exists).

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

    import nfl_data_py as nfl

    print(f"nfl_data_py version: {getattr(nfl, '__version__', 'unknown')}")
    print(f"Querying season: {season}")

    weekly = nfl.import_weekly_data([season])
    _print_frame("import_weekly_data", weekly)
    if "position" in weekly.columns:
        print("distinct positions:", sorted(weekly["position"].dropna().unique().tolist()))
    if "season_type" in weekly.columns:
        print("distinct season_type:", sorted(weekly["season_type"].dropna().unique().tolist()))

    schedules = nfl.import_schedules([season])
    _print_frame("import_schedules", schedules)
    if "weekday" in schedules.columns:
        print("distinct weekday:", sorted(schedules["weekday"].dropna().unique().tolist()))
    if "game_type" in schedules.columns:
        print("distinct game_type:", sorted(schedules["game_type"].dropna().unique().tolist()))

    try:
        injuries = nfl.import_injuries([season])
        _print_frame("import_injuries", injuries)
        if "report_status" in injuries.columns:
            print("distinct report_status:", sorted(injuries["report_status"].dropna().unique().tolist()))
    except Exception as exc:
        print(f"\n=== import_injuries === FAILED: {exc}")

    try:
        rosters = nfl.import_rosters([season])
        _print_frame("import_rosters", rosters)
    except Exception as exc:
        print(f"\n=== import_rosters === FAILED: {exc}")

    try:
        ids = nfl.import_ids()
        _print_frame("import_ids", ids, n_rows=1)
    except Exception as exc:
        print(f"\n=== import_ids === FAILED: {exc}")

    # DST inputs: check for a ready-made team-defense table first.
    try:
        team_stats = nfl.import_team_desc()
        _print_frame("import_team_desc", team_stats)
    except Exception as exc:
        print(f"\n=== import_team_desc === FAILED (may not exist): {exc}")

    # Fallback: play-by-play columns needed to aggregate DST stats by hand.
    try:
        pbp = nfl.import_pbp_data([season], downcast=True)
        defense_cols = [
            c for c in pbp.columns
            if c in {
                "defteam", "sack", "interception", "fumble_lost", "touchdown", "td_team",
                "return_touchdown", "safety", "week", "posteam", "home_team", "away_team",
            }
        ]
        print(f"\n=== import_pbp_data ({season}) - defense-relevant columns only ===")
        print(f"full shape: {pbp.shape}")
        print("defense-relevant columns present:", sorted(defense_cols))
        if defense_cols:
            print(pbp[defense_cols].head(5).to_string())
    except Exception as exc:
        print(f"\n=== import_pbp_data === FAILED: {exc}")

    # Historical depth probe: try a much older season and see what happens.
    for probe_season in (2015, 2010, 1999):
        try:
            probe = nfl.import_weekly_data([probe_season])
            print(f"\nimport_weekly_data([{probe_season}]) -> {len(probe)} rows")
        except Exception as exc:
            print(f"\nimport_weekly_data([{probe_season}]) -> FAILED: {exc}")


if __name__ == "__main__":
    main()
