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
Kept around (not deleted) as a reusable bootstrapping tool - reused most
recently (2026-08-11) to confirm load_ff_rankings/load_ff_playerids' real
shape for the NFL Bestball Draft Assistant's real ECR-based reach/value
signal, the same way it was reused earlier to confirm fetch_snap_counts's
real shape.
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
    # The workflow always passes the season input as an argument, an empty
    # string when the workflow_dispatch field is left blank - not simply
    # absent - so a truthiness check is needed, not just len(sys.argv).
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    season = int(arg) if arg else datetime.date.today().year - 1

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

    # Draft Assistant Phase 0 (2026-08-11): real FantasyPros Expert Consensus
    # Rank data + a real crosswalk to this project's own player_id (GSIS id)
    # space, to serve as a real, honestly-labeled substitute for raw ADP -
    # every raw ADP-provider domain is blocked by this sandbox's egress
    # policy (confirmed live: Fantasy Football Calculator, FantasyPros,
    # MyFantasyLeague, even nflverse's own docs domain all return
    # EGRESS_BLOCKED), and this project has no HTTP client dependency, so
    # nothing about load_ff_rankings/load_ff_playerids' real shape is
    # assumed here - print everything needed to design nfl_ff_rankings.py
    # against real, confirmed field names.
    ff_rankings = None
    try:
        ff_rankings = nfl.load_ff_rankings().to_pandas()
        _print_frame("load_ff_rankings", ff_rankings, n_rows=5)
        for candidate_col in ("page", "type", "scoring", "format", "ecr_type", "position"):
            if candidate_col in ff_rankings.columns:
                print(f"distinct {candidate_col}:", sorted(ff_rankings[candidate_col].dropna().unique().tolist()))
    except Exception as exc:
        print(f"\n=== load_ff_rankings === FAILED: {exc}")

    ff_playerids = None
    try:
        ff_playerids = nfl.load_ff_playerids().to_pandas()
        _print_frame("load_ff_playerids", ff_playerids, n_rows=5)
    except Exception as exc:
        print(f"\n=== load_ff_playerids === FAILED: {exc}")

    # Real crosswalk overlap check, mirroring the same real check already
    # done for the pfr_id crosswalk (compute_player_snap_share) - only
    # meaningful if both real frames above loaded AND a real committed
    # rosters_weekly parquet is present in this checkout.
    if ff_playerids is not None:
        import os

        rosters_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "nfl", f"rosters_weekly_{season}.parquet")
        if os.path.exists(rosters_path):
            import pandas as pd

            rosters_weekly = pd.read_parquet(rosters_path)
            real_gsis_ids = set(rosters_weekly["gsis_id"].dropna().unique()) if "gsis_id" in rosters_weekly.columns else set()
            for id_col in ff_playerids.columns:
                if "gsis" in id_col.lower():
                    ff_gsis_ids = set(ff_playerids[id_col].dropna().unique())
                    overlap = real_gsis_ids & ff_gsis_ids
                    print(
                        f"\nreal crosswalk check: ff_playerids['{id_col}'] overlaps "
                        f"{len(overlap)}/{len(real_gsis_ids)} real {season} rostered gsis_ids "
                        f"({len(overlap) / len(real_gsis_ids):.1%})" if real_gsis_ids else "(no real gsis_ids to compare)"
                    )
        else:
            print(f"\n(no committed {rosters_path} to check real crosswalk overlap against)")


if __name__ == "__main__":
    main()
