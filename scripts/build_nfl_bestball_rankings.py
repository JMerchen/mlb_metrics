"""Build docs/data/nfl_bestball_rankings.csv: a preseason bestball
draft-strategy ranking of real QB/RB/WR/TE production from last season
(config.NFL_SEASON - 1), with real games-played/missed as an honest,
cheap injury-history proxy - see nfl_bestball.py's module docstring for
why this is a genuinely different question from the weekly DFS pipeline
(a realized season total, not a forward-looking projection).

A manual/one-time build, not part of the daily/weekly cron - real 2025
season stats don't change, and (per this feature's own scope) this isn't
meant to be a live-updated feed. Re-run manually
(`.github/workflows/build_nfl_bestball_rankings.yml`, workflow_dispatch
only) later in preseason if wanted, e.g. once more roster/depth-chart
news has settled.

Needs data/raw/nfl/weekly_<season>.parquet and
data/raw/nfl/schedules_<season>.parquet (see scripts/fetch_nfl_historical.py)
for both the target season and one prior season (for the
games_missed_prior_season repeat-injury-risk column) - all already
persisted for 2016-2025 as of this script's writing.

Usage:
    python scripts/build_nfl_bestball_rankings.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import config, nfl_bestball, nfl_data


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default=config.NFL_RAW_DATA_DIR)
    parser.add_argument("--data-dir", default="docs/data")
    parser.add_argument("--season", type=int, default=config.NFL_SEASON - 1)
    args = parser.parse_args()

    prior_season = args.season - 1

    weekly = nfl_data.load_persisted_table(args.raw_dir, "weekly", args.season)
    schedules = nfl_data.load_persisted_table(args.raw_dir, "schedules", args.season)
    if weekly is None or schedules is None:
        print(f"No persisted weekly/schedules data for {args.season} in {args.raw_dir} - "
              f"run scripts/fetch_nfl_historical.py first.")
        return

    prior_weekly = nfl_data.load_persisted_table(args.raw_dir, "weekly", prior_season)
    prior_schedules = nfl_data.load_persisted_table(args.raw_dir, "schedules", prior_season)
    if prior_weekly is None or prior_schedules is None:
        print(f"No persisted weekly/schedules data for prior season {prior_season} - "
              f"proceeding without games_missed_prior_season.")

    rankings = nfl_bestball.build_bestball_rankings(
        weekly, schedules, args.season,
        prior_season=prior_season if prior_weekly is not None and prior_schedules is not None else None,
        prior_weekly_df=prior_weekly, prior_schedules_df=prior_schedules,
    )

    os.makedirs(args.data_dir, exist_ok=True)
    out_path = os.path.join(args.data_dir, "nfl_bestball_rankings.csv")
    rankings.to_csv(out_path, index=False)
    print(f"Wrote nfl_bestball_rankings.csv ({len(rankings)} rows) for {args.season}.")


if __name__ == "__main__":
    main()
