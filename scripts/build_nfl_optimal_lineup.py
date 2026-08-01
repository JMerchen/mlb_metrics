"""Build docs/data/nfl_optimal_lineup.csv (the DK Classic NFL salary-cap,
9-slot optimal lineup - see nfl_dfs_optimizer.py's module docstring) from
docs/data/nfl_qb.csv/nfl_skill.csv/nfl_dst.csv, which nfl_pipeline.py
(Phase 8, not yet built) is responsible for writing each week. Until that
pipeline exists, this script has nothing real to run against - it's
buildable and independently testable now (see
tests/test_build_nfl_optimal_lineup_script.py) via the same "leave prior
output in place on any missing/empty input" resilience pattern
scripts/build_optimal_lineup.py already uses, so it's ready the moment
Phase 8 lands rather than needing to be written then.

IMPORTANT: Estimated_Salary here is a MODELED price, not a real
DraftKings salary - DraftKings has no public salary API. See
nfl_estimated_salary.py's module docstring for the full disclaimer.
Never treat docs/data/nfl_optimal_lineup.csv's Estimated_Salary column as
a real, submittable DraftKings price.

Unlike MLB's build_optimal_lineup.py, position eligibility here needs NO
external fetch (nfl_roster_positions.py is a pure mapping over the
`position` column nfl_skill.csv already carries - see that module's
docstring for why NFL doesn't need MLB's roster_positions.py-style MLB
Stats API lookup).

Also writes docs/data/nfl_dfs_salary_pool.csv (every eligible QB/skill/
DST considered, with dk_slot/DK_Points/Estimated_Salary) - not required
by the page, real transparency into what the optimizer saw, matching
this project's "don't hide the ingredients" culture.

Usage:
    python scripts/build_nfl_optimal_lineup.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from mlb_metrics import config, nfl_dfs_optimizer, nfl_roster_positions


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="docs/data")
    args = parser.parse_args()

    qb_path = os.path.join(args.data_dir, "nfl_qb.csv")
    skill_path = os.path.join(args.data_dir, "nfl_skill.csv")
    dst_path = os.path.join(args.data_dir, "nfl_dst.csv")
    if not (os.path.exists(qb_path) and os.path.exists(skill_path) and os.path.exists(dst_path)):
        print(f"No nfl_qb.csv/nfl_skill.csv/nfl_dst.csv in {args.data_dir} - run the NFL weekly pipeline first.")
        return

    qb_dk = pd.read_csv(qb_path)
    skill_dk = pd.read_csv(skill_path)
    dst_dk = pd.read_csv(dst_path)

    if qb_dk.empty or skill_dk.empty or dst_dk.empty:
        print("No qualified QB/skill/DST rows this week - leaving prior nfl_optimal_lineup.csv in place, if any.")
        return

    eligibility = nfl_roster_positions.build_eligibility_table(skill_dk[["player_id", "position"]])

    pool = nfl_dfs_optimizer.build_player_pool(qb_dk, skill_dk, dst_dk, eligibility)
    if pool.empty:
        print("No players resolved a DK roster slot this week - leaving prior nfl_optimal_lineup.csv in place, if any.")
        return

    lineup = nfl_dfs_optimizer.solve_optimal_lineup(pool, salary_cap=config.NFL_DFS_SALARY_CAP, roster_slots=config.NFL_DFS_ROSTER_SLOTS)
    if lineup is None:
        print("Optimizer could not fill a full lineup under the salary cap this week (too few eligible players at "
              "some position, or the cap is infeasible) - leaving prior nfl_optimal_lineup.csv in place, if any.")
        return

    os.makedirs(args.data_dir, exist_ok=True)
    pool.to_csv(os.path.join(args.data_dir, "nfl_dfs_salary_pool.csv"), index=False)
    lineup.to_csv(os.path.join(args.data_dir, "nfl_optimal_lineup.csv"), index=False)
    print(
        f"Wrote nfl_optimal_lineup.csv ({len(lineup)} players, "
        f"{lineup['Estimated_Salary'].sum():.0f}/{config.NFL_DFS_SALARY_CAP} of cap, "
        f"{lineup['DK_Points'].sum():.2f} total DK_Points "
        f"[ESTIMATED salaries, not real DraftKings prices]) "
        f"and nfl_dfs_salary_pool.csv ({len(pool)} eligible players/DSTs)."
    )


if __name__ == "__main__":
    main()
