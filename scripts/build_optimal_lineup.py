"""Build docs/data/optimal_lineup.csv (the DK Classic MLB salary-cap,
position-slot optimal lineup - see dfs_optimizer.py's module docstring)
from the SAME docs/data/dfs_hitters.csv/dfs_pitchers.csv
scripts/build_dfs_rankings.py already wrote earlier in this run, plus a
fresh position-eligibility fetch (roster_positions.py).

IMPORTANT: Estimated_Salary here is a MODELED price, not a real
DraftKings salary - DraftKings has no public salary API. See
estimated_salary.py's module docstring for the full disclaimer and why.
Never treat docs/data/optimal_lineup.csv's Estimated_Salary column as a
real, submittable DraftKings price.

Run this DAILY, immediately after scripts/build_dfs_rankings.py, as part
of daily_update.yml - same cadence reasoning (today's projections/
matchups change every day).

Resilience, matching build_dfs_rankings.py's own pattern: missing
dfs_hitters.csv/dfs_pitchers.csv, a failed position-eligibility fetch, or
an infeasible optimizer solve (too few eligible players at some slot, or
even the cheapest full roster exceeds the cap) all leave yesterday's
optimal_lineup.csv in place rather than writing anything.

Also writes docs/data/dfs_salary_pool.csv (every eligible player considered,
with their dk_slot/DK_Points/Ceiling_DK_Points/Boom_Adjusted_DK_Points/
Value_Score/Estimated_Salary) - not required by the page, but real
transparency into what the optimizer actually saw, matching this
project's "don't hide the ingredients" culture.

`--objective {mean,ceiling,boom,value}` selects what the optimizer
maximizes: "mean" (the default, preserving all prior behavior) uses
DK_Points_Hitter/Pitcher; "ceiling" uses dfs_ceiling.py's Ceiling_DK_Points
instead, for a tournament/GPP-style lineup built from a single real
historical spike game; "boom" uses Boom_Adjusted_DK_Points (mean + k times
a player's real historical UPSIDE-ONLY volatility) - a middle ground the
user explicitly asked for, rewarding real, frequent boom potential without
chasing one outlier game the way pure ceiling does; "value" uses
Value_Score (boom-adjusted points PER $1,000 of Estimated_Salary) - real
evidence showed "mean"/"ceiling"/"boom" all independently converge on the
same two most expensive pitchers (Estimated_Salary's fixed floor makes
ANY high scorer look structurally cheaper per point on average, so a raw
points-maximizer always overpays for the biggest scorers regardless of
position), leaving no budget to build a real mix of consistent-floor and
genuine-boom hitters. "value" explicitly rewards being underpriced
relative to real upside (a "star") instead of chasing raw point totals
(a "superstar" that's already fully priced in) - see
config.DFS_VALUE_BOOM_K_PITCHER's docstring for the full reasoning. None
of "ceiling"/"boom"/"value" is the default - see dfs_ceiling.py's and
dfs_optimizer.py's module docstrings for the backtest evidence (or, for
"value," the explicit reasoning where backtest evidence doesn't directly
apply) behind each.

"value" also passes solve_optimal_lineup a min_salary floor
(config.DFS_VALUE_MIN_SALARY_FRACTION * DFS_SALARY_CAP) - maximizing a
per-dollar ratio alone carries no pressure to spend near the cap, and a
real sanity check confirmed it: "value" with no floor left $10,300 of
$50,000 unspent. See config.DFS_VALUE_MIN_SALARY_FRACTION's docstring
for the real numbers behind the chosen floor. The other three objectives
pass no min_salary (unchanged behavior).

Usage:
    python scripts/build_optimal_lineup.py
    python scripts/build_optimal_lineup.py --objective ceiling
    python scripts/build_optimal_lineup.py --objective boom
    python scripts/build_optimal_lineup.py --objective value
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from mlb_metrics import config, dfs_optimizer, roster_positions

OBJECTIVE_COLUMNS = {
    "mean": "DK_Points", "ceiling": "Ceiling_DK_Points",
    "boom": "Boom_Adjusted_DK_Points", "value": "Value_Score",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="docs/data")
    parser.add_argument("--objective", choices=sorted(OBJECTIVE_COLUMNS), default="mean")
    args = parser.parse_args()

    hitters_path = os.path.join(args.data_dir, "dfs_hitters.csv")
    pitchers_path = os.path.join(args.data_dir, "dfs_pitchers.csv")
    if not (os.path.exists(hitters_path) and os.path.exists(pitchers_path)):
        print(f"No dfs_hitters.csv/dfs_pitchers.csv in {args.data_dir} - run scripts/build_dfs_rankings.py first.")
        return

    hitters = pd.read_csv(hitters_path)
    pitchers = pd.read_csv(pitchers_path)
    # dfs_hitters.csv now carries its own dk_slot (build_dfs_rankings.py,
    # for the dashboard's per-position subtabs) - drop it before
    # build_player_pool does its own independent eligibility fetch/merge
    # below, which would otherwise collide and get silently suffixed to
    # dk_slot_x/dk_slot_y.
    hitters = hitters.drop(columns=["dk_slot"], errors="ignore")

    if hitters.empty:
        print("No qualified hitters today - leaving yesterday's optimal_lineup.csv in place, if any.")
        return

    try:
        eligibility = roster_positions.fetch_position_eligibility(hitters["key_mlbam"].tolist())
    except Exception as exc:
        print(f"WARNING: failed to fetch today's position eligibility ({exc}); "
              f"leaving yesterday's optimal_lineup.csv in place, if any.")
        return

    pool = dfs_optimizer.build_player_pool(hitters, pitchers, eligibility)
    if pool.empty:
        print("No players resolved a DK roster slot today - leaving yesterday's optimal_lineup.csv in place, if any.")
        return

    min_salary = config.DFS_VALUE_MIN_SALARY_FRACTION * config.DFS_SALARY_CAP if args.objective == "value" else None
    lineup = dfs_optimizer.solve_optimal_lineup(
        pool, objective_column=OBJECTIVE_COLUMNS[args.objective], min_salary=min_salary
    )
    if lineup is None:
        print("Optimizer could not fill a full lineup under the salary cap today (too few eligible players at "
              "some position, or the cap is infeasible) - leaving yesterday's optimal_lineup.csv in place, if any.")
        return

    os.makedirs(args.data_dir, exist_ok=True)
    pool.to_csv(os.path.join(args.data_dir, "dfs_salary_pool.csv"), index=False)
    lineup.to_csv(os.path.join(args.data_dir, "optimal_lineup.csv"), index=False)
    print(
        f"Wrote optimal_lineup.csv (objective={args.objective}, {len(lineup)} players, "
        f"{lineup['Estimated_Salary'].sum():.0f}/{config.DFS_SALARY_CAP} of cap, "
        f"{lineup['DK_Points'].sum():.2f} total DK_Points / {lineup['Ceiling_DK_Points'].sum():.2f} total "
        f"Ceiling_DK_Points [ESTIMATED salaries, not real DraftKings prices]) "
        f"and dfs_salary_pool.csv ({len(pool)} eligible players)."
    )


if __name__ == "__main__":
    main()
