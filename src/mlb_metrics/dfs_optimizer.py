"""DraftKings Classic MLB salary-cap, position-slot lineup optimizer - an
exact MILP (mixed-integer linear program) solved via PuLP/CBC
(pulp.PULP_CBC_CMD, bundled with the `pulp` package, no separate solver
install needed).

## Why an exact solver, not a greedy pick-per-slot heuristic

A typical slate's hitter pool is large enough (several hundred qualified
batters) that the OF slot alone (3 of maybe 80-100 OF-eligible candidates)
is already a `C(90, 3)` ~ 117,000-combination choice, before considering
every other slot sharing the same salary-cap budget. A greedy "best points
per dollar" pick per slot does NOT guarantee the true optimum: a locally
good OF pick can foreclose a globally better combination once the shared
budget constraint is considered - that's exactly the property that makes
this a real knapsack/assignment problem, not a sorting problem.

## Why PuLP/ILP over a hand-rolled DP

Since build_player_pool gives each player exactly one dk_slot, the slot
groups ARE disjoint - the problem technically decomposes into a bounded
knapsack per group merged via a final 1-D knapsack, solvable without a new
dependency. PuLP was chosen anyway because: (a) it directly expresses the
real constraints (`sum(x[p] for p in OF_pool) == 3`,
`sum(salary * x) <= cap`) rather than needing careful group-merge
bookkeeping to hand-verify correct at "real dollar stakes"; (b) it survives
the near-certain v1.1 follow-up already flagged in roster_positions.py's
docstring (real DK multi-position eligibility breaks the disjoint-groups
assumption a DP would depend on - ILP trivially generalizes by giving a
multi-eligible player rows in more than one group, unioned by the same
"used at most once" constraint this module already needs for two-way
players, see below); (c) the actual problem size here (a few hundred
binary variables, ~10 constraints) solves via CBC in well under a second,
so there's no real performance case for avoiding it.

## Two-way players

A player who appears in BOTH the hitter pool and the pitcher pool (e.g. a
true two-way player) gets one pool row per role, each with its own dk_slot
- without an explicit guard, the optimizer could select both rows (using
one real person to fill two different roster slots at once), which isn't
a legal DK lineup. solve_optimal_lineup adds a `<= 1` constraint across any
key_mlbam appearing more than once in the pool to prevent this.

## Objective: mean points (default), ceiling, boom-adjusted, or value

solve_optimal_lineup's `objective_column` selects what gets maximized:
"DK_Points" (the default, preserving all prior behavior - the existing
mean projection), "Ceiling_DK_Points" (dfs_ceiling.py's real-history
90th-percentile upside signal, for a boom-or-bust tournament/GPP lineup),
"Boom_Adjusted_DK_Points" (dfs_ceiling.py's mean + k*upside-deviation
blend - rewards a player's real, FREQUENT volatility without chasing a
single outlier game the way pure ceiling does), or "Value_Score"
(boom-adjusted points PER $1,000 of Estimated_Salary).

**Why Value_Score exists**: real evidence from an actual DK contest
showed that maximizing ANY of the point-like objectives above - mean,
ceiling, or boom-adjusted - independently converges on the same two most
expensive pitchers, because Estimated_Salary's fixed floor means a high
scorer of ANY position gets a structurally better AVERAGE dollars-per-
point rate (the floor is a smaller fraction of a bigger price), so a
budget-constrained points-maximizer will always rationally overpay for
the biggest scorers first. Value_Score inverts that: it explicitly
rewards being UNDERPRICED relative to real upside (a "star") over being
already fully priced-in (a "superstar") - see config.DFS_VALUE_BOOM_K_PITCHER's
docstring for the full reasoning and an explicit caveat about what's
validated here vs. not.

**Value_Score's own flaw, found and fixed in the same pass**: maximizing
a per-dollar ratio carries no pressure to spend near the cap - a real
sanity check left $10,300 of $50,000 unspent. solve_optimal_lineup's
`min_salary` parameter fixes this (see config.DFS_VALUE_MIN_SALARY_FRACTION's
docstring for the real numbers behind the chosen floor); build_optimal_lineup.py
passes it only when objective="value".

None of the three alternatives is the default - see dfs_ceiling.py's
module docstring for why each ships opt-in. A player with no real scored
history has no per-player ceiling/deviation to fall back to salary-cap
math on, so build_player_pool defaults every alternative column to its
own DK_Points (the mean projection) rather than leaving it NaN, which
would otherwise make that player silently unselectable under a non-mean
objective."""

import pandas as pd
import pulp

from mlb_metrics import config, dfs_ceiling, estimated_salary

POOL_COLUMNS = [
    "key_mlbam", "name_first", "name_last", "team", "opponent", "is_home", "dk_slot",
    "game_pk", "game_datetime",
    "DK_Points", "Ceiling_DK_Points", "Boom_Adjusted_DK_Points", "Value_Score", "Estimated_Salary",
]


def build_player_pool(hitters: pd.DataFrame, pitchers: pd.DataFrame, eligibility: pd.DataFrame) -> pd.DataFrame:
    """One row per player eligible for today's optimizer, in POOL_COLUMNS
    shape. `hitters`/`pitchers` are dfs.compute_hitter_dk_points/
    compute_pitcher_dk_points's own output (post dfs_ml.apply_ml_overrides -
    i.e. this project's best current point projection). `eligibility` is
    roster_positions.fetch_position_eligibility's output - a hitter with no
    resolvable DK slot (unmatched, or primary position has no DK Classic
    slot - most commonly a DH) is excluded via the inner join, not
    defaulted. Pitchers need no eligibility lookup - already restricted to
    today's probable starters (dfs.py's own scope), mapped to dk_slot="P"
    directly."""
    hitters = hitters.merge(eligibility[["key_mlbam", "dk_slot"]], on="key_mlbam", how="inner").copy()
    hitters["DK_Points"] = hitters["DK_Points_Hitter"]
    for column in ("Ceiling_DK_Points", "Boom_Adjusted_DK_Points"):
        if column not in hitters.columns:
            hitters[column] = pd.NA
        hitters[column] = hitters[column].fillna(hitters["DK_Points_Hitter"])
    hitters["Estimated_Salary"] = estimated_salary.compute_hitter_estimated_salary(hitters["DK_Points_Hitter"])
    # Reuses hitters' own Boom_Adjusted_DK_Points (already k=1.0, the
    # validated hitter value) as Value_Score's numerator - no separate
    # hitter constant needed, unlike pitchers below.
    hitters["Value_Score"] = hitters["Boom_Adjusted_DK_Points"] / (hitters["Estimated_Salary"] / 1000)

    pitchers = pitchers.copy()
    pitchers["dk_slot"] = "P"
    pitchers["DK_Points"] = pitchers["DK_Points_Pitcher"]
    for column in ("Ceiling_DK_Points", "Boom_Adjusted_DK_Points"):
        if column not in pitchers.columns:
            pitchers[column] = pd.NA
        pitchers[column] = pitchers[column].fillna(pitchers["DK_Points_Pitcher"])
    pitchers["Estimated_Salary"] = estimated_salary.compute_pitcher_estimated_salary(pitchers["DK_Points_Pitcher"])
    # Value_Score's pitcher numerator is DELIBERATELY NOT the same as
    # pitchers' own Boom_Adjusted_DK_Points column (which uses k=0.0 - see
    # config.DFS_BOOM_ADJUSTED_K_PITCHER's docstring, validated against a
    # different question that k didn't help). Recomputed here with
    # config.DFS_VALUE_BOOM_K_PITCHER instead - see that constant's
    # docstring for why this is a deliberate strategic choice, not a
    # re-validated statistical claim.
    if "Upside_Deviation" not in pitchers.columns:
        pitchers["Upside_Deviation"] = pd.NA
    pitcher_upside_deviation = pitchers["Upside_Deviation"].fillna(0)
    pitcher_value_points = dfs_ceiling.compute_boom_adjusted_score(
        pitchers["DK_Points_Pitcher"], pitcher_upside_deviation, config.DFS_VALUE_BOOM_K_PITCHER
    )
    pitchers["Value_Score"] = pitcher_value_points / (pitchers["Estimated_Salary"] / 1000)

    # is_home/game_pk/game_datetime feed the dashboard's slate-filtering
    # optimizer UI (which game is each pool row in?) and the Optimal
    # Lineup table's home/away rendering - a hitters/pitchers frame built
    # by hand (e.g. in tests, or an older dfs_hitters.csv/dfs_pitchers.csv
    # snapshot) that lacks them degrades to NA rather than a KeyError on
    # the POOL_COLUMNS selection below.
    for frame in (hitters, pitchers):
        for column in ("is_home", "game_pk", "game_datetime"):
            if column not in frame.columns:
                frame[column] = pd.NA

    return pd.concat([hitters[POOL_COLUMNS], pitchers[POOL_COLUMNS]], ignore_index=True)


def solve_optimal_lineup(
    pool: pd.DataFrame,
    salary_cap: float = config.DFS_SALARY_CAP,
    roster_slots: dict = config.DFS_ROSTER_SLOTS,
    objective_column: str = "DK_Points",
    min_salary: float | None = None,
) -> pd.DataFrame | None:
    """Exact MILP: maximize total `objective_column` (default "DK_Points",
    the existing mean projection - pass "Ceiling_DK_Points" for a
    tournament/GPP-style upside lineup instead, see module docstring)
    subject to sum(Estimated_Salary) <= salary_cap and, for every slot in
    `roster_slots`, exactly that many players with a matching dk_slot are
    selected. Returns None (not an exception) if infeasible - too few
    eligible players at some slot, or the cheapest possible full roster
    already exceeds the cap - the same documented-fallback resilience
    pattern used elsewhere in this project (e.g.
    dfs.compute_pitcher_dk_points's unannounced-starter exclusion,
    pipeline.run()'s failed-schedule-fetch handling).

    `min_salary` (default None, i.e. no floor) adds
    sum(Estimated_Salary) >= min_salary. Needed specifically for
    objective_column="Value_Score": maximizing a per-dollar ratio carries
    no pressure to spend near the cap on its own (confirmed via a real
    sanity check - see config.DFS_VALUE_MIN_SALARY_FRACTION's docstring),
    so scripts/build_optimal_lineup.py passes one only for that
    objective."""
    if pool.empty:
        return None

    pool = pool.reset_index(drop=True)
    problem = pulp.LpProblem("optimal_lineup", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in pool.index}

    problem += pulp.lpSum(x[i] * pool.loc[i, objective_column] for i in pool.index)
    problem += pulp.lpSum(x[i] * pool.loc[i, "Estimated_Salary"] for i in pool.index) <= salary_cap
    if min_salary is not None:
        problem += pulp.lpSum(x[i] * pool.loc[i, "Estimated_Salary"] for i in pool.index) >= min_salary

    for slot, count in roster_slots.items():
        slot_indices = pool.index[pool["dk_slot"] == slot]
        problem += pulp.lpSum(x[i] for i in slot_indices) == count

    for key_mlbam, group in pool.groupby("key_mlbam"):
        if len(group) > 1:
            problem += pulp.lpSum(x[i] for i in group.index) <= 1

    problem.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[problem.status] != "Optimal":
        return None

    selected = [i for i in pool.index if x[i].value() == 1]
    return pool.loc[selected].sort_values("dk_slot").reset_index(drop=True)
