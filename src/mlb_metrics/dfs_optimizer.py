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

## Objective: mean points (default) or ceiling

solve_optimal_lineup's `objective_column` selects what gets maximized:
"DK_Points" (the default, preserving all prior behavior - the existing
mean projection) or "Ceiling_DK_Points" (dfs_ceiling.py's real-history
upside signal, for a tournament/GPP-style lineup built from spike-game
players instead of reliable-average ones). Ceiling is NOT the default -
see dfs_ceiling.py's module docstring for why it ships opt-in until a
real backtest validates it actually helps. A player with no real scored
history has no per-player ceiling to fall back to salary-cap math on, so
build_player_pool defaults their Ceiling_DK_Points to their own DK_Points
(the mean projection) rather than leaving it NaN, which would otherwise
make them silently unselectable under a ceiling objective."""

import pandas as pd
import pulp

from mlb_metrics import config, estimated_salary

POOL_COLUMNS = [
    "key_mlbam", "name_first", "name_last", "team", "opponent", "dk_slot",
    "DK_Points", "Ceiling_DK_Points", "Estimated_Salary",
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
    if "Ceiling_DK_Points" not in hitters.columns:
        hitters["Ceiling_DK_Points"] = pd.NA
    hitters["Ceiling_DK_Points"] = hitters["Ceiling_DK_Points"].fillna(hitters["DK_Points_Hitter"])
    hitters["Estimated_Salary"] = estimated_salary.compute_hitter_estimated_salary(hitters["DK_Points_Hitter"])

    pitchers = pitchers.copy()
    pitchers["dk_slot"] = "P"
    pitchers["DK_Points"] = pitchers["DK_Points_Pitcher"]
    if "Ceiling_DK_Points" not in pitchers.columns:
        pitchers["Ceiling_DK_Points"] = pd.NA
    pitchers["Ceiling_DK_Points"] = pitchers["Ceiling_DK_Points"].fillna(pitchers["DK_Points_Pitcher"])
    pitchers["Estimated_Salary"] = estimated_salary.compute_pitcher_estimated_salary(pitchers["DK_Points_Pitcher"])

    return pd.concat([hitters[POOL_COLUMNS], pitchers[POOL_COLUMNS]], ignore_index=True)


def solve_optimal_lineup(
    pool: pd.DataFrame,
    salary_cap: float = config.DFS_SALARY_CAP,
    roster_slots: dict = config.DFS_ROSTER_SLOTS,
    objective_column: str = "DK_Points",
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
    pipeline.run()'s failed-schedule-fetch handling)."""
    if pool.empty:
        return None

    pool = pool.reset_index(drop=True)
    problem = pulp.LpProblem("optimal_lineup", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in pool.index}

    problem += pulp.lpSum(x[i] * pool.loc[i, objective_column] for i in pool.index)
    problem += pulp.lpSum(x[i] * pool.loc[i, "Estimated_Salary"] for i in pool.index) <= salary_cap

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
