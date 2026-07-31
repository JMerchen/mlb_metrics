"""Cross-language parity check for docs/dfs_solver.js's exact knapsack DP
against dfs_optimizer.solve_optimal_lineup's exact PuLP/CBC MILP - the
highest-value test for this feature, since a Playwright screenshot can
confirm the UI wires up correctly but can never catch a knapsack DP that's
subtly wrong in a way that still LOOKS like a plausible lineup. Both
solvers are exact algorithms over the same problem, so on any real pool
they must agree on the achievable objective total (not necessarily the
exact same PLAYER set - ties are legitimately ambiguous between two
different exact solvers) and on feasibility.

Skipped entirely if `node` isn't on PATH, so a Node-less environment still
passes the rest of the suite - see .github/workflows/ci.yml for where this
actually runs.
"""

import json
import random
import shutil
import subprocess

import pytest

from mlb_metrics import config, dfs_optimizer

DFS_SOLVER_JS = "docs/dfs_solver.js"

NODE_AVAILABLE = shutil.which("node") is not None
pytestmark = pytest.mark.skipif(not NODE_AVAILABLE, reason="node not available in this environment")


def _run_js_solver(pool_rows, objective_column, salary_cap, min_salary=None, roster_slots=None):
    payload = {
        "pool": pool_rows,
        "options": {
            "objectiveColumn": objective_column,
            "salaryCap": salary_cap,
            "minSalary": min_salary,
        },
    }
    if roster_slots is not None:
        payload["options"]["rosterSlots"] = roster_slots

    script = f"""
const {{ solveOptimalLineupDP }} = require('./{DFS_SOLVER_JS}');
let input = '';
process.stdin.on('data', d => input += d);
process.stdin.on('end', () => {{
  const payload = JSON.parse(input);
  const result = solveOptimalLineupDP(payload.pool, payload.options);
  console.log(JSON.stringify(result ? {{totalObjective: result.totalObjective, totalSalary: result.totalSalary}} : null));
}});
"""
    proc = subprocess.run(
        ["node", "-e", script], input=json.dumps(payload), capture_output=True, text=True, check=True,
    )
    return json.loads(proc.stdout.strip())


def _random_pool(rng, roster_slots, salary_min=2000, salary_max=11000, extra_per_slot=(1, 4)):
    """One pool row per candidate, `count + extra` candidates per slot -
    same real-data shape dfs_salary_pool.csv has (more eligible players
    than roster spots at every position)."""
    rows = []
    key = 1
    for slot, count in roster_slots.items():
        n_candidates = count + rng.randint(*extra_per_slot)
        for _ in range(n_candidates):
            salary = rng.randint(salary_min // 100, salary_max // 100) * 100
            value = round(rng.uniform(1.0, 25.0), 4)
            rows.append({
                "key_mlbam": key, "name_first": "P", "name_last": str(key), "team": "BOS", "opponent": "NYY",
                "dk_slot": slot, "DK_Points": value, "Estimated_Salary": salary,
            })
            key += 1
    return rows


def _pool_df(rows):
    import pandas as pd
    return pd.DataFrame(rows)


@pytest.mark.parametrize("seed", range(20))
def test_dp_matches_pulp_on_random_pools(seed):
    rng = random.Random(seed)
    rows = _random_pool(rng, config.DFS_ROSTER_SLOTS)
    pool = _pool_df(rows)

    python_result = dfs_optimizer.solve_optimal_lineup(
        pool, salary_cap=config.DFS_SALARY_CAP, roster_slots=config.DFS_ROSTER_SLOTS, objective_column="DK_Points",
    )
    js_result = _run_js_solver(rows, "DK_Points", config.DFS_SALARY_CAP)

    if python_result is None:
        assert js_result is None, f"seed={seed}: Python infeasible but JS found a lineup"
    else:
        assert js_result is not None, f"seed={seed}: JS infeasible but Python found a lineup"
        python_total = python_result["DK_Points"].sum()
        assert js_result["totalObjective"] == pytest.approx(python_total, abs=1e-6), f"seed={seed}"
        assert js_result["totalSalary"] <= config.DFS_SALARY_CAP


@pytest.mark.parametrize("seed", range(10))
def test_dp_matches_pulp_with_min_salary_floor(seed):
    rng = random.Random(seed + 1000)
    rows = _random_pool(rng, config.DFS_ROSTER_SLOTS)
    pool = _pool_df(rows)
    min_salary = config.DFS_VALUE_MIN_SALARY_FRACTION * config.DFS_SALARY_CAP

    python_result = dfs_optimizer.solve_optimal_lineup(
        pool, salary_cap=config.DFS_SALARY_CAP, roster_slots=config.DFS_ROSTER_SLOTS,
        objective_column="DK_Points", min_salary=min_salary,
    )
    js_result = _run_js_solver(rows, "DK_Points", config.DFS_SALARY_CAP, min_salary=min_salary)

    if python_result is None:
        assert js_result is None, f"seed={seed}"
    else:
        assert js_result is not None, f"seed={seed}"
        python_total = python_result["DK_Points"].sum()
        assert js_result["totalObjective"] == pytest.approx(python_total, abs=1e-6), f"seed={seed}"
        assert js_result["totalSalary"] >= min_salary - 1e-6


def test_dp_matches_pulp_with_a_two_way_player():
    rng = random.Random(42)
    rows = _random_pool(rng, config.DFS_ROSTER_SLOTS)
    # Make one hitter and one pitcher share a key_mlbam - a real two-way
    # player, legal at most once total under both solvers' constraints.
    hitter_row = next(r for r in rows if r["dk_slot"] != "P")
    pitcher_row = next(r for r in rows if r["dk_slot"] == "P")
    pitcher_row["key_mlbam"] = hitter_row["key_mlbam"]
    # Make both roles attractively cheap and high-value so the solver
    # would actually WANT to double-dip if the constraint were broken.
    hitter_row["DK_Points"] = 50.0
    hitter_row["Estimated_Salary"] = 2000
    pitcher_row["DK_Points"] = 50.0
    pitcher_row["Estimated_Salary"] = 2000

    pool = _pool_df(rows)
    python_result = dfs_optimizer.solve_optimal_lineup(
        pool, salary_cap=config.DFS_SALARY_CAP, roster_slots=config.DFS_ROSTER_SLOTS, objective_column="DK_Points",
    )
    js_result = _run_js_solver(rows, "DK_Points", config.DFS_SALARY_CAP)

    assert python_result is not None and js_result is not None
    python_key_counts = python_result["key_mlbam"].value_counts()
    assert python_key_counts[hitter_row["key_mlbam"]] == 1
    python_total = python_result["DK_Points"].sum()
    assert js_result["totalObjective"] == pytest.approx(python_total, abs=1e-6)


def test_js_solver_constants_match_config():
    """Constant-drift guard: docs/dfs_solver.js hardcodes its own copies of
    config.py's DFS constants (no build step to import them directly) -
    this is the tax for that, and this project already cares about
    exactly this class of drift (see schedule.TEAM_ID_TO_ABBREV's own
    docs/app.js-parity test)."""
    with open(DFS_SOLVER_JS) as f:
        source = f.read()

    assert f"const SALARY_CAP = {config.DFS_SALARY_CAP};" in source
    assert f"const SALARY_STEP = {config.DFS_ESTIMATED_SALARY_ROUND_TO};" in source
    assert f"const VALUE_MIN_SALARY_FRACTION = {config.DFS_VALUE_MIN_SALARY_FRACTION};" in source
    for slot, count in config.DFS_ROSTER_SLOTS.items():
        assert f'{slot}: {count}' in source or f'"{slot}": {count}' in source, (
            f"docs/dfs_solver.js's ROSTER_SLOTS is missing or wrong for {slot}"
        )
