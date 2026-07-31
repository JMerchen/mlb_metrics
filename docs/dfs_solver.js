// Client-side re-implementation of dfs_optimizer.solve_optimal_lineup (see
// that module's Python docstring for the full "why a DP, not a JS MILP
// library" reasoning). Lets the dashboard re-solve the DraftKings Classic
// optimal lineup for a USER-CHOSEN subset of today's games ("slate
// filtering") entirely in the browser, using data already published in
// docs/data/dfs_salary_pool.csv - no new backend, no new network call.
//
// Why this is an EXACT algorithm, not a heuristic: dfs_optimizer.build_player_pool
// gives every pool row exactly one dk_slot, so the DraftKings roster slots
// (config.DFS_ROSTER_SLOTS) are disjoint groups. The problem is therefore a
// multiple-choice knapsack (pick exactly n players from each disjoint
// group to maximize total objective value) merged over ONE shared salary
// budget - solvable by dynamic programming, not just approximately. Every
// Estimated_Salary is already a multiple of $100 (config.DFS_ESTIMATED_SALARY_ROUND_TO),
// so the budget dimension has only 501 discrete states at the $50,000 cap
// - small enough to make this genuinely fast (single-digit milliseconds on
// a real slate) and exact, not a performance compromise.
//
// The one thing this DP can't express natively is dfs_optimizer.py's
// cross-group "<=1 per key_mlbam" constraint (a true two-way player
// appearing in both the hitter and pitcher pools). Handled below by
// solving multiple times over the small number of "which role stays
// available" combinations when duplicates exist - see
// solveOptimalLineupDP's docstring.
//
// KNOWN v1 LIMITATION, matching roster_positions.py's own documented gap:
// this DP depends on the disjoint-slot-groups property. If this project
// ever implements real DraftKings multi-position eligibility (a player
// legally fillable at more than one slot - flagged as a v1.1 follow-up in
// roster_positions.py's docstring), a player would appear in more than one
// slot group and this DP would need to be revisited; the Python MILP
// (dfs_optimizer.solve_optimal_lineup) already generalizes to that case
// for free, since ILP constraints don't assume disjoint groups.

const SALARY_CAP = 50000; // config.DFS_SALARY_CAP
const SALARY_STEP = 100; // config.DFS_ESTIMATED_SALARY_ROUND_TO
const ROSTER_SLOTS = { P: 2, C: 1, "1B": 1, "2B": 1, "3B": 1, SS: 1, OF: 3 }; // config.DFS_ROSTER_SLOTS
const VALUE_MIN_SALARY_FRACTION = 0.85; // config.DFS_VALUE_MIN_SALARY_FRACTION
const OBJECTIVE_COLUMNS = {
  mean: "DK_Points",
  ceiling: "Ceiling_DK_Points",
  boom: "Boom_Adjusted_DK_Points",
  value: "Value_Score",
}; // scripts/build_optimal_lineup.py's OBJECTIVE_COLUMNS

function toNumber(value) {
  // PapaParse returns every field as a string, and JS's Number("") is 0,
  // not NaN - a real footgun here since an empty/missing cell must be
  // treated as unusable, not a free $0 salary or 0-point player.
  if (value === "" || value === null || value === undefined) return NaN;
  return Number(value);
}

function makeGrid(rows, cols, fillValue) {
  const grid = new Array(rows + 1);
  for (let i = 0; i <= rows; i++) grid[i] = new Array(cols + 1).fill(fillValue);
  return grid;
}

function cloneGrid(grid) {
  return grid.map((row) => row.slice());
}

// Bounded multiple-choice knapsack for ONE slot: choose exactly `n` of
// `items` (each {cost, value}, cost/value already numeric) to maximize
// total value, for every possible total cost 0..B. Returns
// {finalLayer, reconstruct} - finalLayer[j][b] is the best value using
// exactly j items at cost exactly b (-Infinity if unreachable);
// reconstruct(j, b) returns the LOCAL indices into `items` achieving that
// cell, or [] if unreachable.
//
// Implementation keeps one full grid "layer" per item processed (not an
// in-place update) specifically so reconstruction is a simple, provably
// correct comparison (layer[i] differs from layer[i-1] at (j,b) if and
// only if item i-1 was used to reach that cell's value) rather than a
// harder-to-verify shared-state walk - see this file's own tests for the
// exact case (a hand-computed pool where greedy would pick wrong) this
// guards against.
function solveSlotSubset(items, n, B) {
  const m = items.length;
  const layers = new Array(m + 1);
  layers[0] = makeGrid(n, B, -Infinity);
  layers[0][0][0] = 0;

  for (let i = 1; i <= m; i++) {
    const prev = layers[i - 1];
    const { cost, value } = items[i - 1];
    const cur = cloneGrid(prev);
    const maxJ = Math.min(i, n);
    for (let j = 1; j <= maxJ; j++) {
      for (let b = cost; b <= B; b++) {
        const candidate = prev[j - 1][b - cost];
        if (candidate === -Infinity) continue;
        if (candidate + value > cur[j][b]) {
          cur[j][b] = candidate + value;
        }
      }
    }
    layers[i] = cur;
  }

  function reconstruct(j, b) {
    const chosen = [];
    let curJ = j;
    let curB = b;
    for (let i = m; i >= 1 && curJ > 0; i--) {
      if (layers[i][curJ][curB] !== layers[i - 1][curJ][curB]) {
        chosen.push(i - 1);
        curJ -= 1;
        curB -= items[i - 1].cost;
      }
    }
    return chosen;
  }

  return { finalLayer: layers[m], reconstruct };
}

// Merges every slot's per-budget best-value row (finalLayer[n_slot]) over
// the ONE shared budget dimension - a knapsack-of-knapsacks. Returns
// {g, splits}: g[b] = best total value using total cost exactly b across
// every slot; splits[k][b] = how much of that budget slot k itself spent,
// needed to walk the reconstruction back apart by slot.
function mergeSlots(slotResults, slotOrder, rosterSlots, B) {
  let g = new Array(B + 1).fill(-Infinity);
  g[0] = 0;
  const splits = [];

  for (let k = 0; k < slotOrder.length; k++) {
    const slot = slotOrder[k];
    const n = rosterSlots[slot];
    const slotBest = slotResults[slot].finalLayer[n];
    const newG = new Array(B + 1).fill(-Infinity);
    const split = new Array(B + 1).fill(-1);
    for (let used = 0; used <= B; used++) {
      if (g[used] === -Infinity) continue;
      for (let c = 0; used + c <= B; c++) {
        const v = slotBest[c];
        if (v === -Infinity) continue;
        const total = g[used] + v;
        if (total > newG[used + c]) {
          newG[used + c] = total;
          split[used + c] = c;
        }
      }
    }
    g = newG;
    splits.push(split);
  }

  return { g, splits };
}

// One full solve over a fixed, already-deduplicated-by-key row set. Not
// exported directly - solveOptimalLineupDP is the public entry point,
// since it also has to handle the cross-group two-way-player constraint
// before this function ever runs.
function solveOnce(rows, pool, objectiveColumn, salaryCap, minSalary, rosterSlots) {
  const B = Math.floor(salaryCap / SALARY_STEP);
  const minBudgetUnits = minSalary != null ? Math.ceil(minSalary / SALARY_STEP) : 0;

  const bySlot = {};
  for (const slot of Object.keys(rosterSlots)) bySlot[slot] = [];
  for (const r of rows) {
    if (!(r.dk_slot in bySlot)) continue;
    const cost = Math.ceil(r.salary / SALARY_STEP);
    bySlot[r.dk_slot].push({ idx: r.idx, cost, value: r.value });
  }

  const slotOrder = Object.keys(rosterSlots);
  const slotResults = {};
  for (const slot of slotOrder) {
    const n = rosterSlots[slot];
    if (bySlot[slot].length < n) return null; // not enough eligible players at this slot
    slotResults[slot] = solveSlotSubset(bySlot[slot], n, B);
  }

  const { g, splits } = mergeSlots(slotResults, slotOrder, rosterSlots, B);

  let bStar = -1;
  let bestValue = -Infinity;
  for (let b = minBudgetUnits; b <= B; b++) {
    if (g[b] > bestValue) {
      bestValue = g[b];
      bStar = b;
    }
  }
  if (bStar === -1 || bestValue === -Infinity) return null;

  let remaining = bStar;
  const selectedIdx = [];
  for (let k = slotOrder.length - 1; k >= 0; k--) {
    const slot = slotOrder[k];
    const n = rosterSlots[slot];
    const c = splits[k][remaining];
    const localChosen = slotResults[slot].reconstruct(n, c);
    for (const li of localChosen) selectedIdx.push(bySlot[slot][li].idx);
    remaining -= c;
  }

  const players = selectedIdx.map((idx) => pool[idx]);
  const totalSalary = selectedIdx.reduce((sum, idx) => sum + toNumber(pool[idx].Estimated_Salary), 0);
  return { players, totalObjective: bestValue, totalSalary };
}

function cartesianProduct(arraysOfChoices) {
  return arraysOfChoices.reduce(
    (acc, choices) => acc.flatMap((combo) => choices.map((choice) => [...combo, choice])),
    [[]]
  );
}

// Exact solve, mirroring dfs_optimizer.solve_optimal_lineup: `pool` is an
// array of dfs_salary_pool.csv-shaped row objects (POOL_COLUMNS - dk_slot,
// Estimated_Salary, and whichever objective column is selected, at
// minimum). Returns {players, totalObjective, totalSalary} or `null` if
// infeasible (too few eligible players at some slot, or even the cheapest
// full roster exceeds the cap) - same "None on infeasible, never throws"
// contract as the Python version.
//
// Two-way players (the same key_mlbam appearing under more than one
// dk_slot - a true two-way player with both a hitter-pool and a
// pitcher-pool row): dfs_optimizer.py enforces "at most one role total"
// via a cross-group MILP constraint this DP can't express directly.
// Instead, solve once per combination of "which single role stays
// available" for each duplicated player (at most 2^3 = 8 solves for up to
// 3 duplicated players - each solve is exact, so the best of the
// combinations IS the true optimum) and keep the best. On real data this
// essentially never triggers (see dfs_optimizer.py's own module
// docstring: the pitcher pool is today's probable starters only, and
// hitters/pitchers are built from disjoint queries). More than 3
// duplicated players (should not happen in practice) falls back to
// keeping only each duplicated player's higher-objective row and flags
// the result `approximate: true` rather than an exponential blow-up -
// stated here as a known v1 simplification, not hidden.
function solveOptimalLineupDP(pool, options) {
  const objectiveColumn = options.objectiveColumn;
  const salaryCap = options.salaryCap != null ? options.salaryCap : SALARY_CAP;
  const minSalary = options.minSalary != null ? options.minSalary : null;
  const rosterSlots = options.rosterSlots || ROSTER_SLOTS;

  const rows = [];
  pool.forEach((row, idx) => {
    const value = toNumber(row[objectiveColumn]);
    const salary = toNumber(row.Estimated_Salary);
    if (!Number.isFinite(value) || !Number.isFinite(salary)) return;
    rows.push({ idx, key_mlbam: row.key_mlbam, dk_slot: row.dk_slot, value, salary });
  });

  const slotsByKey = new Map();
  for (const r of rows) {
    if (!slotsByKey.has(r.key_mlbam)) slotsByKey.set(r.key_mlbam, new Set());
    slotsByKey.get(r.key_mlbam).add(r.dk_slot);
  }
  const duplicated = [...slotsByKey.entries()].filter(([, slots]) => slots.size > 1).map(([key]) => key);

  if (duplicated.length === 0) {
    return solveOnce(rows, pool, objectiveColumn, salaryCap, minSalary, rosterSlots);
  }

  if (duplicated.length > 3) {
    const bestForKey = new Map();
    for (const r of rows) {
      if (!duplicated.includes(r.key_mlbam)) continue;
      const cur = bestForKey.get(r.key_mlbam);
      if (!cur || r.value > cur.value) bestForKey.set(r.key_mlbam, r);
    }
    const filteredRows = rows.filter((r) => {
      if (!duplicated.includes(r.key_mlbam)) return true;
      return bestForKey.get(r.key_mlbam) === r;
    });
    const result = solveOnce(filteredRows, pool, objectiveColumn, salaryCap, minSalary, rosterSlots);
    if (result) result.approximate = true;
    return result;
  }

  const choicesPerKey = duplicated.map((key) => [...slotsByKey.get(key)]);
  let best = null;
  for (const combo of cartesianProduct(choicesPerKey)) {
    const filteredRows = rows.filter((r) => {
      const dupIdx = duplicated.indexOf(r.key_mlbam);
      if (dupIdx === -1) return true;
      return r.dk_slot === combo[dupIdx];
    });
    const result = solveOnce(filteredRows, pool, objectiveColumn, salaryCap, minSalary, rosterSlots);
    if (result && (!best || result.totalObjective > best.totalObjective)) best = result;
  }
  return best;
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    solveOptimalLineupDP,
    SALARY_CAP,
    SALARY_STEP,
    ROSTER_SLOTS,
    VALUE_MIN_SALARY_FRACTION,
    OBJECTIVE_COLUMNS,
    // exported for direct unit testing of the DP internals
    solveSlotSubset,
    mergeSlots,
    toNumber,
  };
}
