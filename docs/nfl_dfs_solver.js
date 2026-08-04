// Client-side re-implementation of nfl_dfs_optimizer.solve_optimal_lineup
// (dfs_optimizer.solve_optimal_lineup, reused unmodified there - see that
// Python module's docstring). Lets docs/nfl.html re-solve the optimal DK
// Classic NFL lineup entirely in the browser from data already published
// in docs/data/nfl_dfs_salary_pool.csv - no new backend, no new network
// call. Same underlying algorithm as docs/dfs_solver.js (a disjoint-slot
// bounded-knapsack-per-slot, merged into one shared-budget knapsack -
// see that file's own docstring for the full "why a DP, not a JS MILP
// library" reasoning) - solveSlotSubset/mergeSlots/solveOnce below are
// duplicated from it verbatim rather than shared via a require(), since
// this project's docs/ pages are each fully self-contained (own JS,
// sharing only style.css - see docs/age-curves.html's established
// precedent, reaffirmed for this NFL page specifically).
//
// ## The one real difference from MLB: FLEX
//
// dfs_optimizer's DP depends on DK slots being DISJOINT groups - true for
// MLB (every hitter has exactly one dk_slot). NFL's FLEX slot breaks that:
// nfl_roster_positions.build_eligibility_table gives every RB/WR/TE TWO
// pool rows (their own slot AND "FLEX"), so naively running this same DP
// with a literal "FLEX" slot group would double-count a player's value
// across two knapsacks that both allow selecting them.
//
// The exact fix (no approximation): since DK scores a FLEX RB identically
// to an RB-slot RB, "2 RB + 3 WR + 1 TE + 1 FLEX" is equivalent to
// enumerating the 3 possible ways that 1 extra skill-position slot could
// be spent - {RB:3, WR:3, TE:1}, {RB:2, WR:4, TE:1}, {RB:2, WR:3, TE:2} -
// and solving each as an ordinary NO-FLEX disjoint-groups problem (using
// only each player's OWN-slot row, never their "FLEX" row) via the exact
// same solveOnce this file already has. Whichever of the 3 hypotheses
// yields the highest total objective IS the true global optimum - there's
// no 4th possibility a real DK lineup could take that isn't covered by
// "which single position absorbs the extra slot." O(3) solves, no new DP
// dimension, no approximation flag needed (unlike dfs_solver.js's own
// >3-duplicate-players fallback, which genuinely does approximate).
//
// Once a hypothesis wins, WHICH specific selected player at the
// overflowing position is displayed under dk_slot="FLEX" (vs. their real
// position) is purely cosmetic - DK scores them identically either way.
// relabelFlexForDisplay below picks the lowest-Estimated_Salary player at
// that position as an arbitrary, deterministic tie-break, purely for a
// stable/readable UI - not a modeling choice.

const SALARY_CAP = 50000; // config.NFL_DFS_SALARY_CAP
const SALARY_STEP = 100; // config.NFL_DFS_ESTIMATED_SALARY_ROUND_TO
const ROSTER_SLOTS = { QB: 1, RB: 2, WR: 3, TE: 1, FLEX: 1, DST: 1 }; // config.NFL_DFS_ROSTER_SLOTS

// The 3 FLEX hypotheses - each a complete, disjoint-groups slot table
// with NO "FLEX" key at all (QB/DST counts never change). See module
// docstring for why exactly these 3 exhaust the real possibilities.
const FLEX_HYPOTHESES = [
  { QB: 1, RB: 3, WR: 3, TE: 1, DST: 1 },
  { QB: 1, RB: 2, WR: 4, TE: 1, DST: 1 },
  { QB: 1, RB: 2, WR: 3, TE: 2, DST: 1 },
];

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

// Bounded multiple-choice knapsack for ONE slot - identical algorithm to
// docs/dfs_solver.js's solveSlotSubset (see that file for the full
// per-cell reconstruction reasoning); duplicated here rather than shared
// per this file's own module docstring.
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

// Knapsack-of-knapsacks over the ONE shared budget dimension - identical
// to docs/dfs_solver.js's mergeSlots.
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

// One full disjoint-groups solve - identical shape to docs/dfs_solver.js's
// solveOnce. `rosterSlots` here is always one of FLEX_HYPOTHESES (never
// has a "FLEX" key), so `rows` must already be pre-filtered to exclude
// every "FLEX"-labeled row before calling this (solveOptimalLineupDP does
// that filtering).
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

// Relabels the lowest-Estimated_Salary selected player at the winning
// hypothesis's overflowing position to dk_slot="FLEX", purely for display
// - see module docstring's "purely cosmetic" note. Mutates neither the
// input pool nor result.players in place (returns new player objects).
function relabelFlexForDisplay(result, hypothesis) {
  if (!result) return result;
  const overflowSlot = Object.keys(hypothesis).find(
    (slot) => slot !== "QB" && slot !== "DST" && hypothesis[slot] > ROSTER_SLOTS[slot]
  );
  if (!overflowSlot) return result;

  const candidateIndices = result.players
    .map((p, i) => i)
    .filter((i) => result.players[i].dk_slot === overflowSlot);
  candidateIndices.sort(
    (a, b) => toNumber(result.players[a].Estimated_Salary) - toNumber(result.players[b].Estimated_Salary)
  );
  const flexIdx = candidateIndices[0];

  return {
    ...result,
    players: result.players.map((p, i) => (i === flexIdx ? { ...p, dk_slot: "FLEX" } : p)),
  };
}

// Exact solve, mirroring nfl_dfs_optimizer.build_player_pool's output
// shape: `pool` is an array of nfl_dfs_salary_pool.csv-shaped row objects
// (key_mlbam, dk_slot, Estimated_Salary, and whichever objective column
// is selected, at minimum - every RB/WR/TE appears TWICE, once per
// eligible slot). Returns {players, totalObjective, totalSalary} with
// `players` in the real 9-slot shape (QB/RB/RB/WR/WR/WR/TE/FLEX/DST), or
// `null` if infeasible under EVERY hypothesis.
function solveOptimalLineupDP(pool, options) {
  const objectiveColumn = options.objectiveColumn;
  const salaryCap = options.salaryCap != null ? options.salaryCap : SALARY_CAP;
  const minSalary = options.minSalary != null ? options.minSalary : null;

  const rows = [];
  pool.forEach((row, idx) => {
    const value = toNumber(row[objectiveColumn]);
    const salary = toNumber(row.Estimated_Salary);
    if (!Number.isFinite(value) || !Number.isFinite(salary)) return;
    rows.push({ idx, key_mlbam: row.key_mlbam, dk_slot: row.dk_slot, value, salary });
  });

  let best = null;
  let bestHypothesis = null;
  for (const hypothesis of FLEX_HYPOTHESES) {
    const nonFlexRows = rows.filter((r) => r.dk_slot in hypothesis);
    const result = solveOnce(nonFlexRows, pool, objectiveColumn, salaryCap, minSalary, hypothesis);
    if (result && (!best || result.totalObjective > best.totalObjective)) {
      best = result;
      bestHypothesis = hypothesis;
    }
  }

  return relabelFlexForDisplay(best, bestHypothesis);
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    solveOptimalLineupDP,
    SALARY_CAP,
    SALARY_STEP,
    ROSTER_SLOTS,
    FLEX_HYPOTHESES,
    // exported for direct unit testing of the DP internals
    solveSlotSubset,
    mergeSlots,
    solveOnce,
    relabelFlexForDisplay,
    toNumber,
  };
}
