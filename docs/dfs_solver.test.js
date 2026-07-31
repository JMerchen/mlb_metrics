// node --test docs/dfs_solver.test.js
const test = require("node:test");
const assert = require("node:assert/strict");
const {
  solveOptimalLineupDP,
  SALARY_CAP,
  ROSTER_SLOTS,
} = require("./dfs_solver.js");

const SMALL_ROSTER = { P: 1, C: 1, OF: 1 };

function player(key_mlbam, dk_slot, DK_Points, Estimated_Salary, extra) {
  return { key_mlbam, dk_slot, DK_Points, Estimated_Salary, ...extra };
}

test("respects exact slot counts and the salary cap", () => {
  const pool = [
    player(1, "P", 10, 5000),
    player(2, "C", 8, 4000),
    player(3, "OF", 6, 3000),
    player(4, "OF", 5, 2000), // extra OF candidate, not selected (only 1 needed)
  ];

  const result = solveOptimalLineupDP(pool, { objectiveColumn: "DK_Points", salaryCap: 50000, rosterSlots: SMALL_ROSTER });

  assert.ok(result);
  assert.equal(result.players.length, 3);
  const bySlot = Object.fromEntries(result.players.map((p) => [p.dk_slot, p.key_mlbam]));
  assert.equal(bySlot.P, 1);
  assert.equal(bySlot.C, 2);
  assert.equal(bySlot.OF, 3); // higher-value OF wins when cap isn't binding
  assert.ok(result.totalSalary <= 50000);
});

test("a binding cap forces the cheaper-but-lower-points pick", () => {
  const pool = [
    player(1, "P", 10, 100),
    player(2, "C", 10, 100),
    player(3, "OF", 100, 50000), // best OF points, but alone busts the cap
    player(4, "OF", 5, 100), // far cheaper, must be chosen instead
  ];

  const result = solveOptimalLineupDP(pool, { objectiveColumn: "DK_Points", salaryCap: 300, rosterSlots: SMALL_ROSTER });

  assert.ok(result);
  const of = result.players.find((p) => p.dk_slot === "OF");
  assert.equal(of.key_mlbam, 4);
});

test("too few candidates at a slot returns null", () => {
  const pool = [player(1, "P", 10, 100), player(2, "C", 10, 100)]; // no OF candidate at all

  const result = solveOptimalLineupDP(pool, { objectiveColumn: "DK_Points", salaryCap: 50000, rosterSlots: SMALL_ROSTER });

  assert.equal(result, null);
});

test("cheapest possible roster over the cap returns null", () => {
  const pool = [player(1, "P", 10, 20000), player(2, "C", 10, 20000), player(3, "OF", 10, 20000)];

  const result = solveOptimalLineupDP(pool, { objectiveColumn: "DK_Points", salaryCap: 50000, rosterSlots: SMALL_ROSTER });

  assert.equal(result, null); // 60000 total, cap is 50000
});

test("empty pool returns null, not a crash", () => {
  const result = solveOptimalLineupDP([], { objectiveColumn: "DK_Points", salaryCap: 50000, rosterSlots: SMALL_ROSTER });
  assert.equal(result, null);
});

test("minSalary forces the pricier pick when the cheap one would otherwise win", () => {
  const pool = [
    player(1, "P", 10, 5000),
    player(2, "C", 10, 5000),
    player(3, "OF", 10, 100), // same points as OF 4, but far cheaper - would win without a floor
    player(4, "OF", 10, 40000),
  ];

  const result = solveOptimalLineupDP(pool, {
    objectiveColumn: "DK_Points",
    salaryCap: 50000,
    minSalary: 45000,
    rosterSlots: SMALL_ROSTER,
  });

  assert.ok(result);
  const of = result.players.find((p) => p.dk_slot === "OF");
  assert.equal(of.key_mlbam, 4);
  assert.ok(result.totalSalary >= 45000);
});

test("minSalary that's infeasible returns null", () => {
  const pool = [player(1, "P", 10, 100), player(2, "C", 10, 100), player(3, "OF", 10, 100)];

  const result = solveOptimalLineupDP(pool, {
    objectiveColumn: "DK_Points",
    salaryCap: 50000,
    minSalary: 49000, // far more than the whole 3-player roster could ever spend
    rosterSlots: SMALL_ROSTER,
  });

  assert.equal(result, null);
});

test("a two-way player (same key_mlbam, two slots) is selected at most once", () => {
  const pool = [
    player(1, "P", 20, 10000), // two-way player as a pitcher
    player(1, "C", 20, 10000), // same key_mlbam as a hitter
    player(2, "C", 5, 100), // weaker alternative catcher
    player(3, "OF", 10, 100),
  ];

  const result = solveOptimalLineupDP(pool, { objectiveColumn: "DK_Points", salaryCap: 50000, rosterSlots: SMALL_ROSTER });

  assert.ok(result);
  const keyMlbams = result.players.map((p) => p.key_mlbam);
  assert.equal(keyMlbams.filter((k) => k === 1).length, 1); // never selected twice
  assert.equal(result.players.length, 3); // still a full roster (P, C, OF)
});

test("proves the DP is exact: greedy-per-slot would pick the wrong combination here", () => {
  // Two OF candidates: A is the single best OF alone (11 pts, $30000), B is
  // slightly worse alone (10 pts, $10000). A greedy "best OF first" pick
  // takes A, leaving only $20000 for P+C - both of which need at least
  // $15000 each here, busting the cap. The DP must recognize that B (with
  // the P/C combo it leaves room for) achieves a higher TOTAL.
  const pool = [
    player(1, "P", 10, 15000),
    player(2, "C", 10, 15000),
    player(3, "OF", 11, 30000), // best alone, but forces an infeasible P+C budget
    player(4, "OF", 10, 10000), // slightly worse alone, but the only way to a feasible full roster
  ];

  const result = solveOptimalLineupDP(pool, { objectiveColumn: "DK_Points", salaryCap: 40000, rosterSlots: SMALL_ROSTER });

  assert.ok(result); // must find a feasible answer at all
  const of = result.players.find((p) => p.dk_slot === "OF");
  assert.equal(of.key_mlbam, 4);
  assert.equal(result.totalObjective, 30); // 10 + 10 + 10
});

test("real DraftKings roster shape (2 P, C, 1B, 2B, 3B, SS, 3 OF) solves under the real cap", () => {
  const pool = [];
  let key = 1;
  for (const [slot, count] of Object.entries(ROSTER_SLOTS)) {
    // a few candidates per slot, cheap enough that a full roster fits comfortably under $50,000
    for (let i = 0; i < count + 2; i++) {
      pool.push(player(key, slot, 5 + i, 2000 + i * 100));
      key += 1;
    }
  }

  const result = solveOptimalLineupDP(pool, { objectiveColumn: "DK_Points", salaryCap: SALARY_CAP });

  assert.ok(result);
  const counts = {};
  for (const p of result.players) counts[p.dk_slot] = (counts[p.dk_slot] || 0) + 1;
  for (const [slot, n] of Object.entries(ROSTER_SLOTS)) {
    assert.equal(counts[slot], n, `expected ${n} at ${slot}, got ${counts[slot]}`);
  }
  assert.ok(result.totalSalary <= SALARY_CAP);
});

test("blank/missing objective or salary cells are excluded, not treated as 0", () => {
  const pool = [
    player(1, "P", "", 5000), // blank DK_Points - PapaParse shape, must be excluded not coerced to 0
    player(2, "P", 10, 5000),
    player(3, "C", 8, 4000),
    player(4, "OF", 6, 3000),
  ];

  const result = solveOptimalLineupDP(pool, { objectiveColumn: "DK_Points", salaryCap: 50000, rosterSlots: SMALL_ROSTER });

  assert.ok(result);
  const p = result.players.find((row) => row.dk_slot === "P");
  assert.equal(p.key_mlbam, 2); // player 1 excluded, not silently valued at 0
});
