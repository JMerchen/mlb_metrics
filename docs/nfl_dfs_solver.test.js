// node --test docs/nfl_dfs_solver.test.js
const test = require("node:test");
const assert = require("node:assert/strict");
const { solveOptimalLineupDP, SALARY_CAP, ROSTER_SLOTS, relabelFlexForDisplay, FLEX_HYPOTHESES } = require("./nfl_dfs_solver.js");

function player(key_mlbam, dk_slot, DK_Points, Estimated_Salary, extra) {
  return { key_mlbam, dk_slot, DK_Points, Estimated_Salary, ...extra };
}

// Two rows per RB/WR/TE (own slot + FLEX), one row for QB/DST - the real
// shape nfl_dfs_optimizer.build_player_pool emits.
function skillPlayer(key_mlbam, ownSlot, DK_Points, Estimated_Salary) {
  return [player(key_mlbam, ownSlot, DK_Points, Estimated_Salary), player(key_mlbam, "FLEX", DK_Points, Estimated_Salary)];
}

test("respects exact slot counts (including FLEX) and the salary cap on a minimal full roster", () => {
  const pool = [
    player("qb1", "QB", 20, 5000),
    player("dst1", "DST", 8, 3000),
    ...skillPlayer("rb1", "RB", 10, 4000),
    ...skillPlayer("rb2", "RB", 9, 4000),
    ...skillPlayer("rb3", "RB", 6, 3000), // the 7th skill player - fills FLEX via the RB:3 hypothesis
    ...skillPlayer("wr1", "WR", 12, 5000),
    ...skillPlayer("wr2", "WR", 11, 5000),
    ...skillPlayer("wr3", "WR", 10, 4500),
    ...skillPlayer("te1", "TE", 7, 3500),
  ];
  // Exactly the minimum roster shape PLUS one real overflow candidate
  // (rb3) - a full 9-player roster needs 7 skill players total
  // (RB:2+WR:3+TE:1+FLEX:1), so every one of these 7 must be selected,
  // one of them relabeled to FLEX for display.

  const result = solveOptimalLineupDP(pool, { objectiveColumn: "DK_Points", salaryCap: SALARY_CAP });

  assert.ok(result);
  assert.equal(result.players.length, 9);
  const counts = {};
  for (const p of result.players) counts[p.dk_slot] = (counts[p.dk_slot] || 0) + 1;
  assert.deepEqual(counts, ROSTER_SLOTS);
  assert.ok(result.totalSalary <= SALARY_CAP);
});

test("FLEX picks the best value across positions, not just whichever position has overflow", () => {
  // Mirrors the Python go/no-go test exactly: rb_c/wr_d/te_b are each
  // clearly the best "extra" candidate at their position; the true
  // optimum must include all three somewhere (own slot or FLEX).
  const pool = [
    player("qb1", "QB", 20, 100),
    player("dst1", "DST", 8, 100),
    ...skillPlayer("rb_a", "RB", 1, 100),
    ...skillPlayer("rb_b", "RB", 1, 100),
    ...skillPlayer("rb_c", "RB", 15, 100),
    ...skillPlayer("wr_a", "WR", 1, 100),
    ...skillPlayer("wr_b", "WR", 1, 100),
    ...skillPlayer("wr_c", "WR", 1, 100),
    ...skillPlayer("wr_d", "WR", 12, 100),
    ...skillPlayer("te_a", "TE", 1, 100),
    ...skillPlayer("te_b", "TE", 8, 100),
  ];

  const result = solveOptimalLineupDP(pool, { objectiveColumn: "DK_Points", salaryCap: SALARY_CAP });

  assert.ok(result);
  assert.equal(result.players.length, 9);
  const keyMlbams = new Set(result.players.map((p) => p.key_mlbam));
  for (const key of ["qb1", "dst1", "rb_c", "wr_d", "te_b"]) {
    assert.ok(keyMlbams.has(key), `expected ${key} to be selected`);
  }
  assert.equal(result.totalObjective, 67); // 20 + 8 + 15 + 12 + 8 + four 1-point fillers
});

test("a player is never selected twice across their own-slot and FLEX rows", () => {
  const pool = [
    player("qb1", "QB", 20, 100),
    player("dst1", "DST", 8, 100),
    ...skillPlayer("rb1", "RB", 20, 100), // by far the best RB - could tempt double-selection
    ...skillPlayer("rb2", "RB", 1, 100),
    ...skillPlayer("rb3", "RB", 1, 100), // the 7th skill player, fills FLEX
    ...skillPlayer("wr1", "WR", 1, 100),
    ...skillPlayer("wr2", "WR", 1, 100),
    ...skillPlayer("wr3", "WR", 1, 100),
    ...skillPlayer("te1", "TE", 1, 100),
  ];

  const result = solveOptimalLineupDP(pool, { objectiveColumn: "DK_Points", salaryCap: SALARY_CAP });

  assert.ok(result);
  const rb1Count = result.players.filter((p) => p.key_mlbam === "rb1").length;
  assert.equal(rb1Count, 1);
});

test("relabelFlexForDisplay picks the lowest-salary overflow player, purely cosmetic", () => {
  const hypothesis = FLEX_HYPOTHESES[0]; // RB:3 overflow
  const result = {
    players: [
      player("rb1", "RB", 10, 5000),
      player("rb2", "RB", 9, 3000), // cheapest RB - should become FLEX
      player("rb3", "RB", 8, 4000),
    ],
    totalObjective: 27,
    totalSalary: 12000,
  };

  const relabeled = relabelFlexForDisplay(result, hypothesis);

  const flexPlayer = relabeled.players.find((p) => p.dk_slot === "FLEX");
  assert.equal(flexPlayer.key_mlbam, "rb2");
  assert.equal(relabeled.totalObjective, 27); // unchanged - purely cosmetic
  assert.equal(relabeled.totalSalary, 12000);
});

test("too few candidates at a required slot returns null", () => {
  const pool = [
    player("qb1", "QB", 20, 100),
    player("dst1", "DST", 8, 100),
    ...skillPlayer("rb1", "RB", 10, 100), // only 1 RB - RB slot alone needs 2
    ...skillPlayer("wr1", "WR", 1, 100),
    ...skillPlayer("wr2", "WR", 1, 100),
    ...skillPlayer("wr3", "WR", 1, 100),
    ...skillPlayer("te1", "TE", 1, 100),
  ];

  const result = solveOptimalLineupDP(pool, { objectiveColumn: "DK_Points", salaryCap: SALARY_CAP });

  assert.equal(result, null);
});

test("cheapest possible roster over the cap returns null", () => {
  const pool = [
    player("qb1", "QB", 20, 20000),
    player("dst1", "DST", 8, 20000),
    ...skillPlayer("rb1", "RB", 10, 20000),
    ...skillPlayer("rb2", "RB", 9, 20000),
    ...skillPlayer("wr1", "WR", 12, 20000),
    ...skillPlayer("wr2", "WR", 11, 20000),
    ...skillPlayer("wr3", "WR", 10, 20000),
    ...skillPlayer("te1", "TE", 7, 20000),
  ];

  const result = solveOptimalLineupDP(pool, { objectiveColumn: "DK_Points", salaryCap: SALARY_CAP });

  assert.equal(result, null); // way more than $50,000 total
});

test("empty pool returns null, not a crash", () => {
  const result = solveOptimalLineupDP([], { objectiveColumn: "DK_Points", salaryCap: SALARY_CAP });
  assert.equal(result, null);
});

test("minSalary forces the pricier pick when the cheap one would otherwise win", () => {
  // Only the RB:3 hypothesis is feasible here (WR has exactly 3
  // candidates, TE exactly 1 - no 4th WR or 2nd TE exists), isolating
  // the test to a single real choice: which of two EQUAL-points RB
  // overflow candidates fills the 3rd RB spot, cheap or pricey.
  const pool = [
    player("qb1", "QB", 20, 5000),
    player("dst1", "DST", 8, 3000),
    ...skillPlayer("rb1", "RB", 10, 4000),
    ...skillPlayer("rb2", "RB", 10, 4000),
    ...skillPlayer("rb_cheap", "RB", 10, 100), // same points as rb_pricey, far cheaper
    ...skillPlayer("rb_pricey", "RB", 10, 15000),
    ...skillPlayer("wr1", "WR", 12, 5000),
    ...skillPlayer("wr2", "WR", 11, 5000),
    ...skillPlayer("wr3", "WR", 10, 4500),
    ...skillPlayer("te1", "TE", 7, 3500),
  ];

  const unconstrained = solveOptimalLineupDP(pool, { objectiveColumn: "DK_Points", salaryCap: SALARY_CAP });
  assert.ok(unconstrained);
  const cheapKeys = unconstrained.players.map((p) => p.key_mlbam);
  assert.ok(cheapKeys.includes("rb_cheap"));
  assert.ok(!cheapKeys.includes("rb_pricey"));
  assert.ok(unconstrained.totalSalary < 45000);

  const withFloor = solveOptimalLineupDP(pool, { objectiveColumn: "DK_Points", salaryCap: SALARY_CAP, minSalary: 45000 });
  assert.ok(withFloor);
  const floorKeys = withFloor.players.map((p) => p.key_mlbam);
  // The floor forces at least the pricier option INTO the lineup (the
  // cheap-only combination can't reach $45,000) - it doesn't necessarily
  // exclude rb_cheap too, since rb1+rb_cheap+rb_pricey ($45,100) is a
  // real, legitimately CHEAPER way to clear the floor than rb1+rb2+
  // rb_pricey ($49,000) at the same tied objective value - the solver
  // correctly prefers the smallest budget that still clears the floor.
  assert.ok(floorKeys.includes("rb_pricey"));
  assert.ok(withFloor.totalSalary >= 45000);
  assert.ok(withFloor.totalSalary > unconstrained.totalSalary);
});

test("blank/missing objective or salary cells are excluded, not treated as 0", () => {
  const pool = [
    player("qb1", "QB", "", 5000), // blank DK_Points - PapaParse shape, must be excluded not coerced to 0
    player("qb2", "QB", 20, 5000),
    player("dst1", "DST", 8, 100),
    ...skillPlayer("rb1", "RB", 10, 100),
    ...skillPlayer("rb2", "RB", 9, 100),
    ...skillPlayer("rb3", "RB", 6, 100), // the 7th skill player, fills FLEX
    ...skillPlayer("wr1", "WR", 12, 100),
    ...skillPlayer("wr2", "WR", 11, 100),
    ...skillPlayer("wr3", "WR", 10, 100),
    ...skillPlayer("te1", "TE", 7, 100),
  ];

  const result = solveOptimalLineupDP(pool, { objectiveColumn: "DK_Points", salaryCap: SALARY_CAP });

  assert.ok(result);
  const qb = result.players.find((p) => p.dk_slot === "QB");
  assert.equal(qb.key_mlbam, "qb2");
});
