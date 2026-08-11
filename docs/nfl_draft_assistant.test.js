// node --test docs/nfl_draft_assistant.test.js
const test = require("node:test")
const assert = require("node:assert/strict")
const {
  computeRosterGap,
  computeSnakeDraftPicksUntilNextTurn,
  computeReachValueRead,
  pickNumberForRoundAndSlot,
} = require("./nfl_draft_assistant.js")

test("computeRosterGap: below-min position reads need", () => {
  const gap = computeRosterGap({ QB: 0, RB: 3, WR: 2, TE: 0 })
  assert.equal(gap.RB.status, "need") // 3 < real min 5
  assert.equal(gap.RB.current, 3)
  assert.equal(gap.RB.min, 5)
  assert.equal(gap.RB.max, 7)
})

test("computeRosterGap: within-range position reads optional", () => {
  const gap = computeRosterGap({ QB: 2, RB: 6, WR: 7, TE: 1 })
  assert.equal(gap.QB.status, "optional") // real target 2-3, 2 is within range
  assert.equal(gap.RB.status, "optional")
})

test("computeRosterGap: at-or-above-max position reads full", () => {
  const gap = computeRosterGap({ QB: 3, RB: 7, WR: 8, TE: 2 })
  assert.equal(gap.QB.status, "full")
  assert.equal(gap.TE.status, "full")
})

test("computeRosterGap: missing position defaults to a real 0 count", () => {
  const gap = computeRosterGap({})
  assert.equal(gap.TE.current, 0)
  assert.equal(gap.TE.status, "need")
})

test("computeRosterGap: honors a custom rosterTarget override", () => {
  const gap = computeRosterGap({ QB: 1 }, { QB: [1, 1] })
  assert.equal(gap.QB.status, "full") // 1 >= real max 1
})

test("pickNumberForRoundAndSlot: real snake alternation across rounds", () => {
  // Real 12-team pod, slot 5: round 1 picks slot 5 at overall #5 (odd
  // round, forward order); round 2 reverses (picks 13-24 go slot 12..1),
  // so slot 5's real round-2 pick is #20.
  assert.equal(pickNumberForRoundAndSlot(1, 5, 12), 5)
  assert.equal(pickNumberForRoundAndSlot(2, 5, 12), 20)
  assert.equal(pickNumberForRoundAndSlot(3, 5, 12), 29)
})

test("computeSnakeDraftPicksUntilNextTurn: zero when it's currently your real turn", () => {
  assert.equal(computeSnakeDraftPicksUntilNextTurn(5, 12, 5), 0)
})

test("computeSnakeDraftPicksUntilNextTurn: real count of picks before your next turn, same round", () => {
  // Slot 5 hasn't picked yet in round 1 - from pick #1, 4 real picks
  // happen before slot 5's own real pick #5.
  assert.equal(computeSnakeDraftPicksUntilNextTurn(5, 12, 1), 4)
})

test("computeSnakeDraftPicksUntilNextTurn: real count crossing into the next (reversed) round", () => {
  // Just after slot 5's round-1 pick (#5); round 2 reverses, so slot 5's
  // next real pick is #20 - 14 real picks happen in between.
  assert.equal(computeSnakeDraftPicksUntilNextTurn(5, 12, 6), 14)
})

test("computeReachValueRead: real reach when the pick number falls well ahead of the expert range", () => {
  assert.equal(computeReachValueRead(20, 3, 10), "reach") // pick 10 < ecr(20) - sd(3) = 17
})

test("computeReachValueRead: real value when the pick number falls well past the expert range", () => {
  assert.equal(computeReachValueRead(20, 3, 30), "value") // pick 30 > ecr(20) + sd(3) = 23
})

test("computeReachValueRead: as expected within the real expert range", () => {
  assert.equal(computeReachValueRead(20, 3, 21), "as expected")
})

test("computeReachValueRead: null (not a fabricated read) when ECR data is missing", () => {
  assert.equal(computeReachValueRead(undefined, undefined, 10), null)
  assert.equal(computeReachValueRead(20, 3, undefined), null)
  assert.equal(computeReachValueRead(NaN, 3, 10), null)
})
