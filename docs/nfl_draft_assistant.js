// Pure, DOM-free Draft Assistant logic - mirrors the docs/nfl_dfs_solver.js
// precedent (the only other client-side logic in this project with real
// node --test coverage, docs/nfl_dfs_solver.test.js): meaningful real
// calculation logic deserves real tests, not just a visual spot-check.
// See docs/nfl.js's "My Draft" section for the DOM wiring that calls these.

// Real per-team roster-construction target ranges, mirrored from
// config.NFL_BESTBALL_ROSTER_TARGET - duplicated here (not shared via a
// JSON fetch) the same way docs/nfl_dfs_solver.js already duplicates real
// DK roster-slot constants rather than importing a Python module.
const ROSTER_TARGET = { QB: [2, 3], RB: [5, 7], WR: [6, 8], TE: [1, 2] }

// Real DK Best Ball Mania 12-team pod size (config.NFL_BESTBALL_DRAFT_POD_SIZE).
const DRAFT_POD_SIZE = 12

// currentCountsByPosition: {QB: n, RB: n, WR: n, TE: n} - real counts from
// the roster builder. rosterTarget defaults to the real ROSTER_TARGET
// above (overridable for tests). Returns one real {current, min, max,
// status} object per position - status is "need" (below the real min),
// "full" (at/above the real max), or "optional" (within the real target
// range) - never an invented composite score, just the real gap.
function computeRosterGap(currentCountsByPosition, rosterTarget = ROSTER_TARGET) {
  const result = {}
  for (const position of Object.keys(rosterTarget)) {
    const [min, max] = rosterTarget[position]
    const current = currentCountsByPosition[position] || 0
    let status
    if (current < min) { status = "need" }
    else if (current >= max) { status = "full" }
    else { status = "optional" }
    result[position] = { current, min, max, status }
  }
  return result
}

// Real snake-draft arithmetic: the real overall pick number for a given
// real round and draft slot (1-based), alternating direction each round
// (odd rounds go slot 1..podSize, even rounds reverse).
function pickNumberForRoundAndSlot(round, draftSlot, podSize) {
  return round % 2 === 1
    ? (round - 1) * podSize + draftSlot
    : (round - 1) * podSize + (podSize - draftSlot + 1)
}

// How many real picks happen between currentPickNumber and this
// draftSlot's own next real selection (0 if currentPickNumber IS this
// slot's own pick). Pure integer math, no new data needed beyond the
// real draft slot/pod size/current pick number.
function computeSnakeDraftPicksUntilNextTurn(draftSlot, podSize, currentPickNumber) {
  let round = Math.floor((currentPickNumber - 1) / podSize) + 1
  let candidate = pickNumberForRoundAndSlot(round, draftSlot, podSize)
  if (candidate < currentPickNumber) {
    round += 1
    candidate = pickNumberForRoundAndSlot(round, draftSlot, podSize)
  }
  return candidate - currentPickNumber
}

// Real, honest reach/value read: comparing a real pick number against a
// player's real FantasyPros Best Ball Overall ECR +/- their own real
// expert-rank standard deviation (see src/mlb_metrics/nfl_ff_rankings.py)
// - no invented probability model, just "does this pick number fall
// inside the real expert range." Returns null (not a fabricated read)
// when ecr/ecrSd/pickNumber aren't all real, known numbers.
function computeReachValueRead(ecr, ecrSd, pickNumber) {
  if (ecr === null || ecr === undefined || isNaN(ecr)) { return null }
  if (ecrSd === null || ecrSd === undefined || isNaN(ecrSd)) { return null }
  if (pickNumber === null || pickNumber === undefined || isNaN(pickNumber)) { return null }
  if (pickNumber < ecr - ecrSd) { return "reach" }
  if (pickNumber > ecr + ecrSd) { return "value" }
  return "as expected"
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    ROSTER_TARGET,
    DRAFT_POD_SIZE,
    computeRosterGap,
    pickNumberForRoundAndSlot,
    computeSnakeDraftPicksUntilNextTurn,
    computeReachValueRead,
  }
}
