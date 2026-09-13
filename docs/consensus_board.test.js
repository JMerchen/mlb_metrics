// node --test docs/consensus_board.test.js
const test = require("node:test")
const assert = require("node:assert/strict")
const {
  prepareConsensusColumns,
  formatConsensusValue,
  formatColumnLabel,
} = require("./consensus_board.js")

test("prepareConsensusColumns: core columns first, in a fixed real order", () => {
  const data = [{ sources_ranked_by: 2, player_name: "Alice", consensus_rank: 1, consensus_score: 1.5 }]

  const columns = prepareConsensusColumns(data)

  assert.deepEqual(columns, ["consensus_rank", "player_name", "consensus_score", "sources_ranked_by"])
})

test("prepareConsensusColumns: extra real columns (e.g. Pos/College) appended after the core 4", () => {
  const data = [{
    consensus_rank: 1, player_name: "Alice", consensus_score: 1.5, sources_ranked_by: 2,
    Pos: "SS", College: "UCLA",
  }]

  const columns = prepareConsensusColumns(data)

  assert.deepEqual(columns, ["consensus_rank", "player_name", "consensus_score", "sources_ranked_by", "Pos", "College"])
})

test("prepareConsensusColumns: hides the raw per-source dict/url columns", () => {
  const data = [{
    consensus_rank: 1, player_name: "Alice", consensus_score: 1.5, sources_ranked_by: 2,
    source_ranks: "{'A': 1.0}", source_url: "https://example.com",
  }]

  const columns = prepareConsensusColumns(data)

  assert.deepEqual(columns, ["consensus_rank", "player_name", "consensus_score", "sources_ranked_by"])
})

test("prepareConsensusColumns: real, honest handling of a missing core column", () => {
  // A caller might pass a partial real shape (e.g. before sources_ranked_by
  // existed) - never fabricate a column that isn't really there.
  const data = [{ consensus_rank: 1, player_name: "Alice", consensus_score: 1.5 }]

  const columns = prepareConsensusColumns(data)

  assert.deepEqual(columns, ["consensus_rank", "player_name", "consensus_score"])
})

test("prepareConsensusColumns: empty data returns no columns", () => {
  assert.deepEqual(prepareConsensusColumns([]), [])
})

test("formatConsensusValue: rounds consensus_score to 1 real decimal", () => {
  assert.equal(formatConsensusValue("consensus_score", 12.3333333), "12.3")
})

test("formatConsensusValue: passes through a non-numeric consensus_score unchanged", () => {
  assert.equal(formatConsensusValue("consensus_score", "n/a"), "n/a")
})

test("formatConsensusValue: passes through every other real column unchanged", () => {
  assert.equal(formatConsensusValue("player_name", "Alice"), "Alice")
  assert.equal(formatConsensusValue("consensus_rank", 1), 1)
})

test("formatColumnLabel: real friendly labels for the 4 synthetic columns", () => {
  assert.equal(formatColumnLabel("consensus_rank"), "Rank")
  assert.equal(formatColumnLabel("player_name"), "Player")
  assert.equal(formatColumnLabel("consensus_score"), "Consensus Score")
  assert.equal(formatColumnLabel("sources_ranked_by"), "Sources")
})

test("formatColumnLabel: passes through a real source-provided column unchanged", () => {
  assert.equal(formatColumnLabel("Pos"), "Pos")
})
