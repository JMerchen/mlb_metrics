// node --test docs/player_props.test.js
const test = require("node:test")
const assert = require("node:assert/strict")
const {
  formatPlayerPropsValue,
  PLAYER_PROPS_COLUMNS,
  PLAYER_PROPS_COLUMN_LABELS,
} = require("./player_props.js")

test("formatPlayerPropsValue: rounds real per-game rates to one decimal", () => {
  assert.equal(formatPlayerPropsValue("player_rate", 4.611765), "4.6")
  assert.equal(formatPlayerPropsValue("opponent_allowed_rate", 24.731618), "24.7")
  assert.equal(formatPlayerPropsValue("league_rate", 20.135214), "20.1")
})

test("formatPlayerPropsValue: ratio above 1 shows as a real positive edge", () => {
  assert.equal(formatPlayerPropsValue("ratio", 1.2), "+20%")
})

test("formatPlayerPropsValue: ratio below 1 shows as a real negative edge", () => {
  assert.equal(formatPlayerPropsValue("ratio", 0.8), "-20%")
})

test("formatPlayerPropsValue: ratio exactly 1 (neutral matchup) shows +0%", () => {
  assert.equal(formatPlayerPropsValue("ratio", 1.0), "+0%")
})

test("formatPlayerPropsValue: a non-numeric ratio (e.g. a real missing column) passes through unchanged", () => {
  // A real JS quirk this test exists to guard against: Number("") is 0
  // (not NaN), so an empty string would silently format as "-100%"
  // instead of passing through - undefined (a real missing CSV column)
  // is genuinely non-numeric.
  assert.equal(formatPlayerPropsValue("ratio", undefined), undefined)
})

test("formatPlayerPropsValue: an unformatted column (e.g. player_name) passes through unchanged", () => {
  assert.equal(formatPlayerPropsValue("player_name", "Puka Nacua"), "Puka Nacua")
})

test("PLAYER_PROPS_COLUMNS: every column has a real display label", () => {
  PLAYER_PROPS_COLUMNS.forEach(c => {
    assert.ok(PLAYER_PROPS_COLUMN_LABELS[c], `missing a real label for column "${c}"`)
  })
})
