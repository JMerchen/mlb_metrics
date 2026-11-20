const test = require("node:test")
const assert = require("node:assert")
const { rangeFor, chanceOver, fairPrice, rgMarginSpan, gameRangeLines, propRangeHtml, propLineText } = require("./ranges.js")

// Misses spread evenly from -20 to +20 (in scale units of 1).
const even = (low, high) => Array.from({ length: 201 }, (_, i) => low + (high - low) * i / 200)
const FLAT = { n: 999, scale: null, edges: [], groups: [{ n: 999, scores: even(-20, 20) }] }
// Two groups by size, misses scaled to the projection.
const SIZED = {
  n: 2000, scale: { intercept: 2, slope: 0.4, floor: 1 }, edges: [40], floor: 0,
  groups: [{ n: 1000, scores: even(-1, 3) }, { n: 1000, scores: even(-2, 2) }],
}
const text = html => html.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim()

test("a range is read off the past misses at the stated share", () => {
  const [low, high] = rangeFor(FLAT, 45, 0.8)
  assert.ok(Math.abs(low - 29) < 0.2 && Math.abs(high - 61) < 0.2)
  const half = rangeFor(FLAT, 45, 0.5)
  assert.ok(Math.abs(half[0] - 35) < 0.2 && Math.abs(half[1] - 55) < 0.2)
  assert.equal(rangeFor(null, 45, 0.8), null)
  assert.equal(rangeFor(FLAT, "", 0.8), null)
})

test("grouped and scaled: a bigger projection gets its own misses and a wider range, never below the floor", () => {
  const small = rangeFor(SIZED, 10, 0.8, 0), large = rangeFor(SIZED, 80, 0.8, 0)
  // 10 yards: scale 6, misses -1 to 3 -> about 10 - 0.6*6 to 10 + 2.6*6, clipped at 0... the low end stays above it here.
  assert.ok(Math.abs(small[0] - 6.4) < 0.3 && Math.abs(small[1] - 25.6) < 0.3)
  assert.ok(large[1] - large[0] > 3 * (small[1] - small[0]))
  assert.equal(rangeFor(SIZED, 2, 0.8, 0)[0] >= 0, true)
})

test("the chance over a line comes from the same misses and is never certain", () => {
  assert.ok(Math.abs(chanceOver(FLAT, 45, 45) - 0.5) < 0.01)
  assert.ok(Math.abs(chanceOver(FLAT, 45, 55) - 0.25) < 0.01)
  assert.ok(chanceOver(FLAT, 45, 500) > 0 && chanceOver(FLAT, 45, 500) < 0.001)
  assert.ok(chanceOver(FLAT, 45, -500) < 1)
  assert.equal(chanceOver(FLAT, 45, "x"), null)
})

test("a chance becomes a fair price", () => {
  assert.equal(fairPrice(0.6), "−150")
  assert.equal(fairPrice(0.4), "+150")
  assert.equal(fairPrice(0.5), "−100")
  assert.equal(fairPrice(1), "")
})

test("a game card gets its total 4 times in 5 and its margin both ways, in team names", () => {
  assert.equal(rgMarginSpan(-1.4, 16.8, "Texans", "Colts"), "Colts by 1 to Texans by 17")
  assert.equal(rgMarginSpan(0.2, 9, "Texans", "Colts"), "even to Texans by 9")
  const lines = gameRangeLines({ total: FLAT, margin: FLAT }, { predicted_total: "45", predicted_margin: "8" }, "Texans", "Colts")
  assert.deepEqual(lines, [
    "4 times in 5 the total lands in 29–61",
    "Margin: half the time Colts by 2 to Texans by 18; 4 times in 5, Colts by 8 to Texans by 24",
  ])
  assert.deepEqual(gameRangeLines(null, { predicted_total: 45 }, "A", "B"), [])
  assert.deepEqual(gameRangeLines({ total: FLAT }, { predicted_total: "", predicted_margin: "" }, "A", "B"), [])
})

test("a prop card gets its range and a line checker starting at the player's average", () => {
  const row = { category: "Receiving Yards", projection: "22.3", player_rate: "38.01" }
  const html = propRangeHtml({ props: { "Receiving Yards": SIZED } }, row, "yards")
  assert.match(text(html), /^Likely yards: \d+–\d+ half the time, \d+–\d+ 4 times in 5\. Against a line of over \d+% \(fair at [+−]\d+\) · under \d+% \(fair at [+−]\d+\)$/)
  assert.match(html, /value="38.5"/)
  assert.match(html, /data-prop-category="Receiving Yards" data-prop-projection="22.3"/)
  // Sacks have no projection, and a category without a calibration no range.
  assert.equal(propRangeHtml({ props: {} }, row, "yards"), "")
  assert.equal(propRangeHtml({ props: { "Receiving Yards": SIZED } }, { ...row, projection: "" }, "yards"), "")
  assert.equal(propLineText(SIZED, 22.3, ""), "")
  assert.match(text(propLineText(SIZED, 22.3, 16.5)), /^over 8\d%/)
})
