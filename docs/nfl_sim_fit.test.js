const test = require("node:test")
const assert = require("node:assert")
const { summaries, matchups, priorDraw, realGames } = require("../scripts/fit_nfl_sim_form.js")

test("the six numbers describe how far scores land from the lines", () => {
  const s = summaries([-2, 2, -25, 25], [0, 0, 10, -30])
  assert.ok(Math.abs(s[0] - Math.sqrt((4 + 4 + 625 + 625) / 4)) < 1e-9)   // margin sd
  assert.equal(s[2], 0.5)     // margins within 4
  assert.equal(s[3], 0.5)     // margins beyond 20
  assert.equal(s[4], 0.5)     // totals within 4
  assert.equal(s[5], 0.25)    // totals beyond 20
})

test("candidate settings are drawn inside the search range, the same every run", () => {
  for (let i = 0; i < 50; i++) {
    const d = priorDraw(i)
    assert.ok(d.team >= 0.04 && d.team <= 0.13 && d.game >= 0 && d.game <= 0.08)
  }
  assert.deepEqual(priorDraw(3), priorDraw(3))
  assert.notDeepEqual(priorDraw(3), priorDraw(4))
})

test("the matchups span the real spreads, from big road favorites to big home ones", () => {
  const real = realGames()
  assert.ok(real.length > 2000 && real.every(g => g.season >= 2018))
  const list = matchups(real, 40)
  assert.equal(list.length, 40)
  assert.ok(list[0].spread <= -7 && list[39].spread >= 10)
  assert.ok(list.every(m => m.home !== m.away && m.line > 30 && m.line < 60))
})
