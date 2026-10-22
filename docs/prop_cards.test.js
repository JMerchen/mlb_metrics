// node --test docs/prop_cards.test.js
const test = require("node:test")
const assert = require("node:assert/strict")
global.teamName = require("./teams.js").teamName
global.PROP_UNIT_CONTEXT = undefined
const cards = require("./prop_cards.js")

const text = html => html.replace(/<[^>]+>/g, " ").replace(/&middot;/g, "·").replace(/\s+/g, " ").trim()

const WALKER = {
  rank: "1", player_name: "Kenneth Walker III", team: "KC", opponent: "SEA", position: "RB",
  category: "Rushing Yards", direction: "Under", player_rate: "114.2", projection: "72.5",
  projected_carries: "15.1", opponent_allowed_rate: "3.74", league_rate: "4.28", ratio: "0.87",
  context_note: "", recent_values: "116, 62, 135, 173, 117, 70, 177, 128",
  recent_weeks: "2025 W17, 2025 W18, 2026 W1, 2026 W2, 2026 W3, 2026 W4, 2026 W5, 2026 W6",
}
const SIMMONS = {
  rank: "7", player_name: "Jeffery Simmons", team: "TEN", opponent: "CLE", position: "DT", category: "Sacks",
  direction: "Over", player_rate: "0.739", projection: "", opponent_allowed_rate: "3.15", league_rate: "2.35",
  ratio: "1.339", recent_values: "1, 1, 0, 1, 1, 0, 0, 2", recent_weeks: "",
}

test("lean gap: projection vs his average, or the opponent ratio when there is no projection", () => {
  assert.ok(Math.abs(cards.propLeanGap(WALKER).gap - (72.5 / 114.2 - 1)) < 1e-9)
  assert.equal(cards.propLeanGap(WALKER).basis, "projection")
  assert.ok(Math.abs(cards.propLeanGap(SIMMONS).gap - 0.339) < 1e-9)
  assert.equal(cards.propLeanGap(SIMMONS).basis, "opponent")
  assert.equal(cards.propLeanGap({}).gap, null)
})

test("recent games parse oldest first, and the cleared count is on the side we lean", () => {
  const games = cards.propRecentGames(WALKER)
  assert.equal(games.length, 8)
  assert.deepEqual(games[0], { value: 116, label: "2025 W17" })
  // under 72.5: 62 and 70
  assert.deepEqual(cards.propClearedCount(WALKER), { hits: 2, games: 8, line: 72.5 })
  // sacks: no projection, so his 0.739 average is the line; over it in 5 of 8
  assert.deepEqual(cards.propClearedCount(SIMMONS), { hits: 5, games: 8, line: 0.739 })
  assert.equal(cards.propClearedCount({ ...WALKER, recent_values: "" }), null)
})

test("headline says the projection against his average in words", () => {
  assert.equal(text(cards.propHeadline(WALKER)), "We project 72.5 rushing yards, 37% below his 114.2 average")
  assert.equal(text(cards.propHeadline(SIMMONS)), "He averages 0.74 sacks a game")
})

test("card: who, lean, the dots, the cleared count and the reasons", () => {
  const units = { KC: { rushing_plus: "158" }, SEA: { run_defense_plus: "148" } }
  global.PROP_UNIT_CONTEXT = { "Rushing Yards": [["His Rushing+", "own", "rushing_plus"], ["Opp Run D+", "opp", "run_defense_plus"]] }
  const html = cards.propCardHtml(WALKER, units)
  const t = text(html)
  assert.ok(t.startsWith("#1 Kenneth Walker III RB · Chiefs vs Seahawks Under Rushing yards We project 72.5"))
  assert.ok(t.includes("Under 72.5 in 2 of his last 8 games"))
  assert.ok(t.includes("Seahawks allow 3.7 yards per carry to RBs (league 4.3): 13% less than average"))
  assert.ok(t.includes("His Rushing+ 158 vs Opp Run D+ 148: +10 edge"))
  assert.ok(t.includes("About 15.1 carries expected"))
  assert.equal((html.match(/class="dotUnder"/g) || []).length, 2)     // the two games under 72.5
  assert.equal((html.match(/class="dotOther"/g) || []).length, 6)
  assert.ok(html.includes("<title>2026 W4: 70 yards (under)</title>"))
  assert.ok(html.includes('id="prop-card-1"'))
  global.PROP_UNIT_CONTEXT = undefined
})

test("card: a sacks pick has no projection line, and names are escaped", () => {
  const html = cards.propCardHtml({ ...SIMMONS, player_name: "A <b>Bold</b> Name" }, null)
  assert.ok(!html.includes("stripProjection"))
  assert.ok(html.includes("A &lt;b&gt;Bold&lt;/b&gt; Name"))
  assert.ok(text(html).includes("Browns give up 3.1 sacks a game (league 2.4): 34% more than average"))
  assert.ok(text(html).includes("Over 0.7 in 5 of his last 8 games"))
})

test("lean chart: one bar per pick, unders left of centre, overs right, labelled in words", () => {
  const html = cards.leanChartHtml([WALKER, SIMMONS])
  const t = text(html)
  assert.ok(t.includes("Kenneth Walker III rushing yards Under −37%"))
  assert.ok(t.includes("Jeffery Simmons sacks Over +34%"))
  assert.ok(html.includes('class="leanBar leanUnder" style="right:50%'))
  assert.ok(html.includes('class="leanBar leanOver" style="left:50%'))
  assert.ok(html.includes('href="#prop-card-7"'))
  assert.equal(cards.leanChartHtml([]), "")
})
