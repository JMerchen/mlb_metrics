const test = require("node:test")
const assert = require("node:assert")
const { MW_POSITIONS, mwKindOf, mwState, mwShownKind, mwColumns, mwFilter, mwSort, mwNumber, mwControlsHtml, mwTableHtml } = require("./mlb_war.js")

const text = html => html.replace(/<[^>]+>/g, " ").replace(/&amp;/g, "&").replace(/\s+/g, " ").trim()
const H = (name, pos, team, div, age, war, extra = {}) => ({ name, pos, kind: "hitter", team, div, age, games: 150, war, raa: war * 6, rar: war * 9.7, pa: 600, batting: war * 5, fielding: war, ...extra })
const P = (name, pos, team, div, age, war, extra = {}) => ({ name, pos, kind: "pitcher", team, div, age, games: 30, war, raa: war * 7, rar: war * 9.7, bf: 700, pitching: war * 7, ...extra })
const DATA = {
  season: 2026, through: "2026-09-27", runs_per_win: 9.7,
  players: [
    H("Pete Crow-Armstrong", "CF", "CHC", "NL Central", 24, 8.97, { fielding: 13.5 }),
    H("Zach Neto", "SS", "LAA", "AL West", 25, 6.17, { fielding: 7.0 }),
    H("Gunnar Henderson", "SS", "BAL", "AL East", 25, 5.4, { fielding: 10.1 }),
    H("Mike Trout", "CF", "LAA", "AL West", 35, 6.12),
    H("Shohei Ohtani", "DH", "LAD", "NL West", 32, 5.92, { bf: 340, pitching: 12.9 }),
    P("Jacob Misiorowski", "SP", "MIL", "NL Central", 24, 5.14),
    P("Chris Sale", "SP", "ATL", "NL East", 37, 4.62, { fielding: 1.7 }),
    P("Some Reliever", "RP", "NYY", "AL East", null, -0.4),
  ],
}

test("a position is a hitter's or a pitcher's, and the state is kept consistent with it", () => {
  assert.strictEqual(mwKindOf("SS"), "hitter")
  assert.strictEqual(mwKindOf("RP"), "pitcher")
  assert.strictEqual(mwState({ kind: "pitcher", pos: "SS" }).pos, "all")
  assert.strictEqual(mwState({ kind: "hitter", pos: "SS" }).pos, "SS")
  assert.strictEqual(mwState({ kind: "nonsense" }).kind, "all")
  assert.strictEqual(mwShownKind(mwState({ pos: "SP" })), "pitcher")
  assert.strictEqual(MW_POSITIONS.length, 11)
})

test("filters: hitters or pitchers, position, age, league or division, and name", () => {
  const names = state => mwFilter(DATA.players, state).map(p => p.name)
  assert.strictEqual(names({}).length, 8)
  assert.deepStrictEqual(names({ kind: "pitcher" }), ["Jacob Misiorowski", "Chris Sale", "Some Reliever"])
  assert.deepStrictEqual(names({ pos: "SS" }), ["Zach Neto", "Gunnar Henderson"])
  // 25 and under; a player with no age on record is left out of any age group.
  assert.deepStrictEqual(names({ age: "young" }), ["Pete Crow-Armstrong", "Zach Neto", "Gunnar Henderson", "Jacob Misiorowski"])
  assert.deepStrictEqual(names({ age: "veteran" }), ["Mike Trout", "Chris Sale"])
  assert.deepStrictEqual(names({ group: "AL" }), ["Zach Neto", "Gunnar Henderson", "Mike Trout", "Some Reliever"])
  assert.deepStrictEqual(names({ group: "AL West" }), ["Zach Neto", "Mike Trout"])
  assert.deepStrictEqual(names({ group: "AL", pos: "SS", age: "young" }), ["Zach Neto", "Gunnar Henderson"])
  assert.deepStrictEqual(names({ search: "ohtani" }), ["Shohei Ohtani"])
})

test("sorting: biggest first by default, a missing value last, names A to Z", () => {
  const by = (key, dir) => mwSort(DATA.players, key, dir).map(p => p.name)
  assert.strictEqual(by("war", -1)[0], "Pete Crow-Armstrong")
  assert.strictEqual(by("war", 1)[0], "Some Reliever")
  assert.deepStrictEqual(by("fielding", -1).slice(0, 2), ["Pete Crow-Armstrong", "Gunnar Henderson"])
  // Pitching: the two-way player counts; hitters with none come last, in WAR order.
  assert.deepStrictEqual(by("pitching", -1).slice(0, 2), ["Jacob Misiorowski", "Chris Sale"])
  assert.strictEqual(by("pitching", -1)[4], "Pete Crow-Armstrong")
  assert.strictEqual(by("name", -1)[0], "Chris Sale")
})

test("the columns follow who is shown", () => {
  const keys = state => mwColumns(mwState(state)).map(c => c[0])
  assert.deepStrictEqual(keys({}), ["games", "pa", "bf", "war", "rar", "raa", "batting", "fielding", "pitching"])
  assert.deepStrictEqual(keys({ kind: "hitter" }), ["games", "pa", "war", "rar", "raa", "batting", "fielding"])
  assert.deepStrictEqual(keys({ pos: "RP" }), ["games", "bf", "war", "rar", "raa", "fielding", "pitching"])
})

test("numbers: fixed digits, a real minus sign, and no minus zero", () => {
  assert.strictEqual(mwNumber(8.968, 2), "8.97")
  assert.strictEqual(mwNumber(-3.4, 1), "−3.4")
  assert.strictEqual(mwNumber(-0.04, 1), "0.0")
  assert.strictEqual(mwNumber(undefined, 1), "")
})

test("the table ranks the filtered players and says what is left out", () => {
  const html = mwTableHtml(DATA, {})
  const t = text(html)
  assert.ok(t.includes("8 players · 2026 season through 2026-09-27 · 9.7 runs = 1 win"))
  assert.ok(t.includes("Baserunning and a catcher's framing are not in it yet."))
  assert.ok(t.indexOf("Pete Crow-Armstrong") < t.indexOf("Zach Neto") && t.indexOf("Zach Neto") < t.indexOf("Some Reliever"))
  assert.ok(t.includes("CHC · CF · age 24") && t.includes("NYY · RP"))
  assert.ok(t.includes("8.97") && t.includes("−0.40") && t.includes("WAR ▼"))
  const pitchers = text(mwTableHtml(DATA, { kind: "pitcher" }))
  assert.ok(pitchers.includes("3 players") && pitchers.includes("Batters faced") && !pitchers.includes("Plate appearances"))
})

test("a long table is cut with a way to see the rest; nothing matching and no data say so", () => {
  const html = mwTableHtml(DATA, { limit: 3 })
  assert.strictEqual((html.match(/<tr>/g) || []).length, 4)
  assert.ok(html.includes("data-mw-more") && text(html).includes("Show all 8"))
  assert.ok(!mwTableHtml(DATA, {}).includes("data-mw-more"))
  assert.ok(text(mwTableHtml(DATA, { search: "zzz" })).includes("Nobody matches those filters."))
  assert.ok(text(mwTableHtml({ players: [] }, {})).includes("No player values for this season yet."))
})

test("the controls offer every filter and mark what is chosen", () => {
  const html = mwControlsHtml([2026, 2025], { season: 2025, kind: "pitcher", pos: "RP", age: "prime", group: "NL West", search: "x" })
  assert.ok(html.includes('<option value="2025" selected>') && html.includes('<option value="RP" selected>'))
  assert.ok(html.includes('<option value="prime" selected>') && html.includes('<option value="NL West" selected>'))
  assert.ok(!html.includes('value="SS"') && html.includes('value="SP"'))
  assert.ok(/class="tabButton active" data-mw-kind="pitcher"/.test(html))
  const all = mwControlsHtml([2026], {})
  assert.ok(all.includes('value="SS"') && all.includes("American League") && /class="tabButton active" data-mw-kind="all"/.test(all))
})
