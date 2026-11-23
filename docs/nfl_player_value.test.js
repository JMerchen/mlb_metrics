const test = require("node:test")
const assert = require("node:assert")
const { PV_POSITIONS, pvYear, pvSideOf, pvState, pvShownSide, pvColumns, pvFilter, pvSort, pvNumber, pvControlsHtml, pvSummaryHtml, pvTableHtml } = require("./nfl_player_value.js")

const text = html => html.replace(/<[^>]+>/g, " ").replace(/&amp;/g, "&").replace(/\s+/g, " ").trim()
const P = (name, pos, side, team, div, exp, war, extra = {}) => ({ name, pos, side, team, div, exp, games: 10, volume: 300, war, paa: war * 20, par: war * 35, ...extra })
const DATA = {
  season: 2026, through_week: 11, points_per_win: 35,
  players: [
    P("Dak Prescott", "QB", "offense", "DAL", "NFC East", 10, 3.1, { passing: 105, passing_plays: 450, rushing: 3.7, rushing_plays: 14 }),
    P("Drake Maye", "QB", "offense", "NE", "AFC East", 2, 1.96, { passing: 66, passing_plays: 320 }),
    P("Puka Nacua", "WR", "offense", "LA", "NFC West", 3, 0.8, { receiving: 28, receiving_plays: 110 }),
    P("Rookie Receiver", "WR", "offense", "SEA", "NFC West", 0, 0.2, { receiving: 7, receiving_plays: 60 }),
    P("Will Anderson Jr.", "ED", "defense", "HOU", "AFC South", 3, 0.47, { points: 32.8, sacks: 11.5, interceptions: 0, passes_defended: 1, tackles_for_loss: 15, tackles: 36 }),
    P("Kyle Hamilton", "S", "defense", "BAL", "AFC North", 4, 0.34, { points: 27.2, sacks: 0, interceptions: 1, passes_defended: 5, tackles_for_loss: 10, tackles: 37 }),
    P("Cam Little", "K", "special", "JAX", "AFC South", 2, 0.26, { fg_made: 20, fg_tried: 22, games: undefined }),
    P("Bad Passer", "QB", "offense", "NYJ", "AFC East", null, -0.5, { passing: -17, passing_plays: 180 }),
  ],
}

test("a year in the league reads as a rookie or an ordinal", () => {
  assert.strictEqual(pvYear(0), "Rookie")
  assert.strictEqual(pvYear(1), "2nd year")
  assert.strictEqual(pvYear(2), "3rd year")
  assert.strictEqual(pvYear(10), "11th year")
  assert.strictEqual(pvYear(null), "")
})

test("a position belongs to one side, and the state is kept consistent with it", () => {
  assert.strictEqual(pvSideOf("WR"), "offense")
  assert.strictEqual(pvSideOf("ED"), "defense")
  assert.strictEqual(pvSideOf("K"), "special")
  assert.strictEqual(pvState({ side: "defense", pos: "WR" }).pos, "all")      // a receiver is not a defender
  assert.strictEqual(pvState({ side: "offense", pos: "WR" }).pos, "WR")
  assert.strictEqual(pvState({ side: "nonsense" }).side, "all")
  assert.strictEqual(pvShownSide(pvState({ pos: "CB" })), "defense")           // a position alone shows its side's columns
  assert.deepStrictEqual(pvState().sort, "war")
  assert.strictEqual(PV_POSITIONS.length, 10)
})

test("filters: side, position, years in the league, conference or division, and name", () => {
  const names = state => pvFilter(DATA.players, state).map(p => p.name)
  assert.strictEqual(names({}).length, 8)
  assert.deepStrictEqual(names({ side: "defense" }), ["Will Anderson Jr.", "Kyle Hamilton"])
  assert.deepStrictEqual(names({ pos: "WR" }), ["Puka Nacua", "Rookie Receiver"])
  assert.deepStrictEqual(names({ exp: "rookie" }), ["Rookie Receiver"])
  // 2nd to 4th year is one to three seasons behind him; a player with no record of it is left out.
  assert.deepStrictEqual(names({ exp: "early" }), ["Drake Maye", "Puka Nacua", "Will Anderson Jr.", "Cam Little"])
  assert.deepStrictEqual(names({ exp: "veteran" }), ["Dak Prescott"])
  assert.deepStrictEqual(names({ group: "AFC" }), ["Drake Maye", "Will Anderson Jr.", "Kyle Hamilton", "Cam Little", "Bad Passer"])
  assert.deepStrictEqual(names({ group: "NFC West" }), ["Puka Nacua", "Rookie Receiver"])
  assert.deepStrictEqual(names({ group: "AFC", side: "offense", pos: "QB" }), ["Drake Maye", "Bad Passer"])
  assert.deepStrictEqual(names({ search: "  ander " }), ["Will Anderson Jr."])
})

test("sorting: biggest first by default, a missing value last, names A to Z", () => {
  const by = (key, dir) => pvSort(DATA.players, key, dir).map(p => p.name)
  assert.strictEqual(by("war", -1)[0], "Dak Prescott")
  assert.strictEqual(by("war", 1)[0], "Bad Passer")
  assert.deepStrictEqual(by("sacks", -1).slice(0, 2), ["Will Anderson Jr.", "Kyle Hamilton"])
  // Players with no sacks column come after those with one, either way, in WAR order.
  assert.strictEqual(by("sacks", 1)[2], "Dak Prescott")
  assert.strictEqual(by("name", -1)[0], "Bad Passer")
  assert.strictEqual(by("name", 1)[0], "Will Anderson Jr.")
})

test("numbers: fixed digits, a real minus sign, and no minus zero", () => {
  assert.strictEqual(pvNumber(3.104, 2), "3.10")
  assert.strictEqual(pvNumber(-0.5, 2), "−0.50")
  assert.strictEqual(pvNumber(-0.01, 1), "0.0")
  assert.strictEqual(pvNumber(undefined, 1), "")
  assert.strictEqual(pvNumber(11, 0), "11")
})

test("the columns follow the side shown", () => {
  const keys = state => pvColumns(pvState(state)).map(c => c[0])
  assert.deepStrictEqual(keys({}), ["games", "volume", "war", "par", "paa"])
  assert.deepStrictEqual(keys({ side: "offense" }).slice(5), ["passing", "rushing", "receiving"])
  assert.ok(keys({ pos: "S" }).includes("passes_defended"))
  assert.deepStrictEqual(keys({ side: "special" }).slice(5), ["fg_made", "fg_tried"])
})

test("the table ranks the filtered players and says what the group cannot show", () => {
  const html = pvTableHtml(DATA, {})
  const t = text(html)
  assert.ok(t.includes("8 players · 2026 season through week 11 · 35 points = 1 win"))
  assert.ok(t.includes("offensive linemen are not rated"))
  assert.ok(t.indexOf("Dak Prescott") < t.indexOf("Drake Maye") && t.indexOf("Drake Maye") < t.indexOf("Bad Passer"))
  assert.ok(t.includes("DAL · QB · 11th year") && t.includes("SEA · WR · Rookie"))
  assert.ok(t.includes("3.10") && t.includes("−0.50"))
  assert.ok(html.includes('data-sort="war"') && t.includes("WAR ▼"))
  const defense = text(pvTableHtml(DATA, { side: "defense" }))
  assert.ok(defense.includes("2 players") && defense.includes("coverage nobody tested is invisible") && defense.includes("Snaps"))
  assert.ok(defense.includes("11.5") && !defense.includes("Dak Prescott"))
  assert.ok(text(pvTableHtml(DATA, { side: "special" })).includes("Kicks"))
})

test("a long table is cut with a way to see the rest; nothing matching says so; no data says so", () => {
  const html = pvTableHtml(DATA, { limit: 3 })
  assert.strictEqual((html.match(/<tr>/g) || []).length, 4)           // the heading row and three players
  assert.ok(html.includes("data-pv-more") && text(html).includes("Show all 8"))
  assert.ok(!pvTableHtml(DATA, { limit: 50 }).includes("data-pv-more"))
  assert.ok(text(pvTableHtml(DATA, { search: "zzz" })).includes("Nobody matches those filters."))
  assert.ok(text(pvTableHtml({ players: [] }, {})).includes("No player values for this season yet."))
  assert.ok(text(pvTableHtml(null, {})).includes("No player values"))
})

test("the controls offer every filter and mark what is chosen", () => {
  const html = pvControlsHtml([2026, 2025], { season: 2025, side: "defense", pos: "CB", exp: "rookie", group: "AFC North", search: "a\"b" })
  assert.ok(html.includes('<option value="2025" selected>') && html.includes('<option value="CB" selected>'))
  assert.ok(html.includes('<option value="rookie" selected>') && html.includes('<option value="AFC North" selected>'))
  assert.ok(html.includes('value="a&quot;b"'))
  // With a side chosen, only its positions are offered.
  assert.ok(!html.includes('value="WR"') && html.includes('value="ED"'))
  assert.ok(/class="tabButton active" data-pv-side="defense"/.test(html))
  const all = pvControlsHtml([2026], {})
  assert.ok(all.includes('value="WR"') && all.includes('value="K"') && all.includes("NFC West"))
  assert.ok(/class="tabButton active" data-pv-side="all"/.test(all))
  assert.ok(text(pvSummaryHtml(DATA, { side: "offense" }, 1)).startsWith("1 player ·"))
})
