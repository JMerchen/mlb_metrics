const test = require("node:test")
const assert = require("node:assert")
const { wtSigned, wtRecord, wtSort, wtTeamsHtml } = require("./war_teams.js")

const text = html => html.replace(/<[^>]+>/g, " ").replace(/&amp;/g, "&").replace(/\s+/g, " ").trim()
const DATA = {
  season: 2026, replacement_win_rate: 0.273,
  teams: [
    { team: "LAD", wins: 100, losses: 62, war: 59.1, expected: 103.3, offense: 110, defense: 112, uncredited: -1.4, close: -1.9 },
    { team: "MIL", wins: 103, losses: 59, war: 50.3, expected: 94.6, offense: 13, defense: 112, uncredited: 7.9, close: 0.5 },
    { team: "COL", wins: 58, losses: 104, war: 16.2, expected: 60.5, offense: -53, defense: -168, uncredited: 3.9, close: -6.4 },
  ],
}
const NAMES = code => ({ LAD: "Dodgers", MIL: "Brewers", COL: "Rockies" }[code] || code)

test("signed numbers carry a plus or a real minus, and zero carries neither", () => {
  assert.strictEqual(wtSigned(7.94, 1), "+7.9")
  assert.strictEqual(wtSigned(-6.4, 1), "−6.4")
  assert.strictEqual(wtSigned(-0.04, 1), "0.0")
  assert.strictEqual(wtSigned(110, 0), "+110")
})

test("a record is wins and losses, with ties as a third number", () => {
  assert.strictEqual(wtRecord({ wins: 100, losses: 62 }), "100-62")
  assert.strictEqual(wtRecord({ wins: 9.5, losses: 6.5 }), "9-6-1")
  assert.strictEqual(wtRecord({ wins: 8, losses: 8 }), "8-8")
})

test("teams sort by any column, biggest first by default, names A to Z", () => {
  const by = (key, dir) => wtSort(DATA.teams.map(t => ({ ...t, name: NAMES(t.team) })), key, dir).map(t => t.team)
  assert.deepStrictEqual(by("expected", -1), ["LAD", "MIL", "COL"])
  assert.deepStrictEqual(by("wins", -1), ["MIL", "LAD", "COL"])
  assert.deepStrictEqual(by("close", 1), ["COL", "LAD", "MIL"])
  assert.deepStrictEqual(by("team", -1), ["MIL", "LAD", "COL"])            // Brewers, Dodgers, Rockies
})

test("the table says what a replacement team would win and lays each team's season out", () => {
  const html = wtTeamsHtml(DATA, { unit: "runs", seasonGames: 162, name: NAMES })
  const t = text(html)
  assert.ok(t.includes("A replacement-level team would win about 44 of 162 games (27%)."))
  assert.ok(t.includes("Team WAR is the wins a team's players add on top of that."))
  assert.ok(t.indexOf("Dodgers") < t.indexOf("Brewers") && t.indexOf("Brewers") < t.indexOf("Rockies"))
  assert.ok(t.includes("Dodgers 100-62 59.1 103.3 −1.4 −1.9 +110 +112"))
  assert.ok(t.includes("Brewers 103-59 50.3 94.6 +7.9 +0.5"))
  assert.ok(t.includes("Expected wins Exp. ▼"))
  assert.ok(html.includes('data-team-sort="close"') && html.includes("wtUp") && html.includes("wtDown"))
  assert.ok(html.includes("no player&#39;s ledger") || html.includes("no player's ledger"))
})

test("football: fewer games, points, and a sort the page asked for", () => {
  const nfl = { replacement_win_rate: 0.1832, teams: [
    { team: "LA", games: 17, wins: 12, losses: 5, war: 8.4, expected: 11.6, offense: 99, defense: 34, uncredited: 0.7, close: -0.3 },
    { team: "NYJ", games: 17, wins: 3, losses: 14, war: 2.2, expected: 5.3, offense: -84, defense: -36, uncredited: -1.5, close: -0.9 }] }
  const html = wtTeamsHtml(nfl, { unit: "points", seasonGames: 17, sort: "uncredited", dir: 1 })
  const t = text(html)
  assert.ok(t.includes("would win about 3.1 of 17 games (18%)"))
  assert.ok(t.indexOf("NYJ") < t.indexOf("LA 12-5") && t.includes("Not cred. ▲"))
  assert.ok(html.includes("point differential"))
})

test("no teams, no table", () => {
  assert.strictEqual(wtTeamsHtml({ teams: [] }, {}), "")
  assert.strictEqual(wtTeamsHtml(null, {}), "")
})
