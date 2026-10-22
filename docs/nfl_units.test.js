// node --test docs/nfl_units.test.js
const test = require("node:test")
const assert = require("node:assert/strict")
global.teamName = require("./teams.js").teamName
const { unitsTableHtml, sortUnitRows, unitCellClass, propUnitContext, unitsByTeam } = require("./nfl_units.js")

const text = html => html.replace(/<\/t[dh]>/g, " ").replace(/<[^>]+>/g, "").replace(/\s+/g, " ").trim()

const ROWS = [
  { team: "KC", opponent: "DEN", qb_name: "Patrick Mahomes", qb_plus: "112", receiving_plus: "104", rushing_plus: "96",
    pass_defense_plus: "108", run_defense_plus: "88", week: "7" },
  { team: "DEN", opponent: "KC", qb_name: "Bo Nix", qb_plus: "95", receiving_plus: "93", rushing_plus: "102",
    pass_defense_plus: "121", run_defense_plus: "105", week: "7" },
  { team: "BUF", opponent: "", qb_name: "Josh Allen", qb_plus: "118", receiving_plus: "115", rushing_plus: "110",
    pass_defense_plus: "90", run_defense_plus: "", week: "7" },
]

test("units table: nicknames, opponent or bye, whole numbers, colour bands", () => {
  const html = unitsTableHtml(ROWS, "team")
  assert.equal(text(html).split(" Bills")[0], "Team ▼ QB QB+ Receiving+ Rushing+ Pass D+ Run D+")
  assert.ok(text(html).includes("Bills bye Josh Allen 118 115 110 90 -"))
  assert.ok(text(html).includes("Chiefs vs Broncos Patrick Mahomes 112 104 96 108 88"))
  assert.equal(unitCellClass("100"), "")
  assert.equal(unitsTableHtml([], "team"), "Team unit ratings appear after the next weekly update.")
})

test("units sort: by team name, or strongest first with blanks last", () => {
  assert.deepEqual(sortUnitRows(ROWS, "team").map(r => r.team), ["BUF", "DEN", "KC"])
  assert.deepEqual(sortUnitRows(ROWS, "pass_defense_plus").map(r => r.team), ["DEN", "KC", "BUF"])
  assert.deepEqual(sortUnitRows(ROWS, "run_defense_plus").map(r => r.team), ["DEN", "KC", "BUF"])
})

test("prop rows: the units each category leans on", () => {
  const units = unitsByTeam(ROWS)
  assert.equal(propUnitContext({ category: "Receiving Yards", team: "KC", opponent: "DEN" }, units), "His QB+ 112 · Opp Pass D+ 121")
  assert.equal(propUnitContext({ category: "Rushing Yards", team: "DEN", opponent: "KC" }, units), "His Rushing+ 102 · Opp Run D+ 88")
  assert.equal(propUnitContext({ category: "Passing Yards", team: "KC", opponent: "DEN" }, units), "His Receiving+ 104 · Opp Pass D+ 121")
  assert.equal(propUnitContext({ category: "Sacks", team: "DEN", opponent: "KC" }, units), "Opp QB+ 112")
  assert.equal(propUnitContext({ category: "Rushing Yards", team: "KC", opponent: "BUF" }, units), "His Rushing+ 96")
  assert.equal(propUnitContext({ category: "Receiving Yards", team: "NYJ", opponent: "MIA" }, units), "")
  assert.equal(propUnitContext({ category: "Receiving Yards", team: "KC", opponent: "DEN" }, null), "")
})

test("colour bands follow 30 points per standard deviation", () => {
  assert.equal(unitCellClass("130"), "unitStrong")
  assert.equal(unitCellClass("129"), "")
  assert.equal(unitCellClass(70), "unitWeak")
})

test("matchup differentials: passing is the QB/Receiving average minus the opponent's Pass D+", () => {
  const { matchupDifferentials } = require("./nfl_units.js")
  const rows = [
    { team: "SF", opponent: "ATL", qb_plus: "117", receiving_plus: "111", rushing_plus: "111", pass_defense_plus: "98", run_defense_plus: "112" },
    { team: "ATL", opponent: "SF", qb_plus: "99", receiving_plus: "100", rushing_plus: "116", pass_defense_plus: "102", run_defense_plus: "101" },
    { team: "BUF", opponent: "", qb_plus: "111", receiving_plus: "115", rushing_plus: "115", pass_defense_plus: "90", run_defense_plus: "96" },
  ]
  const passing = matchupDifferentials(rows, "passing")
  // the user's example: (117 + 111) / 2 - 102 = 12
  assert.deepEqual(passing[0], { team: "SF", opponent: "ATL", offense: 114, defense: 102, differential: 12 })
  assert.equal(passing[1].team, "ATL")
  assert.equal(passing[1].differential, 99.5 - 98)
  assert.equal(passing[2].team, "BUF")            // bye: no opponent, no differential, listed last
  assert.equal(passing[2].differential, null)

  const rushing = matchupDifferentials(rows, "rushing")
  assert.deepEqual(rushing.map(r => [r.team, r.differential]), [["ATL", 4], ["SF", 10]].sort((a, b) => b[1] - a[1]).concat([["BUF", null]]))

})

test("matchup cards: one per game, away team first, both differentials; byes in their own table", () => {
  const { matchupGames, matchupCardsHtml, byeTableHtml } = require("./nfl_units.js")
  const rows = [
    { team: "DET", opponent: "GB", home: "1", qb_plus: "146", receiving_plus: "146", rushing_plus: "105", pass_defense_plus: "57", run_defense_plus: "51" },
    { team: "GB", opponent: "DET", home: "0", qb_plus: "125", receiving_plus: "124", rushing_plus: "55", pass_defense_plus: "126", run_defense_plus: "32" },
    { team: "SF", opponent: "ATL", home: "0", qb_plus: "151", receiving_plus: "133", rushing_plus: "133", pass_defense_plus: "93", run_defense_plus: "136" },
    { team: "ATL", opponent: "SF", home: "1", qb_plus: "97", receiving_plus: "101", rushing_plus: "149", pass_defense_plus: "105", run_defense_plus: "103" },
    { team: "BUF", opponent: "", home: "", qb_name: "Josh Allen", qb_plus: "132" },
    { team: "JAX", opponent: "", home: "", qb_name: "Trevor Lawrence", qb_plus: "130" },
  ]
  const games = matchupGames(rows)
  assert.deepEqual(games.map(g => g.title), ["Packers at Lions", "49ers at Falcons"])   // +73 rushing makes DET-GB the most lopsided
  assert.deepEqual(games[0].teams, [
    { team: "GB", passing: 124.5 - 57, rushing: 55 - 51 },
    { team: "DET", passing: 146 - 126, rushing: 105 - 32 },
  ])
  const html = text(matchupCardsHtml(rows).replace(/<\/t[dh]>/g, " "))
  assert.ok(html.startsWith("Packers at Lions Passing differential Rushing differential Packers +67.5 +4 Lions +20 +73"))
  assert.equal(text(byeTableHtml(rows).replace(/<\/t[dh]>/g, " ")), "Team QB Bills Josh Allen Jaguars Trevor Lawrence")
  assert.equal(byeTableHtml(rows.slice(0, 4)), "No teams on bye this week.")
})

test("matchup cards without a home flag (older file): alphabetical, 'vs'", () => {
  const { matchupGames } = require("./nfl_units.js")
  const rows = [
    { team: "DET", opponent: "GB", rushing_plus: "105", run_defense_plus: "51" },
    { team: "GB", opponent: "DET", rushing_plus: "55", run_defense_plus: "32" },
  ]
  assert.equal(matchupGames(rows)[0].title, "Lions vs Packers")
})
