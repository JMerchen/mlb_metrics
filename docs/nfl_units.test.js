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
  assert.equal(unitCellClass("110"), "unitStrong")
  assert.equal(unitCellClass(90), "unitWeak")
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
