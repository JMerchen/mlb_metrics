// node --test docs/age_render.test.js
const test = require("node:test")
const assert = require("node:assert/strict")
global.teamName = require("./teams.js").teamName
Object.assign(global, require("./mlb_render.js"))
const a = require("./age_render.js")

const text = html => html.replace(/<\/(div|span|td|button)>/g, " ").replace(/<[^>]+>/g, "").replace(/\s+/g, " ").trim()

const PLAYER = { key_mlbam: "670541", name: "Yordan Álvarez", metric: "OPS", age: "29", value: "0.9998",
  n_comparables: "25", n_with_next_season: "2", projected_value_mean: "0.946", projected_value_p25: "0.858", projected_value_p75: "1.046" }
const COMPS = [
  { key_mlbam: "670541", metric: "OPS", name: "Miguel Cabrera", yearID: "2012", age: "29", value: "0.999", next_value: "1.078" },
  { key_mlbam: "670541", metric: "OPS", name: "Charlie Blackmon", yearID: "2017", age: "30", value: "1.000", next_value: "0.860" },
  { key_mlbam: "670541", metric: "OPS", name: "Pedro Guerrero", yearID: "1985", age: "29", value: "0.999", next_value: "" },
  { key_mlbam: "1", metric: "OPS", name: "Someone Else", yearID: "2000", age: "29", value: "0.7", next_value: "0.7" },
]
const LEAGUE = [
  { metric: "OPS", age: "28", value: "0.735", n_seasons: "3020" }, { metric: "OPS", age: "29", value: "0.7375", n_seasons: "2775" },
  { metric: "OPS", age: "45", value: "0.771", n_seasons: "4" },
]
const HISTORY = [{ key_mlbam: "670541", metric: "OPS", age: "27", value: "0.96" }, { key_mlbam: "670541", metric: "OPS", age: "28", value: "0.98" }]

test("formatting: rate stats drop the leading zero, per-nine keep two decimals; direction by stat", () => {
  assert.equal(a.ageValue("0.9461", "OPS"), ".946")
  assert.equal(a.ageValue("1.046", "OPS"), "1.046")
  assert.equal(a.ageValue("9.41", "K9"), "9.41")
  assert.equal(a.ageDelta(-0.03, "OPS"), "−.030")
  assert.equal(a.ageIsBetter(-0.5, "FIP"), true)
  assert.equal(a.ageIsBetter(-0.5, "K9"), false)
  assert.equal(a.ageMedian([3, 1, 2, null]), 2)
})

test("comparables: only his, with where they went next", () => {
  const comps = a.ageComparables(COMPS, "670541", "OPS")
  assert.equal(comps.length, 3)
  assert.equal(comps[0].delta.toFixed(3), "0.079")
  assert.equal(comps[2].delta, null)
  const list = text(a.ageComparablesHtml(comps, "OPS"))
  assert.ok(list.includes("2012 · age 29 .999 → 1.078 +.079"))
  assert.ok(list.includes("2017 · age 30 1.000 → .860 −.140"))
  assert.ok(list.includes("no next year"))
})

test("hero: this season vs. his age's average, projection range and what matches did", () => {
  const comps = a.ageComparables(COMPS, "670541", "OPS")
  const hero = text(a.ageHeroHtml(PLAYER, "OPS", LEAGUE, comps, "HOU"))
  assert.ok(hero.includes("Astros · age 29 · hitter"))
  assert.ok(hero.includes("This season 1.000 OPS"))
  assert.ok(hero.includes("+.262 vs. the average 29-year-old"))
  assert.ok(hero.includes(".946 OPS likely between .858 and 1.046"))
  assert.ok(hero.includes("Of his 3 closest matches at age 29, 2 played the next season; their OPS changed by a median of −.030."))
  assert.ok(a.ageHeroHtml(Object.assign({}, PLAYER, { projected_value_mean: "" }), "OPS", LEAGUE, comps).includes("Not enough comparable"))
})

test("chart: league curve only where enough seasons, his seasons, comparables' steps and the projection", () => {
  const comps = a.ageComparables(COMPS, "670541", "OPS")
  const svg = a.ageChartSvg(PLAYER, "OPS", LEAGUE, HISTORY, comps, 600)
  assert.ok(svg.startsWith('<svg class="ageChart" width="600"'))
  assert.equal((svg.match(/class="ageOwnDot"/g) || []).length, 3)          // ages 27, 28 and this season
  assert.equal((svg.match(/class="ageComp"/g) || []).length, 2)           // the two with a next season
  assert.ok(svg.includes("ageCompUp") && svg.includes("ageCompDown"))
  assert.ok(svg.includes("Next season, at 30: about .946, likely .858 to 1.046"))
  assert.ok(!svg.includes(">45<"))                                          // 4-season league point dropped
  assert.deepEqual(Object.keys(a.ageLeagueByAge(LEAGUE, "OPS")), ["28", "29"])
})
