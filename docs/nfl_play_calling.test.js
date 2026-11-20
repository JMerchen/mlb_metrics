const test = require("node:test")
const assert = require("node:assert")
const { pcExpected, pcTells, pcPhrase, pcOrdinal, pcBoxHtml, playCallingHtml, playCallingMatchupHtml } = require("./nfl_play_calling.js")

const text = html => html.replace(/<[^>]+>/g, " ").replace(/&amp;/g, "&").replace(/\s+/g, " ").trim()
const cell = (plays, passes, typical, rate) => ({ plays, passes, typical, rate, lean: Math.round((rate - typical) * 1e4) / 1e4 })
const STATES = [
  { key: "open", label: "To open a drive", group: "1st down" },
  { key: "first_after_run", label: "After a run moved the chains", group: "1st down" },
  { key: "second_long_after_pass", label: "2nd & long, after a pass that failed", group: "2nd down" },
  { key: "third_long", label: "3rd & long", group: "3rd down" },
]
const HABITS = [{ key: "after_two_runs", label: "After two runs in a row" }, { key: "red_zone", label: "Inside the 20" }]
const league = (pass, run) => ({ plays: 1000, pass_rate: 0.5, lean: 0, pass_success: pass, run_success: run })
const team = (offense, habits, extra = {}) => ({
  offense, habits, offense_effect: { pass: 0, run: 0 }, defense_effect: { pass: 0, run: 0 },
  defense: Object.fromEntries(STATES.map(s => [s.key, { passes: 20, pass_success: 9, runs: 10, run_success: 4 }])),
  readable: { plays: 500, situation: 0.71, habits: 0.74, rank: 2 }, ...extra,
})
const DATA = {
  season: 2026, through_week: 10, states: STATES, habits: HABITS,
  league: { open: league(0.5, 0.36), first_after_run: league(0.47, 0.38), second_long_after_pass: league(0.42, 0.28), third_long: league(0.31, 0.35),
            after_two_runs: league(0.45, 0.4), red_zone: league(0.44, 0.42) },
  teams: {
    IND: team({ open: cell(114, 56, 0.48, 0.49), first_after_run: cell(67, 21, 0.47, 0.38), second_long_after_pass: cell(57, 49, 0.68, 0.78), third_long: cell(12, 12, 0.87, 0.93) },
              { after_two_runs: cell(115, 55, 0.56, 0.5), red_zone: cell(0, 0, null, null) },
              { offense_effect: { pass: -0.026, run: 0.017 } }),
    HOU: team({ open: cell(98, 52, 0.5, 0.52), first_after_run: cell(42, 19, 0.45, 0.45), second_long_after_pass: cell(48, 33, 0.72, 0.71), third_long: cell(62, 55, 0.87, 0.88) },
              { after_two_runs: cell(68, 43, 0.54, 0.59), red_zone: cell(83, 47, 0.56, 0.56) },
              { defense_effect: { pass: -0.014, run: 0.002 }, readable: { plays: 500, situation: 0.73, habits: 0.73, rank: 10 } }),
  },
}
const NAMES = code => ({ IND: "Colts", HOU: "Texans" }[code] || code)

test("what to expect from a call is the league's rate there moved by both teams' overall effects", () => {
  const e = pcExpected(DATA, "IND", "HOU", "second_long_after_pass")
  assert.ok(Math.abs(e.pass - (0.42 - 0.026 - 0.014)) < 1e-9)
  assert.ok(Math.abs(e.run - (0.28 + 0.017 + 0.002)) < 1e-9)
})

test("tells are the biggest leans with enough plays behind them, across situations and habits", () => {
  const tells = pcTells(DATA, "IND")
  assert.deepEqual(tells.map(t => t.key), ["second_long_after_pass", "first_after_run"])
  // 12 plays is not enough, however one-sided; and Houston has nothing 5 points from typical but its two-run habit.
  assert.ok(!pcTells(DATA, "IND", 10).some(t => t.key === "third_long"))
  assert.deepEqual(pcTells(DATA, "HOU").map(t => t.key), ["after_two_runs"])
})

test("a situation reads naturally inside a sentence", () => {
  assert.equal(pcPhrase({ ...STATES[0], habit: false }), "to open a drive")
  assert.equal(pcPhrase({ ...STATES[1], habit: false }), "on 1st down after a run moved the chains")
  assert.equal(pcPhrase({ ...STATES[2], habit: false }), "on 2nd & long, after a pass that failed")
  assert.equal(pcPhrase({ ...HABITS[0], habit: true }), "after two runs in a row")
  assert.equal(pcOrdinal(1), "1st")
  assert.equal(pcOrdinal(22), "22nd")
  assert.equal(pcOrdinal(13), "13th")
})

test("a box shows the split, the typical team's tick, the lean when it is a tell, and the defense's record in the hover text", () => {
  const row = { ...STATES[2], ...DATA.teams.IND.offense.second_long_after_pass, habit: false }
  const html = pcBoxHtml(DATA, "IND", "HOU", row, NAMES)
  assert.match(html, /class="callBox callTell"/)
  assert.match(html, /class="callPass" style="width:78\.0%"/)
  assert.match(html, /class="callTick" style="left:68\.0%"/)
  assert.match(text(html), /2nd & long, after a pass that failed 78% pass typical 68% · 57 plays 10 points more passing than typical Against Texans: a pass works 38%, a run 30%/)
  assert.match(html, /data-tip="Colts: 49 passes in 57 plays here \(86%\); after allowing for the small count, 78%\. A typical team in the same plays: 68%\. Texans defense here: passes worked 9 of 20 times, runs 4 of 10\."/)
  // Too few plays: no tell, however far the rate. No plays: said so.
  const thin = pcBoxHtml(DATA, "IND", "HOU", { ...STATES[3], ...DATA.teams.IND.offense.third_long, habit: false }, NAMES)
  assert.doesNotMatch(thin, /callTell|more passing/)
  assert.match(text(pcBoxHtml(DATA, "IND", "HOU", { ...HABITS[1], ...DATA.teams.IND.habits.red_zone, habit: true }, NAMES)), /Inside the 20 No plays yet/)
})

test("one offense's map: the read, the down-by-down tree and the habits", () => {
  const html = playCallingHtml(DATA, "IND", "HOU", NAMES)
  const t = text(html)
  assert.match(t, /^Colts offense against the Texans defense pass run a typical team/)
  assert.match(t, /Biggest tells: on 2nd & long, after a pass that failed, they throw 78% of the time \(a typical team 68%\); on 1st down after a run moved the chains, they run 62% of the time \(a typical team 53%\)\./)
  assert.match(t, /right 71% of the time against them; knowing these habits, 74% \(the 2nd easiest offense of 2 to read\)/)
  assert.match(t, /their passes should work 4\.0 points less often and their runs 1\.9 points more often/)
  assert.equal((html.match(/<div class="callCol">/g) || []).length, 3)
  assert.match(t, /1st down To open a drive .* 2nd down 2nd & long.* 3rd down 3rd & long .* Habits that cut across downs After two runs in a row/)
  assert.match(text(playCallingHtml(DATA, "HOU", "IND", NAMES)), /Biggest tells: after two runs in a row, they throw 59%/)
  assert.equal(playCallingHtml(DATA, "IND", "XXX", NAMES), "")
  assert.equal(playCallingHtml(null, "IND", "HOU", NAMES), "")
})

test("a game shows the road offense first, then the home offense", () => {
  const html = playCallingMatchupHtml(DATA, "HOU", "IND", NAMES)
  assert.ok(html.indexOf("Colts offense") < html.indexOf("Texans offense") && html.indexOf("Colts offense") >= 0)
  assert.equal(playCallingMatchupHtml(DATA, "AAA", "BBB", NAMES), "")
})
