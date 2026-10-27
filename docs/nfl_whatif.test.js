// node --test docs/nfl_whatif.test.js
const test = require("node:test")
const assert = require("node:assert/strict")
global.teamName = require("./teams.js").teamName
const {
  whatifProbability, whatifPanelHtml, whatifEmptyState, whatifPoints, whatifUnitPcts, whatifDefaultPct,
  whatifMissedState, whatifWeekSummary, whatifParsePct, whatifPostedState,
} = require("./nfl_whatif.js")

const text = html => html.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim()
const logit = p => Math.log(p / (1 - p))
const sigmoid = x => 1 / (1 + Math.exp(-x))
const close = (a, b) => assert.ok(Math.abs(a - b) < 1e-9, `${a} vs ${b}`)

// 100 yards a game in every unit, so 1% of a unit = 1 yard = 0.07 points.
const UNITS = {
  run_offense: { ypp: 4, plays: 25 }, pass_offense: { ypp: 5, plays: 20 },
  run_defense: { ypp: 4, plays: 25 }, pass_defense: { ypp: 5, plays: 20 },
}
const GAME = {
  game_id: "2026_08_CAR_GB", home_win_probability: 0.6,
  home: {
    team: "GB", units: UNITS,
    qbs: [
      { id: "love", name: "Jordan Love", qb_plus: 0.12, listed: true },
      { id: "taylor", name: "Tyrod Taylor", qb_plus: -0.08, listed: false },
      { id: "average", name: "Average starter", qb_plus: 0.04, listed: false },
    ],
    regulars: [
      { id: "rb1", name: "Josh Jacobs", group: "RB", share: 0.75, record: { run_offense: { pct: 0.95, games: 3 } } },
      { id: "lt", name: "Left Tackle", group: "OL", share: 1.0, record: {} },
    ],
  },
  away: {
    team: "CAR", units: UNITS,
    qbs: [{ id: "young", name: "Bryce Young", qb_plus: 0.05, listed: true }],
    regulars: [{ id: "cb1", name: "Top Corner", group: "CB", share: 1.0, record: {} }],
  },
  pairing_log_odds: { "love|young": 0.30, "taylor|young": -0.10, "average|young": 0.10 },
}
const DATA = {
  group_units: { RB: ["run_offense"], OL: ["pass_offense", "run_offense"], CB: ["pass_defense"], WR: ["pass_offense"] },
  group_default_pct: { RB: 0.866, OL: 0.939, CB: 0.913, WR: 0.885 },
  average_regular_share: { RB: 0.7, OL: 0.92, CB: 0.8, WR: 0.74 },
  points_per_yard: 0.07, logodds_per_point: 0.137, games: [GAME],
}

test("nothing changed: the posted probability", () => {
  const r = whatifProbability(GAME, DATA, whatifEmptyState())
  close(r.probability, 0.6)
  assert.equal(r.qbShift, 0)
  assert.equal(r.margin, 0)
})

test("a backup QB moves the posted log-odds by the model's change", () => {
  const state = whatifEmptyState()
  state.homeQb = "taylor"
  const r = whatifProbability(GAME, DATA, state)
  close(r.qbShift, -0.4)
  close(r.probability, sigmoid(logit(0.6) - 0.4))
})

test("a player out lowers his units by his position's default times his snap share", () => {
  close(whatifDefaultPct(DATA, "RB", 0.75), 1 - 0.134 * 0.75)
  const state = whatifEmptyState()
  state.out.home.add("rb1")
  const pcts = whatifUnitPcts(GAME, "home", DATA, state)
  close(pcts.run_offense, 0.8995)
  close(pcts.pass_offense, 1)
  const r = whatifProbability(GAME, DATA, state)
  // 10.05% of 100 run yards = 10.05 yards = 0.7035 points off the home margin.
  close(r.margin, -0.7035)
  close(r.probability, sigmoid(logit(0.6) - 0.137 * 0.7035))
})

test("a typed player percent replaces his default - Williams makes the run game 50% as good", () => {
  const state = whatifEmptyState()
  state.out.home.add("rb1")
  state.playerPct.home["rb1|run_offense"] = 0.5
  close(whatifUnitPcts(GAME, "home", DATA, state).run_offense, 0.5)
  close(whatifProbability(GAME, DATA, state).margin, -3.5)
})

test("a typed unit percent overrides the absences", () => {
  const state = whatifEmptyState()
  state.out.home.add("rb1")
  state.unitPct.home.run_offense = 0.95
  close(whatifUnitPcts(GAME, "home", DATA, state).run_offense, 0.95)
  close(whatifProbability(GAME, DATA, state).margin, -0.35)
})

test("a weaker defense helps the other side; absences in a unit multiply", () => {
  const state = whatifEmptyState()
  state.out.away.add("cb1")
  state.out.home.add("lt")
  state.out.home.add("average:OL")
  const away = whatifUnitPcts(GAME, "away", DATA, state)
  close(away.pass_defense, 0.913)
  const home = whatifUnitPcts(GAME, "home", DATA, state)
  close(home.run_offense, 0.939 * (1 - 0.061 * 0.92))
  const r = whatifProbability(GAME, DATA, state)
  const homeNet = (home.run_offense - 1) * 7 + (home.pass_offense - 1) * 7
  close(r.margin, homeNet - (0.913 - 1) * 7)
})

test("typed percents parse, and blank clears", () => {
  assert.equal(whatifParsePct("50"), 0.5)
  assert.equal(whatifParsePct(""), undefined)
  assert.equal(whatifParsePct("abc"), undefined)
})

test("panel shows units, QB menus, players with their units when out, and his record", () => {
  const state = whatifEmptyState()
  state.homeQb = "taylor"
  state.out.home.add("rb1")
  const html = whatifPanelHtml(DATA, GAME.game_id, state)
  const out = text(html)
  assert.match(out, /Game Panthers at Packers/)
  assert.match(out, /Tyrod Taylor · −0.08 EPA\/dropback/)
  assert.match(out, /Average starter · \+0.04 EPA\/dropback/)
  assert.match(html, /data-whatif-unit="home\|run_offense"[^>]*placeholder="90"/)
  assert.match(out, /Run game % from absences · −0.7 pts/)
  assert.match(html, /data-whatif-player="home\|rb1\|run_offense"[^>]*placeholder="90"/)
  assert.match(out, /position average/)
  assert.match(out, /In his 3 games missed: run game 95%/)
  assert.match(out, /Running backs .*An average regular \(not listed\)/)
  assert.match(out, /QB change −10 pts; units −0.7 pts on the margin, −2 pts/)
  assert.match(html, /data-whatif-action="reset"/)
})

// A replayed week: final score, who didn't play, the QBs who started.
const REPLAY = {
  ...GAME,
  final: { home_score: 17, away_score: 24 },
  home: { ...GAME.home, qbs: GAME.home.qbs.map(q => ({ ...q, started: q.id === "love" })),
    regulars: GAME.home.regulars.map(r => ({ ...r, played: r.id !== "lt" })) },
  away: { ...GAME.away, regulars: GAME.away.regulars.map(r => ({ ...r, played: true })) },
}

test("the missed state marks every listed player who took no snaps", () => {
  const state = whatifMissedState(REPLAY)
  assert.deepEqual([...state.out.home], ["lt"])
  assert.deepEqual([...state.out.away], [])
})

test("a replayed week tallies picks as posted and with the real absences", () => {
  const t = whatifWeekSummary({ ...DATA, games: [REPLAY] })
  assert.equal(t.n, 1)
  assert.equal(t.posted.right, 0)
  close(t.posted.winner, 0.4)
  // Left tackle out: both offense units at 93.9% -> -0.061 * 200 yards * 0.07.
  close(t.missed.winner, 1 - sigmoid(logit(0.6) - 0.137 * 0.061 * 200 * 0.07))
})

test("a replayed game shows the final, the starters, who didn't play, the summary and the mark-missed button", () => {
  const modes = [{ key: "this", label: "This week", active: false }, { key: "last", label: "Last week (with results)", active: true }]
  const html = whatifPanelHtml({ ...DATA, games: [REPLAY] }, REPLAY.game_id, whatifEmptyState(), { modes })
  const out = text(html)
  assert.match(html, /data-whatif-mode="last"/)
  assert.match(out, /Final: Panthers 24, Packers 17 · wrong side/)
  assert.match(out, /Jordan Love \(started\)/)
  assert.match(out, /Left Tackle didn't play/)
  assert.match(out, /on the winning side in 0 of 1 games as posted/)
  assert.match(html, /data-whatif-action="missed"/)
})

test("this week's view has no replay pieces; an empty file says so", () => {
  const out = text(whatifPanelHtml(DATA, GAME.game_id, whatifEmptyState()))
  assert.doesNotMatch(out, /Final:|winning side|actually missed/)
  assert.equal(whatifPanelHtml({ games: [] }, null, whatifEmptyState()), "No upcoming games to try yet")
})

test("points read as signed whole percentage points", () => {
  assert.equal(whatifPoints(0.034), "+3 pts")
  assert.equal(whatifPoints(-0.1), "−10 pts")
  assert.equal(whatifPoints(0.001), "no change")
})

test("a game repriced after its injury report opens on that report", () => {
  const game = { ...GAME, injury_report: { out: { home: ["rb1"], away: [] }, qb: { home: "taylor" }, note: "Jordan Love (QB; Tyrod Taylor starts), Josh Jacobs" } }
  const state = whatifPostedState(game)
  assert.deepEqual([...state.out.home], ["rb1"])
  assert.equal(state.homeQb, "taylor")
  assert.deepEqual([...whatifPostedState(GAME).out.home], [])
  const out = text(whatifPanelHtml({ ...DATA, games: [game] }, game.game_id, state))
  assert.match(out, /Posted after the final injury report \(out: Jordan Love \(QB; Tyrod Taylor starts\), Josh Jacobs\)/)
  assert.match(out, /Packers' chance: 60% before the injury report → /)
  assert.match(out, /Without the injury report/)
})
