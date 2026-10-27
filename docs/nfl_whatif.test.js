// node --test docs/nfl_whatif.test.js
const test = require("node:test")
const assert = require("node:assert/strict")
global.teamName = require("./teams.js").teamName
const { whatifLost, whatifProbability, whatifPanelHtml, whatifEmptyState, whatifPoints, whatifMissedState, whatifWeekSummary } = require("./nfl_whatif.js")

const text = html => html.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim()
const logit = p => Math.log(p / (1 - p))
const sigmoid = x => 1 / (1 + Math.exp(-x))
const close = (a, b) => assert.ok(Math.abs(a - b) < 1e-9, `${a} vs ${b}`)

const GAME = {
  game_id: "2026_08_CAR_GB", home_win_probability: 0.6,
  home: {
    team: "GB",
    qbs: [
      { id: "love", name: "Jordan Love", qb_plus: 0.12, listed: true },
      { id: "taylor", name: "Tyrod Taylor", qb_plus: -0.08, listed: false },
      { id: "average", name: "Average starter", qb_plus: 0.04, listed: false },
    ],
    regulars: [
      { id: "lt", name: "Left Tackle", group: "OL", share: 1.0 },
      { id: "rg", name: "Right Guard", group: "OL", share: 1.0 },
      { id: "wr1", name: "Top Receiver", group: "WR", share: 0.8 },
    ],
  },
  away: {
    team: "CAR",
    qbs: [{ id: "young", name: "Bryce Young", qb_plus: 0.05, listed: true }],
    regulars: [{ id: "cb1", name: "Top Corner", group: "CB", share: 0.8 }, { id: "cb2", name: "Second Corner", group: "CB", share: 0.6 }],
  },
  pairing_log_odds: { "love|young": 0.30, "taylor|young": -0.10, "average|young": 0.10 },
}
const ODDS = { WR: -0.25, OL: -0.19, CB: -0.19, RB: -0.13 }
const AVERAGES = { WR: 0.75, OL: 0.9, CB: 0.8, RB: 0.7 }

test("nothing changed: the posted probability", () => {
  const r = whatifProbability(GAME, ODDS, whatifEmptyState(), AVERAGES)
  close(r.probability, 0.6)
  assert.equal(r.qbShift, 0)
  assert.equal(r.injuryShift, 0)
})

test("a backup QB moves the posted log-odds by the model's change", () => {
  const state = whatifEmptyState()
  state.homeQb = "taylor"
  const r = whatifProbability(GAME, ODDS, state, AVERAGES)
  close(r.qbShift, -0.4)
  close(r.probability, sigmoid(logit(0.6) - 0.4))
})

test("an average starter is priced like any other QB", () => {
  const state = whatifEmptyState()
  state.homeQb = "average"
  close(whatifProbability(GAME, ODDS, state, AVERAGES).qbShift, -0.2)
})

test("each player counts his snap share; an average regular the league's", () => {
  const lost = whatifLost(GAME.home, new Set(["lt", "wr1", "average:RB"]), AVERAGES)
  assert.deepEqual(lost, { OL: 1.0, WR: 0.8, RB: 0.7 })
})

test("home absences lower and away absences raise the home chance, player by player", () => {
  const state = whatifEmptyState()
  state.out.home = new Set(["lt", "rg"])
  state.out.away = new Set(["cb1", "cb2"])
  const r = whatifProbability(GAME, ODDS, state, AVERAGES)
  // OL: -0.19 x 2 regulars at home; CB: -0.19 x 1.4 for the away corners.
  close(r.injuryShift, -0.19 * 2 + 0.19 * 1.4)
  close(r.probability, sigmoid(logit(0.6) - 0.38 + 0.266))
})

test("panel shows the QB menus, every position with an average option, and each player's worth", () => {
  const data = { injury_log_odds: ODDS, average_regular_share: AVERAGES, games: [GAME] }
  const state = whatifEmptyState()
  state.homeQb = "taylor"
  const out = text(whatifPanelHtml(data, GAME.game_id, state))
  assert.match(out, /Game Panthers at Packers/)
  assert.match(out, /Packers' chance: 60% as posted → 50%/)
  assert.match(out, /Tyrod Taylor · −0.08 EPA\/dropback/)
  assert.match(out, /Average starter · \+0.04 EPA\/dropback/)
  assert.match(out, /Receivers Top Receiver 80% of snaps · ~5 pts An average regular \(not listed\) ~5 pts/)
  assert.match(out, /Offensive line Left Tackle 100% of snaps · ~5 pts/)
  // Carolina lists no running back, but the average option is still there.
  assert.match(out, /Panthers .*Running backs An average regular \(not listed\) ~2 pts/)
  assert.match(out, /\(QB change −10 pts\)/)
})

test("an empty file says so", () => {
  assert.equal(whatifPanelHtml({ games: [] }, null, whatifEmptyState()), "No upcoming games to try yet")
})

test("points read as signed whole percentage points", () => {
  assert.equal(whatifPoints(0.034), "+3 pts")
  assert.equal(whatifPoints(-0.1), "−10 pts")
  assert.equal(whatifPoints(0.001), "no change")
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
  // Home at 60% lost: wrong as posted; Left Tackle out (-0.19) leaves home at 55%, still wrong.
  const t = whatifWeekSummary({ injury_log_odds: ODDS, average_regular_share: AVERAGES, games: [REPLAY] })
  assert.equal(t.n, 1)
  assert.equal(t.posted.right, 0)
  close(t.posted.winner, 0.4)
  close(t.missed.winner, 1 - sigmoid(logit(0.6) - 0.19))
})

test("a replayed game shows the final, the starters, who didn't play, the summary and the mark-missed button", () => {
  const data = { injury_log_odds: ODDS, average_regular_share: AVERAGES, games: [REPLAY] }
  const modes = [{ key: "this", label: "This week", active: false }, { key: "last", label: "Last week (with results)", active: true }]
  const html = whatifPanelHtml(data, REPLAY.game_id, whatifEmptyState(), { modes })
  const out = text(html)
  assert.match(html, /data-whatif-mode="last"/)
  assert.match(out, /This week Last week \(with results\)/)
  assert.match(out, /Final: Panthers 24, Packers 17 · wrong side/)
  assert.match(out, /Jordan Love \(started\)/)
  assert.match(out, /Left Tackle didn't play/)
  assert.match(out, /on the winning side in 0 of 1 games as posted/)
  assert.match(html, /data-whatif-action="missed"/)
})

test("this week's view has no replay pieces", () => {
  const out = text(whatifPanelHtml({ injury_log_odds: ODDS, games: [GAME] }, GAME.game_id, whatifEmptyState()))
  assert.doesNotMatch(out, /Final:|winning side|actually missed/)
})
