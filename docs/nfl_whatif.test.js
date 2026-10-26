// node --test docs/nfl_whatif.test.js
const test = require("node:test")
const assert = require("node:assert/strict")
global.teamName = require("./teams.js").teamName
const { whatifGroupShares, whatifProbability, whatifPanelHtml, whatifEmptyState, whatifPoints } = require("./nfl_whatif.js")

const text = html => html.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim()
const logit = p => Math.log(p / (1 - p))
const sigmoid = x => 1 / (1 + Math.exp(-x))

const GAME = {
  game_id: "2026_08_CAR_GB", home_win_probability: 0.6,
  home: {
    team: "GB",
    qbs: [{ id: "love", name: "Jordan Love", qb_plus: 0.12, listed: true }, { id: "taylor", name: "Tyrod Taylor", qb_plus: -0.08, listed: false }],
    regulars: [
      { id: "lt", name: "Left Tackle", group: "OL", share: 1.0 },
      { id: "rg", name: "Right Guard", group: "OL", share: 1.0 },
      { id: "wr1", name: "Top Receiver", group: "WR", share: 0.9 },
    ],
  },
  away: {
    team: "CAR",
    qbs: [{ id: "young", name: "Bryce Young", qb_plus: 0.05, listed: true }],
    regulars: [{ id: "cb1", name: "Top Corner", group: "CB", share: 0.8 }, { id: "cb2", name: "Second Corner", group: "CB", share: 0.8 }],
  },
  pairing_log_odds: { "love|young": 0.30, "taylor|young": -0.10 },
}
const ODDS = { OL: -0.85, CB: -0.45, WR: -0.39 }

test("nothing changed: the posted probability", () => {
  const r = whatifProbability(GAME, ODDS, whatifEmptyState())
  assert.ok(Math.abs(r.probability - 0.6) < 1e-9)
  assert.equal(r.qbShift, 0)
  assert.equal(r.injuryShift, 0)
})

test("a backup QB moves the posted log-odds by the model's change", () => {
  const state = whatifEmptyState()
  state.homeQb = "taylor"
  const r = whatifProbability(GAME, ODDS, state)
  assert.ok(Math.abs(r.qbShift - -0.4) < 1e-9)
  assert.ok(Math.abs(r.probability - sigmoid(logit(0.6) - 0.4)) < 1e-9)
})

test("group shares weight absences by snap share", () => {
  assert.deepEqual(whatifGroupShares(GAME.home, new Set(["lt"])), { OL: 0.5, WR: 0 })
})

test("home absences lower and away absences raise the home chance", () => {
  const state = whatifEmptyState()
  state.out.home = new Set(["lt"])
  state.out.away = new Set(["cb1", "cb2"])
  const r = whatifProbability(GAME, ODDS, state)
  // OL: -0.85 * 0.5 at home; CB: -0.45 * (0 - 1) for the away team's corners.
  assert.ok(Math.abs(r.injuryShift - (-0.425 + 0.45)) < 1e-9)
  assert.ok(Math.abs(r.probability - sigmoid(logit(0.6) + 0.025)) < 1e-9)
})

test("panel shows the game picker, both QB menus and the regulars", () => {
  const data = { injury_log_odds: ODDS, games: [GAME] }
  const state = whatifEmptyState()
  state.homeQb = "taylor"
  const out = text(whatifPanelHtml(data, GAME.game_id, state))
  assert.match(out, /Game Panthers at Packers/)
  assert.match(out, /Packers' chance: 60% as posted → 50%/)
  assert.match(out, /Tyrod Taylor · −0.08 EPA\/dropback/)
  assert.match(out, /Jordan Love \(listed\)/)
  assert.match(out, /Offensive line Left Tackle 100% of snaps/)
  assert.match(out, /as posted → /)
  assert.match(out, /QB change −10 pts/)
})

test("an empty file says so", () => {
  assert.equal(whatifPanelHtml({ games: [] }, null, whatifEmptyState()), "No upcoming games to try yet")
})

test("points read as signed whole percentage points", () => {
  assert.equal(whatifPoints(0.034), "+3 pts")
  assert.equal(whatifPoints(-0.1), "−10 pts")
  assert.equal(whatifPoints(0.001), "no change")
})
