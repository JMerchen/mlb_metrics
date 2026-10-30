// node --test docs/nfl_sim.test.js
const test = require("node:test")
const assert = require("node:assert/strict")
global.teamName = require("./teams.js").teamName
const {
  simRng, simDraw, simFactors, simulateGame, simAverage, simCalibrate, simSpot, simClock,
  simScoreboardHtml, simDriveChartHtml, simBoxScoreHtml, simTickerHtml, simFieldSvg, simFinalHtml,
} = require("./nfl_sim.js")

const text = html => html.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim()

// A league in round numbers, and two teams with their own styles.
const PROBS = [0.003, ...Array.from({ length: 39 }, (_, i) => 0.025 + i * 0.025), 0.997]
// Gains bunched short with a long tail (mean lo + (hi - lo) / 3).
const spread = (lo, hi, power = 2) => PROBS.map((p, i) => lo + (hi - lo) * (i / (PROBS.length - 1)) ** power)
const LEAGUE = {
  neutral_pass_rate: 0.52, sack_rate: 0.065, comp_rate: 0.645, int_rate: 0.022,
  run_fumble_rate: 0.006, catch_fumble_rate: 0.005, sack_yards: -6.5,
  rush_yards: spread(-3, 20), completion_yards: spread(-1, 32),
  kickoff_start: spread(15, 45, 1), punt_net: spread(25, 60, 1), punt_touchback_rate: 0.07,
  field_goal: { a: 6.1, b: -0.098 }, xp_rate: 0.95, scrimmage_plays_per_team_game: 61, quantile_probs: PROBS,
}
const PROFILES = {
  season: 2026, through_week: 7, league: LEAGUE,
  teams: { GB: { neutral_pass_rate: 0.45 }, CAR: { neutral_pass_rate: 0.62, sack_rate: 0.09 } },
}
const SCRIMMAGE = new Set(["run", "pass", "incomplete", "sack", "interception", "kneel"])

test("the random numbers repeat by seed and stay in [0, 1)", () => {
  const a = simRng(42), b = simRng(42), c = simRng(43)
  const xs = Array.from({ length: 1000 }, () => a())
  assert.deepEqual(xs.slice(0, 5), Array.from({ length: 5 }, () => b()))
  assert.notEqual(xs[0], c())
  assert.ok(xs.every(x => x >= 0 && x < 1))
  const mean = xs.reduce((s, x) => s + x, 0) / xs.length
  assert.ok(Math.abs(mean - 0.5) < 0.05)
})

test("a draw interpolates between quantiles inside the outer ones", () => {
  const q = PROBS.map((p, i) => i)
  assert.equal(simDraw(q, PROBS, 0), 0)
  assert.equal(simDraw(q, PROBS, 1), PROBS.length - 1)
  const mid = simDraw(q, PROBS, 0.5)
  assert.ok(mid > 19 && mid < 21, `${mid}`)
})

test("unit percents multiply an offense's yards, and a weak defense gives up more", () => {
  assert.deepEqual(simFactors(1, {}, {}), { run: 1, pass: 1 })
  const f = simFactors(1.1, { run_offense: 0.9 }, { pass_defense: 0.8 })
  assert.ok(Math.abs(f.run - 0.99) < 1e-12 && Math.abs(f.pass - 1.32) < 1e-12)
})

test("spots and clocks read like a broadcast", () => {
  assert.equal(simSpot(35, "KC", "BUF"), "KC 35")
  assert.equal(simSpot(50, "KC", "BUF"), "50")
  assert.equal(simSpot(80, "KC", "BUF"), "BUF 20")
  assert.equal(simClock(65), "1:05")
  assert.equal(simClock(-3), "0:00")
})

test("a game is the same from the same seed, and its books balance", () => {
  const g = simulateGame(PROFILES, "GB", "CAR", { seed: 9 })
  assert.deepEqual(simulateGame(PROFILES, "GB", "CAR", { seed: 9 }).final, g.final)
  for (let seed = 1; seed <= 300; seed++) {
    const game = simulateGame(PROFILES, "GB", "CAR", { seed })
    const last = game.plays[game.plays.length - 1]
    assert.deepEqual([last.home, last.away], [game.final.home, game.final.away])
    for (const side of ["home", "away"]) {
      const b = game.box[side]
      assert.equal(b.points, game.final[side])
      assert.equal(b.yards, b.pass_yards - b.sack_yards + b.rush_yards)
      assert.ok(b.pass_comp <= b.pass_att && b.third_conv <= b.third_att && b.fg_made <= b.fg_att)
      const scrimmage = game.plays.filter(p => p.poss === side && SCRIMMAGE.has(p.kind)).length
      assert.equal(b.plays, scrimmage)
    }
    // Possession adds up to the 60 minutes (plus any overtime played).
    const used = game.box.home.possession + game.box.away.possession
    assert.ok(game.final.overtime ? used > 3600 && used <= 4200 : used === 3600, `${used}`)
    game.plays.forEach((p, i) => {
      assert.ok(p.ball >= 0 && p.ball <= 100 && p.end >= 0 && p.end <= 100)
      const next = game.plays[i + 1]
      // The next snap starts where the last one ended, on the same drive.
      if (next && next.drive === p.drive && next.kind !== "kickoff") { assert.equal(next.ball, p.end) }
      if (next && next.qtr === p.qtr) { assert.ok(next.clock <= p.clock) }
    })
    for (const d of game.drives) { assert.ok(d.result, "every drive ends") }
    // Overtime only after a tie, and a game ends tied only after it.
    if (game.final.home === game.final.away) { assert.ok(game.final.overtime) }
  }
})

test("left alone, a game looks like an NFL game", () => {
  const n = 600
  const sum = { points: 0, plays: 0, yards: 0, turnovers: 0, punts: 0 }
  for (let seed = 1; seed <= n; seed++) {
    const g = simulateGame(PROFILES, "GB", "CAR", { seed })
    for (const side of ["home", "away"]) {
      for (const k of Object.keys(sum)) { sum[k] += g.box[side][k] / (2 * n) }
    }
  }
  assert.ok(sum.points > 14 && sum.points < 30, `points ${sum.points}`)
  assert.ok(sum.plays > 52 && sum.plays < 72, `plays ${sum.plays}`)
  assert.ok(sum.turnovers > 0.5 && sum.turnovers < 2, `turnovers ${sum.turnovers}`)
  assert.ok(sum.punts > 2 && sum.punts < 7, `punts ${sum.punts}`)
})

test("a pass-happy team throws more, and a bad pass defense gives up more through the air", () => {
  let gbAtt = 0, carAtt = 0, base = 0, leaky = 0
  for (let seed = 1; seed <= 200; seed++) {
    const g = simulateGame(PROFILES, "GB", "CAR", { seed })
    gbAtt += g.box.home.pass_att
    carAtt += g.box.away.pass_att
    base += g.box.home.pass_yards
    leaky += simulateGame(PROFILES, "GB", "CAR", { seed, unitPcts: { away: { pass_defense: 0.6 } } }).box.home.pass_yards
  }
  assert.ok(carAtt > gbAtt)
  assert.ok(leaky > base * 1.15, `${leaky} vs ${base}`)
})

test("calibration lands the average game on the target margin and total", () => {
  for (const [margin, total] of [[6, 44], [-3, 50]]) {
    const c = simCalibrate(PROFILES, "GB", "CAR", margin, total, { games: 200 })
    assert.ok(Math.abs(c.margin - margin) < 1, `margin ${c.margin} vs ${margin}`)
    assert.ok(Math.abs(c.total - total) < 1.5, `total ${c.total} vs ${total}`)
    assert.equal(Math.sign(c.edge), Math.sign(margin))
    // The same edge and scale on fresh seeds stay close.
    const fresh = simAverage(PROFILES, "GB", "CAR", { edge: c.edge, scale: c.scale }, 200)
    assert.ok(Math.abs(fresh.margin - margin) < 2.5, `fresh margin ${fresh.margin}`)
  }
})

test("calibration with unit settings still lands on the target", () => {
  const unitPcts = { home: { pass_offense: 0.7 }, away: {} }
  const c = simCalibrate(PROFILES, "GB", "CAR", 3, 44, { games: 200, unitPcts })
  assert.ok(Math.abs(c.margin - 3) < 1, `${c.margin}`)
})

test("the drive chart shows finished drives in full and the one under way so far", () => {
  const g = simulateGame(PROFILES, "GB", "CAR", { seed: 5 })
  const i = g.plays.findIndex((p, j) => j > 0 && p.drive === 1 && SCRIMMAGE.has(p.kind))
  const partial = simDriveChartHtml(g, i)
  assert.equal((partial.match(/<tr/g) || []).length, 1 + 2)
  assert.match(partial, /simLiveDrive/)
  assert.match(text(partial), /…/)
  const full = simDriveChartHtml(g)
  assert.equal((full.match(/<tr/g) || []).length, 1 + g.drives.length)
  assert.doesNotMatch(full, /simLiveDrive/)
})

test("the scoreboard shows what's next, then Final; the box and final read right", () => {
  const g = simulateGame(PROFILES, "GB", "CAR", { seed: 11 })
  const k = g.plays.findIndex(p => p.kind === "run" || p.kind === "pass")
  const board = text(simScoreboardHtml(g, g.plays[k], g.plays[k + 1]))
  assert.match(board, /Panthers .* Packers/)
  const last = g.plays[g.plays.length - 1]
  assert.match(text(simScoreboardHtml(g, last)), /Final/)
  const box = text(simBoxScoreHtml(g))
  assert.match(box, /^CAR GB First downs/)
  assert.match(box, /Time of possession \d+:\d\d \d+:\d\d/)
  const f = g.final
  const headline = text(simFinalHtml(g))
  assert.match(headline, f.home === f.away ? /Tie/ : f.home > f.away ? /Packers win/ : /Panthers win/)
  assert.equal((simTickerHtml(g, 20, 5).match(/<li/g) || []).length, 5)
  assert.match(simFieldSvg(g.plays[k], "GB", "CAR"), /<svg[^>]*aria-label="Ball at /)
})
