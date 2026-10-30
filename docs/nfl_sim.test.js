// node --test docs/nfl_sim.test.js
const test = require("node:test")
const assert = require("node:assert/strict")
global.teamName = require("./teams.js").teamName
const {
  simRng, simDraw, simFactors, simulateGame, simAverage, simCalibrate, simSpot, simClock,
  simScoreboardHtml, simDriveChartHtml, simBoxScoreHtml, simTickerHtml, simFieldSvg, simFinalHtml,
  simPick, simShortName, simLineup, simSummary, simMarginBins, simSummaryHtml, simPlayerStatsHtml,
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
  penalties: {
    offense_rate: 0.05, defense_rate: 0.035, dpi_yards: spread(2, 45, 1),
    types: [
      { name: "False Start", side: "O", share: 0.5, yards: 5, first_down: 0, presnap: true, after: false },
      { name: "Offensive Holding", side: "O", share: 0.4, yards: 10, first_down: 0, presnap: false, after: false },
      { name: "Intentional Grounding", side: "O", share: 0.1, yards: 10, first_down: 0, presnap: false, after: false },
      { name: "Defensive Pass Interference", side: "D", share: 0.3, yards: 14, first_down: 1, presnap: false, after: false },
      { name: "Defensive Offside", side: "D", share: 0.4, yards: 5, first_down: 0, presnap: true, after: false },
      { name: "Unnecessary Roughness", side: "D", share: 0.3, yards: 15, first_down: 0.95, presnap: false, after: true },
    ],
  },
}
const player = (gsis, pfr, name, pos, targets, carries, more = {}) =>
  ({ gsis, pfr, name, pos, targets, carries, catch: 1, ypr: 1, ypc: 1, ...more })
const GB_PLAYERS = {
  qb: { gsis: "q1", name: "Jordan Love" }, kicker: { gsis: "k1", name: "Brandon McManus" },
  players: [
    player("w1", "GoldMa02", "Matthew Golden", "WR", 0.4, 0, { catch: 0.9, ypr: 1.3 }),
    player("w2", "WatsCh00", "Christian Watson", "WR", 0.3, 0),
    player("r1", "JacoJo01", "Josh Jacobs", "RB", 0.1, 0.6, { ypc: 1.1 }),
    player("t1", "KrafTu00", "Tucker Kraft", "TE", 0.2, 0.1),
    { role: "QB", gsis: null, pfr: null, name: "QB", pos: "QB", targets: 0, carries: 0.3, catch: 1, ypr: 1, ypc: 1.2 },
  ],
}
const PROFILES = {
  season: 2026, through_week: 7, league: LEAGUE,
  teams: { GB: { neutral_pass_rate: 0.45, ...GB_PLAYERS }, CAR: { neutral_pass_rate: 0.62, sack_rate: 0.09 } },
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

test("short names and picks by share", () => {
  assert.equal(simShortName("Jordan Love"), "J. Love")
  assert.equal(simShortName("Marvin Harrison Jr."), "M. Harrison")
  assert.equal(simShortName("Bijan"), "Bijan")
  const list = [{ key: "a", share: 0.25 }, { key: "b", share: 0.75 }]
  assert.equal(simPick(list, 0.1).key, "a")
  assert.equal(simPick(list, 0.3).key, "b")
  assert.equal(simPick([], 0.3), null)
})

test("a lineup splits targets and carries among who plays, keeping the team's average", () => {
  const full = simLineup(PROFILES, "GB")
  assert.equal(full.qb.name, "Jordan Love")
  assert.equal(full.kicker.short, "B. McManus")
  const sum = (xs, f) => xs.reduce((a, x) => a + f(x), 0)
  assert.ok(Math.abs(sum(full.receivers, r => r.share) - 1) < 1e-9)
  assert.ok(Math.abs(sum(full.rushers, r => r.share) - 1) < 1e-9)
  // Share-weighted, the catch and gain multipliers average 1.
  assert.ok(Math.abs(sum(full.receivers, r => r.share * r.catch) - 1) < 1e-9)
  assert.ok(Math.abs(sum(full.rushers, r => r.share * r.ypc) - 1) < 1e-9)
  // Jacobs out (by his What if? id): his carries go to the rest, the QB's
  // runs go to whoever starts.
  const out = simLineup(PROFILES, "GB", new Set(["JacoJo01"]), { id: "q2", name: "Malik Willis" })
  assert.ok(!out.rushers.some(r => r.name === "Josh Jacobs") && !out.receivers.some(r => r.name === "Josh Jacobs"))
  const qbRuns = out.rushers.find(r => r.pos === "QB")
  assert.equal(qbRuns.name, "Malik Willis")
  assert.ok(Math.abs(qbRuns.share - 0.75) < 1e-9)
  assert.equal(out.qb.short, "M. Willis")
  // A team without player profiles still plays, credited to the team.
  const bare = simLineup(PROFILES, "CAR")
  assert.equal(bare.receivers.length, 0)
  assert.equal(bare.qb.name, "CAR QB")
})

test("players' stat lines add up to the team box score", () => {
  const lineups = { home: simLineup(PROFILES, "GB", new Set(["JacoJo01"])), away: simLineup(PROFILES, "CAR") }
  for (let seed = 1; seed <= 150; seed++) {
    const g = simulateGame(PROFILES, "GB", "CAR", { seed, lineups })
    const ps = g.players.home
    const total = k => ps.reduce((a, p) => a + p[k], 0)
    const b = g.box.home
    assert.equal(total("att"), b.pass_att)
    assert.equal(total("cmp"), b.pass_comp)
    assert.equal(total("pass_yds"), b.pass_yards)
    assert.equal(total("rec_yds"), b.pass_yards)
    assert.equal(total("tgt"), b.pass_att)
    assert.equal(total("rush_yds"), b.rush_yards)
    assert.equal(total("rush_att"), b.rush_att)
    assert.equal(total("pass_td"), b.pass_td)
    assert.equal(total("rec_td") + total("rush_td"), b.pass_td + b.rush_td)
    assert.equal(total("ints"), b.ints)
    assert.equal(total("sacks"), b.sacks)
    assert.equal(total("fga"), b.fg_att)
    assert.ok(!ps.some(p => p.name === "Josh Jacobs"))
  }
})

test("a thousand games: wins add up, players average sensible lines", () => {
  const lineups = { home: simLineup(PROFILES, "GB"), away: simLineup(PROFILES, "CAR") }
  const sum = simSummary(PROFILES, "GB", "CAR", { seed: 3, lineups }, 400)
  assert.equal(sum.homeWins + sum.awayWins + sum.ties, 400)
  assert.equal(sum.margins.length, 400)
  assert.ok(sum.marginRange[0] <= sum.marginRange[1] && sum.marginRange[1] <= sum.marginRange[2])
  const love = sum.players.home.find(p => p.role === "QB")
  assert.ok(love.att > 20 && love.att < 50, `${love.att}`)
  const golden = sum.players.home.find(p => p.name === "Matthew Golden")
  const watson = sum.players.home.find(p => p.name === "Christian Watson")
  assert.ok(golden.tgt > watson.tgt)
  assert.ok(golden.td_games > 0 && golden.td_games < 1)
  const bins = simMarginBins(sum.margins)
  assert.ok(Math.abs(bins.reduce((a, b) => a + b.share, 0) - 1) < 1e-9)
  assert.equal(bins.find(b => b.bin === 0).side, "tie")
  const html = text(simSummaryHtml(sum, { margin: 3, total: 44, probability: 0.6 }))
  assert.match(html, /^(Packers|Panthers) win [\d,]+ of 400/)
  assert.match(html, /Matthew Golden WR/)
  assert.match(html, /score a TD/)
})

test("one game's player tables list who played", () => {
  const lineups = { home: simLineup(PROFILES, "GB"), away: simLineup(PROFILES, "CAR") }
  const g = simulateGame(PROFILES, "GB", "CAR", { seed: 2, lineups })
  const html = text(simPlayerStatsHtml(g))
  assert.match(html, /Panthers Passing C\/Att/)
  assert.match(html, /Packers Passing C\/Att Yds TD INT Sk Jordan Love/)
  assert.ok(g.plays.some(p => / pass (to|incomplete for) [A-Z]\. /.test(p.text)))
})

test("penalties: flags at about the set rates, walked off within the field, in the box score", () => {
  let flagsSeen = 0, snaps = 0
  for (let seed = 1; seed <= 300; seed++) {
    const g = simulateGame(PROFILES, "GB", "CAR", { seed })
    const wiped = g.plays.filter(p => p.kind === "penalty")
    const added = g.plays.filter(p => p.kind !== "penalty" && /PENALTY/.test(p.text))
    assert.equal(g.box.home.penalties + g.box.away.penalties, wiped.length + added.length)
    for (const p of wiped) {
      assert.ok(p.end >= 1 && p.end <= 99, `${p.ball} -> ${p.end}`)
      // Half the distance to the goal at most (DPI goes to the 1).
      if (/False Start|Holding/.test(p.text)) { assert.ok(p.ball - p.end <= Math.max(1, p.ball / 2) + 1e-9) }
    }
    flagsSeen += wiped.length + added.length
    snaps += g.plays.filter(p => SCRIMMAGE.has(p.kind) || p.kind === "penalty").length
  }
  const rate = flagsSeen / snaps
  assert.ok(rate > 0.06 && rate < 0.11, `${rate}`)
})

test("timeouts: three a half (two in overtime), never below zero; one two-minute warning a half", () => {
  for (let seed = 1; seed <= 300; seed++) {
    const g = simulateGame(PROFILES, "GB", "CAR", { seed })
    for (const p of g.plays) {
      for (const side of ["home", "away"]) {
        assert.ok(p.timeouts[side] >= 0 && p.timeouts[side] <= (p.qtr >= 5 ? 2 : 3))
      }
    }
    const used = g.box.home.timeouts + g.box.away.timeouts
    assert.equal(used, g.plays.filter(p => p.kind === "timeout").length)
    assert.ok(used <= 12 + (g.final.overtime ? 4 : 0))
    for (const q of [2, 4]) {
      assert.ok(g.plays.filter(p => p.kind === "warning" && p.qtr === q).length <= 1)
    }
  }
})

test("late and close, the defense spends its timeouts; the leader kneels only when it can run out the clock", () => {
  let defenseLate = 0
  for (let seed = 1; seed <= 400; seed++) {
    const g = simulateGame(PROFILES, "GB", "CAR", { seed })
    for (const p of g.plays) {
      if (p.kind === "timeout" && p.qtr === 4 && p.clock <= 180) {
        const caller = p.text.includes("GB") ? "home" : "away"
        const callerMargin = caller === "home" ? p.home - p.away : p.away - p.home
        if (caller !== p.poss) { defenseLate += 1; assert.ok(callerMargin < 0) }
      }
    }
    // A kneel-down never hands the ball back with the leader's lead in danger
    // of more than one possession of clock.
    const kneels = g.plays.filter(p => p.kind === "kneel")
    for (const k of kneels) { assert.ok(k.qtr >= 4) }
  }
  assert.ok(defenseLate > 0)
})
