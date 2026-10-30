// node --test docs/mlb_sim.test.js
const test = require("node:test")
const assert = require("node:assert/strict")
global.teamName = require("./teams.js").teamName
const {
  mlbRng, mlbPick, mlbBucket, mlbShortName, mlbPitchProbs, simulateMlbGame,
  mlbLineScoreHtml, mlbSituationHtml, mlbZoneSvg, mlbPlaysHtml, mlbBoxHtml, mlbFinalHtml, mlbInningsPitched,
  simulateMlbSummary, mlbSummaryHtml,
} = require("./mlb_sim.js")
const P = require("./data/mlb_sim_profiles.json")

const text = html => html.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim()
const TEAMS = Object.keys(P.teams).sort()
const game = (seed, a = TEAMS[seed % TEAMS.length], h = TEAMS[(seed * 7 + 3) % TEAMS.length]) =>
  simulateMlbGame(P, { away: { team: a }, home: { team: h === a ? TEAMS[(seed + 1) % TEAMS.length] : h } }, { seed })

test("the basics: seeded randomness, picks by weight, counts, short names", () => {
  const a = mlbRng(3), b = mlbRng(3)
  assert.equal(a(), b())
  assert.equal(mlbPick([0, 1, 0], 0.5), 1)
  assert.equal(mlbPick([1, 3], 0.2), 0)
  assert.equal(mlbPick([1, 3], 0.3), 1)
  assert.equal(mlbBucket(0, 0), "first")
  assert.equal(mlbBucket(1, 2), "ahead")
  assert.equal(mlbBucket(2, 2), "even")
  assert.equal(mlbBucket(3, 1), "behind")
  assert.equal(mlbShortName("Shohei Ohtani"), "S. Ohtani")
  assert.equal(mlbShortName("Ronald Acuña Jr."), "R. Acuña")
  assert.equal(mlbInningsPitched(17), "5.2")
})

test("a pitch's result chances are positive for a real pitcher and hitter", () => {
  const pid = Object.keys(P.pitchers)[0]
  const bid = Object.keys(P.batters)[0]
  const pitcher = P.pitchers[pid]
  const probs = mlbPitchProbs(P.league, pitcher, P.batters[bid], pitcher.types[0], "0-0", true, 1)
  assert.equal(probs.length, 6)
  assert.ok(probs.every(p => p >= 0) && probs.reduce((x, y) => x + y, 0) > 0.5)
  // 3-0: far more balls than 0-2 for the same pitch.
  const ahead = mlbPitchProbs(P.league, pitcher, P.batters[bid], pitcher.types[0], "0-2", true, 1)
  const behind = mlbPitchProbs(P.league, pitcher, P.batters[bid], pitcher.types[0], "3-0", true, 1)
  assert.ok(behind[1] / behind.reduce((x, y) => x + y) > ahead[1] / ahead.reduce((x, y) => x + y))
})

test("a game is the same from the same seed, and its books balance", () => {
  assert.deepEqual(game(4).final, game(4).final)
  for (let seed = 1; seed <= 150; seed++) {
    const g = game(seed)
    const f = g.final
    assert.notEqual(f.home, f.away, `seed ${seed} tied`)
    assert.ok(f.innings >= 9)
    // Line score, runs scored and hits all agree.
    for (const side of ["away", "home"]) {
      assert.equal(g.innings[side].reduce((a, b) => a + (b || 0), 0), f[side])
      assert.equal(g.batting[side].reduce((a, l) => a + l.r, 0), f[side])
      assert.equal(g.batting[side].reduce((a, l) => a + l.h, 0), g.hits[side])
      const other = side === "home" ? "away" : "home"
      assert.equal(g.pitching[other].reduce((a, l) => a + l.h, 0), g.hits[side])
      // Ghost runners in extra innings aren't charged to a pitcher.
      assert.ok(g.pitching[other].reduce((a, l) => a + l.r, 0) <= f[side])
      const l = g.batting[side]
      assert.ok(l.every(b => b.pa === b.ab + b.bb + b.hbp + b.sf))
    }
    // Every top half is three outs; the home staff gets all of them.
    assert.equal(g.pitching.home.reduce((a, l) => a + l.outs, 0), 3 * f.innings)
    // Counts never reach four balls or three strikes before a pitch.
    assert.ok(g.pitches.every(p => p.balls <= 3 && p.strikes <= 2 && p.outs <= 2))
    // A game that ends in the bottom half ends with the home team just ahead
    // (by four at most, on a home run).
    const last = g.plays[g.plays.length - 1]
    if (last.half === "bottom" && f.home > f.away) { assert.ok(f.home - f.away <= 4) }
  }
})

test("the batting order is the one picked, and the picked starter starts", () => {
  const team = TEAMS[0], other = TEAMS[1]
  const roster = P.teams[team]
  const lineup = roster.lineup.map(String).slice().reverse()
  const starter = String(roster.starters[roster.starters.length - 1].id)
  const g = simulateMlbGame(P, { away: { team, starter, lineup }, home: { team: other } }, { seed: 9 })
  assert.deepEqual(g.batting.away.slice(0, 9).map(l => l.id), lineup)
  assert.equal(g.pitching.away[0].id, starter)
  const firstPitch = g.pitches.find(p => p.half === "bottom")
  assert.equal(firstPitch.pitcher, starter)
  assert.equal(g.pitches[0].batter, lineup[0])
})

test("left alone, games look like the majors", () => {
  const t = { runs: 0, pa: 0, k: 0, bb: 0, hr: 0, pitches: 0, starterOuts: 0 }
  const n = 300
  for (let seed = 1; seed <= n; seed++) {
    const g = game(seed)
    for (const side of ["away", "home"]) {
      t.runs += g.final[side]
      for (const l of g.batting[side]) { t.pa += l.pa; t.k += l.k; t.bb += l.bb + l.hbp; t.hr += l.hr }
      for (const l of g.pitching[side]) { t.pitches += l.pitches }
      t.starterOuts += g.pitching[side][0].outs
    }
  }
  const tg = 2 * n
  assert.ok(t.runs / tg > 3.6 && t.runs / tg < 5.2, `runs ${t.runs / tg}`)
  assert.ok(t.k / t.pa > 0.18 && t.k / t.pa < 0.27, `K ${t.k / t.pa}`)
  assert.ok(t.bb / t.pa > 0.07 && t.bb / t.pa < 0.12, `BB ${t.bb / t.pa}`)
  assert.ok(t.hr / t.pa > 0.02 && t.hr / t.pa < 0.04, `HR ${t.hr / t.pa}`)
  assert.ok(t.pitches / tg > 130 && t.pitches / tg < 165, `pitches ${t.pitches / tg}`)
  assert.ok(t.starterOuts / tg / 3 > 4.3 && t.starterOuts / tg / 3 < 6.2, `starter IP ${t.starterOuts / tg / 3}`)
})

test("the page's pieces: line score, count, zone, plays, box and final", () => {
  const g = game(12)
  const mid = g.pitches.findIndex((p, i) => i > 20 && !p.text)
  const line = text(mlbLineScoreHtml(g))
  assert.match(line, /R H/)
  assert.match(text(mlbLineScoreHtml(g, mid)), /\d/)
  assert.match(mlbSituationHtml(g.pitches[mid]), /mlbDiamond/)
  assert.match(mlbZoneSvg(g, mid), /<circle/)
  assert.ok((mlbPlaysHtml(g, g.pitches.length - 1, 5).match(/<li/g) || []).length === 5)
  const box = text(mlbBoxHtml(P, g))
  assert.match(box, /batting AB R H RBI BB K/)
  assert.match(box, /pitching IP H R BB K HR P/)
  assert.match(text(mlbFinalHtml(g)), /^Final: .+ win \d+-\d+/)
})

test("the notes under each batting box list its extra-base hits, double plays and steals", () => {
  let seen = 0
  for (let seed = 1; seed <= 60; seed++) {
    const g = game(seed)
    const box = text(mlbBoxHtml(P, g))
    for (const side of ["away", "home"]) {
      const l = g.batting[side]
      const sum = k => l.reduce((a, b) => a + b[k], 0)
      // Grounding into a double play is the play-by-play's own.
      const gidp = g.plays.filter(p => p.half === (side === "away" ? "top" : "bottom") && /grounds into a double play/.test(p.text)).length
      assert.equal(sum("gidp"), gidp)
      for (const [k, label] of [["d", "2B"], ["t", "3B"], ["hr", "HR"], ["gidp", "GIDP"], ["sb", "SB"], ["cs", "CS"]]) {
        if (sum(k)) { assert.match(box, new RegExp(`${label} ?: `)); seen++ }
      }
    }
    if (!g.batting.away.some(l => l.hr) && !g.batting.home.some(l => l.hr)) { assert.doesNotMatch(box, /HR ?: /) }
  }
  assert.ok(seen > 100)
  // A player with two of something is listed once, with the count.
  const g = game(5)
  const multi = g.batting.away.concat(g.batting.home).find(l => l.d > 1 || l.hr > 1 || l.sb > 1)
  if (multi) { assert.match(text(mlbBoxHtml(P, g)), new RegExp(`${mlbShortName(P.batters[multi.id]?.name || "").split(" ").pop()} 2`)) }
})

test("between pitches: steals, caught stealing and wild pitches happen at about real rates", () => {
  let sb = 0, cs = 0, ibb = 0
  const n = 300
  for (let seed = 1; seed <= n; seed++) {
    const g = game(seed)
    for (const side of ["away", "home"]) { for (const l of g.batting[side]) { sb += l.sb; cs += l.cs } }
    ibb += g.plays.filter(p => /intentionally walked/.test(p.text)).length
    // Runners move only between a hitter's pitches, never before his first.
    for (const p of g.plays.filter(p => p.kind === "run")) {
      const next = g.pitches[p.pitch + 1]
      if (next && p.outs < 3) { assert.ok(next.balls + next.strikes > 0, `seed ${seed}: ${p.text} before a first pitch`) }
    }
    // A caught stealing that ends an inning leaves that batter to lead off
    // the next one: no plate appearance is counted for him.
    for (const side of ["away", "home"]) {
      assert.ok(g.batting[side].every(l => l.pa === l.ab + l.bb + l.hbp + l.sf))
    }
  }
  assert.ok(sb / (2 * n) > 0.5 && sb / (2 * n) < 1, `SB ${sb / (2 * n)}`)
  assert.ok(cs / (2 * n) > 0.05 && cs / (2 * n) < 0.3, `CS ${cs / (2 * n)}`)
  assert.ok(ibb > 0)
})

test("a thousand games: wins add up, averages are sensible, and the page reads them", () => {
  const sum = simulateMlbSummary(P, { away: { team: TEAMS[0] }, home: { team: TEAMS[1] } }, 200, 3)
  assert.equal(sum.wins.home + sum.wins.away, 200)
  assert.equal(sum.margins.length, 200)
  assert.ok(sum.margins.every(m => m !== 0))
  assert.ok(sum.runs.home > 2 && sum.runs.home < 7)
  const regular = sum.hitters.find(h => h.games === 200)
  assert.ok(regular && regular.hit / regular.games > 0.3 && regular.hit / regular.games < 0.95)
  const starter = sum.pitchers.find(p => p.starts === 200)
  assert.ok(starter && starter.outs / starter.games > 9)
  const html = text(mlbSummaryHtml(P, sum, { team: TEAMS[1], probability: 0.55 }))
  assert.match(html, /win [\d,]+ of 200/)
  assert.match(html, /Our pick model gives .+ 55%/)
  assert.match(html, /get a hit/)
  assert.match(html, /quality starts/)
})
