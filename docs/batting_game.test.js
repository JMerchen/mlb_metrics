// node --test docs/batting_game.test.js
const test = require("node:test")
const assert = require("node:assert/strict")
const g = require("./batting_game.js")

function rng(seed){
  let a = seed >>> 0
  return () => {
    a = (a + 0x6D2B79F5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}
const ace = {
  name: "Test Ace", throws: "R", types: ["FF", "SL", "CH"], velo: [[98, 1.2], [88, 1.4], [89, 1.3]],
  mix: {
    R: { first: [0.6, 0.3, 0.1], ahead: [0.3, 0.6, 0.1], even: [0.5, 0.4, 0.1], behind: [0.9, 0.1, 0] },
    L: { first: [0.5, 0.1, 0.4], ahead: [0.3, 0.2, 0.5], even: [0.5, 0.1, 0.4], behind: [0.8, 0, 0.2] },
  },
}
const middle = { type: "FF", name: "Four-seam fastball", velo: 95, flight: 0.41, x: 0, z: 2.5, x0: -1.9, z0: 5.9, breakX: -0.6, breakZ: -1.4, inZone: true }
const fixed = v => () => v      // a "random" that always says the same thing
const noLuck = fixed(0.25)       // every bell-curve draw comes out 0

test("flight time: a 100 mph pitch gets there in under four tenths of a second", () => {
  assert.ok(Math.abs(g.batFlightSeconds(100) - 0.391) < 0.002)
  assert.ok(g.batFlightSeconds(80) > g.batFlightSeconds(95))
  assert.ok(Math.abs(g.batFlightSeconds(80) / g.batFlightSeconds(100) - 1.25) < 1e-9)
})

test("the count picks the pitch mix", () => {
  assert.equal(g.batSituation(0, 0), "first")
  assert.equal(g.batSituation(0, 2), "ahead")
  assert.equal(g.batSituation(3, 1), "behind")
  assert.equal(g.batSituation(2, 2), "even")
  assert.deepEqual(g.batMix(ace, "R", 3, 0).map(m => m.share), [0.9, 0.1, 0])
  assert.deepEqual(g.batMix(ace, "L", 0, 2).map(m => m.type), ["FF", "SL", "CH"])
  // No mix on file for the spot: his pitches in equal shares.
  assert.deepEqual(g.batMix({ types: ["FF", "CU"], velo: [[94, 1], [80, 1]] }, "R", 0, 0).map(m => m.share), [0.5, 0.5])
  const random = rng(4)
  const thrown = { first: {}, behind: {} }
  for(let i = 0; i < 3000; i++){
    const a = g.batNextPitch(ace, "R", 0, 0, random).type, b = g.batNextPitch(ace, "R", 3, 0, random).type
    thrown.first[a] = (thrown.first[a] || 0) + 1
    thrown.behind[b] = (thrown.behind[b] || 0) + 1
  }
  assert.ok(Math.abs(thrown.first.FF / 3000 - 0.6) < 0.04 && Math.abs(thrown.first.SL / 3000 - 0.3) < 0.04)
  assert.ok(Math.abs(thrown.behind.FF / 3000 - 0.9) < 0.03 && thrown.behind.CH === undefined)
})

test("the scouting card: every pitch with its speed, most used first", () => {
  const card = g.batArsenal(ace, "L")
  assert.deepEqual(card.map(c => c.type), ["FF", "CH", "SL"])
  assert.equal(card[0].name, "Four-seam fastball")
  assert.equal(card[0].velo, 98)
  assert.ok(Math.abs(card.reduce((s, c) => s + c.share, 0) - 1) < 1e-9)
  assert.equal(g.batPitchName("ZZ"), "Pitch")
})

test("a pitch: his speed, a path from the hand to where it crosses, and about half are strikes", () => {
  const random = rng(7)
  let strikes = 0, three = 0
  for(let i = 0; i < 4000; i++){
    const pitch = g.batNextPitch(ace, i % 2 ? "L" : "R", 0, 0, random)
    const index = ace.types.indexOf(pitch.type)
    assert.ok(Math.abs(pitch.velo - ace.velo[index][0]) <= 2.5 * ace.velo[index][1] + 0.06)
    assert.ok(Math.abs(pitch.flight - g.batFlightSeconds(pitch.velo)) < 1e-3)
    const start = g.batBallAt(pitch, 0), end = g.batBallAt(pitch, 1)
    assert.deepEqual([start.x, start.y, start.z], [-1.9, 54.5, 5.9])
    assert.ok(Math.abs(end.x - pitch.x) < 1e-9 && Math.abs(end.z - pitch.z) < 1e-9 && end.y === 0)
    assert.equal(pitch.inZone, Math.abs(pitch.x) <= 0.83 && pitch.z >= 1.5 && pitch.z <= 3.5)
    strikes += pitch.inZone ? 1 : 0
    three += g.batNextPitch(ace, "R", 3, 0, random).inZone ? 1 : 0
  }
  assert.ok(strikes / 4000 > 0.4 && strikes / 4000 < 0.58)
  assert.ok(three > strikes)                 // with three balls he comes to the middle
  // Everything drops (gravity), and a slider drops more than a fastball.
  const fastball = g.batNextPitch({ ...ace, types: ["FF"], velo: [[98, 0]], mix: {} }, "R", 0, 0, random)
  const slider = g.batNextPitch({ ...ace, types: ["SL"], velo: [[88, 0]], mix: {} }, "R", 0, 0, random)
  assert.ok(fastball.breakZ < 0 && slider.breakZ < fastball.breakZ)
  // A left-hander is the mirror image.
  const lefty = g.batNextPitch({ ...ace, throws: "L", types: ["SL"], velo: [[88, 0]], mix: {} }, "R", 0, 0, random)
  assert.equal(lefty.x0, 1.9)
  assert.equal(lefty.breakX, -slider.breakX)
})

test("taking a pitch: a strike in the zone, a ball out of it", () => {
  assert.equal(g.batSwingResult(middle, null, "R", fixed(0.5)).kind, "called_strike")
  assert.equal(g.batSwingResult({ ...middle, x: 1.4, inZone: false }, null, "R", fixed(0.5)).kind, "ball")
})

test("swinging: too early, too late or too far away is a miss", () => {
  const swing = (dt, dx, dz) => g.batSwingResult(middle, { time: middle.flight + dt, x: dx, z: 2.5 + dz }, "R", noLuck)
  assert.equal(swing(-0.09, 0, 0).kind, "swinging_strike")
  assert.equal(swing(0.07, 0, 0).kind, "swinging_strike")
  assert.equal(swing(0, 1.0, 0).kind, "swinging_strike")
  const late = swing(0.07, 0.2, 0)
  assert.ok(Math.abs(late.timing - 0.07) < 1e-9 && Math.abs(late.miss - 0.2) < 1e-9)
  // Close on both but at the edge of the timing: fouled off.
  assert.equal(swing(0.057, 0, 0).kind, "foul")
  assert.equal(swing(-0.057, 0, 0).kind, "foul")
})

test("swinging: square contact is hit hard, and timing and aim decide where it goes", () => {
  const swing = (dt, dz, side = "R") => g.batSwingResult(middle, { time: middle.flight + dt, x: 0, z: 2.5 + dz }, side, noLuck)
  const square = swing(0, 0)
  assert.ok(square.exitVelo >= 110 && Math.abs(square.spray) < 1 && square.launch === 16)
  assert.equal(square.kind, "double")
  // A little under it, on time: in the air and over the fence.
  const lifted = swing(0, -0.25)
  assert.equal(lifted.kind, "homer")
  assert.ok(lifted.launch > 24 && lifted.distance >= 400 && lifted.how === "Home run")
  // Over the top: on the ground.
  const topped = swing(0, 0.3)
  assert.ok(topped.launch < 8 && ["single", "out"].includes(topped.kind))
  // Way under: popped up.
  const popped = swing(0, -0.8)
  assert.equal(popped.kind, "out")
  assert.equal(popped.how, "Popped out")
  // Early pulls it (left for a right-handed batter, right for a left-handed one); late goes the other way.
  assert.ok(swing(-0.03, 0).spray < -15 && swing(0.03, 0).spray > 15)
  assert.ok(swing(-0.03, 0, "L").spray > 15 && swing(0.03, 0, "L").spray < -15)
  // Worse timing, weaker contact.
  assert.ok(swing(0.03, 0).exitVelo < square.exitVelo)
})

test("carry and the fence", () => {
  assert.equal(g.batFence(0), 400)
  assert.equal(g.batFence(-45), 330)
  assert.equal(g.batFence(60), 330)
  assert.ok(g.batCarry(105, 28) > 400 && g.batCarry(90, 28) < 330)
  assert.ok(g.batCarry(105, 55) < g.batCarry(105, 28))
  assert.equal(g.batCarry(100, -10), 0)
})

test("the count: four balls walk, three strikes are out, a foul can't be strike three", () => {
  let game = g.batNewGame()
  let step
  for(let i = 0; i < 4; i++){ step = g.batApply(game, { kind: "ball" }); game = step.game }
  assert.equal(step.ended, "walk")
  assert.deepEqual(game.bases, [true, false, false])
  assert.deepEqual([game.balls, game.strikes, game.walks, game.plateAppearances], [0, 0, 1, 1])
  game = g.batApply(game, { kind: "called_strike" }).game
  game = g.batApply(game, { kind: "foul" }).game
  game = g.batApply(game, { kind: "foul" }).game
  game = g.batApply(game, { kind: "foul" }).game
  assert.equal(game.strikes, 2)
  step = g.batApply(game, { kind: "swinging_strike" })
  assert.equal(step.ended, "strikeout")
  assert.deepEqual([step.game.outs, step.game.strikeouts, step.game.pitches], [1, 1, 9])
})

test("baserunning: hits move everyone up, walks only force, and three outs clear the bases", () => {
  const on = (bases, kind, extra = {}) => g.batApply({ ...g.batNewGame(), bases }, { kind, ...extra })
  assert.deepEqual(on([true, false, false], "single").game.bases, [true, true, false])
  assert.equal(on([false, true, false], "single").runs, 1)                       // scores from second on a single
  assert.deepEqual(on([true, false, true], "single").game.bases, [true, true, false])
  assert.equal(on([true, false, false], "double").runs, 1)                       // and from first on a double
  assert.deepEqual(on([true, false, false], "double").game.bases, [false, true, false])
  assert.equal(on([true, true, true], "triple").runs, 3)
  assert.deepEqual(on([true, true, true], "triple").game.bases, [false, false, true])
  const slam = on([true, true, true], "homer", { exitVelo: 109, distance: 431 })
  assert.deepEqual([slam.runs, slam.game.runs, slam.game.homers, slam.game.hits, slam.game.hardest, slam.game.longest], [4, 4, 1, 1, 109, 431])
  assert.deepEqual(slam.game.bases, [false, false, false])
  const walk = g.batApply({ ...g.batNewGame(), balls: 3, bases: [true, true, true] }, { kind: "ball" })
  assert.equal(walk.runs, 1)
  assert.deepEqual(walk.game.bases, [true, true, true])
  assert.deepEqual(g.batApply({ ...g.batNewGame(), balls: 3, bases: [false, true, false] }, { kind: "ball" }).game.bases, [true, true, false])
  // Third out: the inning turns over and the runners are gone; the last inning's third out ends it.
  const turn = g.batApply({ ...g.batNewGame(), outs: 2, bases: [true, false, true] }, { kind: "out" })
  assert.ok(turn.inningOver && !turn.game.over)
  assert.deepEqual([turn.game.inning, turn.game.outs], [2, 0])
  assert.deepEqual(turn.game.bases, [false, false, false])
  const end = g.batApply({ ...g.batNewGame(), inning: 3, outs: 2 }, { kind: "out" })
  assert.ok(end.game.over)
  assert.equal(g.batApply(end.game, { kind: "homer" }).game.runs, 0)             // nothing counts after the game
})

test("a whole game played by a careful batter ends after nine outs with a sane line", () => {
  const random = rng(21)
  let game = g.batNewGame(), pitches = 0
  while(!game.over && pitches < 2000){
    const pitch = g.batNextPitch(ace, "R", game.balls, game.strikes, random)
    const swing = pitch.inZone ? { time: pitch.flight + 0.02 * (random() - 0.5), x: pitch.x + 0.3 * (random() - 0.5), z: pitch.z + 0.3 * (random() - 0.5) } : null
    game = g.batApply(game, g.batSwingResult(pitch, swing, "R", random)).game
    pitches += 1
  }
  assert.ok(game.over && game.inning === 3 && game.outs === 3)
  assert.equal(game.pitches, pitches)
  assert.ok(game.hits > 0 && game.runs >= game.homers && game.plateAppearances >= 9)
})

test("the call, in words", () => {
  assert.equal(g.batCallText(middle, { kind: "ball" }, { ended: null, runs: 0 }), "Ball. Four-seam fastball, 95 mph.")
  assert.equal(g.batCallText(middle, { kind: "ball" }, { ended: "walk", runs: 1 }), "Ball four, take your base. Four-seam fastball, 95 mph. A run scores.")
  assert.equal(g.batCallText(middle, { kind: "called_strike" }, { ended: "strikeout", runs: 0 }), "Strike three looking. Four-seam fastball, 95 mph.")
  assert.equal(g.batCallText(middle, { kind: "swinging_strike", timing: -0.038, miss: 0.1, under: 0.1 }, { ended: null, runs: 0 }),
    "Swing and a miss. Four-seam fastball, 95 mph: 38 ms early.")
  assert.equal(g.batCallText(middle, { kind: "swinging_strike", timing: 0.004, miss: 0.5, under: 0.5 }, { ended: "strikeout", runs: 0 }),
    "Strike three swinging. Four-seam fastball, 95 mph: right on time, 6 inches under it.")
  assert.equal(g.batSwingNote({ timing: 0.05, miss: 0.5, under: -0.5 }), "50 ms late, 6 inches over it")
  assert.equal(g.batSwingNote({ timing: 0.05, miss: 0.5, under: 0.05 }), "50 ms late, 6 inches off it")
  assert.equal(g.batSwingNote({ kind: "ball" }), "")
  assert.equal(g.batCallText(middle, { kind: "homer", spray: -30, exitVelo: 108, distance: 421, timing: 0, miss: 0, under: 0 }, { ended: "homer", runs: 2 }),
    "Home run! 421 feet to left, 108 mph off the bat. Four-seam fastball, 95 mph. 2 runs score.")
  assert.equal(g.batCallText(middle, { kind: "double", how: "Off the wall", spray: 30, exitVelo: 101, timing: 0, miss: 0, under: 0 }, { ended: "double", runs: 0 }),
    "Double! Off the wall to right, 101 mph. Four-seam fastball, 95 mph.")
  assert.equal(g.batCallText(middle, { kind: "out", how: "Grounded out", spray: 2, exitVelo: 84, timing: 0.02, miss: 0.3, under: -0.3 }, { ended: "out", runs: 0 }),
    "Grounded out to center, 84 mph. Four-seam fastball, 95 mph: 20 ms late, 4 inches over it.")
})

test("the pitcher list: by team, starters by starts then relievers, only pitchers with pitches on file", () => {
  const profiles = {
    teams: {
      NYY: { starters: [{ id: 2, starts: 5 }, { id: 1, starts: 30 }, { id: 99, starts: 9 }], bullpen: [{ id: 3 }, { id: 4 }] },
      BOS: { starters: [], bullpen: [] },
    },
    pitchers: { 1: ace, 2: { ...ace, name: "Fifth Starter" }, 3: { ...ace, name: "Closer", velo: [[101, 1], [90, 1], [88, 1]] }, 4: { name: "No Pitches", types: [], velo: [] } },
  }
  const teams = g.batPitchersByTeam(profiles)
  assert.equal(teams.length, 1)
  assert.deepEqual(teams[0].pitchers.map(p => [p.id, p.name, p.role]), [["1", "Test Ace", "starter"], ["2", "Fifth Starter", "starter"], ["3", "Closer", "reliever"]])
  assert.equal(teams[0].pitchers[2].top, 101)
  assert.deepEqual(g.batPitchersByTeam(null), [])
})

test("every pitcher in the real file can throw a pitch", () => {
  const profiles = require("./data/mlb_sim_profiles.json")
  const random = rng(2)
  const teams = g.batPitchersByTeam(profiles)
  assert.equal(teams.length, 30)
  teams.forEach(t => t.pitchers.forEach(p => {
    const pitcher = profiles.pitchers[p.id]
    ;["L", "R"].forEach(side => {
      const pitch = g.batNextPitch(pitcher, side, 1, 1, random)
      assert.ok(pitch.velo > 30 && pitch.velo < 108 && Number.isFinite(pitch.x) && Number.isFinite(pitch.z) && pitch.name !== "Pitch", `${p.name} ${JSON.stringify(pitch)}`)
      assert.ok(g.batArsenal(pitcher, side).length >= 1)
    })
  }))
})
