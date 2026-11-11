// Step In: a batting game against real pitchers. This file is the rules
// and the physics, with no drawing in it (batting.js draws and listens).
//
// What is real: the pitcher's pitch types, how fast he throws each (and how
// much that varies), and how often he throws each in each kind of count to
// a left- or right-handed batter - all from docs/data/mlb_sim_profiles.json.
// What is typical rather than his own: how each kind of pitch moves and
// where it is aimed. What is made up: how a swing turns into a batted ball
// (good timing and aim are rewarded; it is a game, not a model).
//
// Distances are in feet, from the catcher's view: x across the plate
// (positive toward first base), z up from the ground, y out toward the mound.

const BAT_PITCH_NAMES = {
FF: "Four-seam fastball", SI: "Sinker", FC: "Cutter", SL: "Slider", ST: "Sweeper", CU: "Curveball",
KC: "Knuckle curve", SV: "Slurve", CH: "Changeup", FS: "Splitter",
}
// A right-hander's typical movement: [sideways, lift] against a pitch with
// no spin. A left-hander's sideways movement is mirrored.
const BAT_MOVEMENT = {
FF: [-0.6, 1.3], SI: [-1.25, 0.65], FC: [0.2, 0.7], SL: [0.5, 0.15], ST: [1.2, 0.1], CU: [0.75, -0.9],
KC: [0.5, -0.9], SV: [0.9, -0.5], CH: [-1.15, 0.5], FS: [-0.9, 0.25],
}
// Where a right-hander tends to aim each kind: [across the plate, height].
const BAT_TARGETS = {
FF: [0, 2.9], SI: [-0.3, 2.1], FC: [0.4, 2.5], SL: [0.55, 1.85], ST: [0.7, 2.0], CU: [0.2, 1.7],
KC: [0.2, 1.65], SV: [0.5, 1.8], CH: [-0.45, 1.85], FS: [0, 1.6],
}
const BAT_RELEASE_DISTANCE = 54.5      // from the hand to the front of the plate
const BAT_RELEASE = [-1.9, 5.9]         // a right-hander's hand at release: [across, height]
// The strike zone, widened by the ball's radius (a ball that nicks it is a strike).
const BAT_ZONE = {halfWidth: 0.83, bottom: 1.5, top: 3.5}
// A swing connects when it is within this long of the ball arriving, and
// this close to where it crosses.
const BAT_TIMING_WINDOW = 0.06
const BAT_REACH = 0.9
const BAT_INNINGS = 3

function batPitchName(type){ return BAT_PITCH_NAMES[type] || "Pitch" }

// Seconds from the hand to the plate (a pitch loses about a tenth of its speed on the way).
function batFlightSeconds(velo){ return BAT_RELEASE_DISTANCE / (velo * 1.4667 * 0.95) }

function batNormal(random){ return Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random()) }

// The kind of count the pitcher's pitch mix is split by.
function batSituation(balls, strikes){
if(balls === 0 && strikes === 0){ return "first" }
return strikes > balls ? "ahead" : balls > strikes ? "behind" : "even"
}

// The pitcher's mix in this count to this batter, as [{type, share}] (his
// types in equal shares if the profile has no mix for the spot).
function batMix(pitcher, side, balls, strikes){
const types = pitcher.types || []
const shares = ((pitcher.mix || {})[side] || {})[batSituation(balls, strikes)]
const total = (shares || []).reduce((sum, v) => sum + v, 0)
return types.map((type, i) => ({type, share: total > 0 ? shares[i] / total : 1 / types.length}))
}

// His arsenal for the scouting card: every pitch, its speed, and how often
// he throws it to this batter over a typical at-bat. Most used first.
function batArsenal(pitcher, side){
const counts = [[0, 0], [0, 1], [1, 1], [1, 0]]
return (pitcher.types || []).map((type, i) => ({
type, name: batPitchName(type), velo: pitcher.velo[i][0],
share: counts.reduce((sum, c) => sum + batMix(pitcher, side, c[0], c[1])[i].share, 0) / counts.length,
})).sort((x, y) => y.share - x.share)
}

// The next pitch: what it is, how fast, where it will cross the plate, and
// the path it takes to get there.
function batNextPitch(pitcher, side, balls, strikes, random){
const mix = batMix(pitcher, side, balls, strikes)
let roll = random(), index = mix.length - 1
for(let i = 0; i < mix.length; i++){
roll -= mix[i].share
if(roll < 0){ index = i; break }
}
const type = mix[index].type
const [mean, spread] = pitcher.velo[index]
const velo = Math.max(35, mean + spread * Math.max(-2.5, Math.min(2.5, batNormal(random))))      // a catcher lobbing it in is on file too
const hand = pitcher.throws === "L" ? -1 : 1
const flight = batFlightSeconds(velo)
const move = BAT_MOVEMENT[type] || [0, 0.5]
const target = BAT_TARGETS[type] || [0, 2.4]
// Fastballs go to either edge; with three balls he comes to the middle,
// with two strikes he works further off it.
let aimX = type === "FF" ? (random() < 0.5 ? -0.35 : 0.35) : target[0] * hand
let aimZ = target[1], wild = 1
if(balls === 3){ aimX *= 0.5; aimZ = 2.5 + (aimZ - 2.5) * 0.5; wild = 0.8 }
else if(strikes === 2){ aimX *= 1.3; aimZ = 2.5 + (aimZ - 2.5) * 1.3 }
const x = Math.max(-2.3, Math.min(2.3, aimX + 0.78 * wild * batNormal(random)))
const z = Math.max(0.3, Math.min(4.8, aimZ + 0.85 * wild * batNormal(random)))
return {
type, name: batPitchName(type), velo: Math.round(velo * 10) / 10, flight, x, z,
x0: BAT_RELEASE[0] * hand, z0: BAT_RELEASE[1],
breakX: move[0] * hand, breakZ: move[1] - 16.1 * flight * flight,
inZone: Math.abs(x) <= BAT_ZONE.halfWidth && z >= BAT_ZONE.bottom && z <= BAT_ZONE.top,
}
}

// Where the ball is when it has gone `progress` of the way to the plate
// (0 at the hand, 1 at the plate, a little past 1 into the catcher's
// glove). It leaves the hand on a straight line and the break and the drop
// build as it goes, so the last part of the flight bends the most.
function batBallAt(pitch, progress){
const s = progress
return {
x: pitch.x0 + (pitch.x - pitch.breakX - pitch.x0) * s + pitch.breakX * s * s,
y: BAT_RELEASE_DISTANCE * (1 - s),
z: pitch.z0 + (pitch.z - pitch.breakZ - pitch.z0) * s + pitch.breakZ * s * s,
}
}

// How far a fly ball carries, from how hard and how steeply it was hit.
function batCarry(exitVelo, launch){
const steep = 1 - Math.max(0, launch - 30) / 60
return Math.max(0, 0.046 * exitVelo * exitVelo * Math.sin(2 * launch * Math.PI / 180) * steep)
}

// The fence: 330 feet down the lines, 400 to center.
function batFence(spray){ return 400 - 70 * Math.min(1, Math.abs(spray) / 45) }

// What happens on a pitch. `swing` is null for a take, or {time, x, z}:
// seconds after the ball left the hand, and where the bat was aimed as the
// plate is crossed. Returns {kind, ...}: kind is "ball", "called_strike",
// "swinging_strike", "foul", "out", "single", "double", "triple" or
// "homer". A swing also carries `timing` (seconds; negative is early) and
// `miss` (feet between the bat and the ball), and a ball in play carries
// exitVelo, launch (degrees), spray (degrees; negative is left field),
// distance (feet to where it lands or is fielded) and `how` in words.
function batSwingResult(pitch, swing, side, random){
if(!swing){ return {kind: pitch.inZone ? "called_strike" : "ball"} }
const timing = swing.time - pitch.flight
const miss = Math.hypot(swing.x - pitch.x, swing.z - pitch.z)
const base = {timing, miss, under: pitch.z - swing.z}
if(Math.abs(timing) > BAT_TIMING_WINDOW || miss > BAT_REACH){ return {kind: "swinging_strike", ...base} }
const quality = (1 - Math.pow(Math.abs(timing) / BAT_TIMING_WINDOW, 1.5)) * (1 - Math.pow(miss / BAT_REACH, 1.5))
// Early pulls the ball, late pushes it the other way.
const spray = timing / BAT_TIMING_WINDOW * 50 * (side === "L" ? -1 : 1) + 4 * batNormal(random)
if(Math.abs(spray) > 45 || quality < 0.12){ return {kind: "foul", ...base, spray} }
const exitVelo = Math.max(45, Math.min(118, 58 + 54 * quality + (pitch.velo - 90) * 0.15 + 2.5 * batNormal(random)))
// Under the ball lifts it, over the top beats it into the ground.
const launch = Math.max(-35, Math.min(65, 16 + base.under * 42 + (pitch.z - 2.5) * 5 + 7 * batNormal(random)))
const ball = {...base, spray, exitVelo: Math.round(exitVelo), launch: Math.round(launch)}
const roll = random()
if(launch < 8){
const through = exitVelo >= 95 ? 0.42 : exitVelo >= 80 ? 0.24 : 0.1
return roll < through ? {kind: "single", ...ball, distance: 190 + 40 * random(), how: "Ground ball through the infield"}
: {kind: "out", ...ball, distance: 95 + 60 * random(), how: "Grounded out"}
}
if(launch < 24){
const lands = exitVelo >= 100 ? 0.78 : exitVelo >= 88 ? 0.62 : 0.38
if(roll >= lands){ return {kind: "out", ...ball, distance: 150 + 130 * random(), how: "Lined out"} }
if(exitVelo >= 98 && random() < 0.45){
return random() < 0.1 ? {kind: "triple", ...ball, distance: 330 + 30 * random(), how: "Line drive into the gap"}
: {kind: "double", ...ball, distance: 300 + 40 * random(), how: "Line drive into the gap"}
}
return {kind: "single", ...ball, distance: 210 + 60 * random(), how: "Line drive"}
}
const carry = batCarry(exitVelo, launch), fence = batFence(spray)
if(carry >= fence){ return {kind: "homer", ...ball, distance: Math.round(carry), how: "Home run"} }
if(launch > 48 || carry < 140){ return {kind: "out", ...ball, distance: Math.max(60, carry), how: "Popped out"} }
if(carry > fence - 35 && roll < 0.45){
return random() < 0.12 ? {kind: "triple", ...ball, distance: fence - 6, how: "Off the wall"} : {kind: "double", ...ball, distance: fence - 6, how: "Off the wall"}
}
if(carry < 235 && roll < 0.4){ return {kind: "single", ...ball, distance: carry, how: "Dropped in front of the outfielder"} }
return {kind: "out", ...ball, distance: carry, how: "Flied out"}
}

// A new game: three innings, three outs each.
function batNewGame(innings = BAT_INNINGS){
return {innings, inning: 1, outs: 0, balls: 0, strikes: 0, bases: [false, false, false], runs: 0, hits: 0, homers: 0,
walks: 0, strikeouts: 0, plateAppearances: 0, pitches: 0, hardest: 0, longest: 0, over: false}
}

// Applies one pitch's result to the game. Returns {game, ended, runs,
// inningOver}: `ended` is how the at-bat finished ("walk", "strikeout",
// "out", "single", ...) or null if it goes on, and `runs` is what scored.
// On a hit every runner moves up as many bases as the batter and one more
// from second on a single or from first on a double.
function batApply(game, result){
const g = {...game, bases: game.bases.slice()}
if(g.over){ return {game: g, ended: null, runs: 0, inningOver: false} }
const kind = result.kind
g.pitches += 1
let ended = null
if(kind === "ball"){ g.balls += 1; if(g.balls === 4){ ended = "walk" } }
else if(kind === "called_strike" || kind === "swinging_strike"){ g.strikes += 1; if(g.strikes === 3){ ended = "strikeout" } }
else if(kind === "foul"){ g.strikes = Math.min(2, g.strikes + 1) }
else{ ended = kind }
let runs = 0, inningOver = false
if(ended){
g.plateAppearances += 1
g.balls = 0; g.strikes = 0
if(ended === "walk"){
g.walks += 1
if(g.bases[0] && g.bases[1] && g.bases[2]){ runs += 1 }
else if(g.bases[0] && g.bases[1]){ g.bases[2] = true }
else if(g.bases[0]){ g.bases[1] = true }
g.bases[0] = true
}else if(ended === "strikeout" || ended === "out"){
g.outs += 1
if(ended === "strikeout"){ g.strikeouts += 1 }
}else{
const bases = {single: 1, double: 2, triple: 3, homer: 4}[ended]
const extra = bases <= 2 ? 1 : 0
const next = [false, false, false]
g.bases.forEach((on, i) => {
if(!on){ return }
const to = i + bases + (i + 1 + bases >= 3 ? extra : 0)
if(to >= 3){ runs += 1 }else{ next[to] = true }
})
if(bases === 4){ runs += 1; g.homers += 1 }else{ next[bases - 1] = true }
g.bases = next
g.hits += 1
g.hardest = Math.max(g.hardest, result.exitVelo || 0)
if(ended === "homer"){ g.longest = Math.max(g.longest, result.distance || 0) }
}
g.runs += runs
if(g.outs === 3){
inningOver = true
if(g.inning >= g.innings){ g.over = true }
else{ g.inning += 1; g.outs = 0; g.bases = [false, false, false] }
}
}
return {game: g, ended, runs, inningOver}
}

// How a swing missed, in words: "38 ms late", "5 inches under it".
function batSwingNote(result){
if(result.timing === undefined){ return "" }
const parts = []
const ms = Math.round(Math.abs(result.timing) * 1000)
parts.push(ms <= 8 ? "right on time" : `${ms} ms ${result.timing < 0 ? "early" : "late"}`)
const inches = Math.round(result.miss * 12)
if(inches >= 3){ parts.push(`${inches} inches ${Math.abs(result.under) >= result.miss * 0.6 ? (result.under > 0 ? "under it" : "over it") : "off it"}`) }
return parts.join(", ")
}

// The call on a pitch, in a sentence.
function batCallText(pitch, result, outcome){
const thrown = `${pitch.name}, ${Math.round(pitch.velo)} mph`
const field = result.spray === undefined ? "" : result.spray < -15 ? "left" : result.spray > 15 ? "right" : "center"
const scored = outcome && outcome.runs ? ` ${outcome.runs === 1 ? "A run scores" : `${outcome.runs} runs score`}.` : ""
const note = batSwingNote(result)
switch(result.kind){
case "ball": return `${outcome && outcome.ended === "walk" ? "Ball four, take your base." : "Ball."} ${thrown}.${scored}`
case "called_strike": return `${outcome && outcome.ended === "strikeout" ? "Strike three looking." : "Strike looking."} ${thrown}.`
case "swinging_strike": return `${outcome && outcome.ended === "strikeout" ? "Strike three swinging." : "Swing and a miss."} ${thrown}: ${note}.`
case "foul": return `Foul ball. ${thrown}: ${note}.`
case "homer": return `Home run! ${result.distance} feet to ${field}, ${result.exitVelo} mph off the bat. ${thrown}.${scored}`
case "out": return `${result.how} to ${field}, ${result.exitVelo} mph. ${thrown}: ${note}.`
default: return `${result.kind[0].toUpperCase()}${result.kind.slice(1)}! ${result.how} to ${field}, ${result.exitVelo} mph. ${thrown}.${scored}`
}
}

// Every pitcher in the profiles, by team: [{team, pitchers: [{id, name,
// role, top}]}] with starters first (most starts first), then relievers.
// `top` is his fastest pitch's usual speed.
function batPitchersByTeam(profiles){
const out = []
Object.keys((profiles && profiles.teams) || {}).sort().forEach(team => {
const t = profiles.teams[team]
const row = (entry, role) => {
const p = profiles.pitchers[String(entry.id)]
return p && (p.types || []).length ? {id: String(entry.id), name: p.name, role, starts: entry.starts || 0, top: Math.max(...p.velo.map(v => v[0]))} : null
}
const starters = (t.starters || []).slice().sort((x, y) => (y.starts || 0) - (x.starts || 0)).map(e => row(e, "starter")).filter(Boolean)
const relievers = (t.bullpen || []).map(e => row(e, "reliever")).filter(Boolean)
if(starters.length + relievers.length){ out.push({team, pitchers: starters.concat(relievers)}) }
})
return out
}

if(typeof module !== "undefined" && module.exports){
module.exports = {
BAT_PITCH_NAMES, BAT_MOVEMENT, BAT_TARGETS, BAT_RELEASE_DISTANCE, BAT_ZONE, BAT_TIMING_WINDOW, BAT_REACH,
batPitchName, batFlightSeconds, batSituation, batMix, batArsenal, batNextPitch, batBallAt, batCarry, batFence,
batSwingResult, batNewGame, batApply, batSwingNote, batCallText, batPitchersByTeam,
}
}
