// MLB game simulation: a game played pitch by pitch from the players'
// profiles (mlb_sim.py's docs/data/mlb_sim_profiles.json). The pitcher
// picks a pitch the way he does in that count against that side of the
// plate; its result (ball, called strike, swinging strike, foul, in play,
// hit batter) is the league's rate for that pitch type and count times the
// pitcher's own rate for it, the hitter's, the platoon and the times
// through the order; a ball in play becomes a groundout, flyout, lineout,
// single, double, triple or home run the same way; and the runners move as
// they did after that kind of play from that base state with that many
// outs. Starters go about as deep as they do; the bullpen finishes, its
// late-inning arms late. Extra innings start with a runner on second.
// Pure functions returning data or HTML strings, so node tests can check
// them; mlb_sim_view.js does the page.

const MLB_PITCH_NAMES = {
FF: "four-seam fastball", SI: "sinker", FC: "cutter", SL: "slider", ST: "sweeper", CU: "curveball",
KC: "knuckle curve", SV: "slurve", CH: "changeup", FS: "splitter",
}
const MLB_RESULTS = ["B", "C", "S", "F", "X", "H"]
const MLB_CONTACT = ["GO", "FO", "LO", "1B", "2B", "3B", "HR"]
const MLB_MAX_INNINGS = 20

function mlbEsc(value){
return String(value ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;")
}

// Seeded random numbers (mulberry32).
function mlbRng(seed){
let a = (Number(seed) >>> 0) || 1
return function(){
a = (a + 0x6D2B79F5) >>> 0
let t = a
t = Math.imul(t ^ (t >>> 15), t | 1)
t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
return ((t ^ (t >>> 14)) >>> 0) / 4294967296
}
}

// The index picked by u from weights (any scale).
function mlbPick(weights, u){
const total = weights.reduce((a, b) => a + b, 0)
let left = u * total
for(let i = 0; i < weights.length; i++){
left -= weights[i]
if(left < 0){ return i }
}
return weights.length - 1
}

function mlbBucket(balls, strikes){
if(balls === 0 && strikes === 0){ return "first" }
return strikes > balls ? "ahead" : balls === strikes ? "even" : "behind"
}

function mlbShortName(name){
const parts = String(name || "").trim().split(/\s+/)
if(parts.length < 2){ return parts[0] || "" }
const suffix = /^(Jr\.?|Sr\.?|II|III|IV)$/.test(parts[parts.length - 1]) && parts.length > 2
return `${parts[0][0]}. ${suffix ? parts[parts.length - 2] : parts[parts.length - 1]}`
}

function mlbPlayerName(profiles, id){
const p = (profiles.batters || {})[id] || (profiles.pitchers || {})[id]
return p && p.name && !/^\d+$/.test(p.name) ? p.name : `#${id}`
}

// The probabilities of each pitch result for one pitch.
function mlbPitchProbs(league, pitcher, batter, type, count, sameHand, timesThrough){
const base = (league.pitch_table[type] || {})[count]
const pr = (pitcher.ratios || {})[type] || [1, 1, 1, 1, 1, 1]
const br = batter.ratios || [1, 1, 1, 1, 1, 1]
const plat = league.platoon[sameHand ? "same" : "opposite"]
const order = league.order[String(Math.min(3, Math.max(1, timesThrough)))]
const cal = league.calibration || [1, 1, 1, 1, 1, 1]
return base.map((v, i) => v * pr[i] * br[i] * plat[i] * order[i] * cal[i])
}

function mlbContactProbs(league, pitcher, batter){
const cal = league.contact_calibration || league.contact.map(() => 1)
return league.contact.map((v, i) => v * (pitcher.contact || [])[i] * (batter.contact || [])[i] * cal[i])
}

// One team's side for a game: who starts, the batting order and the
// bullpen (late-inning arms first), with stat lines.
function mlbSide(profiles, team, starterId, lineup){
const roster = (profiles.teams || {})[team] || {}
const order = (lineup && lineup.length === 9 ? lineup : roster.lineup || []).map(String)
const starter = String(starterId || ((roster.starters || [])[0] || {}).id)
const pen = (roster.bullpen || []).map(r => String(r.id)).filter(id => id !== starter && (profiles.pitchers || {})[id])
const apps = Object.fromEntries((roster.bullpen || []).map(r => [String(r.id), r.apps]))
return {team, order, starter, pen, apps, used: [starter]}
}

function mlbEmptyBat(id){
return {id, pa: 0, ab: 0, r: 0, h: 0, d: 0, t: 0, hr: 0, rbi: 0, bb: 0, hbp: 0, k: 0, sf: 0}
}

function mlbEmptyPitch(id){
return {id, outs: 0, h: 0, r: 0, er: 0, bb: 0, hbp: 0, k: 0, hr: 0, pitches: 0, strikes: 0, bf: 0}
}

function mlbBaseText(mask){
const names = []
if(mask & 1){ names.push("1st") }
if(mask & 2){ names.push("2nd") }
if(mask & 4){ names.push("3rd") }
return names.length === 3 ? "bases loaded" : names.length ? `runner${names.length > 1 ? "s" : ""} on ${names.join(" and ")}` : "bases empty"
}

const MLB_PLAY_TEXT = {
"1B": "singles", "2B": "doubles", "3B": "triples", HR: "homers",
GO: "grounds out", FO: "flies out", LO: "lines out",
}

// setup: {away: {team, starter, lineup}, home: {...}}; opts: {seed}.
// Returns {away, home, pitches, plays, innings: {away: [runs], home:
// [runs]}, batting, pitching, final, hits, starters}. A pitch: {inning,
// half, balls, strikes (before it), outs, bases, pitcher, batter, type,
// velo, result, x, z, home, away (score after), pa (index of its plate
// appearance), text (when the plate appearance ends)}.
function simulateMlbGame(profiles, setup, opts = {}){
const rng = mlbRng(opts.seed ?? 1)
const lg = profiles.league
const pitchers = profiles.pitchers || {}
const batters = profiles.batters || {}
const sides = {
away: mlbSide(profiles, setup.away.team, setup.away.starter, setup.away.lineup),
home: mlbSide(profiles, setup.home.team, setup.home.starter, setup.home.lineup),
}
const other = s => s === "home" ? "away" : "home"
const score = {away: 0, home: 0}
const hits = {away: 0, home: 0}
const innings = {away: [], home: []}
const batting = {away: new Map(), home: new Map()}
const pitching = {away: new Map(), home: new Map()}
const lineupSpot = {away: 0, home: 0}
const pitches = []
const plays = []
const current = {}
const fallbackPitcher = {ratios: {}, contact: lg.contact.map(() => 1), types: ["FF"], velo: [[93, 2]], mix: null}
const fallbackBatter = {ratios: [1, 1, 1, 1, 1, 1], contact: lg.contact.map(() => 1), bats: {L: "R", R: "R"}}
for(const side of ["away", "home"]){
const st = pitchers[sides[side].starter] || {}
const [mean, sd] = st.stamina || lg.stamina
// How many pitches this starter goes today.
const draw = mean + sd * (rng() + rng() + rng() - 1.5) * 1.15
current[side] = {id: sides[side].starter, pitches: 0, bf: 0, starter: true, limit: Math.max(45, Math.min(115, draw)), inningRuns: 0, enteredInning: 1}
}
const batLine = (side, id) => {
if(!batting[side].has(id)){ batting[side].set(id, mlbEmptyBat(id)) }
return batting[side].get(id)
}
const pitchLine = (side, id) => {
if(!pitching[side].has(id)){ pitching[side].set(id, mlbEmptyPitch(id)) }
return pitching[side].get(id)
}
const name = id => mlbShortName(mlbPlayerName(profiles, id))

let inning = 1
let half = "top"
let over = false

// A new pitcher for the fielding side: the closer for a save in the 9th or
// later, the setup man in the 8th of a close game, otherwise any other arm,
// as often as each really pitches (so the whole bullpen gets used, not
// just its best).
const bringIn = (field, why) => {
const s = sides[field]
const lead = score[field] - score[other(field)]
const fresh = s.pen.filter(id => !s.used.includes(id))
if(!fresh.length){ return false }
let pick
if(inning >= 9 && lead >= 0 && lead <= 3 && fresh.includes(s.pen[0])){ pick = s.pen[0] }
else if(inning === 8 && Math.abs(lead) <= 3 && fresh.includes(s.pen[1])){ pick = s.pen[1] }
else{
const middle = fresh.filter(id => id !== s.pen[0] && id !== s.pen[1])
const pool = middle.length ? middle : fresh
pick = pool[mlbPick(pool.map(id => s.apps[id] || 1), rng())]
}
s.used.push(pick)
current[field] = {id: pick, pitches: 0, bf: 0, starter: false, limit: 30, inningRuns: 0, enteredInning: inning}
plays.push({inning, half, kind: "change", text: `Pitching change: ${name(pick)} replaces ${why}`, pitch: pitches.length})
return true
}

// Before a plate appearance (or the half-inning): is the pitcher done?
const maybeChange = (field, startOfHalf) => {
const c = current[field]
const prev = name(c.id)
if(c.starter){
const tired = startOfHalf ? c.pitches >= c.limit - 6 : c.pitches >= c.limit + 10
const shelled = c.inningRuns >= 5 || (c.inningRuns >= 4 && c.pitches >= 70)
if(tired || shelled){ bringIn(field, prev) }
return
}
// A reliever gets his inning (two in a long outing), unless it's gone
// badly or long.
const done = startOfHalf ? (inning > c.enteredInning && !(c.pitches < 12 && inning === c.enteredInning + 1 && score[field] - score[other(field)] > 4)) : c.pitches >= 35 || c.inningRuns >= 3
if(done){ bringIn(field, prev) }
}

const runners = [null, null, null] // first, second, third: {id, pitcher, side}
let outs = 0
const maskOf = () => (runners[0] ? 1 : 0) | (runners[1] ? 2 : 0) | (runners[2] ? 4 : 0)

// A play's runner movement: the league's outcome for this kind of play,
// base state and outs, applied lead runner first.
const advance = (kind, bat, batId, field, pitcherId) => {
const key = `${kind}|${maskOf()}|${outs}`
const options = lg.transitions[key] || [[1, kind === "HR" ? 0 : kind === "K" || kind[1] === "O" ? maskOf() : maskOf() | 1, kind === "K" || kind[1] === "O" ? 1 : 0, kind === "HR" ? 1 + [0, 1, 2].filter(i => runners[i]).length : 0]]
const [, after, outsAdded, runsDrawn] = options[mlbPick(options.map(o => o[0]), rng())]
const order = [runners[2], runners[1], runners[0]].filter(Boolean)
if(kind !== "K"){ order.push({id: batId, pitcher: pitcherId, side: bat}) }
let runs = Math.min(runsDrawn, order.length)
// A walk-off ends it once the winning run scores (a home run counts all).
const walkOff = bat === "home" && inning >= 9 && score.home + runs > score.away
if(walkOff && kind !== "HR"){ runs = Math.min(runs, score.away - score.home + 1) }
const scorers = order.slice(0, runs)
const rest = order.slice(runs)
const slots = [2, 1, 0].filter(i => after & (1 << i))
runners[0] = runners[1] = runners[2] = null
slots.forEach((slot, i) => { if(rest[i]){ runners[slot] = rest[i] } })
for(const r of scorers){
score[bat] += 1
const line = batting[bat].get(r.id)
if(line){ line.r += 1 }
if(r.pitcher){
const pl = pitching[field].get(r.pitcher)
if(pl){
pl.r += 1
pl.er += 1
}
}
}
current[field].inningRuns += scorers.length
innings[bat][inning - 1] = (innings[bat][inning - 1] || 0) + scorers.length
outs += outsAdded
return {runs: scorers.length, outsAdded, after: maskOf(), walkOff}
}

for(inning = 1; inning <= MLB_MAX_INNINGS && !over; inning++){
for(const h of ["top", "bottom"]){
half = h
const bat = h === "top" ? "away" : "home"
const field = other(bat)
// The home team doesn't bat in the 9th (or later) with the lead.
if(h === "bottom" && inning >= 9 && score.home > score.away){
over = true
break
}
innings[bat][inning - 1] = 0
outs = 0
runners[0] = runners[1] = runners[2] = null
// Extra innings: the last batter up starts on second.
if(inning >= 10){
const n = sides[bat].order.length
runners[1] = {id: sides[bat].order[(lineupSpot[bat] + n - 1) % n], pitcher: null, side: bat}
}
current[field].inningRuns = 0
if(inning > 1 || h === "bottom"){ maybeChange(field, true) }
plays.push({inning, half: h, kind: "half", text: `${h === "top" ? "Top" : "Bottom"} of the ${mlbOrdinal(inning)}`, pitch: pitches.length})
while(outs < 3){
maybeChange(field, false)
const c = current[field]
const pitcher = pitchers[c.id] || fallbackPitcher
const order = sides[bat].order
const batId = order[lineupSpot[bat] % order.length]
lineupSpot[bat] += 1
const batter = batters[batId] || fallbackBatter
const stand = (batter.bats || {})[pitcher.throws || "R"] || "R"
const sameHand = stand === (pitcher.throws || "R")
const timesThrough = Math.floor(c.bf / 9) + 1
const bl = batLine(bat, batId)
const pl = pitchLine(field, c.id)
bl.pa += 1
pl.bf += 1
c.bf += 1
let balls = 0, strikes = 0
let ending = null
const paIndex = plays.length
while(!ending){
const bucket = mlbBucket(balls, strikes)
const mix = pitcher.mix ? pitcher.mix[stand][bucket] : [1]
const ti = mlbPick(mix, rng())
const type = pitcher.types[ti] || "FF"
const [vm, vs] = (pitcher.velo || [])[ti] || [93, 2]
const velo = Math.round(((vm || 93) + (vs || 1) * (rng() + rng() + rng() - 1.5) * 1.4) * 10) / 10
const count = `${balls}-${strikes}`
const probs = mlbPitchProbs(lg, pitcher, batter, type, count, sameHand, timesThrough)
const result = MLB_RESULTS[mlbPick(probs, rng())]
const group = lg.pitch_groups[type] || "FB"
const spots = ((lg.locations || {})[result] || {})[group] || [[0, 0.5]]
const [xb, zn] = spots[Math.floor(rng() * spots.length)]
c.pitches += 1
pl.pitches += 1
if(result !== "B"){ pl.strikes += result === "H" ? 0 : 1 }
const pitch = {
inning, half: h, balls, strikes, outs, bases: maskOf(), pitcher: c.id, batter: batId, type, velo, result,
x: stand === "R" ? -xb : xb, z: zn, stand, pa: paIndex, home: score.home, away: score.away,
}
pitches.push(pitch)
if(result === "B"){ balls += 1; if(balls === 4){ ending = "BB" } }
else if(result === "C" || result === "S"){ strikes += 1; if(strikes === 3){ ending = "K" } }
else if(result === "F"){ if(strikes < 2){ strikes += 1 } }
else if(result === "H"){ ending = "HBP" }
else{ ending = MLB_CONTACT[mlbPick(mlbContactProbs(lg, pitcher, batter), rng())] }
// The count after it (for showing it mid-at-bat).
pitch.count = [Math.min(balls, 3), Math.min(strikes, 2)]
}
const kind = ending === "HBP" ? "BB" : ending
const before = maskOf()
const outsBefore = outs
const moved = advance(kind, bat, batId, field, c.id)
// The batter's and pitcher's lines.
const sac = kind === "FO" && moved.runs > 0 && outsBefore < 2 && moved.outsAdded === 1
if(ending === "BB"){ bl.bb += 1; pl.bb += 1 }
else if(ending === "HBP"){ bl.hbp += 1; pl.hbp += 1 }
else if(sac){ bl.sf += 1 }
else{ bl.ab += 1 }
if(kind === "K"){ bl.k += 1; pl.k += 1 }
if(["1B", "2B", "3B", "HR"].includes(kind)){
bl.h += 1
pl.h += 1
hits[bat] += 1
if(kind === "2B"){ bl.d += 1 }
if(kind === "3B"){ bl.t += 1 }
if(kind === "HR"){ bl.hr += 1; pl.hr += 1 }
}
if(moved.outsAdded < 2 && kind !== "K"){ bl.rbi += moved.runs }
pl.outs += moved.outsAdded
const reached = kind[1] === "O" && moved.outsAdded === 0
const verb = ending === "HBP" ? "is hit by a pitch" : ending === "BB" ? "walks" : kind === "K" ? (pitches[pitches.length - 1].result === "C" ? "strikes out looking" : "strikes out swinging")
: reached ? "reaches on a fielder's choice or error" : moved.outsAdded >= 2 ? "grounds into a double play" : sac ? "hits a sacrifice fly" : MLB_PLAY_TEXT[kind]
const runText = moved.runs ? `; ${moved.runs} run${moved.runs > 1 ? "s" : ""} score${moved.runs > 1 ? "" : "s"}` : ""
const text = `${name(batId)} ${verb}${runText}`
const last = pitches[pitches.length - 1]
last.text = text
last.after = {outs, bases: moved.after, home: score.home, away: score.away}
last.home = score.home
last.away = score.away
plays.push({inning, half: h, kind, text, pitch: pitches.length - 1, runs: moved.runs, outs, before, after: moved.after,
scoring: moved.runs > 0, home: score.home, away: score.away, batter: batId, pitcher: c.id})
if(moved.walkOff){
over = true
break
}
}
if(over){ break }
// Game over after the top of the 9th or later if the home team leads...
if(h === "bottom" && inning >= 9 && score.home !== score.away){ over = true }
}
}
const lines = side => ({
batting: sides[side].order.map(id => batting[side].get(id) || mlbEmptyBat(id))
.concat([...batting[side].values()].filter(l => !sides[side].order.includes(l.id))),
pitching: sides[side].used.map(id => pitching[side].get(id)).filter(Boolean),
})
return {
away: sides.away.team, home: sides.home.team, pitches, plays, innings, hits,
batting: {away: lines("away").batting, home: lines("home").batting},
pitching: {away: lines("away").pitching, home: lines("home").pitching},
final: {away: score.away, home: score.home, innings: Math.max(innings.away.length, innings.home.length)},
}
}

function mlbOrdinal(n){
const s = ["th", "st", "nd", "rd"]
const v = n % 100
return `${n}${s[(v - 20) % 10] || s[v] || s[0]}`
}

function mlbInningsPitched(outs){
return `${Math.floor(outs / 3)}.${outs % 3}`
}

// Line score: runs by inning, then R, H.
function mlbLineScoreHtml(game, upToPitch){
const n = Math.max(9, game.final.innings)
const shown = upToPitch === undefined ? null : game.pitches[upToPitch]
const runsIn = (side, i) => {
if(!shown){ return game.innings[side][i] ?? (i < game.final.innings ? "x" : "") }
// As of this pitch: innings over in full, the current one so far.
const pastHalf = i + 1 < shown.inning || (i + 1 === shown.inning && side === "away" && shown.half === "bottom")
const now = i + 1 === shown.inning && ((side === "away" && shown.half === "top") || (side === "home" && shown.half === "bottom"))
if(pastHalf){ return game.innings[side][i] ?? "" }
if(now){
const before = game.innings[side].slice(0, i).reduce((a, b) => a + (b || 0), 0)
return (side === "away" ? shown.away : shown.home) - before
}
return ""
}
const total = side => shown ? (side === "away" ? shown.away : shown.home) : game.final[side]
const hitsSoFar = side => shown
? game.plays.filter(p => p.pitch <= upToPitch && ["1B", "2B", "3B", "HR"].includes(p.kind) && (p.half === "top") === (side === "away")).length
: game.hits[side]
const name = s => mlbEsc(typeof teamName === "function" ? teamName(game[s], "mlb") : game[s])
const head = Array.from({length: n}, (_, i) => `<th>${i + 1}</th>`).join("")
const row = side => `<tr><th>${name(side)}</th>${Array.from({length: n}, (_, i) => `<td>${runsIn(side, i)}</td>`).join("")}<td class="mlbTot">${total(side)}</td><td>${hitsSoFar(side)}</td></tr>`
return `<table class="mlbLine"><thead><tr><th></th>${head}<th>R</th><th>H</th></tr></thead><tbody>${row("away")}${row("home")}</tbody></table>`
}

// The count, outs and runners, with a diamond.
function mlbSituationHtml(pitch, next){
const p = next || pitch
const count = pitch.count || [pitch.balls, pitch.strikes]
const balls = next ? next.balls : pitch.after ? 0 : count[0]
const strikes = next ? next.strikes : pitch.after ? 0 : count[1]
const outs = next ? next.outs : pitch.after ? pitch.after.outs : pitch.outs
const bases = next ? next.bases : pitch.after ? pitch.after.bases : pitch.bases
const dot = on => `<i class="${on ? "mlbOn" : ""}"></i>`
const base = (on, x, y) => `<rect x="${x}" y="${y}" width="16" height="16" transform="rotate(45 ${x + 8} ${y + 8})" class="${on ? "mlbBaseOn" : "mlbBase"}"/>`
return `<div class="mlbSituation">
<svg class="mlbDiamond" viewBox="0 0 76 56" aria-label="${mlbBaseText(bases)}">${base(bases & 2, 30, 4)}${base(bases & 4, 8, 26)}${base(bases & 1, 52, 26)}</svg>
<div class="mlbCount"><span>B ${dot(balls > 0)}${dot(balls > 1)}${dot(balls > 2)}</span><span>S ${dot(strikes > 0)}${dot(strikes > 1)}</span><span>O ${dot(outs > 0)}${dot(outs > 1)}</span></div>
<div class="mlbInning">${p.half === "top" ? "▲" : "▼"} ${mlbOrdinal(p.inning)}</div>
</div>`
}

const MLB_RESULT_WORDS = {B: "ball", C: "called strike", S: "swinging strike", F: "foul", X: "in play", H: "hit by pitch"}

// The strike zone, from the catcher's view, with this plate appearance's
// pitches numbered (the latest last).
function mlbZoneSvg(game, index){
const p = game.pitches[index]
const these = game.pitches.slice(0, index + 1).filter(q => q.pa === p.pa)
const x = v => Math.min(152, Math.max(8, 80 + v * 60))
const z = v => Math.min(182, Math.max(8, 160 - v * 100))
const dots = these.map((q, i) => {
const cls = q.result === "B" || q.result === "H" ? "mlbBall" : q.result === "X" ? "mlbInPlay" : "mlbStrike"
const last = i === these.length - 1
return `<g class="${cls}${last ? " mlbLast" : ""}"><circle cx="${x(q.x).toFixed(1)}" cy="${z(q.z).toFixed(1)}" r="${last ? 9 : 7.5}"/><text x="${x(q.x).toFixed(1)}" y="${(z(q.z) + 3.5).toFixed(1)}">${i + 1}</text></g>`
}).join("")
return `<svg class="mlbZone" viewBox="0 0 160 205" role="img" aria-label="Pitch ${these.length}: ${mlbEsc(MLB_PITCH_NAMES[p.type] || p.type)}, ${MLB_RESULT_WORDS[p.result]}">
<rect x="${x(-0.83)}" y="${z(1)}" width="${(1.66 * 60).toFixed(1)}" height="100" class="mlbZoneBox"/>
<line x1="${x(-0.28)}" y1="${z(1)}" x2="${x(-0.28)}" y2="${z(0)}" class="mlbZoneGrid"/><line x1="${x(0.28)}" y1="${z(1)}" x2="${x(0.28)}" y2="${z(0)}" class="mlbZoneGrid"/>
<line x1="${x(-0.83)}" y1="${z(0.667)}" x2="${x(0.83)}" y2="${z(0.667)}" class="mlbZoneGrid"/><line x1="${x(-0.83)}" y1="${z(0.333)}" x2="${x(0.83)}" y2="${z(0.333)}" class="mlbZoneGrid"/>
<path d="M${x(-0.71)},190 L${x(0.71)},190 L${x(0.71)},195 L80,201 L${x(-0.71)},195 Z" class="mlbPlate"/>
<text x="${p.stand === "R" ? 14 : 146}" y="110" class="mlbBatterSide">${p.stand === "R" ? "RHB" : "LHB"}</text>
${dots}</svg>`
}

function mlbPitchLine(profiles, pitch){
return `${mlbEsc(MLB_PITCH_NAMES[pitch.type] || pitch.type)}, ${pitch.velo.toFixed(1)} mph - ${MLB_RESULT_WORDS[pitch.result]}`
}

// The latest plays (plate appearances, pitching changes), newest first.
function mlbPlaysHtml(game, upToPitch, count){
const shown = game.plays.filter(p => p.kind !== "half" && p.pitch <= upToPitch)
const list = count ? shown.slice(-count) : shown
return `<ol class="simTicker mlbPlays">${list.reverse().map(p =>
`<li${p.scoring ? " class=\"simScoring\"" : p.kind === "change" ? " class=\"mlbChange\"" : ""}><span class="simWhen">${p.half === "top" ? "▲" : "▼"}${p.inning}</span> ${mlbEsc(p.text)}</li>`).join("")}</ol>`
}

function mlbBoxHtml(profiles, game){
const team = s => mlbEsc(typeof teamName === "function" ? teamName(game[s], "mlb") : game[s])
const nm = id => mlbEsc(mlbPlayerName(profiles, id))
const bat = s => {
const rows = game.batting[s].filter(l => l.pa > 0).map(l => {
const notes = [l.d ? `${l.d > 1 ? l.d : ""}2B` : "", l.t ? `${l.t > 1 ? l.t : ""}3B` : "", l.hr ? `${l.hr > 1 ? l.hr : ""}HR` : ""].filter(Boolean).join(", ")
return `<tr><td>${nm(l.id)}${notes ? ` <small>${notes}</small>` : ""}</td><td>${l.ab}</td><td>${l.r}</td><td>${l.h}</td><td>${l.rbi}</td><td>${l.bb + l.hbp}</td><td>${l.k}</td></tr>`
}).join("")
return `<table class="simBox mlbBox"><thead><tr><th>${team(s)} batting</th><th>AB</th><th>R</th><th>H</th><th>RBI</th><th>BB</th><th>K</th></tr></thead><tbody>${rows}</tbody></table>`
}
const pit = s => {
const rows = game.pitching[s].map(l => `<tr><td>${nm(l.id)}</td><td>${mlbInningsPitched(l.outs)}</td><td>${l.h}</td><td>${l.r}</td><td>${l.bb + l.hbp}</td><td>${l.k}</td><td>${l.hr}</td><td>${l.pitches}</td></tr>`).join("")
return `<table class="simBox mlbBox"><thead><tr><th>${team(s)} pitching</th><th>IP</th><th>H</th><th>R</th><th>BB</th><th>K</th><th>HR</th><th>P</th></tr></thead><tbody>${rows}</tbody></table>`
}
return `<div class="mlbBoxes"><div class="scrollbox">${bat("away")}</div><div class="scrollbox">${bat("home")}</div></div>
<div class="mlbBoxes"><div class="scrollbox">${pit("away")}</div><div class="scrollbox">${pit("home")}</div></div>`
}

function mlbFinalHtml(game){
const f = game.final
const name = s => mlbEsc(typeof teamName === "function" ? teamName(game[s], "mlb") : game[s])
const winner = f.home > f.away ? "home" : "away"
return `<div class="whatIfHeadline">Final: ${name(winner)} win ${Math.max(f.home, f.away)}-${Math.min(f.home, f.away)}${f.innings > 9 ? ` in ${f.innings}` : ""}</div>`
}

if (typeof module !== "undefined" && module.exports) {
module.exports = {
mlbRng, mlbPick, mlbBucket, mlbShortName, mlbPitchProbs, mlbContactProbs, mlbSide, simulateMlbGame,
mlbLineScoreHtml, mlbSituationHtml, mlbZoneSvg, mlbPitchLine, mlbPlaysHtml, mlbBoxHtml, mlbFinalHtml, mlbInningsPitched,
MLB_PITCH_NAMES,
}
}
