// NFL game simulation for the What if? panel: plays a game snap by snap
// from the teams' play profiles (nfl_sim.py's docs/data/nfl_sim_profiles.json)
// and returns every play, the drive chart and a team box score. nfl.js
// plays it back over about five minutes.
//
// The profiles set how a team plays - how often it throws, how its runs
// and completions are spread, how often it's sacked or picked off - but
// not how good it is. Strength comes from calibration: simCalibrate finds
// a yards edge for the home offense (home gains x (1 + edge), away x
// (1 - edge)) and a scale on both, so that a few hundred seeded games
// average the posted pick's margin (its win chance turned into points, as
// the What if? panel does) and its predicted total. The What if? unit
// percents then multiply each side's run and pass yards on top, so a pass
// defense at 80% gives up more through the air in the simulated game.
// Pure functions returning data or HTML strings, so node tests can check
// them; nfl.js does the DOM wiring and the playback.

const SIM_QUARTER_SECONDS = 900
const SIM_OT_SECONDS = 600
const SIM_CALIBRATION_GAMES = 300
const SIM_TIMEOUTS = 3
const SIM_OT_TIMEOUTS = 2
// Timeouts outside the end-of-half and late-game rules (to save a play
// clock, before a big fourth down...): about 1.3 a team a game, so this
// often after a play that keeps the clock running.
const SIM_OTHER_TIMEOUT_RATE = 0.025
// Seconds a flag that wipes out the snap takes (the clock restarts on the
// referee's signal; real median 11-13), and in a hurry-up.
const SIM_SHORT_YARDAGE_STUFF = 0.12
const SIM_FLAG_SECONDS = 12
const SIM_FLAG_HURRY_SECONDS = 6
const SIM_SIDES = ["home", "away"]
// Field zones: inside the opponent's 10 ("goal"), its 11-20 ("red") and the
// offense's own 10 ("backed"). The league's ratios to the middle of the
// field (profiles.league.field_zones) scale the pass rate, sacks and
// completions; on top of the end zone cutting gains off, gains shrink as the
// field does, and losses shrink when pinned deep (real offenses there keep
// the ball in front of them). Tuned so simulated yards per run, touchdowns
// per play and safeties match the league's in each zone.
const SIM_ZONE_GAIN = {goal: 0.9}
const SIM_BACKED_LOSS = 0.6
// Fourth down inside the opponent's 10, outside the last five minutes: how
// often real teams go for it by yards to go (2024-26: 4th and 2, 61%).
const SIM_GOAL_LINE_GO = [[1, 0.92], [2, 0.61], [3, 0.34], [5, 0.19], [99, 0.04]]

function simEsc(value){
return String(value ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;")
}

// A spot's field zone (ball: yards from the offense's own goal line).
function simZone(ball){
return ball >= 90 ? "goal" : ball >= 80 ? "red" : ball <= 10 ? "backed" : null
}

// Seeded random numbers (mulberry32), so a game replays the same way.
function simRng(seed){
let a = (Number(seed) >>> 0) || 1
return function(){
a = (a + 0x6D2B79F5) >>> 0
let t = a
t = Math.imul(t ^ (t >>> 15), t | 1)
t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
return ((t ^ (t >>> 14)) >>> 0) / 4294967296
}
}

// A draw from a distribution kept as quantiles at probs: pick a point
// between the outer probs and interpolate between its neighbors.
function simDraw(quantiles, probs, u){
const lo = probs[0], hi = probs[probs.length - 1]
const p = lo + u * (hi - lo)
let i = 1
while(i < probs.length - 1 && probs[i] < p){ i += 1 }
const f = (p - probs[i - 1]) / (probs[i] - probs[i - 1])
return quantiles[i - 1] + f * (quantiles[i] - quantiles[i - 1])
}

function simTeamProfile(profiles, code){
return Object.assign({}, profiles.league, (profiles.teams || {})[code] || {})
}

// Yards multipliers for one offense: the calibrated strength times the
// What if? percents of its own offense and the other side's defense (a
// defense at 90% allows 10% more).
function simFactors(strength, ownPcts, otherPcts){
const own = ownPcts || {}, other = otherPcts || {}
const pct = (o, k) => o[k] === undefined ? 1 : Number(o[k])
return {
run: strength * pct(own, "run_offense") * (2 - pct(other, "run_defense")),
pass: strength * pct(own, "pass_offense") * (2 - pct(other, "pass_defense")),
}
}

function simEmptyBox(){
return {
points: 0, plays: 0, yards: 0, first_downs: 0,
pass_att: 0, pass_comp: 0, pass_yards: 0, pass_td: 0, ints: 0, sacks: 0, sack_yards: 0,
rush_att: 0, rush_yards: 0, rush_td: 0,
third_att: 0, third_conv: 0, fourth_att: 0, fourth_conv: 0,
fumbles_lost: 0, turnovers: 0, punts: 0, fg_att: 0, fg_made: 0, possession: 0,
penalties: 0, penalty_yards: 0, timeouts: 0, return_td: 0,
}
}

function simEmptyLine(who){
return {
key: who.key, name: who.name, pos: who.pos || "", role: who.role || "",
cmp: 0, att: 0, pass_yds: 0, pass_td: 0, ints: 0, sacks: 0, sack_yds: 0,
rush_att: 0, rush_yds: 0, rush_td: 0, rush_long: 0,
tgt: 0, rec: 0, rec_yds: 0, rec_td: 0, rec_long: 0, fumbles: 0,
fgm: 0, fga: 0, fg_long: 0, xpm: 0, xpa: 0,
}
}

// An entry of list (each with a share) picked by u in [0, 1).
function simPick(list, u){
if(!list || !list.length){ return null }
let left = u
for(const item of list){
left -= item.share
if(left < 0){ return item }
}
return list[list.length - 1]
}

// A list's shares squared and renormalized: the leaders get more.
function simSharpen(list){
if(!list || list.length < 2){ return list }
const total = list.reduce((sum, item) => sum + item.share * item.share, 0)
return total > 0 ? list.map(item => ({...item, share: item.share * item.share / total})) : list
}

// "J. Love" from "Jordan Love".
function simShortName(name){
const parts = String(name || "").trim().split(/\s+/)
if(parts.length < 2){ return parts[0] || "" }
const suffix = /^(Jr\.?|Sr\.?|II|III|IV|V)$/.test(parts[parts.length - 1]) && parts.length > 2
const last = suffix ? parts[parts.length - 2] : parts[parts.length - 1]
return `${parts[0][0]}. ${last}`
}

// One team's players for a simulated game, from its profile: out (pfr or
// gsis ids, as the What if? panel marks them) sit out, and the rest split
// their targets and carries; qb: {id, name} of who starts (the profile's
// starter if left out) - he throws, takes the sacks and gets the QB's own
// runs. Each player's catch rate and gains are relative to his teammates
// who play, so a lineup moves stats between players, not the team's
// totals - the What if? unit percents and the calibration set those.
function simLineup(profiles, code, out, qb){
const team = (profiles.teams || {})[code] || {}
const skip = out || new Set()
const starter = qb && qb.name ? qb : team.qb ? {id: team.qb.gsis, name: team.qb.name} : {id: "qb", name: `${code} QB`}
const qbEntry = {key: `qb:${starter.id}`, name: starter.name, short: starter.short || simShortName(starter.name) || starter.name, pos: "QB", role: "QB"}
const playing = (team.players || []).filter(p => p.role === "QB" || !(skip.has(p.pfr) || skip.has(p.gsis)))
const entry = p => p.role === "QB"
? {...qbEntry}
: {key: p.gsis, name: p.name, short: simShortName(p.name), pos: p.pos || ""}
const normalize = (list, shareKey) => {
const total = list.reduce((sum, p) => sum + (p[shareKey] || 0), 0)
return total > 0 ? list.filter(p => p[shareKey] > 0).map(p => ({...p, share: p[shareKey] / total})) : []
}
const targets = normalize(playing.filter(p => p.role !== "QB"), "targets")
const meanCatch = targets.reduce((sum, p) => sum + p.share * p.catch, 0) || 1
const meanYpr = targets.reduce((sum, p) => sum + p.share * p.catch * p.ypr, 0) / meanCatch || 1
const carries = normalize(playing, "carries")
const meanYpc = carries.reduce((sum, p) => sum + p.share * p.ypc, 0) || 1
return {
qb: qbEntry,
kicker: team.kicker ? {key: `k:${team.kicker.gsis}`, name: team.kicker.name, short: simShortName(team.kicker.name), pos: "K", role: "K"} : null,
receivers: targets.map(p => ({...entry(p), share: p.share, catch: p.catch / meanCatch, ypr: p.ypr / meanYpr})),
rushers: carries.map(p => ({...entry(p), share: p.share, ypc: p.ypc / meanYpc})),
}
}

function simOrdinal(n){
return ["", "1st", "2nd", "3rd", "4th"][n] || `${n}th`
}

// "KC 35", "50" or "BUF 20" for a ball spot in yards from team's own goal.
function simSpot(ball, team, other){
const b = Math.round(ball)
if(b === 50){ return "50" }
return b < 50 ? `${team} ${b}` : `${other} ${100 - b}`
}

function simClock(seconds){
const s = Math.max(0, Math.round(seconds))
return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`
}

function simQuarterLabel(qtr){
return qtr > 4 ? "OT" : `Q${qtr}`
}

// A punt from ball (yards from the punting team's goal): a real punt
// kicked from about there - how far, and whether it was returned, fair
// caught, downed, out of bounds or a touchback. Returns {gross, landing,
// at (where the other side starts), ret, touchdown, text(spot, side)}.
function simPunt(lg, ball, rng){
const buckets = (lg.punts || {}).buckets
if(!buckets){
let net = Math.round(simDraw(lg.punt_net, lg.quantile_probs, rng()))
const tb = ball + net >= 100 || rng() < (lg.punt_touchback_rate || 0.07) * (ball > 50 ? 2 : 0.5)
const at = tb ? 20 : Math.max(1, 100 - (ball + Math.max(net, 0)))
return {gross: net, landing: 100 - at, at, ret: 0, touchdown: false,
text: (spot, side) => `punt, ${tb ? "touchback; " : ""}taken over at the ${spot(at, side)}`}
}
let key = Math.min(9, Math.max(0, Math.floor(ball / 10)))
while(!(buckets[key] || []).length && key > 0){ key -= 1 }
const list = buckets[key] || buckets[Object.keys(buckets)[0]]
const [gross0, end, ret0] = list[Math.floor(rng() * list.length)]
let gross = gross0
let ending = end
// Into the end zone (or a touchback where this punt was kicked from).
if(ball + gross >= 100 || ending === "t"){
gross = Math.max(gross, 100 - ball)
return {gross, landing: 100, at: 20, ret: 0, touchdown: false,
text: () => `punts ${gross} yards into the end zone, touchback`}
}
const landing = ball + gross
const caughtAt = 100 - landing
if(ending !== "r"){
const how = {f: "fair catch", d: "downed", o: "out of bounds"}[ending] || "downed"
return {gross, landing, at: Math.max(1, caughtAt), ret: 0, touchdown: false,
text: (spot, side) => `punts ${gross} yards to the ${spot(caughtAt, side)}, ${how}`}
}
const ret = ret0
const at = caughtAt + ret
if(at >= 100){ return {gross, landing, at: 99, ret, touchdown: true, text: () => ""} }
return {gross, landing, at: Math.max(1, at), ret, touchdown: false,
text: (spot, side) => `punts ${gross} yards to the ${spot(caughtAt, side)}, returned ${ret} yard${ret === 1 ? "" : "s"} to the ${spot(Math.max(1, at), side)}`}
}

// opts: {seed, edge, scale, unitPcts: {home: {unit: pct}, away: {...}},
// lineups: {home: simLineup(...), away: ...}}. Returns {home, away, plays,
// drives, box, final, players}. A play: {qtr, clock (left in the quarter
// after it), poss, down, togo, ball (before it, from the offense's goal),
// end (after it), kind, yards, text, home, away (score after it), drive,
// scoring}. With lineups, each play goes to a player (targets and carries
// by share) and players: {home: [line], away: [...]} has their stat lines.
function simulateGame(profiles, homeCode, awayCode, opts = {}){
const rng = simRng(opts.seed ?? 1)
const lg = profiles.league
const probs = lg.quantile_probs
const edge = Number(opts.edge) || 0
const scale = Number(opts.scale) || 1
const pcts = opts.unitPcts || {}
const code = {home: homeCode, away: awayCode}
const other = side => side === "home" ? "away" : "home"
const prof = {home: simTeamProfile(profiles, homeCode), away: simTeamProfile(profiles, awayCode)}
const factors = {
home: simFactors(scale * (1 + edge), pcts.home, pcts.away),
away: simFactors(scale * (1 - edge), pcts.away, pcts.home),
}
const fg = lg.field_goal || {a: 6, b: -0.1}
const lineups = opts.lineups || {}
const lines = {home: new Map(), away: new Map()}
// The stat line of a lineup entry (made the first time it's needed).
const line = (side, who) => {
if(!who){ return null }
let l = lines[side].get(who.key)
if(!l){
l = simEmptyLine(who)
lines[side].set(who.key, l)
}
return l
}
const name = (side, who) => who ? who.short : code[side]
const score = {home: 0, away: 0}
const box = {home: simEmptyBox(), away: simEmptyBox()}
const plays = []
const drives = []
const firstReceiver = rng() < 0.5 ? "home" : "away"

let qtr = 1
let clock = SIM_QUARTER_SECONDS
let poss = firstReceiver
let ball = 25, down = 1, togo = 10
let drive = null
let over = false
let ot = {on: false, possessions: 0}
let timeouts = {home: SIM_TIMEOUTS, away: SIM_TIMEOUTS}
// Timeouts and the two-minute warning, written into the play-by-play
// once the play that set them off is.
const notes = []
const flags = {O: [], D: []}
for(const t of (lg.penalties || {}).types || []){ flags[t.side].push({...t}) }

const spot = (b, side) => simSpot(b, code[side], code[other(side)])
const lead = side => score[side] - score[other(side)]
const record = (kind, yards, text, extra = {}) => {
const play = {
qtr, clock, poss, down: extra.down ?? down, togo: extra.togo ?? togo, ball: extra.ball ?? ball, end: extra.end ?? ball, kind, yards: Math.round(yards), text,
home: score.home, away: score.away, drive: drives.length - 1, scoring: extra.scoring || null, timeouts: {...timeouts},
change: !!extra.change,
}
plays.push(play)
return play
}
const startDrive = (side, at, how) => {
// Once overtime is decided, nobody gets the ball back.
if(over){ return }
poss = side
ball = Math.min(99, Math.max(1, Math.round(at)))
down = 1
togo = Math.min(10, 100 - ball)
drive = {team: side, start: ball, how, qtr, clock, plays: 0, yards: 0, seconds: 0, result: null, end: ball}
drives.push(drive)
}
const endDrive = result => {
if(drive && !drive.result){
drive.result = result
drive.end = ball
}
if(ot.on){
ot.possessions += 1
// Overtime: both teams get the ball; after that, any lead ends it.
if(ot.possessions >= 2 && score.home !== score.away){ over = true }
}
}
// Runs the clock off a play (the main loop ends the period at 0:00). The
// two-minute warning stops it at 2:00 - or right after a play running
// at the time.
const tick = seconds => {
let used = Math.min(clock, seconds)
if((qtr === 2 || qtr === 4) && clock > 120 && clock - used < 120){
used = clock - 120 >= 6 ? clock - 120 : Math.min(used, 6)
notes.push({kind: "warning", text: "Two-minute warning"})
}
clock -= used
box[poss].possession += used
if(drive){ drive.seconds += used }
}
// A play that ends in bounds: the play's own seconds, then the runoff to
// the next snap - unless someone calls time. The defense does when it's
// behind by two scores or less late in the game, or the offense is pinned
// deep near the end of the half (to get the ball back); the offense does
// when it's behind late, or driving at the end of the half.
const runClock = (playTime, runoff) => {
const after = clock - playTime
const defense = other(poss)
const margin = lead(poss)
let caller = null
if(after > 0 && !(qtr === 4 && clock > 120 && after < 120)){
if(timeouts[defense] > 0 && qtr >= 4 && margin > 0 && margin <= 16 && after <= 180){ caller = defense }
else if(timeouts[poss] > 0 && qtr >= 4 && margin < 0 && after <= 120 && rng() < 0.7){ caller = poss }
else if(timeouts[poss] > 0 && qtr === 5 && margin <= 0 && after <= 120){ caller = poss }
else if(timeouts[poss] > 0 && qtr === 2 && after <= 60 && ball >= 35){ caller = poss }
else if(timeouts[defense] > 0 && qtr === 2 && after <= 110 && ball < 40){ caller = defense }
else if(!(qtr === 2 && after <= 120) && !(qtr >= 4 && after <= 300) && rng() < SIM_OTHER_TIMEOUT_RATE){
const who = rng() < 0.5 ? poss : defense
if(timeouts[who] > 1){ caller = who }
}
}
if(!caller){
tick(playTime + runoff)
return
}
timeouts[caller] -= 1
box[caller].timeouts += 1
tick(playTime)
notes.push({kind: "timeout", text: `Timeout ${code[caller]} (${timeouts[caller]} left)`})
}
// An onside kick only from a team behind (by 24 or less) in the 4th
// quarter: in the last two minutes or so, or with four left when it needs
// more than a score - as teams do. Most fail and hand the other side the
// ball near midfield.
// Whether kneeling on every down left runs out the clock: each kneel is
// 2 seconds when the defense stops it with a timeout (as runClock would
// call one), 40 otherwise, and the two-minute warning stops the clock.
const kneelsRunOut = (defense, margin) => {
let c = clock
let left = timeouts[defense]
for(let k = down; k <= 4; k++){
const after = c - 2
if(after <= 0){ return true }
if(left > 0 && margin <= 16 && after <= 180 && !(qtr === 4 && c > 120 && after < 120)){
left -= 1
c = after
continue
}
let used = Math.min(c, 40)
if((qtr === 2 || qtr === 4) && c > 120 && c - used < 120){ used = c - 120 >= 6 ? c - 120 : Math.min(used, 6) }
c -= used
if(c <= 0){ return true }
}
return false
}
const kickoff = (kicker, how = "Kickoff") => {
if(over){ return }
const receiver = other(kicker)
const behind = -lead(kicker)
const onside = lg.onside && qtr === 4 && behind > 0 && behind <= 24 && (clock <= 130 || (clock <= 240 && behind > 8))
if(onside){
const recovered = rng() < (lg.onside.recovery_rate || 0.075)
const failedStart = (lg.onside.failed_start || []).length ? simDraw(lg.onside.failed_start, probs, rng()) : 52
startDrive(recovered ? kicker : receiver, recovered ? 45 + Math.floor(5 * rng()) : failedStart, recovered ? "Onside kick" : how)
tick(5)
record("kickoff", 0, recovered
? `${code[kicker]} onside kick RECOVERED by ${code[kicker]} at ${spot(ball, kicker)}`
: `${code[kicker]} onside kick, ${code[receiver]} recover at ${spot(ball, receiver)}`)
return
}
const kicks = lg.kickoffs
if(!kicks || !(kicks.returns || []).length){
const start = simDraw(lg.kickoff_start, probs, rng())
startDrive(receiver, start, how)
tick(5)
record("kickoff", 0, `${code[kicker]} kicks off; ${code[receiver]} start at ${spot(ball, receiver)}`)
return
}
// A real kickoff: a touchback, out of bounds, or a catch and a return.
const u = rng()
if(u < kicks.touchback_rate || u < kicks.touchback_rate + kicks.out_of_bounds_rate){
const tb = u < kicks.touchback_rate
startDrive(receiver, tb ? kicks.touchback_spot : kicks.out_of_bounds_spot, how)
tick(5)
record("kickoff", 0, `${code[kicker]} kicks off, ${tb ? "touchback" : "out of bounds"}; ${code[receiver]} start at ${spot(ball, receiver)}`)
return
}
const [caught, ret] = kicks.returns[Math.floor(rng() * kicks.returns.length)]
const where = caught < 0 ? `${-caught} yard${caught === -1 ? "" : "s"} deep in the end zone` : caught === 0 ? "the goal line" : `the ${spot(caught, receiver)}`
const kickText = `${code[kicker]} kicks off, caught at ${where}`
if(caught + ret >= 100){
returnTouchdown(receiver, "kickoff", `${kickText}, returned ${ret} yards`, how)
return
}
startDrive(receiver, caught + ret, how)
tick(caught + ret > 50 ? 10 : 6)
// A kick landing short of the landing zone is spotted at the 40.
record("kickoff", 0, ret === 0 && caught >= 35
? `${code[kicker]} kicks off short of the landing zone; ${code[receiver]} start at the ${spot(ball, receiver)}`
: ret === 0
? `${kickText}, no return`
: `${kickText}, returned ${ret} yard${ret === 1 ? "" : "s"} to the ${spot(ball, receiver)}`)
}
// A kick or punt returned all the way: six points (and the try) for the
// returning team, which then kicks off.
function returnTouchdown(side, kind, text, how){
startDrive(side, 99, how)
tick(12)
score[side] += 6
const good = rng() < (lg.xp_rate || 0.95)
if(good){ score[side] += 1 }
const kicker = line(side, (lineups[side] || {}).kicker)
if(kicker){
kicker.xpa += 1
kicker.xpm += good ? 1 : 0
}
box[side].return_td += 1
record(kind, 0, `${text}, TOUCHDOWN ${code[side]}; extra point ${good ? "is good" : "is no good"}`, {end: 100, scoring: "TD"})
ball = 100
endDrive("Return TD")
kickoff(side)
}
const scoreTouchdown = how => {
score[poss] += 6
box[poss][how === "pass" ? "pass_td" : "rush_td"] += 1
const kicker = line(poss, (lineups[poss] || {}).kicker)
const trail = -lead(poss)
// Down 2 late after the touchdown: go for two to tie.
const twoPoint = qtr >= 4 && trail === 2
let text
if(twoPoint){
const good = rng() < 0.48
if(good){ score[poss] += 2 }
text = `two-point try ${good ? "is good" : "fails"}`
}else{
const good = rng() < (lg.xp_rate || 0.95)
if(good){ score[poss] += 1 }
if(kicker){
kicker.xpa += 1
kicker.xpm += good ? 1 : 0
}
text = `extra point ${good ? "is good" : "is no good"}`
}
return text
}
const turnover = (side, at, result) => {
endDrive(result)
startDrive(other(side), 100 - at, result)
}
// The half (or game) ends: settle the drive and decide what's next.
const endOfPeriod = () => {
if(qtr === 2){
endDrive("End of half")
qtr = 3
clock = SIM_QUARTER_SECONDS
timeouts = {home: SIM_TIMEOUTS, away: SIM_TIMEOUTS}
kickoff(firstReceiver, "Second-half kickoff")
return
}
if(qtr === 1 || qtr === 3){
// The drive carries over to the next quarter.
qtr += 1
clock = SIM_QUARTER_SECONDS
return
}
endDrive(qtr >= 5 ? "End of overtime" : "End of game")
if(qtr === 4 && score.home === score.away){
qtr = 5
clock = SIM_OT_SECONDS
ot = {on: true, possessions: 0}
timeouts = {home: SIM_OT_TIMEOUTS, away: SIM_OT_TIMEOUTS}
kickoff(rng() < 0.5 ? "home" : "away", "Overtime kickoff")
return
}
over = true
}

kickoff(other(firstReceiver), "Opening kickoff")

let guard = 0
while(!over && guard < 500){
guard += 1
while(notes.length){
const note = notes.shift()
record(note.kind, 0, note.text)
}
if(clock <= 0){
endOfPeriod()
continue
}
const p = prof[poss]
const def = other(poss)
const b = box[poss]
const margin = lead(poss)
const late = qtr >= 4 && clock <= 300
const hurry = (qtr === 2 && clock <= 120) || (late && margin < 0) || (qtr === 5 && margin < 0)
const fgDistance = 117 - ball
const fgChance = 1 / (1 + Math.exp(-(fg.a + fg.b * fgDistance)))

// Kneel out a won game - only when kneeling on every down left runs the
// clock out, counting the defense's timeouts and the two-minute warning
// (otherwise a kneel at 2:01 stops at 1:55 and the plan falls apart).
if(qtr >= 4 && margin > 0 && kneelsRunOut(def, margin)){
drive.plays += 1
runClock(2, 38)
const qbLine = line(poss, (lineups[poss] || {}).qb)
if(qbLine){
qbLine.rush_att += 1
qbLine.rush_yds -= 1
}
record("kneel", -1, `${name(poss, (lineups[poss] || {}).qb)} kneel down`, {end: Math.max(1, ball - 1)})
ball = Math.max(1, ball - 1)
b.plays += 1
b.rush_att += 1
b.rush_yards -= 1
b.yards -= 1
drive.yards -= 1
down += 1
togo += 1
if(down > 4){
plays[plays.length - 1].change = true
turnover(poss, ball, "Downs")
}
continue
}

// Field goal as time runs out, or on 4th down.
const lastKick = clock <= 12 && fgDistance <= 58 && (qtr === 2 || ((qtr === 4 || qtr === 5) && margin <= 0 && margin >= -3))
let fourth = null
// Fourth down, as teams decide now: 4th and 1 between the 40s goes 86%
// of the time, 4th and 2 about half; behind by a score or less in the 4th
// quarter, 4th and 3 or less near midfield goes 72%.
if(down === 4 && !lastKick){
const needTd = late && margin < -3
const desperate = qtr >= 4 && clock <= 150 && margin < 0
const behindLate = qtr >= 4 && margin < 0 && margin >= -8 && clock <= 480
const fgTiesOrWins = margin >= -3 && fgDistance <= 50
if(desperate && !(margin >= -3 && fgDistance <= 55)){ fourth = "go" }
else if(needTd && ball >= 40){ fourth = "go" }
else if(behindLate && !(fgTiesOrWins && clock <= 120) && ((togo <= 2 && ball >= 25) || (togo <= 4 && ball >= 40))){ fourth = "go" }
else if(togo <= 1 && ball >= 30){ fourth = "go" }
else if(ball >= 90 && rng() < SIM_GOAL_LINE_GO.find(([most]) => togo <= most)[1]){ fourth = "go" }
else if(fgDistance <= 57 && fgChance >= 0.35){ fourth = "fg" }
else if(togo <= 2 && ball >= 50){ fourth = "go" }
else if(togo <= 3 && ball >= 55){ fourth = "go" }
else{ fourth = "punt" }
if(fourth === "fg" && qtr >= 4 && clock <= 150 && margin < -3){ fourth = "go" }
}
if(lastKick || fourth === "fg"){
b.fg_att += 1
const made = rng() < fgChance
const kicker = line(poss, (lineups[poss] || {}).kicker)
if(kicker){
kicker.fga += 1
kicker.fgm += made ? 1 : 0
if(made){ kicker.fg_long = Math.max(kicker.fg_long, Math.round(fgDistance)) }
}
tick(6)
if(made){
score[poss] += 3
b.fg_made += 1
record("field_goal", 0, `${name(poss, (lineups[poss] || {}).kicker)} ${Math.round(fgDistance)}-yard field goal is GOOD`, {scoring: "FG"})
endDrive("Field goal")
kickoff(poss)
}else{
const at = Math.max(20, 100 - (ball - 7))
record("field_goal", 0, `${name(poss, (lineups[poss] || {}).kicker)} ${Math.round(fgDistance)}-yard field goal is NO GOOD`, {change: true})
endDrive("Missed FG")
startDrive(def, at, "Missed FG")
}
continue
}
if(fourth === "punt"){
b.punts += 1
tick(8)
const punt = simPunt(lg, ball, rng)
const kicker = code[poss]
if(punt.touchdown){
record("punt", 0, `${kicker} punts ${punt.gross} yards to the ${spot(100 - punt.landing, def)}`, {change: true})
endDrive("Punt")
returnTouchdown(def, "punt", `${code[def]} return the punt ${punt.ret} yards`, "Punt")
continue
}
record("punt", 0, `${kicker} ${punt.text(spot, def)}`, {change: true})
endDrive("Punt")
startDrive(def, punt.at, "Punt")
continue
}

// A flag on the snap: by the offense (at its rate a snap) or the defense
// (at its own). Most wipe out the play and are walked off from the line,
// half the distance to the goal at most; a few personal fouls by the
// defense are added on after the play instead.
let flag = null
const flagU = rng()
const offRate = p.offense_penalty_rate ?? (lg.penalties || {}).offense_rate ?? 0
const defRate = prof[def].defense_penalty_rate ?? (lg.penalties || {}).defense_rate ?? 0
if(flagU < offRate + defRate){
const side = flagU < offRate ? "O" : "D"
flag = simPick(flags[side], rng())
}
if(flag && !flag.after){
const before = ball
const downBefore = down, togoBefore = togo
const by = flag.side === "O" ? poss : def
const situation = `${simOrdinal(down)} & ${ball + togo >= 100 ? "goal" : togo}`
let yds
let first = false
if(flag.side === "O"){
yds = Math.min(flag.yards, Math.floor(ball / 2))
ball -= yds
togo += yds
// Intentional grounding also costs the down.
if(flag.name === "Intentional Grounding"){ down += 1 }
}else{
const dpi = /Pass Interference/.test(flag.name) && ((lg.penalties || {}).dpi_yards || []).length
yds = dpi ? Math.max(1, Math.round(simDraw(lg.penalties.dpi_yards, probs, rng()))) : flag.yards
// Never into the end zone: at the 1, half the distance is nothing.
yds = Math.max(0, dpi ? Math.min(yds, 99 - ball) : Math.min(yds, Math.floor((100 - ball) / 2), 99 - ball))
ball += yds
first = rng() < flag.first_down || yds >= togo
if(first){
down = 1
togo = Math.min(10, 100 - ball)
b.first_downs += 1
}else{
togo -= yds
}
}
box[by].penalties += 1
box[by].penalty_yards += yds
drive.yards += flag.side === "O" ? -yds : yds
tick(hurry ? SIM_FLAG_HURRY_SECONDS : SIM_FLAG_SECONDS)
const play = record("penalty", flag.side === "O" ? -yds : yds,
`${situation}: PENALTY ${code[by]}, ${flag.name}, ${yds ? `${yds} yard${yds === 1 ? "" : "s"}` : "half the distance to the goal"}${first ? ", first down" : ""}`,
{ball: before, end: ball, down: downBefore, togo: togoBefore})
play.first_down = first
play.line = ball + togo
if(down > 4){
play.change = true
turnover(poss, 100 - ball, "Downs")
}
continue
}

// A play from scrimmage: run or pass. Short yardage runs (3rd and 1:
// 77% runs in real games); ahead late, a team runs the clock - in the last
// four minutes real teams throw on 6% of first downs, 21% of seconds and
// 42% of thirds - and an incompletion would stop it.
let passRate = p.neutral_pass_rate
if(down === 3 || down === 4){ passRate = togo <= 1 ? 0.25 : togo <= 2 ? 0.4 : togo <= 4 ? 0.65 : 0.85 }
else if(down === 2 && togo >= 8){ passRate += 0.1 }
else if(down === 2 && togo <= 3){ passRate -= 0.1 }
if(hurry){ passRate += 0.3 }
const protecting = qtr >= 4 && margin > 0 && clock <= 240
if(protecting){ passRate = down === 1 ? 0.06 : down === 2 ? 0.2 : togo <= 3 ? 0.25 : 0.55 }
else if(qtr >= 4 && margin >= 9){ passRate -= 0.25 }
if(qtr >= 3 && margin <= -14){ passRate += 0.15 }
const zone = simZone(ball)
const zoneRates = ((lg.field_zones || {})[zone]) || {}
if(zoneRates.pass && !protecting){ passRate *= zoneRates.pass }
// Gains shrink near the goal line; losses shrink pinned deep.
const zoneYards = raw => raw > 0 ? raw * (SIM_ZONE_GAIN[zone] || 1) : zone === "backed" ? raw * SIM_BACKED_LOSS : raw
const from = ball
const isPass = rng() < Math.min(0.95, Math.max(0.1, passRate))
if(down === 3){ b.third_att += 1 }
if(down === 4){ b.fourth_att += 1 }
b.plays += 1
drive.plays += 1
const situation = `${simOrdinal(down)} & ${ball + togo >= 100 ? "goal" : togo}`
let yards = 0
let kind
let text
let clockUsed = hurry ? 16 : 40
let lost = null
const lineup = lineups[poss] || {}
const qb = line(poss, lineup.qb)
let carrier = null
let target = null
if(isPass){
if(rng() < p.sack_rate * (zoneRates.sack || 1)){
kind = "sack"
yards = Math.min(-1, Math.round(zoneYards(lg.sack_yards * (0.4 + 1.2 * rng()))))
b.sacks += 1
b.sack_yards -= yards
if(qb){
qb.sacks += 1
qb.sack_yds -= yards
}
text = `${name(poss, lineup.qb)} sacked for a loss of ${-yards}`
}else{
b.pass_att += 1
target = simPick(lineup.receivers, rng())
const tLine = line(poss, target)
if(qb){ qb.att += 1 }
if(tLine){ tLine.tgt += 1 }
if(rng() < p.int_rate){
kind = "interception"
const air = Math.round(5 + 25 * rng())
const caught = Math.min(ball + air, 99)
const ret = Math.round(15 * rng() * rng())
b.ints += 1
b.turnovers += 1
if(qb){ qb.ints += 1 }
lost = {at: Math.min(99, Math.max(1, 100 - caught + ret)), result: "Interception"}
text = `${name(poss, lineup.qb)} pass${target ? ` for ${target.short}` : ""} INTERCEPTED by ${code[def]} at the ${spot(100 - caught, def)}, returned ${ret} yard${ret === 1 ? "" : "s"}`
clockUsed = 8
}else if(rng() < Math.min(0.95, p.comp_rate * (zoneRates.comp || 1) * (target ? target.catch : 1))){
kind = "pass"
const raw = simDraw(p.completion_yards, probs, rng())
yards = Math.min(100 - ball, Math.round(zoneYards(raw > 0 ? raw * factors[poss].pass * (target ? target.ypr : 1) : raw)))
b.pass_comp += 1
b.pass_yards += yards
if(qb){
qb.cmp += 1
qb.pass_yds += yards
}
if(tLine){
tLine.rec += 1
tLine.rec_yds += yards
tLine.rec_long = Math.max(tLine.rec_long, yards)
}
text = target
? `${name(poss, lineup.qb)} pass to ${target.short} for ${yards} yard${Math.abs(yards) === 1 ? "" : "s"}`
: `${code[poss]} pass complete for ${yards} yard${Math.abs(yards) === 1 ? "" : "s"}`
if(rng() < p.catch_fumble_rate && ball + yards < 100){
lost = {at: Math.min(99, Math.max(1, 100 - (ball + yards))), result: "Fumble"}
text += `, FUMBLE recovered by ${code[def]}`
}
}else{
kind = "incomplete"
text = target ? `${name(poss, lineup.qb)} pass incomplete for ${target.short}` : `${code[poss]} pass incomplete`
clockUsed = 6
}
}
}else{
kind = "run"
// Short yardage, at the goal line, or late in a close game, the lead back
// gets the ball: shares squared (83% of the carries becomes about 98%).
const leverage = togo <= 2 || ball >= 95 || (qtr >= 4 && Math.abs(margin) <= 8 && clock <= 300)
carrier = simPick(leverage ? simSharpen(lineup.rushers) : lineup.rushers, rng())
const raw = simDraw(p.rush_yards, probs, rng())
yards = Math.min(100 - ball, Math.round(zoneYards(raw > 0 ? raw * factors[poss].run * (carrier ? carrier.ypc : 1) : raw)))
// Against a stacked box on 3rd or 4th and short a run is stuffed more
// often: real ones convert 73% of the time.
if((down === 3 || down === 4) && togo <= 2 && rng() < SIM_SHORT_YARDAGE_STUFF){ yards = Math.min(yards, 0) }
b.rush_att += 1
b.rush_yards += yards
const cLine = line(poss, carrier)
if(cLine){
cLine.rush_att += 1
cLine.rush_yds += yards
cLine.rush_long = Math.max(cLine.rush_long, yards)
}
text = `${name(poss, carrier)} run for ${yards} yard${Math.abs(yards) === 1 ? "" : "s"}`
if(rng() < p.run_fumble_rate && ball + yards < 100){
lost = {at: Math.min(99, Math.max(1, 100 - (ball + yards))), result: "Fumble"}
text += `, FUMBLE recovered by ${code[def]}`
}
}
// Total yards are net of sacks, as in a box score.
b.yards += yards
drive.yards += yards
if(lost){
const fumbler = line(poss, kind === "run" ? carrier : target)
if(fumbler && lost.result === "Fumble"){ fumbler.fumbles += 1 }
b.fumbles_lost += lost.result === "Fumble" ? 1 : 0
b.turnovers += lost.result === "Fumble" ? 1 : 0
tick(clockUsed === 6 ? 6 : 8)
ball += yards
record(kind, yards, `${situation}: ${text}`, {ball: from, end: ball, change: true})
turnover(poss, 100 - lost.at, lost.result)
continue
}
ball += yards
if(ball >= 100){
ball = 100
const after = scoreTouchdown(kind === "pass" ? "pass" : "run")
if(kind === "pass"){
if(qb){ qb.pass_td += 1 }
const tLine = line(poss, target)
if(tLine){ tLine.rec_td += 1 }
}else{
const cLine = line(poss, carrier)
if(cLine){ cLine.rush_td += 1 }
}
if(down === 3){ b.third_conv += 1 }
if(down === 4){ b.fourth_conv += 1 }
b.first_downs += 1
tick(clockUsed === 6 ? 6 : 8)
record(kind, yards, `${situation}: ${text}, TOUCHDOWN ${code[poss]}; ${after}`, {ball: from, end: 100, scoring: "TD"})
endDrive("Touchdown")
if(!over){ kickoff(poss) }
continue
}
if(ball <= 0){
score[def] += 2
tick(clockUsed)
record(kind, yards, `${situation}: ${text}, SAFETY`, {ball: from, end: 0, scoring: "Safety"})
ball = 0
endDrive("Safety")
if(!over){
const start = simDraw(lg.kickoff_start, probs, rng()) + 10
startDrive(def, start, "Free kick")
}
continue
}
if(clockUsed === 6){ tick(6) }else{ runClock(6, clockUsed - 6) }
const play = record(kind, yards, `${situation}: ${text}`, {ball: from, end: ball})
if(yards >= togo){
if(down === 3){ b.third_conv += 1 }
if(down === 4){ b.fourth_conv += 1 }
b.first_downs += 1
down = 1
togo = Math.min(10, 100 - ball)
play.first_down = true
}else{
down += 1
togo -= yards
}
// A personal foul after the play: 15 more yards, nearly always a first down.
if(flag && flag.after){
const yds = Math.max(0, Math.min(flag.yards, Math.floor((100 - ball) / 2), 99 - ball))
ball += yds
box[def].penalties += 1
box[def].penalty_yards += yds
drive.yards += yds
if(!play.first_down && (rng() < flag.first_down || yds >= togo)){
b.first_downs += 1
play.first_down = true
}
if(play.first_down){
down = 1
togo = Math.min(10, 100 - ball)
}else{
togo -= yds
}
play.end = ball
play.text += `; PENALTY ${code[def]}, ${flag.name}, ${yds ? `${yds} yards` : "half the distance to the goal"}`
}
play.line = ball + togo
if(down > 4){
play.change = true
turnover(poss, 100 - ball, "Downs")
}
}
for(const side of SIM_SIDES){ box[side].points = score[side] }
return {
home: homeCode, away: awayCode, plays, drives, box,
final: {home: score.home, away: score.away, overtime: qtr >= 5},
players: {home: [...lines.home.values()], away: [...lines.away.values()]},
}
}

// Averages over n seeded games (the same seeds each call, so a change in
// edge or scale isn't lost in the noise).
function simAverage(profiles, homeCode, awayCode, opts, n){
let margin = 0, total = 0, homeWins = 0
for(let i = 0; i < n; i++){
const g = simulateGame(profiles, homeCode, awayCode, {...opts, seed: 1000 + i})
margin += (g.final.home - g.final.away) / n
total += (g.final.home + g.final.away) / n
homeWins += (g.final.home > g.final.away ? 1 : g.final.home === g.final.away ? 0.5 : 0) / n
}
return {margin, total, homeWins}
}

// The edge and scale that make the average game land on targetMargin
// (home points minus away) and targetTotal, with opts.unitPcts (the What
// if? unit percents) and opts.lineups applied and opts.games games
// (default 300). Broyden's
// method: slopes measured once by nudging each, then updated from every
// step taken; the closest try wins if it runs out of steps.
function simCalibrate(profiles, homeCode, awayCode, targetMargin, targetTotal, opts = {}){
const n = opts.games || SIM_CALIBRATION_GAMES
const at = (edge, scale) => {
const r = simAverage(profiles, homeCode, awayCode, {edge, scale, unitPcts: opts.unitPcts, lineups: opts.lineups}, n)
return {...r, edge, scale, off: [r.margin - targetMargin, r.total - targetTotal]}
}
const miss = r => Math.abs(r.off[0]) / 0.25 + Math.abs(r.off[1]) / 0.5
let now = at(0, 1)
const de = 0.08, ds = 0.1
const e1 = at(de, 1)
const s1 = at(0, 1 + ds)
// J = [[dMargin/dEdge, dMargin/dScale], [dTotal/dEdge, dTotal/dScale]]
const J = [
[(e1.off[0] - now.off[0]) / de, (s1.off[0] - now.off[0]) / ds],
[(e1.off[1] - now.off[1]) / de, (s1.off[1] - now.off[1]) / ds],
]
J[0][0] = Math.max(5, J[0][0])
J[1][1] = Math.max(10, J[1][1])
let best = now
for(let step = 0; step < 6 && miss(best) > 1; step++){
const det = J[0][0] * J[1][1] - J[0][1] * J[1][0]
if(Math.abs(det) < 1e-6){ break }
const dEdge = -(J[1][1] * now.off[0] - J[0][1] * now.off[1]) / det
const dScale = -(-J[1][0] * now.off[0] + J[0][0] * now.off[1]) / det
const edge = Math.min(0.45, Math.max(-0.45, now.edge + dEdge))
const scale = Math.min(1.8, Math.max(0.55, now.scale + dScale))
const next = at(edge, scale)
const dx = [edge - now.edge, scale - now.scale]
const dd = dx[0] * dx[0] + dx[1] * dx[1]
if(dd > 1e-10){
for(let r = 0; r < 2; r++){
const predicted = J[r][0] * dx[0] + J[r][1] * dx[1]
const surprise = (next.off[r] - now.off[r]) - predicted
J[r][0] += surprise * dx[0] / dd
J[r][1] += surprise * dx[1] / dd
}
}
J[0][0] = Math.max(5, J[0][0])
J[1][1] = Math.max(10, J[1][1])
now = next
if(miss(now) < miss(best)){ best = now }
}
return {edge: best.edge, scale: best.scale, margin: best.margin, total: best.total, homeWins: best.homeWins, games: n}
}

// n games with the same settings (seeds seed, seed + 1, ...): how often
// each side wins, the average score, the spread of margins (home minus
// away) and totals, the commonest finals and each player's averages.
const SIM_SUMMARY_STATS = ["cmp", "att", "pass_yds", "pass_td", "ints", "rush_att", "rush_yds", "rush_td", "tgt", "rec", "rec_yds", "rec_td", "fgm", "fga"]

function simSummary(profiles, homeCode, awayCode, opts = {}, n = 1000){
const seed = Number(opts.seed) || 1
const out = {
n, home: homeCode, away: awayCode, homeWins: 0, awayWins: 0, ties: 0, overtime: 0,
points: {home: 0, away: 0}, margins: [], totals: [], scores: new Map(), players: {home: new Map(), away: new Map()},
}
for(let i = 0; i < n; i++){
const g = simulateGame(profiles, homeCode, awayCode, {...opts, seed: seed + i})
const f = g.final
if(f.home > f.away){ out.homeWins += 1 }else if(f.away > f.home){ out.awayWins += 1 }else{ out.ties += 1 }
if(f.overtime){ out.overtime += 1 }
out.points.home += f.home / n
out.points.away += f.away / n
out.margins.push(f.home - f.away)
out.totals.push(f.home + f.away)
const key = `${Math.max(f.home, f.away)}-${Math.min(f.home, f.away)}|${f.home > f.away ? "home" : f.away > f.home ? "away" : "tie"}`
out.scores.set(key, (out.scores.get(key) || 0) + 1)
for(const side of SIM_SIDES){
for(const p of g.players[side]){
let a = out.players[side].get(p.key)
if(!a){
a = {key: p.key, name: p.name, pos: p.pos, role: p.role, games: 0, td_games: 0, rush_100: 0, rec_100: 0, pass_300: 0}
for(const k of SIM_SUMMARY_STATS){ a[k] = 0 }
out.players[side].set(p.key, a)
}
for(const k of SIM_SUMMARY_STATS){ a[k] += p[k] / n }
if(p.rush_td + p.rec_td > 0){ a.td_games += 1 / n }
if(p.rush_yds >= 100){ a.rush_100 += 1 / n }
if(p.rec_yds >= 100){ a.rec_100 += 1 / n }
if(p.pass_yds >= 300){ a.pass_300 += 1 / n }
}
}
}
const sorted = xs => [...xs].sort((a, b) => a - b)
const pctile = (xs, q) => xs[Math.min(xs.length - 1, Math.floor(q * xs.length))]
const m = sorted(out.margins), t = sorted(out.totals)
out.marginRange = [pctile(m, 0.1), pctile(m, 0.5), pctile(m, 0.9)]
out.totalRange = [pctile(t, 0.1), pctile(t, 0.5), pctile(t, 0.9)]
out.commonScores = [...out.scores.entries()].sort((a, b) => b[1] - a[1]).slice(0, 5)
.map(([key, count]) => {
const [score, winner] = key.split("|")
return {score, winner, share: count / n}
})
out.players = {home: [...out.players.home.values()], away: [...out.players.away.values()]}
delete out.scores
return out
}

// Margin chart: share of games by final margin in 3-point bins, a team's
// wins on its own side of the tie line (home to the right).
function simMarginBins(margins, width = 3, cap = 30){
const bins = new Map()
for(const m of margins){
const c = Math.max(-cap, Math.min(cap, m))
const b = c === 0 ? 0 : Math.sign(c) * Math.ceil(Math.abs(c) / width)
bins.set(b, (bins.get(b) || 0) + 1)
}
const top = Math.ceil(cap / width)
const out = []
for(let b = -top; b <= top; b++){
const lo = b === 0 ? 0 : (Math.abs(b) - 1) * width + 1
const hi = b === 0 ? 0 : Math.abs(b) === top ? null : Math.abs(b) * width
out.push({bin: b, side: b > 0 ? "home" : b < 0 ? "away" : "tie", lo, hi, share: (bins.get(b) || 0) / margins.length})
}
return out
}

function simMarginChartHtml(summary){
const bins = simMarginBins(summary.margins)
const name = side => simEsc(simTeamName(summary[side]))
const w = 640, h = 196, left = 8, right = 8, top = 22, bottom = 48
const slot = (w - left - right) / bins.length
const bar = Math.min(24, slot - 2)
const max = Math.max(...bins.map(b => b.share), 0.01)
const y = v => top + (h - top - bottom) * (1 - v / max)
const range = b => b.side === "tie" ? "a tie" : `${name(b.side)} by ${b.lo}${b.hi === null ? "+" : b.hi === b.lo ? "" : `-${b.hi}`}`
const rects = bins.map((b, i) => {
const x = left + i * slot + (slot - bar) / 2
const y0 = y(b.share), base = h - bottom
const height = Math.max(0, base - y0)
const r = Math.min(4, height / 2, bar / 2)
const path = height <= 0 ? "" : `M${x},${base} V${y0 + r} Q${x},${y0} ${x + r},${y0} H${x + bar - r} Q${x + bar},${y0} ${x + bar},${y0 + r} V${base} Z`
const tip = `${range(b)}: ${(b.share * 100).toFixed(1)}% of games`
return `<g class="simBin"><title>${tip}</title><rect x="${left + i * slot}" y="${top}" width="${slot}" height="${h - top - bottom}" class="simHit"/>${path ? `<path d="${path}" class="simBar simBar-${b.side}"/>` : ""}</g>`
}).join("")
const mid = left + (bins.length / 2) * slot
const ticks = [-30, -21, -14, -7, 0, 7, 14, 21, 30].map(v => {
const i = bins.findIndex(b => v === 0 ? b.bin === 0 : Math.sign(b.bin) === Math.sign(v) && Math.abs(v) >= b.lo && (b.hi === null || Math.abs(v) <= b.hi))
const x = left + (i + 0.5) * slot
return `<text x="${x}" y="${h - bottom + 20}" class="simAxis">${v === 0 ? "Tie" : `${Math.abs(v)}${Math.abs(v) === 30 ? "+" : ""}`}</text>`
}).join("")
return `<figure class="simChart">
<figcaption>Final margin across ${summary.n.toLocaleString()} games</figcaption>
<div class="simLegend"><span><i class="simSwatch simBar-away"></i>${name("away")} win</span><span><i class="simSwatch simBar-home"></i>${name("home")} win</span></div>
<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Final margins: ${name("home")} won ${Math.round(summary.homeWins / summary.n * 100)}%, ${name("away")} ${Math.round(summary.awayWins / summary.n * 100)}%">
<line x1="${left}" y1="${h - bottom}" x2="${w - right}" y2="${h - bottom}" class="simBaseline"/>
<line x1="${mid}" y1="${top - 8}" x2="${mid}" y2="${h - bottom}" class="simTieLine"/>
${rects}${ticks}
<text x="${left}" y="${h - 4}" class="simAxis simAxisStart">← ${name("away")} by more</text>
<text x="${w - right}" y="${h - 4}" class="simAxis simAxisEnd">${name("home")} by more →</text>
</svg>
<details class="simAllPlays"><summary>As a table</summary><table class="simBox simCommon"><thead><tr><th>Final margin</th><th>Games</th></tr></thead><tbody>${
bins.filter(b => b.share > 0).map(b => `<tr><td>${range(b)}</td><td>${(b.share * 100).toFixed(1)}%</td></tr>`).join("")}</tbody></table></details></figure>`
}

const simFixed = (v, d = 1) => Number(v).toFixed(d)
const simPct = v => `${Math.round(v * 100)}%`

// The summary: who wins how often, the score, the spreads, the commonest
// finals, and each team's players on average (the regulars: anyone
// averaging a carry, a target or a pass).
function simSummaryHtml(summary, target){
const n = summary.n
const name = side => simEsc(simTeamName(summary[side]))
const fav = summary.homeWins >= summary.awayWins ? "home" : "away"
const wins = fav === "home" ? summary.homeWins : summary.awayWins
const [m10, m50, m90] = summary.marginRange
const [t10, t50, t90] = summary.totalRange
const marginWords = v => v === 0 ? "a tie" : v > 0 ? `${name("home")} by ${v}` : `${name("away")} by ${-v}`
const headline = `${name(fav)} win ${wins.toLocaleString()} of ${n.toLocaleString()} (${simPct(wins / n)})`
const stats = `<div class="simStats">
<div><b>${simFixed(summary.points.away)}-${simFixed(summary.points.home)}</b><span>average score (${name("away")}-${name("home")})</span></div>
<div><b>${marginWords(m50)}</b><span>median margin; 8 in 10 between ${marginWords(m10)} and ${marginWords(m90)}</span></div>
<div><b>${t50}</b><span>median total; 8 in 10 between ${t10} and ${t90}</span></div>
<div><b>${simPct(summary.overtime / n)}</b><span>went to overtime${summary.ties ? `; ${simPct(summary.ties / n)} ended tied` : ""}</span></div>
</div>`
const scores = `<table class="simBox simCommon"><thead><tr><th>Commonest finals</th><th>Games</th></tr></thead><tbody>${
summary.commonScores.map(c => `<tr><td>${c.winner === "tie" ? `Tie ${c.score}` : `${name(c.winner)} ${c.score}`}</td><td>${simPct(c.share)}</td></tr>`).join("")}</tbody></table>`
const note = target
? `<p class="simIntro">Tuned to ${name(target.margin >= 0 ? "home" : "away")} by ${Math.abs(target.margin).toFixed(1)} and ${target.total.toFixed(0)} total points; the What if? panel gives ${name("home")} ${simPct(target.probability)}.</p>`
: ""
const teams = SIM_SIDES.slice().reverse().map(side => `<h4>${name(side)}</h4>${simPlayerAveragesHtml(summary.players[side])}`).join("")
return `<div class="simSummary"><div class="whatIfHeadline">${headline}</div>${note}${stats}
${simMarginChartHtml(summary)}
<div class="scrollbox">${scores}</div>
<h4>Players, per game on average</h4>${teams}</div>`
}

function simPlayerAveragesHtml(players){
const qb = players.filter(p => p.role === "QB" && p.att >= 1)
const skill = players.filter(p => p.role !== "QB" && p.role !== "K" && (p.rush_att >= 1 || p.tgt >= 1))
.sort((a, b) => (b.rush_yds + b.rec_yds) - (a.rush_yds + a.rec_yds))
const kick = players.filter(p => p.role === "K")
const row = (p, parts) => `<li><span class="simAvgName">${simEsc(p.name)} <small>${simEsc(p.pos)}</small></span><span class="simAvgLine">${parts.filter(Boolean).join(" · ")}</span></li>`
const rows = []
for(const p of qb){
rows.push(row(p, [`${simFixed(p.cmp)}/${simFixed(p.att)}, ${Math.round(p.pass_yds)} yds, ${simFixed(p.pass_td)} TD, ${simFixed(p.ints)} INT`,
`${Math.round(p.rush_yds)} rush yds`, `${simPct(p.pass_300)} throw for 300+`]))
}
for(const p of skill){
rows.push(row(p, [
p.rush_att >= 1 ? `${simFixed(p.rush_att)} car, ${Math.round(p.rush_yds)} yds` : "",
p.tgt >= 1 ? `${simFixed(p.rec)} rec, ${Math.round(p.rec_yds)} yds` : "",
`${simPct(p.td_games)} score a TD`,
p.rush_100 >= 0.05 ? `${simPct(p.rush_100)} run for 100+` : "",
p.rec_100 >= 0.05 ? `${simPct(p.rec_100)} 100+ receiving` : "",
]))
}
for(const p of kick){ rows.push(row(p, [`${simFixed(p.fgm)}/${simFixed(p.fga)} FG`])) }
return `<ul class="simAverages">${rows.join("")}</ul>`
}

// One game's player stat lines, a team at a time: passing, rushing,
// receiving and kicking.
function simPlayerStatsHtml(game){
const block = side => {
const ps = game.players[side] || []
const passing = ps.filter(p => p.att > 0 || p.sacks > 0)
const rushing = ps.filter(p => p.rush_att > 0).sort((a, b) => b.rush_yds - a.rush_yds)
const receiving = ps.filter(p => p.tgt > 0).sort((a, b) => b.rec_yds - a.rec_yds)
const kicking = ps.filter(p => p.fga > 0 || p.xpa > 0)
const table = (title, head, rows) => rows.length
? `<table class="simBox simLines"><thead><tr><th>${title}</th>${head.map(h => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows.join("")}</tbody></table>`
: ""
const who = p => `<td>${simEsc(p.name)}</td>`
return `<h4>${simEsc(simTeamName(game[side]))}</h4><div class="scrollbox">
${table("Passing", ["C/Att", "Yds", "TD", "INT", "Sk"], passing.map(p => `<tr>${who(p)}<td>${p.cmp}/${p.att}</td><td>${p.pass_yds}</td><td>${p.pass_td}</td><td>${p.ints}</td><td>${p.sacks}</td></tr>`))}
${table("Rushing", ["Car", "Yds", "Avg", "TD", "Long"], rushing.map(p => `<tr>${who(p)}<td>${p.rush_att}</td><td>${p.rush_yds}</td><td>${simFixed(p.rush_yds / p.rush_att)}</td><td>${p.rush_td}</td><td>${p.rush_long}</td></tr>`))}
${table("Receiving", ["Rec/Tgt", "Yds", "Avg", "TD", "Long"], receiving.map(p => `<tr>${who(p)}<td>${p.rec}/${p.tgt}</td><td>${p.rec_yds}</td><td>${p.rec ? simFixed(p.rec_yds / p.rec) : "-"}</td><td>${p.rec_td}</td><td>${p.rec ? p.rec_long : "-"}</td></tr>`))}
${table("Kicking", ["FG", "XP", "Long"], kicking.map(p => `<tr>${who(p)}<td>${p.fgm}/${p.fga}</td><td>${p.xpm}/${p.xpa}</td><td>${p.fgm ? p.fg_long : "-"}</td></tr>`))}
</div>`
}
return block("away") + block("home")
}

// Field picture: the offense's ball spot, the first-down line and the end
// zones, home driving to the right (so home's own end zone is on the left).
function simFieldSvg(play, homeCode, awayCode){
const x = yards => 40 + yards * 5.2
const toRight = play.poss === "home"
const at = b => x(toRight ? b : 100 - b)
const ball = at(play.end ?? play.ball)
const marker = play.kind === "kickoff" || play.kind === "punt" || play.kind === "field_goal" || play.scoring
? null : at(Math.min(100, play.line ?? ((play.end ?? play.ball) + (play.first_down ? 10 : Math.max(0, play.togo - play.yards)))))
const lines = []
for(let y = 10; y <= 90; y += 10){
lines.push(`<line x1="${x(y)}" y1="6" x2="${x(y)}" y2="74" class="simYardLine"/>`)
lines.push(`<text x="${x(y)}" y="70" class="simYardNum">${y <= 50 ? y : 100 - y}</text>`)
}
return `<svg class="simField" viewBox="0 0 600 80" role="img" aria-label="Ball at ${simEsc(simSpot(play.end ?? play.ball, play.poss === "home" ? homeCode : awayCode, play.poss === "home" ? awayCode : homeCode))}">
<rect x="0" y="4" width="600" height="72" rx="6" class="simGrass"/>
<rect x="0" y="4" width="40" height="72" class="simEndZone"/><rect x="560" y="4" width="40" height="72" class="simEndZone"/>
<text x="20" y="44" class="simEndText" transform="rotate(-90 20 40)">${simEsc(homeCode)}</text>
<text x="580" y="44" class="simEndText" transform="rotate(90 580 40)">${simEsc(awayCode)}</text>
${lines.join("")}
${marker === null ? "" : `<line x1="${marker}" y1="6" x2="${marker}" y2="74" class="simFirstDown"/>`}
<ellipse cx="${ball}" cy="38" rx="7" ry="4.5" class="simBall"/>
</svg>`
}

function simTeamName(code){
return typeof teamName === "function" ? teamName(code, "nfl") : code
}

// The score and clock after play, and the down and distance coming up
// (next, the play after it). After the last play it reads Final.
function simScoreboardHtml(game, play, next){
const p = play || {qtr: 1, clock: SIM_QUARTER_SECONDS, home: 0, away: 0}
const poss = next ? next.poss : null
// Timeouts left, as three dots under the name.
const dots = s => {
const left = (p.timeouts || {})[s]
if(left === undefined){ return "" }
return `<span class="simTimeouts" aria-label="${left} timeout${left === 1 ? "" : "s"} left">${
[0, 1, 2].map(i => `<i class="${i < left ? "simTo" : "simToUsed"}"></i>`).join("")}</span>`
}
const side = s => `<div class="simTeam${poss === s ? " simHasBall" : ""}"><span class="simTeamName"><span class="simLong">${simEsc(simTeamName(game[s]))}</span><span class="simShort">${simEsc(game[s])}</span>${dots(s)}</span><span class="simPoints">${p[s]}</span></div>`
const final = !next && play === game.plays[game.plays.length - 1]
const when = final
? `<span class="simQuarter">&nbsp;</span><span class="simClock">Final${game.final.overtime ? "/OT" : ""}</span>`
: `<span class="simQuarter">${simQuarterLabel(p.qtr)}</span><span class="simClock">${simClock(p.clock)}</span>`
const situation = next && next.kind !== "kickoff" && next.down
? `${simOrdinal(next.down)} & ${next.ball + next.togo >= 100 ? "goal" : next.togo} at ${simEsc(simSpot(next.ball, game[next.poss], game[next.poss === "home" ? "away" : "home"]))}`
: ""
return `<div class="simScoreboard">${side("away")}<div class="simClockBox">${when}<span class="simDown">${situation}</span></div>${side("home")}</div>`
}

const SIM_SCRIMMAGE = new Set(["run", "pass", "incomplete", "sack", "interception", "kneel"])

// The drive chart as of play playIndex (all of it when left out): drives
// over so far in full, the one under way with its plays and yards so far.
function simDriveChartHtml(game, playIndex){
const plays = game.plays
const at = playIndex === undefined ? plays.length - 1 : playIndex
const current = plays[at] ? plays[at].drive : game.drives.length - 1
const lastPlay = game.drives.map(() => -1)
plays.forEach((p, i) => { lastPlay[p.drive] = i })
const finished = at >= plays.length - 1
const rows = []
game.drives.forEach((d, i) => {
if(i > current && !finished){ return }
const team = game[d.team]
const opp = game[d.team === "home" ? "away" : "home"]
const over = finished || i < current || at >= lastPlay[i]
let count = d.plays, yards = d.yards, time = simClock(d.seconds), result = d.result || ""
if(!over){
const sofar = plays.filter((p, j) => j <= at && p.drive === i && SIM_SCRIMMAGE.has(p.kind))
count = sofar.length
yards = sofar.reduce((sum, p) => sum + p.yards, 0)
time = "…"
result = "…"
}
const cls = over && (result === "Touchdown" || result === "Return TD" || result === "Field goal") ? " class=\"simScoreDrive\"" : !over ? " class=\"simLiveDrive\"" : ""
rows.push(`<tr${cls}><td>${simEsc(team)}</td><td>${simQuarterLabel(d.qtr)} ${simClock(d.clock)}</td><td>${simEsc(simSpot(d.start, team, opp))}</td><td>${count}</td><td>${yards}</td><td>${time}</td><td>${simEsc(result)}</td></tr>`)
})
if(!rows.length){ return "" }
return `<table class="simDrives"><thead><tr><th>Team</th><th>Start</th><th>From</th><th>Plays</th><th>Yards</th><th>Time</th><th>Result</th></tr></thead><tbody>${rows.join("")}</tbody></table>`
}

// The latest plays up to playIndex, newest first (count of them, or all).
function simTickerHtml(game, playIndex, count){
const upTo = game.plays.slice(0, playIndex + 1)
const shown = count ? upTo.slice(-count) : upTo
return `<ol class="simTicker">${shown.reverse().map(p =>
`<li${p.scoring ? " class=\"simScoring\"" : p.change ? " class=\"simChange\"" : ""}><span class="simWhen">${simQuarterLabel(p.qtr)} ${simClock(p.clock)}</span> ${simEsc(p.text)}</li>`).join("")}</ol>`
}

function simBoxScoreHtml(game){
const a = game.box.away, h = game.box.home
const pct = (n, d) => `${n}-${d}`
const rows = [
["First downs", a.first_downs, h.first_downs],
["Total yards", a.yards, h.yards],
["Plays", a.plays, h.plays],
["Yards per play", (a.yards / Math.max(1, a.plays)).toFixed(1), (h.yards / Math.max(1, h.plays)).toFixed(1)],
["Passing (comp-att, yards)", `${a.pass_comp}-${a.pass_att}, ${a.pass_yards - a.sack_yards}`, `${h.pass_comp}-${h.pass_att}, ${h.pass_yards - h.sack_yards}`],
["Sacked (yards)", `${a.sacks} (${a.sack_yards})`, `${h.sacks} (${h.sack_yards})`],
["Rushing (att, yards)", `${a.rush_att}, ${a.rush_yards}`, `${h.rush_att}, ${h.rush_yards}`],
["Touchdowns (pass, rush, return)", `${a.pass_td}, ${a.rush_td}, ${a.return_td}`, `${h.pass_td}, ${h.rush_td}, ${h.return_td}`],
["Field goals", pct(a.fg_made, a.fg_att), pct(h.fg_made, h.fg_att)],
["3rd down", pct(a.third_conv, a.third_att), pct(h.third_conv, h.third_att)],
["4th down", pct(a.fourth_conv, a.fourth_att), pct(h.fourth_conv, h.fourth_att)],
["Turnovers (INT, fumbles)", `${a.turnovers} (${a.ints}, ${a.fumbles_lost})`, `${h.turnovers} (${h.ints}, ${h.fumbles_lost})`],
["Punts", a.punts, h.punts],
["Penalties (yards)", `${a.penalties} (${a.penalty_yards})`, `${h.penalties} (${h.penalty_yards})`],
["Time of possession", simClock(a.possession), simClock(h.possession)],
]
return `<table class="simBox"><thead><tr><th></th><th>${simEsc(game.away)}</th><th>${simEsc(game.home)}</th></tr></thead><tbody>${
rows.map(r => `<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td></tr>`).join("")}</tbody></table>`
}

// One line under the final: what the simulation was tuned to.
function simCalibrationNote(game, calibration, target){
const home = simTeamName(game.home)
const fav = target.margin >= 0 ? home : simTeamName(game.away)
return `Tuned so that ${calibration.games} simulated games average ${simEsc(fav)} by ${Math.abs(target.margin).toFixed(1)} and ${target.total.toFixed(0)} total points (they came out ${Math.abs(calibration.margin).toFixed(1)} and ${calibration.total.toFixed(1)}; ${home} won ${Math.round(calibration.homeWins * 100)}%). This is one of those games.`
}

function simFinalHtml(game){
const f = game.final
const winner = f.home === f.away ? null : f.home > f.away ? "home" : "away"
const name = s => simEsc(simTeamName(game[s]))
const headline = winner
? `${name(winner)} win ${Math.max(f.home, f.away)}-${Math.min(f.home, f.away)}${f.overtime ? " in overtime" : ""}`
: `Tie, ${f.home}-${f.away}`
return `<div class="simFinal"><div class="whatIfHeadline">Final: ${headline}</div></div>`
}

if (typeof module !== "undefined" && module.exports) {
module.exports = {
simRng, simDraw, simFactors, simZone, simPunt, simPick, simSharpen, simShortName, simLineup, simulateGame, simAverage, simCalibrate, simSpot, simClock,
simSummary, simMarginBins, simMarginChartHtml, simSummaryHtml, simPlayerStatsHtml,
simFieldSvg, simScoreboardHtml, simDriveChartHtml, simTickerHtml, simBoxScoreHtml, simCalibrationNote, simFinalHtml,
}
}
