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
const SIM_SIDES = ["home", "away"]

function simEsc(value){
return String(value ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;")
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

// opts: {seed, edge, scale, unitPcts: {home: {unit: pct}, away: {...}}}
// Returns {home, away, plays, drives, box, final}. A play: {qtr, clock (left
// in the quarter after it), poss, down, togo, ball (before it, from the
// offense's goal), end (after it), kind, yards, text, home, away (score
// after it), drive, scoring}.
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

const spot = (b, side) => simSpot(b, code[side], code[other(side)])
const lead = side => score[side] - score[other(side)]
const record = (kind, yards, text, extra = {}) => {
const play = {
qtr, clock, poss, down, togo, ball: extra.ball ?? ball, end: extra.end ?? ball, kind, yards: Math.round(yards), text,
home: score.home, away: score.away, drive: drives.length - 1, scoring: extra.scoring || null,
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
// Runs the clock off a play (the main loop ends the period at 0:00).
const tick = seconds => {
const used = Math.min(clock, seconds)
clock -= used
box[poss].possession += used
if(drive){ drive.seconds += used }
}
const kickoff = (kicker, how = "Kickoff") => {
if(over){ return }
const receiver = other(kicker)
const start = simDraw(lg.kickoff_start, probs, rng())
startDrive(receiver, start, how)
tick(5)
record("kickoff", 0, `${code[kicker]} kicks off; ${code[receiver]} start at ${spot(ball, receiver)}`)
}
const scoreTouchdown = how => {
score[poss] += 6
box[poss][how === "pass" ? "pass_td" : "rush_td"] += 1
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
kickoff(rng() < 0.5 ? "home" : "away", "Overtime kickoff")
return
}
over = true
}

kickoff(other(firstReceiver), "Opening kickoff")

let guard = 0
while(!over && guard < 400){
guard += 1
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

// Kneel out a won game.
if(qtr >= 4 && margin > 0 && clock <= 42 * (5 - down)){
drive.plays += 1
tick(40)
record("kneel", -1, `${code[poss]} kneel down`, {end: Math.max(1, ball - 1)})
ball = Math.max(1, ball - 1)
b.plays += 1
b.rush_att += 1
b.rush_yards -= 1
b.yards -= 1
drive.yards -= 1
down += 1
togo += 1
if(down > 4){
turnover(poss, ball, "Downs")
}
continue
}

// Field goal as time runs out, or on 4th down.
const lastKick = clock <= 12 && fgDistance <= 58 && (qtr === 2 || ((qtr === 4 || qtr === 5) && margin <= 0 && margin >= -3))
let fourth = null
if(down === 4 && !lastKick){
const needTd = late && margin < -3
const desperate = qtr >= 4 && clock <= 150 && margin < 0
if(desperate && !(margin >= -3 && fgDistance <= 55)){ fourth = "go" }
else if(needTd && ball >= 40){ fourth = "go" }
else if(togo <= 1 && ball >= 35){ fourth = "go" }
else if(fgDistance <= 57 && fgChance >= 0.35){ fourth = "fg" }
else if(togo <= 3 && ball >= 55){ fourth = "go" }
else{ fourth = "punt" }
if(fourth === "fg" && qtr >= 4 && clock <= 150 && margin < -3){ fourth = "go" }
}
if(lastKick || fourth === "fg"){
b.fg_att += 1
const made = rng() < fgChance
tick(6)
if(made){
score[poss] += 3
b.fg_made += 1
record("field_goal", 0, `${code[poss]} ${Math.round(fgDistance)}-yard field goal is GOOD`, {scoring: "FG"})
endDrive("Field goal")
kickoff(poss)
}else{
const at = Math.max(20, 100 - (ball - 7))
record("field_goal", 0, `${code[poss]} ${Math.round(fgDistance)}-yard field goal is NO GOOD`)
endDrive("Missed FG")
startDrive(def, at, "Missed FG")
}
continue
}
if(fourth === "punt"){
b.punts += 1
tick(8)
let at
let net = Math.round(simDraw(lg.punt_net, probs, rng()))
if(ball + net >= 100 || rng() < (lg.punt_touchback_rate || 0.07) * (ball > 50 ? 2 : 0.5)){
at = 20
net = 80 - ball
}else{
at = Math.max(1, 100 - (ball + Math.max(net, 0)))
}
record("punt", 0, `${code[poss]} punt, ${code[def]} take over at ${spot(at, def)}`)
endDrive("Punt")
startDrive(def, at, "Punt")
continue
}

// A play from scrimmage: run or pass.
let passRate = p.neutral_pass_rate
if(down === 3 || down === 4){ passRate = togo <= 2 ? 0.4 : togo <= 4 ? 0.65 : 0.85 }
else if(down === 2 && togo >= 8){ passRate += 0.1 }
else if(down === 2 && togo <= 3){ passRate -= 0.1 }
if(hurry){ passRate += 0.3 }
if(qtr >= 4 && margin >= 9){ passRate -= 0.25 }
if(qtr >= 3 && margin <= -14){ passRate += 0.15 }
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
let clockUsed = hurry ? 16 : 38
let lost = null
if(isPass){
if(rng() < p.sack_rate){
kind = "sack"
yards = Math.min(-1, Math.round(lg.sack_yards * (0.4 + 1.2 * rng())))
b.sacks += 1
b.sack_yards -= yards
text = `${code[poss]} sacked for a loss of ${-yards}`
}else{
b.pass_att += 1
if(rng() < p.int_rate){
kind = "interception"
const air = Math.round(5 + 25 * rng())
const caught = Math.min(ball + air, 99)
const ret = Math.round(15 * rng() * rng())
b.ints += 1
b.turnovers += 1
lost = {at: Math.min(99, Math.max(1, 100 - caught + ret)), result: "Interception"}
text = `${code[poss]} pass INTERCEPTED by ${code[def]}`
clockUsed = 8
}else if(rng() < p.comp_rate){
kind = "pass"
const raw = simDraw(p.completion_yards, probs, rng())
yards = Math.min(100 - ball, Math.round(raw > 0 ? raw * factors[poss].pass : raw))
b.pass_comp += 1
b.pass_yards += yards
text = `${code[poss]} pass complete for ${yards} yard${Math.abs(yards) === 1 ? "" : "s"}`
if(rng() < p.catch_fumble_rate && ball + yards < 100){
lost = {at: Math.min(99, Math.max(1, 100 - (ball + yards))), result: "Fumble"}
text += `, FUMBLE recovered by ${code[def]}`
}
}else{
kind = "incomplete"
text = `${code[poss]} pass incomplete`
clockUsed = 6
}
}
}else{
kind = "run"
const raw = simDraw(p.rush_yards, probs, rng())
yards = Math.min(100 - ball, Math.round(raw > 0 ? raw * factors[poss].run : raw))
b.rush_att += 1
b.rush_yards += yards
text = `${code[poss]} run for ${yards} yard${Math.abs(yards) === 1 ? "" : "s"}`
if(rng() < p.run_fumble_rate && ball + yards < 100){
lost = {at: Math.min(99, Math.max(1, 100 - (ball + yards))), result: "Fumble"}
text += `, FUMBLE recovered by ${code[def]}`
}
}
// Total yards are net of sacks, as in a box score.
b.yards += yards
drive.yards += yards
if(lost){
b.fumbles_lost += lost.result === "Fumble" ? 1 : 0
b.turnovers += lost.result === "Fumble" ? 1 : 0
tick(clockUsed)
ball += yards
record(kind, yards, `${situation}: ${text}`, {ball: from, end: ball})
turnover(poss, 100 - lost.at, lost.result)
continue
}
ball += yards
if(ball >= 100){
ball = 100
const after = scoreTouchdown(kind === "pass" ? "pass" : "run")
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
tick(clockUsed)
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
if(down > 4){
turnover(poss, 100 - ball, "Downs")
}
}
}
for(const side of SIM_SIDES){ box[side].points = score[side] }
return {
home: homeCode, away: awayCode, plays, drives, box,
final: {home: score.home, away: score.away, overtime: qtr >= 5},
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
// if? unit percents) applied and opts.games games (default 300). Broyden's
// method: slopes measured once by nudging each, then updated from every
// step taken; the closest try wins if it runs out of steps.
function simCalibrate(profiles, homeCode, awayCode, targetMargin, targetTotal, opts = {}){
const n = opts.games || SIM_CALIBRATION_GAMES
const at = (edge, scale) => {
const r = simAverage(profiles, homeCode, awayCode, {edge, scale, unitPcts: opts.unitPcts}, n)
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

// Field picture: the offense's ball spot, the first-down line and the end
// zones, home driving to the right (so home's own end zone is on the left).
function simFieldSvg(play, homeCode, awayCode){
const x = yards => 40 + yards * 5.2
const toRight = play.poss === "home"
const at = b => x(toRight ? b : 100 - b)
const ball = at(play.end ?? play.ball)
const marker = play.kind === "kickoff" || play.kind === "punt" || play.kind === "field_goal" || play.scoring
? null : at(Math.min(100, (play.end ?? play.ball) + (play.first_down ? 10 : Math.max(0, play.togo - play.yards))))
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
const side = s => `<div class="simTeam${poss === s ? " simHasBall" : ""}"><span class="simTeamName"><span class="simLong">${simEsc(simTeamName(game[s]))}</span><span class="simShort">${simEsc(game[s])}</span></span><span class="simPoints">${p[s]}</span></div>`
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
const cls = over && (result === "Touchdown" || result === "Field goal") ? " class=\"simScoreDrive\"" : !over ? " class=\"simLiveDrive\"" : ""
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
`<li${p.scoring ? " class=\"simScoring\"" : ""}><span class="simWhen">${simQuarterLabel(p.qtr)} ${simClock(p.clock)}</span> ${simEsc(p.text)}</li>`).join("")}</ol>`
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
["Touchdowns (pass, rush)", `${a.pass_td}, ${a.rush_td}`, `${h.pass_td}, ${h.rush_td}`],
["Field goals", pct(a.fg_made, a.fg_att), pct(h.fg_made, h.fg_att)],
["3rd down", pct(a.third_conv, a.third_att), pct(h.third_conv, h.third_att)],
["4th down", pct(a.fourth_conv, a.fourth_att), pct(h.fourth_conv, h.fourth_att)],
["Turnovers (INT, fumbles)", `${a.turnovers} (${a.ints}, ${a.fumbles_lost})`, `${h.turnovers} (${h.ints}, ${h.fumbles_lost})`],
["Punts", a.punts, h.punts],
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
simRng, simDraw, simFactors, simulateGame, simAverage, simCalibrate, simSpot, simClock,
simFieldSvg, simScoreboardHtml, simDriveChartHtml, simTickerHtml, simBoxScoreHtml, simCalibrationNote, simFinalHtml,
}
}
