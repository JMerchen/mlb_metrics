// NFL game picks "What if?" panel (nfl_whatif.py's docs/data/nfl_whatif.json):
// swap a team's starting QB or mark players out and the home team's win
// chance is recomputed. The shown probability is moved in log-odds by
// (1) the model's change from the listed QB pairing to the chosen one and
// (2) each player marked out: his position group's measured cost of losing
// a regular, times his snap share ("an average regular" uses the league's
// average share at the position). Pure functions returning numbers or HTML
// strings, so node tests can check them; nfl.js does the DOM wiring.

const WHATIF_GROUP_LABELS = {
OL: "Offensive line", WR: "Receivers", TE: "Tight ends", RB: "Running backs",
DL: "Defensive line", LB: "Linebackers", CB: "Cornerbacks", S: "Safeties",
}

function whatifEsc(value){
return String(value ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;")
}

function whatifLogit(p){
const q = Math.min(Math.max(Number(p), 1e-4), 1 - 1e-4)
return Math.log(q / (1 - q))
}

function whatifSigmoid(x){
return 1 / (1 + Math.exp(-x))
}

function whatifListedQb(team){
const listed = (team.qbs || []).find(q => q.listed) || (team.qbs || [])[0]
return listed ? listed.id : null
}

const WHATIF_AVERAGE_PREFIX = "average:"

// Regulars lost per group for one side: the snap shares of the players
// marked out, an "average regular" counting the league's average share.
function whatifLost(team, outIds, averageShares){
const lost = {}
for(const r of team.regulars || []){
if(outIds.has(r.id)){ lost[r.group] = (lost[r.group] || 0) + Number(r.share) }
}
for(const id of outIds){
if(id.startsWith(WHATIF_AVERAGE_PREFIX)){
const group = id.slice(WHATIF_AVERAGE_PREFIX.length)
lost[group] = (lost[group] || 0) + (Number((averageShares || {})[group]) || 0)
}
}
return lost
}

// state: {homeQb, awayQb, out: {home: Set, away: Set}}
function whatifProbability(game, injuryLogOdds, state, averageShares){
const base = whatifLogit(game.home_win_probability)
const listedKey = `${whatifListedQb(game.home)}|${whatifListedQb(game.away)}`
const chosenKey = `${state.homeQb || whatifListedQb(game.home)}|${state.awayQb || whatifListedQb(game.away)}`
const pairs = game.pairing_log_odds || {}
const qbShift = pairs[chosenKey] !== undefined && pairs[listedKey] !== undefined ? pairs[chosenKey] - pairs[listedKey] : 0
const home = whatifLost(game.home, state.out.home, averageShares)
const away = whatifLost(game.away, state.out.away, averageShares)
let injuryShift = 0
for(const [group, coef] of Object.entries(injuryLogOdds || {})){
injuryShift += Number(coef) * ((home[group] || 0) - (away[group] || 0))
}
return {
base: game.home_win_probability,
probability: whatifSigmoid(base + qbShift + injuryShift),
qbProbability: whatifSigmoid(base + qbShift),
qbShift, injuryShift,
}
}

function whatifPct(p){
return `${Math.round(Number(p) * 100)}%`
}

function whatifPoints(delta){
const pts = Math.round(delta * 100)
return pts === 0 ? "no change" : `${pts > 0 ? "+" : "−"}${Math.abs(pts)} pts`
}

function whatifQbLabel(q){
const rating = q.qb_plus === null || q.qb_plus === undefined ? "no passes yet" : `${q.qb_plus >= 0 ? "+" : "−"}${Math.abs(q.qb_plus).toFixed(2)} EPA/dropback`
return `${q.name}${q.started ? " (started)" : q.listed ? " (listed)" : ""} · ${rating}`
}

// Roughly what one absence is worth in win chance (a log-odds change of x
// is about x/4 near a coin flip).
function whatifWorth(coef, share){
const pts = Math.round(Math.abs(Number(coef) * Number(share)) * 25)
return pts < 1 ? "under 1 pt" : `~${pts} pt${pts === 1 ? "" : "s"}`
}

function whatifCheckbox(side, id, state, label, detail){
return `<label class="whatIfPlayer"><input type="checkbox" data-whatif-out="${side}" value="${whatifEsc(id)}"${state.out[side].has(id) ? " checked" : ""}> ${label} <small>${detail}</small></label>`
}

function whatifTeamHtml(side, team, injuryLogOdds, state, averageShares){
const name = typeof teamName === "function" ? teamName(team.team, "nfl") : team.team
const chosen = (side === "home" ? state.homeQb : state.awayQb) || whatifListedQb(team)
const qbSelect = `<label class="whatIfField">Starting QB
<select data-whatif-qb="${side}">${(team.qbs || []).map(q =>
`<option value="${whatifEsc(q.id)}"${q.id === chosen ? " selected" : ""}>${whatifEsc(whatifQbLabel(q))}</option>`).join("")}</select></label>`
const groups = {}
for(const r of team.regulars || []){ (groups[r.group] = groups[r.group] || []).push(r) }
// Costliest positions first; every position gets an "average regular"
// for a player who isn't listed or whose role isn't known yet.
const order = Object.keys(injuryLogOdds).sort((a, b) => Number(injuryLogOdds[a]) - Number(injuryLogOdds[b]))
const groupHtml = order.map(group => {
const coef = injuryLogOdds[group]
const players = (groups[group] || []).map(r => whatifCheckbox(side, r.id, state,
`${whatifEsc(r.name)}${r.played === false ? ' <span class="whatIfMissed">didn\'t play</span>' : ""}`,
`${Math.round(Number(r.share) * 100)}% of snaps · ${whatifWorth(coef, r.share)}`)).join("")
const average = (averageShares || {})[group]
const averageBox = average ? whatifCheckbox(side, `${WHATIF_AVERAGE_PREFIX}${group}`, state, "<i>An average regular (not listed)</i>",
whatifWorth(coef, average)) : ""
return `<div class="whatIfGroup"><div class="whatIfGroupName">${WHATIF_GROUP_LABELS[group] || group}</div>${players}${averageBox}</div>`
}).join("")
return `<div class="whatIfTeam"><h4>${whatifEsc(name)}${side === "home" ? " (home)" : ""}</h4>${qbSelect}
<div class="whatIfOutLabel">Mark players out</div>${groupHtml}</div>`
}

function whatifResultHtml(game, result){
const home = typeof teamName === "function" ? teamName(game.home.team, "nfl") : game.home.team
const away = typeof teamName === "function" ? teamName(game.away.team, "nfl") : game.away.team
const homeWins = result.probability >= 0.5
const leader = homeWins ? home : away
const leaderP = homeWins ? result.probability : 1 - result.probability
const changed = Math.abs(result.probability - result.base) >= 0.005
const qbDelta = result.qbProbability - result.base
const injuryDelta = result.probability - result.qbProbability
const final = game.final
? `<div class="whatIfFinal">Final: ${whatifEsc(away)} ${game.final.away_score}, ${whatifEsc(home)} ${game.final.home_score}${
game.final.home_score === game.final.away_score ? " (tie)" : (game.final.home_score > game.final.away_score) === homeWins ? " · right side" : " · wrong side"}</div>`
: ""
return `<div class="whatIfResult">
<div class="whatIfHeadline">${whatifEsc(leader)} ${whatifPct(leaderP)}</div>${final}
<span class="chanceTrack"><span class="chanceFill" style="width:${(leaderP * 100).toFixed(1)}%"></span></span>
<div class="whatIfBreakdown">${whatifEsc(home)}${/s$/.test(home) ? "'" : "'s"} chance: ${whatifPct(result.base)} as posted${changed
? ` → <b>${whatifPct(result.probability)}</b> (${[
Math.round(qbDelta * 100) !== 0 ? `QB change ${whatifPoints(qbDelta)}` : "",
Math.round(injuryDelta * 100) !== 0 ? `absences ${whatifPoints(injuryDelta)}` : "",
].filter(Boolean).join(", ") || "under a point"})`
: " (no changes yet)"}</div>
</div>`
}

function whatifGameLabel(game){
const name = code => typeof teamName === "function" ? teamName(code, "nfl") : code
return `${name(game.away.team)} at ${name(game.home.team)}`
}

// A replayed week's state with every listed player who took no snaps
// marked out (the QBs who started are already the listed ones).
function whatifMissedState(game){
const state = whatifEmptyState()
for(const side of ["home", "away"]){
for(const r of game[side].regulars || []){ if(r.played === false){ state.out[side].add(r.id) } }
}
return state
}

// How a replayed week's picks did as posted and with the real absences
// marked: picks on the winning side, and the average chance given to the
// team that won.
function whatifWeekSummary(data){
const games = ((data && data.games) || []).filter(g => g.final && g.final.home_score !== g.final.away_score)
const odds = data.injury_log_odds || {}
const averages = data.average_regular_share || {}
const tally = {posted: {right: 0, winner: 0}, missed: {right: 0, winner: 0}, n: games.length}
for(const g of games){
const homeWon = g.final.home_score > g.final.away_score
for(const [key, state] of [["posted", whatifEmptyState()], ["missed", whatifMissedState(g)]]){
const p = whatifProbability(g, odds, state, averages).probability
const winner = homeWon ? p : 1 - p
tally[key].winner += winner / games.length
if(winner > 0.5){ tally[key].right += 1 }
}
}
return tally
}

function whatifSummaryHtml(data){
const t = whatifWeekSummary(data)
if(!t.n){ return "" }
return `<div class="whatIfSummary">Replayed with only what we knew before kickoff: our picks were on the winning side in
<b>${t.posted.right} of ${t.n}</b> games as posted (average chance given to the winner ${whatifPct(t.posted.winner)}),
and <b>${t.missed.right} of ${t.n}</b> with everyone who actually missed marked out (${whatifPct(t.missed.winner)}).</div>`
}

// options.modes: [{key, label, active}] for the This week / Last week
// switch; a replayed week (games with a final score) also gets its summary
// and a button to mark everyone who actually missed the game.
function whatifPanelHtml(data, gameId, state, options = {}){
const modes = options.modes && options.modes.length > 1
? `<div class="tabGroup whatIfModes">${options.modes.map(m => `<button class="tabButton${m.active ? " active" : ""}" data-whatif-mode="${whatifEsc(m.key)}">${whatifEsc(m.label)}</button>`).join("")}</div>`
: ""
const games = (data && data.games) || []
if(!games.length){ return `${modes}No upcoming games to try yet` }
const game = games.find(g => g.game_id === gameId) || games[0]
const picker = `<label class="whatIfField">Game
<select data-whatif="game">${games.map(g => `<option value="${whatifEsc(g.game_id)}"${g.game_id === game.game_id ? " selected" : ""}>${whatifEsc(whatifGameLabel(g))}</option>`).join("")}</select></label>`
const odds = data.injury_log_odds || {}
const averages = data.average_regular_share || {}
const result = whatifProbability(game, odds, state, averages)
const replay = game.final
? `${whatifSummaryHtml(data)}<button class="whatIfAction" data-whatif-action="missed">Mark everyone who actually missed this game</button>`
: ""
return `${modes}${replay}${picker}${whatifResultHtml(game, result)}
<div class="whatIfTeams">${whatifTeamHtml("away", game.away, odds, state, averages)}${whatifTeamHtml("home", game.home, odds, state, averages)}</div>`
}

function whatifEmptyState(){
return {homeQb: null, awayQb: null, out: {home: new Set(), away: new Set()}}
}

if (typeof module !== "undefined" && module.exports) {
module.exports = {
whatifLost, whatifProbability, whatifPanelHtml, whatifEmptyState, whatifListedQb, whatifPoints,
whatifMissedState, whatifWeekSummary,
}
}
