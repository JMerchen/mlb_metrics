// NFL game picks "What if?" panel (nfl_whatif.py's docs/data/nfl_whatif.json):
// swap a team's starting QB or mark regulars out and the home team's win
// chance is recomputed. The shown probability is moved in log-odds by
// (1) the model's change from the listed QB pairing to the chosen one and
// (2) each position group's measured cost of an absence, scaled by the
// snap share of the regulars marked out. Pure functions returning numbers
// or HTML strings, so node tests can check them; nfl.js does the DOM wiring.

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

// Each group's missing share for one side: snap share of the regulars
// marked out over all the group's regulars' (0 to 1).
function whatifGroupShares(team, outIds){
const totals = {}
const missing = {}
for(const r of team.regulars || []){
totals[r.group] = (totals[r.group] || 0) + Number(r.share)
if(outIds.has(r.id)){ missing[r.group] = (missing[r.group] || 0) + Number(r.share) }
}
const shares = {}
for(const group of Object.keys(totals)){
shares[group] = totals[group] > 0 ? (missing[group] || 0) / totals[group] : 0
}
return shares
}

// state: {homeQb, awayQb, out: {home: Set, away: Set}}
function whatifProbability(game, injuryLogOdds, state){
const base = whatifLogit(game.home_win_probability)
const listedKey = `${whatifListedQb(game.home)}|${whatifListedQb(game.away)}`
const chosenKey = `${state.homeQb || whatifListedQb(game.home)}|${state.awayQb || whatifListedQb(game.away)}`
const pairs = game.pairing_log_odds || {}
const qbShift = pairs[chosenKey] !== undefined && pairs[listedKey] !== undefined ? pairs[chosenKey] - pairs[listedKey] : 0
const home = whatifGroupShares(game.home, state.out.home)
const away = whatifGroupShares(game.away, state.out.away)
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
return `${q.name}${q.listed ? " (listed)" : ""} · ${rating}`
}

function whatifTeamHtml(side, team, injuryLogOdds, state){
const name = typeof teamName === "function" ? teamName(team.team, "nfl") : team.team
const chosen = (side === "home" ? state.homeQb : state.awayQb) || whatifListedQb(team)
const qbSelect = `<label class="whatIfField">Starting QB
<select data-whatif-qb="${side}">${(team.qbs || []).map(q =>
`<option value="${whatifEsc(q.id)}"${q.id === chosen ? " selected" : ""}>${whatifEsc(whatifQbLabel(q))}</option>`).join("")}</select></label>`
const groups = {}
for(const r of team.regulars || []){ (groups[r.group] = groups[r.group] || []).push(r) }
const order = Object.keys(groups).sort((a, b) => (Number(injuryLogOdds[a]) || 0) - (Number(injuryLogOdds[b]) || 0))
const groupHtml = order.map(group => {
const players = groups[group].map(r => `<label class="whatIfPlayer"><input type="checkbox" data-whatif-out="${side}" value="${whatifEsc(r.id)}"${state.out[side].has(r.id) ? " checked" : ""}> ${whatifEsc(r.name)} <small>${Math.round(Number(r.share) * 100)}% of snaps</small></label>`).join("")
return `<div class="whatIfGroup"><div class="whatIfGroupName">${WHATIF_GROUP_LABELS[group] || group}</div>${players}</div>`
}).join("")
return `<div class="whatIfTeam"><h4>${whatifEsc(name)}${side === "home" ? " (home)" : ""}</h4>${qbSelect}
<div class="whatIfOutLabel">Mark regulars out</div>${groupHtml || "<p class=\"sectionIntro\">No snap counts yet.</p>"}</div>`
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
return `<div class="whatIfResult">
<div class="whatIfHeadline">${whatifEsc(leader)} ${whatifPct(leaderP)}</div>
<span class="chanceTrack"><span class="chanceFill" style="width:${(leaderP * 100).toFixed(1)}%"></span></span>
<div class="whatIfBreakdown">${whatifEsc(home)}${/s$/.test(home) ? "'" : "'s"} chance: ${whatifPct(result.base)} as posted${changed
? ` → <b>${whatifPct(result.probability)}</b> (QB change ${whatifPoints(qbDelta)}, absences ${whatifPoints(injuryDelta)})`
: " (no changes yet)"}</div>
</div>`
}

function whatifGameLabel(game){
const name = code => typeof teamName === "function" ? teamName(code, "nfl") : code
return `${name(game.away.team)} at ${name(game.home.team)}`
}

function whatifPanelHtml(data, gameId, state){
const games = (data && data.games) || []
if(!games.length){ return "No upcoming games to try yet" }
const game = games.find(g => g.game_id === gameId) || games[0]
const picker = `<label class="whatIfField">Game
<select data-whatif="game">${games.map(g => `<option value="${whatifEsc(g.game_id)}"${g.game_id === game.game_id ? " selected" : ""}>${whatifEsc(whatifGameLabel(g))}</option>`).join("")}</select></label>`
const result = whatifProbability(game, data.injury_log_odds, state)
return `${picker}${whatifResultHtml(game, result)}
<div class="whatIfTeams">${whatifTeamHtml("away", game.away, data.injury_log_odds || {}, state)}${whatifTeamHtml("home", game.home, data.injury_log_odds || {}, state)}</div>`
}

function whatifEmptyState(){
return {homeQb: null, awayQb: null, out: {home: new Set(), away: new Set()}}
}

if (typeof module !== "undefined" && module.exports) {
module.exports = {
whatifGroupShares, whatifProbability, whatifPanelHtml, whatifEmptyState, whatifListedQb, whatifPoints,
}
}
