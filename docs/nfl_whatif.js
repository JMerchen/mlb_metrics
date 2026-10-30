// NFL game picks "What if?" panel (nfl_whatif.py's docs/data/nfl_whatif.json
// and nfl_whatif_last_week.json). Swap a team's starting QB, mark players
// out, and set how well each unit plays: run and pass offense and defense,
// in percent of their normal yards per play (defense: 90% allows 10% more).
// A player out lowers the units his position touches by a default you can
// type over; a unit's percent can be typed over outright. The shown
// probability then moves in log-odds by the model's change from the listed
// QB pairing to the chosen one, plus the units' change in yards turned into
// points (data.points_per_yard) and points into log-odds
// (data.logodds_per_point). Pure functions returning numbers or HTML
// strings, so node tests can check them; nfl.js does the DOM wiring.

const WHATIF_GROUP_LABELS = {
OL: "Offensive line", WR: "Receivers", TE: "Tight ends", RB: "Running backs",
DL: "Defensive line", LB: "Linebackers", CB: "Cornerbacks", S: "Safeties",
}
const WHATIF_UNITS = ["run_offense", "pass_offense", "run_defense", "pass_defense"]
const WHATIF_UNIT_LABELS = {
run_offense: "Run game", pass_offense: "Pass game", run_defense: "Run defense", pass_defense: "Pass defense",
}
const WHATIF_AVERAGE_PREFIX = "average:"

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

function whatifEmptyState(){
return {
homeQb: null, awayQb: null,
out: {home: new Set(), away: new Set()},
// "<player id>|<unit>" -> percent (as a fraction) typed for that player out
playerPct: {home: {}, away: {}},
// unit -> percent (as a fraction) typed for the whole unit
unitPct: {home: {}, away: {}},
}
}

// A game's state as posted: for a game repriced after its final injury
// report, that report's players out and backup QBs; otherwise nothing.
function whatifPostedState(game){
const state = whatifEmptyState()
const report = game && game.injury_report
if(report){
for(const side of ["home", "away"]){
for(const id of (report.out || {})[side] || []){ state.out[side].add(id) }
}
state.homeQb = (report.qb || {}).home || null
state.awayQb = (report.qb || {}).away || null
}
return state
}

// The players a side has marked out, each {id, group, share} - an "average
// regular" counts the league's average share at his position.
function whatifOutPlayers(team, outIds, data){
const players = (team.regulars || []).filter(r => outIds.has(r.id))
for(const id of outIds){
if(id.startsWith(WHATIF_AVERAGE_PREFIX)){
const group = id.slice(WHATIF_AVERAGE_PREFIX.length)
players.push({id, group, share: Number((data.average_regular_share || {})[group]) || 0})
}
}
return players
}

// A player's default percent for a unit: his position's every-down drop,
// times his snap share.
function whatifDefaultPct(data, group, share){
const full = Number((data.group_default_pct || {})[group])
return Number.isFinite(full) ? 1 - (1 - full) * Number(share) : 1
}

function whatifPlayerPct(data, state, side, player, unit){
const typed = state.playerPct[side][`${player.id}|${unit}`]
return typed !== undefined ? typed : whatifDefaultPct(data, player.group, player.share)
}

// {unit: percent} for one side: a typed unit percent, or the product of
// the marked players' percents for that unit.
function whatifUnitPcts(game, side, data, state){
const pcts = {}
for(const unit of WHATIF_UNITS){ pcts[unit] = 1 }
for(const player of whatifOutPlayers(game[side], state.out[side], data)){
for(const unit of (data.group_units || {})[player.group] || []){
pcts[unit] *= whatifPlayerPct(data, state, side, player, unit)
}
}
for(const unit of WHATIF_UNITS){
const typed = state.unitPct[side][unit]
if(typed !== undefined){ pcts[unit] = typed }
}
return pcts
}

// Points a unit's percent is worth to its own team a game: an offense at
// 90% gains 10% fewer yards, a defense at 90% allows 10% more - both cost
// that many yards' worth of points.
function whatifUnitPoints(data, team, unit, pct){
const base = (team.units || {})[unit]
if(!base){ return 0 }
return (pct - 1) * Number(base.ypp) * Number(base.plays) * (Number(data.points_per_yard) || 0)
}

// state: see whatifEmptyState
function whatifProbability(game, data, state){
const base = whatifLogit(game.home_win_probability)
const listedKey = `${whatifListedQb(game.home)}|${whatifListedQb(game.away)}`
const chosenKey = `${state.homeQb || whatifListedQb(game.home)}|${state.awayQb || whatifListedQb(game.away)}`
const pairs = game.pairing_log_odds || {}
const qbShift = pairs[chosenKey] !== undefined && pairs[listedKey] !== undefined ? pairs[chosenKey] - pairs[listedKey] : 0
const net = side => {
const pcts = whatifUnitPcts(game, side, data, state)
return WHATIF_UNITS.reduce((sum, unit) => sum + whatifUnitPoints(data, game[side], unit, pcts[unit]), 0)
}
const margin = net("home") - net("away")
const unitShift = (Number(data.logodds_per_point) || 0) * margin
return {
base: game.home_win_probability,
probability: whatifSigmoid(base + qbShift + unitShift),
qbProbability: whatifSigmoid(base + qbShift),
qbShift, unitShift, margin,
}
}

function whatifPct(p){
return `${Math.round(Number(p) * 100)}%`
}

function whatifPoints(delta){
const pts = Math.round(delta * 100)
return pts === 0 ? "no change" : `${pts > 0 ? "+" : "−"}${Math.abs(pts)} pts`
}

function whatifGamePoints(points){
const r = Math.round(points * 10) / 10
return r === 0 ? "±0.0" : `${r > 0 ? "+" : "−"}${Math.abs(r).toFixed(1)}`
}

function whatifQbLabel(q){
const rating = q.qb_plus === null || q.qb_plus === undefined ? "no passes yet" : `${q.qb_plus >= 0 ? "+" : "−"}${Math.abs(q.qb_plus).toFixed(2)} EPA/dropback`
return `${q.name}${q.started ? " (started)" : q.listed ? " (listed)" : ""} · ${rating}`
}

function whatifPctInput(attr, value, placeholder, label){
return `<span class="whatIfPctWrap"><input class="whatIfPct" type="number" inputmode="numeric" min="0" max="200" step="1" ${attr} value="${value === undefined ? "" : Math.round(value * 100)}" placeholder="${Math.round(placeholder * 100)}" aria-label="${whatifEsc(label)} percent">%</span>`
}

function whatifUnitsHtml(side, game, data, state){
const team = game[side]
const pcts = whatifUnitPcts(game, side, data, state)
return `<div class="whatIfUnits">${WHATIF_UNITS.filter(u => (team.units || {})[u]).map(unit => {
const typed = state.unitPct[side][unit]
const points = whatifUnitPoints(data, team, unit, pcts[unit])
const source = typed !== undefined ? "set by you" : Math.abs(pcts[unit] - 1) < 0.0005 ? "normal" : "from absences"
return `<label class="whatIfUnit"><span>${WHATIF_UNIT_LABELS[unit]}</span>${whatifPctInput(`data-whatif-unit="${side}|${unit}"`, typed, pcts[unit], WHATIF_UNIT_LABELS[unit])}<small>${source} · ${whatifGamePoints(points)} pts</small></label>`
}).join("")}</div>`
}

function whatifPlayerHtml(side, player, data, state, label, detail){
const checked = state.out[side].has(player.id)
const units = (data.group_units || {})[player.group] || []
const typedAny = Object.keys(state.playerPct[side]).some(k => k.startsWith(`${player.id}|`))
const inputs = checked
? `<div class="whatIfPlayerUnits">${units.map(unit => `<span>${WHATIF_UNIT_LABELS[unit]} ${whatifPctInput(
`data-whatif-player="${side}|${whatifEsc(player.id)}|${unit}"`, state.playerPct[side][`${player.id}|${unit}`],
whatifDefaultPct(data, player.group, player.share), WHATIF_UNIT_LABELS[unit])}</span>`).join("")}<small>${typedAny ? "set by you" : "position average"}</small></div>`
: ""
const record = Object.entries(player.record || {})
const recordLine = checked && record.length
? `<div class="whatIfRecord">In his ${record[0][1].games} games missed: ${record.map(([unit, r]) =>
`${WHATIF_UNIT_LABELS[unit].toLowerCase()} ${Math.round(r.pct * 100)}%`).join(", ")} (a handful of games - mostly noise)</div>`
: ""
return `<div class="whatIfPlayerRow"><label class="whatIfPlayer"><input type="checkbox" data-whatif-out="${side}" value="${whatifEsc(player.id)}"${checked ? " checked" : ""}> ${label} <small>${detail}</small></label>${inputs}${recordLine}</div>`
}

function whatifTeamHtml(side, game, data, state){
const team = game[side]
const name = typeof teamName === "function" ? teamName(team.team, "nfl") : team.team
const chosen = (side === "home" ? state.homeQb : state.awayQb) || whatifListedQb(team)
const qbSelect = `<label class="whatIfField">Starting QB
<select data-whatif-qb="${side}">${(team.qbs || []).map(q =>
`<option value="${whatifEsc(q.id)}"${q.id === chosen ? " selected" : ""}>${whatifEsc(whatifQbLabel(q))}</option>`).join("")}</select></label>`
const groups = {}
for(const r of team.regulars || []){ (groups[r.group] = groups[r.group] || []).push(r) }
const defaults = data.group_default_pct || {}
// Costliest positions first; every position gets "an average regular" for
// a player who isn't listed or whose role isn't known yet.
const order = Object.keys(defaults).sort((a, b) => Number(defaults[a]) - Number(defaults[b]))
const groupHtml = order.map(group => {
const players = (groups[group] || []).map(r => whatifPlayerHtml(side, r, data, state,
`${whatifEsc(r.name)}${r.played === false ? ' <span class="whatIfMissed">didn\'t play</span>' : ""}`,
`${Math.round(Number(r.share) * 100)}% of snaps`)).join("")
const average = (data.average_regular_share || {})[group]
const averageRow = average ? whatifPlayerHtml(side, {id: `${WHATIF_AVERAGE_PREFIX}${group}`, group, share: average}, data, state,
"<i>An average regular (not listed)</i>", "") : ""
return `<div class="whatIfGroup"><div class="whatIfGroupName">${WHATIF_GROUP_LABELS[group] || group}</div>${players}${averageRow}</div>`
}).join("")
return `<div class="whatIfTeam"><h4>${whatifEsc(name)}${side === "home" ? " (home)" : ""}</h4>${qbSelect}
<div class="whatIfOutLabel">Units, % of normal yards per play (type to set)</div>${whatifUnitsHtml(side, game, data, state)}
<div class="whatIfOutLabel">Mark players out</div>${groupHtml}</div>`
}

function whatifResultHtml(game, result){
const home = typeof teamName === "function" ? teamName(game.home.team, "nfl") : game.home.team
const away = typeof teamName === "function" ? teamName(game.away.team, "nfl") : game.away.team
const homeWins = result.probability >= 0.5
const leader = homeWins ? home : away
const leaderP = homeWins ? result.probability : 1 - result.probability
const qbDelta = result.qbProbability - result.base
const unitDelta = result.probability - result.qbProbability
const final = game.final
? `<div class="whatIfFinal">Final: ${whatifEsc(away)} ${game.final.away_score}, ${whatifEsc(home)} ${game.final.home_score}${
game.final.home_score === game.final.away_score ? " (tie)" : (game.final.home_score > game.final.away_score) === homeWins ? " · right side" : " · wrong side"}</div>`
: ""
const parts = [
Math.round(qbDelta * 100) !== 0 ? `QB change ${whatifPoints(qbDelta)}` : "",
Math.abs(result.margin) >= 0.05 ? `units ${whatifGamePoints(result.margin)} pts on the margin, ${whatifPoints(unitDelta)}` : "",
].filter(Boolean)
return `<div class="whatIfResult">
<div class="whatIfHeadline">${whatifEsc(leader)} ${whatifPct(leaderP)}</div>${final}
<span class="chanceTrack"><span class="chanceFill" style="width:${(leaderP * 100).toFixed(1)}%"></span></span>
${game.injury_report ? `<div class="whatIfFinal">Posted after the final injury report${game.injury_report.note ? ` (out: ${whatifEsc(game.injury_report.note)})` : ""} - those players and QBs are marked below.</div>` : ""}
<div class="whatIfBreakdown">${whatifEsc(home)}${/s$/.test(home) ? "'" : "'s"} chance: ${whatifPct(result.base)} ${game.injury_report ? "before the injury report" : "as posted"}${parts.length
? ` → <b>${whatifPct(result.probability)}</b> (${parts.join("; ")})`
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
// marked at their defaults: picks on the winning side, and the average
// chance given to the team that won.
function whatifWeekSummary(data){
const games = ((data && data.games) || []).filter(g => g.final && g.final.home_score !== g.final.away_score)
const tally = {posted: {right: 0, winner: 0}, missed: {right: 0, winner: 0}, n: games.length}
for(const g of games){
const homeWon = g.final.home_score > g.final.away_score
for(const [key, state] of [["posted", whatifEmptyState()], ["missed", whatifMissedState(g)]]){
const p = whatifProbability(g, data, state).probability
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
and <b>${t.missed.right} of ${t.n}</b> with everyone who actually missed marked out at the defaults (${whatifPct(t.missed.winner)}).</div>`
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
const result = whatifProbability(game, data, state)
const replay = game.final
? `${whatifSummaryHtml(data)}<button class="whatIfAction" data-whatif-action="missed">Mark everyone who actually missed this game</button>`
: ""
return `${modes}${replay}${picker}${whatifResultHtml(game, result)}
<button class="whatIfAction" data-whatif-action="reset">Reset this game</button>${game.injury_report ? `
<button class="whatIfAction" data-whatif-action="clear">Without the injury report</button>` : ""}
<div class="whatIfTeams">${whatifTeamHtml("away", game, data, state)}${whatifTeamHtml("home", game, data, state)}</div>`
}

// A state as a string, to tell whether anything was changed since.
function whatifStateKey(state){
const sorted = set => [...set].sort()
return JSON.stringify({
homeQb: state.homeQb, awayQb: state.awayQb,
out: {home: sorted(state.out.home), away: sorted(state.out.away)},
playerPct: state.playerPct, unitPct: state.unitPct,
})
}

// What a state changes from the game as listed, in words: QB swaps,
// players out and units typed over ("Packers QB Tyrod Taylor", "Packers
// without Josh Jacobs", "Panthers pass defense 80%"). The units the
// absences lower go unsaid - the names already say it.
function whatifSettingsSummary(game, data, state){
const out = []
for(const side of ["away", "home"]){
const team = game[side]
const name = typeof teamName === "function" ? teamName(team.team, "nfl") : team.team
const qb = side === "home" ? state.homeQb : state.awayQb
if(qb && qb !== whatifListedQb(team)){
const q = (team.qbs || []).find(x => x.id === qb)
out.push(`${name} QB ${q ? q.name : qb}`)
}
const missing = whatifOutPlayers(team, state.out[side], data).map(p => p.name || `an average ${WHATIF_GROUP_LABELS[p.group] ? WHATIF_GROUP_LABELS[p.group].toLowerCase() : p.group} regular`)
if(missing.length){ out.push(`${name} without ${missing.join(", ")}`) }
for(const unit of WHATIF_UNITS){
const typed = state.unitPct[side][unit]
if(typed !== undefined){ out.push(`${name} ${WHATIF_UNIT_LABELS[unit].toLowerCase()} ${Math.round(typed * 100)}%`) }
}
}
return out
}

// A typed percent ("" clears it) as a fraction, or undefined.
function whatifParsePct(text){
const value = String(text ?? "").trim()
if(value === ""){ return undefined }
const n = Number(value)
return Number.isFinite(n) && n >= 0 ? n / 100 : undefined
}

if (typeof module !== "undefined" && module.exports) {
module.exports = {
whatifProbability, whatifPanelHtml, whatifEmptyState, whatifListedQb, whatifPoints, whatifUnitPcts,
whatifUnitPoints, whatifDefaultPct, whatifMissedState, whatifWeekSummary, whatifParsePct, whatifOutPlayers,
whatifPostedState, whatifStateKey, whatifSettingsSummary,
}
}
