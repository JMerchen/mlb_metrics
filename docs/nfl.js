// NFL page - separate from app.js/age-curves.js since this page is
// standalone (see docs/nfl.html). loadCSV/buildTable are copied (not
// shared - no build step in docs/, same convention every other page's JS
// already documents).

async function loadCSV(path){
const response = await fetch(`${path}?t=${Date.now()}`, {cache:"no-store"})
if(!response.ok){ throw new Error(`Failed to load ${path}: ${response.status}`) }
const text = await response.text()
return Papa.parse(text, {header:true, skipEmptyLines:true}).data
}

function buildTable(data, id, limit=null){
if(limit){ data = data.slice(0, limit) }
const el = document.getElementById(id)
if(!data.length){ el.innerHTML = "No data"; return }
let html = "<table><tr>"
Object.keys(data[0]).forEach(k=>{ html += `<th>${k}</th>` })
html += "</tr>"
data.forEach(row=>{
html += "<tr>"
Object.values(row).forEach(v=>{ html += `<td>${v}</td>` })
html += "</tr>"
})
html += "</table>"
el.innerHTML = html
}

// Copied from app.js (same "no shared build step in docs/" convention
// as loadCSV/buildTable above).
function ciLabel(low, high){
if(low === undefined || low === "" || high === undefined || high === ""){
return ""
}
return `95% CI ${(Number(low) * 100).toFixed(0)}–${(Number(high) * 100).toFixed(0)}%`
}

function significanceLabel(p){
if(p === undefined || p === "" || Number.isNaN(Number(p))){
return ""
}
const value = Number(p)
return value < 0.05
? `significant (p=${value.toFixed(3)})`
: `not significant (p=${value.toFixed(2)})`
}

const NFL_TABS = ["picks", "stats", "playerprops", "prospects"]
// Old tab names in links: "#gamepicks" (now Picks), "#teamunits" (now
// Stats), "#draftboard" (now Prospects).
const NFL_TAB_ALIASES = {gamepicks: "picks", teamunits: "stats", draftboard: "prospects"}

// Which tab the page opens on: a "#stats"-style link, else Picks.
function nflDefaultTab(hash){
const name = (hash || "").replace("#", "")
const tab = NFL_TAB_ALIASES[name] || name
return NFL_TABS.includes(tab) ? tab : "picks"
}

function selectNflTab(tab){
document.querySelectorAll("#nflTabs .tabButton").forEach(btn=>{
btn.classList.toggle("active", btn.dataset.nflTab === tab)
})
// Keep the open tab in the address, so a link or a refresh lands on it.
if(window.history && window.history.replaceState){
window.history.replaceState(null, "", `#${tab}`)
}
document.getElementById("gamePicksSection").style.display = tab === "picks" ? "" : "none"
document.getElementById("draftBoardSection").style.display = tab === "prospects" ? "" : "none"
document.getElementById("playerPropsSection").style.display = tab === "playerprops" ? "" : "none"
document.getElementById("teamUnitsSection").style.display = tab === "stats" ? "" : "none"
}

// --- NFL Automated Game Picks (nfl_game_evaluation.build_game_picks_export's
// own docs/data/nfl_game_picks_*.csv exports) - direct structural mirror
// of app.js's own game-picks section, adapted for game_id/season/week in
// place of game_pk/date-only, and "This Week's Picks" (the most recent
// real week logged, not necessarily "today" the way MLB's daily cadence
// means) in place of "Today's Picks". ---

async function loadNflGamePicks(){

let summary = []
let picks = []

try{
summary = await loadCSV("./data/nfl_game_picks_summary.csv")
}catch(e){
console.log("no nfl_game_picks_summary.csv yet", e)
}

try{
picks = await loadCSV("./data/nfl_game_picks_picks.csv")
}catch(e){
console.log("no nfl_game_picks_picks.csv yet", e)
}

renderNflGamePickStats(summary)
renderNflTodaysGamePicks(picks)
renderNflGamePickHistory(picks)
loadNflWhatIf()

}

// "What if?" panel (docs/nfl_whatif.js): this week's games, or last week's
// replayed as a test case; per view, the chosen game, QBs and absences.
const NFL_WHATIF_MODES = [
{key: "this", label: "This week", file: "nfl_whatif"},
{key: "last", label: "Last week (with results)", file: "nfl_whatif_last_week"},
]
const nflWhatIf = {mode: "this", data: {}, game: {}, state: {}}

async function loadNflWhatIfFile(file){
if(window.SITE_DATA && window.SITE_DATA[file]){ return window.SITE_DATA[file] }
const response = await fetch(`./data/${file}.json?t=${Date.now()}`, {cache:"no-store"})
if(!response.ok){ throw new Error(`HTTP ${response.status}`) }
return response.json()
}

async function loadNflWhatIf(){

const el = document.getElementById("nflWhatIf")

for(const m of NFL_WHATIF_MODES){
try{
nflWhatIf.data[m.key] = await loadNflWhatIfFile(m.file)
nflWhatIf.state[m.key] = whatifPostedState((nflWhatIf.data[m.key].games || [])[0])
}catch(e){
console.log(`no ${m.file}.json yet`, e)
}
}
if(!nflWhatIf.data.this && !nflWhatIf.data.last){
el.innerHTML = "No upcoming games to try yet"
return
}
if(!nflWhatIf.data.this){ nflWhatIf.mode = "last" }

renderNflWhatIf()
el.addEventListener("change", onNflWhatIfChange)
el.addEventListener("click", onNflWhatIfClick)

}

function renderNflWhatIf(focusSelector){

const el = document.getElementById("nflWhatIf")
const mode = nflWhatIf.mode
const modes = NFL_WHATIF_MODES.filter(m => nflWhatIf.data[m.key]).map(m => ({...m, active: m.key === mode}))
el.innerHTML = whatifPanelHtml(nflWhatIf.data[mode], nflWhatIf.game[mode], nflWhatIf.state[mode], {modes})
// Re-rendering replaces the control that changed; keep keyboard focus on it.
const again = focusSelector ? el.querySelector(focusSelector) : null
if(again){ again.focus() }

}

function onNflWhatIfChange(event){

const t = event.target
const mode = nflWhatIf.mode
const state = nflWhatIf.state[mode]
let focus = null
if(t.dataset.whatif === "game"){
nflWhatIf.game[mode] = t.value
nflWhatIf.state[mode] = whatifPostedState(nflWhatIfGameData(mode))
focus = '[data-whatif="game"]'
}else if(t.dataset.whatifQb){
state[t.dataset.whatifQb === "home" ? "homeQb" : "awayQb"] = t.value
focus = `[data-whatif-qb="${t.dataset.whatifQb}"]`
}else if(t.dataset.whatifOut){
const out = state.out[t.dataset.whatifOut]
if(t.checked){ out.add(t.value) }else{ out.delete(t.value) }
focus = `[data-whatif-out="${t.dataset.whatifOut}"][value="${CSS.escape(t.value)}"]`
}else if(t.dataset.whatifUnit){
// A unit's percent typed over (blank goes back to the absences' percent).
const [side, unit] = t.dataset.whatifUnit.split("|")
const pct = whatifParsePct(t.value)
if(pct === undefined){ delete state.unitPct[side][unit] }else{ state.unitPct[side][unit] = pct }
focus = `[data-whatif-unit="${t.dataset.whatifUnit}"]`
}else if(t.dataset.whatifPlayer){
// A player's percent for one unit (blank goes back to his default).
const [side, id, unit] = t.dataset.whatifPlayer.split("|")
const pct = whatifParsePct(t.value)
if(pct === undefined){ delete state.playerPct[side][`${id}|${unit}`] }else{ state.playerPct[side][`${id}|${unit}`] = pct }
focus = `[data-whatif-player="${CSS.escape(t.dataset.whatifPlayer)}"]`
}else{
return
}
renderNflWhatIf(focus)

}

// The game the view has open (its first until one is picked).
function nflWhatIfGameData(mode){
const games = (nflWhatIf.data[mode] || {}).games || []
return games.find(g => g.game_id === nflWhatIf.game[mode]) || games[0]
}

function onNflWhatIfClick(event){

const t = event.target.closest("button")
if(!t){ return }
if(t.dataset.whatifMode){
nflWhatIf.mode = t.dataset.whatifMode
renderNflWhatIf(`[data-whatif-mode="${t.dataset.whatifMode}"]`)
}else if(t.dataset.whatifAction === "missed"){
const mode = nflWhatIf.mode
nflWhatIf.state[mode] = whatifMissedState(nflWhatIfGameData(mode))
renderNflWhatIf('[data-whatif-action="missed"]')
}else if(t.dataset.whatifAction === "reset"){
// Back to the game as posted (with its injury report, if it has one).
nflWhatIf.state[nflWhatIf.mode] = whatifPostedState(nflWhatIfGameData(nflWhatIf.mode))
renderNflWhatIf('[data-whatif-action="reset"]')
}else if(t.dataset.whatifAction === "clear"){
nflWhatIf.state[nflWhatIf.mode] = whatifEmptyState()
renderNflWhatIf('[data-whatif-action="clear"]')
}

}

function renderNflGamePickStats(summary){

const el = document.getElementById("nflGamePickStats")

if(!summary.length){
el.innerHTML = "No picks tracked yet"
return
}

const s = summary[0]

const stat = (value, label, sub) =>
`<div class="streakStat"><div class="value">${value}</div><div class="label">${label}</div>${sub ? `<div class="sub">${sub}</div>` : ""}</div>`
const pct0 = v=>v === undefined || v === "" || Number.isNaN(Number(v)) ? "-" : Math.round(Number(v) * 100) + "%"

// Plain-language headline: our picks next to Vegas's on the same games,
// then the bet record as "won N of M" and units up or down.
let html = ""
const n = Number(s.model_vs_market_n) || 0
if(n){
html += stat(`${Math.round(Number(s.model_pick_accuracy) * n)} of ${n}`, "Our picks right", "games with a Vegas line")
html += stat(`${Math.round(Number(s.market_pick_accuracy) * n)} of ${n}`, "Vegas's picks right", "same games")
}
const bets = Number(s.n_bets_advised) || 0
if(bets){
const units = Number(s.total_profit_units)
html += stat(`${s.bets_won} of ${bets}`, "Bets won", `the odds said about ${pct0(s.odds_implied_win_rate)} would win`)
html += `<div class="streakStat"><div class="value" style="color:${units >= 0 ? "var(--success)" : "var(--danger)"}">${units >= 0 ? "+" : "−"}${Math.abs(units).toFixed(1)}</div><div class="label">Units won or lost</div><div class="sub">${bets < 100 ? "too few bets to judge yet" : ""}</div></div>`
}else{
html += stat("-", "Bets won", "no bets settled yet")
}
const totals = Number(s.points_total_n) || 0
if(totals){
html += stat(`${Number(s.points_total_mae).toFixed(1)} pts`, "Our point totals missed by", `on average over ${totals} games &middot; Vegas missed by ${Number(s.market_points_total_mae).toFixed(1)}`)
}
el.innerHTML = html

}

function renderNflTodaysGamePicks(picks){

const el = document.getElementById("nflTodaysGamePicks")

if(!picks.length){
el.innerHTML = "No game picks yet"
return
}

const latestWeek = picks
.map(p=>`${p.season}_${String(p.week).padStart(2,"0")}`)
.sort()
.slice(-1)[0]

const thisWeek = picks
.filter(p=>`${p.season}_${String(p.week).padStart(2,"0")}` === latestWeek)
.sort((a,b)=>Number(b.predicted_probability) - Number(a.predicted_probability))

if(!thisWeek.length){
el.innerHTML = "No games this week"
return
}

const statusLabels = {
win: "We were right",
loss: "We were wrong",
not_played: "Postponed",
pending: "Not played yet",
}
const pct0 = v=>v !== undefined && v !== null && v !== "" && !Number.isNaN(Number(v)) ? Math.round(Number(v) * 100) + "%" : ""

el.innerHTML = thisWeek
.map(p=>{

const away = teamName(p.away_team, "nfl")
const home = teamName(p.home_team, "nfl")
const pick = teamName(p.predicted_winner, "nfl")
const matchup = p.predicted_winner === p.home_team ? `${away} at <b>${home}</b>` : `<b>${away}</b> at ${home}`
const vegas = pct0(p.market_predicted_winner_probability)

const betUnits = Number(p.bet_units)
const betAdvised = betUnits > 0
const betLine = betAdvised
? `<div class="pickStatus">Bet ${teamName(p.bet_team, "nfl")} &middot; ${betUnits.toFixed(1)} units</div>`
: ""

const report = p.injury_report === "after" ? "after" : "before"
const changed = report === "after" && p.before_report_winner
&& (p.before_report_winner !== p.predicted_winner || Math.round(Number(p.before_report_probability) * 100) !== Math.round(Number(p.predicted_probability) * 100))
const reportNote = report === "after"
? `<div class="reportNote"><b>After the final injury report</b>${p.injury_note ? ` &middot; Out: ${mlbEsc(p.injury_note)}` : " &middot; no regulars out"}${changed ? ` &middot; before it: ${teamName(p.before_report_winner, "nfl")} ${pct0(p.before_report_probability)}` : ""}</div>`
: `<div class="reportNote"><b>Before the final injury report</b></div>`

return `
<div class="pickCard ${p.status} report-${report}${betAdvised ? " recommended" : ""}">
<div class="pickName">${matchup}</div>
<div class="pickProb">We say ${pick} ${pct0(p.predicted_probability)}${vegas ? ` &middot; Vegas ${vegas}` : ""}</div>
${totalPointsText(p) ? `<div class="pickProb">${totalPointsText(p)}</div>` : ""}
${teamEpaHtml(p, "nfl")}
${reportNote}
<div class="pickStatus">${statusLabels[p.status] || p.status}</div>
${betLine}
</div>
`

})
.join("")

}

function renderNflGamePickHistory(picks){

const sorted = picks
.slice()
.sort((a,b)=>b.date.localeCompare(a.date) || Number(b.predicted_probability) - Number(a.predicted_probability))

const results = {
win: "✓ Right",
loss: "✗ Wrong",
not_played: "Postponed",
pending: "Not played yet",
}
const pct0 = v=>v !== undefined && v !== null && v !== "" && !Number.isNaN(Number(v)) ? Math.round(Number(v) * 100) + "%" : "-"

const shown = sorted.slice(0, 100)
const anyBets = shown.some(p=>Number(p.bet_units) > 0)
const anyTotals = shown.some(p=>p.predicted_total !== undefined && p.predicted_total !== "")
const betCell = p=>{
if(!(Number(p.bet_units) > 0)){ return "-" }
const profit = Number(p.bet_profit_units)
const settled = p.bet_profit_units !== undefined && p.bet_profit_units !== "" && !Number.isNaN(profit)
? (profit >= 0 ? `, won ${profit.toFixed(1)}` : `, lost ${Math.abs(profit).toFixed(1)}`)
: ""
return `${teamName(p.bet_team, "nfl")} ${Number(p.bet_units).toFixed(1)} units${settled}`
}

const formatted = shown.map(p=>{
const row = {
"Week": p.week,
"Game": `${teamName(p.away_team, "nfl")} at ${teamName(p.home_team, "nfl")}`,
"Our pick": teamName(p.predicted_winner, "nfl"),
"We said": pct0(p.predicted_probability),
"Vegas said": pct0(p.market_predicted_winner_probability),
"Result": results[p.status] || p.status,
}
if(anyTotals){
const num = v=>v !== undefined && v !== "" && !Number.isNaN(Number(v)) ? Number(v) : null
row["Total: we said"] = num(p.predicted_total) === null ? "-" : Math.round(num(p.predicted_total))
row["Total: Vegas"] = num(p.market_total) === null ? "-" : num(p.market_total)
row["Total: actual"] = num(p.actual_total) === null ? "-" : num(p.actual_total)
}
if(anyBets){ row["Bet"] = betCell(p) }
return row
})

buildTable(
formatted,
"nflGamePickHistoryTable"
)

}

// Real NFL draft-eligible college consensus board - rendering logic
// (prepareConsensusColumns/formatConsensusValue/buildConsensusTable)
// lives in docs/consensus_board.js (loaded via its own script tag
// before this file - see that file's own comment for why), shared with
// docs/prospects.js's own two MLB boards rather than reimplemented here.
async function loadNflDraftBoard(){
try{
const draftBoard = await loadCSV("./data/nfl_draft_board.csv")
buildConsensusTable(draftBoard, "nflDraftBoardTable",
"The big boards for the next NFL draft haven't been published yet - this board fills in once they are.")
}catch(e){
console.log("no nfl_draft_board.csv yet", e)
buildConsensusTable([], "nflDraftBoardTable")
}
}

// Real, matchup-based player prop rankings - rendering logic
// (formatPlayerPropsValue/buildPlayerPropsTable) lives in
// docs/player_props.js (loaded via its own script tag before this
// file - see that file's own comment for why), a different real shape
// from a consensus board's own "multiple sources agree on one rank"
// output, so it isn't forced through consensus_board.js's
// buildConsensusTable.
let nflTeamUnits = []
let nflTeamUnitsSort = "team"

function renderNflTeamUnits(){
const recency = document.getElementById("nflUnitsRecency")
const sos = document.getElementById("nflUnitsSos")
const rows = unitsView(nflTeamUnits, unitsViewKey(recency && recency.checked, sos && sos.checked))
document.getElementById("nflUnitMatchups").innerHTML =
matchupCardsHtml(rows) || "Matchups appear after the next weekly update."
document.getElementById("nflUnitByes").innerHTML =
rows.length ? byeTableHtml(rows) : ""
const el = document.getElementById("nflTeamUnitsTable")
el.innerHTML = unitsTableHtml(rows, nflTeamUnitsSort)
el.querySelectorAll("th[data-units-sort]").forEach(th=>{
th.onclick = ()=>{ nflTeamUnitsSort = th.dataset.unitsSort; renderNflTeamUnits() }
})
}

;["nflUnitsRecency", "nflUnitsSos"].forEach(id=>{
const box = document.getElementById(id)
if(box){ box.onchange = ()=>renderNflTeamUnits() }
})

async function loadNflTeamUnits(){
try{
nflTeamUnits = await loadCSV("./data/nfl_team_units.csv")
}catch(e){
console.log("no nfl_team_units.csv yet", e)
nflTeamUnits = []
}
const week = nflTeamUnits.length ? nflTeamUnits[0].week : ""
const asOf = document.getElementById("nflTeamUnitsWeek")
if(asOf){ asOf.textContent = week ? `Going into week ${week}.` : "" }
renderNflTeamUnits()
}

// Cards (the default) or the sortable table, same rows either way.
function selectNflPropView(view){
document.querySelectorAll(".propViewSwitch .tabButton").forEach(btn=>{
btn.classList.toggle("active", btn.dataset.propView === view)
})
document.getElementById("nflPropCardsView").style.display = view === "cards" ? "" : "none"
document.getElementById("nflPlayerPropsTable").style.display = view === "table" ? "" : "none"
}

async function loadNflPlayerProps(){
// Team units first, so each prop row can show the units behind it.
await loadNflTeamUnits()
try{
const props = await loadCSV("./data/nfl_player_props.csv")
const units = unitsByTeam(nflTeamUnits)
buildPlayerPropsSection(props, "nflPlayerPropsGame", "nflPlayerPropsTable", units, rows=>{
document.getElementById("nflPropLean").innerHTML = leanChartHtml(rows)
document.getElementById("nflPropCards").innerHTML = propCardsHtml(rows, units)
})
}catch(e){
console.log("no nfl_player_props.csv yet", e)
buildPlayerPropsSection([], "nflPlayerPropsGame", "nflPlayerPropsTable", null, ()=>{
document.getElementById("nflPropLean").innerHTML = ""
document.getElementById("nflPropCards").innerHTML = propCardsHtml([], null)
})
}
try{
const floors = await loadCSV("./data/nfl_position_floors.csv")
buildPositionFloorsSection(floors, "nflPositionFloorTabs", "nflPositionFloorGame", "nflPositionFloorTable")
}catch(e){
console.log("no nfl_position_floors.csv yet", e)
buildPositionFloorsSection([], "nflPositionFloorTabs", "nflPositionFloorGame", "nflPositionFloorTable")
}
}

// ---- Stats tab (nfl_stats.js) ----
// Picking a team crumples the schedule sheet on show and unfolds the new
// one (CSS animations shared with the MLB Stats page).

let nflScheduleRows = []
let nflScheduleTeam = ""
let nflTeamRowsCache = []
let nflSelectedTeams = []
let nflPlayers = []
let nflPlayerPosition = "QB"
let nflPlayerView = "season"
let nflPlayerSort = ""
let nflPlayerMatches = []
let nflUnfoldTimer = null

function showNflSchedule(team){
const sheet = document.getElementById("nflScheduleSheet")
if(!sheet){ return }
nflScheduleTeam = team
document.getElementById("nflSchedulePicker").innerHTML =
nflSchedulePickerHtml([...new Set(nflScheduleRows.map(r => r.team))], team)
const unfold = () => {
sheet.classList.remove("crumpling", "unfolding")
sheet.innerHTML = nflScheduleSheetHtml(nflScheduleRows, team)
void sheet.offsetWidth
sheet.classList.add("unfolding")
clearTimeout(nflUnfoldTimer)
nflUnfoldTimer = setTimeout(() => sheet.classList.remove("unfolding"), 1400)
}
const reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches
if(sheet.childElementCount && !reduced){
sheet.classList.remove("unfolding")
sheet.classList.add("crumpling")
setTimeout(unfold, 650)
}else{
unfold()
}
}

function renderNflTeams(){
document.getElementById("nflPowerRankings").innerHTML = nflPowerRankingsHtml(nflTeamRowsCache, nflSelectedTeams)
document.getElementById("nflTeamProfile").innerHTML = nflTeamProfileHtml(nflSelectedTeams, nflTeamRowsCache)
}

function renderNflPlayers(){
document.querySelectorAll("#nflPlayerPositions [data-position]").forEach(b => b.classList.toggle("active", b.dataset.position === nflPlayerPosition))
document.querySelectorAll("#nflPlayerView [data-player-view]").forEach(b => b.classList.toggle("active", b.dataset.playerView === nflPlayerView))
document.getElementById("nflPlayerTable").innerHTML = nflPlayerTableHtml(nflPlayers, nflPlayerPosition, nflPlayerSort, nflPlayerView)
}

function showNflPlayer(player){
document.getElementById("nflPlayerResult").innerHTML = player ? nflPlayerCardHtml(player, nflPlayers, nflPlayerView) : ""
}

function wireNflStats(){
document.getElementById("nflSchedulePicker").addEventListener("click", e => {
const button = e.target.closest("[data-sched-team]")
if(button && button.dataset.schedTeam !== nflScheduleTeam){ showNflSchedule(button.dataset.schedTeam) }
})
document.getElementById("nflPowerRankings").addEventListener("click", e => {
const button = e.target.closest("[data-team]")
if(!button){ return }
const team = button.dataset.team
if(nflSelectedTeams.includes(team) && nflSelectedTeams.length > 1){
nflSelectedTeams = nflSelectedTeams.filter(t => t !== team)
}else{
nflSelectedTeams = nflSelectedTeams.filter(t => t !== team).concat(team).slice(-2)
}
renderNflTeams()
document.getElementById("nflTeamProfile").scrollIntoView({ behavior: "smooth", block: "nearest" })
})
document.getElementById("nflPlayerPositions").addEventListener("click", e => {
const button = e.target.closest("[data-position]")
if(button){ nflPlayerPosition = button.dataset.position; nflPlayerSort = ""; renderNflPlayers() }
})
document.getElementById("nflPlayerView").addEventListener("click", e => {
const button = e.target.closest("[data-player-view]")
if(button){ nflPlayerView = button.dataset.playerView; renderNflPlayers() }
})
document.getElementById("nflPlayerTable").addEventListener("click", e => {
const head = e.target.closest("[data-sort]")
if(head){ nflPlayerSort = head.dataset.sort; renderNflPlayers() }
})
const input = document.getElementById("nflPlayerSearch")
const search = () => {
const query = input.value.trim().toLowerCase()
nflPlayerMatches = query ? nflPlayers.filter(p => String(p.player_name).toLowerCase().includes(query)).slice(0, 8) : []
document.getElementById("nflPlayerChoices").innerHTML = query ? nflPlayerSuggestionsHtml(nflPlayerMatches) : ""
}
input.addEventListener("input", search)
input.addEventListener("keydown", e => {
if(e.key === "Enter"){ search(); showNflPlayer(nflPlayerMatches[0]); document.getElementById("nflPlayerChoices").innerHTML = "" }
})
document.getElementById("nflPlayerChoices").addEventListener("click", e => {
const button = e.target.closest(".playerChoice")
if(!button){ return }
showNflPlayer(nflPlayerMatches[Number(button.dataset.index)])
document.getElementById("nflPlayerChoices").innerHTML = ""
})
}

async function loadNflStats(){
if(!document.getElementById("nflScheduleSheet")){ return }
const optional = async path => { try{ return await loadCSV(path) }catch(e){ console.log(`no ${path} yet`, e); return [] } }
const [schedule, stats, players, units] = await Promise.all([
optional("./data/nfl_team_schedule.csv"), optional("./data/nfl_team_stats.csv"),
optional("./data/nfl_player_stats.csv"), optional("./data/nfl_team_units.csv"),
])
nflScheduleRows = schedule
nflTeamRowsCache = nflTeamRows(stats, units)
nflPlayers = players
const top = nflTeamRowsCache.slice().sort((a, b) => Number(b.net_epa_play) - Number(a.net_epa_play))[0]
if(top){ nflSelectedTeams = [top.team] }
renderNflTeams()
renderNflPlayers()
const best = players.filter(p => p.position === "QB").sort((a, b) => Number(b.epa_db) - Number(a.epa_db))[0]
showNflPlayer(best)
if(schedule.length){ showNflSchedule(top && schedule.some(r => r.team === top.team) ? top.team : schedule[0].team) }
else{ document.getElementById("nflSchedulePicker").innerHTML = "Schedules appear after the next weekly update." }
}

installVizTips()
wireNflStats()


loadNflStats()

loadNflGamePicks()

loadNflDraftBoard()

loadNflPlayerProps()

selectNflTab(nflDefaultTab(window.location.hash))
