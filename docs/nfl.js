// NFL page - separate from app.js/dfs.js/age-curves.js since this page
// is standalone (see docs/nfl.html). loadCSV/buildTable are copied (not
// shared - no build step in docs/, same convention every other page's JS
// already documents).
//
// See nfl_bestball.py's module docstring and the README for the full
// bestball-rankings methodology and known limitations (real games-played
// as an honest injury proxy, not a full medical history; the Preseason
// Notes list is a one-time hand-curated snapshot, not a live feed).

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

let nflBestball = []
let nflPositionScarcity = []
let nflDraftStrategy = []
let nflPositionNecessity = []
let nflFfRankings = []
let nflDraftNotes = []

// "My Draft" (Draft Assistant) - real, live-draft state, entirely
// client-side (see docs/nfl_draft_assistant.js for the pure calculation
// functions this wiring calls). Persisted to localStorage only - no
// account/server, matches this being a public static GitHub Pages site.
const DRAFT_ASSISTANT_STORAGE_KEY = "nflDraftAssistant.v1"
let myDraftRoster = [] // [{player_id, player_display_name, position}]
let myDraftConsidering = [] // [{player_id, player_display_name, position}] - players being weighed for the CURRENT pick, not yet drafted

const NFL_TABS = ["preseason", "gamepicks", "teamunits", "draftboard", "playerprops"]

// Which tab the page opens on: a "#gamepicks"-style link wins; otherwise
// Game Picks during the season (September through February) and
// Preseason in the offseason, so the page opens on what's current.
function nflDefaultTab(date, hash){
const fromHash = (hash || "").replace("#", "")
if(NFL_TABS.includes(fromHash)){ return fromHash }
const month = date.getMonth() + 1
return month >= 9 || month <= 2 ? "gamepicks" : "preseason"
}

function selectNflTab(tab){
document.querySelectorAll("#nflTabs .tabButton").forEach(btn=>{
btn.classList.toggle("active", btn.dataset.nflTab === tab)
})
// Keep the open tab in the address, so a link or a refresh lands on it.
if(window.history && window.history.replaceState){
window.history.replaceState(null, "", `#${tab}`)
}
document.getElementById("preseasonSection").style.display = tab === "preseason" ? "" : "none"
document.getElementById("gamePicksSection").style.display = tab === "gamepicks" ? "" : "none"
document.getElementById("draftBoardSection").style.display = tab === "draftboard" ? "" : "none"
document.getElementById("playerPropsSection").style.display = tab === "playerprops" ? "" : "none"
document.getElementById("teamUnitsSection").style.display = tab === "teamunits" ? "" : "none"
}

function selectBestballPosition(position){
document.querySelectorAll("#bestballPositionTabs .tabButton").forEach(btn=>{
btn.classList.toggle("active", btn.dataset.bestballPosition === position)
})
renderBestball()
}

function renderBestball(){
const activeBtn = document.querySelector("#bestballPositionTabs .tabButton.active")
const position = activeBtn ? activeBtn.dataset.bestballPosition : "all"
let filtered = position === "all" ? nflBestball : nflBestball.filter(p=>p.position === position)

const searchEl = document.getElementById("bestballSearch")
const search = searchEl ? searchEl.value.trim().toLowerCase() : ""
if(search){
filtered = filtered.filter(p=>(p.player_display_name || "").toLowerCase().includes(search))
}

const rows = filtered.map(p=>{
const zScore = p.points_z_score !== undefined && p.points_z_score !== "" && !isNaN(Number(p.points_z_score))
? Number(p.points_z_score) : null
const posRank = p.position_rank && p.position_rank !== ""
? (zScore !== null ? `${p.position_rank} (${zScore>=0?"+":""}${zScore.toFixed(1)}σ)` : p.position_rank)
: "-"
return {
"Rank": p.overall_rank && p.overall_rank !== "" ? p.overall_rank : "-",
"Pos Rank": posRank,
"Player": p.player_display_name,
"Pos": p.position,
"Team": p.team,
"Pts/Game": p.dk_points_per_game && p.dk_points_per_game !== "" ? Number(p.dk_points_per_game).toFixed(1) : "-",
"Missed": p.games_missed && p.games_missed !== "" ? p.games_missed : "0",
"Missed (Prior Yr)": p.games_missed_prior_season && p.games_missed_prior_season !== "" ? p.games_missed_prior_season : "-",
"R1 Score": p.r1_dk_points !== undefined && p.r1_dk_points !== "" ? Number(p.r1_dk_points).toFixed(1) : "0.0",
"R2-R4 Score": p.r2_r4_dk_points !== undefined && p.r2_r4_dk_points !== "" ? Number(p.r2_r4_dk_points).toFixed(1) : "0.0",
"Value": p.points_above_replacement !== undefined && p.points_above_replacement !== "" ? Number(p.points_above_replacement).toFixed(1) : "-",
[`${p.season} Total`]: p.dk_points_total && p.dk_points_total !== "" ? Number(p.dk_points_total).toFixed(1) : "-",
}
})
buildTable(rows, "bestballTable")
}

function renderPositionScarcity(){
const bucketColumns = [
["below_-3sd", "< -3 SD"],
["-3sd_to_-2sd", "-3 to -2 SD"],
["-2sd_to_-1sd", "-2 to -1 SD"],
["-1sd_to_-0.5sd", "-1 to -0.5 SD"],
["-0.5sd_to_0sd", "-0.5 to 0 SD"],
["0sd_to_0.5sd", "0 to +0.5 SD"],
["0.5sd_to_1sd", "+0.5 to +1 SD"],
["1sd_to_2sd", "+1 to +2 SD"],
["2sd_to_3sd", "+2 to +3 SD"],
["above_3sd", "> +3 SD"],
]
const rows = nflPositionScarcity.map(p=>{
const row = {
"Pos": p.position,
"Total Players": p.total_players,
"Qualified": p.qualified_players,
"Points Floor": p.points_floor && p.points_floor !== "" ? Number(p.points_floor).toFixed(1) : "-",
"Outliers Excluded": p.outliers_removed && p.outliers_removed !== "" ? p.outliers_removed : "0",
"Mean": p.mean_dk_points && p.mean_dk_points !== "" ? Number(p.mean_dk_points).toFixed(1) : "-",
"Std": p.std_dk_points && p.std_dk_points !== "" ? Number(p.std_dk_points).toFixed(1) : "-",
"CV": p.coefficient_of_variation && p.coefficient_of_variation !== "" ? Number(p.coefficient_of_variation).toFixed(2) : "-",
}
bucketColumns.forEach(([key, label])=>{ row[label] = p[key] !== undefined && p[key] !== "" ? p[key] : "0" })
return row
})
buildTable(rows, "positionScarcityTable")
}

function renderDraftStrategy(){
const rows = nflDraftStrategy.map(t=>({
"Pos": t.position,
"Dispersion Rank": t.dispersion_rank && t.dispersion_rank !== "" ? t.dispersion_rank : "-",
"CV": t.coefficient_of_variation && t.coefficient_of_variation !== "" ? Number(t.coefficient_of_variation).toFixed(2) : "-",
"Takeaway": t.takeaway,
}))
buildTable(rows, "draftStrategyTable")
}

function renderPositionNecessity(){
const rows = nflPositionNecessity.map(n=>({
"Pos": n.position,
"Roster Target": n.roster_target_min && n.roster_target_max ? `${n.roster_target_min}-${n.roster_target_max}/team` : "-",
"Pod Demand": n.pod_demand && n.pod_demand !== "" ? n.pod_demand : "-",
"Available": n.available !== undefined && n.available !== "" ? n.available : "-",
"Necessity Ratio": n.necessity_ratio && n.necessity_ratio !== "" ? Number(n.necessity_ratio).toFixed(2) : "-",
"Read": n.read,
}))
buildTable(rows, "positionNecessityTable")
}

function loadDraftAssistantState(){
try{
const saved = JSON.parse(localStorage.getItem(DRAFT_ASSISTANT_STORAGE_KEY) || "{}")
myDraftRoster = saved.roster || []
myDraftConsidering = saved.considering || []
if(saved.draftSlot){ document.getElementById("draftAssistantSlot").value = saved.draftSlot }
if(saved.pickNumber){ document.getElementById("draftAssistantPick").value = saved.pickNumber }
}catch(e){
console.log("could not load saved My Draft state", e)
}
}

function saveDraftAssistantState(){
const draftSlot = document.getElementById("draftAssistantSlot").value
const pickNumber = document.getElementById("draftAssistantPick").value
localStorage.setItem(DRAFT_ASSISTANT_STORAGE_KEY, JSON.stringify({roster: myDraftRoster, considering: myDraftConsidering, draftSlot, pickNumber}))
}

function clearDraftAssistant(){
myDraftRoster = []
myDraftConsidering = []
document.getElementById("draftAssistantSlot").value = ""
document.getElementById("draftAssistantPick").value = ""
document.getElementById("draftAssistantSearch").value = ""
document.getElementById("draftAssistantConsiderSearch").value = ""
localStorage.removeItem(DRAFT_ASSISTANT_STORAGE_KEY)
renderDraftAssistantCandidates()
renderDraftAssistantRoster()
renderDraftAssistantConsiderCandidates()
renderDraftAssistantConsiderList()
renderDraftAssistant()
}

function addToDraftRoster(playerId){
const player = nflBestball.find(p=>p.player_id === playerId)
if(!player){ return }
myDraftRoster.push({player_id: player.player_id, player_display_name: player.player_display_name, position: player.position})
// A player just drafted is no longer "under consideration" for this pick.
myDraftConsidering = myDraftConsidering.filter(p=>p.player_id !== playerId)
document.getElementById("draftAssistantSearch").value = ""
renderDraftAssistantCandidates()
renderDraftAssistantRoster()
renderDraftAssistantConsiderCandidates()
renderDraftAssistantConsiderList()
renderDraftAssistant() // also saves state - see its own comment
}

function removeFromDraftRoster(playerId){
myDraftRoster = myDraftRoster.filter(p=>p.player_id !== playerId)
renderDraftAssistantCandidates()
renderDraftAssistantRoster()
renderDraftAssistant() // also saves state - see its own comment
}

function addToConsidering(playerId){
if(myDraftConsidering.some(p=>p.player_id === playerId)){ return }
const player = nflBestball.find(p=>p.player_id === playerId)
if(!player){ return }
myDraftConsidering.push({player_id: player.player_id, player_display_name: player.player_display_name, position: player.position})
document.getElementById("draftAssistantConsiderSearch").value = ""
renderDraftAssistantConsiderCandidates()
renderDraftAssistantConsiderList()
renderDraftAssistant() // also saves state - see its own comment
}

function removeFromConsidering(playerId){
myDraftConsidering = myDraftConsidering.filter(p=>p.player_id !== playerId)
renderDraftAssistantConsiderCandidates()
renderDraftAssistantConsiderList()
renderDraftAssistant() // also saves state - see its own comment
}

function renderDraftAssistantCandidates(){
const searchEl = document.getElementById("draftAssistantSearch")
const search = searchEl ? searchEl.value.trim().toLowerCase() : ""
const el = document.getElementById("draftAssistantCandidates")
if(!search){ el.innerHTML = ""; return }

const rosterIds = new Set(myDraftRoster.map(p=>p.player_id))
const matches = nflBestball
.filter(p=>!rosterIds.has(p.player_id) && (p.player_display_name || "").toLowerCase().includes(search))
.slice(0, 10)

if(!matches.length){ el.innerHTML = "No players found"; return }
el.innerHTML = matches.map(p=>
`<button onclick="addToDraftRoster('${p.player_id}')">+ ${p.player_display_name} (${p.position}, ${p.team})</button>`
).join(" ")
}

function renderDraftAssistantRoster(){
const el = document.getElementById("draftAssistantRoster")
if(!myDraftRoster.length){ el.innerHTML = "No players added yet - search above to build your real roster."; return }
el.innerHTML = myDraftRoster.map(p=>
`<div class="pickCard">${p.player_display_name} (${p.position}) <button class="removePitcher" onclick="removeFromDraftRoster('${p.player_id}')">x</button></div>`
).join("")
}

function renderDraftAssistantConsiderCandidates(){
const searchEl = document.getElementById("draftAssistantConsiderSearch")
const search = searchEl ? searchEl.value.trim().toLowerCase() : ""
const el = document.getElementById("draftAssistantConsiderCandidates")
if(!search){ el.innerHTML = ""; return }

// A player already on your roster is drafted, not "under consideration";
// a player already in the considering list doesn't need to be re-added.
const excludeIds = new Set([...myDraftRoster, ...myDraftConsidering].map(p=>p.player_id))
const matches = nflBestball
.filter(p=>!excludeIds.has(p.player_id) && (p.player_display_name || "").toLowerCase().includes(search))
.slice(0, 10)

if(!matches.length){ el.innerHTML = "No players found"; return }
el.innerHTML = matches.map(p=>
`<button onclick="addToConsidering('${p.player_id}')">+ ${p.player_display_name} (${p.position}, ${p.team})</button>`
).join(" ")
}

function renderDraftAssistantConsiderList(){
const el = document.getElementById("draftAssistantConsiderList")
if(!myDraftConsidering.length){ el.innerHTML = "No players added yet - search above to add players you're weighing for this pick."; return }
el.innerHTML = myDraftConsidering.map(p=>
`<div class="pickCard">${p.player_display_name} (${p.position}) <button class="removePitcher" onclick="removeFromConsidering('${p.player_id}')">x</button></div>`
).join("")
}

function renderDraftAssistant(){
saveDraftAssistantState() // also called on every draft-slot/pick-number input change, not just roster edits
const countsByPosition = {}
myDraftRoster.forEach(p=>{ countsByPosition[p.position] = (countsByPosition[p.position] || 0) + 1 })

const necessityByPosition = {}
nflPositionNecessity.forEach(n=>{ necessityByPosition[n.position] = n })

const ffByPlayerId = {}
nflFfRankings.forEach(f=>{ ffByPlayerId[f.player_id] = f })

const bestballByPlayerId = {}
nflBestball.forEach(p=>{ bestballByPlayerId[p.player_id] = p })

const draftSlot = Number(document.getElementById("draftAssistantSlot").value) || null
const pickNumber = Number(document.getElementById("draftAssistantPick").value) || null
const picksUntilNextTurn = (draftSlot && pickNumber)
? computeSnakeDraftPicksUntilNextTurn(draftSlot, DRAFT_POD_SIZE, pickNumber) : null

const picksUntilTurnEl = document.getElementById("draftAssistantPicksUntilTurn")
if(picksUntilTurnEl){
picksUntilTurnEl.textContent = picksUntilNextTurn !== null
? `Picks until your next turn: ${picksUntilNextTurn}`
: ""
}

const gap = computeRosterGap(countsByPosition)

// Real roster gap / necessity, per position - no prediction of who's
// still on the board, just your own real team construction.
const gapRows = Object.keys(ROSTER_TARGET).map(position=>{
const g = gap[position]
const necessity = necessityByPosition[position]
return {
"Pos": position,
"Roster": `${g.current}/${g.min}-${g.max}`,
"Status": g.status,
"Necessity Ratio": necessity && necessity.necessity_ratio ? Number(necessity.necessity_ratio).toFixed(2) : "-",
}
})
buildTable(gapRows, "draftAssistantGapTable")

// Real assessment of ONLY the players you said you're considering for
// this pick - sorted by real value above replacement, best first.
const considerRows = myDraftConsidering
.map(p=>{
const full = bestballByPlayerId[p.player_id]
const g = gap[p.position]
const ff = ffByPlayerId[p.player_id]
let valueRead = "-"
if(ff && pickNumber){
const read = computeReachValueRead(Number(ff.ecr), Number(ff.ecr_sd), pickNumber)
valueRead = read || "no ECR match"
}
return {
"Player": p.player_display_name,
"Pos": p.position,
"Team": full ? full.team : "-",
"Value": full && full.points_above_replacement !== undefined && full.points_above_replacement !== ""
? Number(full.points_above_replacement).toFixed(1) : "-",
"Roster Status": g ? g.status : "-",
"Necessity Ratio": necessityByPosition[p.position] && necessityByPosition[p.position].necessity_ratio
? Number(necessityByPosition[p.position].necessity_ratio).toFixed(2) : "-",
"Reach/Value vs ECR": valueRead,
"_sortValue": full && full.points_above_replacement !== undefined && full.points_above_replacement !== ""
? Number(full.points_above_replacement) : -Infinity,
}
})
.sort((a, b)=>b._sortValue - a._sortValue)
.map(row=>{ delete row._sortValue; return row })

if(!myDraftConsidering.length){
document.getElementById("draftAssistantTable").innerHTML =
"Add players above to see a real assessment of each one for this pick."
}else{
buildTable(considerRows, "draftAssistantTable")
}
}

function renderDraftNotes(){
const rows = nflDraftNotes.map(n=>({
"Player / Topic": n.player_or_topic,
"Note": n.note,
"Source": n.source_url ? `<a href="${n.source_url}" target="_blank" rel="noopener">${n.source_name}</a>` : (n.source_name || "-"),
}))
buildTable(rows, "draftNotesTable")
}

async function loadAll(){

try{
nflBestball = await loadCSV("./data/nfl_bestball_rankings.csv")
}catch(e){
console.log("no nfl_bestball_rankings.csv yet", e)
}

try{
nflPositionScarcity = await loadCSV("./data/nfl_position_scarcity.csv")
}catch(e){
console.log("no nfl_position_scarcity.csv yet", e)
}

try{
nflDraftStrategy = await loadCSV("./data/nfl_draft_strategy_takeaways.csv")
}catch(e){
console.log("no nfl_draft_strategy_takeaways.csv yet", e)
}

try{
nflPositionNecessity = await loadCSV("./data/nfl_position_necessity.csv")
}catch(e){
console.log("no nfl_position_necessity.csv yet", e)
}

try{
nflFfRankings = await loadCSV("./data/nfl_ff_rankings.csv")
}catch(e){
console.log("no nfl_ff_rankings.csv yet - the Draft Assistant's reach/value read will have nothing to show", e)
}

try{
nflDraftNotes = await loadCSV("./data/nfl_draft_notes.csv")
}catch(e){
console.log("no nfl_draft_notes.csv yet", e)
}

if(!nflBestball.length){
document.getElementById("bestballTable").innerHTML =
"No bestball rankings published yet - this page needs scripts/build_nfl_bestball_rankings.py to have been run at least once."
}else{
renderBestball()
}

if(!nflPositionScarcity.length){
document.getElementById("positionScarcityTable").innerHTML =
"No position scarcity data published yet - this page needs scripts/build_nfl_bestball_rankings.py to have been run at least once."
}else{
renderPositionScarcity()
}

if(!nflDraftStrategy.length){
document.getElementById("draftStrategyTable").innerHTML =
"No draft strategy analysis published yet - this page needs scripts/build_nfl_bestball_rankings.py to have been run at least once."
}else{
renderDraftStrategy()
}

if(!nflPositionNecessity.length){
document.getElementById("positionNecessityTable").innerHTML =
"No position necessity data published yet - this page needs scripts/build_nfl_bestball_rankings.py to have been run at least once."
}else{
renderPositionNecessity()
}

if(!nflDraftNotes.length){
document.getElementById("draftNotesTable").innerHTML = "No preseason notes published yet."
}else{
renderDraftNotes()
}

if(nflBestball.length){
loadDraftAssistantState()
renderDraftAssistantCandidates()
renderDraftAssistantRoster()
renderDraftAssistantConsiderCandidates()
renderDraftAssistantConsiderList()
renderDraftAssistant()
}else{
const noDataMessage = "No bestball rankings published yet - the Draft Assistant needs scripts/build_nfl_bestball_rankings.py to have been run at least once."
document.getElementById("draftAssistantGapTable").innerHTML = noDataMessage
document.getElementById("draftAssistantTable").innerHTML = noDataMessage
}

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

return `
<div class="pickCard ${p.status}${betAdvised ? " recommended" : ""}">
<div class="pickName">${matchup}</div>
<div class="pickProb">We say ${pick} ${pct0(p.predicted_probability)}${vegas ? ` &middot; Vegas ${vegas}` : ""}</div>
${totalPointsText(p) ? `<div class="pickProb">${totalPointsText(p)}</div>` : ""}
${teamEpaHtml(p, "nfl")}
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
buildConsensusTable(draftBoard, "nflDraftBoardTable")
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
const el = document.getElementById("nflTeamUnitsTable")
el.innerHTML = unitsTableHtml(nflTeamUnits, nflTeamUnitsSort)
el.querySelectorAll("th[data-units-sort]").forEach(th=>{
th.onclick = ()=>{ nflTeamUnitsSort = th.dataset.unitsSort; renderNflTeamUnits() }
})
}

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

async function loadNflPlayerProps(){
// Team units first, so each prop row can show the units behind it.
await loadNflTeamUnits()
try{
const props = await loadCSV("./data/nfl_player_props.csv")
buildPlayerPropsSection(props, "nflPlayerPropsGame", "nflPlayerPropsTable", unitsByTeam(nflTeamUnits))
}catch(e){
console.log("no nfl_player_props.csv yet", e)
buildPlayerPropsSection([], "nflPlayerPropsGame", "nflPlayerPropsTable")
}
try{
const floors = await loadCSV("./data/nfl_position_floors.csv")
buildPositionFloorsSection(floors, "nflPositionFloorTabs", "nflPositionFloorGame", "nflPositionFloorTable")
}catch(e){
console.log("no nfl_position_floors.csv yet", e)
buildPositionFloorsSection([], "nflPositionFloorTabs", "nflPositionFloorGame", "nflPositionFloorTable")
}
}

loadAll()

loadNflGamePicks()

loadNflDraftBoard()

loadNflPlayerProps()

selectNflTab(nflDefaultTab(new Date(), window.location.hash))
