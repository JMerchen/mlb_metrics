// MLB "Picks & stats" page (predictive.html): loads the CSVs and wires
// the DOM. The rendering itself lives in mlb_render.js.

async function loadCSV(path){

const response =

await fetch(

`${path}?t=${Date.now()}`,

{

cache:
"no-store"

}

)

if(!response.ok){
throw new Error(`Failed to load ${path}: ${response.status}`)
}

const text =
await response.text()

return Papa.parse(
text,
{
header:true,
skipEmptyLines:true
}
).data

}



// Quant-analytics item #5 ("backtest scope and statistical significance"):
// shared formatting for the Wilson CIs / p-values evaluation.py and
// game_evaluation.py now attach to every real backtest rate, so a bare
// percentage on the dashboard is never read as more certain than its real
// sample size supports. `low`/`high` are the *_ci_low/*_ci_high CSV
// columns (already 0-1 fractions, like the rate itself); returns "" when
// either is missing/blank rather than a broken "NaN%" string.
function ciLabel(low, high){
if(low === undefined || low === "" || high === undefined || high === ""){
return ""
}
return `95% CI ${(Number(low) * 100).toFixed(0)}–${(Number(high) * 100).toFixed(0)}%`
}

// `p` is a *_p_value CSV column (a real two-sided test p-value, see
// evaluation.binomial_significance/mean_significance). Below 0.05 is
// labeled "significant" - conventional, not a claim of certainty; above
// is labeled "not significant" explicitly rather than just omitted, since
// "no significance note shown" would otherwise read as "no evidence
// either way" instead of "checked, and it isn't distinguishable from
// noise yet." Returns "" when missing/blank (e.g. n=0, nothing to test).
function significanceLabel(p){
if(p === undefined || p === "" || Number.isNaN(Number(p))){
return ""
}
const value = Number(p)
return value < 0.05
? `significant (p=${value.toFixed(3)})`
: `not significant (p=${value.toFixed(2)})`
}



function buildTable(
data,
id,
limit=null
){

if(
limit
){

data =
data.slice(
0,
limit
)

}

if(
!data.length
){

document
.getElementById(
id
).innerHTML =
"No data"

return

}

// Quant-analytics item #3, slice 3 ("confidence intervals"): fold any
// "<X>_CI_Low"/"<X>_CI_High" pair - alongside a matching bare "<X>"
// column - into ONE formatted cell on X ("28.4% (22.1%-35.2%)"),
// dropping the two raw CI columns from the table entirely, rather than
// showing three separate raw, unformatted numbers. Data-driven (keys
// present on data[0]), so this is a no-op for any table with no CI
// columns (e.g. Top PAVE Players) - and forward-compatible: any future
// column following this same naming convention gets folded in for free.
const allKeys =
Object.keys(
data[0]
)
const ciBases =
allKeys
.filter(
k=>k.endsWith("_CI_Low")
)
.map(
k=>k.slice(0, -"_CI_Low".length)
)
.filter(
base=>allKeys.includes(base) && allKeys.includes(base + "_CI_High")
)
const hiddenKeys =
new Set(
ciBases.flatMap(base=>[base + "_CI_Low", base + "_CI_High"])
)
const columns =
allKeys.filter(k=>!hiddenKeys.has(k))
const pct =
v=>(Number(v) * 100).toFixed(1) + "%"

let html =
"<table>"

html +=
"<tr>"

columns
.forEach(
c=>
html +=
`<th>${c}</th>`
)

html +=
"</tr>"

data.forEach(
row=>{

html +=
"<tr>"

columns
.forEach(
c=>{
if(ciBases.includes(c)){
html += `<td>${pct(row[c])} (${pct(row[c + "_CI_Low"])}-${pct(row[c + "_CI_High"])})</td>`
} else {
html += `<td>${row[c]}</td>`
}
}
)

html +=
"</tr>"

}
)

html +=
"</table>"

document
.getElementById(
id
)
.innerHTML =
html

}



// ---- Hover / tap tooltips ----
// Any element with data-tip shows it in one shared tooltip on hover,
// keyboard focus or tap. A data-mark="x%,y%" also places the chart dot
// (.vegasDot) at that point of its plot.
function installVizTips(){
const tip = document.createElement("div")
tip.className = "vizTip"
tip.setAttribute("role", "tooltip")
document.body.appendChild(tip)
const show = target => {
const el = target && target.closest ? target.closest("[data-tip]") : null
if(!el){ tip.style.display = "none"; return }
tip.textContent = el.getAttribute("data-tip")
tip.style.display = "block"
const box = el.getBoundingClientRect()
const width = tip.offsetWidth
const left = Math.min(Math.max(8, box.left + box.width / 2 - width / 2), window.innerWidth - width - 8)
const above = box.top - tip.offsetHeight - 8
tip.style.left = `${left + window.scrollX}px`
tip.style.top = `${(above > 8 ? above : box.bottom + 8) + window.scrollY}px`
const mark = el.getAttribute("data-mark")
const plot = el.parentElement
const dot = plot && plot.querySelector(".vegasDot")
if(mark && dot){
const [x, y] = mark.split(",")
dot.style.left = `${x}%`
dot.style.top = `${y}%`
dot.style.display = "block"
}
}
const hide = () => {
tip.style.display = "none"
document.querySelectorAll(".vegasDot").forEach(d => { d.style.display = "none" })
}
document.addEventListener("mouseover", e => show(e.target))
document.addEventListener("focusin", e => show(e.target))
document.addEventListener("click", e => { if(e.target.closest && e.target.closest("[data-tip]")){ show(e.target) } })
document.addEventListener("mouseout", e => { if(!e.relatedTarget || !e.relatedTarget.closest || !e.relatedTarget.closest("[data-tip]")){ hide() } })
document.addEventListener("focusout", hide)
window.addEventListener("scroll", hide, { passive: true })
}

// ---- Data ----

let wave = []
let pave = []
let confidence = []
let hitterHitPredictions = []
let hitStreaks = []
let qualifiedHitters = []
let qualifiedPitchers = []
let wavePlayersById = {}
let playerMatches = []
let selectedTeams = []

async function loadOptionalCSV(path){
try{
return await loadCSV(path)
}catch(e){
console.log(`no ${path} yet`, e)
return []
}
}

function playerKey(id){
return String(id ?? "").trim().replace(/\.0$/, "")
}

// "Regular" hitters: at least the average number of plate appearances.
// Regular pitchers: at least a quarter of the most at-bats faced.
function qualifyPlayers(){
const pa = p => Number(p.PA_L || 0) + Number(p.PA_R || 0)
const meanPA = wave.length ? wave.reduce((sum, p) => sum + pa(p), 0) / wave.length : 0
qualifiedHitters = wave.filter(p => pa(p) >= meanPA)
const maxAB = Math.max(0, ...pave.map(p => Number(p.at_bats || 0)))
qualifiedPitchers = pave.filter(p => Number(p.at_bats || 0) >= maxAB * 0.25)
wavePlayersById = {}
wave.forEach(p => { wavePlayersById[playerKey(p.key_mlbam)] = p })
}

// ---- Player lookup ----

function searchPlayer(){
const query = document.getElementById("playerSearch").value.trim().toLowerCase()
const type = document.getElementById("playerType").value
const source = type === "wave" ? wave : pave
playerMatches = source.filter(p => `${p.name_first} ${p.name_last}`.toLowerCase().includes(query)).slice(0, 8)
document.getElementById("playerChoices").innerHTML = query ? playerSuggestionsHtml(playerMatches) : ""
}

function showPlayer(index){
const player = playerMatches[index]
if(!player){ return }
const hitter = document.getElementById("playerType").value === "wave"
const key = playerKey(player.key_mlbam)
const today = hitterHitPredictions.find(p => playerKey(p.key_mlbam) === key)
const streak = hitStreaks.find(p => playerKey(p.key_mlbam) === key)
const extra = hitter ? {
todayChance: today ? today.Model_Hit_Probability : null,
opponent: today ? today.opponent : "",
isHome: today ? String(today.is_home).toLowerCase() === "true" : false,
hitStreak: streak && Number(streak.Current_Hit_Streak) >= 3 ? Number(streak.Current_Hit_Streak) : 0,
} : {}
document.getElementById("playerResult").innerHTML = playerCardHtml(player, hitter ? "hitter" : "pitcher", hitter ? qualifiedHitters : qualifiedPitchers, extra)
}

function wirePlayerLookup(){
const input = document.getElementById("playerSearch")
input.addEventListener("input", searchPlayer)
input.addEventListener("keydown", e => {
if(e.key === "Enter"){
searchPlayer()
showPlayer(0)
document.getElementById("playerChoices").innerHTML = ""
}
})
document.getElementById("playerType").addEventListener("change", () => {
searchPlayer()
if(!input.value.trim()){ showDefaultPlayer() }
})
document.getElementById("playerChoices").addEventListener("click", e => {
const button = e.target.closest(".playerChoice")
if(!button){ return }
showPlayer(Number(button.dataset.index))
document.getElementById("playerChoices").innerHTML = ""
})
}

// Open on someone rather than an empty card: the regular hitter (or
// pitcher) the model likes most.
function showDefaultPlayer(){
const hitter = document.getElementById("playerType").value === "wave"
const pool = hitter ? qualifiedHitters : qualifiedPitchers
const key = hitter ? "Game_Hit_Probability" : "PAVE"
const best = pool.slice().sort((a, b) => hitter ? Number(b[key]) - Number(a[key]) : Number(a[key]) - Number(b[key]))[0]
if(!best){ return }
playerMatches = [best]
showPlayer(0)
}

// ---- Teams ----

function renderTeams(){
document.getElementById("powerRankings").innerHTML = powerRankingsHtml(confidence, selectedTeams)
document.getElementById("teamResult").innerHTML = teamProfileHtml(selectedTeams, confidence)
}

// Click one team to see it, a second to compare; a third replaces the
// first. Clicking a picked team again drops it.
function selectTeam(team){
if(selectedTeams.includes(team) && selectedTeams.length > 1){
selectedTeams = selectedTeams.filter(t => t !== team)
}else{
selectedTeams = selectedTeams.filter(t => t !== team)
selectedTeams.push(team)
if(selectedTeams.length > 2){ selectedTeams.shift() }
}
renderTeams()
}

function wireTeams(){
document.getElementById("powerRankings").addEventListener("click", e => {
const button = e.target.closest("[data-team]")
if(!button){ return }
selectTeam(button.dataset.team)
document.getElementById("teamResult").scrollIntoView({ behavior: "smooth", block: "nearest" })
})
}

// ---- Beat the Streak ----

function selectBtsView(view){
document.querySelectorAll("#btsTabs .tabButton").forEach(btn=>{
btn.classList.toggle("active", btn.dataset.view === view)
})
document.getElementById("ourPicksSection").style.display = view === "ourPicks" ? "" : "none"
document.getElementById("hitStreaksSection").style.display = view === "hitStreaks" ? "" : "none"
document.getElementById("modelOddsSection").style.display = view === "modelOdds" ? "" : "none"
}

function renderStreakStats(summary){
const el = document.getElementById("streakStats")
if(!summary.length){
el.innerHTML = "No picks tracked yet"
return
}
const s = summary[0]
const over = btsSeasonOver(s)
const days = n=>`${n} day${Number(n) === 1 ? "" : "s"}`
const stat = (value, label, sub) =>
`<div class="streakStat"><div class="value">${value}</div><div class="label">${label}</div>${sub ? `<div class="sub">${sub}</div>` : ""}</div>`
el.innerHTML =
stat(days(s.current_streak || 0), over ? "Final streak" : "Hit streak", over ? "at the end of the season" : "in a row with a hit") +
stat(days(s.longest_streak || 0), "Best streak", "our longest so far") +
stat(srPct(s.day_survival_rate), "Days with a hit", `over ${days(s.n_days_resolved || 0)}`)
document.getElementById("btsSeasonNote").innerHTML = over ? btsSeasonOverHtml(s) : ""
}

function renderTodaysPicks(picks, summary){
const el = document.getElementById("todaysPicks")
const heading = document.getElementById("btsTodayHeading")
if(!picks.length){
el.innerHTML = "No picks tracked yet"
return
}
const latestDate = picks.map(p=>p.date).sort().slice(-1)[0]
const todays = picks.filter(p=>p.date === latestDate && p.status !== "no_pick").sort((a,b)=>Number(a.rank) - Number(b.rank))
const over = summary.length && btsSeasonOver(summary[0])
heading.textContent = over ? `Last picks of the season · ${mlbShortDate(latestDate)}` : `Today · ${mlbShortDate(latestDate)}`
if(!todays.length){
el.innerHTML = `<div class="pickCard no_pick"><div class="pickName">No games or eligible batters for ${mlbShortDate(latestDate)}</div></div>`
return
}
el.innerHTML = btsPickCardsHtml(todays, wavePlayersById)
}

let btsPicks = []
let btsProgression = []

function renderStreakLog(limit){
document.getElementById("streakHistoryTable").innerHTML = btsLogHtml(btsPicks, btsProgression, limit)
const days = new Set(btsPicks.map(p => p.date)).size
const more = document.getElementById("streakHistoryMore")
more.style.display = days > limit ? "" : "none"
more.onclick = () => renderStreakLog(limit + 30)
}

async function loadBeatTheStreak(){
const [summary, picks, progression] = await Promise.all([
loadOptionalCSV("./data/beat_the_streak_summary.csv"),
loadOptionalCSV("./data/beat_the_streak_picks.csv"),
loadOptionalCSV("./data/beat_the_streak_progression.csv"),
])
btsPicks = picks
btsProgression = progression
renderStreakStats(summary)
renderTodaysPicks(picks, summary)
document.getElementById("streakChart").innerHTML = streakChartHtml(progression, picks)
renderStreakLog(14)
}

// ---- Game picks ----

// The by-version summary holds "all_time" plus one row for the model now
// making picks (config.GAME_PICK_MODEL_VERSION).
function currentVersionRow(byVersion){
return byVersion.find(r=>r.model_version && r.model_version !== "all_time") || null
}

// While bet advice is switched off (config.GAME_PICK_BETTING_ENABLED), the
// headline is the CURRENT model against the market on the same games - not
// a bet record that is no longer being added to and was built by a
// retired model. That older record moves to a small line underneath.
function renderPausedGamePickStats(allTime, current){
const note = document.getElementById("gamePickNote")
note.style.display = ""
note.innerHTML = "<b>We're not suggesting bets right now.</b> Our picks are about as good as Vegas's, but over four seasons they weren't better, so these are picks to follow, not bets."
const stat = (value, label, sub) =>
`<div class="streakStat"><div class="value">${value}</div><div class="label">${label}</div>${sub ? `<div class="sub">${sub}</div>` : ""}</div>`
const el = document.getElementById("gamePickStats")
const n = current ? Number(current.model_vs_market_n) || 0 : 0
if(!n){
el.innerHTML = stat("-", "Our picks right", "no finished games yet")
}else{
const ours = Math.round(Number(current.model_pick_accuracy) * n)
const vegas = Math.round(Number(current.market_pick_accuracy) * n)
const early = n < 100 ? "still early" : `${srPct(current.model_pick_accuracy)} of games`
el.innerHTML =
stat(`${ours} of ${n}`, "Our picks right", `${early} · current model`) +
stat(`${vegas} of ${n}`, "Vegas's picks right", "same games")
}
const record = document.getElementById("gamePickBetRecord")
const bets = Number(allTime.n_bets_advised) || 0
if(bets){
const units = Number(allTime.total_profit_units)
record.innerHTML = `Before we stopped suggesting bets, our bets won ${allTime.bets_won} of ${bets} and ${units >= 0 ? "made" : "lost"} ${Math.abs(units).toFixed(1)} units.`
}else{
record.innerHTML = ""
}
}

function renderGamePickStats(summary){
const el = document.getElementById("gamePickStats")
if(!summary.length){
el.innerHTML = "No picks tracked yet"
return
}
const s = summary[0]
const stat = (value, label, sub) =>
`<div class="streakStat"><div class="value">${value}</div><div class="label">${label}</div>${sub ? `<div class="sub">${sub}</div>` : ""}</div>`
// While bet advice is on, the headline is the bet record in plain terms:
// bets won, money made or lost, and the win rate those odds called for.
const bets = Number(s.n_bets_advised) || 0
if(!bets){
el.innerHTML = stat("-", "Bets won", "no bets settled yet")
return
}
const units = Number(s.total_profit_units)
el.innerHTML =
stat(`${s.bets_won} of ${bets}`, "Bets won", `the odds said about ${srPct(s.odds_implied_win_rate)} would win`) +
`<div class="streakStat"><div class="value" style="color:${units >= 0 ? "var(--success)" : "var(--danger)"}">${units >= 0 ? "+" : "−"}${Math.abs(units).toFixed(1)}</div><div class="label">Units won or lost</div></div>` +
stat(s.current_bet_streak || 0, "Winning days in a row")
}

function renderTodaysGamePicks(picks){
const el = document.getElementById("todaysGamePicks")
if(!picks.length){
el.innerHTML = "No game picks yet"
return
}
const latestDate = picks.map(p=>p.date).sort().slice(-1)[0]
const todays = picks.filter(p=>p.date === latestDate)
.sort((a,b)=>Number(b.predicted_probability) - Number(a.predicted_probability))
document.getElementById("gameTodayHeading").textContent = `Latest games · ${mlbShortDate(latestDate)}`
el.innerHTML = todays.length ? todays.map(gamePickCardHtml).join("") : "No games today"
}

async function loadGamePicks(){
const [summary, byVersion, picks] = await Promise.all([
loadOptionalCSV("./data/game_picks_summary.csv"),
loadOptionalCSV("./data/game_picks_summary_by_version.csv"),
loadOptionalCSV("./data/game_picks_picks.csv"),
])
const allTime = summary[0]
const bettingPaused = Boolean(allTime) && String(allTime.betting_enabled).toLowerCase() === "false"
if(bettingPaused){
renderPausedGamePickStats(allTime, currentVersionRow(byVersion))
}else{
renderGamePickStats(summary)
}
renderTodaysGamePicks(picks)
document.getElementById("gamePickCharts").innerHTML = versusVegasChartHtml(picks) + confidenceBucketsHtml(picks)
document.getElementById("gamePickHistoryTable").innerHTML = gameHistoryHtml(picks, 30)
}

// ---- Everything that needs the player files ----

async function loadAll(){
const [waveRows, paveRows, confidenceRows, predictions, streaks, probables] = await Promise.all([
loadOptionalCSV("./data/wave.csv"),
loadOptionalCSV("./data/pave.csv"),
loadOptionalCSV("./data/confidence.csv"),
loadOptionalCSV("./data/hitter_hit_predictions.csv"),
loadOptionalCSV("./data/hit_streaks.csv"),
loadOptionalCSV("./data/probable_pitchers.csv"),
])
wave = waveRows
pave = paveRows
confidence = confidenceRows.sort((a, b) => String(a.team).localeCompare(String(b.team)))
hitterHitPredictions = predictions
hitStreaks = streaks
qualifyPlayers()

document.getElementById("hitStreaksTable").innerHTML = hitStreakListHtml(hitStreaks)
document.getElementById("modelOddsTable").innerHTML = bestChancesHtml(hitterHitPredictions)

// Probable starters carry key_mlbam for the headshot; older files
// without it are matched to pave.csv by name.
const byName = {}
pave.forEach(p => { byName[`${p.name_first} ${p.name_last}`] = p.key_mlbam })
document.getElementById("probablePitchersTable").innerHTML =
pitcherDuelsHtml(probables.map(r => r.key_mlbam ? r : Object.assign({}, r, { key_mlbam: byName[r.pitcher_name] || "" })))

document.getElementById("waveTable").innerHTML = hitterLeadersHtml(
qualifiedHitters.slice().sort((a, b) => Number(b.Game_Hit_Probability) - Number(a.Game_Hit_Probability)))
document.getElementById("paveTable").innerHTML = pitcherLeadersHtml(
qualifiedPitchers.slice().sort((a, b) => Number(b.PAVE) - Number(a.PAVE)))

const top = confidence.reduce((best, r) => !best || Number(r.Confidence) > Number(best.Confidence) ? r : best, null)
if(top && !selectedTeams.length){ selectedTeams = [top.team] }
renderTeams()
showDefaultPlayer()
}

installVizTips()
wirePlayerLookup()
wireTeams()

// The pick cards need wave.csv for each player's team, so Beat the
// Streak renders once the player files are in.
loadAll().then(loadBeatTheStreak)

loadGamePicks()
