// Age Curves page (age-curves.html): loads the CSVs and wires the DOM; the
// rendering lives in age_render.js. Search covers hitters and pitchers;
// the stat chips show whichever set the chosen player has.

async function loadCSV(path){
const response = await fetch(`${path}?t=${Date.now()}`, {cache:"no-store"})
if(!response.ok){ throw new Error(`Failed to load ${path}: ${response.status}`) }
const text = await response.text()
return Papa.parse(text, {header:true, skipEmptyLines:true}).data
}

let ageProjections = []
let ageComparableRows = []
let ageLeague = []
let ageHistory = []
let ageTeams = {}
let ageMatches = []
let agePlayerKey = null
let ageMetricKey = "OPS"

async function ageOptionalCSV(path){
try{ return await loadCSV(path) }catch(e){ console.log(`no ${path} yet`, e); return [] }
}

function agePlayerRow(key, metric){
return ageProjections.find(r => String(r.key_mlbam) === String(key) && r.metric === metric)
}

// The stats this player has rows for, in AGE_METRICS order.
function ageMetricsFor(key){
const have = new Set(ageProjections.filter(r => String(r.key_mlbam) === String(key)).map(r => r.metric))
return AGE_METRICS.filter(m => have.has(m[0])).map(m => m[0])
}

function renderAgeMetricTabs(){
const metrics = ageMetricsFor(agePlayerKey)
document.getElementById("ageMetricTabs").innerHTML = metrics.map(k =>
`<button class="tabButton${k === ageMetricKey ? " active" : ""}" data-age-metric="${k}">${ageMetric(k).label}</button>`).join("")
}

// Drawn at the card's real width so labels stay legible; a hidden card
// (the artifact's other tabs) gets a full-width drawing that scales down.
function ageChartWidth(){
const width = document.getElementById("ageChart").clientWidth
return width > 100 ? width - 32 : 720
}

function renderAgePlayer(){
const p = agePlayerRow(agePlayerKey, ageMetricKey)
const hero = document.getElementById("ageHero")
const chart = document.getElementById("ageChart")
const list = document.getElementById("ageComparables")
renderAgeMetricTabs()
if(!p){
hero.innerHTML = `<div class="vizCard">No ${ageMetric(ageMetricKey).label} for this player.</div>`
chart.innerHTML = ""
list.innerHTML = ""
return
}
const comps = ageComparables(ageComparableRows, p.key_mlbam, ageMetricKey)
hero.innerHTML = ageHeroHtml(p, ageMetricKey, ageLeague, comps, ageTeams[String(p.key_mlbam)])
chart.innerHTML = `<div class="vizHead"><b>${mlbEsc(p.name)}'s ${ageMetric(ageMetricKey).label} by age</b>
<span class="vizKey"><i class="keyOwn"></i>His seasons <i class="keyProj"></i>Next season (likely range) <i class="keyLeague"></i>League average <i class="keyUp"></i>A similar player got better <i class="keyDown"></i>worse</span></div>
<div class="ageChartBox">${ageChartSvg(p, ageMetricKey, ageLeague, ageHistory, comps, ageChartWidth())}</div>`
list.innerHTML = ageComparablesHtml(comps, ageMetricKey)
}

function selectAgePlayer(key){
agePlayerKey = key
const metrics = ageMetricsFor(key)
if(!metrics.includes(ageMetricKey)){ ageMetricKey = metrics[0] || "OPS" }
renderAgePlayer()
}

function wireAgeCurves(){
const input = document.getElementById("agePlayerSearch")
const search = () => {
const query = input.value.trim().toLowerCase()
const seen = new Set()
ageMatches = query ? ageProjections.filter(r => String(r.name || "").toLowerCase().includes(query) && !seen.has(r.key_mlbam) && seen.add(r.key_mlbam)).slice(0, 8) : []
document.getElementById("agePlayerChoices").innerHTML = query ? ageSuggestionsHtml(ageMatches) : ""
}
input.addEventListener("input", search)
input.addEventListener("keydown", e => {
if(e.key === "Enter"){ search(); if(ageMatches[0]){ selectAgePlayer(ageMatches[0].key_mlbam) } document.getElementById("agePlayerChoices").innerHTML = "" }
})
document.getElementById("agePlayerChoices").addEventListener("click", e => {
const button = e.target.closest(".playerChoice")
if(!button){ return }
selectAgePlayer(ageMatches[Number(button.dataset.index)].key_mlbam)
document.getElementById("agePlayerChoices").innerHTML = ""
})
document.getElementById("ageMetricTabs").addEventListener("click", e => {
const button = e.target.closest("[data-age-metric]")
if(button){ ageMetricKey = button.dataset.ageMetric; renderAgePlayer() }
})
let resizeTimer = null
window.addEventListener("resize", () => {
clearTimeout(resizeTimer)
resizeTimer = setTimeout(() => { if(agePlayerKey !== null){ renderAgePlayer() } }, 150)
})
}

async function loadAgeCurves(){
const [projections, comparables, league, history, wave, pave] = await Promise.all([
ageOptionalCSV("./data/age_curve_projections.csv"), ageOptionalCSV("./data/age_curve_comparables.csv"),
ageOptionalCSV("./data/age_curve_league.csv"), ageOptionalCSV("./data/age_curve_player_history.csv"),
ageOptionalCSV("./data/wave.csv"), ageOptionalCSV("./data/pave.csv"),
])
ageProjections = projections
ageComparableRows = comparables
ageLeague = league
ageHistory = history
pave.concat(wave).forEach(r => { ageTeams[String(r.key_mlbam)] = r.team })
if(!projections.length){
document.getElementById("ageHero").innerHTML = "No age curve data published yet."
return
}
// Open on the hitter with the best OPS this season.
const best = projections.filter(r => r.metric === "OPS").sort((a, b) => Number(b.value) - Number(a.value))[0]
selectAgePlayer(best ? best.key_mlbam : projections[0].key_mlbam)
}

installVizTips()
wireAgeCurves()
loadAgeCurves()
