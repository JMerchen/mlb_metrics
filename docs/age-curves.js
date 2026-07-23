// Age Curves page - separate from app.js since this page is standalone
// (see docs/age-curves.html). loadCSV/buildTable are copied from app.js
// rather than shared, since docs/ has no build step/shared-utils file to
// import from (see app.js's own docstring-equivalent comments).
//
// A separate curve/projection per metric (hitters: AVG/OBP/SLG/OPS;
// pitchers: K9/BB9/HR9/FIP) - not one blended number - since power, plate
// discipline, contact rate, and pitching components all age differently
// (see age_curve.py's module docstring). The metric selector re-renders
// whichever player is currently selected without re-searching.

async function loadCSV(path){
const response = await fetch(`${path}?t=${Date.now()}`, {cache:"no-store"})
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

let projections = []
let comparablesAll = []
let leagueCurve = []
let matches = []
let selectedKeyMlbam = null
let chart = null

function currentMetric(){
return document.getElementById("metricSelect").value
}

async function loadAll(){

try{
projections = await loadCSV("./data/age_curve_projections.csv")
}catch(e){
console.log("no age_curve_projections.csv yet", e)
}

try{
comparablesAll = await loadCSV("./data/age_curve_comparables.csv")
}catch(e){
console.log("no age_curve_comparables.csv yet", e)
}

try{
leagueCurve = await loadCSV("./data/age_curve_league.csv")
}catch(e){
console.log("no age_curve_league.csv yet", e)
}

if(!projections.length){
document.getElementById("playerChoices").innerHTML =
"No age curve data published yet - this page needs scripts/fetch_lahman.py and scripts/build_age_curves.py to have been run at least once."
}

}

function onMetricChange(){
if(selectedKeyMlbam !== null){
renderPlayer(selectedKeyMlbam)
}
}

function searchPlayer(){

const query = document.getElementById("playerSearch").value.trim().toLowerCase()
const metric = currentMetric()

// One row per (player, metric) in `projections` - dedupe to one entry
// per player for the choice list regardless of which metric is selected.
const seen = new Set()
matches = projections
.filter(p=>p.metric === metric && (p.name || "").toLowerCase().includes(query))
.filter(p=>{
if(seen.has(p.key_mlbam)){ return false }
seen.add(p.key_mlbam)
return true
})
.slice(0, 10)

const el = document.getElementById("playerChoices")

if(!matches.length){
el.innerHTML = "No players found"
document.getElementById("playerResult").innerHTML = ""
return
}

el.innerHTML = matches
.map((p, i)=>`<button onclick="showPlayer(${i})">${p.name} (age ${p.age})</button>`)
.join("")

}

function showPlayer(index){
selectedKeyMlbam = matches[index].key_mlbam
renderPlayer(selectedKeyMlbam)
}

function renderPlayer(keyMlbam){

const metric = currentMetric()
const p = projections.find(row=>row.key_mlbam === keyMlbam && row.metric === metric)
const el = document.getElementById("playerResult")

if(!p){
el.innerHTML = `<div class="infoCard">No ${metric} data for this player.</div>`
document.getElementById("comparablesTable").innerHTML = ""
if(chart){ chart.destroy(); chart = null }
return
}

const hasProjection = p.projected_value_mean && p.projected_value_mean !== "" && p.projected_value_mean !== "nan"

if(!hasProjection){
el.innerHTML = `<div class="infoCard">No comparable historical seasons with a resolvable next season for ${p.name} (${metric}) - can't project.</div>`
document.getElementById("comparablesTable").innerHTML = ""
if(chart){ chart.destroy(); chart = null }
return
}

el.innerHTML = `
<div class="infoCard">
<div class="cardHeader">
<div class="cardTitle">${p.name}</div>
<div class="cardBadge">${metric}</div>
</div>
<div class="cardStats">
<div>Age</div>
<div>${p.age}</div>
<div>Current Season ${metric}</div>
<div>${Number(p.value).toFixed(3)}</div>
<div>Projected Next-Season ${metric}</div>
<div>${Number(p.projected_value_mean).toFixed(3)} (range: ${Number(p.projected_value_p25).toFixed(3)} - ${Number(p.projected_value_p75).toFixed(3)})</div>
<div>Comparables Used</div>
<div>${p.n_with_next_season} of ${p.n_comparables} nearest same-age comparables had a next season on record</div>
</div>
</div>
`

renderChart(p, metric)
renderComparablesTable(p, metric)

}

function renderChart(p, metric){

const canvas = document.getElementById("ageCurveChart")
if(!canvas || typeof Chart === "undefined"){ return }

const curveRows = leagueCurve.filter(r=>r.metric === metric)
const ages = curveRows.map(r=>Number(r.age)).sort((a, b)=>a - b)
const valueByAge = {}
curveRows.forEach(r=>{ valueByAge[Number(r.age)] = Number(r.value) })

const playerSeries = ages.map(a=>{
if(a === Number(p.age)){ return Number(p.value) }
if(a === Number(p.age) + 1){ return Number(p.projected_value_mean) }
return null
})

if(chart){ chart.destroy() }

chart = new Chart(canvas.getContext("2d"), {
type: "line",
data: {
labels: ages,
datasets: [
{
label: `League average ${metric} by age`,
data: ages.map(a=>valueByAge[a]),
borderColor: "#5c7cfa",
pointRadius: 0,
spanGaps: true,
},
{
label: p.name,
data: playerSeries,
borderColor: "#4caf50",
backgroundColor: "#4caf50",
pointRadius: 6,
spanGaps: false,
},
],
},
options: {
responsive: true,
scales: {
y: {title: {display: true, text: metric}},
x: {title: {display: true, text: "Age"}},
},
},
})

}

function renderComparablesTable(p, metric){

const rows = comparablesAll
.filter(r=>String(r.key_mlbam) === String(p.key_mlbam) && r.metric === metric)
.map(r=>({
"Player": r.name,
"Year": r.yearID,
"Age": r.age,
[metric]: r.value && r.value !== "" ? Number(r.value).toFixed(3) : "-",
[`Next Season ${metric}`]: r.next_value && r.next_value !== "" && r.next_value !== "nan" ? Number(r.next_value).toFixed(3) : "no next season",
}))

buildTable(rows, "comparablesTable")

}

document.getElementById("playerSearch").addEventListener("keydown", function(e){
if(e.key === "Enter"){ searchPlayer() }
})

loadAll()
