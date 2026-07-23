// Age Curves page - separate from app.js since this page is standalone
// (see docs/age-curves.html). loadCSV/buildTable are copied from app.js
// rather than shared, since docs/ has no build step/shared-utils file to
// import from (see app.js's own docstring-equivalent comments).

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
let chart = null

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

function searchPlayer(){

const query = document.getElementById("playerSearch").value.trim().toLowerCase()
matches = projections.filter(p=>(p.name||"").toLowerCase().includes(query)).slice(0, 10)

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

const p = matches[index]
const el = document.getElementById("playerResult")

const hasProjection = p.projected_ops_mean && p.projected_ops_mean !== "" && p.projected_ops_mean !== "nan"

if(!hasProjection){
el.innerHTML = `<div class="infoCard">No comparable historical seasons with a resolvable next season for ${p.name} - can't project.</div>`
document.getElementById("comparablesTable").innerHTML = ""
if(chart){ chart.destroy(); chart = null }
return
}

el.innerHTML = `
<div class="infoCard">
<div class="cardHeader">
<div class="cardTitle">${p.name}</div>
</div>
<div class="cardStats">
<div>Age</div>
<div>${p.age}</div>
<div>Current Season OPS</div>
<div>${Number(p.OPS).toFixed(3)}</div>
<div>Projected Next-Season OPS</div>
<div>${Number(p.projected_ops_mean).toFixed(3)} (range: ${Number(p.projected_ops_p25).toFixed(3)} - ${Number(p.projected_ops_p75).toFixed(3)})</div>
<div>Comparables Used</div>
<div>${p.n_with_next_season} of ${p.n_comparables} nearest same-age comparables had a next season on record</div>
</div>
</div>
`

renderChart(p)
renderComparablesTable(p)

}

function renderChart(p){

const canvas = document.getElementById("ageCurveChart")
if(!canvas || typeof Chart === "undefined"){ return }

const ages = leagueCurve.map(r=>Number(r.age)).sort((a, b)=>a - b)
const opsByAge = {}
leagueCurve.forEach(r=>{ opsByAge[Number(r.age)] = Number(r.OPS) })

const playerSeries = ages.map(a=>{
if(a === Number(p.age)){ return Number(p.OPS) }
if(a === Number(p.age) + 1){ return Number(p.projected_ops_mean) }
return null
})

if(chart){ chart.destroy() }

chart = new Chart(canvas.getContext("2d"), {
type: "line",
data: {
labels: ages,
datasets: [
{
label: "League average OPS by age",
data: ages.map(a=>opsByAge[a]),
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
y: {title: {display: true, text: "OPS"}},
x: {title: {display: true, text: "Age"}},
},
},
})

}

function renderComparablesTable(p){

const rows = comparablesAll
.filter(r=>String(r.key_mlbam) === String(p.key_mlbam))
.map(r=>({
"Player": r.name,
"Year": r.yearID,
"Age": r.age,
"OPS": r.OPS && r.OPS !== "" ? Number(r.OPS).toFixed(3) : "-",
"Next Season OPS": r.next_OPS && r.next_OPS !== "" && r.next_OPS !== "nan" ? Number(r.next_OPS).toFixed(3) : "no next season",
}))

buildTable(rows, "comparablesTable")

}

loadAll()
