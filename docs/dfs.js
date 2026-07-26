// DFS Rankings page - separate from app.js/age-curves.js since this page
// is standalone (see docs/dfs.html). loadCSV/buildTable are copied (not
// shared - no build step in docs/, same convention age-curves.js already
// documents).
//
// See dfs.py's/dfs_ceiling.py's/estimated_salary.py's module docstrings
// and the README for the full DK-points methodology, the Ceiling/
// Boom-Adjusted/Boom Rate/Matchup Boom signals, backtest numbers, and
// known limitations (including that Estimated_Salary is a MODELED
// estimate, not a real DraftKings price).

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

let dfsHitters = []
let dfsPitchers = []
let dfsOptimalLineup = []

function selectPosition(position){
document.querySelectorAll("#positionTabs .tabButton").forEach(btn=>{
btn.classList.toggle("active", btn.dataset.position === position)
})
document.getElementById("hittersTable").style.display = position === "hitters" ? "" : "none"
document.getElementById("pitchersTable").style.display = position === "pitchers" ? "" : "none"
document.getElementById("optimalLineupSection").style.display = position === "optimal" ? "" : "none"
}

function renderHitters(){
const rows = dfsHitters.map(p=>({
"Player": `${p.name_first} ${p.name_last}`,
"Team": p.team,
"Opp": `${p.is_home === "True" || p.is_home === "true" ? "vs" : "@"} ${p.opponent}`,
"Expected Bases": p.Expected_Bases && p.Expected_Bases !== "" ? Number(p.Expected_Bases).toFixed(2) : "-",
"Matchup Ratio": p.Matchup_Ratio && p.Matchup_Ratio !== "" ? Number(p.Matchup_Ratio).toFixed(2) : "-",
"Exp BB": p.Expected_BB && p.Expected_BB !== "" ? Number(p.Expected_BB).toFixed(2) : "-",
"Exp HBP": p.Expected_HBP && p.Expected_HBP !== "" ? Number(p.Expected_HBP).toFixed(2) : "-",
"Exp RBI": p.Expected_RBI && p.Expected_RBI !== "" ? Number(p.Expected_RBI).toFixed(2) : "-",
"DK Pts": p.DK_Points_Hitter && p.DK_Points_Hitter !== "" ? Number(p.DK_Points_Hitter).toFixed(2) : "-",
"Ceiling (P90)": p.Ceiling_DK_Points && p.Ceiling_DK_Points !== "" ? Number(p.Ceiling_DK_Points).toFixed(2) : "-",
"Boom-Adjusted": p.Boom_Adjusted_DK_Points && p.Boom_Adjusted_DK_Points !== "" ? Number(p.Boom_Adjusted_DK_Points).toFixed(2) : "-",
"Boom Rate": p.Boom_Rate && p.Boom_Rate !== "" ? Number(p.Boom_Rate).toFixed(3) : "-",
"Matchup Boom": p.Matchup_Boom_Score && p.Matchup_Boom_Score !== "" ? Number(p.Matchup_Boom_Score).toFixed(3) : "-",
}))
buildTable(rows, "hittersTable")
}

function renderPitchers(){
const rows = dfsPitchers.map(p=>({
"Player": `${p.name_first} ${p.name_last}`,
"Team": p.team,
"Opp": `${p.is_home === "True" || p.is_home === "true" ? "vs" : "@"} ${p.opponent}`,
"K9": p.K9 && p.K9 !== "" ? Number(p.K9).toFixed(1) : "-",
"BB9": p.BB9 && p.BB9 !== "" ? Number(p.BB9).toFixed(1) : "-",
"HR9": p.HR9 && p.HR9 !== "" ? Number(p.HR9).toFixed(1) : "-",
"Exp IP": p.Expected_IP && p.Expected_IP !== "" ? Number(p.Expected_IP).toFixed(1) : "-",
"DK Pts": p.DK_Points_Pitcher && p.DK_Points_Pitcher !== "" ? Number(p.DK_Points_Pitcher).toFixed(2) : "-",
"Ceiling (P90)": p.Ceiling_DK_Points && p.Ceiling_DK_Points !== "" ? Number(p.Ceiling_DK_Points).toFixed(2) : "-",
"Boom-Adjusted": p.Boom_Adjusted_DK_Points && p.Boom_Adjusted_DK_Points !== "" ? Number(p.Boom_Adjusted_DK_Points).toFixed(2) : "-",
}))
buildTable(rows, "pitchersTable")
}

function renderOptimalLineup(){
const rows = dfsOptimalLineup.map(p=>({
"Slot": p.dk_slot,
"Player": `${p.name_first} ${p.name_last}`,
"Team": p.team,
"Opp": `${p.is_home === "True" || p.is_home === "true" ? "vs" : "@"} ${p.opponent}`,
"DK Pts": p.DK_Points && p.DK_Points !== "" ? Number(p.DK_Points).toFixed(2) : "-",
"Est. Salary": p.Estimated_Salary && p.Estimated_Salary !== "" ? `$${Number(p.Estimated_Salary).toLocaleString()}` : "-",
}))
buildTable(rows, "optimalLineupTable")
}

async function loadAll(){

try{
dfsHitters = await loadCSV("./data/dfs_hitters.csv")
}catch(e){
console.log("no dfs_hitters.csv yet", e)
}

try{
dfsPitchers = await loadCSV("./data/dfs_pitchers.csv")
}catch(e){
console.log("no dfs_pitchers.csv yet", e)
}

try{
dfsOptimalLineup = await loadCSV("./data/optimal_lineup.csv")
}catch(e){
console.log("no optimal_lineup.csv yet", e)
}

if(!dfsHitters.length && !dfsPitchers.length){
document.getElementById("hittersTable").innerHTML =
"No DFS rankings published yet - this page needs scripts/build_dfs_rankings.py to have been run at least once."
return
}

renderHitters()
renderPitchers()

if(!dfsOptimalLineup.length){
document.getElementById("optimalLineupTable").innerHTML =
"No optimal lineup published yet - this page needs scripts/build_optimal_lineup.py to have been run at least once."
}else{
renderOptimalLineup()
}

}

loadAll()
