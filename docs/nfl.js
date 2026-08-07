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

let nflBestball = []
let nflPositionScarcity = []
let nflDraftStrategy = []
let nflDraftNotes = []

function selectNflTab(tab){
document.querySelectorAll("#nflTabs .tabButton").forEach(btn=>{
btn.classList.toggle("active", btn.dataset.nflTab === tab)
})
document.getElementById("preseasonSection").style.display = tab === "preseason" ? "" : "none"
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
const filtered = position === "all" ? nflBestball : nflBestball.filter(p=>p.position === position)
const rows = filtered.map(p=>({
"Player": p.player_display_name,
"Pos": p.position,
"Team": p.team,
[`${p.season} Pts`]: p.dk_points_total && p.dk_points_total !== "" ? Number(p.dk_points_total).toFixed(1) : "-",
"Pts/Game": p.dk_points_per_game && p.dk_points_per_game !== "" ? Number(p.dk_points_per_game).toFixed(1) : "-",
"Games": p.games_played && p.possible_games ? `${p.games_played}/${p.possible_games}` : "-",
"Missed": p.games_missed && p.games_missed !== "" ? p.games_missed : "0",
"Missed (Prior Yr)": p.games_missed_prior_season && p.games_missed_prior_season !== "" ? p.games_missed_prior_season : "-",
"Snap %": p.avg_offense_pct && p.avg_offense_pct !== "" ? `${(Number(p.avg_offense_pct)*100).toFixed(0)}%` : "-",
}))
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

if(!nflDraftNotes.length){
document.getElementById("draftNotesTable").innerHTML = "No preseason notes published yet."
}else{
renderDraftNotes()
}

}

loadAll()
