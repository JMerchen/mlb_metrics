// College Hoops page: loads docs/data/cbb.json (cbb_pipeline.py) and wires
// the four tabs. The rendering and the matchup math are in cbb_render.js.

const CBB_TABS = ["madness", "games", "rankings", "matchup"]
const CBB_SECTIONS = {madness: "cbbMadnessSection", games: "cbbGamesSection", rankings: "cbbRankingsSection", matchup: "cbbMatchupSection"}
const cbb = {data: null, byId: new Map(), day: null, allOdds: false, allGames: false, allTeams: false}

function selectCbbTab(tab){
if(!CBB_TABS.includes(tab)){ tab = "madness" }
document.querySelectorAll("#cbbTabs .tabButton").forEach(b => b.classList.toggle("active", b.dataset.cbbTab === tab))
CBB_TABS.forEach(t => { document.getElementById(CBB_SECTIONS[t]).style.display = t === tab ? "" : "none" })
if(window.location.hash.replace("#", "") !== tab){ history.replaceState(null, "", `#${tab}`) }
}

function renderCbbMadness(){
const d = cbb.data
document.getElementById("cbbBracketNote").textContent = cbbBracketNote(d)
document.getElementById("cbbFavorites").innerHTML = cbbFavoritesHtml(d.bracket, cbb.byId)
document.getElementById("cbbOdds").innerHTML = cbbBracketOddsHtml(d.bracket, cbb.byId, cbb.allOdds ? null : 16)
const more = document.getElementById("cbbOddsMore")
more.style.display = cbb.allOdds ? "none" : ""
more.onclick = () => { cbb.allOdds = true; renderCbbMadness() }
const regions = [].concat(...(d.bracket.final_four_pairs || []))
document.getElementById("cbbUpsets").innerHTML = cbbUpsetsHtml(d.bracket, cbb.byId)
document.getElementById("cbbRegions").innerHTML = regions.map(r => cbbRegionHtml(d.bracket, cbb.byId, r, d.league)).join("")
const playOut = random => {
document.getElementById("cbbPlayed").innerHTML = cbbPlayedBracketHtml(cbbPlayBracket(d.bracket, cbb.byId, d.league, random), cbb.byId)
}
document.getElementById("cbbPlayRandom").onclick = () => playOut(cbbRng(Date.now()))
document.getElementById("cbbPlayFavorites").onclick = () => playOut(null)
const backtest = cbbBacktestHtml(d.backtest)
document.getElementById("cbbBacktest").innerHTML = backtest
document.getElementById("cbbBacktestSection").style.display = backtest ? "" : "none"
}

function renderCbbGames(){
const d = cbb.data
document.getElementById("cbbRecord").innerHTML = cbbRecordHtml(d.record)
const dates = [...new Set((d.games || []).map(g => g.date))].sort()
if(!dates.length){
document.getElementById("cbbDays").innerHTML = ""
document.getElementById("cbbGames").innerHTML = '<div class="empty-note">No games in the next few days.</div>'
return
}
if(!cbb.day || !dates.includes(cbb.day)){ cbb.day = dates.includes(d.updated) ? d.updated : dates.find(x => x >= d.updated) || dates[dates.length - 1] }
document.getElementById("cbbDays").innerHTML = dates.map(x =>
`<button class="tabButton${x === cbb.day ? " active" : ""}" data-cbb-day="${x}">${x === d.updated ? "Today" : cbbDateLabel(x)}</button>`).join("")
document.querySelectorAll("#cbbDays [data-cbb-day]").forEach(b => { b.onclick = () => { cbb.day = b.dataset.cbbDay; cbb.allGames = false; renderCbbGames() } })
const count = d.games.filter(g => g.date === cbb.day).length
document.getElementById("cbbGames").innerHTML = cbbGamesHtml(d.games, cbb.byId, cbb.day, cbb.allGames ? null : 15)
const more = document.getElementById("cbbGamesMore")
more.style.display = !cbb.allGames && count > 15 ? "" : "none"
more.onclick = () => { cbb.allGames = true; renderCbbGames() }
}

function renderCbbRankings(){
const search = document.getElementById("cbbSearch").value
const conference = document.getElementById("cbbConference").value
const filtered = Boolean(search || conference)
document.getElementById("cbbRankings").innerHTML = cbbRankingsHtml(cbb.data.teams, {search, conference, league: cbb.data.league, limit: cbb.allTeams || filtered ? null : 50})
const more = document.getElementById("cbbRankingsMore")
more.style.display = cbb.allTeams || filtered ? "none" : ""
more.onclick = () => { cbb.allTeams = true; renderCbbRankings() }
}

function renderCbbMatchup(){
const a = cbb.byId.get(Number(document.getElementById("cbbTeamA").value))
const b = cbb.byId.get(Number(document.getElementById("cbbTeamB").value))
document.getElementById("cbbMatchup").innerHTML = cbbMatchupHtml(a, b, document.getElementById("cbbSite").value, cbb.data.league)
}

function wireCbb(){
const d = cbb.data
const conferences = [...new Set(d.teams.map(t => t.conference))].sort()
document.getElementById("cbbConference").innerHTML = '<option value="">All conferences</option>'
+ conferences.map(c => `<option value="${cbbEsc(c)}">${cbbEsc(c)}</option>`).join("")
const byName = d.teams.slice().sort((x, y) => x.team.localeCompare(y.team))
const options = byName.map(t => `<option value="${t.team_id}">${cbbEsc(t.team)}</option>`).join("")
const a = document.getElementById("cbbTeamA"), b = document.getElementById("cbbTeamB")
a.innerHTML = options; b.innerHTML = options
if(d.teams.length > 1){ a.value = d.teams[0].team_id; b.value = d.teams[1].team_id }
;["cbbTeamA", "cbbTeamB", "cbbSite"].forEach(id => { document.getElementById(id).onchange = renderCbbMatchup })
document.getElementById("cbbSearch").oninput = renderCbbRankings
document.getElementById("cbbConference").onchange = renderCbbRankings
}

async function loadCbb(){
const note = document.getElementById("cbbUpdated")
try{
const response = await fetch(`./data/cbb.json?t=${Date.now()}`, {cache: "no-store"})
if(!response.ok){ throw new Error(`cbb.json ${response.status}`) }
cbb.data = await response.json()
}catch(e){
console.log("no cbb.json yet", e)
note.textContent = "Men's Division I. No data yet - it fills in with the first daily update."
return
}
const d = cbb.data
cbb.byId = cbbTeamIndex(d.teams)
const through = d.through ? cbbDateLabel(d.through) : "the preseason"
note.textContent = `Men's Division I, ${d.season - 1}–${String(d.season).slice(2)} season. ${Number(d.games_played).toLocaleString("en-US")} games through ${through}.`
wireCbb()
renderCbbMadness()
renderCbbGames()
renderCbbRankings()
renderCbbMatchup()
}

selectCbbTab(window.location.hash.replace("#", ""))
loadCbb()
