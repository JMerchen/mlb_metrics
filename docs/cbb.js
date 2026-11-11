// College Hoops page: loads docs/data/cbb.json (cbb_pipeline.py) and wires
// the tabs. The Time Machine tab has its own file, docs/data/cbb_history.json
// (scripts/build_cbb_history.py), fetched the first time the tab is opened.
// The rendering and the matchup math are in cbb_render.js.

const CBB_TABS = ["madness", "games", "rankings", "matchup", "era"]
const CBB_SECTIONS = {madness: "cbbMadnessSection", games: "cbbGamesSection", rankings: "cbbRankingsSection", matchup: "cbbMatchupSection", era: "cbbEraSection"}
// How many times the all-time bracket is played for its odds (a fixed
// seed, so the numbers are the same on every visit).
const CBB_ERA_SIMS = 5000
const cbbEra = {started: false, teams: null, byId: null, league: null, bracket: null, tally: null, allOdds: false, bestLimit: 25}
const cbb = {data: null, byId: new Map(), day: null, allOdds: false, allGames: false, allTeams: false}

function selectCbbTab(tab){
if(!CBB_TABS.includes(tab)){ tab = "madness" }
document.querySelectorAll("#cbbTabs .tabButton").forEach(b => b.classList.toggle("active", b.dataset.cbbTab === tab))
CBB_TABS.forEach(t => { document.getElementById(CBB_SECTIONS[t]).style.display = t === tab ? "" : "none" })
if(window.location.hash.replace("#", "") !== tab){ history.replaceState(null, "", `#${tab}`) }
if(tab === "era" && !cbbEra.started){ loadCbbEra() }
}

// ---- Time Machine ----

function cbbEraPicked(side){
return cbbEra.byId.get(Number(document.getElementById(`cbbEraTeam${side}`).value))
}

// Fills one side's team list with its season's teams (A to Z) and selects `teamId`.
function fillCbbEraTeams(side, teamId){
const season = Number(document.getElementById(`cbbEraSeason${side}`).value)
const teams = cbbEra.teams.filter(t => t.season === season).sort((x, y) => x.school.localeCompare(y.school))
const select = document.getElementById(`cbbEraTeam${side}`)
select.innerHTML = teams.map(t => `<option value="${t.team_id}">${cbbEsc(t.school)}</option>`).join("")
const best = teams.slice().sort((x, y) => x.rank - y.rank)[0]
select.value = teams.some(t => t.team_id === teamId) ? teamId : best.team_id
}

function setCbbEraTeam(side, team){
document.getElementById(`cbbEraSeason${side}`).value = team.season
fillCbbEraTeams(side, team.team_id)
}

function renderCbbEraMatchup(){
document.getElementById("cbbEraMatchup").innerHTML = cbbEraMatchupHtml(cbbEraPicked("A"), cbbEraPicked("B"), cbbEra.league, cbbEra.tally)
}

function playCbbEraGames(count){
const a = cbbEraPicked("A"), b = cbbEraPicked("B")
if(!a || !b || a.team_id === b.team_id){ return }
const random = cbbRng(Date.now())
const tally = cbbEra.tally || {games: 0, winsA: 0, last: null}
for(let i = 0; i < count; i++){
tally.last = cbbPlayGame(a, b, cbbEra.league, random)
tally.games += 1
tally.winsA += tally.last.scoreA > tally.last.scoreB ? 1 : 0
}
cbbEra.tally = tally
renderCbbEraMatchup()
}

function renderCbbEraBracket(){
const b = cbbEra.bracket
const champions = b.teams.filter(t => cbbEra.byId.get(t.team_id).march === 8).length
document.getElementById("cbbEraBracketNote").textContent = `One bracket for the whole era: all ${champions} national champions since 2009, then the best of everyone else, ${b.teams.length} teams seeded straight down the ratings and played out ${CBB_ERA_SIMS.toLocaleString("en-US")} times. There are no easy games in a field like this - most of the 15 and 16 seeds were top-five teams in their own year - so even the favorite rarely gets through.`
document.getElementById("cbbEraFavorites").innerHTML = cbbFavoritesHtml(b, cbbEra.byId)
document.getElementById("cbbEraOdds").innerHTML = cbbAllTimeOddsHtml(b, cbbEra.byId, cbbEra.allOdds ? null : 16)
const more = document.getElementById("cbbEraOddsMore")
more.style.display = cbbEra.allOdds ? "none" : ""
more.onclick = () => { cbbEra.allOdds = true; renderCbbEraBracket() }
}

function renderCbbEraBest(){
const search = document.getElementById("cbbEraSearch").value
document.getElementById("cbbEraBest").innerHTML = cbbBestEverHtml(cbbEra.teams, search ? 60 : cbbEra.bestLimit, search)
const more = document.getElementById("cbbEraBestMore")
more.style.display = search || cbbEra.bestLimit >= 100 ? "none" : ""
more.onclick = () => { cbbEra.bestLimit = 100; renderCbbEraBest() }
}

async function loadCbbEra(){
cbbEra.started = true
let history
try{
const response = await fetch("./data/cbb_history.json")
if(!response.ok){ throw new Error(`cbb_history.json ${response.status}`) }
history = await response.json()
}catch(e){
console.log("no cbb_history.json", e)
cbbEra.started = false
document.getElementById("cbbEraMatchup").innerHTML = '<div class="empty-note">The Time Machine couldn\'t load. Try again in a moment.</div>'
return
}
cbbEra.teams = cbbEraTeams(history)
cbbEra.byId = cbbTeamIndex(cbbEra.teams)
cbbEra.league = cbbEraLeague(history)
const seasons = [...new Set(cbbEra.teams.map(t => t.season))].sort((x, y) => y - x)
const seasonOptions = seasons.map(s => `<option value="${s}">${s - 1}–${String(s).slice(2)}</option>`).join("")
;["A", "B"].forEach(side => {
document.getElementById(`cbbEraSeason${side}`).innerHTML = seasonOptions
document.getElementById(`cbbEraSeason${side}`).onchange = () => { fillCbbEraTeams(side, null); cbbEra.tally = null; renderCbbEraMatchup() }
document.getElementById(`cbbEraTeam${side}`).onchange = () => { cbbEra.tally = null; renderCbbEraMatchup() }
})
// Opens on the best team of the era against the latest champion (or the next best team).
const first = cbbEra.teams[0]
const latestChampion = cbbEra.teams.filter(t => t.march === 8 && t.team_id !== first.team_id).sort((x, y) => y.season - x.season)[0]
setCbbEraTeam("A", first)
setCbbEraTeam("B", latestChampion || cbbEra.teams[1])
document.getElementById("cbbEraPlay").onclick = () => playCbbEraGames(1)
document.getElementById("cbbEraSeries").onclick = () => playCbbEraGames(100)
document.getElementById("cbbEraRandom").onclick = () => {
const random = cbbRng(Date.now())
const pool = cbbEra.teams.slice(0, 150)
const a = pool[Math.floor(random() * pool.length)]
let b = a
while(b.team_id === a.team_id || b.season === a.season){ b = pool[Math.floor(random() * pool.length)] }
setCbbEraTeam("A", a); setCbbEraTeam("B", b)
cbbEra.tally = null
renderCbbEraMatchup()
}
renderCbbEraMatchup()
document.getElementById("cbbEraSearch").oninput = renderCbbEraBest
renderCbbEraBest()
// The bracket's odds take a moment, so the rest of the tab is drawn first.
cbbEra.bracket = cbbAllTimeBracket(cbbEra.teams)
setTimeout(() => {
const odds = cbbBracketOdds(cbbEra.bracket, cbbEra.byId, cbbEra.league, CBB_ERA_SIMS, cbbRng(2009))
cbbEra.bracket.teams.forEach(t => { Object.assign(t, odds.get(t.team_id) || {final_four: 0, champion: 0}) })
renderCbbEraBracket()
const playOut = random => {
document.getElementById("cbbEraPlayed").innerHTML = cbbPlayedBracketHtml(cbbPlayBracket(cbbEra.bracket, cbbEra.byId, cbbEra.league, random), cbbEra.byId)
}
document.getElementById("cbbEraPlayRandom").onclick = () => playOut(cbbRng(Date.now()))
document.getElementById("cbbEraPlayFavorites").onclick = () => playOut(null)
}, 0)
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
