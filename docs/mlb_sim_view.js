// MLB Picks page: the pitch-by-pitch game simulation (docs/mlb_sim.js).
// Pick the teams, starting pitchers and batting orders; the game plays out
// over about five minutes - line score, count and runners, each pitch in
// the strike zone, the play-by-play - and ends with the box score. The
// player profiles (docs/data/mlb_sim_profiles.json, ~0.5 MB) load only
// when someone opens the simulator.

const MLB_SIM_SECONDS = 300
const MLB_SIM_SPEEDS = [1, 4, 16]
const mlbSim = {profiles: null, setup: null, run: null, probables: [], picks: []}
const MLB_SIM_SUMMARY_GAMES = 1000

function mlbSimEl(){
return document.getElementById("mlbSim")
}

async function mlbSimLoadProfiles(){
if(window.SITE_DATA && window.SITE_DATA.mlb_sim_profiles){ return window.SITE_DATA.mlb_sim_profiles }
const response = await fetch(`./data/mlb_sim_profiles.json?t=${Date.now()}`, {cache: "no-store"})
if(!response.ok){ throw new Error(`HTTP ${response.status}`) }
return response.json()
}

function mlbSimStart(){
const el = mlbSimEl()
if(!el){ return }
el.innerHTML = `<p class="simIntro">Pick two teams, their starting pitchers and batting orders, and watch the game play out pitch by pitch - about five minutes, with the box score at the end - or play it 1,000 times at once for how often each side wins and what each player does on average. Every pitch comes from the pitcher's real pitch mix and each matchup's real tendencies.</p>
<button class="whatIfAction" data-mlbsim="open">Set up a game</button>`
el.addEventListener("click", onMlbSimClick)
el.addEventListener("change", onMlbSimChange)
}

async function mlbSimOpen(){
const el = mlbSimEl()
el.innerHTML = `<p class="simIntro">Loading players…</p>`
try{
mlbSim.profiles = await mlbSimLoadProfiles()
}catch(e){
console.log("no mlb_sim_profiles.json yet", e)
el.innerHTML = `<p class="simIntro">The simulator's player data isn't available yet.</p>`
return
}
try{
mlbSim.probables = typeof loadCSV === "function" ? await loadCSV("./data/probable_pitchers.csv") : []
}catch(e){
mlbSim.probables = []
}
try{
mlbSim.picks = typeof loadCSV === "function" ? await loadCSV("./data/game_picks_picks.csv") : []
}catch(e){
mlbSim.picks = []
}
mlbSim.setup = mlbSimDefaultSetup()
mlbSimRenderSetup()
}

// The latest scheduled game with probable starters, or two teams.
function mlbSimDefaultSetup(){
const teams = Object.keys(mlbSim.profiles.teams || {}).sort()
const rows = (mlbSim.probables || []).filter(r => r.team && r.opponent)
const latest = rows.map(r => r.date).sort().pop()
const game = rows.find(r => r.date === latest && String(r.is_home).toLowerCase() === "true" && mlbSim.profiles.teams[r.team] && mlbSim.profiles.teams[r.opponent])
const side = (team, probable) => {
const roster = mlbSim.profiles.teams[team] || {}
const listed = probable && probable.key_mlbam ? String(Math.round(Number(probable.key_mlbam))) : null
return {team, starter: listed && mlbSim.profiles.pitchers[listed] ? listed : mlbSimRotationNext(team), lineup: (roster.lineup || []).map(String)}
}
if(game){
const awayRow = rows.find(r => r.date === latest && r.team === game.opponent)
return {home: side(game.team, game), away: side(game.opponent, awayRow)}
}
return {away: side(teams[0]), home: side(teams[1])}
}

// The next starter up: of the team's recent real starters, the one whose
// last start was longest ago.
function mlbSimRotationNext(team){
const starters = ((mlbSim.profiles.teams[team] || {}).starters || []).filter(s => s.starts >= 3 && mlbSim.profiles.pitchers[String(s.id)])
const rotation = starters.slice(0, 5).sort((a, b) => a.last.localeCompare(b.last))
const pick = rotation[0] || ((mlbSim.profiles.teams[team] || {}).starters || [])[0]
return pick ? String(pick.id) : null
}

function mlbSimName(id){
const p = mlbSim.profiles.batters[id] || mlbSim.profiles.pitchers[id]
return p && p.name && !/^\d+$/.test(p.name) ? p.name : `#${id}`
}

function mlbSimTeamName(code){
return typeof teamName === "function" ? teamName(code, "mlb") : code
}

function mlbSimSideHtml(side){
const s = mlbSim.setup[side]
const P = mlbSim.profiles
const roster = P.teams[s.team] || {}
const teams = Object.keys(P.teams).sort()
const teamSelect = `<select data-mlbsim-team="${side}">${teams.map(t => `<option value="${t}"${t === s.team ? " selected" : ""}>${mlbEsc(mlbSimTeamName(t))}</option>`).join("")}</select>`
const starters = (roster.starters || []).filter(x => P.pitchers[String(x.id)])
const pen = (roster.bullpen || []).filter(x => P.pitchers[String(x.id)] && !starters.some(st => st.id === x.id))
const opt = (id, label) => `<option value="${id}"${String(id) === String(s.starter) ? " selected" : ""}>${mlbEsc(label)}</option>`
const starterSelect = `<select data-mlbsim-starter="${side}">
<optgroup label="Starters">${starters.map(x => opt(x.id, `${mlbSimName(String(x.id))} (${P.pitchers[String(x.id)].throws}HP, last start ${x.last.slice(5)})`)).join("")}</optgroup>
${pen.length ? `<optgroup label="Bullpen (as an opener)">${pen.map(x => opt(x.id, `${mlbSimName(String(x.id))} (${P.pitchers[String(x.id)].throws}HP)`)).join("")}</optgroup>` : ""}
</select>`
const hitters = [...new Set((roster.hitters || []).concat(roster.lineup || []).map(String))].filter(id => P.batters[id])
.sort((a, b) => mlbSimName(a).localeCompare(mlbSimName(b)))
const bats = id => {
const b = P.batters[id].bats || {}
return b.L !== b.R ? "S" : b.R
}
const slots = s.lineup.map((id, i) => `<label class="mlbSlot"><span>${i + 1}</span><select data-mlbsim-slot="${side}|${i}">${
hitters.map(h => `<option value="${h}"${h === id ? " selected" : ""}>${mlbEsc(mlbSimName(h))} (${bats(h)})</option>`).join("")}</select></label>`).join("")
return `<div class="mlbSetupSide"><h4>${side === "home" ? "Home" : "Away"}</h4>
<label class="whatIfField">Team ${teamSelect}</label>
<label class="whatIfField">Starting pitcher ${starterSelect}</label>
<div class="whatIfOutLabel">Batting order</div><div class="mlbSlots">${slots}</div>
<button class="whatIfAction mlbSmall" data-mlbsim="regulars|${side}">Back to the regulars</button></div>`
}

function mlbSimRenderSetup(note){
mlbSimStop()
mlbSim.run = null
const P = mlbSim.profiles
const dupes = ["away", "home"].some(side => new Set(mlbSim.setup[side].lineup).size < 9)
mlbSimEl().innerHTML = `<p class="simIntro">Players and tendencies through ${mlbEsc(P.through)}. Relievers come in on their own - the closer for a save, the setup man in the 8th of a close game.</p>
<div class="mlbSetup">${mlbSimSideHtml("away")}${mlbSimSideHtml("home")}</div>
${dupes ? `<p class="statusNote">A player is in a batting order twice.</p>` : ""}${note ? `<p class="statusNote">${mlbEsc(note)}</p>` : ""}
<div class="simControls"><button class="whatIfAction" data-mlbsim="play"${dupes ? " disabled" : ""}>Play ball</button>
<button class="whatIfAction" data-mlbsim="summary"${dupes ? " disabled" : ""}>Simulate it ${MLB_SIM_SUMMARY_GAMES.toLocaleString()} times</button></div>`
}

function onMlbSimChange(event){
const t = event.target
if(!mlbSim.setup){ return }
if(t.dataset.mlbsimTeam){
const side = t.dataset.mlbsimTeam
const roster = mlbSim.profiles.teams[t.value] || {}
mlbSim.setup[side] = {team: t.value, starter: mlbSimRotationNext(t.value), lineup: (roster.lineup || []).map(String)}
mlbSimRenderSetup()
}else if(t.dataset.mlbsimStarter){
mlbSim.setup[t.dataset.mlbsimStarter].starter = t.value
}else if(t.dataset.mlbsimSlot){
const [side, i] = t.dataset.mlbsimSlot.split("|")
const lineup = mlbSim.setup[side].lineup
// Picking someone already in the order swaps the two.
const was = lineup.indexOf(t.value)
if(was >= 0 && was !== Number(i)){ lineup[was] = lineup[Number(i)] }
lineup[Number(i)] = t.value
mlbSimRenderSetup()
}
}

function onMlbSimClick(event){
const t = event.target.closest("button")
if(!t){ return }
const action = t.dataset.mlbsim
const run = mlbSim.run
if(action === "open"){ mlbSimOpen() }
else if(action && action.startsWith("regulars|")){
const side = action.split("|")[1]
mlbSim.setup[side].lineup = ((mlbSim.profiles.teams[mlbSim.setup[side].team] || {}).lineup || []).map(String)
mlbSimRenderSetup()
}
else if(action === "play"){ mlbSimPlay() }
else if(action === "summary"){ mlbSimSummary() }
else if(action === "setup"){ mlbSimRenderSetup() }
else if(!run){ return }
else if(action === "pause"){
run.paused = !run.paused
t.textContent = run.paused ? "Resume" : "Pause"
mlbSimStop()
mlbSimSchedule()
}else if(action === "skip"){
mlbSimStop()
run.index = run.game.pitches.length - 1
mlbSimFrame()
}else if(t.dataset.mlbsimSpeed){
run.speed = Number(t.dataset.mlbsimSpeed)
t.parentElement.querySelectorAll("[data-mlbsim-speed]").forEach(b => b.classList.toggle("active", b === t))
mlbSimStop()
mlbSimSchedule()
}
}

function mlbSimPlay(){
const setup = JSON.parse(JSON.stringify(mlbSim.setup))
const game = simulateMlbGame(mlbSim.profiles, setup, {seed: Math.floor(Math.random() * 2147483647)})
mlbSim.run = {game, setup, index: 0, speed: mlbSim.run && mlbSim.run.speed ? mlbSim.run.speed : 1, paused: false, timer: null}
const run = mlbSim.run
const speeds = MLB_SIM_SPEEDS.map(s => `<button class="tabButton${s === run.speed ? " active" : ""}" data-mlbsim-speed="${s}">${s}×</button>`).join("")
mlbSimEl().innerHTML = `<h4>${mlbEsc(mlbSimTeamName(setup.away.team))} at ${mlbEsc(mlbSimTeamName(setup.home.team))}</h4>
<div class="simControls"><div class="tabGroup">${speeds}</div>
<button class="whatIfAction" data-mlbsim="pause">Pause</button>
<button class="whatIfAction" data-mlbsim="skip">Skip to the final</button>
<button class="whatIfAction" data-mlbsim="setup">Change the lineups</button></div>
<div class="mlbLive"></div>`
mlbSimFrame()
mlbSimSchedule()
}

// Our pick model's chance for this matchup, when it's on the latest slate.
function mlbSimModelPick(setup){
const rows = (mlbSim.picks || []).filter(r => r.home_team === setup.home.team && r.away_team === setup.away.team)
if(!rows.length){ return null }
const row = rows.sort((a, b) => String(a.date).localeCompare(String(b.date))).pop()
const latest = (mlbSim.picks || []).map(r => r.date).sort().pop()
const p = Number(row.predicted_probability)
return row.date === latest && Number.isFinite(p) ? {team: row.predicted_winner, probability: p} : null
}

function mlbSimSummary(){
const setup = JSON.parse(JSON.stringify(mlbSim.setup))
mlbSimStop()
const el = mlbSimEl()
el.innerHTML = `<h4>${mlbEsc(mlbSimTeamName(setup.away.team))} at ${mlbEsc(mlbSimTeamName(setup.home.team))}</h4><p class="simIntro">Playing ${MLB_SIM_SUMMARY_GAMES.toLocaleString()} games…</p>`
// Let the message paint before the second or so of games.
setTimeout(() => {
const summary = simulateMlbSummary(mlbSim.profiles, setup, MLB_SIM_SUMMARY_GAMES, Math.floor(Math.random() * 2147483647))
mlbSim.run = null
el.innerHTML = `<h4>${mlbEsc(mlbSimTeamName(setup.away.team))} at ${mlbEsc(mlbSimTeamName(setup.home.team))}, ${MLB_SIM_SUMMARY_GAMES.toLocaleString()} times</h4>
<div class="simControls"><button class="whatIfAction" data-mlbsim="play">Watch one game</button>
<button class="whatIfAction" data-mlbsim="setup">Change the lineups</button></div>
${mlbSummaryHtml(mlbSim.profiles, summary, mlbSimModelPick(setup))}`
}, 30)
}

function mlbSimStop(){
if(mlbSim.run && mlbSim.run.timer){
clearTimeout(mlbSim.run.timer)
mlbSim.run.timer = null
}
}

function mlbSimSchedule(){
const run = mlbSim.run
if(!run || run.paused || run.index >= run.game.pitches.length - 1){ return }
// A pitch that ends a plate appearance holds a beat longer.
const p = run.game.pitches[run.index]
const base = MLB_SIM_SECONDS * 1000 / run.game.pitches.length / run.speed
run.timer = setTimeout(() => {
run.timer = null
run.index += 1
mlbSimFrame()
mlbSimSchedule()
}, p.text ? base * 1.6 : base * 0.85)
}

// "G. Cole: 78 pitches" and the batter's day so far.
function mlbSimMatchup(game, pitch){
const P = mlbSim.profiles
const thrown = game.pitches.slice(0, mlbSim.run.index + 1).filter(q => q.pitcher === pitch.pitcher).length
const before = game.plays.filter(p => p.batter === pitch.batter && p.pitch < mlbSim.run.index && p.kind !== "change")
const hits = before.filter(p => ["1B", "2B", "3B", "HR"].includes(p.kind)).length
const abs = before.filter(p => !["BB"].includes(p.kind)).length
return `<div class="mlbMatchup"><span><b>${mlbEsc(mlbSimName(pitch.pitcher))}</b> ${(P.pitchers[pitch.pitcher] || {}).throws || ""}HP · ${thrown} pitches</span>
<span><b>${mlbEsc(mlbSimName(pitch.batter))}</b> · ${hits}-for-${abs} today</span></div>`
}

function mlbSimFrame(){
const run = mlbSim.run
const game = run.game
const el = mlbSimEl()
const pitch = game.pitches[run.index]
if(run.index >= game.pitches.length - 1){
el.innerHTML = `<h4>${mlbEsc(mlbSimTeamName(game.away))} at ${mlbEsc(mlbSimTeamName(game.home))}</h4>
${mlbFinalHtml(game)}
<div class="simControls"><button class="whatIfAction" data-mlbsim="play">Play it again</button>
<button class="whatIfAction" data-mlbsim="summary">Simulate it ${MLB_SIM_SUMMARY_GAMES.toLocaleString()} times</button>
<button class="whatIfAction" data-mlbsim="setup">Change the lineups</button></div>
<div class="scrollbox">${mlbLineScoreHtml(game)}</div>
<h3 class="simSubhead">Box score</h3>${mlbBoxHtml(mlbSim.profiles, game)}
<details class="simAllPlays"><summary>Every play</summary>${mlbPlaysHtml(game, game.pitches.length - 1)}</details>`
return
}
const next = pitch.text ? game.pitches[run.index + 1] : null
el.querySelector(".mlbLive").innerHTML = `<div class="scrollbox">${mlbLineScoreHtml(game, run.index)}</div>
<div class="mlbStage">
<div class="mlbStageLeft">${mlbSituationHtml(pitch, next)}${mlbSimMatchup(game, pitch)}
<div class="mlbPitch">Pitch ${game.pitches.slice(0, run.index + 1).filter(q => q.pa === pitch.pa).length}: ${mlbPitchLine(mlbSim.profiles, pitch)}</div>
${pitch.text ? `<div class="mlbResult${game.plays.find(p => p.pitch === run.index && p.scoring) ? " simScoring" : ""}">${mlbEsc(pitch.text)}</div>` : ""}
${mlbPlaysHtml(game, run.index, 6)}</div>
<div class="mlbStageRight">${mlbZoneSvg(game, run.index)}</div>
</div>`
}

if(typeof window !== "undefined" && window.document){
mlbSimStart()
}
