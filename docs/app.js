let wave = []

let pave = []

let confidence = []

let hitterHitPredictions = []

let playerMatches = []

let selectedTeams =
[]



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



function searchPlayer(){

const query =

document
.getElementById(
"playerSearch"
)
.value
.trim()
.toLowerCase()

const type =

document
.getElementById(
"playerType"
)
.value

const source =

type==="wave"
?
wave
:
pave

playerMatches =

source
.filter(
p=>{

const full =

`${p.name_first}
${p.name_last}`

.toLowerCase()

return full.includes(
query
)

}
)

.slice(
0,
10
)

if(
playerMatches.length===0
){

document
.getElementById(
"playerChoices"
)
.innerHTML =
"No players found"

document
.getElementById(
"playerResult"
)
.innerHTML =
""

return

}

let html =
""

playerMatches
.forEach(
(
p,
i
)=>{

html +=

`

<button

onclick="showPlayer(
${i}
)"

>

${p.name_first}
${p.name_last}

</button>

`

}

)

document
.getElementById(
"playerChoices"
)
.innerHTML =
html

}



function showPlayer(index){

const player =

playerMatches[
index
]

const playerId =

player.key_mlbam

const modelPrediction =

hitterHitPredictions.find(
p=>
p.key_mlbam
===
playerId
)

const isPave =

document
.getElementById(
"playerType"
)
.value

===

"pave"

document
.getElementById(
"playerResult"
)
.innerHTML =

`

<div
class="infoCard">

<div
class="cardHeader">

<div
style="
display:flex;
gap:16px;
align-items:center;
">

<img

src="${playerImage(playerId)}"

class="playerHeadshot"

alt="${player.name_first} ${player.name_last}"

>

<div>

<div
class="cardTitle">

${player.name_first}

${player.name_last}

</div>

<div
style="
opacity:.7;
margin-top:4px;
display:flex;
align-items:center;
gap:8px;
">

<img

src=
"${teamLogo(
player.team
)}"

class=
"smallTeamLogo"

onerror=
"this.style.display='none'"

>

${player.team}

•

${
isPave
?
"PAVE"
:
"WAVE"
}

</div>

</div>

</div>

</div>

<div
class="cardStats">

<div>

Team

</div>

<div>

${player.team || "-"}

</div>

${
isPave

?

`

<div>

Expected Hits

</div>

<div>

${Number(
player.Expected_Hits
||
0
)
.toFixed(
2
)}

</div>

<div>

Expected Bases

</div>

<div>

${Number(
player.Expected_Bases
||
0
)
.toFixed(
2
)}

</div>

<div>

Expected HR

</div>

<div>

${Number(
player.Expected_HRs
||
0
)
.toFixed(
2
)}

</div>

<div>

At Bats

</div>

<div>

${player.at_bats}

</div>

`

:

`

<div>

Heuristic

</div>

<div>

${(
Number(
player.Approach
||
0
)
*
100
)
.toFixed(
1
)}%

</div>

<div>

Model Odds

</div>

<div>

${
modelPrediction
?
(
Number(
modelPrediction.Model_Hit_Probability
||
0
)
*
100
)
.toFixed(
1
)+"%"
:
"-"
}

</div>

<div>

Expected Bases

</div>

<div>

${Number(
player.Expected_Bases
||
0
)
.toFixed(
2
)}

</div>

<div>

PA

</div>

<div>

${
Number(
player.PA_L
||
0
)

+

Number(
player.PA_R
||
0
)

}

</div>

<div>

Decision Score

</div>

<div>

${(
Number(
player.Decision_Score
||
0
)
)
.toFixed(
1
)}%

</div>

`

}

</div>

</div>

`

}



function loadTeamExplorer(){

const container =

document
.getElementById(
"teamButtons"
)

container.innerHTML =
""

const values =

confidence
.map(
r=>

Number(
r.Confidence
||
0
)

)

const min =

Math.min(
...values
)

const max =

Math.max(
...values
)

container.style.display =
"flex"

container.style.flexWrap =
"wrap"

container.style.gap =
"6px"

confidence.forEach(
team=>{

const val =

Number(
team.Confidence
)

const pct =

(
val
-
min
)

/

(
max
-
min
)

const red =

Math.round(
255
*
pct
)

const blue =

Math.round(
255
*
(
1
-
pct
)
)

const btn =

document.createElement(
"button"
)

btn.style.width =
"60px"

btn.style.height =
"44px"

btn.style.position =
"relative"

btn.style.border =
"none"

btn.style.borderRadius =
"8px"

btn.style.background =
"#222"

btn.style.color =
"white"

btn.style.cursor =
"pointer"

btn.innerHTML =

`

<div
style="
font-weight:bold;
">

${team.team}

</div>

<div
style="
position:absolute;
left:0;
bottom:0;
height:6px;
width:100%;
border-radius:0 0 8px 8px;

background:

linear-gradient(
90deg,

rgb(
${red},
80,
${blue}
),

rgb(
${red},
80,
${blue}
)

);

">

</div>

`

btn.onclick =
()=>{

selectTeam(
team.team
)

}

container
.appendChild(
btn
)

}

)

// Open on the highest-rated team rather than an empty "Select a team"
// panel; clicking another team then compares the two.
const top = confidence.reduce((best, r)=>Number(r.Confidence) > Number(best.Confidence) ? r : best, confidence[0])
if(top && selectedTeams.length === 0){
selectTeam(top.team)
}

}

function selectTeam(team){

selectedTeams =

selectedTeams.filter(
t=>
t!==team
)

selectedTeams.push(
team
)

if(
selectedTeams.length
>
2
){

selectedTeams.shift()

}

showTeam()

}

function playerImage(id){

if(
!id
){

return ""

}

return `https://img.mlbstatic.com/mlb-photos/image/upload/w_240,q_auto:best/v1/people/${String(id).trim()}/headshot/67/current`

}

function teamLogo(team){

const ids = {

AZ:109,
ATL:144,
BAL:110,
BOS:111,
CHC:112,
CWS:145,
CIN:113,
CLE:114,
COL:115,
DET:116,
HOU:117,
KC:118,
LAA:108,
LAD:119,
MIA:146,
MIL:158,
MIN:142,
NYM:121,
NYY:147,
ATH:133,
PHI:143,
PIT:134,
SD:135,
SF:137,
SEA:136,
STL:138,
TB:139,
TEX:140,
TOR:141,
WSH:120

}

const key =

String(
team
)

.trim()

.toUpperCase()

return ids[key]

?

`https://www.mlbstatic.com/team-logos/team-cap-on-light/${ids[key]}.svg`

:

""

}

// Plain names for the team card's stats (the CSV columns are model names).
const TEAM_STAT_LABELS = {
current: "Recent win rate",
Strength: "Win strength",
pyth_Strength: "Run-differential strength",
SOS: "Schedule strength",
pyth_SOS: "Schedule strength (by runs)",
Confidence: "Overall rating",
pyth_Confidence: "Overall rating (by runs)",
true_power: "Overall power",
offensive_edge: "Offense vs. average",
suppression_resistance: "Hitting vs. tough pitching",
home_run_reliance: "Reliance on home runs",
homer_per_game: "Home runs per game",
game_homer_rate: "Games with a home run",
team_home_run_rate: "Home runs at home",
away_hr_rate: "Home runs on the road",
Park_Factor: "Ballpark factor",
}

function showTeam(){

const results =

selectedTeams

.map(
team=>

confidence.find(
r=>

r.team===team

)

)

.filter(
Boolean
)

if(
results.length===0
){

document
.getElementById(
"teamResult"
)
.innerHTML =
""

return

}

const values =

confidence.map(
r=>

Number(
r.Confidence
)

)

const min =

Math.min(
...values
)

const max =

Math.max(
...values
)

const winning = [

"current",

"Strength",

"pyth_Strength",

"SOS",

"pyth_SOS",

"Confidence",

"pyth_Confidence"

]

const offense = [

"true_power",

"offensive_edge",

"suppression_resistance",

"home_run_reliance",

"homer_per_game",

"game_homer_rate"

]

const park = [

"team_home_run_rate",

"away_hr_rate",

"Park_Factor"

]

function rows(
result,
list
){

const other =

results.find(
r=>

r.team
!==

result.team

)

return list

.map(
k=>{

const value =

Number(
result[k]
||
0
)

let arrow =
""

if(
results.length===2
&&
other
){

const compare =

Number(
other[k]
||
0
)

if(
value
>
compare
){

arrow =

'<span class="compareUp">▲</span>'

}

else if(
value
<
compare
){

arrow =

'<span class="compareDown">▼</span>'

}

}

return `

<div>

${TEAM_STAT_LABELS[k] || k.replaceAll("_", " ")}

</div>

<div>

${value.toFixed(
3
)}

${arrow}

</div>

`

})

.join(
""

)

}

document
.getElementById(
"teamResult"
)
.innerHTML =

results

.map(
result=>{

const pct =

(

Number(
result.Confidence
)

-

min

)

/

(

max

-

min

)

const color =

`rgb(

${Math.round(
255*pct
)},

80,

${Math.round(
255*(1-pct)
)}

)`

return `

<div
class="infoCard">

<div
class="cardHeader">

<div
style="
display:flex;
align-items:center;
gap:12px;
">

<img

src=
"${teamLogo(
result.team
)}"

class=
"teamLogo"

onerror=
"this.style.display='none'"

>

<div
class="cardTitle">

${result.team}

</div>

</div>

<div
class="cardBadge"

style="
background:
${color}
"

>

</div>

</div>

<div
class="profileColumns">

<div
class="profileColumn">

<div
class="profileHeader">

Winning Profile

</div>

<div
class="profileRows">

${rows(
result,
winning
)}

</div>

</div>

<div
class="profileColumn">

<div
class="profileHeader">

Offensive Profile

</div>

<div
class="profileRows">

${rows(
result,
offense
)}

</div>

</div>

<div
class="profileColumn">

<div
class="profileHeader">

Ballpark Profile

</div>

<div
class="profileRows">

${rows(
result,
park
)}

</div>

</div>

</div>

</div>

`

}

)

.join(
""

)

}

// Display helpers for the leaderboards and history tables: the CSVs keep
// full precision for the models, but a reader needs a few significant
// digits, plain-English headers and no internal ids.
function fmtNum(v, digits){
return v === undefined || v === null || v === "" || Number.isNaN(Number(v)) ? "-" : Number(v).toFixed(digits)
}

function fmtPctRange(v, low, high){
const p = x=>(Number(x) * 100).toFixed(1)
if(v === undefined || v === null || v === "" || Number.isNaN(Number(v))){ return "-" }
if(low === undefined || low === "" || high === undefined || high === ""){ return p(v) + "%" }
return `${p(v)}% <span class="dim">(${p(low)}–${p(high)})</span>`
}

function waveLeaderboardRow(p){
return {
"Player": `${p.name_first} ${p.name_last}`,
"Team": teamName(p.team, "mlb"),
"Plate appearances": Math.round(Number(p.PA_L || 0) + Number(p.PA_R || 0)),
// No range here: WAVE_CI_Low/High describe the raw full-season hit rate,
// not WAVE's recency blend (hitters.compute_wave), so beside WAVE they can
// even exclude it (Moreno: 23.6% shown against 23.7-31.2%).
"Hit rate": fmtPctRange(p.WAVE),
"Chance of a hit today": fmtPctRange(p.Game_Hit_Probability),
"Barrel rate": fmtPctRange(p.Barrel_Rate),
}
}

// "+78%" / "-12%" from a ratio to league average (1.78, 0.88).
function vsAverage(ratio){
const n = Number(ratio)
if(ratio === undefined || ratio === "" || Number.isNaN(n)){ return "-" }
const pct = Math.round((n - 1) * 100)
return `${pct > 0 ? "+" : ""}${pct}%`
}

function paveLeaderboardRow(p){
return {
"Pitcher": `${p.name_first} ${p.name_last}`,
"Team": teamName(p.team, "mlb"),
"Throws": p.Throws === "L" ? "Left" : p.Throws === "R" ? "Right" : "-",
"At-bats faced": p.at_bats,
"Hit rate allowed": fmtPctRange(p.PAVE),
"vs. average pitcher": vsAverage(p.PAVE_PLUS),
}
}

async function loadAll(){

wave =
await loadCSV(
"./data/wave.csv"
)

pave =
await loadCSV(
"./data/pave.csv"
)

confidence =
await loadCSV(
"./data/confidence.csv"
)

confidence =

confidence

.sort(
(
a,
b
)=>

String(
a.team
)
.localeCompare(

String(
b.team
)

)

)

const meanPA =

wave
.reduce(
(
sum,
p
)=>

sum

+

(

Number(
p.PA_L
||
0
)

+

Number(
p.PA_R
||
0
)

),

0

)

/

wave.length

const filteredWave =

wave
.filter(
p=>

(

Number(
p.PA_L
||
0
)

+

Number(
p.PA_R
||
0
)

)

>=

meanPA

)

buildTable(

filteredWave.slice(0, 20).map(waveLeaderboardRow),

"waveTable"

)

const maxAB =

Math.max(

...

pave.map(
p=>

Number(
p.at_bats
||
0
)

)

)

const minAB =

maxAB
*
0.25

const filteredPave =

pave
.filter(
p=>

Number(
p.at_bats
||
0
)

>=

minAB

)

buildTable(

filteredPave.slice(0, 20).map(paveLeaderboardRow),

"paveTable"

)

loadTeamExplorer()

}

function selectBtsView(view){
document.querySelectorAll("#btsTabs .tabButton").forEach(btn=>{
btn.classList.toggle("active", btn.dataset.view === view)
})
document.getElementById("ourPicksSection").style.display = view === "ourPicks" ? "" : "none"
document.getElementById("hitStreaksSection").style.display = view === "hitStreaks" ? "" : "none"
document.getElementById("modelOddsSection").style.display = view === "modelOdds" ? "" : "none"
}

async function loadBeatTheStreak(){

let summary = []
let picks = []

try{
summary = await loadCSV("./data/beat_the_streak_summary.csv")
}catch(e){
console.log("no beat_the_streak_summary.csv yet", e)
}

try{
picks = await loadCSV("./data/beat_the_streak_picks.csv")
}catch(e){
console.log("no beat_the_streak_picks.csv yet", e)
}

renderStreakStats(summary)
renderTodaysPicks(picks)
renderStreakHistory(picks)

}

function renderStreakStats(summary){

const el = document.getElementById("streakStats")

if(!summary.length){
el.innerHTML = "No picks tracked yet"
return
}

const s = summary[0]
const days = n=>`${n} day${Number(n) === 1 ? "" : "s"}`

const stat = (value, label, sub) =>
`<div class="streakStat"><div class="value">${value}</div><div class="label">${label}</div>${sub ? `<div class="sub">${sub}</div>` : ""}</div>`

el.innerHTML =
stat(days(s.current_streak || 0), "Hit streak", "in a row with a hit") +
stat(days(s.longest_streak || 0), "Best streak", "our longest so far") +
stat(srPct(s.day_survival_rate), "Days with a hit", `over ${days(s.n_days_resolved || 0)}`)

}

function renderTodaysPicks(picks){

const el = document.getElementById("todaysPicks")

// A date with ZERO logged candidates at all (a genuine off day - an
// All-Star break, a rainout across the whole slate, or a pipeline gap)
// still gets a "no_pick" row (see evaluation.build_beat_the_streak_export)
// so the latest date here is always the actual most recent day the
// pipeline ran. A weak-slate day no longer gets this treatment - it shows
// its real top candidates graded "speculative" instead (see below).
if(!picks.length){
el.innerHTML = "No picks tracked yet"
return
}

const latestDate = picks
.map(p=>p.date)
.sort()
.slice(-1)[0]

const todays = picks
.filter(p=>p.date === latestDate && p.status !== "no_pick")
.sort((a,b)=>Number(a.rank) - Number(b.rank))

if(!todays.length){
el.innerHTML = `<div class="pickCard no_pick"><div class="pickName">No games or eligible batters for ${latestDate}</div></div>`
return
}

const statusLabels = {
hit: "Got a hit",
miss: "No hit",
no_game: "Didn't play",
pending: "Game not finished yet",
}

const gradeLabels = {
recommended: "Strong pick",
speculative: "Weaker pick",
}

el.innerHTML = todays
.map(p=>{

const prob =
p.predicted_probability && p.predicted_probability !== ""
? srPct(p.predicted_probability) + " to get a hit"
: ""

// "recommended" reuses the same highlight ring the Automated Game
// Picks cards already use for a confident pick (see .pickCard.recommended
// in style.css); "speculative" is a real, already-qualified candidate
// that just didn't clear the recommendation bar - shown, not hidden,
// but visually dimmed and explicitly labeled rather than blended in
// as if it were equally confident.
return `
<div class="pickCard ${p.status} ${p.grade || ""}">
<div class="pickName">${p.name}</div>
<div class="pickGrade ${p.grade || ""}">${gradeLabels[p.grade] || ""}</div>
<div class="pickProb">${prob}</div>
<div class="pickStatus">${statusLabels[p.status] || p.status}</div>
</div>
`

})
.join("")

}

function renderStreakHistory(picks){

const sorted = picks
.slice()
.sort((a,b)=>b.date.localeCompare(a.date) || Number(a.rank) - Number(b.rank))

const results = {hit: "✓ Got a hit", miss: "✗ No hit", no_game: "Didn't play", pending: "Not finished", no_pick: "No pick"}
const grades = {recommended: "Strong", speculative: "Weaker"}

buildTable(
sorted.slice(0, 100).map(p=>({
"Date": p.date,
"Player": p.name || "-",
"Chance of a hit": srPct(p.combined_probability),
"Result": results[p.status] || p.status,
"Pick": grades[p.grade] || "-",
})),
"streakHistoryTable"
)

}

async function loadHitStreaks(){

let streaks = []

try{
streaks = await loadCSV("./data/hit_streaks.csv")
}catch(e){
console.log("no hit_streaks.csv yet", e)
}

renderHitStreaks(streaks)

}

function renderHitStreaks(streaks){

if(!streaks.length){
document.getElementById("hitStreaksTable").innerHTML = "No hit streak data yet"
return
}

const top = streaks
.slice()
.sort((a,b)=>Number(b.Current_Hit_Streak) - Number(a.Current_Hit_Streak))
.slice(0, 10)
.map(p=>({
"Player": `${p.name_first} ${p.name_last}`,
"Team": teamName(p.team, "mlb"),
"Games in a row with a hit": p.Current_Hit_Streak,
}))

buildTable(top, "hitStreaksTable")

}

async function loadModelOdds(){

try{
hitterHitPredictions = await loadCSV("./data/hitter_hit_predictions.csv")
}catch(e){
console.log("no hitter_hit_predictions.csv yet", e)
}

renderModelOdds(hitterHitPredictions)

}

function renderModelOdds(predictions){

if(!predictions.length){
document.getElementById("modelOddsTable").innerHTML = "No chances posted yet today"
return
}

const top = predictions
.slice()
.sort((a,b)=>Number(b.Model_Hit_Probability) - Number(a.Model_Hit_Probability))
.slice(0, 10)
.map(p=>({
"Player": `${p.name_first} ${p.name_last}`,
"Team": teamName(p.team, "mlb"),
"Game": `${p.is_home === "True" || p.is_home === "true" ? "vs." : "at"} ${teamName(p.opponent, "mlb")}`,
"Chance of a hit": srPct(p.Model_Hit_Probability),
}))

buildTable(top, "modelOddsTable")

}

async function loadGamePicks(){

let summary = []
let picks = []

let byVersion = []

try{
summary = await loadCSV("./data/game_picks_summary.csv")
}catch(e){
console.log("no game_picks_summary.csv yet", e)
}

try{
byVersion = await loadCSV("./data/game_picks_summary_by_version.csv")
}catch(e){
console.log("no game_picks_summary_by_version.csv yet", e)
}

try{
picks = await loadCSV("./data/game_picks_picks.csv")
}catch(e){
console.log("no game_picks_picks.csv yet", e)
}

const allTime = summary[0]
const bettingPaused = Boolean(allTime) && String(allTime.betting_enabled).toLowerCase() === "false"
if(bettingPaused){
renderPausedGamePickStats(allTime, currentVersionRow(byVersion))
}else{
renderGamePickStats(summary)
}
renderTodaysGamePicks(picks, bettingPaused)
renderGamePickHistory(picks)

}

// The by-version summary holds "all_time" plus one row for the model now
// making picks (config.GAME_PICK_MODEL_VERSION).
function currentVersionRow(byVersion){
return byVersion.find(r=>r.model_version && r.model_version !== "all_time") || null
}

function fmtPct1(v){
return v === undefined || v === null || v === "" || Number.isNaN(Number(v)) ? "-" : (Number(v) * 100).toFixed(1) + "%"
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
stat(`${ours} of ${n}`, "Our picks right", early) +
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

function renderTodaysGamePicks(picks, bettingPaused){

const el = document.getElementById("todaysGamePicks")

if(!picks.length){
el.innerHTML = "No game picks yet"
return
}

const latestDate = picks
.map(p=>p.date)
.sort()
.slice(-1)[0]

const todays = picks
.filter(p=>p.date === latestDate)
.sort((a,b)=>Number(b.predicted_probability) - Number(a.predicted_probability))

if(!todays.length){
el.innerHTML = "No games today"
return
}

const statusLabels = {
win: "We were right",
loss: "We were wrong",
not_played: "Postponed",
pending: "Not played yet",
}

el.innerHTML = todays
.map(p=>{

const away = teamName(p.away_team, "mlb")
const home = teamName(p.home_team, "mlb")
const pick = teamName(p.predicted_winner, "mlb")
const matchup = p.predicted_winner === p.home_team ? `${away} at <b>${home}</b>` : `<b>${away}</b> at ${home}`
const vegas = fmtPct1(p.market_predicted_winner_probability) !== "-" ? ` &middot; Vegas ${srPct(p.market_predicted_winner_probability)}` : ""

// bet_units > 0 is the "was a bet suggested" signal.
const betUnits = Number(p.bet_units)
const betAdvised = betUnits > 0
const betLine = betAdvised
? `<div class="pickStatus">Bet ${teamName(p.bet_team, "mlb")} &middot; ${betUnits.toFixed(1)} units</div>`
: ""

return `
<div class="pickCard ${p.status}${betAdvised ? " recommended" : ""}">
<div class="pickName">${matchup}</div>
<div class="pickProb">We say ${pick} ${srPct(p.predicted_probability)}${vegas}</div>
<div class="pickStatus">${statusLabels[p.status] || p.status}</div>
${betLine}
</div>
`

})
.join("")

}

function renderGamePickHistory(picks){

const sorted = picks
.slice()
.sort((a,b)=>b.date.localeCompare(a.date) || Number(b.predicted_probability) - Number(a.predicted_probability))

const results = {
win: "✓ Right",
loss: "✗ Wrong",
not_played: "Postponed",
pending: "Not played yet",
}

// A "Bet" column only when a row in view actually had a bet; while bet
// advice is off, a column of dashes would just crowd the useful ones.
const shown = sorted.slice(0, 100)
const anyBets = shown.some(p=>Number(p.bet_units) > 0)
const betCell = p=>{
if(!(Number(p.bet_units) > 0)){ return "-" }
const units = Number(p.bet_units).toFixed(1)
const profit = Number(p.bet_profit_units)
const settled = p.bet_profit_units !== undefined && p.bet_profit_units !== "" && !Number.isNaN(profit)
? (profit >= 0 ? `, won ${profit.toFixed(1)}` : `, lost ${Math.abs(profit).toFixed(1)}`)
: ""
return `${teamName(p.bet_team, "mlb")} ${units} units${settled}`
}
const formatted = shown.map(p=>{
const row = {
"Date": p.date,
"Game": `${teamName(p.away_team, "mlb")} at ${teamName(p.home_team, "mlb")}`,
"Our pick": teamName(p.predicted_winner, "mlb"),
"We said": srPct(p.predicted_probability),
"Vegas said": fmtPct1(p.market_predicted_winner_probability) === "-" ? "-" : srPct(p.market_predicted_winner_probability),
"Result": results[p.status] || p.status,
}
if(anyBets){ row["Bet"] = betCell(p) }
return row
})

buildTable(
formatted,
"gamePickHistoryTable"
)

}

async function loadProbablePitchers(){

let rows = []

try{
rows = await loadCSV("./data/probable_pitchers.csv")
}catch(e){
console.log("no probable_pitchers.csv yet", e)
}

const el = document.getElementById("probablePitchersTable")

if(!rows.length){
el.innerHTML = "No probable pitchers yet"
return
}

const formatted = rows.map(r=>({
"Team": teamName(r.team, "mlb"),
"Game": `${r.is_home === "True" || r.is_home === "true" ? "vs." : "at"} ${teamName(r.opponent, "mlb")}`,
"Starting pitcher": r.pitcher_name || "Not announced",
"Throws": r.Throws === "L" ? "Left" : r.Throws === "R" ? "Right" : "-",
"Hit rate allowed": r.PAVE && r.PAVE !== "" ? fmtPctRange(r.PAVE) : "-",
"vs. average pitcher": r.PAVE_PLUS && r.PAVE_PLUS !== "" ? vsAverage(r.PAVE_PLUS) : "-",
}))

buildTable(
formatted,
"probablePitchersTable"
)

}

loadAll()

loadBeatTheStreak()

loadHitStreaks()

loadModelOdds()

loadGamePicks()

loadProbablePitchers()
