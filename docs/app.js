let wave = []

let pave = []

let confidence = []

let playerMatches = []

let starterA =
[]

let starterB =
[]

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

let html =
"<table>"

html +=
"<tr>"

Object
.keys(
data[0]
)
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

Object
.values(
row
)
.forEach(
v=>
html +=
`<td>${v}</td>`
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
class="cardTitle">

${player.name_first}

${player.name_last}

</div>

<div
style="
font-size:14px;
opacity:.7;
">

${
isPave
?
"PAVE"
:
"WAVE"
}

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

Probability

</div>

<div>

${(
Number(
player.probability
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

Game Hit

</div>

<div>

${(
Number(
player.Game_Hit_Probability
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

"away_hr_rate"

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

${k.replaceAll(
"_",
" "
)}

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


function searchPitcher(side){

const query =

document
.getElementById(
`pitcherSearch${side}`
)
.value
.trim()
.toLowerCase()

const container =

document
.getElementById(
`pitcherChoices${side}`
)

container.innerHTML =
""

if(
query.length
<
2
){

return

}

const matches =

pave
.filter(
p=>{

const full =

`${p.name_first} ${p.name_last}`
.toLowerCase()

return full.includes(
query)

}

)

.slice(
0,
10)

matches.forEach(
p=>{

const full =

`${p.name_first} ${p.name_last}`

const btn =
document.createElement(
"button"
)

btn.textContent =
full

btn.addEventListener(
"click",

()=>{

selectPitcher(
side,
full
)

}

)

container.appendChild(
btn
)

}

)

}



function selectPitcher(
side,
name
){

if(
side==="A"
){

starterA.push(
String(
name
)
)
  
}

else{

starterB.push(
String(
name
)
)

}

document
.getElementById(
`pitcherSelected${side}`
)
.innerHTML =

`
✓ Selected:
<b>${name}</b>
`

document
.getElementById(
`pitcherSearch${side}`
)
.value =
""

document
.getElementById(
`pitcherChoices${side}`
)
.innerHTML =
""

const player =

pave.find(
p=>

`${p.name_first} ${p.name_last}`

===

name

)

const stats =

document
.getElementById(
`pitcherStats${side}`
)

if(
!player
){

stats.innerHTML =
""

return

}

const existing =

stats.querySelector(

`[data-name="${name}"]`

)

if(
existing
){

existing.remove()

}

const card =

document
.createElement(
"div"
)

card.className =
"pitcherCard"

card.dataset.name =
name

card.innerHTML =

`

<div
style="
display:flex;
justify-content:space-between;
align-items:center;
margin-bottom:8px;
">

<b>

${name}

</b>

<button
type="button"
class="removePitcher">

×

</button>

</div>

<table>

<tr>

<td>

Hits

</td>

<td>

${player.Expected_Hits}

</td>

</tr>

<tr>

<td>

Bases

</td>

<td>

${player.Expected_Bases}

</td>

</tr>

<tr>

<td>

HR

</td>

<td>

${player.Expected_HRs}

</td>

</tr>

</table>

`

card
.querySelector(
".removePitcher"
)

.onclick =

()=>{

card.remove()

if(
side==="A"
&&
starterA===name
){

starterA =

starterA.filter(
x=>
x!==name
)

}

if(
side==="B"
&&
starterB===name
){

starterB =

starterB.filter(
x=>
x!==name
)

}

}

stats.appendChild(
card
)
}



function getPitchAdjustment(name){

if(
name===
"League Average"
){

return 1

}

const player =

pave.find(
p=>

`${p.name_first} ${p.name_last}`

===

name

)

if(
!player
){

return 1

}

const hits =

Number(
player.Expected_Hits
||
4.8
)

const bases =

Number(
player.Expected_Bases
||
10.5
)

/*
League average starter:
4.8 hits
10.5 bases
*/

const hitFactor =
hits
/
4.8

const baseFactor =
bases
/
10.5

return(

hitFactor
*
0.5

+

baseFactor
*
0.5

)

}

function getHRAdjustment(
offense,
starter,
parkFactor
){

if(
starter===
"League Average"
){

return 1

}

const pitcher =

pave.find(
p=>

`${p.name_first} ${p.name_last}`

===

starter

)

if(
!pitcher
){

return 1

}

const expectedHR =

Number(
pitcher.Expected_HRs
||
1
)

/*
normalize to average starter
*/

const pitcherFactor =

expectedHR
/
1

const reliance =

Number(
offense.home_run_reliance
||
0.3
)

const bonus =

1
+

(

(
pitcherFactor
-
1
)
*
0.35

+

(
reliance
-
0.3
)
*
0.30

+

(
parkFactor
-
1
)
*
0.35

)

return Math.max(
0.85,
Math.min(
1.15,
bonus
)
)

}

function showHitters(
side
){

const team =

document
.getElementById(
`team${side}`
)
.value

const target =

document
.getElementById(
`team${side}Hitters`
)

if(
!team
){

target.innerHTML =
""

return

}

const hitters =

wave

.filter(
p=>

String(
p.team
)
.trim()

===

team

)

.map(
p=>({

...p,

PA:

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

})

)

.sort(
(
a,
b
)=>

b.PA
-
a.PA
)

.slice(
0,
10)

if(
!hitters.length
){

target.innerHTML =
"No hitters"

return

}

buildTable(
hitters,
target.id
)

}

function confidenceLabel(prob){

const pct =
prob
*
100

if(
pct
>=
73
){

return(
"🟡 Dominant"
)

}

if(
pct
>=
68
){

return(
"🟣 Heavy Favorite"
)

}

if(
pct
>=
63
){

return(
"🔵 Favorite"
)

}

if(
pct
>=
58
){

return(
"🟩 Strong Lean"
)

}

if(
pct
>=
53
){

return(
"🟢 Lean"
)

}

if(
pct
>=
47
){

return(
"⚪ Toss Up"
)

}

if(
pct
>=
40
){

return(
"🟠 Fade"
)

}

if(
pct
>=
35
){

return(
"🔴 Strong Fade"
)

}

if(
pct
>=
30
){

return(
"🟤 Underdog"
)

}

return(
"⚫ Heavy Underdog"
)

}



function americanOdds(prob){

if(
isNaN(
prob
)
){

return "—"

}

if(
prob
<=
0
){

return "—"

}

if(
prob
>=
1
){

return "—"

}

if(
prob
>
0.5
){

return(

"-"

+

Math.round(

(
prob
/

(
1
-
prob
)

)

*
100

)

)

}

return(

"+"

+

Math.round(

(
1
-
prob
)

/
prob

*
100

)

)

}

function predictOdds(){

const games =

Math.max(

starterA.length,

starterB.length,

1

)

let allGames =
""

let allProbs =
[]
  
const A =

confidence.find(
r=>

r.team===

document
.getElementById(
"teamA"
)
.value

)

const B =

confidence.find(
r=>

r.team===

document
.getElementById(
"teamB"
)
.value

)

const baseA =

(
Number(
A.Confidence
)

+

Number(
A.pyth_Confidence
)

+

Number(
A.true_power
)

)
/3

const baseB =

(
Number(
B.Confidence
)

+

Number(
B.pyth_Confidence
)

+

Number(
B.true_power
)

)
/3

for(
let game=0;
game<games;
game++
){

const currentStarterA =

starterA[
game
]

||
"League Average"

const currentStarterB =

starterB[
game
]

||
"League Average"

const pitchA =

getPitchAdjustment(
currentStarterB
)

const pitchB =

getPitchAdjustment(
currentStarterA
)

const parkFactor =

(

Number(
A.team_home_run_rate
)

+

Number(
A.away_hr_rate
)

)

/*
league avg
*/

/
2.1

const hrA =

getHRAdjustment(
A,
currentStarterB,
parkFactor
)

const hrB =

getHRAdjustment(
B,
currentStarterA,
parkFactor
)

function getPitcher(name){

if(
name===
"League Average"
){

return {

Expected_Bases:10,

Expected_HRs:1

}

}

return (

pave.find(
p=>

`${p.name_first} ${p.name_last}`

===

name

)

||

{

Expected_Bases:10,

Expected_HRs:1

}

)

}

const scoreA =

baseA

*

(

0.65

+

0.20
*
(
pitchA
-
1
)

+

0.15
*
(
hrA
-
1
)

)

const scoreB =

baseB

*

(

0.65

+

0.20
*
(
pitchB
-
1
)

+

0.15
*
(
hrB
-
1
)

)
  
const probA =

scoreA
/
(
scoreA
+
scoreB
)

const probB =
1
-
probA

const winner =

probA
>=
probB

?
"A"

:
"B"

const favored =

winner==="A"
?
A
:
B

const dog =

winner==="A"
?
B
:
A

const favPitch =

getPitcher(
winner==="A"
?
currentStarterA
:
currentStarterB
)

const dogPitch =

getPitcher(
winner==="A"
?
currentStarterB
:
currentStarterA
)

const pitchingEdge =

(

Number(
dogPitch.Expected_Bases
||
10
)

-

Number(
favPitch.Expected_Bases
||
10
)

)

/*
positive helps favorite
*/

const hrFav =

(

Number(
dogPitch.Expected_HRs
||
1
)

*

Number(
favored.homer_per_game
||
1
)

*

Number(
favored.home_run_reliance
||
0.3
)

*

parkFactor

)

const hrDog =

(

Number(
favPitch.Expected_HRs
||
1
)

*

Number(
dog.homer_per_game
||
1
)

*

Number(
dog.home_run_reliance
||
0.3
)

*

parkFactor

)

const factors = {

"True Power":

Number(
favored.true_power
)

-

Number(
dog.true_power
),

"Confidence":

Number(
favored.Confidence
)

-

Number(
dog.Confidence
),

"Pitching":

pitchingEdge
/
10,

"HR Environment":

hrFav
-
hrDog

}

const biggest =

Object
.entries(
factors
)

.filter(
x=>

x[1]
>
0

)

.sort(
(
a,
b
)=>

b[1]
-
a[1]

)[0]

allGames +=

`

<h2>

Game

${game+1}

</h2>

<div
class="gameRow">

<div
class="gameCol">

<div
class="gameCard">

<div
style="
display:flex;
align-items:center;
gap:10px;
margin-bottom:8px;
">

<img

src=
"${teamLogo(
A.team
)}"

class=
"teamLogo"

onerror=
"this.style.display='none'"

>

<h3>

${A.team}

</h3>

</div>

Win Probability:

<b>

${(
probA
*
100
)
.toFixed(
1
)
}%

</b>

<br>

Confidence:

${confidenceLabel(
probA
)}

<br>

Fair Line:

${americanOdds(
probA
)}

</div>

</div>

<div
class="gameCol">

<div
class="gameCard">

<div
style="
display:flex;
align-items:center;
gap:10px;
margin-bottom:8px;
">

<img

src=
"${teamLogo(
B.team
)}"

class=
"teamLogo"

onerror=
"this.style.display='none'"

>

<h3>

${B.team}

</h3>

</div>

Win Probability:

<b>

${(
probB
*
100
)
.toFixed(
1
)
}%

</b>

<br>

Confidence:

${confidenceLabel(
probB
)}

<br>

Fair Line:

${americanOdds(
probB
)}

</div>

</div>

</div>

<div
class="gameExplain">

Why the Model Likes

<b>

${favored.team}

</b>

<br><br>

True Power:

${factors[
"True Power"
]
.toFixed(
3
)}

<br>

Confidence:

${factors[
"Confidence"
]
.toFixed(
3
)}

<br>

Pitching:

${factors[
"Pitching"
]
.toFixed(
3
)}

<br>

HR Environment:

${factors[
"HR Environment"
]
.toFixed(
3
)}

<br><br>

Biggest Edge:

<b>

${biggest?.[0] || "Balanced"}

</b>

</div>

<hr>

`

}

document
.getElementById(
"oddsResult"
)
.innerHTML =

allGames

allProbs.push(
probA
)

}

document
.addEventListener(
"keydown",

e=>{

if(
e.key==="Enter"
){

if(

document
.activeElement
.id

===

"pitcherSearchA"

){

searchPitcher(
"A"
)

}

if(

document
.activeElement
.id

===

"pitcherSearchB"

){

searchPitcher(
"B"
)

}

}

}
)

function loadTeams(){

const teamA =

document
.getElementById(
"teamA"
)

const teamB =

document
.getElementById(
"teamB"
)

teamA.innerHTML =
""

teamB.innerHTML =
""

confidence.forEach(
t=>{

teamA.innerHTML +=

`

<option
value="${t.team}">

${t.team}

</option>

`

teamB.innerHTML +=

`

<option
value="${t.team}">

${t.team}

</option>

`

}

)

showHitters(
"A"
)

showHitters(
"B"
)

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

filteredWave,

"waveTable",

20

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

filteredPave,

"paveTable",

20

)

loadTeamExplorer()

loadTeams()

document
.getElementById(
"teamA"
)
.addEventListener(
"change",

()=>{

showHitters(
"A"
)

}
)

document
.getElementById(
"teamB"
)
.addEventListener(
"change",

()=>{

showHitters(
"B"
)

}
)

}

function resetMatchup(){

starterA =
"League Average"

starterB =
"League Average"

document
.getElementById(
"pitcherSearchA"
)
.value =
""

document
.getElementById(
"pitcherSearchB"
)
.value =
""

document
.getElementById(
"pitcherChoicesA"
)
.innerHTML =
""

document
.getElementById(
"pitcherChoicesB"
)
.innerHTML =
""

document
.getElementById(
"pitcherStatsA"
)
.innerHTML =
""

document
.getElementById(
"pitcherStatsB"
)
.innerHTML =
""

document
.getElementById(
"oddsResult"
)
.innerHTML =
""

document
.getElementById(
"teamA"
)
.selectedIndex =
0

document
.getElementById(
"teamB"
)
.selectedIndex =
0

showHitters(
"A"
)

showHitters(
"B"
)

console.log(
"reset complete"
)

}

loadAll()
