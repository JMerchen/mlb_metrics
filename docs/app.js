let wave = []

let pave = []

let confidence = []

let playerMatches = []

let starterA =
"League Average"

let starterB =
"League Average"



async function loadCSV(path){

const response =
await fetch(path)

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

buildTable(

[
playerMatches[
index
]
],

"playerResult"

)

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

showTeam(
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


function showTeam(team){

const result =

confidence.find(
r=>

r.team
===

team

)

if(
!result
){

return

}

const scores =

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
...scores
)

const max =

Math.max(
...scores
)

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

const color =

`rgb(
${red},
80,
${blue}
)`

document
.getElementById(
"teamResult"
)
.innerHTML =

`

<div
style="
font-size:28px;
font-weight:bold;
"

>

${result.team}

</div>

<div
style="
height:14px;
width:100%;
background:${color};
border-radius:8px;
margin-top:8px;
margin-bottom:15px;
"

>

</div>

<div
id="teamStats">

</div>

`

buildTable(

[
result
],

"teamStats"

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

starterA =
String(
name
)

}

else{

starterB =
String(
name
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

stats.innerHTML =

`
<table>

<tr>
<th>Expected Hits</th>
<td>${player.Expected_Hits}</td>
</tr>

<tr>
<th>Expected Bases</th>
<td>${player.Expected_Bases}</td>
</tr>

<tr>
<th>Expected HRs</th>
<td>${player.Expected_HRs}</td>
</tr>

</table>
`

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

const pitchA =
getPitchAdjustment(
starterB
)

const pitchB =
getPitchAdjustment(
starterA
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
starterB,
parkFactor
)

const hrB =
getHRAdjustment(
B,
starterA,
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
starterA
:
starterB
)

const dogPitch =

getPitcher(
winner==="A"
?
starterB
:
starterA
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
  
document
.getElementById(
"oddsResult"
)
.innerHTML =

`

<h2>

${A.team}

vs

${B.team}

</h2>

<hr>

<h3>

${A.team}

</h3>

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

<b>

${confidenceLabel(
probA
)}

</b>

<br>

Fair Line:

<b>

${americanOdds(
probA
)}

</b>

<hr>

<h3>

${B.team}

</h3>

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

<b>

${confidenceLabel(
probB
)}

</b>

<br>

Fair Line:

<b>

${americanOdds(
probB
)}

</b>

<hr>

<h3>

Why the Model Likes

${favored.team}

</h3>

<table>

<tr>

<th>

True Power

</th>

<td>

${factors["True Power"]
.toFixed(
3
)}

</td>

</tr>

<tr>

<th>

Confidence

</th>

<td>

${factors["Confidence"]
.toFixed(
3
)}

</td>

</tr>

<tr>

<th>

Pitching

</th>

<td>

${factors["Pitching"]
.toFixed(
3
)}

</td>

</tr>

<tr>

<th>

HR Environment

</th>

<td>

${factors["HR Environment"]
.toFixed(
3
)}

</td>

</tr>

</table>

<br>

<b>

Biggest Edge:

</b>

${biggest?.[0] || "Balanced"}

`

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


loadAll()
