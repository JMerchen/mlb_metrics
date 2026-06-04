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

let html =
""

confidence
.forEach(
team=>{

html +=

`

<button

onclick="showTeam(
'${team.team}'
)"

>

${team.team}

</button>

`

}

)

document
.getElementById(
"teamButtons"
)
.innerHTML =
html

}



function showTeam(team){

const result =

confidence
.find(
r=>
r.team===team
)

buildTable(

[result],

"teamResult"

)

}



function loadTeams(){

const a =
document
.getElementById(
"teamA"
)

const b =
document
.getElementById(
"teamB"
)

confidence
.forEach(
t=>{

a.innerHTML +=

`
<option>

${t.team}

</option>
`

b.innerHTML +=

`
<option>

${t.team}

</option>
`

}

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

function showHitters(){

const home =

document
.getElementById(
"teamA"
)
.value

const away =

document
.getElementById(
"teamB"
)
.value

function render(team){

const rows =

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

.sort(
(a,b)=>

(

Number(
b.PA_L||0
)

+

Number(
b.PA_R||0
)

)

-

(

Number(
a.PA_L||0
)

+

Number(
a.PA_R||0
)

)

)

.slice(
0,
10
)

if(
rows.length===0
){

return
"No hitters found"

}

let html =
"<table>"

rows.forEach(
p=>{

const pa =

Number(
p.PA_L||0
)

+

Number(
p.PA_R||0
)

html +=

`
<tr>

<td>

${p.name_first}
${p.name_last}

</td>

<td>

PA:
${pa}

</td>

</tr>
`

}

)

html +=
"</table>"

return html

}

document
.getElementById(
"teamAHitters"
)
.innerHTML =
render(
home
)

document
.getElementById(
"teamBHitters"
)
.innerHTML =
render(
away
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

showHitters()
  
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

<p>

${A.team}

:
${(
probA
*
100
)
.toFixed(
1
)
}%

</p>

<p>

${B.team}

:
${(
probB
*
100
)
.toFixed(
1
)
}%

</p>

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

buildTable(
pave,
"paveTable",
20
)

loadTeamExplorer()

loadTeams()

}


loadAll()
