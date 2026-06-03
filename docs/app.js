let wave=[]

let pave=[]

let confidence=[]

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
limit
){

const rows =
limit
?
data.slice(
0,
limit
)
:
data

if(
rows.length===0
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
rows[0]
)
.forEach(
c=>{

html +=
`<th>${c}</th>`

}
)

html +=
"</tr>"

rows.forEach(
row=>{

html +=
"<tr>"

Object
.values(
row
)
.forEach(
v=>{

html +=
`<td>${v}</td>`

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
).innerHTML =
html

}

function loadTeams(data){

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

data.forEach(
row=>{

const team =
row.team

if(
team
){

a.innerHTML +=
`<option>${team}</option>`

b.innerHTML +=
`<option>${team}</option>`

}

}
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

buildTable(
wave,
"waveTable",
20
)

buildTable(
pave,
"paveTable",
20
)

loadTeamExplorer()

loadTeams(
confidence
)

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

const source =
document
.getElementById(
"playerType"
)
.value

const players =
source==="wave"
?
wave
:
pave

const matches =
players.filter(
p=>{

const first =
(
p.name_first
||
""
)
.toLowerCase()

const last =
(
p.name_last
||
""
)
.toLowerCase()

return(

first.includes(
query
)

||

last.includes(
query
)

||

`${first} ${last}`
.includes(
query
)

)

}
)

if(
matches.length===0
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

matches.forEach(
(
player,
index
)=>{

html +=

`
<button
onclick="showPlayer(${index})">

${player.name_first}
${player.name_last}

</button>

`

}
)

window.playerMatches =
matches

document
.getElementById(
"playerChoices"
)
.innerHTML =
html

document
.getElementById(
"playerResult"
)
.innerHTML =
""

}

function loadTeamExplorer(){

let html =
""

confidence.forEach(
row=>{

const team =
row.team

if(
team
){

html +=

`
<button
onclick="showTeam(
'${team}'
)">

${team}

</button>

`

}

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
confidence.find(
r=>

r.team
===
team

)

buildTable(

[result],

"teamResult"

)

}

function predictOdds(){

const teamA =
document
.getElementById(
"teamA"
)
.value

const teamB =
document
.getElementById(
"teamB"
)
.value

if(
teamA===teamB
){

document
.getElementById(
"oddsResult"
)
.innerHTML =
"Choose different teams"

return

}

const A =
confidence.find(
r=>
r.team===teamA
)

const B =
confidence.find(
r=>
r.team===teamB
)

const scoreA =

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

const scoreB =

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

document
.getElementById(
"oddsResult"
)
.innerHTML =

`

<h3>

${teamA}
vs
${teamB}

</h3>

<p>

${teamA}

Win:

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

</p>

<p>

${teamB}

Win:

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

</p>

`

}

function showPlayer(index){

const player =
window.playerMatches[
index
]

buildTable(

[player],

"playerResult"

)

}

loadAll()
