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

if(
query.length
<
2
){

document
.getElementById(
`pitcherChoices${side}`
)
.innerHTML =
""

return

}

const matches =

pave
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

let html =
""

matches
.forEach(
p=>{

const full =

`${p.name_first}
${p.name_last}`

html +=

`

<button

onclick="selectPitcher(
'${side}',
'${full}'
)"

>

${full}

</button>

`

}

)

document
.getElementById(
`pitcherChoices${side}`
)
.innerHTML =
html

}



function selectPitcher(
side,
name
){

if(
side==="A"
){

starterA =
name

}

else{

starterB =
name

}

document
.getElementById(
`pitcherSelected${side}`
)
.innerHTML =

`
✓
Selected:

<b>

${name}

</b>
`

document
.getElementById(
`pitcherChoices${side}`
)
.innerHTML =
""

}



function getPave(name){

if(
name===
"League Average"
){

return 1

}

const player =

pave.find(
p=>{

const full =

`${p.name_first}
${p.name_last}`

return(
full===name
)

}

)

if(
!player
){

return 1

}

const possible =

[
"PAVE",
"pave",
"probability",
"value",
"score"
]

for(
let col
of
possible
){

if(
player[col]
){

console.log(
name,
col,
player[col]
)

return Number(
player[col]
)

}

}

console.log(
player
)

return 1

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

const scoreA =

baseA
*
getPave(
starterB
)

const scoreB =

baseB
*
getPave(
starterA
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

loadTeams()

}


loadAll()
