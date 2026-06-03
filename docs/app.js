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

const wave =
await loadCSV(
"./data/wave.csv"
)

const pave =
await loadCSV(
"./data/pave.csv"
)

const confidence =
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

buildTable(
confidence,
"confidenceTable"
)

loadTeams(
confidence
)

}

loadAll()
