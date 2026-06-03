async function loadCSV(path){

const response =
await fetch(path)

const text =
await response.text()

const rows =
text
.trim()
.split("\n")

const headers =
rows[0]
.split(",")

return rows
.slice(1)
.map(row=>{

const values =
row.split(",")

let obj={}

headers.forEach(
(h,i)=>
obj[h]=values[i]
)

return obj

})

}

function makeTable(
data,
container,
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
data.length===0
){

document
.getElementById(
container
)
.innerHTML =
"No data"

return

}

const cols =
Object.keys(
data[0]
)

let html =
"<table>"

html +=
"<tr>"

cols.forEach(
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

cols.forEach(
c=>{

html +=
`<td>${row[c]}</td>`

})

html +=
"</tr>"

}
)

html +=
"</table>"

document
.getElementById(
container
)
.innerHTML =
html

}

function loadTeams(
data
){

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
t=>{

if(
t.team
){

a.innerHTML +=
`<option>
${t.team}
</option>`

b.innerHTML +=
`<option>
${t.team}
</option>`

}

}
)

}

async function loadAll(){

try{

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

makeTable(
wave,
"waveTable",
20
)

makeTable(
pave,
"paveTable",
20
)

makeTable(
confidence,
"confidenceTable"
)

loadTeams(
confidence
)

}

catch(err){

console.log(
err
)

}

}

loadAll()
