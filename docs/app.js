async function loadWave(){

try{

const response =
await fetch(
"./data/wave.csv"
)

const text =
await response.text()

const rows =
text
.trim()
.split("\n")

let html =
"<table>"

rows
.slice(
0,
20
)
.forEach(
(
row,
index
)=>{

const cols =
row.split(",")

html +=
"<tr>"

cols.forEach(
value=>{

html +=
index===0
?
`<th>${value}</th>`
:
`<td>${value}</td>`

})

html +=
"</tr>"

}
)

html +=
"</table>"

document
.getElementById(
"output"
)
.innerHTML =
html

}

catch(err){

document
.getElementById(
"output"
)
.innerHTML =
"Could not load CSV"

console.log(err)

}

}

loadWave()
