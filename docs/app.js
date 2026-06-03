async function test(){

try{

document
.getElementById(
"waveTable"
)
.innerHTML =
"Loading WAVE..."

const response =
await fetch(
"./data/wave.csv"
)

document
.getElementById(
"waveTable"
)
.innerHTML =
`
Status:
${response.status}
`

const text =
await response.text()

document
.getElementById(
"waveTable"
)
.innerHTML =
`
Loaded

${text.slice(0,300)}
`

}

catch(err){

document
.getElementById(
"waveTable"
)
.innerHTML =
`
ERROR:

${err}
`

}

}

test()
