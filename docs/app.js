async function loadWave(){

document
.getElementById(
"output"
)
.innerHTML =
"attempting fetch"

try{

const response =
await fetch(
"../data/wave.csv"
)

const text =
await response.text()

document
.getElementById(
"output"
)
.innerHTML =
"<pre>"
+
text.slice(
0,
1500
)
+
"</pre>"

}

catch(err){

document
.getElementById(
"output"
)
.innerHTML =
err

}

}

loadWave()
