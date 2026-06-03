async function loadWave(){

const response =
await fetch(
"./data/wave.csv"
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

loadWave()
