async function loadWave(){

const response =
await fetch(
"/mlb_metrics/data/wave.csv"
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
text
.substring(
0,
1500
)
+
"</pre>"

}

loadWave()
