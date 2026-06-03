async function loadWave(){

try{

const response =
await fetch(
"./data/wave.csv"
)

if(!response.ok){

throw new Error(
response.status
)

}

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

catch(err){

document
.getElementById(
"output"
)
.innerHTML =
"ERROR:<br><br>"
+
err

console.log(err)

}

}

loadWave()
