// Wins above replacement table (MLB Stats page): our own measure of what
// each player was worth in a season, filtered by hitters or pitchers,
// position, age and league or division, and sorted by any column (user
// request, 2026-11-23: "our own version of war").
//
// Data: docs/data/mlb_war_<season>.json (scripts/build_mlb_war.py; the
// method, and what it cannot see, are in src/mlb_metrics/mlb_war.py). The
// functions that build HTML are pure, so node tests can check them; the
// block at the bottom keeps the filter state and wires the page. Shares
// the NFL table's styles (.pv*, style.css).

const MW_KINDS = [["all", "Everyone"], ["hitter", "Hitters"], ["pitcher", "Pitchers"]]
// [code, label, kind]
const MW_POSITIONS = [
["C", "Catchers", "hitter"], ["1B", "First basemen", "hitter"], ["2B", "Second basemen", "hitter"], ["3B", "Third basemen", "hitter"],
["SS", "Shortstops", "hitter"], ["LF", "Left fielders", "hitter"], ["CF", "Center fielders", "hitter"], ["RF", "Right fielders", "hitter"],
["DH", "Designated hitters", "hitter"], ["SP", "Starting pitchers", "pitcher"], ["RP", "Relief pitchers", "pitcher"],
]
// [key, label, youngest, oldest]
const MW_AGES = [["all", "Any", 0, 99], ["young", "25 and under", 0, 25], ["prime", "26 to 29", 26, 29], ["thirties", "30 to 33", 30, 33], ["veteran", "34 and over", 34, 99]]
const MW_DIVISIONS = ["AL East", "AL Central", "AL West", "NL East", "NL Central", "NL West"]
// [key, long heading, short heading, digits, what it is, kinds it shows for]
const MW_COLUMNS = [
["games", "Games", "G", 0, "Games he appeared in", "all hitter pitcher"],
["pa", "Plate appearances", "PA", 0, "Times at the plate", "all hitter"],
["bf", "Batters faced", "BF", 0, "Batters he pitched to", "all pitcher"],
["war", "WAR", "WAR", 2, "Wins above replacement: runs above replacement divided by the runs a win takes", "all hitter pitcher"],
["rar", "Above replacement", "vs. repl.", 1, "Runs he was worth over a replacement player in his playing time", "all hitter pitcher"],
["raa", "Above average", "vs. avg.", 1, "Runs he was worth over the average player at his position", "all hitter pitcher"],
["batting", "Batting", "Bat", 1, "Runs at the plate against the average hitter at his position, each plate appearance valued in its situation and graded for the pitchers and parks he faced", "all hitter"],
["running", "Running", "Run", 1, "Runs on the bases against the average runner: steals and caught stealings, and extra bases (or outs) on teammates' hits and outs against what is usual for that play", "all hitter"],
["fielding", "Fielding", "Fld", 1, "Runs in the field against the average at his position: outs made over what balls hit that way usually become", "all hitter pitcher"],
["pitching", "Pitching", "Pit", 1, "Runs saved against the average starter or reliever, with his fielders' part taken out", "all pitcher"],
]
// Columns a phone leaves out, to keep the names readable.
const MW_NARROW_HIDE = ["games", "raa"]

function mwEsc(value){
return String(value ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;")
}

function mwKindOf(position){
const found = MW_POSITIONS.find(p => p[0] === position)
return found ? found[2] : "all"
}

// A filter state made consistent: a position is a hitter's or a
// pitcher's, so picking one picks its kind, and a kind without the chosen
// position drops it.
function mwState(state = {}){
const out = {kind: "all", pos: "all", age: "all", group: "all", search: "", sort: "war", dir: -1, limit: 50, ...state}
if(!MW_KINDS.some(k => k[0] === out.kind)){ out.kind = "all" }
if(out.pos !== "all" && out.kind !== "all" && mwKindOf(out.pos) !== out.kind){ out.pos = "all" }
return out
}

// Whose columns show: the chosen kind, or the chosen position's.
function mwShownKind(state){
return state.kind !== "all" ? state.kind : state.pos !== "all" ? mwKindOf(state.pos) : "all"
}

function mwColumns(state){
const kind = mwShownKind(state)
return MW_COLUMNS.filter(c => c[5].split(" ").includes(kind))
}

function mwFilter(players, state){
const s = mwState(state)
const ages = MW_AGES.find(a => a[0] === s.age) || MW_AGES[0]
const words = String(s.search || "").trim().toLowerCase()
return (players || []).filter(p => {
if(s.kind !== "all" && p.kind !== s.kind){ return false }
if(s.pos !== "all" && p.pos !== s.pos){ return false }
if(s.age !== "all" && (p.age === null || p.age === undefined || p.age < ages[2] || p.age > ages[3])){ return false }
if(s.group !== "all" && !(p.div === s.group || String(p.div || "").split(" ")[0] === s.group)){ return false }
if(words && !String(p.name || "").toLowerCase().includes(words)){ return false }
return true
})
}

// Sorted by a column (a missing value last either way), ties by WAR.
function mwSort(rows, key, dir = -1){
const value = p => key === "name" ? String(p.name || "").toLowerCase() : (p[key] === undefined || p[key] === null ? null : Number(p[key]))
return rows.slice().sort((a, b) => {
const x = value(a), y = value(b)
if(x === null && y === null){ return (b.war || 0) - (a.war || 0) }
if(x === null){ return 1 }
if(y === null){ return -1 }
if(x === y){ return (b.war || 0) - (a.war || 0) }
// dir -1 is biggest first; for names, A first.
return (x < y ? -1 : 1) * (dir < 0 ? -1 : 1) * (key === "name" ? -1 : 1)
})
}

function mwNumber(value, digits){
if(value === undefined || value === null || value === ""){ return "" }
const n = Number(value)
const text = Math.abs(n).toFixed(digits)
// Not "-0.0".
return n < 0 && Number(text) !== 0 ? `−${text}` : text
}

function mwTeam(code){
return typeof mlbTeam === "function" ? mlbTeam(code) : (code || "")
}

function mwLogo(code){
return typeof mlbLogo === "function" ? mlbLogo(code, "xs", true) : ""
}

function mwControlsHtml(seasons, state){
const s = mwState(state)
const option = (value, label, chosen) => `<option value="${mwEsc(value)}"${String(value) === String(chosen) ? " selected" : ""}>${mwEsc(label)}</option>`
const shown = mwShownKind(s)
const positions = MW_POSITIONS.filter(p => s.kind === "all" || p[2] === s.kind)
const groups = [["hitter", "Hitters"], ["pitcher", "Pitchers"]].map(([kind, label]) => {
const own = positions.filter(p => p[2] === kind)
return own.length ? `<optgroup label="${label}">${own.map(p => option(p[0], p[1], s.pos)).join("")}</optgroup>` : ""
}).join("")
return `<div class="tabGroup pvSides">${MW_KINDS.map(([key, label]) => `<button class="tabButton${shown === key ? " active" : ""}" data-mw-kind="${key}">${label}</button>`).join("")}</div>
<div class="pvSelects">
<label>Season <select data-mw="season">${(seasons || []).map(year => option(year, year, s.season)).join("")}</select></label>
<label>Position <select data-mw="pos">${option("all", "All", s.pos)}${groups}</select></label>
<label>Age <select data-mw="age">${MW_AGES.map(a => option(a[0], a[1], s.age)).join("")}</select></label>
<label>League or division <select data-mw="group">${option("all", "All", s.group)}${option("AL", "American League", s.group)}${option("NL", "National League", s.group)}${MW_DIVISIONS.map(d => option(d, d, s.group)).join("")}</select></label>
<label>Player <input type="search" data-mw="search" placeholder="Find a player" value="${mwEsc(s.search)}" autocomplete="off"></label>
</div>`
}

function mwSummaryHtml(data, count){
const through = data.through ? ` through ${mwEsc(data.through)}` : ""
return `<div class="pvSummary"><span><b>${count}</b> player${count === 1 ? "" : "s"} · ${mwEsc(data.season)} season${through} · ${mwNumber(data.runs_per_win, 1)} runs = 1 win</span>
<span class="pvNote">A catcher's framing and throwing are not in it yet.</span></div>`
}

function mwTableHtml(data, state){
if(!data || !data.players || !data.players.length){ return `<p class="pvEmpty">No player values for this season yet.</p>` }
const s = mwState(state)
const columns = mwColumns(s)
const key = s.sort === "name" || columns.some(c => c[0] === s.sort) ? s.sort : "war"
const rows = mwSort(mwFilter(data.players, s), key, s.dir)
if(!rows.length){ return `${mwSummaryHtml(data, 0)}<p class="pvEmpty">Nobody matches those filters.</p>` }
const arrow = s.dir < 0 ? " ▼" : " ▲"
const head = `<th><button class="sortHead${key === "name" ? " sorted" : ""}" data-sort="name">Player${key === "name" ? arrow : ""}</button></th>`
+ columns.map(([col, label, short, , tip]) => `<th class="num${MW_NARROW_HIDE.includes(col) ? " pvWide" : ""}"><button class="sortHead${key === col ? " sorted" : ""}" data-sort="${col}" data-tip="${mwEsc(tip)}"><span class="headLong">${label}</span><span class="headShort">${short}</span>${key === col ? arrow : ""}</button></th>`).join("")
const body = rows.slice(0, s.limit).map((p, i) => `<tr>
<td><span class="statWho pvWho"><span class="rankNo">${i + 1}</span><span class="rankWho"><b>${mwEsc(p.name)}</b><small>${mwLogo(p.team)} ${mwEsc(mwTeam(p.team))} · ${mwEsc(p.pos)}${p.age ? ` · age ${mwEsc(p.age)}` : ""}</small></span></span></td>
${columns.map(([col, , , digits]) => `<td class="num${col === "war" ? " pvWar" : ""}${MW_NARROW_HIDE.includes(col) ? " pvWide" : ""}">${col === "war" ? `<b>${mwNumber(p[col], digits)}</b>` : mwNumber(p[col], digits)}</td>`).join("")}
</tr>`).join("")
const more = rows.length > s.limit ? `<button class="tabButton pvMore" data-mw-more="1">Show all ${rows.length}</button>` : ""
return `${mwSummaryHtml(data, rows.length)}<div class="scrollbox statTableBox"><table class="statTable pvTable"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>${more}`
}

if (typeof module !== "undefined" && module.exports) {
module.exports = { MW_POSITIONS, MW_AGES, MW_DIVISIONS, mwKindOf, mwState, mwShownKind, mwColumns, mwFilter, mwSort, mwNumber, mwControlsHtml, mwSummaryHtml, mwTableHtml }
}

// ---- the page ----

if(typeof document !== "undefined"){ (function(){
const view = {seasons: [], data: {}, state: {season: null}, teamSort: "expected", teamDir: -1}
const controls = () => document.getElementById("mlbWarControls")
const table = () => document.getElementById("mlbWarTable")
const fetchJson = async file => {
const response = await fetch(`./data/${file}.json?t=${Date.now()}`, {cache: "no-store"})
if(!response.ok){ throw new Error(`HTTP ${response.status}`) }
return response.json()
}

function render(withControls = true){
if(!table()){ return }
view.state = mwState(view.state)
if(withControls){ controls().innerHTML = mwControlsHtml(view.seasons, view.state) }
const data = view.data[view.state.season]
table().innerHTML = data ? mwTableHtml(data, view.state) : "Loading..."
renderTeams()
if(typeof installVizTips === "function"){ installVizTips() }
}

function renderTeams(){
const el = document.getElementById("mlbWarTeams"), data = view.data[view.state.season]
if(!el || typeof wtTeamsHtml !== "function"){ return }
el.innerHTML = data ? wtTeamsHtml(data, {unit: "runs", seasonGames: 162, sort: view.teamSort, dir: view.teamDir, name: mwTeam, logo: mwLogo}) : ""
}

function showSeason(season){
view.state.season = season
render()
if(view.data[season]){ return }
fetchJson(`mlb_war_${season}`).then(data => { view.data[season] = data; if(view.state.season === season){ render() } })
.catch(() => { table().innerHTML = "Player values appear after the next update." })
}

function start(){
if(!table() || !controls()){ return }
const set = changes => { Object.assign(view.state, changes, {limit: 50}); render() }
controls().addEventListener("click", e => {
const kind = e.target.closest("[data-mw-kind]")
if(kind){ set({kind: kind.dataset.mwKind, pos: "all"}) }
})
controls().addEventListener("change", e => {
const field = e.target.closest("select[data-mw]")
if(!field){ return }
if(field.dataset.mw === "season"){ showSeason(Number(field.value)) }
else if(field.dataset.mw === "pos"){ set({pos: field.value, kind: field.value === "all" ? view.state.kind : mwKindOf(field.value)}) }
else{ set({[field.dataset.mw]: field.value}) }
})
// Typing redraws the table only, so the box keeps its cursor.
controls().addEventListener("input", e => {
const field = e.target.closest("input[data-mw='search']")
if(field){ Object.assign(view.state, {search: field.value, limit: 50}); render(false) }
})
table().addEventListener("click", e => {
const sort = e.target.closest("[data-sort]")
if(sort){
const key = sort.dataset.sort
view.state.dir = view.state.sort === key ? -view.state.dir : -1
view.state.sort = key
render(false)
return
}
if(e.target.closest("[data-mw-more]")){ view.state.limit = 100000; render(false) }
})
const teams = document.getElementById("mlbWarTeams")
if(teams){ teams.addEventListener("click", e => {
const sort = e.target.closest("[data-team-sort]")
if(!sort){ return }
view.teamDir = view.teamSort === sort.dataset.teamSort ? -view.teamDir : -1
view.teamSort = sort.dataset.teamSort
renderTeams()
if(typeof installVizTips === "function"){ installVizTips() }
}) }
fetchJson("mlb_war").then(index => {
view.seasons = index.seasons || []
if(view.seasons.length){ showSeason(view.seasons[0]) }
}).catch(() => { table().innerHTML = "Player values appear after the next update." })
}

if(document.readyState === "loading"){ document.addEventListener("DOMContentLoaded", start) }else{ start() }
})() }
