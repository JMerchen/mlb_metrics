// Wins above replacement table (NFL Stats tab): our own measure of what
// each player has been worth this season, filtered by side of the ball,
// position, years in the league and conference or division, and sorted by
// any column (user request, 2026-11-23).
//
// Data: docs/data/nfl_player_value_<season>.json
// (scripts/build_nfl_player_value.py; the method, and what it cannot see,
// are in src/mlb_metrics/nfl_player_value.py). Pure functions returning
// HTML strings; nfl.js keeps the filter state and wires the controls.

const PV_SIDES = [["all", "Everyone"], ["offense", "Offense"], ["defense", "Defense"], ["special", "Kickers"]]
// [code, label, side]
const PV_POSITIONS = [
["QB", "Quarterbacks", "offense"], ["RB", "Running backs", "offense"], ["WR", "Wide receivers", "offense"], ["TE", "Tight ends", "offense"],
["ED", "Edge rushers", "defense"], ["DT", "Interior linemen", "defense"], ["LB", "Linebackers", "defense"],
["CB", "Cornerbacks", "defense"], ["S", "Safeties", "defense"], ["K", "Kickers", "special"],
]
// [key, label, first season in the league, last]: exp counts seasons before this one.
const PV_EXPERIENCE = [["all", "Any", 0, 99], ["rookie", "Rookies", 0, 0], ["early", "2nd to 4th year", 1, 3],
["prime", "5th to 8th year", 4, 7], ["veteran", "9th year or later", 8, 99]]
const PV_DIVISIONS = ["AFC East", "AFC North", "AFC South", "AFC West", "NFC East", "NFC North", "NFC South", "NFC West"]
// [key, long heading, short heading, digits, what it is]
const PV_COMMON = [
["games", "Games", "G", 0, "Games he has played in"],
["volume", "Plays", "Plays", 0, "Dropbacks, carries and targets on offense; snaps on defense; kicks for a kicker"],
["war", "WAR", "WAR", 2, "Wins above replacement: points above replacement divided by the points a win takes"],
["par", "Above replacement", "vs. repl.", 1, "Points he has been worth over a reserve at his position in his plays"],
["paa", "Above average", "vs. avg.", 1, "Points he has been worth over the average player at his position in his plays"],
]
const PV_EXTRA = {
offense: [["passing", "Passing", "Pass", 1, "Points above replacement as a passer"], ["rushing", "Rushing", "Rush", 1, "Points above replacement as a runner"],
["receiving", "Receiving", "Rec", 1, "Points above replacement as a receiver"]],
defense: [["points", "Points taken", "Pts", 1, "What his credited plays cost offenses, before any comparison"], ["sacks", "Sacks", "Sk", 1, "Sacks (a shared sack is half)"],
["interceptions", "Int", "Int", 0, "Interceptions"], ["passes_defended", "PD", "PD", 0, "Passes defended: broken up and fell incomplete"],
["tackles_for_loss", "TFL", "TFL", 0, "Tackles for loss, behind the line"], ["tackles", "Stops", "Stops", 0, "Tackles on plays that lost the offense expected points"]],
special: [["fg_made", "Field goals made", "FGM", 0, "Field goals made"], ["fg_tried", "Field goals tried", "FGA", 0, "Field goals tried"]],
}

// Columns a phone leaves out, to keep the names readable.
const PV_NARROW_HIDE = ["games", "paa"]

function pvEsc(value){
return String(value ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;")
}

function pvOrdinal(n){
const s = ["th", "st", "nd", "rd"], v = n % 100
return n + (s[(v - 20) % 10] || s[v] || s[0])
}

// "Rookie", "4th year": exp is his seasons before this one.
function pvYear(exp){
if(exp === null || exp === undefined){ return "" }
return Number(exp) === 0 ? "Rookie" : `${pvOrdinal(Number(exp) + 1)} year`
}

function pvSideOf(position){
const found = PV_POSITIONS.find(p => p[0] === position)
return found ? found[2] : "all"
}

// A filter state made consistent: a position belongs to one side, so
// picking one picks its side, and a side that does not have the chosen
// position drops it.
function pvState(state = {}){
const out = {side: "all", pos: "all", exp: "all", group: "all", search: "", sort: "war", dir: -1, limit: 50, ...state}
if(out.pos !== "all" && out.side !== "all" && pvSideOf(out.pos) !== out.side){ out.pos = "all" }
if(!PV_SIDES.some(s => s[0] === out.side)){ out.side = "all" }
return out
}

// The side whose extra columns show: the chosen side, or the chosen position's.
function pvShownSide(state){
return state.side !== "all" ? state.side : state.pos !== "all" ? pvSideOf(state.pos) : "all"
}

function pvColumns(state){
return PV_COMMON.concat(PV_EXTRA[pvShownSide(state)] || [])
}

function pvFilter(players, state){
const s = pvState(state)
const years = PV_EXPERIENCE.find(e => e[0] === s.exp) || PV_EXPERIENCE[0]
const words = String(s.search || "").trim().toLowerCase()
return (players || []).filter(p => {
if(s.side !== "all" && p.side !== s.side){ return false }
if(s.pos !== "all" && p.pos !== s.pos){ return false }
if(s.exp !== "all" && (p.exp === null || p.exp === undefined || p.exp < years[2] || p.exp > years[3])){ return false }
if(s.group !== "all" && !(p.div === s.group || String(p.div || "").split(" ")[0] === s.group)){ return false }
if(words && !String(p.name || "").toLowerCase().includes(words)){ return false }
return true
})
}

// Sorted by a column (a missing value last either way), ties by WAR.
function pvSort(rows, key, dir = -1){
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

function pvNumber(value, digits){
if(value === undefined || value === null || value === ""){ return "" }
const n = Number(value)
const text = Math.abs(n).toFixed(digits)
// Not "-0.0".
return n < 0 && Number(text) !== 0 ? `−${text}` : text
}

function pvTeam(code){
return typeof nflTeam === "function" ? nflTeam(code) : (code || "")
}

function pvLogo(code){
return typeof nflLogo === "function" ? nflLogo(code, "xs") : ""
}

// The filter controls. `seasons` newest first; `state.season` the one shown.
function pvControlsHtml(seasons, state){
const s = pvState(state)
const option = (value, label, chosen) => `<option value="${pvEsc(value)}"${String(value) === String(chosen) ? " selected" : ""}>${pvEsc(label)}</option>`
const shown = pvShownSide(s)
const positions = PV_POSITIONS.filter(p => s.side === "all" || p[2] === s.side)
const groups = [["offense", "Offense"], ["defense", "Defense"], ["special", "Kickers"]].map(([side, label]) => {
const own = positions.filter(p => p[2] === side)
return own.length ? `<optgroup label="${label}">${own.map(p => option(p[0], p[1], s.pos)).join("")}</optgroup>` : ""
}).join("")
return `<div class="tabGroup pvSides">${PV_SIDES.map(([key, label]) => `<button class="tabButton${shown === key || (key === "all" && shown === "all") ? " active" : ""}" data-pv-side="${key}">${label}</button>`).join("")}</div>
<div class="pvSelects">
<label>Season <select data-pv="season">${(seasons || []).map(year => option(year, year, s.season)).join("")}</select></label>
<label>Position <select data-pv="pos">${option("all", "All", s.pos)}${groups}</select></label>
<label>Years in the league <select data-pv="exp">${PV_EXPERIENCE.map(e => option(e[0], e[1], s.exp)).join("")}</select></label>
<label>Conference or division <select data-pv="group">${option("all", "All", s.group)}${option("AFC", "AFC", s.group)}${option("NFC", "NFC", s.group)}${PV_DIVISIONS.map(d => option(d, d, s.group)).join("")}</select></label>
<label>Player <input type="search" data-pv="search" placeholder="Find a player" value="${pvEsc(s.search)}" autocomplete="off"></label>
</div>`
}

// The line above the table: how many players, through which week, and
// what the chosen group cannot show.
function pvSummaryHtml(data, state, count){
const s = pvState(state), shown = pvShownSide(s)
const parts = [`<b>${count}</b> player${count === 1 ? "" : "s"}`, `${pvEsc(data.season)} season through week ${pvEsc(data.through_week)}`,
`${pvNumber(data.points_per_win, 0)} points = 1 win`]
const note = shown === "defense" ? "Defenders are rated on the plays they made (sacks, stops, passes broken up) - coverage nobody tested is invisible."
: shown === "all" ? "Defenders are rated on plays made only, so their numbers run lower than offense's; offensive linemen are not rated."
: shown === "special" ? "Kickers: points over what each kick was expected to be worth." : ""
return `<div class="pvSummary"><span>${parts.join(" · ")}</span>${note ? `<span class="pvNote">${note}</span>` : ""}</div>`
}

function pvTableHtml(data, state){
if(!data || !data.players || !data.players.length){ return `<p class="pvEmpty">No player values for this season yet.</p>` }
const s = pvState(state)
const columns = pvColumns(s)
const key = s.sort === "name" || columns.some(c => c[0] === s.sort) ? s.sort : "war"
const rows = pvSort(pvFilter(data.players, s), key, s.dir)
const arrow = s.dir < 0 ? " ▼" : " ▲"
const volume = pvShownSide(s) === "defense" ? ["Snaps", "Snaps"] : pvShownSide(s) === "special" ? ["Kicks", "Kicks"] : pvShownSide(s) === "offense" ? ["Plays", "Plays"] : ["Plays or snaps", "Plays"]
const head = `<th><button class="sortHead${key === "name" ? " sorted" : ""}" data-sort="name">Player${key === "name" ? arrow : ""}</button></th>`
+ columns.map(([col, label, short, , tip]) => {
const [long, brief] = col === "volume" ? volume : [label, short]
return `<th class="num${PV_NARROW_HIDE.includes(col) ? " pvWide" : ""}"><button class="sortHead${key === col ? " sorted" : ""}" data-sort="${col}" data-tip="${pvEsc(tip)}"><span class="headLong">${long}</span><span class="headShort">${brief}</span>${key === col ? arrow : ""}</button></th>`
}).join("")
if(!rows.length){ return `${pvSummaryHtml(data, s, 0)}<p class="pvEmpty">Nobody matches those filters.</p>` }
const body = rows.slice(0, s.limit).map((p, i) => `<tr>
<td><span class="statWho pvWho"><span class="rankNo">${i + 1}</span><span class="rankWho"><b>${pvEsc(p.name)}</b><small>${pvLogo(p.team)} ${pvEsc(pvTeam(p.team))} · ${pvEsc(p.pos)}${pvYear(p.exp) ? ` · ${pvYear(p.exp)}` : ""}</small></span></span></td>
${columns.map(([col, , , digits]) => `<td class="num${col === "war" ? " pvWar" : ""}${PV_NARROW_HIDE.includes(col) ? " pvWide" : ""}">${col === "war" ? `<b>${pvNumber(p[col], digits)}</b>` : pvNumber(p[col], digits)}</td>`).join("")}
</tr>`).join("")
const more = rows.length > s.limit ? `<button class="tabButton pvMore" data-pv-more="1">Show all ${rows.length}</button>` : ""
return `${pvSummaryHtml(data, s, rows.length)}<div class="scrollbox statTableBox"><table class="statTable pvTable"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>${more}`
}

if (typeof module !== "undefined" && module.exports) {
module.exports = { PV_SIDES, PV_POSITIONS, PV_EXPERIENCE, PV_DIVISIONS, pvYear, pvSideOf, pvState, pvShownSide, pvColumns, pvFilter, pvSort, pvNumber,
pvControlsHtml, pvSummaryHtml, pvTableHtml }
}
