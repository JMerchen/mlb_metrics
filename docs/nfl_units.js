// NFL Team units tab (nfl_units.py's docs/data/nfl_team_units.csv):
// QB+, Receiving+, Rushing+, Pass D+, Run D+ on a 100 = league average
// scale, higher always stronger. Pure functions returning HTML strings, so
// node tests can check them; nfl.js does the DOM wiring. Also the small
// "units behind it" note on each player-prop row, so an Over or Under can
// be checked against the units it leans on.

const UNIT_COLUMNS = [
["qb_plus", "QB+"],
["receiving_plus", "Receiving+"],
["rushing_plus", "Rushing+"],
["pass_defense_plus", "Pass D+"],
["run_defense_plus", "Run D+"],
]

function unitNum(value){
return value === undefined || value === null || value === "" || Number.isNaN(Number(value)) ? null : Number(value)
}

function unitTeamName(code){
return typeof teamName === "function" ? teamName(code, "nfl") : (code || "")
}

// 30 points is one standard deviation (nfl_units.py): 130+ is about one
// better than average, 70- one worse.
const UNIT_POINTS_PER_SD = 30

function unitCellClass(value){
const n = unitNum(value)
if(n === null){ return "" }
return n >= 100 + UNIT_POINTS_PER_SD ? "unitStrong" : n <= 100 - UNIT_POINTS_PER_SD ? "unitWeak" : ""
}

function sortUnitRows(rows, key){
const sorted = rows.slice()
if(!key || key === "team"){
return sorted.sort((a, b)=>unitTeamName(a.team).localeCompare(unitTeamName(b.team)))
}
return sorted.sort((a, b)=>(unitNum(b[key]) ?? -Infinity) - (unitNum(a[key]) ?? -Infinity))
}

function unitsTableHtml(rows, sortKey){
if(!rows || !rows.length){ return "Team unit ratings appear after the next weekly update." }
const header = [["team", "Team"], ["qb_name", "QB"], ...UNIT_COLUMNS]
.map(([key, label])=>{
const sortable = key !== "qb_name"
const marker = sortKey === key ? " ▼" : ""
return `<th${sortable ? ` data-units-sort="${key}" style="cursor:pointer" title="Sort by ${label}"` : ""}>${label}${marker}</th>`
}).join("")
const body = sortUnitRows(rows, sortKey).map(r=>{
const opponent = r.opponent ? `vs ${unitTeamName(r.opponent)}` : "bye"
const cells = UNIT_COLUMNS.map(([key])=>{
const n = unitNum(r[key])
return `<td class="${unitCellClass(n)}">${n === null ? "-" : Math.round(n)}</td>`
}).join("")
return `<tr><td>${unitTeamName(r.team)} <span class="unitOpp">${opponent}</span></td><td>${r.qb_name || "-"}</td>${cells}</tr>`
}).join("")
return `<table class="unitsTable"><tr>${header}</tr>${body}</table>`
}

// Which units a prop leans on: (label, team side, unit column) pairs.
// "own" is the player's team, "opp" the defense or offense he faces.
const PROP_UNIT_CONTEXT = {
"Receptions": [["His QB+", "own", "qb_plus"], ["Opp Pass D+", "opp", "pass_defense_plus"]],
"Receiving Yards": [["His QB+", "own", "qb_plus"], ["Opp Pass D+", "opp", "pass_defense_plus"]],
"Rushing Yards": [["His Rushing+", "own", "rushing_plus"], ["Opp Run D+", "opp", "run_defense_plus"]],
"Passing Yards": [["His Receiving+", "own", "receiving_plus"], ["Opp Pass D+", "opp", "pass_defense_plus"]],
"Sacks": [["Opp QB+", "opp", "qb_plus"]],
}

// "His QB+ 112 · Opp Pass D+ 95" for one prop row, given unit rows keyed
// by team code. Empty when the category or the teams aren't covered.
function propUnitContext(row, unitsByTeam){
const context = PROP_UNIT_CONTEXT[row.category]
if(!context || !unitsByTeam){ return "" }
const parts = context.map(([label, side, key])=>{
const unit = unitsByTeam[side === "own" ? row.team : row.opponent]
const n = unit ? unitNum(unit[key]) : null
return n === null ? null : `${label} ${Math.round(n)}`
}).filter(Boolean)
return parts.join(" · ")
}

// This week's matchups, one row per team with an opponent (byes listed
// last, no differential):
//   passing  (QB+ + Receiving+) / 2 - the opponent's Pass D+
//   rushing  Rushing+ - the opponent's Run D+
// Higher is a friendlier matchup for the offense.
const MATCHUP_KINDS = {
passing: { offense: r => (unitNum(r.qb_plus) === null || unitNum(r.receiving_plus) === null)
? null : (unitNum(r.qb_plus) + unitNum(r.receiving_plus)) / 2,
defense: "pass_defense_plus", offenseLabel: "QB/Receiving avg", defenseLabel: "Opp Pass D+" },
rushing: { offense: r => unitNum(r.rushing_plus),
defense: "run_defense_plus", offenseLabel: "Rushing+", defenseLabel: "Opp Run D+" },
}

function matchupDifferentials(rows, kind){
const spec = MATCHUP_KINDS[kind]
const byTeam = unitsByTeam(rows)
const out = (rows || []).map(r=>{
const opponent = r.opponent ? byTeam[r.opponent] : null
const offense = spec.offense(r)
const defense = opponent ? unitNum(opponent[spec.defense]) : null
return {
team: r.team, opponent: r.opponent || "", offense, defense,
differential: offense === null || defense === null ? null : offense - defense,
}
})
return out.sort((a, b)=>(b.differential ?? -Infinity) - (a.differential ?? -Infinity)
|| unitTeamName(a.team).localeCompare(unitTeamName(b.team)))
}

// Same colour bands as the units: a gap of a standard deviation or more.
function differentialCellClass(value){
if(value === null){ return "" }
return value >= UNIT_POINTS_PER_SD ? "unitStrong" : value <= -UNIT_POINTS_PER_SD ? "unitWeak" : ""
}

function formatDifferential(value){
if(value === null){ return "-" }
const shown = Number.isInteger(value) ? String(Math.abs(value)) : Math.abs(value).toFixed(1)
return `${value > 0 ? "+" : value < 0 ? "−" : ""}${shown}`
}

// One entry per game this week: its two teams (away first when the file
// says who's home) with each team's passing and rushing differential.
// Ordered by the most lopsided matchup in the game first.
function matchupGames(rows){
const passing = {}, rushing = {}
matchupDifferentials(rows, "passing").forEach(m=>{ passing[m.team] = m.differential })
matchupDifferentials(rows, "rushing").forEach(m=>{ rushing[m.team] = m.differential })
const byTeam = unitsByTeam(rows)
const seen = new Set()
const games = []
;(rows || []).forEach(r=>{
if(!r.opponent || seen.has(r.team) || !byTeam[r.opponent]){ return }
seen.add(r.team); seen.add(r.opponent)
const other = byTeam[r.opponent]
const known = unitNum(r.home) !== null
let away = r, home = other
if(known ? unitNum(r.home) === 1 : unitTeamName(r.team) > unitTeamName(other.team)){ away = other; home = r }
const side = t=>({ team: t.team, passing: passing[t.team] ?? null, rushing: rushing[t.team] ?? null })
games.push({ title: `${unitTeamName(away.team)} ${known ? "at" : "vs"} ${unitTeamName(home.team)}`, teams: [side(away), side(home)] })
})
const biggest = g=>Math.max(...g.teams.flatMap(t=>[t.passing, t.rushing]).filter(v=>v !== null).map(Math.abs), -Infinity)
return games.sort((a, b)=>biggest(b) - biggest(a))
}

function matchupCardsHtml(rows){
const games = matchupGames(rows)
if(!games.length){ return "" }
return games.map(g=>`<div class="pickCard matchupCard">
<div class="pickName">${g.title}</div>
<table class="epaTable"><thead><tr><th></th><th>Passing differential</th><th>Rushing differential</th></tr></thead>
<tbody>${g.teams.map(t=>`<tr><td>${unitTeamName(t.team)}</td>`
+ `<td class="${differentialCellClass(t.passing)}">${formatDifferential(t.passing)}</td>`
+ `<td class="${differentialCellClass(t.rushing)}">${formatDifferential(t.rushing)}</td></tr>`).join("")}</tbody>
</table>
</div>`).join("")
}

// Teams without a game this week.
function byeTableHtml(rows){
const byes = (rows || []).filter(r=>!r.opponent).sort((a, b)=>unitTeamName(a.team).localeCompare(unitTeamName(b.team)))
if(!byes.length){ return "No teams on bye this week." }
return `<table class="unitsTable"><tr><th>Team</th><th>QB</th></tr>${
byes.map(r=>`<tr><td>${unitTeamName(r.team)}</td><td>${r.qb_name || "-"}</td></tr>`).join("")}</table>`
}

function unitsByTeam(rows){
const byTeam = {}
;(rows || []).forEach(r=>{ byTeam[r.team] = r })
return byTeam
}

if (typeof module !== "undefined" && module.exports) {
module.exports = {
UNIT_COLUMNS, unitsTableHtml, sortUnitRows, unitCellClass, propUnitContext, unitsByTeam,
matchupDifferentials, matchupGames, matchupCardsHtml, byeTableHtml, differentialCellClass, formatDifferential,
}
}
