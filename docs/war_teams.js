// Team WAR in context (MLB Stats and NFL Stats): each team's record set
// against what its players' value adds up to, and what is left over
// (user request, 2026-11-23: "make war work in context and explain a
// team's luck versus skill").
//
// A replacement-level team wins some games on its own (about 44 of 162
// in baseball, about 3 of 17 in football: derived and tested, see
// mlb_war.team_table and nfl_player_value.team_table). A team's WAR is
// the wins its players add on top of that, so baseline + WAR = expected
// wins. The gap between expected and actual wins is split in two:
//   uncredited  what the team's real scoring margin says beyond its
//               players' ledgers
//   close       how those runs or points fell across games
// Data: the `teams` list and `replacement_win_rate` in mlb_war_<season>.json
// and nfl_player_value_<season>.json. Pure functions; each page wires
// its own clicks.

// [key, long heading, short heading, digits, signed]
const WT_COLUMNS = [
["wins", "Record", "W-L", 0, false],
["war", "Team WAR", "WAR", 1, false],
["expected", "Expected wins", "Exp.", 1, false],
["uncredited", "Not credited", "Not cred.", 1, true],
["close", "Close games", "Close", 1, true],
["offense", "Offense", "Off", 0, true],
["defense", "Defense", "Def", 0, true],
]

function wtEsc(value){
return String(value ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;")
}

// "+4.2", "−3.0", "0.0": a real minus sign, and no signed zero.
function wtSigned(value, digits){
const n = Number(value), text = Math.abs(n).toFixed(digits)
return Number(text) === 0 ? text : `${n < 0 ? "−" : "+"}${text}`
}

// 12 wins and 5 losses is "12-5"; a tie shows as a third number.
function wtRecord(team){
const ties = 2 * (Number(team.wins) % 1)
const wins = Math.floor(Number(team.wins)), losses = Math.floor(Number(team.losses))
return ties ? `${wins}-${losses}-${Math.round(ties)}` : `${wins}-${losses}`
}

function wtSort(teams, key, dir = -1){
const value = t => key === "team" ? String(t.name || t.team || "").toLowerCase() : Number(t[key])
return (teams || []).slice().sort((a, b) => {
const x = value(a), y = value(b)
if(x === y){ return Number(b.expected) - Number(a.expected) }
return (x < y ? -1 : 1) * (dir < 0 ? -1 : 1) * (key === "team" ? -1 : 1)
})
}

// What each heading means, for its tooltip. `unit` is "runs" or "points".
function wtTips(unit){
return {
wins: "Wins and losses",
war: "Wins its players add on top of a replacement team's: expected wins less the replacement baseline",
expected: `The wins its players' ${unit} make: an average team plus what its players were worth, turned into wins`,
uncredited: `Wins in its real ${unit === "runs" ? "run" : "point"} differential that no player's ledger holds`,
close: `Wins beyond what its ${unit} scored and allowed make: how they fell across games`,
offense: `Its players' ${unit} on offense against the league's average team`,
defense: unit === "points" ? "Points its defense, as a unit, took off the offenses it faced, against the league's average defense"
: `Its players' ${unit} saved on defense against the league's average team`,
}
}

// opts: {unit: "runs" | "points", seasonGames: 162 | 17, sort, dir, name: code => name, logo: code => html}
function wtTeamsHtml(data, opts = {}){
const teams = data && data.teams
if(!teams || !teams.length){ return "" }
const name = opts.name || (code => code), logo = opts.logo || (() => "")
const columns = WT_COLUMNS
const key = opts.sort === "team" || columns.some(c => c[0] === opts.sort) ? opts.sort : "expected"
const dir = opts.dir === 1 ? 1 : -1
const arrow = dir < 0 ? " ▼" : " ▲"
const tips = wtTips(opts.unit || "runs")
const rows = wtSort(teams.map(t => ({...t, name: name(t.team)})), key, dir)
const head = `<th><button class="sortHead${key === "team" ? " sorted" : ""}" data-team-sort="team">Team${key === "team" ? arrow : ""}</button></th>`
+ columns.map(([col, label, short]) => `<th class="num${col === "offense" || col === "defense" ? " pvWide" : ""}"><button class="sortHead${key === col ? " sorted" : ""}" data-team-sort="${col}" data-tip="${wtEsc(tips[col])}"><span class="headLong">${label}</span><span class="headShort">${short}</span>${key === col ? arrow : ""}</button></th>`).join("")
const body = rows.map((t, i) => `<tr>
<td><span class="statWho pvWho"><span class="rankNo">${i + 1}</span><span class="rankWho"><b>${logo(t.team)} ${wtEsc(t.name)}</b></span></span></td>
${columns.map(([col, , , digits, signed]) => {
const text = col === "wins" ? wtRecord(t) : signed ? wtSigned(t[col], digits) : Number(t[col]).toFixed(digits)
const tone = signed && (col === "uncredited" || col === "close") ? (Number(t[col]) >= 0.05 ? " wtUp" : Number(t[col]) <= -0.05 ? " wtDown" : "") : ""
return `<td class="num${tone}${col === "offense" || col === "defense" ? " pvWide" : ""}">${col === "war" ? `<b>${text}</b>` : text}</td>`
}).join("")}
</tr>`).join("")
const games = opts.seasonGames || 162
const rate = Number(data.replacement_win_rate)
const baseline = Number.isFinite(rate) && rate > 0
? `A replacement-level team would win about <b>${(rate * games).toFixed(games > 100 ? 0 : 1)} of ${games}</b> games (${Math.round(100 * rate)}%). Team WAR is the wins a team's players add on top of that.` : ""
return `<div class="pvSummary"><span>${baseline}</span></div>
<div class="scrollbox statTableBox"><table class="statTable pvTable wtTable"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`
}

if (typeof module !== "undefined" && module.exports) {
module.exports = { WT_COLUMNS, wtSigned, wtRecord, wtSort, wtTips, wtTeamsHtml }
}
