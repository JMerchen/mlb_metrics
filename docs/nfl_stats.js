// NFL Stats tab (nfl.html): team schedules on a paper sheet, power
// rankings, team profiles and players' season views. Pure functions
// returning HTML strings, so node tests can check them; nfl.js does the
// DOM wiring. Shares mlb_render.js's helpers (percentile rows, meters,
// formatting) and its CSS (pictures, paper sheet, profile rows).
//
// Data (scripts/build_nfl_stats.py): nfl_team_schedule.csv,
// nfl_team_stats.csv, nfl_player_stats.csv, plus nfl_team_units.csv.

// ESPN's logo codes differ from nflverse's for two teams.
const NFL_ESPN_CODES = { LA: "lar", WAS: "wsh" }

function nflLogoUrl(team){
const code = String(team || "").trim().toUpperCase()
return code ? `https://a.espncdn.com/i/teamlogos/nfl/500/${NFL_ESPN_CODES[code] || code.toLowerCase()}.png` : ""
}

// Headshots come with the player data (nflverse's headshot_url).
function nflHeadshotUrl(url){
return url || ""
}

function nflTeam(code){
return typeof teamName === "function" ? teamName(code, "nfl") : (code || "")
}

function nflLogo(team, size = "sm"){
const url = nflLogoUrl(team)
return `<span class="mlbLogo mlbLogo-${size}" title="${mlbEsc(nflTeam(team))}" aria-hidden="true"><span>${mlbEsc(team || "")}</span>${
url ? `<img src="${url}" alt="" loading="lazy" onload="this.parentNode.classList.add('loaded')" onerror="this.remove()">` : ""}</span>`
}

function nflAvatar(url, name, size = "sm"){
const src = nflHeadshotUrl(url)
return `<span class="mlbAvatar mlbAvatar-${size}" aria-hidden="true"><span>${mlbEsc(mlbInitials(name))}</span>${
src ? `<img src="${mlbEsc(src)}" alt="" loading="lazy" onload="this.parentNode.classList.add('loaded')" onerror="this.remove()">` : ""}</span>`
}

// ---- Team schedule sheet ----

// Ink deepens with the margin, light at 1 point and deepest at 21 (three
// scores) or more. Ties get a grey dash.
const NFL_DEEPEST_MARGIN = 21

function nflScheduleInk(result, margin){
if(result === "T"){ return "rgb(120,112,100)" }
const m = Math.min(NFL_DEEPEST_MARGIN, Math.max(1, Math.abs(Number(margin) || 1)))
const t = (m - 1) / (NFL_DEEPEST_MARGIN - 1)
const mix = (a, b) => a.map((c, i) => Math.round(c + (b[i] - c) * t))
const rgb = result === "W" ? mix([112, 192, 132], [8, 110, 46]) : mix([232, 132, 132], [160, 14, 18])
return `rgb(${rgb.join(",")})`
}

function nflMarkSvg(result, ink){
const shape = result === "W" ? `<ellipse cx="20" cy="20" rx="17" ry="15.5" transform="rotate(-12 20 20)"/>`
: result === "L" ? `<path d="M7 8 L33 32 M33 7 L8 33"/>`
: result === "T" ? `<path d="M6 21 L34 19"/>` : ""
return shape ? `<svg viewBox="0 0 40 40" class="schedMark ${result === "W" ? "schedWin" : "schedLoss"}" style="stroke:${ink}" aria-hidden="true">${shape}</svg>` : ""
}

function nflWeekRow(g){
const week = `<span class="nflWeek">Wk ${mlbEsc(g.week)}<small>${g.date ? mlbShortDate(g.date) : ""}</small></span>`
if(g.result === "BYE"){
return `<div class="nflSchedRow nflBye">${week}<span class="nflByeText">Bye week</span></div>`
}
const where = String(g.home).startsWith("1") ? "vs." : "at"
const played = ["W", "L", "T"].includes(g.result)
const pf = mlbNum(g.points_for), pa = mlbNum(g.points_against)
const score = played ? `${g.result} ${pf}–${pa}` : "Upcoming"
const tip = `Week ${g.week}${g.date ? `, ${mlbShortDate(g.date)}` : ""}: ${where} ${nflTeam(g.opponent)} - ${played ? score : "not played yet"}`
return `<div class="nflSchedRow${played ? "" : " nflUpcoming"}" data-tip="${mlbEsc(tip)}" tabindex="0">${week}
<span class="nflWhere">${where}</span>
<span class="schedGame nflOpp">${nflLogo(g.opponent, "md")}${played ? nflMarkSvg(g.result, nflScheduleInk(g.result, g.margin)) : ""}</span>
<span class="nflOppName">${mlbEsc(nflTeam(g.opponent))}</span>
<span class="nflScore">${mlbEsc(score)}</span>
</div>`
}

// A team's season on one sheet: 18 weeks in three panels of six.
function nflScheduleSheetHtml(rows, team){
const weeks = (rows || []).filter(r => r.team === team).sort((a, b) => Number(a.week) - Number(b.week))
if(!weeks.length){ return `<div class="paperEmpty">No schedule for this team yet</div>` }
const played = weeks.filter(g => ["W", "L", "T"].includes(g.result))
const wins = played.filter(g => g.result === "W").length
const losses = played.filter(g => g.result === "L").length
const ties = played.filter(g => g.result === "T").length
const diff = played.reduce((sum, g) => sum + (mlbNum(g.margin) || 0), 0)
const next = weeks.find(g => g.result === "")
const nextText = next ? ` · next: ${String(next.home).startsWith("1") ? "vs." : "at"} ${nflTeam(next.opponent)}${next.date ? `, ${mlbShortDate(next.date)}` : ""}` : ""
const per = Math.ceil(weeks.length / 3)
const panels = [0, 1, 2].map(c => `<div class="paperPanel" style="--panel:${c}">${weeks.slice(c * per, (c + 1) * per).map(nflWeekRow).join("")}</div>`).join("")
const key = (result, margins) => margins.map(m => `${nflMarkSvg(result, nflScheduleInk(result, result === "L" ? -m : m))}<small>${m === NFL_DEEPEST_MARGIN ? `${m}+` : m}</small>`).join("")
return `<div class="paperHead">
<div class="paperTeam">${nflLogo(team, "lg")}<div><b>${mlbEsc(nflTeam(team))}</b><small>${wins}-${losses}${ties ? `-${ties}` : ""} · point differential ${diff > 0 ? "+" : diff < 0 ? "−" : ""}${Math.abs(diff)}${mlbEsc(nextText)}</small></div></div>
<div class="paperKey"><span>Won by</span>${key("W", [1, 10, NFL_DEEPEST_MARGIN])}<span>Lost by</span>${key("L", [1, 10, NFL_DEEPEST_MARGIN])}</div>
</div>
<div class="paperPanels nflPanels">${panels}</div>`
}

function nflSchedulePickerHtml(teams, selected){
return (teams || []).slice().sort((a, b) => nflTeam(a).localeCompare(nflTeam(b))).map(t =>
`<button class="schedPick${t === selected ? " schedPicked" : ""}" data-sched-team="${mlbEsc(t)}" title="${mlbEsc(nflTeam(t))}" aria-pressed="${t === selected}">${nflLogo(t, "md")}<span>${mlbEsc(t)}</span></button>`).join("")
}

// ---- Power rankings and team profiles ----

function nflRecord(t){
const ties = Number(t.ties) || 0
return `${Number(t.wins) || 0}-${Number(t.losses) || 0}${ties ? `-${ties}` : ""}`
}

function nflSigned(value, digits = 2){
const n = mlbNum(value)
if(n === null){ return "-" }
const fixed = Math.abs(n).toFixed(digits)
return `${n > 0 ? "+" : n < 0 ? "−" : ""}${fixed.replace(/^0(?=\.)/, "")}`
}

// Every team by net EPA per play (offense minus defense), as a diverging
// bar around zero. Each row is a button that opens the team's profile.
function nflPowerRankingsHtml(teams, selected){
const sorted = (teams || []).filter(t => mlbNum(t.net_epa_play) !== null)
.slice().sort((a, b) => mlbNum(b.net_epa_play) - mlbNum(a.net_epa_play))
if(!sorted.length){ return "" }
const reach = Math.max(...sorted.map(t => Math.abs(mlbNum(t.net_epa_play))), 0.05)
const picked = new Set(selected || [])
const row = (t, i) => {
const net = mlbNum(t.net_epa_play)
const width = Math.abs(net) / reach * 50
return `<button class="powerRow${picked.has(t.team) ? " powerPicked" : ""}" data-team="${mlbEsc(t.team)}" data-tip="${mlbEsc(`${nflTeam(t.team)}: ${mlbOrdinal(i + 1)}, ${nflRecord(t)}, net EPA ${nflSigned(net, 3)} a play, point differential ${nflSigned(t.point_diff_pg, 1)} a game`)}">
<span class="rankNo">${i + 1}</span>${nflLogo(t.team, "xs")}<span class="powerName">${mlbEsc(nflTeam(t.team))} <small class="dim">${nflRecord(t)}</small></span>
<span class="mlbMeter"><span class="mlbMeterMid"></span><span class="mlbMeterBar" style="${net >= 0 ? "left:50%" : "right:50%"};width:${width.toFixed(1)}%;background:${net >= 0 ? "var(--mlb-better)" : "var(--mlb-worse)"}"></span></span>
<span class="powerValue">${nflSigned(net, 3)}</span>
</button>`
}
const half = Math.ceil(sorted.length / 2)
return `<div class="vizCard">
<div class="vizHead"><b>Power rankings</b><span class="vizKey"><i class="keyBetter"></i>Net EPA per play above zero <i class="keyWorse"></i>Below</span></div>
<div class="powerCols"><div>${sorted.slice(0, half).map(row).join("")}</div><div>${sorted.slice(half).map((t, i) => row(t, i + half)).join("")}</div></div>
</div>`
}

// [column, label, format, direction] - direction -1 means lower is better.
const NFL_TEAM_PROFILE_GROUPS = [
["Results", [
["win_pct", "Win %", "pct", 1],
["point_diff_pg", "Point differential a game", "signed1", 1],
["ppg", "Points a game", "fixed1", 1],
["papg", "Points allowed a game", "fixed1", -1],
]],
["Offense", [
["off_epa_play", "EPA per play", "signed3", 1],
["off_pass_epa", "EPA per pass", "signed3", 1],
["off_rush_epa", "EPA per run", "signed3", 1],
["off_success", "Success rate", "pct", 1],
["giveaways_pg", "Giveaways a game", "fixed1", -1],
]],
["Defense", [
["def_epa_play", "EPA allowed per play", "signed3", -1],
["def_pass_epa", "EPA allowed per pass", "signed3", -1],
["def_rush_epa", "EPA allowed per run", "signed3", -1],
["def_success", "Success rate allowed", "pct", -1],
["takeaways_pg", "Takeaways a game", "fixed1", 1],
]],
["Units (100 = average)", [
["qb_plus", "QB+", "int", 1],
["receiving_plus", "Receiving+", "int", 1],
["rushing_plus", "Rushing+", "int", 1],
["pass_defense_plus", "Pass D+", "int", 1],
["run_defense_plus", "Run D+", "int", 1],
]],
]

function nflStatText(value, format){
const n = mlbNum(value)
if(n === null){ return "-" }
switch(format){
case "pct": return mlbPct(n, 1)
case "signed1": return nflSigned(n, 1)
case "signed3": return nflSigned(n, 3)
case "fixed1": return n.toFixed(1)
case "fixed2": return n.toFixed(2)
case "int": return String(Math.round(n))
default: return String(n)
}
}

// Team stats joined with their unit ratings (season view) and win %.
function nflTeamRows(stats, units){
const byTeam = {}
;(units || []).forEach(u => { byTeam[u.team] = u })
return (stats || []).map(t => {
const u = byTeam[t.team] || {}
const games = mlbNum(t.games) || 0
const winPct = games ? ((mlbNum(t.wins) || 0) + 0.5 * (mlbNum(t.ties) || 0)) / games : null
return Object.assign({}, t, {
win_pct: winPct,
qb_plus: u.qb_plus_season ?? u.qb_plus, receiving_plus: u.receiving_plus_season ?? u.receiving_plus,
rushing_plus: u.rushing_plus_season ?? u.rushing_plus, pass_defense_plus: u.pass_defense_plus_season ?? u.pass_defense_plus,
run_defense_plus: u.run_defense_plus_season ?? u.run_defense_plus,
})
})
}

function nflTeamRank(teams, team, key, direction){
const value = mlbNum(team[key])
if(value === null){ return null }
const sign = direction < 0 ? -1 : 1
return 1 + teams.filter(t => mlbNum(t[key]) !== null && sign * mlbNum(t[key]) > sign * value).length
}

// One team, or two side by side on the same tracks (as on the MLB Stats page).
function nflTeamProfileHtml(selected, teams){
const chosen = (selected || []).map(code => teams.find(t => t.team === code)).filter(Boolean)
if(!chosen.length){ return "Select a team" }
const n = teams.length
const header = chosen.map(t => `<div class="teamHead">${nflLogo(t.team, "lg")}<div><div class="teamHeadName">${mlbEsc(nflTeam(t.team))}</div>
<div class="teamHeadMeta">${nflRecord(t)} · ${mlbOrdinal(nflTeamRank(teams, t, "net_epa_play", 1) || n)} of ${n} in net EPA</div></div></div>`).join(`<span class="teamVs">vs</span>`)
const groups = NFL_TEAM_PROFILE_GROUPS.map(([title, stats]) => {
const rows = stats.filter(([key]) => chosen.some(t => mlbNum(t[key]) !== null)).map(([key, label, format, direction]) => {
const place = t => { const r = nflTeamRank(teams, t, key, direction); return r === null ? null : Math.round(100 * (n - r) / Math.max(1, n - 1)) }
if(chosen.length === 1){
const t = chosen[0], r = nflTeamRank(teams, t, key, direction)
return mlbPercentileRow(label, nflStatText(t[key], format), place(t),
{ dotText: r === null ? "" : r, tip: `${label}: ${nflStatText(t[key], format)} (${r ? `${mlbOrdinal(r)} of ${n}` : "-"})` })
}
const markers = chosen.map((t, i) => { const p = place(t); return p === null ? "" : `<span class="pctMarker pctMarker${i}" style="left:${p}%">${nflLogo(t.team, "xs")}</span>` }).join("")
const values = chosen.map(t => `<span class="pctPair">${nflLogo(t.team, "xs")}${mlbEsc(nflStatText(t[key], format))}</span>`).join("")
const tip = `${label}: ` + chosen.map(t => `${nflTeam(t.team)} ${nflStatText(t[key], format)} (${mlbOrdinal(nflTeamRank(teams, t, key, direction) || n)})`).join(", ")
return `<div class="pctRow" data-tip="${mlbEsc(tip)}" tabindex="0"><span class="pctLabel">${mlbEsc(label)}</span>
<span class="pctTrack pctTrackCompare"><span class="pctMid"></span>${markers}</span><span class="pctValue">${values}</span></div>`
}).join("")
return rows ? `<div class="profileGroup"><div class="profileGroupTitle">${mlbEsc(title)}</div>${mlbScaleRow("Worst", "Best")}${rows}</div>` : ""
}).join("")
const note = chosen.length === 1 ? `Numbers in the dots are ranks, 1 to ${n}.` : `Each logo sits at that team's rank among the ${n}.`
return `<div class="vizCard teamProfile${chosen.length > 1 ? " teamCompare" : ""}"><div class="teamHeads">${header}</div><div class="vizSub">${note}</div><div class="profileGrid">${groups}</div></div>`
}

// ---- Players ----

// Per position: [column, heading, short heading, format, direction].
const NFL_PLAYER_COLUMNS = {
QB: [["pass_ypg", "Pass yds/g", "Yds/g", "fixed1", 1], ["ypa", "Yards/att", "Y/A", "fixed1", 1], ["comp_pct", "Comp %", "Cmp", "pct", 1],
["epa_db", "EPA/dropback", "EPA", "signed2", 1], ["td_rate", "TD rate", "TD%", "pct", 1], ["int_rate", "INT rate", "INT%", "pct", -1], ["rush_ypg", "Rush yds/g", "Rush", "fixed1", 1]],
RB: [["rush_ypg", "Rush yds/g", "Yds/g", "fixed1", 1], ["carries_pg", "Carries/g", "Car", "fixed1", 1], ["ypc", "Yards/carry", "Y/C", "fixed1", 1],
["epa_carry", "EPA/carry", "EPA", "signed2", 1], ["rec_pg", "Catches/g", "Rec", "fixed1", 1], ["rec_ypg", "Rec yds/g", "RecY", "fixed1", 1]],
WR: [["targets_pg", "Targets/g", "Tgt", "fixed1", 1], ["rec_pg", "Catches/g", "Rec", "fixed1", 1], ["rec_ypg", "Rec yds/g", "Yds/g", "fixed1", 1],
["ypt", "Yards/target", "Y/T", "fixed1", 1], ["epa_target", "EPA/target", "EPA", "signed2", 1], ["target_share", "Target share", "Share", "pct", 1]],
}
NFL_PLAYER_COLUMNS.TE = NFL_PLAYER_COLUMNS.WR

function nflPlayerValue(row, key, view){
return mlbNum(view === "faced" ? row[`${key}_adj`] : row[key])
}

function nflPlayerText(value, format){
if(value === null){ return "-" }
return format === "signed2" ? nflSigned(value, 2) : nflStatText(value, format)
}

function sortNflPlayers(rows, key, view, direction = 1){
const sorted = rows.slice()
if(key === "name"){ return sorted.sort((a, b) => String(a.player_name).localeCompare(String(b.player_name))) }
return sorted.sort((a, b) => direction * ((nflPlayerValue(b, key, view) ?? -Infinity) - (nflPlayerValue(a, key, view) ?? -Infinity)))
}

function nflPlayerTableHtml(players, position, sortKey, view = "season", limit = 40){
const columns = NFL_PLAYER_COLUMNS[position] || []
const rows = (players || []).filter(p => p.position === position)
if(!rows.length){ return "No players yet" }
const key = sortKey && (sortKey === "name" || columns.some(c => c[0] === sortKey)) ? sortKey : columns[0][0]
const direction = (columns.find(c => c[0] === key) || [])[4] === -1 ? -1 : 1
const faced = view === "faced"
const head = `<th><button class="sortHead${key === "name" ? " sorted" : ""}" data-sort="name">Player${key === "name" ? " ▼" : ""}</button></th>` +
columns.map(([col, label, short]) => `<th class="num"><button class="sortHead${key === col ? " sorted" : ""}" data-sort="${col}"><span class="headLong">${label}</span><span class="headShort">${short}</span>${key === col ? (direction < 0 ? " ▲" : " ▼") : ""}</button></th>`).join("")
const body = sortNflPlayers(rows, key, view, direction).slice(0, limit).map((p, i) => `<tr>
<td><span class="statWho"><span class="rankNo">${i + 1}</span>${nflAvatar(p.headshot_url, p.player_name, "sm")}<span class="rankWho"><b>${mlbEsc(p.player_name)}</b><small>${nflLogo(p.team, "xs")} ${mlbEsc(nflTeam(p.team))} · ${mlbEsc(p.games)} g</small></span></span></td>
${columns.map(([col, , , format]) => {
const value = nflPlayerValue(p, col, view)
const base = mlbNum(p[col])
const moved = faced && value !== null && base !== null ? value - base : null
// Below half the last shown digit, the move rounds away: say so.
const smallest = format === "pct" ? 0.0005 : format === "signed2" ? 0.005 : 0.05
const note = moved === null ? "" : `<small class="adjNote">${Math.abs(moved) < smallest ? "no change" : `${moved > 0 ? "+" : "−"}${format === "pct" ? mlbPct(Math.abs(moved), 1) : Math.abs(moved).toFixed(format === "signed2" ? 2 : 1)}`}</small>`
return `<td class="num"><b>${nflPlayerText(value, format)}</b>${note}</td>`
}).join("")}
</tr>`).join("")
return `<div class="scrollbox statTableBox"><table class="statTable"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`
}

// Player card: headshot, team, then each stat as a percentile among
// qualified players at his position.
function nflPlayerCardHtml(player, players, view = "season"){
const columns = NFL_PLAYER_COLUMNS[player.position] || []
const pool = (players || []).filter(p => p.position === player.position)
const bars = columns.map(([col, label, , format, direction]) => {
const value = nflPlayerValue(player, col, view)
const p = mlbPercentile(pool.map(r => nflPlayerValue(r, col, view)), value, direction)
return mlbPercentileRow(label, nflPlayerText(value, format), p,
{ tip: p === null ? `${label}: no data yet` : `${label}: ${nflPlayerText(value, format)} - better than ${p}% of ${player.position}s` })
}).join("")
return `<div class="vizCard playerCard">
<div class="playerCardHead">${nflAvatar(player.headshot_url, player.player_name, "xl")}
<div><div class="teamHeadName">${mlbEsc(player.player_name)}</div><div class="teamHeadMeta">${nflLogo(player.team, "xs")} ${mlbEsc(nflTeam(player.team))} · ${mlbEsc(player.position)} · ${mlbEsc(player.games)} games</div></div></div>
<div class="profileGrid"><div class="profileGroup"><div class="profileGroupTitle">Percentile among ${mlbEsc(player.position)}s${view === "faced" ? ", vs. defenses faced" : ""}</div>${mlbScaleRow("Worst", "Best")}${bars}</div></div>
</div>`
}

function nflPlayerSuggestionsHtml(matches){
if(!matches.length){ return `<span class="dim">No players found</span>` }
return matches.map((p, i) => `<button class="playerChoice" data-index="${i}">${nflAvatar(p.headshot_url, p.player_name, "xs")}<span>${mlbEsc(p.player_name)}</span><small>${mlbEsc(p.position)} · ${mlbEsc(p.team)}</small></button>`).join("")
}

if (typeof module !== "undefined" && module.exports) {
module.exports = {
nflLogoUrl, nflHeadshotUrl, nflLogo, nflAvatar, nflScheduleInk, nflScheduleSheetHtml, nflSchedulePickerHtml,
nflPowerRankingsHtml, nflTeamRows, nflTeamRank, nflTeamProfileHtml, NFL_TEAM_PROFILE_GROUPS,
NFL_PLAYER_COLUMNS, sortNflPlayers, nflPlayerTableHtml, nflPlayerCardHtml, nflPlayerSuggestionsHtml, nflSigned,
}
}
