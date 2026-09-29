// Real, matchup-based NFL player prop rankings - rendering helpers for
// nfl_player_props.py's own real output shape (a different real shape
// from a consensus board's "multiple sources agree on one rank" output
// - see that module's own docstring - so this is its own small file
// rather than forced through consensus_board.js's own
// buildConsensusTable). Same "tested pure-logic file feeds an untested
// bootstrap file's DOM wiring via plain globals" split
// docs/consensus_board.js/docs/nfl_draft_assistant.js already establish
// - this file deliberately has NO top-level side-effecting call, so it
// stays safely require()-able from node --test.

const PLAYER_PROPS_COLUMN_LABELS = {
rank: "#", game: "Game",
player_name: "Player", team: "Team", opponent: "Opp", position: "Pos",
category: "Category", direction: "Bet", player_rate: "Own Avg",
projection: "Projected", opponent_allowed_rate: "Opp Allows",
league_rate: "League Avg", ratio: "Opp vs Avg", games: "Games",
}
const PLAYER_PROPS_COLUMNS = [
"rank", "game", "player_name", "team", "opponent", "position", "category", "direction",
"player_rate", "projection", "opponent_allowed_rate", "league_rate", "ratio", "games",
]

// player_rate and projection are per-GAME quantities; league_rate and
// opponent_allowed_rate are per-PLAY for the three skill categories and
// per-game for Passing Yards/Sacks (the row's own `rate_basis` column
// says which). One decimal place throughout - real precision without a
// false extra digit of certainty - except that a per-play rate like a
// catch rate needs two to be readable at all.
//
// `ratio` is labelled "Opp vs Avg", NOT "Matchup Edge" as it was until
// 2026-09-17, and the rename is the point rather than cosmetic. Under
// the old model this column WAS the board's ranking signal, so calling
// it the edge was fair; it is now just one input among several, because
// the skill categories rank on how far the projection departs from the
// player's own average (see nfl_player_props.top_prop_bets). Leaving it
// named "Matchup Edge" would tell a reader that a +3% row is a weak pick
// when the pick may rest almost entirely on usage instead.
//
// A user reading the old column also, correctly, could not get signal
// out of it: it was clipped to config.NFL_PROP_MATCHUP_CLIP (0.8, 1.2),
// so it could only ever print between -20% and +20%, and 35.6% of
// qualified rows on the real week-2 slate sat exactly ON a boundary. For
// those rows "+20%" meant "at least 20%, we are not saying how much" -
// two rows 3x apart in true extremity printed identically. The
// per-play ratio the projection model feeds this column is unclipped, so
// the number now means what it says.
function formatPlayerPropsValue(column, value){
// Passing Yards and Sacks carry no projection (nfl_prop_projections
// models receiving and rushing only), so those cells arrive EMPTY from
// the CSV. Number("") is 0, not NaN - the same real JS quirk this
// file's own tests already guard for `ratio` - so without this an
// unprojected row would confidently print "0.0" projected yards.
if(column === "projection" && (value === "" || value === null || value === undefined)){
return "-"
}
if(["player_rate", "projection"].includes(column)){
const n = Number(value)
return isNaN(n) ? value : n.toFixed(1)
}
if(["opponent_allowed_rate", "league_rate"].includes(column)){
const n = Number(value)
if(isNaN(n)){ return value }
// A per-play rate can be a catch rate around 0.6, where one decimal
// place collapses genuinely different defenses onto the same "0.6".
return n < 10 ? n.toFixed(2) : n.toFixed(1)
}
if(column === "ratio"){
const n = Number(value)
if(isNaN(n)){ return value }
const pct = (n - 1) * 100
return `${pct >= 0 ? "+" : ""}${pct.toFixed(0)}%`
}
return value
}

// The board carries the overall top 10 UNION the best few rows of every
// individual matchup (see nfl_player_props.build_prop_board), so a game
// filter always has something to show. "All games" deliberately renders
// only the overall top 10 rather than all ~80 rows, so the default view
// is the same board it has always been - the extra rows exist to make
// single-game views possible, not to lengthen the front page.
const PLAYER_PROPS_ALL_GAMES = "All games"

function playerPropsGames(data){
const seen = []
data.forEach(row=>{
const game = row.game
if(game && !seen.includes(game)){ seen.push(game) }
})
// Ordered by best overall rank, which `data` is already sorted by, so
// the most interesting matchup sits at the top of the dropdown.
return seen
}

function filterPlayerProps(data, game, topN){
if(!game || game === PLAYER_PROPS_ALL_GAMES){
return data.slice(0, topN === undefined ? 10 : topN)
}
return data.filter(row=>row.game === game)
}

function buildPlayerPropsTable(data, id){
const el = document.getElementById(id)
if(!data.length){ el.innerHTML = "No player props data yet - this board updates weekly."; return }
let html = "<table><tr>"
PLAYER_PROPS_COLUMNS.forEach(c=>{ html += `<th>${PLAYER_PROPS_COLUMN_LABELS[c] || c}</th>` })
html += "</tr>"
data.forEach(row=>{
html += "<tr>"
PLAYER_PROPS_COLUMNS.forEach(c=>{ html += `<td>${formatPlayerPropsValue(c, row[c])}</td>` })
html += "</tr>"
})
html += "</table>"
el.innerHTML = html
}

function buildPlayerPropsSection(data, selectId, tableId){
const select = document.getElementById(selectId)
const render = ()=> buildPlayerPropsTable(
filterPlayerProps(data, select ? select.value : PLAYER_PROPS_ALL_GAMES), tableId
)
if(select){
const games = playerPropsGames(data)
select.innerHTML = [PLAYER_PROPS_ALL_GAMES].concat(games)
.map(g=>`<option value="${g}">${g}</option>`).join("")
// A board with no game labels at all (an older CSV written before
// they existed) leaves the picker hidden rather than showing a
// dropdown whose only entry does nothing.
select.style.display = games.length ? "" : "none"
select.onchange = render
}
render()
}



// Season floors by position (nfl_position_floors.py) - a HISTORY board,
// not a prediction one. One tab per position, the top 32 there by season
// snaps, and for every stat the lowest number the player has posted in a
// game this season - zero included - with his average and standard
// deviation under it. Games he left injured are left out of all three.
const POSITION_FLOOR_TABS = ["QB", "RB", "WR", "TE", "DL", "DB"]

// Per-tab stat columns, in the order they were asked for. Keys are the
// CSV's own column names; each has a matching `<key>_log` game log.
const POSITION_FLOOR_STATS = {
QB: [["completions", "Completions"], ["passing_tds", "Pass TDs"], ["passing_yards", "Pass Yds"],
["interceptions_thrown", "INTs"], ["anytime_td", "Anytime TD"], ["rushing_yards", "Rush Yds"]],
RB: [["receptions", "Receptions"], ["receiving_yards", "Rec Yds"], ["rushing_yards", "Rush Yds"], ["anytime_td", "Anytime TD"]],
WR: [["receptions", "Receptions"], ["receiving_yards", "Rec Yds"], ["rushing_yards", "Rush Yds"], ["anytime_td", "Anytime TD"]],
TE: [["receptions", "Receptions"], ["receiving_yards", "Rec Yds"], ["rushing_yards", "Rush Yds"], ["anytime_td", "Anytime TD"]],
DL: [["tackles", "Tackles"], ["sacks", "Sacks"], ["def_interceptions", "INTs"]],
DB: [["tackles", "Tackles"], ["sacks", "Sacks"], ["def_interceptions", "INTs"]],
}

function _isBlank(value){
return value === "" || value === null || value === undefined || (typeof value === "number" && isNaN(value))
}

// A floor of zero is a real floor and prints "0". Only a floor with no
// counted games behind it (every game he played was an injury exit)
// prints a dash - there is nothing to take a minimum over.
function formatPositionFloor(value){
if(_isBlank(value)){ return "-" }
const n = Number(value)
// Number() drops a CSV's ".0" and keeps a split sack's "0.5".
return isNaN(n) ? String(value) : String(n)
}

// The line under each floor: "avg 31 ± 5". Yardage-sized numbers
// (mean of 10 or more) round to whole numbers; small counts like
// receptions or touchdowns keep one decimal so 0.3 and 0.7 stay apart.
// SD uses the mean's precision, and is dropped when there is only one
// game to measure it over.
function formatPositionFloorSpread(mean, sd){
if(_isBlank(mean) || isNaN(Number(mean))){ return "" }
const m = Number(mean)
const digits = Math.abs(m) >= 10 ? 0 : 1
const avg = `avg ${m.toFixed(digits)}`
if(_isBlank(sd) || isNaN(Number(sd))){ return avg }
return `${avg} \u00b1 ${Number(sd).toFixed(digits)}`
}

function positionFloorNote(row){
const notes = []
if(!_isBlank(row.excluded_weeks)){
notes.push(`Injured wk ${row.excluded_weeks} - not counted`)
}
if(!_isBlank(row.flagged_week)){
notes.push(`Left wk ${row.flagged_week} early - counted unless he's out next game`)
}
return notes.join("; ")
}

function filterPositionFloors(data, tab, game){
let rows = data.filter(row=>row.tab === tab)
if(game && game !== PLAYER_PROPS_ALL_GAMES){
rows = rows.filter(row=>row.game === game)
}
return rows
}

// Highest floor first; a blank floor sinks to the bottom whichever way.
// With no sort key the board stays in snap-count order.
function sortPositionFloors(rows, key){
const copy = rows.slice()
if(!key){
return copy.sort((a, b)=>Number(a.rank) - Number(b.rank))
}
return copy.sort((a, b)=>{
const av = _isBlank(a[key]) ? -Infinity : Number(a[key])
const bv = _isBlank(b[key]) ? -Infinity : Number(b[key])
if(bv !== av){ return bv - av }
return Number(a.rank) - Number(b.rank)
})
}

function positionFloorGames(data){
const seen = []
data.forEach(row=>{ if(row.game && !seen.includes(row.game)){ seen.push(row.game) } })
return seen.sort()
}

function buildPositionFloorTable(rows, tab, sortKey, id, onSort){
const el = document.getElementById(id)
if(!el){ return }
if(!rows.length){
el.innerHTML = "No players to show for this selection."
return
}
const stats = POSITION_FLOOR_STATS[tab] || []
let html = "<table><tr><th>#</th><th>Player</th><th>Pos</th><th>Team</th><th>Next Game</th><th>Games</th>"
stats.forEach(([key, label])=>{
const marker = key === sortKey ? " &#9660;" : ""
html += `<th data-floor-sort="${key}" style="cursor:pointer" title="Sort by ${label} floor">${label}${marker}</th>`
})
html += "<th>Note</th></tr>"
rows.forEach(row=>{
const counted = Number(row.games_counted)
const played = Number(row.games_played)
const games = counted === played ? `${played}` : `${counted} of ${played}`
html += `<tr><td>${row.rank}</td><td>${row.player_name}</td><td>${row.position}</td><td>${row.team}</td>`
html += `<td>${_isBlank(row.game) ? "bye" : row.game}</td><td>${games}</td>`
stats.forEach(([key])=>{
const spread = formatPositionFloorSpread(row[`${key}_mean`], row[`${key}_sd`])
html += `<td><b>${formatPositionFloor(row[key])}</b>`
if(spread){
html += `<br><span style="font-size:0.8em;opacity:0.7;white-space:nowrap">${spread}</span>`
}
html += "</td>"
})
html += `<td style="font-size:0.85em">${positionFloorNote(row)}</td></tr>`
})
html += "</table>"
el.innerHTML = html
if(onSort){
el.querySelectorAll("th[data-floor-sort]").forEach(th=>{
th.onclick = ()=> onSort(th.getAttribute("data-floor-sort"))
})
}
}

function buildPositionFloorsSection(data, tabsId, gameSelectId, tableId){
const tabsEl = document.getElementById(tabsId)
const gameSelect = document.getElementById(gameSelectId)
const state = { tab: POSITION_FLOOR_TABS[0], sortKey: null }
const render = ()=>{
const rows = sortPositionFloors(
filterPositionFloors(data, state.tab, gameSelect ? gameSelect.value : PLAYER_PROPS_ALL_GAMES),
state.sortKey,
)
buildPositionFloorTable(rows, state.tab, state.sortKey, tableId, key=>{
// Clicking the sorted column again goes back to snap order.
state.sortKey = state.sortKey === key ? null : key
render()
})
if(tabsEl){
tabsEl.querySelectorAll("button").forEach(b=>{
b.classList.toggle("active", b.getAttribute("data-floor-tab") === state.tab)
})
}
}
if(tabsEl){
tabsEl.innerHTML = POSITION_FLOOR_TABS
.map(t=>`<button class="tabButton" data-floor-tab="${t}">${t}</button>`).join("")
tabsEl.querySelectorAll("button").forEach(b=>{
b.onclick = ()=>{
state.tab = b.getAttribute("data-floor-tab")
state.sortKey = null
render()
}
})
}
if(gameSelect){
const games = positionFloorGames(data)
gameSelect.innerHTML = [PLAYER_PROPS_ALL_GAMES].concat(games)
.map(g=>`<option value="${g}">${g}</option>`).join("")
gameSelect.style.display = games.length ? "" : "none"
gameSelect.onchange = render
}
render()
}

if (typeof module !== "undefined" && module.exports) {
module.exports = {
formatPlayerPropsValue,
playerPropsGames,
filterPlayerProps,
PLAYER_PROPS_ALL_GAMES,
PLAYER_PROPS_COLUMNS,
PLAYER_PROPS_COLUMN_LABELS,
formatPositionFloor,
formatPositionFloorSpread,
positionFloorNote,
filterPositionFloors,
sortPositionFloors,
positionFloorGames,
POSITION_FLOOR_TABS,
POSITION_FLOOR_STATS,
}
}
