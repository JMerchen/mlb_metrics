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



// Season-long floors (nfl_prop_streaks.py) - a HISTORY board, not a
// prediction one: the most each player has produced in every game so
// far ("4+ every week" once his receptions go 5, 6, 4), kept where that
// floor is rare among his position peers. Nothing here projects
// anything; every number already happened.
const PLAYER_STREAK_ALL_STATS = "All stats"
// In the "All stats" view, the best few per stat - enough to show what
// every stat looks like without one deep stat (tackles, among 170+ DBs)
// burying the rest. Picking a stat shows every qualifying row for it.
const PLAYER_STREAK_ALL_STATS_PER_STAT = 3

const PLAYER_STREAK_COLUMN_LABELS = {
player_name: "Player", position: "Pos", team: "Team", game: "Next Game",
stat: "Stat", floor: "Every Week", game_log: "Game Log",
season_avg: "Avg", rarity: "How Rare",
}
const PLAYER_STREAK_COLUMNS = [
"player_name", "position", "team", "game", "stat", "floor",
"game_log", "season_avg", "rarity",
]

// Plural label for the peer pool a floor was compared against. Most
// pools are position groups ("WRs", "DBs"); kickers are pooled by exact
// position because the SPEC group also holds punters and long snappers.
const PLAYER_STREAK_POOL_LABELS = { K: "kickers" }

function playerStreakPoolLabel(group){
return PLAYER_STREAK_POOL_LABELS[group] || `${group}s`
}

function formatPlayerStreakValue(column, value, row){
if(column === "floor"){
const n = Number(value)
if(value === "" || value === null || value === undefined || isNaN(n)){ return value }
// Number() already drops a CSV's trailing ".0" (4.0 -> "4") and keeps
// a split sack as "0.5", so no special-casing is needed.
return `${n}+`
}
if(column === "season_avg"){
const n = Number(value)
return isNaN(n) || value === "" ? value : n.toFixed(1)
}
if(column === "rarity"){
// Said the way a person would say it - "2 of 32 kickers" - rather
// than as a fraction, because the counts ARE the information.
if(!row){ return value }
return `${row.players_at_or_above} of ${row.pool_size} ${playerStreakPoolLabel(row.pool_group)}`
}
if(column === "game" && (value === "" || value === null || value === undefined)){
return "bye"
}
return value
}

function playerStreakStats(data){
const seen = []
data.forEach(row=>{ if(row.stat && !seen.includes(row.stat)){ seen.push(row.stat) } })
return seen
}

function filterPlayerStreaks(data, stat, game){
let rows = data
if(game && game !== PLAYER_PROPS_ALL_GAMES){
rows = rows.filter(row=>row.game === game)
}
if(stat && stat !== PLAYER_STREAK_ALL_STATS){
return rows.filter(row=>row.stat === stat)
}
const counts = {}
return rows.filter(row=>{
counts[row.stat] = (counts[row.stat] || 0) + 1
return counts[row.stat] <= PLAYER_STREAK_ALL_STATS_PER_STAT
})
}

function buildPlayerStreakTable(data, id){
const el = document.getElementById(id)
if(!el){ return }
if(!data.length){
el.innerHTML = "No season-long floors to show for this selection."
return
}
let html = "<table><tr>"
PLAYER_STREAK_COLUMNS.forEach(c=>{ html += `<th>${PLAYER_STREAK_COLUMN_LABELS[c] || c}</th>` })
html += "</tr>"
data.forEach(row=>{
html += "<tr>"
PLAYER_STREAK_COLUMNS.forEach(c=>{ html += `<td>${formatPlayerStreakValue(c, row[c], row)}</td>` })
html += "</tr>"
})
html += "</table>"
el.innerHTML = html
}

function buildPlayerStreakSection(data, statSelectId, gameSelectId, tableId){
const statSelect = document.getElementById(statSelectId)
const gameSelect = document.getElementById(gameSelectId)
const render = ()=> buildPlayerStreakTable(
filterPlayerStreaks(
data,
statSelect ? statSelect.value : PLAYER_STREAK_ALL_STATS,
gameSelect ? gameSelect.value : PLAYER_PROPS_ALL_GAMES,
),
tableId,
)
if(statSelect){
statSelect.innerHTML = [PLAYER_STREAK_ALL_STATS].concat(playerStreakStats(data))
.map(s=>`<option value="${s}">${s}</option>`).join("")
statSelect.onchange = render
}
if(gameSelect){
const games = playerPropsGames(data)
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
formatPlayerStreakValue,
playerStreakStats,
filterPlayerStreaks,
playerStreakPoolLabel,
PLAYER_STREAK_ALL_STATS,
PLAYER_STREAK_ALL_STATS_PER_STAT,
PLAYER_STREAK_COLUMNS,
PLAYER_STREAK_COLUMN_LABELS,
}
}
