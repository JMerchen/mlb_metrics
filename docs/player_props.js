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
player_name: "Player", team: "Team", opponent: "Opp", position: "Pos",
category: "Category", direction: "Bet", player_rate: "Own Avg",
projection: "Projected", opponent_allowed_rate: "Opp Allows",
league_rate: "League Avg", ratio: "Opp vs Avg", games: "Games",
}
const PLAYER_PROPS_COLUMNS = [
"player_name", "team", "opponent", "position", "category", "direction",
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

if (typeof module !== "undefined" && module.exports) {
module.exports = {
formatPlayerPropsValue,
PLAYER_PROPS_COLUMNS,
PLAYER_PROPS_COLUMN_LABELS,
}
}
