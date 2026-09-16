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
opponent_allowed_rate: "Opp Allows", league_rate: "League Avg",
ratio: "Matchup Edge", games: "Games",
}
const PLAYER_PROPS_COLUMNS = [
"player_name", "team", "opponent", "position", "category", "direction",
"player_rate", "opponent_allowed_rate", "league_rate", "ratio", "games",
]

// player_rate/opponent_allowed_rate/league_rate are all real per-game
// counting-stat rates (receptions/yards/sacks) - one decimal place is
// real, meaningful precision without a false extra digit of certainty.
// ratio is shown as a real relative edge (+/-% vs a neutral 1.0 = league
// average), not a raw percentage of the opponent's own rate - "+20%"
// reads directly as "this opponent allows 20% more than average," the
// real number this whole feature exists to surface, without the reader
// having to do that subtraction themselves.
function formatPlayerPropsValue(column, value){
if(["player_rate", "opponent_allowed_rate", "league_rate"].includes(column)){
const n = Number(value)
return isNaN(n) ? value : n.toFixed(1)
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
