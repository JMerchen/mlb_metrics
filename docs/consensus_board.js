// Shared, real rendering helpers for the 3 consensus rankings boards
// (consensus_rankings.build_consensus_ranking's own output shape) -
// used by both docs/prospects.js (MLB prospects + MLB draft board) and
// docs/nfl.js (NFL draft board). A real, small, testable file loaded via
// its own <script> tag BEFORE either page's own bootstrap script, same
// "tested pure-logic file feeds an untested bootstrap file's DOM wiring
// via plain globals" split docs/nfl_draft_assistant.js/docs/nfl.js
// already establish - this file deliberately has NO top-level
// side-effecting call (no loadAll()-equivalent), so it stays safely
// require()-able from node --test.

const CONSENSUS_PRIORITY_COLUMNS = ["consensus_rank", "player_name", "consensus_score", "sources_ranked_by"]
const CONSENSUS_HIDDEN_COLUMNS = ["source_ranks", "source_url"]

// Friendly real header labels for the 4 synthetic columns this engine
// itself creates (consensus_rankings.build_consensus_ranking's own
// output) - every OTHER real column (Pos/Team/College/...) already
// carries whatever real header the source's own page used, so it's
// shown as-is rather than relabeled, same "don't reformat a real value
// you didn't create" reasoning formatConsensusValue's own docstring uses.
const CONSENSUS_COLUMN_LABELS = {
consensus_rank: "Rank",
player_name: "Player",
consensus_score: "Consensus Score",
sources_ranked_by: "Sources",
}

function formatColumnLabel(column){
return CONSENSUS_COLUMN_LABELS[column] || column
}

async function loadCSV(path){
const response = await fetch(`${path}?t=${Date.now()}`, {cache:"no-store"})
if(!response.ok){ throw new Error(`Failed to load ${path}: ${response.status}`) }
const text = await response.text()
return Papa.parse(text, {header:true, skipEmptyLines:true}).data
}

// Real, stable column order for a consensus board's own real output
// shape: the 4 core columns first (in a fixed, readable order), then
// whatever other real columns happen to exist (e.g. Pos/Team/College,
// which vary by which source's row was picked as canonical - see
// consensus_rankings.build_consensus_ranking's own docstring) in
// whatever order the data itself carries them, excluding the raw
// per-source dict/url (too noisy for a table cell - source_ranks'
// real per-source breakdown belongs in a tooltip/detail view, not
// inline, a real future enhancement once real data exists to design
// against).
function prepareConsensusColumns(data){
if(!data.length){ return [] }
const allColumns = Object.keys(data[0])
const extra = allColumns.filter(c => !CONSENSUS_PRIORITY_COLUMNS.includes(c) && !CONSENSUS_HIDDEN_COLUMNS.includes(c))
return CONSENSUS_PRIORITY_COLUMNS.filter(c => allColumns.includes(c)).concat(extra)
}

// consensus_score is a real mean-of-ranks float (e.g. 12.333333...) -
// round to 1 decimal for real display readability; every other column
// passes through unchanged (a real player_name/position/rank shouldn't
// be reformatted).
function formatConsensusValue(column, value){
if(column === "consensus_score"){
const n = Number(value)
return isNaN(n) ? value : n.toFixed(1)
}
return value
}

function buildConsensusTable(data, id, emptyMessage){
const el = document.getElementById(id)
if(!data.length){ el.innerHTML = emptyMessage || "No consensus data yet - this board updates weekly."; return }
const columns = prepareConsensusColumns(data)
let html = "<table><tr>"
columns.forEach(c=>{ html += `<th>${formatColumnLabel(c)}</th>` })
html += "</tr>"
data.forEach(row=>{
html += "<tr>"
columns.forEach(c=>{ html += `<td>${formatConsensusValue(c, row[c])}</td>` })
html += "</tr>"
})
html += "</table>"
el.innerHTML = html
}

if (typeof module !== "undefined" && module.exports) {
module.exports = {
prepareConsensusColumns,
formatConsensusValue,
formatColumnLabel,
}
}
