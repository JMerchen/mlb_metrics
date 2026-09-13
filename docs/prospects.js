// Prospects page bootstrap - separate from app.js since this page is
// standalone (see docs/prospects.html). Real rendering logic
// (loadCSV/prepareConsensusColumns/formatConsensusValue/buildConsensusTable)
// lives in docs/consensus_board.js (loaded via its own script tag
// before this file, see that file's own comment for why) - this file is
// purely the page's own DOM wiring, same "tested pure-logic file feeds
// an untested bootstrap file" split as docs/nfl_draft_assistant.js/
// docs/nfl.js.

function selectProspectsTab(tab){
document.querySelectorAll("#prospectsTabs .tabButton").forEach(btn=>{
btn.classList.toggle("active", btn.dataset.prospectsTab === tab)
})
document.getElementById("farmSystemSection").style.display = tab === "farmsystem" ? "" : "none"
document.getElementById("draftBoardSection").style.display = tab === "draftboard" ? "" : "none"
}

async function loadAll(){

try{
const prospects = await loadCSV("./data/prospect_rankings.csv")
buildConsensusTable(prospects, "prospectRankingsTable")
}catch(e){
console.log("no prospect_rankings.csv yet", e)
buildConsensusTable([], "prospectRankingsTable")
}

try{
const draftBoard = await loadCSV("./data/mlb_draft_board.csv")
buildConsensusTable(draftBoard, "mlbDraftBoardTable")
}catch(e){
console.log("no mlb_draft_board.csv yet", e)
buildConsensusTable([], "mlbDraftBoardTable")
}

}

loadAll()
