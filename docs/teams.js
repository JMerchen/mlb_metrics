// Team nicknames for display ("Yankees", "Cowboys") - the data files use
// short codes, which read like jargon on their own. MLB and NFL are kept
// apart because codes overlap (ATL, BAL, CIN, ... are different teams).
// Codes cover every spelling the data uses (AZ/ARI, ATH/OAK, CWS/CHW,
// WSH/WAS). An unknown code falls back to itself rather than disappearing.

const MLB_TEAM_NAMES = {
AZ: "Diamondbacks", ARI: "Diamondbacks", ATL: "Braves", BAL: "Orioles", BOS: "Red Sox",
CHC: "Cubs", CWS: "White Sox", CHW: "White Sox", CIN: "Reds", CLE: "Guardians", COL: "Rockies",
DET: "Tigers", HOU: "Astros", KC: "Royals", LAA: "Angels", LAD: "Dodgers", MIA: "Marlins",
MIL: "Brewers", MIN: "Twins", NYM: "Mets", NYY: "Yankees", ATH: "Athletics", OAK: "Athletics",
PHI: "Phillies", PIT: "Pirates", SD: "Padres", SF: "Giants", SEA: "Mariners", STL: "Cardinals",
TB: "Rays", TEX: "Rangers", TOR: "Blue Jays", WSH: "Nationals", WAS: "Nationals",
}

const NFL_TEAM_NAMES = {
ARI: "Cardinals", ATL: "Falcons", BAL: "Ravens", BUF: "Bills", CAR: "Panthers", CHI: "Bears",
CIN: "Bengals", CLE: "Browns", DAL: "Cowboys", DEN: "Broncos", DET: "Lions", GB: "Packers",
HOU: "Texans", IND: "Colts", JAX: "Jaguars", KC: "Chiefs", LA: "Rams", LAR: "Rams",
LAC: "Chargers", LV: "Raiders", MIA: "Dolphins", MIN: "Vikings", NE: "Patriots", NO: "Saints",
NYG: "Giants", NYJ: "Jets", PHI: "Eagles", PIT: "Steelers", SEA: "Seahawks", SF: "49ers",
TB: "Buccaneers", TEN: "Titans", WAS: "Commanders", WSH: "Commanders",
}

function teamName(code, sport){
const names = sport === "nfl" ? NFL_TEAM_NAMES : MLB_TEAM_NAMES
return names[code] || code || ""
}

if (typeof module !== "undefined" && module.exports) {
module.exports = { teamName, MLB_TEAM_NAMES, NFL_TEAM_NAMES }
}
