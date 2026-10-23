// Plain-language rendering for the Today page (docs/today.js) and the
// full-site artifact's Assistant tab (scripts/templates/bts_assistant_template.html,
// which inlines this file and teams.js). One copy on purpose: the Assistant
// tab used to keep its own duplicate of this code, and the two drifted
// apart (2026-10-10). Pure functions returning HTML strings, so node tests
// can check the wording.
//
// The wording rule: say it the way you would to a friend. Whole-number
// percentages, team nicknames, "we say / Vegas says", records as "right 12
// of 20" - no Brier scores, confidence intervals or model version labels
// on these pages. The detail is still on the MLB/NFL pages and Methodology.

function srEsc(s){ return String(s === undefined || s === null ? "" : s).replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c])) }

function srNum(x){
return x === undefined || x === null || x === "" || Number.isNaN(Number(x)) ? null : Number(x)
}

// 0.589 -> "59%"
function srPct(x){
const n = srNum(x)
return n === null ? "-" : Math.round(n * 100) + "%"
}

function srPlural(n, word){ return `${n} ${word}${n === 1 ? "" : "s"}` }

function srTeam(code, sport){
return typeof teamName === "function" ? teamName(code, sport) : (code || "")
}

// Beat the Streak is a regular-season contest; once it's over the
// pipeline marks the summary (season_over / season_end).
function btsSeasonOver(summary){
return String((summary || {}).season_over).toLowerCase() === "true"
}

// "2026-09-27" -> "Sep 27"
function srShortDate(iso){
const d = new Date(`${iso}T12:00:00`)
return Number.isNaN(d.getTime()) ? String(iso || "") : d.toLocaleDateString("en-US", { month: "short", day: "numeric" })
}

function btsSeasonOverHtml(summary){
const ended = summary && summary.season_end ? ` ended ${srEsc(srShortDate(summary.season_end))}` : " is over"
return `<div class="pick"><span class="name">The regular season${ended}. Beat the Streak picks return on Opening Day.</span></div>`
}

// Top tiles: the Beat the Streak record and today's best hitter (or,
// after the regular season, that the season is over).
function streakTilesHtml(summary, topApproach){
if(!summary || !Object.keys(summary).length){
return '<div class="empty-note">No picks tracked yet.</div>'
}
const over = btsSeasonOver(summary)
const current = srNum(summary.current_streak) || 0
const longest = srNum(summary.longest_streak) || 0
const days = srNum(summary.n_days_resolved) || 0
const tiles = [
{ label: "Hit streak", value: srPlural(current, "day"), sub: over ? "at the end of the season" : "in a row with a hit" },
{ label: "Best streak", value: srPlural(longest, "day"), sub: over ? "our longest this season" : "our longest so far" },
{ label: "Days with a hit", value: srPct(summary.day_survival_rate), sub: `over ${srPlural(days, "day")}` },
over
? { label: "Season", value: "Over", sub: summary.season_end ? `regular season ended ${srEsc(srShortDate(summary.season_end))}` : "see you on Opening Day" }
: topApproach
? { label: "Best hitter today", value: srEsc(String(topApproach.name).split(" ").pop()),
    sub: `${srEsc(srTeam(topApproach.team, "mlb"))}&nbsp;&middot; ${srPct(topApproach.Game_Hit_Probability)} to get a hit` }
: { label: "Best hitter today", value: "-", sub: "no games today" },
]
return tiles.map(t =>
`<div class="stat"><div class="btsLabel">${t.label}</div><div class="btsValue">${t.value}</div><div class="btsSub">${t.sub}</div></div>`
).join("")
}

// Today's Beat the Streak picks.
function btsPicksHtml(picks){
if(!picks || !picks.length){
return '<div class="pick"><span class="name">No picks yet today.</span></div>'
}
return picks.map(p => {
const strong = p.grade === "recommended"
return `<div class="pick">
<span class="name">${srEsc(p.name)}</span>
<span class="prob">${srPct(p.combined_probability)} to get a hit</span>
<span class="grade ${strong ? "btsRecommended" : "btsSpeculative"}">${strong ? "Strong pick" : "Weaker pick"}</span>
</div>`
}).join("")
}

// The one-line record under each sport's heading. While bet advice is off,
// it is how often our picks were right next to how often Vegas's were, on
// the same games; while it's on, it is the bet record.
function gamePicksRecordHtml(summary, current){
const s = summary || {}
if(String(s.betting_enabled).toLowerCase() === "false"){
const n = current ? srNum(current.model_vs_market_n) || 0 : 0
const paused = "Not suggesting bets right now."
if(!n){ return `No results yet this season. ${paused}` }
const ours = Math.round((srNum(current.model_pick_accuracy) || 0) * n)
const vegas = Math.round((srNum(current.market_pick_accuracy) || 0) * n)
return `Right on <b>${ours} of ${n}</b> games so far (Vegas: ${vegas} of ${n}). ${paused}`
}
const bets = srNum(s.n_bets_advised) || 0
if(!bets){ return "No bets settled yet." }
const won = srNum(s.bets_won) || 0
const units = srNum(s.total_profit_units) || 0
const direction = units >= 0 ? "up" : "down"
return `Our bets: won <b>${won} of ${bets}</b>, ${direction} ${Math.abs(units).toFixed(1)} units.`
}

// NFL: "Total points: we say 48 · Vegas 47.5". Empty when there is no
// predicted total (every MLB game, NFL games before totals existed).
function totalPointsText(g){
const ours = srNum(g.predicted_total)
if(ours === null){ return "" }
const vegas = srNum(g.market_total)
return `Total points: we say ${Math.round(ours)}${vegas !== null ? `&nbsp;&middot; Vegas ${vegas}` : ""}`
}

// 3 -> "3rd"
function srOrdinal(n){
const v = n % 100
const suffix = v >= 11 && v <= 13 ? "th" : ({1: "st", 2: "nd", 3: "rd"}[n % 10] || "th")
return `${n}${suffix}`
}

// NFL: each team's offense EPA per play gained and defense EPA per play
// allowed, with league rank (1st = best on both). Empty without ratings.
function teamEpaHtml(g, sport){
const sides = ["away", "home"].map(side => ({
team: srTeam(g[`${side}_team`], sport),
off: srNum(g[`${side}_off_epa`]), offRank: srNum(g[`${side}_off_rank`]),
def: srNum(g[`${side}_def_epa`]), defRank: srNum(g[`${side}_def_rank`]),
}))
if(sides.every(s => s.off === null && s.def === null)){ return "" }
const signed = value => {
const shown = Math.abs(value).toFixed(2)
return shown === "0.00" ? shown : `${value > 0 ? "+" : "−"}${shown}`
}
const cell = (value, rank) => value === null ? "-"
: `${signed(value)}${rank !== null ? ` <span class="epaRank">${srOrdinal(Math.round(rank))}</span>` : ""}`
return `<table class="epaTable">
<thead><tr><th>EPA per play</th><th>Offense</th><th>Defense</th></tr></thead>
<tbody>${sides.map(s => `<tr><td>${srEsc(s.team)}</td><td>${cell(s.off, s.offRank)}</td><td>${cell(s.def, s.defRank)}</td></tr>`).join("")}</tbody>
</table>`
}

// One game: "Rays at Yankees", our pick and our chance next to Vegas's.
function gamePickRowHtml(g, sport){
const homePicked = g.predicted_winner === g.home_team
const away = srEsc(srTeam(g.away_team, sport))
const home = srEsc(srTeam(g.home_team, sport))
const pick = srEsc(srTeam(g.predicted_winner, sport))
const vegas = srNum(g.market_predicted_winner_probability)
const units = srNum(g.bet_units) || 0
const total = totalPointsText(g)
const bet = units > 0
? `<span class="bet-badge">Bet ${srEsc(srTeam(g.bet_team, sport))}&nbsp;&middot; ${units.toFixed(1)} units</span>` : ""
return `<div class="game">
<div class="btsMatchup">
<span class="${homePicked ? "" : "winner"}">${away}</span> at
<span class="${homePicked ? "winner" : ""}">${home}</span>
</div>
<span class="prob">We say <b>${pick} ${srPct(g.predicted_probability)}</b>${vegas !== null ? `<span class="mkt">&nbsp;&middot; Vegas ${srPct(vegas)}</span>` : ""}${total ? `<span class="total">${total}</span>` : ""}</span>
${bet}
</div>`
}

function gamePicksListHtml(picks, sport){
if(!picks || !picks.length){
return '<div class="empty-note">No games scheduled yet.</div>'
}
return picks.map(g => gamePickRowHtml(g, sport)).join("")
}

if (typeof module !== "undefined" && module.exports) {
module.exports = {
srPct, streakTilesHtml, btsPicksHtml, gamePicksRecordHtml, gamePickRowHtml, gamePicksListHtml, totalPointsText,
btsSeasonOver, btsSeasonOverHtml,
teamEpaHtml,
}
}
