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
return gamePicksBaseRecordHtml(summary, current) + totalLeansRecordHtml(summary)
}

function gamePicksBaseRecordHtml(summary, current){
const s = summary || {}
if(String(s.betting_enabled).toLowerCase() === "false"){
const n = current ? srNum(current.model_vs_market_n) || 0 : 0
const paused = "Not suggesting moneyline bets right now."
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

// NFL: " Over/under leans: won 5 of 8." Empty without any settled lean.
function totalLeansRecordHtml(summary){
const s = summary || {}
const won = srNum(s.total_leans_won) || 0, lost = srNum(s.total_leans_lost) || 0
return won + lost ? ` Over/under leans: won <b>${won} of ${won + lost}</b>.` : ""
}

// NFL: "Total points: we say 39–57 · Vegas 47.5" - our total as a range
// (config.NFL_TOTAL_BAND_HALF_WIDTH either side; the real total lands in
// it about half the time), or one number for a pick logged without one.
// Empty when there is no predicted total (every MLB game, NFL games before
// totals existed).
function totalPointsText(g){
const ours = srNum(g.predicted_total)
if(ours === null){ return "" }
const low = srNum(g.predicted_total_low), high = srNum(g.predicted_total_high)
const range = low !== null && high !== null ? `${low}–${high}` : `${Math.round(ours)}`
const vegas = srNum(g.market_total)
return `Total points: we say ${range}${vegas !== null ? `&nbsp;&middot; Vegas ${vegas}` : ""}`
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

// Open-Meteo weather codes (WMO) that mean snow.
const SR_SNOW_CODES = new Set([71, 73, 75, 77, 85, 86])

// NFL: the game's weather from its pick (nfl_weather.py) - {indoor, temp,
// wind, rain, code, snow, icon, label} - or null when there's no forecast
// yet (it's pulled the day before, or the morning of a Monday or Thursday
// game) or the sport has none.
function srWeather(g){
if(srNum(g.weather_indoor) === 1){ return {indoor: true, icon: "🏟️", label: "Indoors"} }
const temp = srNum(g.weather_temp), wind = srNum(g.weather_wind), rain = srNum(g.weather_rain), code = srNum(g.weather_code)
if(temp === null && wind === null && rain === null){ return null }
const snow = code !== null && SR_SNOW_CODES.has(code)
const icon = snow ? "🌨️"
: code !== null && code >= 95 ? "⛈️"
: (rain || 0) >= 0.5 ? "🌧️"
: (rain || 0) >= 0.25 ? "🌦️"
: (wind || 0) >= 18 ? "💨"
: code === 45 || code === 48 ? "🌫️"
: code === 0 || code === 1 ? "☀️"
: code === 2 ? "⛅"
: code === 3 ? "☁️" : "🌤️"
const parts = []
if(temp !== null){ parts.push(`${Math.round(temp)}°F`) }
if(wind !== null){ parts.push(`wind ${Math.round(wind)} mph`) }
if(snow){ parts.push("snow") }else if(rain !== null){ parts.push(`${Math.round(rain * 100)}% chance of rain`) }
return {indoor: false, temp, wind, rain, code, snow, icon, label: parts.join(", ")}
}

// Each team's chance as one bar, away on the left and home on the right,
// our pick's side filled in, with a tick where Vegas has it.
function pickBarHtml(g, sport){
const p = srNum(g.predicted_probability)
if(p === null){ return "" }
const homePicked = g.predicted_winner === g.home_team
const home = homePicked ? p : 1 - p
const vegasHome = srNum(g.market_home_win_probability)
const away = srTeam(g.away_team, sport), homeName = srTeam(g.home_team, sport)
const tick = vegasHome === null ? ""
: `<span class="pickBarVegas" style="left:${((1 - vegasHome) * 100).toFixed(1)}%" title="Vegas: ${srEsc(homeName)} ${srPct(vegasHome)}"></span>`
return `<div class="pickBar" role="img" aria-label="${srEsc(away)} ${srPct(1 - home)}, ${srEsc(homeName)} ${srPct(home)}">
<div class="pickBarLabels"><span>${srEsc(away)} <b>${srPct(1 - home)}</b></span><span><b>${srPct(home)}</b> ${srEsc(homeName)}</span></div>
<div class="pickBarTrack"><span class="pickBarAway${homePicked ? "" : " picked"}" style="width:${((1 - home) * 100).toFixed(1)}%"></span><span class="pickBarHome${homePicked ? " picked" : ""}"></span>${tick}</div>
${vegasHome === null ? "" : `<div class="pickBarKey">The tick is Vegas: ${srEsc(homePicked ? homeName : away)} ${srPct(homePicked ? vegasHome : 1 - vegasHome)}</div>`}
</div>`
}

// NFL: our over/under lean when our total is far enough from Vegas's
// (config.NFL_TOTAL_LEAN_MIN_GAP) - {side, line, gap, result} or null.
function totalLean(g){
const side = g.total_lean === "over" || g.total_lean === "under" ? g.total_lean : null
const line = srNum(g.total_lean_line)
if(!side || line === null){ return null }
const ours = srNum(g.predicted_total)
return {side, line, gap: ours === null ? null : Math.abs(ours - line), result: g.total_lean_result || ""}
}

// "Lean OVER 37.5", and once played "won" / "lost" / "push".
function totalLeanBadgeHtml(g){
const lean = totalLean(g)
if(!lean){ return "" }
const done = {win: " · won", loss: " · lost", push: " · push"}[lean.result] || ""
return `<span class="leanBadge lean-${lean.side}${lean.result ? ` lean-${lean.result}` : ""}">Lean ${lean.side.toUpperCase()} ${lean.line}${done}</span>`
}

// A played game outside our range: "Finished 61 - 5 over our range".
function totalRangeNote(g){
const actual = srNum(g.actual_total)
if(actual === null){ return "" }
const miss = srNum(g.total_range_miss)
if(g.total_range_result === "over" || g.total_range_result === "under"){
return `Finished ${actual} - ${miss} ${g.total_range_result} our range`
}
return g.total_range_result === "in" ? `Finished ${actual} - inside our range` : ""
}

// One game: a card with the matchup and our pick ("Rays at Yankees", "We
// say Yankees 59%"), a color bar down its left edge (NFL: before or after
// the final injury report), and a click to open the rest: each team's
// chance as a bar, the weather, the predicted total and each team's EPA.
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
// NFL picks carry whether they came before or after the final injury
// report (the edge's color).
const report = sport === "nfl" ? (g.injury_report === "after" ? "after" : "before") : ""
const reportNote = report
? `<span class="reportNote"><b>${report === "after" ? "After" : "Before"} the final injury report</b></span>`
: ""
const weather = sport === "nfl" ? srWeather(g) : null
const weatherLine = sport !== "nfl" ? ""
: weather
? `<div class="gameWeather"><span class="wxIcon" aria-hidden="true">${weather.icon}</span><span>${srEsc(weather.label)}</span></div>`
: `<div class="gameWeather"><span class="wxIcon" aria-hidden="true">🗓️</span><span>Forecast comes the day before the game</span></div>`
const epa = sport === "nfl" ? teamEpaHtml(g, sport) : ""
return `<details class="game${report ? ` report-${report}` : ""}">
<summary>
<div class="btsMatchup">
<span class="${homePicked ? "" : "winner"}">${away}</span> at
<span class="${homePicked ? "winner" : ""}">${home}</span>
</div>
<span class="prob">We say <b>${pick} ${srPct(g.predicted_probability)}</b>${vegas !== null ? `<span class="mkt">&nbsp;&middot; Vegas ${srPct(vegas)}</span>` : ""}</span>
${weather && !weather.indoor ? `<span class="wxIcon wxPeek" title="${srEsc(weather.label)}" aria-hidden="true">${weather.icon}</span>` : ""}
${bet}
${sport === "nfl" ? totalLeanBadgeHtml(g) : ""}
<span class="gameMore" aria-hidden="true"></span>
</summary>
<div class="gameDetail">
${pickBarHtml(g, sport)}
${weatherLine}
${total ? `<div class="total">${total}</div>` : ""}
${total && srNum(g.predicted_total_low) !== null ? `<div class="totalNote">About half of games finish inside our range.</div>` : ""}
${sport === "nfl" && totalLean(g) ? `<div class="totalNote leanNote">We lean ${totalLean(g).side} Vegas's ${totalLean(g).line}: our total is ${Math.round(totalLean(g).gap)} points ${totalLean(g).side === "over" ? "higher" : "lower"}.</div>` : ""}
${sport === "nfl" && totalRangeNote(g) ? `<div class="totalNote">${srEsc(totalRangeNote(g))}</div>` : ""}
${epa}
${reportNote}
</div>
</details>`
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
srWeather, pickBarHtml, totalLean, totalLeanBadgeHtml, totalRangeNote,
btsSeasonOver, btsSeasonOverHtml,
teamEpaHtml,
}
}
