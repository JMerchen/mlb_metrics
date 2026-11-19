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
// The model making picks now, or every pick on record while that one
// has no finished games yet (just after a model change).
const row = current && srNum(current.model_vs_market_n) ? current : s
const n = srNum(row.model_vs_market_n) || 0
const paused = "Not suggesting moneyline bets right now."
if(!n){ return `No results yet this season. ${paused}` }
const ours = Math.round((srNum(row.model_pick_accuracy) || 0) * n)
const vegas = Math.round((srNum(row.market_pick_accuracy) || 0) * n)
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
return {side, line, gap: ours === null ? null : Math.abs(ours - line), result: g.total_lean_result || "",
close: srNum(g.total_lean_close), clv: srNum(g.total_lean_clv), weather: srNum(g.total_lean_weather) || 0}
}

// Why we lean: "We lean under Vegas's 46.5: our total is 6 points lower."
// When rain is in the forecast its part is left out of our total first
// (total_lean_weather: the line already prices a forecast, and the lean was
// tested without knowing the rain), and the sentence says so.
function totalLeanReason(g){
const lean = totalLean(g)
if(!lean){ return "" }
const ours = srNum(g.predicted_total)
const rain = lean.weather > 0 ? lean.weather : 0
const diff = ours === null ? null : ours + rain - lean.line
const lead = `We lean ${lean.side} Vegas's ${lean.line}`
if(diff === null || Math.round(Math.abs(diff)) === 0){ return `${lead}.` }
const gap = Math.round(Math.abs(diff))
return `${lead}: our total is ${gap} point${gap === 1 ? "" : "s"} ${diff > 0 ? "higher" : "lower"}${rain >= 0.5 ? ", leaving the rain in the forecast out of it" : ""}.`
}

// ---- This week's bets ----
// Only what has beaten the closing line in testing gets recommended: for
// the NFL that is the over/under lean (see config.NFL_TOTAL_LEAN_BACKTEST's
// audit and scripts/research_nfl_bets.py for everything that didn't).

// The worst number still worth betting: our total (rain left out) less the
// rule's gap for an over, plus it for an under, to the half point.
function betWorstLine(g, minGap){
const lean = totalLean(g)
const ours = srNum(g.predicted_total)
if(!lean || ours === null){ return null }
const total = ours + (lean.weather > 0 ? lean.weather : 0)
const raw = lean.side === "over" ? total - minGap : total + minGap
return lean.side === "over" ? Math.floor(raw * 2) / 2 : Math.ceil(raw * 2) / 2
}

// The price a bet stops being worth taking at, from how often bets like it
// have won in testing, shaded toward break-even (half the tested edge over
// 52.4%): -110 odds need 52.4%, -120 need 54.5%, -130 need 56.5%.
function betPriceFloor(wonRate){
const p = 0.524 + (Number(wonRate) - 0.524) / 2
if(!Number.isFinite(p) || p <= 0.524){ return null }
return -Math.round(100 * p / (1 - p) / 5) * 5
}

// The bets panel: this week's recommended bets (games not yet played with
// a spread bet or an over/under lean), any waiting on the injury report,
// and the record behind each kind.
function betsPanelHtml(picks, summary, options){
const s = summary || {}
// options.single: the panel is for one game (the Picks page's matchup
// view), so "nothing to bet" is about that game, not the week.
const single = !!(options && options.single)
const minGap = srNum(s.total_lean_min_gap) || 5
const sideGap = srNum(s.spread_bet_min_gap) || 3.5
const rate = srNum(s.total_lean_backtest_won_rate), tested = srNum(s.total_lean_backtest_games) || 0
const sideRate = srNum(s.spread_bet_backtest_won_rate), sideTested = srNum(s.spread_bet_backtest_games) || 0
const upcoming = (picks || []).filter(g => g.status === "pending")
const sides = upcoming.filter(g => spreadBet(g, "nfl"))
const bets = upcoming.filter(g => totalLean(g))
const held = upcoming.filter(g => (!totalLean(g) && g.total_lean_held) || (!spreadBet(g, "nfl") && g.spread_bet_held))
const floor = betPriceFloor(rate), sideFloor = betPriceFloor(sideRate)
const name = code => typeof teamName === "function" ? teamName(code, "nfl") : code
const game = g => `<span class="betGame">${srEsc(name(g.away_team))} at ${srEsc(name(g.home_team))}</span>`
const sideRows = sides.map(g => {
const bet = spreadBet(g, "nfl"), worst = spreadBetWorstLine(g, sideGap), alt = spreadBetMoneylineNote(g, s)
return `<div class="betRow">
<div class="betMain"><b>${srEsc(bet.team)} ${bet.number}</b> ${game(g)}</div>
<div class="betWhy">${srEsc(spreadBetReason(g, "nfl"))}</div>
<div class="betTerms">Stake 1 unit.${worst ? ` Worth taking at ${worst} or better` : ""}${sideFloor ? `${worst ? "," : ""} at ${sideFloor} or better.` : "."}</div>
${alt ? `<div class="betTerms betAlt">${srEsc(alt)}</div>` : ""}
</div>`
}).join("")
const rows = bets.map(g => {
const lean = totalLean(g), worst = betWorstLine(g, minGap)
return `<div class="betRow">
<div class="betMain"><b>${lean.side === "over" ? "Over" : "Under"} ${lean.line}</b> ${game(g)}</div>
<div class="betWhy">${srEsc(totalLeanReason(g))}</div>
<div class="betTerms">Stake 1 unit.${worst !== null ? ` Worth taking at ${lean.side === "over" ? `${worst} or lower` : `${worst} or higher`}` : ""}${floor ? `${worst !== null ? "," : ""} at ${floor} or better.` : "."}</div>
</div>`
}).join("")
const waiting = !held.length ? ""
: single ? `<div class="betHeld">A bet here is waiting on the injury report: ${srEsc(held[0].spread_bet_held || held[0].total_lean_held)}.</div>`
: `<div class="betHeld">Waiting on the injury report: ${held.map(g => `${srEsc(name(g.away_team))} at ${srEsc(name(g.home_team))}`).join("; ")}.</div>`
const seasonText = (w, l, units) => w + l ? ` This season: ${w}–${l}${units !== null ? `, ${units > 0 ? "up" : units < 0 ? "down" : "even at"} ${Math.abs(units).toFixed(2)} units` : ""}.` : ""
const closeText = (n, mean) => n ? ` The line has closed ${Math.abs(mean || 0).toFixed(1)} points ${(mean || 0) >= 0 ? "our way" : "against us"} on average after we took it (${n} bet${n === 1 ? "" : "s"}).` : ""
const upDown = v => `${srNum(v) > 0 ? "up" : "down"} ${Math.abs(srNum(v) || 0)} units`
const season = seasonText(srNum(s.total_leans_won) || 0, srNum(s.total_leans_lost) || 0, srNum(s.total_lean_units))
const clv = closeText(srNum(s.total_lean_clv_n) || 0, srNum(s.total_lean_clv_mean))
const record = rate !== null && tested ? `<div class="betRecord"><b>Over/unders.</b> Bets like these won <b>${Math.round(1000 * rate) / 10}% of ${tested}</b> in testing (2019–2025, at the real odds: ${upDown(s.total_lean_backtest_units)}, a winning record in ${s.total_lean_backtest_winning_seasons} of ${s.total_lean_backtest_seasons} seasons; 52.4% breaks even).${season}${clv}</div>` : ""
const recentWon = srNum(s.spread_bet_recent_won), recentLost = srNum(s.spread_bet_recent_lost)
const recent = recentWon !== null && recentLost !== null ? ` The caution: ${s.spread_bet_recent_seasons} went only ${recentWon}–${recentLost}, so treat it as a thinner edge than the over/unders.` : ""
const sideSeason = seasonText(srNum(s.spread_bets_won) || 0, srNum(s.spread_bets_lost) || 0, srNum(s.spread_bet_units))
const sideClv = closeText(srNum(s.spread_bet_clv_n) || 0, srNum(s.spread_bet_clv_mean))
const sideRecord = sideRate !== null && sideTested ? `<div class="betRecord"><b>Spreads.</b> Sides ${sideGap} or more points from the line won <b>${Math.round(1000 * sideRate) / 10}% of ${sideTested}</b> against the closing spread in testing (${s.spread_bet_backtest_first_season}–${s.spread_bet_backtest_last_season}: ${upDown(s.spread_bet_backtest_units)}, a winning record in ${s.spread_bet_backtest_winning_seasons} of ${s.spread_bet_backtest_seasons} seasons).${recent}${sideSeason}${sideClv}</div>` : ""
const none = bets.length || sides.length ? ""
: single ? `<div class="betNone">${held.length ? "" : upcoming.length ? "No bet on this game: neither its spread nor its total is far enough from ours." : "This game has been played."}</div>`
: `<div class="betNone">${upcoming.length ? "No bets yet this week: no game's spread or total is far enough from ours." : "No games left to bet this week."}</div>`
return `<div class="betsPanel">
${sideRows}${rows}${none}${waiting}${sideRecord}${record}
<div class="betFine">We only recommend what has beaten the closing line in testing: spread sides where our margin sits ${sideGap} or more points from Vegas's, and over/unders where our total sits ${minGap} or more from theirs. Take them early: when our number and the opening line disagreed, the line moved toward us far more often than away. Our game winners alone have not beaten the closing line, so they are picks, not bets.</div>
</div>`
}

// ---- The Picks page's matchup view ----
// This week's games for the matchup dropdown (the latest week in the
// picks), in the order they are played: {games: [{id, label, pending,
// bet}], selected}. A label reads "Colts at Texans · Thu Nov 19", with
// "· final" once played and "· bet" when the game has one. `selected` is
// the first game still to play (the last game when none is).
function matchupOptions(picks){
const all = picks || []
const key = p => `${p.season}_${String(p.week).padStart(2, "0")}`
const latest = all.map(key).sort().slice(-1)[0]
const week = all.filter(p => key(p) === latest)
.sort((a, b) => String(a.date).localeCompare(String(b.date)) || String(a.game_id).localeCompare(String(b.game_id)))
const day = p => {
const d = new Date(`${String(p.date).slice(0, 10)}T12:00:00`)
return Number.isNaN(d.getTime()) ? "" : d.toLocaleDateString("en-US", {weekday: "short", month: "short", day: "numeric"}).replace(",", "")
}
const games = week.map(p => {
const pending = p.status === "pending"
const bet = pending && !!(spreadBet(p, "nfl") || totalLean(p))
return {id: p.game_id, pending, bet,
label: `${srTeam(p.away_team, "nfl")} at ${srTeam(p.home_team, "nfl")}${day(p) ? ` · ${day(p)}` : ""}${pending ? "" : " · final"}${bet ? " · bet" : ""}`}
})
const first = games.find(g => g.pending) || games[games.length - 1]
return {games, selected: first ? first.id : null}
}

// Under the dropdown: this week's bets as buttons that open their game
// ("Colts +12.5", "Over 39.5 Rams at Cardinals"), and the games whose bet
// is waiting on the injury report. data-matchup carries the game.
function weekBetsHtml(picks){
// In the order the games are played, like the dropdown.
const byId = new Map((picks || []).map(g => [g.game_id, g]))
const upcoming = matchupOptions(picks).games.filter(g => g.pending).map(g => byId.get(g.id))
const game = g => `${srTeam(g.away_team, "nfl")} at ${srTeam(g.home_team, "nfl")}`
const chip = (g, text, held) => `<button type="button" class="betChip${held ? " held" : ""}" data-matchup="${srEsc(g.game_id)}">${srEsc(text)}</button>`
const chips = []
upcoming.forEach(g => {
const side = spreadBet(g, "nfl"), lean = totalLean(g)
if(side){ chips.push(chip(g, `${side.team} ${side.number}`)) }
if(lean){ chips.push(chip(g, `${lean.side === "over" ? "Over" : "Under"} ${lean.line} ${game(g)}`)) }
})
const held = upcoming.filter(g => (!totalLean(g) && g.total_lean_held) || (!spreadBet(g, "nfl") && g.spread_bet_held))
.map(g => chip(g, game(g), true))
if(!chips.length && !held.length){
return upcoming.length ? `<span class="weekBetsLabel">No bets this week yet.</span>` : ""
}
return `${chips.length ? `<span class="weekBetsLabel">Bets this week</span>${chips.join("")}` : `<span class="weekBetsLabel">No bets this week yet.</span>`}${
held.length ? `<span class="weekBetsLabel">Waiting on the injury report</span>${held.join("")}` : ""}`
}

// "Texans −9.5" for the points Vegas had the home team winning by.
function meetingSpreadText(m, sport){
const line = srNum(m.spread_line)
if(line === null){ return "" }
if(line === 0){ return "a pick'em" }
return `${srTeam(line > 0 ? m.home_team : m.away_team, sport)} −${srPoints(line)}`
}

// One past meeting in words: {date, score, vegas, result}. "Jan 4, 2026",
// "Texans 38, Colts 30", "Texans −9.5 (Texans −535, Colts +400), total
// 37.5", "Colts covered, over".
function meetingInfo(m, sport){
const home = srTeam(m.home_team, sport), away = srTeam(m.away_team, sport)
const hs = srNum(m.home_score), as = srNum(m.away_score)
const d = new Date(`${String(m.date).slice(0, 10)}T12:00:00`)
const date = Number.isNaN(d.getTime()) ? String(m.date || "") : d.toLocaleDateString("en-US", {year: "numeric", month: "short", day: "numeric"})
const score = hs === as ? `${home} ${hs}, ${away} ${as} (tie)` : hs > as ? `${home} ${hs}, ${away} ${as}` : `${away} ${as}, ${home} ${hs}`
const money = v => { const n = srNum(v); return n === null ? null : `${n > 0 ? "+" : "−"}${Math.abs(Math.round(n))}` }
const hm = money(m.home_moneyline), am = money(m.away_moneyline)
const spread = meetingSpreadText(m, sport), total = srNum(m.total_line)
const vegas = [spread + (spread && hm && am ? ` (${home} ${hm}, ${away} ${am})` : ""), total === null ? "" : `total ${total}`].filter(Boolean).join(", ")
const line = srNum(m.spread_line), result = []
if(line !== null && hs !== null && as !== null){
const margin = hs - as - line
result.push(margin === 0 ? "a push on the spread" : `${margin > 0 ? home : away} covered`)
}
if(total !== null && hs !== null && as !== null){
result.push(hs + as === total ? "a push on the total" : hs + as > total ? "over" : "under")
}
return {date, score, vegas, result: result.join(", ")}
}

// The last meetings with the same home team (nfl_matchup_history.csv rows
// for one game, latest first): the date and final score, what Vegas had,
// and how it came out against those numbers.
function pastMeetingsHtml(meetings, game, sport){
const rows = (meetings || []).filter(m => m.game_id === game.game_id)
const home = srTeam(game.home_team, sport), away = srTeam(game.away_team, sport)
if(!rows.length){
return `<div class="pickCard meetings"><div class="meetingNone">The ${srEsc(away)} haven't played at the ${srEsc(home)} since at least 1999.</div></div>`
}
return `<div class="pickCard meetings">${rows.map(m => {
const info = meetingInfo(m, sport)
return `<div class="meeting">
<div class="meetingScore"><span class="meetingDate">${srEsc(info.date)}</span> <b>${srEsc(info.score)}</b></div>
<div class="meetingVegas">${info.vegas ? `Vegas: ${srEsc(info.vegas)}` : "No Vegas line on file"}${info.result ? ` <span class="meetingResult">&middot; ${srEsc(info.result)}</span>` : ""}</div>
</div>`
}).join("")}</div>`
}

// "Lean OVER 37.5", and once played "won" / "lost" / "push". A Tuesday
// lean held for the injury report reads "Lean on hold".
function totalLeanBadgeHtml(g){
const lean = totalLean(g)
if(!lean){ return g.total_lean_held ? `<span class="leanBadge lean-held">Lean on hold</span>` : "" }
const done = {win: " · won", loss: " · lost", push: " · push"}[lean.result] || ""
return `<span class="leanBadge lean-${lean.side}${lean.result ? ` lean-${lean.result}` : ""}">Lean ${lean.side.toUpperCase()} ${lean.line}${done}</span>`
}

// Where the line closed against our lean: "The line closed at 45.5 - 2
// points our way". Empty before the game is played.
function totalLeanCloseNote(g){
const lean = totalLean(g)
if(!lean || lean.close === null || lean.clv === null){ return "" }
const pts = Math.abs(lean.clv)
const amount = `${pts % 1 ? pts.toFixed(1) : pts} point${pts === 1 ? "" : "s"}`
return `The line closed at ${lean.close} - ${lean.clv > 0 ? `${amount} our way` : lean.clv < 0 ? `${amount} against us` : "where we got it"}`
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

// "3.5" / "7" - a point count without a trailing .0
function srPoints(x){
const v = Math.abs(x)
return v % 1 ? v.toFixed(1) : String(Math.round(v))
}

// NFL: Vegas's point spread next to our margin - {line, margin, vegas,
// ours, side, sideLine, result} or null without a spread. `line` and
// `margin` are home minus away (positive = home ahead); `side` is the team
// our margin puts on the right side of the spread and `sideLine` its
// number ("+3.5").
function spreadInfo(g, sport){
const line = srNum(g.spread_line), margin = srNum(g.predicted_margin)
if(line === null || margin === null){ return null }
const home = srTeam(g.home_team, sport), away = srTeam(g.away_team, sport)
const by = (x, even) => Math.abs(x) < 0.25 ? even : `${x > 0 ? home : away} by ${srPoints(Math.round(Math.abs(x) * 2) / 2)}`
const side = g.spread_side === g.home_team || g.spread_side === g.away_team ? g.spread_side : null
const number = side === null ? null : (side === g.home_team ? -line : line)
return {line, margin, vegas: by(line, "a pick'em"), ours: by(margin, "it even"),
side: side === null ? null : srTeam(side, sport),
sideLine: number === null ? "" : number === 0 ? "pick'em" : `${number > 0 ? "+" : "−"}${srPoints(number)}`,
result: g.spread_result || ""}
}

// "Spread: Vegas has Bills by 3.5 · we have Bills by 2 (our side: Vikings +3.5)"
function spreadText(g, sport){
const s = spreadInfo(g, sport)
if(!s){ return "" }
const done = {win: ", covered", loss: ", didn't cover", push: ", push"}[s.result] || ""
return `Spread: Vegas has ${srEsc(s.vegas)}&nbsp;&middot; we have ${srEsc(s.ours)}${s.side ? ` (our side: ${srEsc(s.side)} ${s.sideLine}${done})` : ""}`
}

// A spread side that is a bet: our model's margin sits far enough from the
// line (config.NFL_SPREAD_BET_MIN_GAP; the pipeline decides and sends
// spread_bet). {team, number, gap, result, clv} or null.
function spreadBet(g, sport){
const s = spreadInfo(g, sport)
if(!s || !(g.spread_bet === g.home_team || g.spread_bet === g.away_team)){ return null }
const home = g.spread_bet === g.home_team
// `points`: what the team is getting (positive = it is the underdog).
return {team: srTeam(g.spread_bet, sport), home, number: s.sideLine, points: home ? -s.line : s.line,
gap: Math.abs(s.margin - s.line), result: g.spread_bet_result || "", clv: srNum(g.spread_bet_clv)}
}

// The moneyline as another way to play a spread bet
// (config.NFL_SPREAD_BET_MONEYLINE, sent in the summary): on an underdog
// getting a few points it has returned more than the points while winning
// less often; on one getting more it has lost money. "" for a favorite or
// when the summary has no numbers.
function spreadBetMoneylineNote(g, summary){
const bet = spreadBet(g, "nfl"), s = summary || {}
const cut = srNum(s.spread_bet_moneyline_max_points), n = srNum(s.spread_bet_moneyline_bets)
if(!bet || cut === null || !n || !(bet.points > 0)){ return "" }
const pct = v => `${Math.round(Math.abs(100 * (srNum(v) || 0)))}%`
if(bet.points > cut){
const big = srNum(s.spread_bet_moneyline_big_bets), lost = srNum(s.spread_bet_moneyline_big_return)
return big && lost !== null && lost < 0
? `Take the points, not the moneyline: on underdogs getting more than ${cut}, the moneyline has lost ${pct(lost)} a bet on bets like this (${big} since 2006).` : ""
}
return `Or the moneyline: on underdogs getting ${cut} points or fewer it has returned ${pct(s.spread_bet_moneyline_return)} a bet against ${pct(s.spread_bet_moneyline_points_return)} with the points (${n} bets like this since 2006). It wins less often, ${pct(s.spread_bet_moneyline_won)} of the time against ${pct(s.spread_bet_moneyline_points_covered)}, and the gap between the two isn't proven, so we grade the spread.`
}

// The worst number still worth betting, as our team's line ("+3" or
// "−2.5"): the spread that leaves our margin exactly the rule's gap from
// it, to the safe half point.
function spreadBetWorstLine(g, minGap){
const bet = spreadBet(g, "nfl")
const margin = srNum(g.predicted_margin)
if(!bet || margin === null){ return "" }
// Home-team terms: a home bet needs the home team favored by no more than
// our margin less the gap; a road bet, by no less than our margin plus it.
const homeLine = bet.home ? Math.floor((margin - minGap) * 2) / 2 : Math.ceil((margin + minGap) * 2) / 2
const number = bet.home ? -homeLine : homeLine
return number === 0 ? "pick'em" : `${number > 0 ? "+" : "−"}${srPoints(number)}`
}

// Why a spread side is a bet: "We have the Chiefs by 1; Vegas has them by
// 6.5. That is 5.5 points apart."
function spreadBetReason(g, sport){
const s = spreadInfo(g, sport), bet = spreadBet(g, sport)
if(!s || !bet){ return "" }
const gap = Math.round(bet.gap * 2) / 2
return `Vegas has ${s.vegas}; we have ${s.ours}. That is ${srPoints(gap)} points apart.`
}

// Under the spread line on a card: why this side is, or isn't, a bet.
function spreadNote(g, sport){
if(spreadBet(g, sport)){ return "Our margin is far enough from the line for this side to be a bet: sides like it have beaten the closing line in testing." }
if(g.spread_bet_held){ return `Our margin is far enough from the line to bet this side, but it waits for the injury report: ${g.spread_bet_held}.` }
return "Close to the line our side has been a coin flip in testing, so here it is context, not a bet."
}

// "Bet Broncos +6.5", and once played "won" / "lost" / "push".
function spreadBetBadgeHtml(g, sport){
const bet = spreadBet(g, sport)
if(!bet){ return "" }
const done = {win: " · won", loss: " · lost", push: " · push"}[bet.result] || ""
return `<span class="leanBadge lean-side${bet.result ? ` lean-${bet.result}` : ""}">Bet ${srEsc(bet.team)} ${bet.number}${done}</span>`
}

// The players out on one side, from "Name (QB), Name (WR)".
function srOutList(text){
return String(text === undefined || text === null ? "" : text).split(",").map(x => x.trim()).filter(Boolean)
}

// NFL: the case for the pick and what cuts against it, from what the card
// already knows - {pick, for: [...], against: [...]} of plain sentences.
// A gap of 6+ places in a ranking counts as an edge.
function pickReasons(g, sport){
const pickHome = g.predicted_winner === g.home_team
const mine = pickHome ? "home" : "away", theirs = pickHome ? "away" : "home"
const pick = srTeam(g[`${mine}_team`], sport), opp = srTeam(g[`${theirs}_team`], sport)
// Nicknames are plural: "the Vikings are", "the Vikings' 28th".
const own = name => `the ${name}${/s$/.test(name) ? "'" : "'s"}`
const out = {pick, for: [], against: []}
const rank = (side, key) => srNum(g[`${side}_${key}_rank`])
const GAP = 6
const compare = (key, label) => {
const a = rank(mine, key), b = rank(theirs, key)
if(a === null || b === null){ return }
if(a + GAP <= b){ out.for.push(`Better ${label}: ${srOrdinal(Math.round(a))} to ${own(opp)} ${srOrdinal(Math.round(b))}`) }
else if(b + GAP <= a){ out.against.push(`The ${opp} have the better ${label}: ${srOrdinal(Math.round(b))} to ${srOrdinal(Math.round(a))}`) }
}
compare("off", "offense")
compare("def", "defense")
const myOut = srOutList(g[`injury_out_${mine}`]), theirOut = srOutList(g[`injury_out_${theirs}`])
const lostQb = list => list.find(x => /\(QB\)$/.test(x))
const myQb = g[`${mine}_qb`], theirQb = g[`${theirs}_qb`]
if(lostQb(theirOut)){ out.for.push(`The ${opp} are without ${lostQb(theirOut).replace(/ \(QB\)$/, "")} at quarterback${theirQb ? `; ${theirQb} starts` : ""}`) }
if(lostQb(myOut)){ out.against.push(`The ${pick} are without ${lostQb(myOut).replace(/ \(QB\)$/, "")} at quarterback${myQb ? `; ${myQb} starts` : ""}`) }
const a = rank(mine, "qb"), b = rank(theirs, "qb")
if(a !== null && b !== null && myQb && theirQb){
if(a + GAP <= b){ out.for.push(`Better quarterback play lately: ${myQb} ranks ${srOrdinal(Math.round(a))} among this week's starters, ${theirQb} ${srOrdinal(Math.round(b))}`) }
else if(b + GAP <= a){ out.against.push(`The ${opp} have had the better quarterback play: ${theirQb} ranks ${srOrdinal(Math.round(b))} among this week's starters, ${myQb} ${srOrdinal(Math.round(a))}`) }
}
const others = list => list.filter(x => !/\(QB\)$/.test(x)).length
if(others(theirOut) >= others(myOut) + 2){ out.for.push(`The ${opp} are missing ${others(theirOut)} regulars${others(myOut) ? ` to ${own(pick)} ${others(myOut)}` : ""}`) }
else if(others(myOut) >= others(theirOut) + 2){ out.against.push(`The ${pick} are missing ${others(myOut)} regulars${others(theirOut) ? ` to ${own(opp)} ${others(theirOut)}` : ""}`) }
if(pickHome){ out.for.push("Playing at home") }else{ out.against.push("Playing on the road") }
const vegas = srNum(g.market_predicted_winner_probability)
if(vegas !== null){
if(vegas >= 0.5){ out.for.push(`Vegas agrees: ${pick} ${srPct(vegas)}`) }
else{ out.against.push(`Vegas favors the ${opp} (${srPct(1 - vegas)})`) }
}
return out
}

function pickReasonsHtml(g, sport){
const r = pickReasons(g, sport)
if(!r.for.length && !r.against.length){ return "" }
const list = (items, cls) => items.length ? `<ul class="${cls}">${items.map(x => `<li>${srEsc(x)}</li>`).join("")}</ul>` : ""
return `<div class="pickWhy">
${r.for.length ? `<div class="pickWhyHead">For the ${srEsc(r.pick)}</div>${list(r.for, "whyFor")}` : ""}
${r.against.length ? `<div class="pickWhyHead">Against</div>${list(r.against, "whyAgainst")}` : ""}
</div>`
}

// NFL: who the final injury report took out on each side, or (before it)
// the starters in doubt on the last one. Empty with nothing to say.
function injuryLineHtml(g, sport){
const sides = ["away", "home"].map(side => ({team: srTeam(g[`${side}_team`], sport), out: srOutList(g[`injury_out_${side}`])}))
.filter(s => s.out.length)
if(sides.length){
return `<div class="injuryLine"><b>Out:</b> ${sides.map(s => `${srEsc(s.team)} - ${srEsc(s.out.join(", "))}`).join("&nbsp;&middot; ")}</div>`
}
if(g.injury_report === "after"){
return g.injury_note ? `<div class="injuryLine"><b>Out:</b> ${srEsc(g.injury_note)}</div>`
: `<div class="injuryLine">The final injury report took no regulars out.</div>`
}
return g.starters_in_doubt ? `<div class="injuryLine"><b>In doubt:</b> ${srEsc(g.starters_in_doubt)}</div>` : ""
}

// One "right N% of the time" phrase from a track-record row
// (nfl_pick_confidence.csv): the backtest's rate and games, then this
// season's count. "" when the row has no games either way.
function trackPhrase(row){
if(!row){ return "" }
const games = srNum(row.backtest_games) || 0, right = srNum(row.backtest_right) || 0
const sGames = srNum(row.season_games) || 0, sRight = srNum(row.season_right) || 0
const parts = []
if(games){ parts.push(`right ${Math.round(100 * right / games)}% of the time in testing (${games.toLocaleString("en-US")} games)`) }
if(sGames){ parts.push(`${sRight} of ${sGames} this season`) }
return parts.join(", ")
}

// NFL: how picks like this one have done - by how sure we are, and, when
// we're against Vegas's favorite, how those picks have done.
function trackRecordNote(g, table){
const p = srNum(g.predicted_probability)
if(p === null || !table || !table.length){ return "" }
const band = table.find(r => r.group === "confidence" && p >= srNum(r.low) && (p < srNum(r.high) || srNum(r.high) >= 1))
const lines = []
const phrase = trackPhrase(band)
if(phrase){ lines.push(`Picks this sure (${band.label}) have been ${phrase}.`) }
const vegas = srNum(g.market_predicted_winner_probability)
if(vegas !== null && vegas < 0.5){
const against = trackPhrase(table.find(r => r.group === "vegas" && r.label === "against"))
if(against){ lines.push(`This one goes against Vegas's favorite; those picks have been ${against}.`) }
}
return lines.join(" ")
}

// ---- NFL track record ----
// How sure a pick was, in the bands the pipeline uses
// (nfl_game_evaluation.CONFIDENCE_BANDS).
const PICK_BANDS = [[0.50, 0.55], [0.55, 0.60], [0.60, 0.65], [0.65, 0.70], [0.70, 0.75], [0.75, 1.01]]
const pickBandLabel = ([low, high]) => high > 1 ? `${Math.round(low * 100)}%+` : `${Math.round(low * 100)}–${Math.round(high * 100)}%`
const pickBand = p => PICK_BANDS.findIndex(([low, high]) => p >= low && p < high)

// This season's finished picks as a cross tab: how sure we were (rows)
// against Vegas's chance for the team we picked (columns: "Under 50%" is
// Vegas having the other team, then the same bands), each cell our record
// {won, lost}. `rows` and `columns` are the labels; `rowTotals`,
// `columnTotals` and `total` the margins. A pick with no Vegas number
// counts in its row's total only. null with nothing finished.
function pickCrossTab(picks){
const finished = (picks || []).filter(g => g.status === "win" || g.status === "loss")
const season = finished.reduce((latest, g) => Math.max(latest, srNum(g.season) || 0), 0)
const done = finished.filter(g => (srNum(g.season) || 0) === season)
if(!done.length){ return null }
const blank = () => ({won: 0, lost: 0})
const columns = ["Under 50%", ...PICK_BANDS.map(pickBandLabel)]
const cells = PICK_BANDS.map(() => columns.map(blank))
const rowTotals = PICK_BANDS.map(blank), columnTotals = columns.map(blank), total = blank()
done.forEach(g => {
const ours = srNum(g.predicted_probability), vegas = srNum(g.market_predicted_winner_probability)
const row = ours === null ? -1 : pickBand(Math.max(ours, 1 - ours))
if(row < 0){ return }
const key = g.status === "win" ? "won" : "lost"
rowTotals[row][key]++
total[key]++
if(vegas === null){ return }
const column = vegas < 0.5 ? 0 : 1 + pickBand(vegas)
cells[row][column][key]++
columnTotals[column][key]++
})
return {season, rows: PICK_BANDS.map(pickBandLabel), columns, cells, rowTotals, columnTotals, total}
}

// The cross tab as a table: "12–5" in a cell is 12 right and 5 wrong; an
// empty cell had no such picks. The shaded cells are where we and Vegas
// were about equally sure.
function pickCrossTabHtml(picks){
const tab = pickCrossTab(picks)
if(!tab){ return "" }
const record = c => c.won + c.lost ? `${c.won}–${c.lost}` : ""
const rate = c => c.won + c.lost ? ` title="right ${Math.round(100 * c.won / (c.won + c.lost))}% of ${c.won + c.lost}"` : ""
const body = tab.rows.map((label, r) => `<tr><th scope="row">${label}</th>${
tab.cells[r].map((c, i) => `<td${i === r + 1 ? ' class="same"' : ""}${rate(c)}>${record(c)}</td>`).join("")
}<td class="all"${rate(tab.rowTotals[r])}>${record(tab.rowTotals[r])}</td></tr>`).join("")
return `<table class="trackRecord crossTab">
<thead>
<tr><th></th><th colspan="${tab.columns.length}" class="over">Vegas's chance for the team we picked</th><th></th></tr>
<tr><th class="corner">We say</th>${tab.columns.map(c => `<th scope="col">${c}</th>`).join("")}<th scope="col">All</th></tr>
</thead>
<tbody>${body}
<tr class="all"><th scope="row">All</th>${tab.columnTotals.map(c => `<td${rate(c)}>${record(c)}</td>`).join("")}<td${rate(tab.total)}>${record(tab.total)}</td></tr>
</tbody>
</table>`
}

// Under the cross tab: the kinds of pick it doesn't show, this season -
// with and against Vegas's favorite (and what a unit on each made at the
// closing odds), over/under leans, spread bets and our side of every
// spread. `table` is nfl_pick_confidence.csv, `summary` the picks summary.
function trackRecordTableHtml(table, summary){
const s = summary || {}
const cell = (right, games) => games ? `${right} of ${games} <span class="dim">${Math.round(100 * right / games)}%</span>` : "-"
const units = v => { const u = srNum(v); return u === null ? "" : ` <span class="dim">&middot; ${u > 0 ? "+" : u < 0 ? "−" : ""}${Math.abs(u).toFixed(1)} units</span>` }
const rows = (table || []).filter(r => r.group === "vegas" && srNum(r.season_games)).map(r =>
[r.label === "with" ? "We pick Vegas's favorite" : "We pick against Vegas's favorite",
cell(srNum(r.season_right) || 0, srNum(r.season_games) || 0) + units(r.season_units)])
const pair = (won, lost, label) => {
const w = srNum(won) || 0, l = srNum(lost) || 0
if(w + l){ rows.push([label, cell(w, w + l)]) }
}
pair(s.total_leans_won, s.total_leans_lost, "Over/under leans")
pair(s.spread_bets_won, s.spread_bets_lost, `Spread bets (${srNum(s.spread_bet_min_gap) || 3.5}+ points from the line)`)
pair(s.spread_sides_won, s.spread_sides_lost, "Our side of the spread, every game")
if(!rows.length){ return "" }
return `<table class="trackRecord">
<thead><tr><th>Kind of pick</th><th>This season</th></tr></thead>
<tbody>${rows.map(r => `<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join("")}</tbody>
</table>`
}

// One game: a card with the matchup and our pick ("Rays at Yankees", "We
// say Yankees 59%"), a color bar down its left edge (NFL: before or after
// the final injury report), and a click to open the rest: each team's
// chance as a bar, the weather, the predicted total and each team's EPA.
function gamePickRowHtml(g, sport, extras){
const track = sport === "nfl" ? trackRecordNote(g, extras && extras.confidence) : ""
const spread = sport === "nfl" ? spreadText(g, sport) : ""
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
${sport === "nfl" ? spreadBetBadgeHtml(g, sport) : ""}
${sport === "nfl" ? totalLeanBadgeHtml(g) : ""}
<span class="gameMore" aria-hidden="true"></span>
</summary>
<div class="gameDetail">
${pickBarHtml(g, sport)}
${track ? `<div class="totalNote trackNote">${srEsc(track)}</div>` : ""}
${sport === "nfl" ? pickReasonsHtml(g, sport) : ""}
${sport === "nfl" ? injuryLineHtml(g, sport) : ""}
${spread ? `<div class="total">${spread}</div><div class="totalNote">${srEsc(spreadNote(g, sport))}</div>` : ""}
${weatherLine}
${total ? `<div class="total">${total}</div>` : ""}
${total && srNum(g.predicted_total_low) !== null ? `<div class="totalNote">About half of games finish inside our range.</div>` : ""}
${sport === "nfl" && totalLean(g) ? `<div class="totalNote leanNote">${srEsc(totalLeanReason(g))}</div>` : ""}
${sport === "nfl" && !totalLean(g) && g.total_lean_held ? `<div class="totalNote leanNote">Lean on hold until this week's injury report: ${srEsc(g.total_lean_held)}.</div>` : ""}
${sport === "nfl" && totalLeanCloseNote(g) ? `<div class="totalNote">${srEsc(totalLeanCloseNote(g))}</div>` : ""}
${sport === "nfl" && totalRangeNote(g) ? `<div class="totalNote">${srEsc(totalRangeNote(g))}</div>` : ""}
${epa}
${reportNote}
</div>
</details>`
}

function gamePicksListHtml(picks, sport, extras){
if(!picks || !picks.length){
return '<div class="empty-note">No games scheduled yet.</div>'
}
return picks.map(g => gamePickRowHtml(g, sport, extras)).join("")
}

if (typeof module !== "undefined" && module.exports) {
module.exports = {
srPct, streakTilesHtml, btsPicksHtml, gamePicksRecordHtml, gamePickRowHtml, gamePicksListHtml, totalPointsText,
srWeather, pickBarHtml, totalLean, totalLeanBadgeHtml, totalRangeNote, totalLeanCloseNote, totalLeanReason,
betWorstLine, betPriceFloor, betsPanelHtml, matchupOptions, weekBetsHtml, meetingInfo, pastMeetingsHtml,
btsSeasonOver, btsSeasonOverHtml,
teamEpaHtml, spreadInfo, spreadText, spreadBet, spreadBetWorstLine, spreadBetReason, spreadBetMoneylineNote, spreadNote, spreadBetBadgeHtml, pickReasons, pickReasonsHtml, injuryLineHtml, trackRecordNote, trackRecordTableHtml, pickCrossTab, pickCrossTabHtml,
}
}
