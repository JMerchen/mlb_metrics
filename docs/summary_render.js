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

// Why we lean: "We lean under Vegas's 46.5: our total is 4 points lower,
// and the forecast's rain and wind take off about 3 more." (The weather
// part is what the line still misses for it - config.NFL_TOTAL_LEAN_WEATHER.)
function totalLeanReason(g){
const lean = totalLean(g)
if(!lean){ return "" }
const ours = srNum(g.predicted_total)
const diff = ours === null ? null : ours - lean.line
const parts = [`We lean ${lean.side} Vegas's ${lean.line}`]
if(diff !== null && Math.round(Math.abs(diff)) > 0){
parts.push(`our total is ${Math.round(Math.abs(diff))} point${Math.round(Math.abs(diff)) === 1 ? "" : "s"} ${diff > 0 ? "higher" : "lower"}`)
}
const wx = Math.round(Math.abs(lean.weather))
if(parts.length > 1){
return wx >= 1 ? `${parts[0]}: ${parts[1]}, and the forecast's rain and wind take off about ${wx} more.` : `${parts[0]}: ${parts[1]}.`
}
return wx >= 1 ? `${parts[0]}: the forecast's rain and wind take off about ${wx} point${wx === 1 ? "" : "s"}.` : `${parts[0]}.`
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

// Why the spread line is context and not a lean.
const SPREAD_NOTE = "Our side of the spread has been a coin flip in testing (about half of 1,900 past games), so read it as context, not a lean."

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

// The NFL page's track-record table: each kind of pick, how it has done
// this season and in testing.
function trackRecordTableHtml(table, summary){
const s = summary || {}
const cell = (right, games) => games ? `${right} of ${games} <span class="dim">${Math.round(100 * right / games)}%</span>` : "-"
const tested = (right, games) => games ? `${Math.round(100 * right / games)}% <span class="dim">of ${Number(games).toLocaleString("en-US")}</span>` : "-"
const rows = (table || []).map(r => {
const label = r.group === "confidence" ? `We say ${r.label}`
: r.label === "with" ? "We pick Vegas's favorite" : "We pick against Vegas's favorite"
return [label, cell(srNum(r.season_right) || 0, srNum(r.season_games) || 0), tested(srNum(r.backtest_right) || 0, srNum(r.backtest_games) || 0)]
})
const pair = (won, lost, rate, games, label) => {
const w = srNum(won) || 0, l = srNum(lost) || 0, n = srNum(games) || 0, r = srNum(rate)
if(!(w + l) && !n){ return }
rows.push([label, cell(w, w + l), n && r !== null ? `${Math.round(100 * r)}% <span class="dim">of ${n.toLocaleString("en-US")}</span>` : "-"])
}
pair(s.total_leans_won, s.total_leans_lost, s.total_lean_backtest_won_rate, s.total_lean_backtest_games, "Over/under leans")
pair(s.spread_sides_won, s.spread_sides_lost, s.spread_backtest_won_rate, s.spread_backtest_games, "Our side of the spread")
if(!rows.length){ return "" }
return `<table class="trackRecord">
<thead><tr><th>When</th><th>This season</th><th>In testing</th></tr></thead>
<tbody>${rows.map(r => `<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td></tr>`).join("")}</tbody>
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
${sport === "nfl" ? totalLeanBadgeHtml(g) : ""}
<span class="gameMore" aria-hidden="true"></span>
</summary>
<div class="gameDetail">
${pickBarHtml(g, sport)}
${track ? `<div class="totalNote trackNote">${srEsc(track)}</div>` : ""}
${sport === "nfl" ? pickReasonsHtml(g, sport) : ""}
${sport === "nfl" ? injuryLineHtml(g, sport) : ""}
${spread ? `<div class="total">${spread}</div><div class="totalNote">${SPREAD_NOTE}</div>` : ""}
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
btsSeasonOver, btsSeasonOverHtml,
teamEpaHtml, spreadInfo, spreadText, pickReasons, pickReasonsHtml, injuryLineHtml, trackRecordNote, trackRecordTableHtml,
}
}
