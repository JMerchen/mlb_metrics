// MLB Picks (predictive.html) and Stats (stats.html) pages: headshots,
// logos, charts and cards. Pure functions returning HTML strings so node tests can check
// them; app.js loads the CSVs and does the DOM wiring.
//
// Colour: one diverging pair for "better / worse than average" - blue and
// orange, the same pair the NFL props use (validated with the dataviz
// palette checker on --bg-card) - with a grey midpoint. Text never takes a
// series colour, and every coloured mark has a word or number beside it.

const MLB_TEAM_IDS = {
AZ: 109, ATL: 144, BAL: 110, BOS: 111, CHC: 112, CWS: 145, CIN: 113, CLE: 114, COL: 115, DET: 116,
HOU: 117, KC: 118, LAA: 108, LAD: 119, MIA: 146, MIL: 158, MIN: 142, NYM: 121, NYY: 147, ATH: 133,
PHI: 143, PIT: 134, SD: 135, SF: 137, SEA: 136, STL: 138, TB: 139, TEX: 140, TOR: 141, WSH: 120,
}

function mlbEsc(value){
return String(value ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;")
}

function mlbNum(value){
return value === undefined || value === null || value === "" || Number.isNaN(Number(value)) ? null : Number(value)
}

function mlbTeam(code){
return typeof teamName === "function" ? teamName(code, "mlb") : (code || "")
}

function mlbPct(value, digits = 0){
const n = mlbNum(value)
return n === null ? "-" : `${(n * 100).toFixed(digits)}%`
}

// "+22%" / "−5%" / "avg" from an index where 1.00 is league average.
function mlbIndex(value){
const n = mlbNum(value)
if(n === null){ return "-" }
const pct = Math.round((n - 1) * 100)
return pct === 0 ? "avg" : `${pct > 0 ? "+" : "−"}${Math.abs(pct)}%`
}

function mlbOrdinal(n){
const tens = n % 100, ones = n % 10
const suffix = tens >= 11 && tens <= 13 ? "th" : ones === 1 ? "st" : ones === 2 ? "nd" : ones === 3 ? "rd" : "th"
return `${n}${suffix}`
}

function mlbShortDate(iso){
const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
const [y, m, d] = String(iso || "").split("-").map(Number)
return y && m && d ? `${months[m - 1]} ${d}` : String(iso || "")
}

// ---- Pictures ----
// Headshots and caps come from MLB's CDN. Each picture sits over a text
// fallback (initials, team code) and removes itself if it fails to load,
// so a blocked or missing image never leaves a broken-image icon.

function mlbHeadshotUrl(id){
const key = String(id ?? "").trim().replace(/\.0$/, "")
return key ? `https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_120,q_auto:best/v1/people/${key}/headshot/67/current` : ""
}

// Caps drawn for a dark background, or for a light one (the paper
// schedule sheet) with onLight.
function mlbLogoUrl(team, onLight = false){
const id = MLB_TEAM_IDS[String(team ?? "").trim().toUpperCase()]
return id ? `https://www.mlbstatic.com/team-logos/team-cap-on-${onLight ? "light" : "dark"}/${id}.svg` : ""
}

function mlbInitials(name){
return String(name || "").split(/\s+/).filter(Boolean).map(w => w[0]).slice(0, 2).join("").toUpperCase()
}

function mlbAvatar(id, name, size = "md"){
const url = mlbHeadshotUrl(id)
return `<span class="mlbAvatar mlbAvatar-${size}" aria-hidden="true"><span>${mlbEsc(mlbInitials(name))}</span>${
url ? `<img src="${url}" alt="" loading="lazy" onload="this.parentNode.classList.add('loaded')" onerror="this.remove()">` : ""}</span>`
}

function mlbLogo(team, size = "sm", onLight = false){
const url = mlbLogoUrl(team, onLight)
return `<span class="mlbLogo mlbLogo-${size}" title="${mlbEsc(mlbTeam(team))}" aria-hidden="true"><span>${mlbEsc(team || "")}</span>${
url ? `<img src="${url}" alt="" loading="lazy" onload="this.parentNode.classList.add('loaded')" onerror="this.remove()">` : ""}</span>`
}

// ---- Better / worse than average ----

// Share of `pool` below `value` (ties count half), 0-100. `direction` -1
// flips it for stats where lower is better (whiffs, hits allowed).
function mlbPercentile(pool, value, direction = 1){
const v = mlbNum(value)
const values = (pool || []).map(mlbNum).filter(n => n !== null)
if(v === null || !values.length){ return null }
const below = values.filter(n => n < v).length
const equal = values.filter(n => n === v).length
const p = 100 * (below + equal / 2) / values.length
return Math.round(direction < 0 ? 100 - p : p)
}

// Bar colour for a percentile: orange (poor) through grey (average) to
// blue (great). Neutral stats (no better or worse) stay grey.
function mlbPercentileColor(p, neutral = false){
if(p === null || neutral){ return "var(--mlb-mid)" }
const mix = (a, b, t) => a.map((c, i) => Math.round(c + (b[i] - c) * t))
const under = [217, 89, 38], mid = [91, 98, 112], over = [57, 135, 229]
const rgb = p >= 50 ? mix(mid, over, (p - 50) / 50) : mix(under, mid, p / 50)
return `rgb(${rgb.join(",")})`
}

// One stat as a percentile bar: label, bar with the percentile in a dot
// at its end, and the stat itself on the right.
function mlbPercentileRow(label, valueText, p, opts = {}){
const width = p === null ? 0 : Math.max(3, p)
const tip = opts.tip || (p === null ? label : `${label}: ${valueText} - better than ${p}% of ${opts.poolName || "the league"}`)
return `<div class="pctRow" data-tip="${mlbEsc(tip)}" tabindex="0">
<span class="pctLabel">${mlbEsc(label)}</span>
<span class="pctTrack"><span class="pctFill" style="width:${width}%;background:${mlbPercentileColor(p, opts.neutral)}"></span>${
p === null ? "" : `<span class="pctDot" style="left:${width}%;border-color:${mlbPercentileColor(p, opts.neutral)}">${opts.dotText ?? p}</span>`}</span>
<span class="pctValue">${mlbEsc(valueText)}</span>
</div>`
}

// Worst / best labels lined up over a group's bars.
function mlbScaleRow(left, right){
return `<div class="pctRow pctScaleRow" aria-hidden="true"><span></span><span class="pctScaleInner"><span>${left}</span><span>${right}</span></span><span></span></div>`
}

// ---- Beat the Streak ----

// The streak after every resolved day of the season (one column per day,
// beat_the_streak_progression.csv), with the days a pick went hitless
// marked under the axis. HTML columns rather than an SVG so the labels
// stay readable at phone width.
function streakChartHtml(progression, picks){
const days = (progression || []).filter(d => d.date)
if(!days.length){ return "" }
const values = days.map(d => mlbNum(d.streak) || 0)
const best = Math.max(...values, 1)
const top = Math.max(5, Math.ceil(best / 5) * 5)
const bestIndex = values.lastIndexOf(best)
const byDate = {}
;(picks || []).forEach(p => { (byDate[p.date] = byDate[p.date] || []).push(p) })
const resultWord = { hit: "hit", miss: "no hit", no_game: "didn't play", pending: "not finished" }
const columns = days.map((d, i) => {
const reset = String(d.reset).toLowerCase() === "true"
const who = (byDate[d.date] || []).filter(p => p.grade === "recommended")
.map(p => `${p.name} (${resultWord[p.status] || p.status})`).join(", ")
const tip = `${mlbShortDate(d.date)}: streak ${values[i]}${reset ? " - reset" : ""}${who ? ` · ${who}` : ""}`
const height = (values[i] / top) * 100
const label = i === bestIndex ? `<span class="streakLabel">Best ${best}</span>`
: i === days.length - 1 ? `<span class="streakLabel">Final ${values[i]}</span>` : ""
return `<span class="streakCol" data-tip="${mlbEsc(tip)}" tabindex="0"><span class="streakBar" style="height:${height.toFixed(1)}%">${label}</span>${reset ? `<span class="streakReset"></span>` : ""}</span>`
}).join("")
// Month ticks at the first day of each month in the data.
let lastMonth = ""
const ticks = days.map((d, i) => {
const month = String(d.date).slice(0, 7)
if(month === lastMonth){ return "" }
lastMonth = month
return `<span class="streakTick" style="left:${(100 * i / days.length).toFixed(2)}%">${mlbShortDate(d.date).split(" ")[0]}</span>`
}).join("")
const grid = [0, top / 2, top].map(v => `<span class="streakGrid" style="bottom:${(100 * v / top).toFixed(1)}%"><i>${v}</i></span>`).join("")
const resets = days.filter(d => String(d.reset).toLowerCase() === "true").length
return `<div class="vizCard">
<div class="vizHead"><b>Our streak, day by day</b><span class="vizKey"><i class="keyBar"></i>Streak after the day <i class="keyReset"></i>Reset (a pick went hitless)</span></div>
<div class="streakPlot" role="img" aria-label="${mlbEsc(`Streak over ${days.length} days: best ${best}, reset ${resets} times, ended at ${values[values.length - 1]}`)}">${grid}<div class="streakCols">${columns}</div></div>
<div class="streakTicks">${ticks}</div>
</div>`
}

const BTS_RESULT = {
hit: ["✓", "Got a hit", "resultGood"],
miss: ["✗", "No hit", "resultBad"],
no_game: ["–", "Didn't play", "resultNeutral"],
pending: ["…", "Not finished", "resultNeutral"],
}

function btsResultChip(status){
const [icon, word, cls] = BTS_RESULT[status] || ["", status || "", "resultNeutral"]
return `<span class="resultChip ${cls}">${icon} ${mlbEsc(word)}</span>`
}

// Today's (or the season's last) picks as cards: headshot, team, chance.
// `playersById` maps key_mlbam to a wave.csv row, for the team.
function btsPickCardsHtml(picks, playersById){
return (picks || []).map(p => {
const player = (playersById || {})[String(p.key_mlbam ?? "").replace(/\.0$/, "")] || {}
const team = player.team || p.team || ""
const chance = mlbNum(p.combined_probability) ?? mlbNum(p.predicted_probability)
const grade = p.grade === "recommended" ? "Strong pick" : p.grade === "speculative" ? "Weaker pick" : ""
return `<div class="playerPick ${mlbEsc(p.status || "")}">
${mlbAvatar(p.key_mlbam, p.name, "lg")}
<div class="playerPickBody">
<div class="playerPickName">${mlbEsc(p.name)}</div>
<div class="playerPickTeam">${team ? `${mlbLogo(team)} ${mlbEsc(mlbTeam(team))}` : ""}</div>
<div class="playerPickChance"><b>${chance === null ? "-" : mlbPct(chance)}</b> chance of a hit${grade ? ` · <span class="pickGrade ${mlbEsc(p.grade)}">${grade}</span>` : ""}</div>
${btsResultChip(p.status)}
</div>
</div>`
}).join("")
}

// Day-by-day log, newest first: each day's picks with how they did and
// the streak after the day.
function btsLogHtml(picks, progression, limit = 21){
const streakByDate = {}
;(progression || []).forEach(d => { streakByDate[d.date] = d })
const byDate = {}
;(picks || []).forEach(p => { (byDate[p.date] = byDate[p.date] || []).push(p) })
const dates = Object.keys(byDate).sort().reverse()
if(!dates.length){ return "No picks tracked yet" }
const rows = dates.slice(0, limit).map(date => {
const day = byDate[date].slice().sort((a, b) => Number(a.rank) - Number(b.rank))
const chips = day.filter(p => p.status !== "no_pick").map(p => {
const [icon, word, cls] = BTS_RESULT[p.status] || ["", p.status, "resultNeutral"]
const weak = p.grade === "speculative" ? `<span class="logWeak" title="Weaker pick - doesn't count toward the streak">weaker</span>` : ""
return `<span class="logPick" data-tip="${mlbEsc(`${p.name}: ${mlbPct(p.combined_probability)} chance - ${word}`)}" tabindex="0">${mlbAvatar(p.key_mlbam, p.name, "sm")}<span class="logName">${mlbEsc(p.name)}</span>${weak}<span class="resultIcon ${cls}" aria-label="${mlbEsc(word)}">${icon}</span></span>`
}).join("") || `<span class="logEmpty">No pick</span>`
const step = streakByDate[date]
const streak = step ? `<span class="logStreak${String(step.reset).toLowerCase() === "true" ? " logStreakReset" : ""}">${String(step.reset).toLowerCase() === "true" ? "Reset to 0" : `Streak ${step.streak}`}</span>`
: `<span class="logStreak">${day.some(p => p.grade === "recommended") ? "" : "Sat out"}</span>`
return `<div class="logRow"><span class="logDate">${mlbShortDate(date)}</span><span class="logPicks">${chips}</span>${streak}</div>`
}).join("")
return `<div class="logList">${rows}</div>`
}

// The longest active hit streaks: one pip per game.
function hitStreakListHtml(streaks, limit = 10){
const top = (streaks || []).slice()
.sort((a, b) => (mlbNum(b.Current_Hit_Streak) || 0) - (mlbNum(a.Current_Hit_Streak) || 0))
.slice(0, limit)
if(!top.length){ return "No hit streak data yet" }
return `<div class="rankList">${top.map((p, i) => {
const games = mlbNum(p.Current_Hit_Streak) || 0
const name = `${p.name_first} ${p.name_last}`
const pips = Array.from({ length: Math.min(games, 40) }, () => "<i></i>").join("")
return `<div class="rankRow" data-tip="${mlbEsc(`${name}: hit in ${games} straight games (last game ${mlbShortDate(p.last_game_date)})`)}" tabindex="0">
<span class="rankNo">${i + 1}</span>${mlbAvatar(p.key_mlbam, name, "sm")}
<span class="rankWho"><b>${mlbEsc(name)}</b><small>${mlbLogo(p.team, "xs")} ${mlbEsc(mlbTeam(p.team))}</small></span>
<span class="pips">${pips}</span><span class="rankValue">${games} games</span>
</div>`
}).join("")}</div>`
}

// A simple 0-100% bar for a probability.
function mlbChanceBar(p){
const n = mlbNum(p)
return `<span class="chanceTrack"><span class="chanceFill" style="width:${n === null ? 0 : (n * 100).toFixed(1)}%"></span></span>`
}

// Today's best chances of a hit, with the opponent.
function bestChancesHtml(predictions, limit = 10){
const top = (predictions || []).slice()
.sort((a, b) => (mlbNum(b.Model_Hit_Probability) || 0) - (mlbNum(a.Model_Hit_Probability) || 0))
.slice(0, limit)
if(!top.length){ return "No chances posted yet today" }
return `<div class="rankList">${top.map((p, i) => {
const name = `${p.name_first} ${p.name_last}`
const home = String(p.is_home).toLowerCase() === "true"
return `<div class="rankRow" data-tip="${mlbEsc(`${name}: ${mlbPct(p.Model_Hit_Probability, 1)} chance of a hit ${home ? "vs." : "at"} the ${mlbTeam(p.opponent)}`)}" tabindex="0">
<span class="rankNo">${i + 1}</span>${mlbAvatar(p.key_mlbam, name, "sm")}
<span class="rankWho"><b>${mlbEsc(name)}</b><small>${mlbLogo(p.team, "xs")} ${home ? "vs." : "at"} ${mlbLogo(p.opponent, "xs")} ${mlbEsc(mlbTeam(p.opponent))}</small></span>
${mlbChanceBar(p.Model_Hit_Probability)}<span class="rankValue">${mlbPct(p.Model_Hit_Probability)}</span>
</div>`
}).join("")}</div>`
}

// ---- Starting pitchers ----

// "23% fewer hits than average" from PAVE_PLUS (1.00 = average).
function pitcherVsAverage(ratio){
const n = mlbNum(ratio)
if(n === null){ return "" }
const pct = Math.round((n - 1) * 100)
return pct === 0 ? "an average hit rate" : `${Math.abs(pct)}% ${pct < 0 ? "fewer" : "more"} hits than average`
}

// Small diverging meter around average (±50% fills the half).
function mlbMeter(share, higherIsGood){
if(share === null){ return "" }
const width = Math.min(50, Math.abs(share) * 100)
const right = share >= 0
const good = right === higherIsGood
return `<span class="mlbMeter"><span class="mlbMeterMid"></span><span class="mlbMeterBar" style="${right ? "left:50%" : "right:50%"};width:${width.toFixed(1)}%;background:${good ? "var(--mlb-better)" : "var(--mlb-worse)"}"></span></span>`
}

function pitcherSideHtml(row){
const name = row.pitcher_name || "Not announced"
const throws = row.Throws === "L" ? "Lefty" : row.Throws === "R" ? "Righty" : ""
const ratio = mlbNum(row.PAVE_PLUS)
return `<div class="duelSide">
${mlbAvatar(row.key_mlbam, row.pitcher_name || row.team, "md")}
<div class="duelWho"><b>${mlbEsc(name)}</b><small>${mlbLogo(row.team, "xs")} ${mlbEsc(mlbTeam(row.team))}${throws ? ` · ${throws}` : ""}</small>
${ratio === null ? `<small>No hit rate yet</small>` : `<div class="duelStat"><span>Hits on <b>${mlbPct(row.PAVE)}</b> of at-bats · ${mlbEsc(pitcherVsAverage(ratio))}</span>${mlbMeter(1 - ratio, true)}</div>`}
</div>
</div>`
}

// One card per game: the two probable starters face to face, away first.
function pitcherDuelsHtml(rows){
const list = rows || []
if(!list.length){ return "No probable pitchers yet" }
const seen = new Set()
const cards = []
list.forEach(r => {
if(seen.has(r.team)){ return }
const other = list.find(o => o.team === r.opponent && o.opponent === r.team)
seen.add(r.team)
if(other){ seen.add(other.team) }
const home = String(r.is_home).toLowerCase() === "true"
const away = home ? other : r, host = home ? r : other
cards.push(`<div class="duelCard">
<div class="duelTitle">${away ? mlbLogo(away.team) : ""} ${mlbEsc(mlbTeam(away ? away.team : r.opponent))} <span>at</span> ${host ? mlbLogo(host.team) : ""} ${mlbEsc(mlbTeam(host ? host.team : r.opponent))}</div>
${away ? pitcherSideHtml(away) : ""}${host ? pitcherSideHtml(host) : ""}
</div>`)
})
return `<div class="duelCards">${cards.join("")}</div>`
}

// ---- Game picks ----

const GAME_RESULT = {
win: ["✓", "We were right", "resultGood"],
loss: ["✗", "We were wrong", "resultBad"],
not_played: ["–", "Postponed", "resultNeutral"],
pending: ["…", "Not played yet", "resultNeutral"],
}

// A game card: both caps, then a win-probability bar split between the
// two teams (our numbers), with Vegas's split marked on it.
function gamePickCardHtml(p){
const homePicked = p.predicted_winner === p.home_team
const ours = mlbNum(p.predicted_probability)
const homeChance = ours === null ? null : (homePicked ? ours : 1 - ours)
const vegasHome = mlbNum(p.market_home_win_probability)
const [icon, word, cls] = GAME_RESULT[p.status] || ["", p.status, "resultNeutral"]
const bet = Number(p.bet_units) > 0 ? `<div class="gameBet">Bet ${mlbEsc(mlbTeam(p.bet_team))} · ${Number(p.bet_units).toFixed(1)} units</div>` : ""
const vegasText = vegasHome === null ? "" : ` · Vegas ${mlbPct(homePicked ? vegasHome : 1 - vegasHome)}`
const awayWidth = homeChance === null ? 50 : (1 - homeChance) * 100
return `<div class="gameCard2${Number(p.bet_units) > 0 ? " gameCardBet" : ""}">
<div class="gameTeams">
<span class="gameTeam${homePicked ? "" : " gamePicked"}">${mlbLogo(p.away_team, "md")}<span>${mlbEsc(mlbTeam(p.away_team))}</span></span>
<span class="gameAt">at</span>
<span class="gameTeam gameHome${homePicked ? " gamePicked" : ""}"><span>${mlbEsc(mlbTeam(p.home_team))}</span>${mlbLogo(p.home_team, "md")}</span>
</div>
<div class="winSplit" data-tip="${mlbEsc(`We say ${mlbTeam(p.away_team)} ${mlbPct(1 - (homeChance ?? 0.5))}, ${mlbTeam(p.home_team)} ${mlbPct(homeChance ?? 0.5)}${vegasHome === null ? "" : `; Vegas ${mlbPct(1 - vegasHome)} / ${mlbPct(vegasHome)}`}`)}" tabindex="0">
<span class="winSeg${homePicked ? "" : " winSegPicked"}" style="width:${awayWidth.toFixed(1)}%"><b>${mlbPct(1 - (homeChance ?? 0.5))}</b></span><span class="winSeg${homePicked ? " winSegPicked" : ""}" style="width:${(100 - awayWidth).toFixed(1)}%"><b>${mlbPct(homeChance ?? 0.5)}</b></span>
${vegasHome === null ? "" : `<span class="winVegas" style="left:${((1 - vegasHome) * 100).toFixed(1)}%"><i>Vegas</i></span>`}
</div>
<div class="gameSay">We say <b>${mlbEsc(mlbTeam(p.predicted_winner))}</b> ${mlbPct(ours)}${vegasText}</div>
<div class="gameFoot"><span class="resultChip ${cls}">${icon} ${mlbEsc(word)}</span>${bet}</div>
</div>`
}

// Finished games where Vegas had a line, oldest first, each marked with
// whether we and Vegas picked the winner.
function gamesVersusVegas(picks){
return (picks || []).filter(p => (p.status === "win" || p.status === "loss") && mlbNum(p.market_home_win_probability) !== null)
.slice().sort((a, b) => String(a.date).localeCompare(String(b.date)))
.map(p => {
const weHome = p.predicted_winner === p.home_team
const homeWon = weHome === (p.status === "win")
const vegasHome = mlbNum(p.market_home_win_probability) > 0.5
return { date: p.date, ours: p.status === "win", vegas: vegasHome === homeWon }
})
}

// Running "games ahead of (or behind) Vegas" over the season: +1 when we
// were right and Vegas wasn't, -1 the other way round. One series, so no
// legend; the zero line is labelled.
function versusVegasChartHtml(picks){
const games = gamesVersusVegas(picks)
if(games.length < 10){ return "" }
let net = 0
const points = games.map(g => { net += (g.ours ? 1 : 0) - (g.vegas ? 1 : 0); return { date: g.date, net } })
const lo = Math.min(0, ...points.map(p => p.net)), hi = Math.max(0, ...points.map(p => p.net))
const span = Math.max(4, hi - lo)
const top = hi + span * 0.12, bottom = lo - span * 0.12
const W = 1000, H = 220
const x = i => (W * i / (points.length - 1))
const y = v => H * (top - v) / (top - bottom)
const path = points.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p.net).toFixed(1)}`).join("")
const ours = games.filter(g => g.ours).length, vegas = games.filter(g => g.vegas).length
const last = points[points.length - 1]
const verdict = last.net === 0 ? "level with Vegas" : `${Math.abs(last.net)} game${Math.abs(last.net) === 1 ? "" : "s"} ${last.net > 0 ? "ahead of" : "behind"} Vegas`
// Hover bands: one per ~1% of the season, each reporting the running record.
const bands = 60
const hits = Array.from({ length: bands }, (_, b) => {
const i = Math.min(points.length - 1, Math.round((b + 0.5) * points.length / bands))
const upTo = games.slice(0, i + 1)
const tip = `Through ${mlbShortDate(points[i].date)}: we were right on ${upTo.filter(g => g.ours).length} of ${upTo.length}, Vegas ${upTo.filter(g => g.vegas).length} (${points[i].net > 0 ? "+" : points[i].net < 0 ? "−" : ""}${Math.abs(points[i].net)})`
return `<span class="vegasHit" style="left:${(100 * b / bands).toFixed(2)}%;width:${(100 / bands).toFixed(2)}%" data-tip="${mlbEsc(tip)}" data-mark="${(100 * x(i) / W).toFixed(2)},${(100 * y(points[i].net) / H).toFixed(2)}"></span>`
}).join("")
let lastMonth = ""
const ticks = points.map((p, i) => {
const month = String(p.date).slice(0, 7)
if(month === lastMonth){ return "" }
lastMonth = month
return `<span class="streakTick" style="left:${(100 * i / points.length).toFixed(2)}%">${mlbShortDate(p.date).split(" ")[0]}</span>`
}).join("")
return `<div class="vizCard">
<div class="vizHead"><b>Games ahead of or behind Vegas</b><span class="vizSub">Over ${games.length} finished games with a Vegas line we were right on ${ours} and Vegas on ${vegas}, so we finished ${verdict}.</span></div>
<div class="vegasPlot" role="img" aria-label="${mlbEsc(`Running games ahead of Vegas; ends ${verdict}`)}">
<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none"><line x1="0" x2="${W}" y1="${y(0).toFixed(1)}" y2="${y(0).toFixed(1)}" class="vegasZero"/><path d="${path}" class="vegasLine"/></svg>
<span class="vegasZeroLabel" style="top:${(100 * y(0) / H).toFixed(1)}%">Level with Vegas</span>
<span class="vegasEnd" style="top:${(100 * y(last.net) / H).toFixed(1)}%">${last.net > 0 ? "+" : last.net < 0 ? "−" : ""}${Math.abs(last.net)}</span>
<span class="vegasDot"></span>${hits}
</div>
<div class="streakTicks">${ticks}</div>
</div>`
}

// How often we were right at each confidence level - do our bigger
// numbers really win more often?
const CONFIDENCE_BUCKETS = [[0.5, 0.55, "50–55%"], [0.55, 0.6, "55–60%"], [0.6, 0.65, "60–65%"], [0.65, 1.01, "65% and up"]]

function confidenceBucketsHtml(picks){
const done = (picks || []).filter(p => p.status === "win" || p.status === "loss")
const rows = CONFIDENCE_BUCKETS.map(([lo, hi, label]) => {
const games = done.filter(p => { const n = mlbNum(p.predicted_probability); return n !== null && n >= lo && n < hi })
const right = games.filter(p => p.status === "win").length
return { label, n: games.length, rate: games.length ? right / games.length : null, right, mid: (lo + Math.min(hi, 0.75)) / 2 }
}).filter(r => r.n)
if(!rows.length){ return "" }
return `<div class="vizCard">
<div class="vizHead"><b>When we were more sure, were we right more often?</b><span class="vizKey"><i class="keyBar"></i>How often we were right <i class="keyTick"></i>What we said</span></div>
${rows.map(r => `<div class="bucketRow" data-tip="${mlbEsc(`Picks we gave ${r.label}: right on ${r.right} of ${r.n}`)}" tabindex="0">
<span class="bucketLabel">We said ${r.label}</span>
<span class="chanceTrack bucketTrack"><span class="chanceFill" style="width:${(r.rate * 100).toFixed(1)}%"></span><span class="bucketTick" style="left:${(r.mid * 100).toFixed(1)}%"></span></span>
<span class="bucketValue">${mlbPct(r.rate)} <small>of ${r.n}</small></span>
</div>`).join("")}
</div>`
}

// Recent results, one row per game, with caps.
function gameHistoryHtml(picks, limit = 30){
const rows = (picks || []).slice()
.sort((a, b) => String(b.date).localeCompare(String(a.date)) || (mlbNum(b.predicted_probability) || 0) - (mlbNum(a.predicted_probability) || 0))
.slice(0, limit)
if(!rows.length){ return "No game picks yet" }
return `<div class="logList">${rows.map(p => {
const [icon, word, cls] = GAME_RESULT[p.status] || ["", p.status, "resultNeutral"]
const vegas = mlbNum(p.market_predicted_winner_probability)
return `<div class="logRow gameLogRow"><span class="logDate">${mlbShortDate(p.date)}</span>
<span class="logGame">${mlbLogo(p.away_team, "xs")} ${mlbEsc(mlbTeam(p.away_team))} <span class="dim">at</span> ${mlbLogo(p.home_team, "xs")} ${mlbEsc(mlbTeam(p.home_team))}</span>
<span class="logPickTeam">${mlbEsc(mlbTeam(p.predicted_winner))} <b>${mlbPct(p.predicted_probability)}</b>${vegas === null ? "" : ` <span class="dim">Vegas ${mlbPct(vegas)}</span>`}</span>
<span class="resultIcon ${cls}" title="${mlbEsc(word)}">${icon}<span class="srOnly"> ${mlbEsc(word)}</span></span></div>`
}).join("")}</div>`
}

// ---- Team profiles ----

// [column, label, format, direction]: direction 1 = higher is better,
// -1 = lower is better, 0 = neither (shown, but not graded).
const TEAM_PROFILE_GROUPS = [
["Winning", [
["win_rate", "Win rate", "pct", 1],
["Confidence", "Overall rating", "index", 1],
["current", "Recent form", "index", 1],
["pyth_Strength", "Run differential strength", "index", 1],
["SOS", "Schedule toughness", "index", 0],
]],
["Hitting", [
["true_power", "Overall power", "index", 1],
["offensive_edge", "Bases gained vs. allowed", "index", 1],
["suppression_resistance", "Scoring vs. tough pitching", "index", 1],
["team_bases_pg", "Bases per game", "fixed1", 1],
["homer_per_game", "Home runs per game", "fixed2", 1],
["game_homer_rate", "Games with a home run", "pct", 1],
]],
["Bullpen", [
["Bullpen_PAVE", "Hit rate allowed", "pct", -1],
["Bullpen_Power_A_PLUS", "Power allowed", "index", -1],
["Bullpen_HR_Per", "Home runs per at-bat", "pct1", -1],
]],
["Ballpark", [
["Park_Factor", "Ballpark factor", "index", 0],
["team_home_run_rate", "Home runs a game at home", "fixed2", 0],
["away_hr_rate", "Home runs a game on the road", "fixed2", 0],
]],
]

const TEAM_STAT_NOTES = {
Park_Factor: "above 0% means the park favours hitters",
SOS: "how good their opponents have been",
Confidence: "our blend of record, run differential, schedule and power",
}

function teamStatText(value, format){
const n = mlbNum(value)
if(n === null){ return "-" }
switch(format){
case "pct": return mlbPct(n)
case "pct1": return mlbPct(n, 1)
case "index": return mlbIndex(n)
case "fixed1": return n.toFixed(1)
case "fixed2": return n.toFixed(2)
case "plus": return String(Math.round(n))
default: return String(n)
}
}

// 1 = best of the 30 (highest for neutral stats).
function teamRank(teams, team, key, direction){
const value = mlbNum(team[key])
if(value === null){ return null }
const sign = direction < 0 ? -1 : 1
return 1 + (teams || []).filter(t => mlbNum(t[key]) !== null && sign * mlbNum(t[key]) > sign * value).length
}

function teamRecord(team){
const games = mlbNum(team.games_played), rate = mlbNum(team.win_rate)
if(games === null || rate === null){ return "" }
const wins = Math.round(games * rate)
return `${wins}-${games - wins}`
}

// Every team ranked by overall rating, as a diverging bar around league
// average. Each row is a button that opens that team's profile.
function powerRankingsHtml(teams, selected){
const sorted = (teams || []).filter(t => mlbNum(t.Confidence) !== null)
.slice().sort((a, b) => mlbNum(b.Confidence) - mlbNum(a.Confidence))
if(!sorted.length){ return "" }
const reach = Math.max(...sorted.map(t => Math.abs(mlbNum(t.Confidence) - 1)), 0.05)
const picked = new Set(selected || [])
const row = (t, i) => {
const share = mlbNum(t.Confidence) - 1
const width = Math.abs(share) / reach * 50
return `<button class="powerRow${picked.has(t.team) ? " powerPicked" : ""}" data-team="${mlbEsc(t.team)}" data-tip="${mlbEsc(`${mlbTeam(t.team)}: ${mlbOrdinal(i + 1)}, overall rating ${mlbIndex(t.Confidence)} vs. average, record ${teamRecord(t)}`)}">
<span class="rankNo">${i + 1}</span>${mlbLogo(t.team, "xs")}<span class="powerName">${mlbEsc(mlbTeam(t.team))}</span>
<span class="mlbMeter"><span class="mlbMeterMid"></span><span class="mlbMeterBar" style="${share >= 0 ? "left:50%" : "right:50%"};width:${width.toFixed(1)}%;background:${share >= 0 ? "var(--mlb-better)" : "var(--mlb-worse)"}"></span></span>
<span class="powerValue">${mlbIndex(t.Confidence)}</span>
</button>`
}
const half = Math.ceil(sorted.length / 2)
return `<div class="vizCard">
<div class="vizHead"><b>Power rankings</b><span class="vizKey"><i class="keyBetter"></i>Better than average <i class="keyWorse"></i>Worse</span></div>
<div class="powerCols"><div>${sorted.slice(0, half).map(row).join("")}</div><div>${sorted.slice(half).map((t, i) => row(t, i + half)).join("")}</div></div>
</div>`
}

// One team's profile, or two side by side on the same tracks. Each stat
// is placed by its rank among the 30 teams; with two teams their caps sit
// on the track so the gap reads at a glance.
function teamProfileHtml(selected, teams){
const chosen = (selected || []).map(code => (teams || []).find(t => t.team === code)).filter(Boolean)
if(!chosen.length){ return "Select a team" }
const n = (teams || []).length
const overall = t => teamRank(teams, t, "Confidence", 1)
const header = chosen.map(t => `<div class="teamHead">${mlbLogo(t.team, "lg")}<div><div class="teamHeadName">${mlbEsc(mlbTeam(t.team))}</div>
<div class="teamHeadMeta">${teamRecord(t) ? `${teamRecord(t)} · ` : ""}${overall(t) ? `${mlbOrdinal(overall(t))} of ${n} overall` : ""}</div></div></div>`).join(`<span class="teamVs">vs</span>`)
const groups = TEAM_PROFILE_GROUPS.map(([title, stats]) => {
const rows = stats.filter(([key]) => chosen.some(t => mlbNum(t[key]) !== null)).map(([key, label, format, direction]) => {
const neutral = direction === 0
const place = t => { const r = teamRank(teams, t, key, direction || 1); return r === null ? null : Math.round(100 * (n - r) / Math.max(1, n - 1)) }
if(chosen.length === 1){
const t = chosen[0], r = teamRank(teams, t, key, direction || 1)
const rankWord = r === null ? "" : neutral ? `${mlbOrdinal(r)} highest` : `${mlbOrdinal(r)} of ${n}`
// The dot carries the team's rank (for a stat that is neither good nor
// bad, its rank from the highest), so the value column stays short.
return mlbPercentileRow(label, teamStatText(t[key], format), place(t),
{ neutral, dotText: r === null ? "" : r, tip: `${label}: ${teamStatText(t[key], format)} (${rankWord})${TEAM_STAT_NOTES[key] ? ` - ${TEAM_STAT_NOTES[key]}` : ""}` })
}
const markers = chosen.map((t, i) => {
const p = place(t)
return p === null ? "" : `<span class="pctMarker pctMarker${i}" style="left:${p}%">${mlbLogo(t.team, "xs")}</span>`
}).join("")
const values = chosen.map(t => `<span class="pctPair">${mlbLogo(t.team, "xs")}${mlbEsc(teamStatText(t[key], format))}</span>`).join("")
const tip = `${label}: ` + chosen.map(t => `${mlbTeam(t.team)} ${teamStatText(t[key], format)} (${mlbOrdinal(teamRank(teams, t, key, direction || 1) || 0)})`).join(", ")
return `<div class="pctRow" data-tip="${mlbEsc(tip)}" tabindex="0"><span class="pctLabel">${mlbEsc(label)}</span>
<span class="pctTrack pctTrackCompare"><span class="pctMid"></span>${markers}</span><span class="pctValue">${values}</span></div>`
}).join("")
return rows ? `<div class="profileGroup"><div class="profileGroupTitle">${title}</div>${mlbScaleRow("Worst", "Best")}${rows}</div>` : ""
}).join("")
const note = chosen.length === 1 ? `Numbers in the dots are ranks, 1 to ${n}.` : `Each cap sits at that team's rank among the ${n}.`
return `<div class="vizCard teamProfile${chosen.length > 1 ? " teamCompare" : ""}"><div class="teamHeads">${header}</div><div class="vizSub">${note}</div><div class="profileGrid">${groups}</div></div>`
}

// ---- Player lookup ----

const HITTER_PERCENTILE_STATS = [
["Game_Hit_Probability", "CAVE (games with a hit)", "pct", 1],
["WOPS", "WOPS (OPS)", "avg", 1],
["WAVE", "WAVE (hit rate)", "avg", 1],
["xBA", "Expected batting average", "avg", 1],
["xwOBA", "Expected wOBA", "avg", 1],
["Exit_Velo", "Exit velocity", "mph", 1],
["Barrel_Rate", "Barrel rate", "pct1", 1],
["Whiff_Rate", "Swing-and-miss rate", "pct1", -1],
["Chase_Rate", "Chase rate", "pct1", -1],
["DECISION_PLUS", "Decision+ (swing decisions)", "plus", 1],
]

const HITTER_SPLITS = [
["WAVE_L", "vs. lefties"], ["WAVE_R", "vs. righties"], ["WAVE_Home", "At home"], ["WAVE_Away", "On the road"],
["Fastball_WAVE", "vs. fastballs"], ["Breaking_WAVE", "vs. breaking balls"], ["Offspeed_WAVE", "vs. off-speed"],
]

const PITCHER_PERCENTILE_STATS = [
["STUFF_PLUS", "Stuff+", "plus", 1],
["PAVE", "Hit rate allowed", "pct1", -1],
["Power_A_PLUS", "Power allowed", "index", -1],
]

function playerStatText(value, format){
const n = mlbNum(value)
if(n === null){ return "-" }
if(format === "avg"){ return n.toFixed(3).replace(/^0/, "") }
if(format === "mph"){ return `${n.toFixed(1)} mph` }
return teamStatText(n, format)
}

function mlbMean(values){
const list = values.map(mlbNum).filter(n => n !== null)
return list.length ? list.reduce((a, b) => a + b, 0) / list.length : null
}

// Hit-rate splits as dots on one shared scale, each with the league's
// average for that split as a tick.
function hitterSplitsHtml(player, pool){
const rows = HITTER_SPLITS.map(([key, label]) => ({ key, label, value: mlbNum(player[key]), avg: mlbMean((pool || []).map(p => p[key])) }))
.filter(r => r.value !== null)
if(!rows.length){ return "" }
const all = rows.flatMap(r => [r.value, r.avg]).filter(v => v !== null)
const lo = Math.max(0, Math.floor((Math.min(...all) - 0.02) * 20) / 20), hi = Math.ceil((Math.max(...all) + 0.02) * 20) / 20
const pos = v => (100 * (v - lo) / (hi - lo)).toFixed(1)
return `<div class="profileGroup"><div class="profileGroupTitle">Hit rate by situation <span class="vizKey"><i class="keyDot"></i>${mlbEsc(player.name_first)} <i class="keyTick"></i>League average</span></div>
${rows.map(r => {
const better = r.avg === null || r.value >= r.avg
return `<div class="pctRow" data-tip="${mlbEsc(`${r.label}: ${mlbPct(r.value, 1)} (league ${mlbPct(r.avg, 1)})`)}" tabindex="0"><span class="pctLabel">${mlbEsc(r.label)}</span>
<span class="pctTrack splitTrack">${r.avg === null ? "" : `<span class="splitAvg" style="left:${pos(r.avg)}%"></span>`}<span class="splitDot" style="left:${pos(r.value)}%;background:${better ? "var(--mlb-better)" : "var(--mlb-worse)"}"></span></span>
<span class="pctValue">${mlbPct(r.value, 1)}</span></div>`
}).join("")}</div>`
}

const PITCH_TYPE_NAMES = {
FF: "Four-seam", SI: "Sinker", FC: "Cutter", SL: "Slider", ST: "Sweeper", CU: "Curveball", KC: "Knuckle curve",
SV: "Slurve", CH: "Changeup", FS: "Splitter", FO: "Forkball", KN: "Knuckleball", EP: "Eephus", FA: "Fastball", CS: "Slow curve", SC: "Screwball",
}

// Stuff+ for each pitch he throws enough (100+ this season), most-thrown
// first: a bar from the middle, right for better than average.
function stuffByPitchHtml(player){
const pitches = Object.entries(player.STUFF_TYPES || {}).sort((a, b) => b[1].pitches - a[1].pitches)
if(!pitches.length){ return "" }
return `<div class="profileGroup"><div class="profileGroupTitle">Stuff+ by pitch <span class="vizKey">100 = league average for that pitch</span></div>
${pitches.map(([type, g]) => `<div class="pctRow" data-tip="${mlbEsc(`${PITCH_TYPE_NAMES[type] || type}: Stuff+ ${Math.round(g.grade)} (${g.pitches} thrown)`)}" tabindex="0"><span class="pctLabel">${mlbEsc(PITCH_TYPE_NAMES[type] || type)}</span>${mlbMeter((g.grade - 100) / 60, true)}<span class="pctValue">${Math.round(g.grade)}</span></div>`).join("")}</div>`
}

function pitchMixHtml(player){
const mix = [["Fastball_Rate", "Fastballs"], ["Breaking_Rate", "Breaking balls"], ["Offspeed_Rate", "Off-speed"]]
.map(([key, label]) => ({ label, value: mlbNum(player[key]) })).filter(m => m.value !== null)
if(!mix.length){ return "" }
// Pitches the data doesn't sort into these three show as "Other".
const other = 1 - mix.reduce((sum, m) => sum + m.value, 0)
if(other >= 0.02){ mix.push({ label: "Other / unclassified", value: other }) }
return `<div class="profileGroup"><div class="profileGroupTitle">What he throws</div>
${mix.map(m => `<div class="pctRow"><span class="pctLabel">${m.label}</span>${mlbChanceBar(m.value)}<span class="pctValue">${mlbPct(m.value)}</span></div>`).join("")}</div>`
}

// The player card: headshot, team, percentile bars against qualified
// players, splits (hitters) or pitch mix (pitchers), and today's chance.
// `extra` = { todayChance, opponent, isHome, hitStreak }.
function playerCardHtml(player, kind, pool, extra = {}){
const name = `${player.name_first} ${player.name_last}`
const hitter = kind !== "pitcher"
const stats = hitter ? HITTER_PERCENTILE_STATS : PITCHER_PERCENTILE_STATS
const poolName = hitter ? "regular hitters" : "regular pitchers"
const bars = stats.filter(([key]) => mlbNum(player[key]) !== null).map(([key, label, format, direction]) => {
const p = mlbPercentile((pool || []).map(r => r[key]), player[key], direction)
return mlbPercentileRow(label, playerStatText(player[key], format), p,
{ tip: `${label}: ${playerStatText(player[key], format)} - better than ${p}% of ${poolName}` })
}).join("")
const pa = Math.round((mlbNum(player.PA_L) || 0) + (mlbNum(player.PA_R) || 0))
const meta = hitter ? `${pa} plate appearances` : `${player.Throws === "L" ? "Lefty" : player.Throws === "R" ? "Righty" : "Pitcher"} · ${player.at_bats || 0} at-bats faced`
const chips = []
if(extra.todayChance !== undefined && extra.todayChance !== null){
chips.push(`<span class="factChip"><b>${mlbPct(extra.todayChance)}</b> chance of a hit today${extra.opponent ? ` ${extra.isHome ? "vs." : "at"} ${mlbEsc(mlbTeam(extra.opponent))}` : ""}</span>`)
}
if(extra.hitStreak){ chips.push(`<span class="factChip">On a <b>${extra.hitStreak}-game</b> hit streak</span>`) }
if(!hitter && mlbNum(player.PAVE_PLUS) !== null){ chips.push(`<span class="factChip">Allows <b>${mlbEsc(pitcherVsAverage(player.PAVE_PLUS))}</b></span>`) }
return `<div class="vizCard playerCard">
<div class="playerCardHead">${mlbAvatar(player.key_mlbam, name, "xl")}
<div><div class="teamHeadName">${mlbEsc(name)}</div><div class="teamHeadMeta">${mlbLogo(player.team, "xs")} ${mlbEsc(mlbTeam(player.team))} · ${mlbEsc(meta)}</div>
${chips.length ? `<div class="factChips">${chips.join("")}</div>` : ""}</div></div>
<div class="profileGrid"><div class="profileGroup"><div class="profileGroupTitle">Percentile among ${poolName}</div>${mlbScaleRow("Worst", "Best")}${bars || "<span class=\"dim\">Not enough data yet</span>"}</div>
${hitter ? hitterSplitsHtml(player, pool) : pitchMixHtml(player) + stuffByPitchHtml(player)}</div>
</div>`
}

// Search suggestions with headshots.
function playerSuggestionsHtml(matches){
if(!matches.length){ return `<span class="dim">No players found</span>` }
return matches.map((p, i) => `<button class="playerChoice" data-index="${i}">${mlbAvatar(p.key_mlbam, `${p.name_first} ${p.name_last}`, "xs")}<span>${mlbEsc(`${p.name_first} ${p.name_last}`)}</span><small>${mlbEsc(p.team || "")}</small></button>`).join("")
}

// ---- Leaderboards ----

function pitcherLeadersHtml(rows, limit = 15){
if(!rows.length){ return "No data" }
return `<div class="rankList">${rows.slice(0, limit).map((p, i) => {
const name = `${p.name_first} ${p.name_last}`
const ratio = mlbNum(p.PAVE_PLUS)
return `<div class="rankRow" data-tip="${mlbEsc(`${name}: allows ${mlbPct(p.PAVE, 1)} hits - ${pitcherVsAverage(ratio)} (${p.at_bats} at-bats)`)}" tabindex="0">
<span class="rankNo">${i + 1}</span>${mlbAvatar(p.key_mlbam, name, "sm")}
<span class="rankWho"><b>${mlbEsc(name)}</b><small>${mlbLogo(p.team, "xs")} ${mlbEsc(mlbTeam(p.team))} · ${p.Throws === "L" ? "lefty" : "righty"}</small></span>
${ratio === null ? "<span></span>" : mlbMeter(ratio - 1, false)}<span class="rankValue">${mlbPct(p.PAVE)}</span>
</div>`
}).join("")}</div>`
}

// Pitchers by Stuff+ (stuff_plus.csv merged onto pave.csv's rows).
function stuffLeadersHtml(rows, limit = 15){
const ranked = rows.filter(p => mlbNum(p.STUFF_PLUS) !== null).sort((a, b) => mlbNum(b.STUFF_PLUS) - mlbNum(a.STUFF_PLUS))
if(!ranked.length){ return "No data" }
return `<div class="rankList">${ranked.slice(0, limit).map((p, i) => {
const name = `${p.name_first} ${p.name_last}`
const grade = mlbNum(p.STUFF_PLUS)
const best = Object.entries(p.STUFF_TYPES || {}).sort((a, b) => b[1].grade - a[1].grade)[0]
return `<div class="rankRow" data-tip="${mlbEsc(`${name}: Stuff+ ${Math.round(grade)} over ${p.STUFF_PITCHES} pitches${best ? ` - best pitch: ${PITCH_TYPE_NAMES[best[0]] || best[0]} (${Math.round(best[1].grade)})` : ""}`)}" tabindex="0">
<span class="rankNo">${i + 1}</span>${mlbAvatar(p.key_mlbam, name, "sm")}
<span class="rankWho"><b>${mlbEsc(name)}</b><small>${mlbLogo(p.team, "xs")} ${mlbEsc(mlbTeam(p.team))}${best ? ` · best: ${mlbEsc(PITCH_TYPE_NAMES[best[0]] || best[0])}` : ""}</small></span>
${mlbMeter((grade - 100) / 60, true)}<span class="rankValue">${Math.round(grade)}</span>
</div>`
}).join("")}</div>`
}

// ---- Stats page: hitter table ----
// WAVE, CAVE and WOPS are views of a hitter's season, each blended across
// recent windows (hitter_views.py); "vs. pitchers faced" swaps in the
// versions that credit hits off tough pitchers more.

// Chance of at least one hit in 3.5 at-bats at a WAVE (an at-bat hit
// rate): 1 - (1 - WAVE)^3.5.
const HIT_CHANCE_AT_BATS = 3.5

function hitChanceFromWave(wave){
const n = mlbNum(wave)
return n === null ? null : 1 - Math.pow(1 - n, HIT_CHANCE_AT_BATS)
}

// One row per hitter in `waveRows` (names, team) with hitter_stats.csv's
// views, in the chosen view ("season" or "faced").
function hitterStatRows(waveRows, statRows, view = "season"){
const stats = {}
;(statRows || []).forEach(r => { stats[String(r.key_mlbam).replace(/\.0$/, "")] = r })
const suffix = view === "faced" ? "_adj" : ""
return (waveRows || []).map(p => {
const s = stats[String(p.key_mlbam).replace(/\.0$/, "")] || {}
const wave = mlbNum(s[`WAVE${suffix}`]) ?? (view === "faced" ? null : mlbNum(p.WAVE))
return {
key_mlbam: p.key_mlbam, name: `${p.name_first} ${p.name_last}`, team: p.team,
WAVE: wave, hitChance: hitChanceFromWave(wave),
CAVE: mlbNum(s[`CAVE${suffix}`]) ?? (view === "faced" ? null : mlbNum(p.Game_Hit_Probability)),
WOPS: mlbNum(s[`WOPS${suffix}`]),
decision: mlbNum(p.DECISION_PLUS),
// The plain season numbers, to show how far the adjustment moved them.
base: { WAVE: mlbNum(s.WAVE), CAVE: mlbNum(s.CAVE), WOPS: mlbNum(s.WOPS) },
}
})
}

// [key, heading, short heading for phones]
const HITTER_STAT_COLUMNS = [
["name", "Player", "Player"],
["WAVE", "WAVE", "WAVE"],
["hitChance", "Hit chance (3.5 AB)", "Hit %"],
["CAVE", "CAVE", "CAVE"],
["WOPS", "WOPS", "WOPS"],
["decision", "Decision+", "Dec+"],
]

const HITTER_STAT_TIPS = {
WAVE: "WAVE: at-bat hit rate, recent games counting more",
CAVE: "CAVE (Carried Average): share of games with a hit, recent games counting more",
WOPS: "WOPS: OPS, recent games counting more",
decision: "Decision+: his swing-or-take choices against the league's in the same spot (100 = average, 10 = one standard deviation)",
}

function sortHitterStatRows(rows, key = "hitChance"){
const sorted = rows.slice()
if(key === "name"){ return sorted.sort((a, b) => a.name.localeCompare(b.name)) }
return sorted.sort((a, b) => (b[key] ?? -Infinity) - (a[key] ?? -Infinity) || (b.WAVE ?? 0) - (a.WAVE ?? 0))
}

function mlbRate(value){
return value === null || value === undefined ? "-" : value.toFixed(3).replace(/^0/, "")
}

// "+.004" under an adjusted value: how far "vs. pitchers faced" moved it.
function adjustmentNote(value, base, format){
if(value === null || base === null || base === undefined){ return "" }
const delta = value - base
const shown = format === "pct" ? `${Math.abs(Math.round(delta * 1000) / 10).toFixed(1)}%` : Math.abs(delta).toFixed(3).replace(/^0/, "")
if(Number(shown.replace("%", "")) === 0){ return `<small class="adjNote">no change</small>` }
return `<small class="adjNote">${delta > 0 ? "+" : "−"}${shown}</small>`
}

function hitterStatsTableHtml(rows, sortKey = "hitChance", limit = 40, view = "season"){
if(!rows.length){ return "No hitters yet" }
const faced = view === "faced"
const head = HITTER_STAT_COLUMNS.map(([key, label, short]) =>
`<th${key === "name" ? "" : ` class="num"`}${HITTER_STAT_TIPS[key] ? ` data-tip="${mlbEsc(HITTER_STAT_TIPS[key])}"` : ""}><button class="sortHead${sortKey === key ? " sorted" : ""}" data-sort="${key}"><span class="headLong">${label}</span><span class="headShort">${short}</span>${sortKey === key ? " ▼" : ""}</button></th>`).join("")
const body = sortHitterStatRows(rows, sortKey).slice(0, limit).map((r, i) => `<tr>
<td><span class="statWho"><span class="rankNo">${i + 1}</span>${mlbAvatar(r.key_mlbam, r.name, "sm")}<span class="rankWho"><b>${mlbEsc(r.name)}</b><small>${mlbLogo(r.team, "xs")} ${mlbEsc(mlbTeam(r.team))}</small></span></span></td>
<td class="num"><b>${mlbRate(r.WAVE)}</b>${faced ? adjustmentNote(r.WAVE, r.base.WAVE) : ""}</td>
<td class="num"><span class="statBar">${mlbChanceBar(r.hitChance)}<b>${mlbPct(r.hitChance)}</b></span></td>
<td class="num"><b>${r.CAVE === null ? "-" : mlbPct(r.CAVE, 1)}</b>${faced ? adjustmentNote(r.CAVE, r.base.CAVE, "pct") : ""}</td>
<td class="num"><b>${mlbRate(r.WOPS)}</b>${faced ? adjustmentNote(r.WOPS, r.base.WOPS) : ""}</td>
<td class="num"><b>${r.decision === null ? "-" : Math.round(r.decision)}</b></td>
</tr>`).join("")
return `<div class="scrollbox statTableBox"><table class="statTable"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`
}

// ---- Stats page: team schedule sheet ----

// Pen colour for a game: green circles for wins, red crosses for losses,
// from light (won or lost by 1) to deep (by 9 or more).
const SCHEDULE_DEEPEST_MARGIN = 9

function scheduleInk(result, runDiff){
const margin = Math.min(SCHEDULE_DEEPEST_MARGIN, Math.max(1, Math.abs(Number(runDiff) || 1)))
const t = (margin - 1) / (SCHEDULE_DEEPEST_MARGIN - 1)
const mix = (a, b) => a.map((c, i) => Math.round(c + (b[i] - c) * t))
const rgb = result === "W" ? mix([112, 192, 132], [8, 110, 46]) : mix([232, 132, 132], [160, 14, 18])
return `rgb(${rgb.join(",")})`
}

// Each team's games grouped into its series, in order.
function scheduleSeries(games, team){
const series = []
;(games || []).filter(g => g.team === team)
.sort((a, b) => (mlbNum(a.game_no) || 0) - (mlbNum(b.game_no) || 0))
.forEach(g => {
const n = mlbNum(g.series)
let current = series[series.length - 1]
if(!current || current.n !== n){
current = { n, opponent: g.opponent, home: String(g.home) === "1", games: [] }
series.push(current)
}
current.games.push(g)
})
return series
}

function scheduleGameHtml(g){
const win = g.result === "W"
const ink = scheduleInk(g.result, g.run_diff)
const where = String(g.home) === "1" ? "vs." : "at"
const score = `${win ? "W" : "L"} ${g.runs_for}-${g.runs_against}`
const tip = `${mlbShortDate(g.date)} ${where} ${mlbTeam(g.opponent)}: ${score}`
const mark = win
? `<ellipse cx="20" cy="20" rx="16.5" ry="15.5" transform="rotate(-12 20 20)" />`
: `<path d="M7 8 L33 32 M33 7 L8 33" />`
return `<span class="schedGame" data-tip="${mlbEsc(tip)}" tabindex="0" aria-label="${mlbEsc(tip)}">${mlbLogo(g.opponent, "md", true)}<svg viewBox="0 0 40 40" class="schedMark ${win ? "schedWin" : "schedLoss"}" style="stroke:${ink}" aria-hidden="true">${mark}</svg></span>`
}

// "Mar 26–28", or "Mar 30–Apr 1" across a month end.
function seriesDates(first, last){
const a = mlbShortDate(first), b = mlbShortDate(last)
if(first === last){ return a }
return a.split(" ")[0] === b.split(" ")[0] ? `${a}–${b.split(" ")[1]}` : `${a}–${b}`
}

// The whole season on one sheet: series in order, split into four
// columns (panels the unfold animation opens one by one).
function scheduleSheetHtml(games, team){
const series = scheduleSeries(games, team)
if(!series.length){ return `<div class="paperEmpty">No games for this team yet</div>` }
const all = series.flatMap(s => s.games)
const wins = all.filter(g => g.result === "W").length
const diff = all.reduce((sum, g) => sum + (mlbNum(g.run_diff) || 0), 0)
const won = series.filter(s => { const w = s.games.filter(g => g.result === "W").length; return w * 2 > s.games.length }).length
const lost = series.filter(s => { const w = s.games.filter(g => g.result === "W").length; return w * 2 < s.games.length }).length
const perColumn = Math.ceil(series.length / 4)
const columns = [0, 1, 2, 3].map(c => series.slice(c * perColumn, (c + 1) * perColumn))
const panel = (list, c) => `<div class="paperPanel" style="--panel:${c}">${list.map(s =>
`<div class="schedSeries"><span class="schedDates">${seriesDates(s.games[0].date, s.games[s.games.length - 1].date)}<small>${s.home ? "home" : "away"}</small></span><span class="schedGames">${s.games.map(scheduleGameHtml).join("")}</span></div>`
).join("")}</div>`
return `<div class="paperHead">
<div class="paperTeam">${mlbLogo(team, "lg", true)}<div><b>${mlbEsc(mlbTeam(team))}</b><small>${wins}-${all.length - wins} · run differential ${diff > 0 ? "+" : diff < 0 ? "−" : ""}${Math.abs(diff)} · won ${won} of ${series.length} series${lost + won < series.length ? `, split ${series.length - won - lost}` : ""}</small></div></div>
<div class="paperKey"><span>Won by</span>${[1, 5, 9].map(m => `<svg viewBox="0 0 40 40" class="schedMark schedWin" style="stroke:${scheduleInk("W", m)}"><ellipse cx="20" cy="20" rx="16.5" ry="15.5" transform="rotate(-12 20 20)"/></svg><small>${m === 9 ? "9+" : m}</small>`).join("")}
<span>Lost by</span>${[1, 5, 9].map(m => `<svg viewBox="0 0 40 40" class="schedMark schedLoss" style="stroke:${scheduleInk("L", -m)}"><path d="M7 8 L33 32 M33 7 L8 33"/></svg><small>${m === 9 ? "9+" : m}</small>`).join("")}</div>
</div>
<div class="paperPanels">${columns.map(panel).join("")}</div>`
}

// Team buttons for the schedule.
function scheduleTeamPickerHtml(teams, selected){
return (teams || []).slice().sort((a, b) => mlbTeam(a).localeCompare(mlbTeam(b))).map(t =>
`<button class="schedPick${t === selected ? " schedPicked" : ""}" data-sched-team="${mlbEsc(t)}" title="${mlbEsc(mlbTeam(t))}" aria-pressed="${t === selected}">${mlbLogo(t, "md")}<span>${mlbEsc(t)}</span></button>`).join("")
}

// ---- Hover / tap tooltips ----
// Any element with data-tip shows it in one shared tooltip on hover,
// keyboard focus or tap. A data-mark="x%,y%" also places the chart dot
// (.vegasDot) at that point of its plot.
function installVizTips(){
// Once per page, even when two pages' scripts share it (the artifact).
if(window.vizTipsInstalled){ return }
window.vizTipsInstalled = true
const tip = document.createElement("div")
tip.className = "vizTip"
tip.setAttribute("role", "tooltip")
document.body.appendChild(tip)
const show = target => {
const el = target && target.closest ? target.closest("[data-tip]") : null
if(!el){ tip.style.display = "none"; return }
tip.textContent = el.getAttribute("data-tip")
tip.style.display = "block"
const box = el.getBoundingClientRect()
const width = tip.offsetWidth
const left = Math.min(Math.max(8, box.left + box.width / 2 - width / 2), window.innerWidth - width - 8)
const above = box.top - tip.offsetHeight - 8
tip.style.left = `${left + window.scrollX}px`
tip.style.top = `${(above > 8 ? above : box.bottom + 8) + window.scrollY}px`
const mark = el.getAttribute("data-mark")
const plot = el.parentElement
const dot = plot && plot.querySelector(".vegasDot")
if(mark && dot){
const [x, y] = mark.split(",")
dot.style.left = `${x}%`
dot.style.top = `${y}%`
dot.style.display = "block"
}
}
const hide = () => {
tip.style.display = "none"
document.querySelectorAll(".vegasDot").forEach(d => { d.style.display = "none" })
}
document.addEventListener("mouseover", e => show(e.target))
document.addEventListener("focusin", e => show(e.target))
document.addEventListener("click", e => { if(e.target.closest && e.target.closest("[data-tip]")){ show(e.target) } })
document.addEventListener("mouseout", e => { if(!e.relatedTarget || !e.relatedTarget.closest || !e.relatedTarget.closest("[data-tip]")){ hide() } })
document.addEventListener("focusout", hide)
window.addEventListener("scroll", hide, { passive: true })
}

if (typeof module !== "undefined" && module.exports) {
module.exports = {
mlbEsc, mlbNum, mlbPct, mlbTeam, mlbScaleRow, mlbMeter, mlbChanceBar, installVizTips,
mlbHeadshotUrl, mlbLogoUrl, mlbAvatar, mlbLogo, mlbInitials, mlbIndex, mlbOrdinal, mlbShortDate,
mlbPercentile, mlbPercentileColor, mlbPercentileRow,
streakChartHtml, btsPickCardsHtml, btsLogHtml, hitStreakListHtml, bestChancesHtml,
pitcherVsAverage, pitcherDuelsHtml,
gamePickCardHtml, gamesVersusVegas, versusVegasChartHtml, confidenceBucketsHtml, gameHistoryHtml,
TEAM_PROFILE_GROUPS, teamRank, teamRecord, teamStatText, powerRankingsHtml, teamProfileHtml,
playerCardHtml, playerSuggestionsHtml, hitterSplitsHtml, pitcherLeadersHtml, stuffLeadersHtml, stuffByPitchHtml,
hitChanceFromWave, hitterStatRows, sortHitterStatRows, hitterStatsTableHtml, adjustmentNote,
scheduleInk, scheduleSeries, scheduleSheetHtml, scheduleTeamPickerHtml,
}
}
