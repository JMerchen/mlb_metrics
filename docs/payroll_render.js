// Payroll value - the MLB Payroll page (payroll.html) and the NFL page's
// Payroll tab: who gets the most winning out of each dollar. Pure functions
// returning HTML strings (and the workbook's sheets) so node tests can check
// them; payroll.js loads the CSVs and wires the DOM. Each takes a sport
// (PAY_MLB, the default, or PAY_NFL) last.
//
// The data comes from scripts/build_payroll_value.py and
// build_nfl_payroll_value.py (payroll_value.py has the definitions). Colour runs green (more value) through grey to red
// (less value, overspending), by percentile within what's on screen -
// checked with the dataviz palette validator on --bg-card (colour-blind
// separation ΔE 16), and every coloured mark carries its number. The
// downloaded workbook keeps the original's red-yellow-green.

// What differs between the sports. teamOrder: division order (MLB's as the
// original workbook's grids had it); oldNames: [code, last season, name]
// for nicknames a franchise went by before; yRange/yStep: the wins-per-
// season axes' usual span and grid step.
const PAY_MLB = {
key: "mlb", games: 162, perLabel: "wins per 162", since: 1999,
teamOrder: ["NYY", "BOS", "TB", "BAL", "TOR", "KC", "MIN", "DET", "CWS", "CLE", "TEX", "HOU", "SEA", "ATH",
"LAA", "CHC", "STL", "MIL", "CIN", "PIT", "PHI", "ATL", "NYM", "MIA", "WSH", "LAD", "COL", "SD", "SF", "AZ"],
oldNames: [["WSH", 2004, "Expos"], ["TB", 2007, "Devil Rays"], ["CLE", 2021, "Indians"]],
name: code => mlbTeam(code),
logo: (code, size) => mlbLogo(code, size),
record: r => `${r.wins}-${r.losses}`,
payrollTip: "Opening-day payroll (players under contract on opening day), from The Baseball Cube",
comparableRule: "other teams, 11+ years earlier",
noComparables: "No seasons 11 or more years before",
yRange: [50, 110], yStep: 10, yearStep: 5,
workbook: "Payroll_Effectiveness.xlsx",
files: { seasons: "./data/payroll_seasons.csv", franchises: "./data/payroll_franchises.csv", league: "./data/payroll_league.csv" },
measures: null,
franchiseNote: "Averages over every final season since 1988 (each franchise's seasons under its earlier names included).",
leagueNote: "The average payroll over 81 wins - what an average team paid per win, as if every season were 162 games (2020 played 60; the 1994 and 1995 seasons were cut short by the strike).",
}
const PAY_NFL = {
key: "nfl", games: 17, perLabel: "wins per 17 games", since: 2015,
teamOrder: ["BUF", "MIA", "NE", "NYJ", "BAL", "CIN", "CLE", "PIT", "HOU", "IND", "JAX", "TEN", "DEN", "KC", "LV", "LAC",
"DAL", "NYG", "PHI", "WAS", "CHI", "DET", "GB", "MIN", "ATL", "CAR", "NO", "TB", "ARI", "LA", "SF", "SEA"],
oldNames: [["WAS", 2019, "Redskins"], ["WAS", 2021, "Football Team"]],
name: code => (typeof teamName === "function" ? teamName(code, "nfl") : code),
logo: (code, size) => (typeof nflLogo === "function" ? nflLogo(code, size) : ""),
// Wins and losses count a tie as half of each; show W-L-T.
record: r => {
const t = mlbNum(r.ties) || 0
return t ? `${mlbNum(r.wins) - t / 2}-${mlbNum(r.losses) - t / 2}-${t}` : `${mlbNum(r.wins)}-${mlbNum(r.losses)}`
},
payrollTip: "Cash paid or cap charges that season (see the switch above), summed from OverTheCap's player contracts - dead money not included",
comparableRule: "other teams, any earlier season",
noComparables: "No earlier seasons (the data starts in 2014) before",
yRange: [4, 13], yStep: 2, yearStep: 2,
workbook: "NFL_Payroll_Effectiveness.xlsx",
files: { seasons: "./data/nfl_payroll_seasons.csv", franchises: "./data/nfl_payroll_franchises.csv", league: "./data/nfl_payroll_league.csv" },
measures: [["cash", "Cash paid"], ["cap", "Cap spending"]],
franchiseNote: "Averages over every final season since 2014.",
leagueNote: "The average spend over 8.5 wins - what an average team paid per win, as if every season were 17 games (16 until 2021).",
}

const PAY_GOOD = [71, 171, 79], PAY_MID = [91, 98, 112], PAY_BAD = [165, 50, 52]

// Fill for a percentile (100 = best value), and the ink that reads on it.
function payFill(p){
if(p === null){ return PAY_MID }
const mix = (a, b, t) => a.map((c, i) => Math.round(c + (b[i] - c) * t))
return p >= 50 ? mix(PAY_MID, PAY_GOOD, (p - 50) / 50) : mix(PAY_BAD, PAY_MID, p / 50)
}

function payColor(p){
return `rgb(${payFill(p).join(",")})`
}

// Dark ink on the light greens, white elsewhere (whichever contrasts more).
function payInk(p){
const lum = payFill(p).map(c => { c /= 255; return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4 })
const L = 0.2126 * lum[0] + 0.7152 * lum[1] + 0.0722 * lum[2]
return (L + 0.05) / 0.0548 > 1.05 / (L + 0.05) ? "#0b0d10" : "#fff"
}

// The view's sections; ids are `P` + a name (payroll.js fills them), so
// the MLB page and the NFL tab can share one page in the artifact.
function payViewHtml(S, P){
const measures = S.measures ? `<span class="tabGroup payMeasures" id="${P}Measures" role="group" aria-label="Spending measure">${
S.measures.map(([key, label], i) => `<button class="tabButton${i ? "" : " active"}" data-pay-measure="${key}">${label}</button>`).join("")}</span>` : ""
return `<div class="lookupControls payControls">
<label class="payField">Season <select id="${P}SeasonPick" aria-label="Season"></select></label>
${measures}
<button class="tabButton" id="${P}Download" data-tip="Every season, the Front Office model and the year grids, as an Excel workbook">Download workbook (.xlsx)</button>
</div>
<div id="${P}Hero">Loading...</div>
<div class="vizCard" id="${P}Scatter"></div>
<h3 id="${P}SeasonTitle">Season</h3>
<p class="sectionIntro">Click a column to sort, or a team to see its history.</p>
<div id="${P}Table"></div>
<h3 id="${P}TeamSection">Team history</h3>
<div class="lookupControls"><label class="payField">Team <select id="${P}TeamPick" aria-label="Team"></select></label></div>
<div class="vizCard" id="${P}TeamChart"></div>
<div id="${P}Neighbors"></div>
<h3>All-time value by franchise</h3>
<p class="sectionIntro">${S.franchiseNote}</p>
<div id="${P}Franchises"></div>
<h3>Year by year</h3>
<p class="sectionIntro">Every final season at a glance - by team, or by where a payroll or record ranked that year. Green is the better end of each grid, red the worse.</p>
<div id="${P}GridTabs"></div>
<div id="${P}Grid"></div>
<h3>The price of a win</h3>
<p class="sectionIntro">${S.leagueNote}</p>
<div class="vizCard" id="${P}League"></div>`
}

function payTeamName(code, season, S = PAY_MLB){
const old = S.oldNames.find(([c, last]) => c === code && Number(season) <= last)
return old ? old[2] : S.name(code)
}

function payMoney(value){
const n = mlbNum(value)
if(n === null){ return "-" }
return n >= 1e9 ? `$${(n / 1e9).toFixed(2)}B` : n >= 1e6 ? `$${(n / 1e6).toFixed(1)}M` : `$${Math.round(n / 1e3)}K`
}

function payRatio(value, digits = 2){
const n = mlbNum(value)
return n === null ? "-" : n.toFixed(digits)
}

function paySigned(value, digits = 1){
const n = mlbNum(value)
if(n === null){ return "-" }
const shown = Math.abs(n).toFixed(digits)
return Number(shown) === 0 ? (0).toFixed(digits) : `${n > 0 ? "+" : "−"}${shown}`
}

// Expected wins: whole for baseball, a decimal for football's 17 games.
function payWins(value, S = PAY_MLB){
const n = mlbNum(value)
return n === null ? "-" : S.games > 100 ? String(Math.round(n)) : n.toFixed(1)
}

function payRecord(r, S = PAY_MLB){
return mlbNum(r.wins) === null ? "-" : S.record(r)
}

// Wins over a full season (MLB 2020 played 60, 1994-95 were cut short; the
// NFL played 16 until 2021).
function payPerSeason(r, key = "win_avg", S = PAY_MLB){
const v = mlbNum(r[key])
return v === null ? null : v * S.games / 2
}

function paySeasonRows(rows, season){
return (rows || []).filter(r => Number(r.season) === Number(season))
}

function paySeasons(rows){
return [...new Set((rows || []).map(r => Number(r.season)))].sort((a, b) => b - a)
}

// The season the page opens on: the latest with games played.
function payDefaultSeason(rows){
const played = (rows || []).filter(r => r.status !== "forecast").map(r => Number(r.season))
return played.length ? Math.max(...played) : (paySeasons(rows)[0] ?? null)
}

// [key, label, short label, higher is better (null: neither), tip]
const payColumns = S => [
["team", "Team", "Team", null, ""],
["payroll", "Payroll", "Pay", null, S.payrollTip],
["payroll_rank", "Payroll rank", "Pay #", null, "1 = the biggest payroll that season"],
["wins", "Record", "W-L", true, ""],
["wins_rank", "Wins rank", "W #", false, "1 = the most wins that season"],
["pe_index", "Payroll effectiveness", "PE", true, "Wins over average divided by payroll per win over the league's. Above 1 bought more winning per dollar than the league did"],
["expected_wins", "Expected wins", "xW", null, `What the 10 seasons that spent most like this one won (${S.comparableRule}), plus this front office's recent edge`],
["wins_above_expected", "Wins vs. expected", "+/-", true, "Wins minus expected wins"],
["fo_score", "Front office score", "FO", true, "How much this front office beat its money over the previous three seasons (in wins over average)"],
["cash_flow", "Spending change", "Cash", null, "Payroll next to the league's, minus its average over the previous three seasons. Positive spent up"],
]

function paySortRows(rows, key, S = PAY_MLB){
const value = r => key === "team" ? payTeamName(r.franchise, r.season, S) : mlbNum(r[key])
const ascending = key === "team" || key === "payroll_rank" || key === "wins_rank"
return rows.slice().sort((a, b) => {
const va = value(a), vb = value(b)
if(va === null || vb === null){ return (va === null) - (vb === null) }
if(typeof va === "string"){ return va.localeCompare(vb) }
return ascending ? va - vb : vb - va
})
}

// A coloured chip for a value, by its percentile in `pool`.
function payChip(value, pool, text, higherIsBetter = true){
const p = mlbPercentile(pool, value, higherIsBetter ? 1 : -1)
return `<span class="payChip" style="background:${payColor(p)};color:${payInk(p)}">${mlbEsc(text)}</span>`
}

function paySeasonTableHtml(rows, sortKey = "pe_index", selected = null, S = PAY_MLB){
if(!rows.length){ return "No payrolls for this season yet" }
const forecast = rows.every(r => r.status === "forecast")
const all = payColumns(S)
const cols = forecast ? all.filter(c => ["team", "payroll", "payroll_rank", "expected_wins", "fo_score", "cash_flow"].includes(c[0])) : all
const pool = key => rows.map(r => r[key])
const head = cols.map(([key, label, short, , tip]) =>
`<th${key === "team" ? "" : ` class="num"`}${tip ? ` data-tip="${mlbEsc(tip)}"` : ""}><button class="sortHead${sortKey === key ? " sorted" : ""}" data-pay-sort="${key}"><span class="headLong">${label}</span><span class="headShort">${short}</span>${sortKey === key ? " ▼" : ""}</button></th>`).join("")
const cell = (key, r) => {
switch(key){
case "team": return `<td><button class="payTeamBtn" data-pay-team="${mlbEsc(r.franchise)}">${S.logo(r.franchise, "sm")}<b>${mlbEsc(payTeamName(r.franchise, r.season, S))}</b></button></td>`
case "payroll": return `<td class="num">${payMoney(r.payroll)}</td>`
case "payroll_rank": case "wins_rank": return `<td class="num">${mlbNum(r[key]) === null ? "-" : mlbOrdinal(Number(r[key]))}</td>`
case "wins": return `<td class="num">${payRecord(r, S)}</td>`
case "pe_index": return `<td class="num">${mlbNum(r.pe_index) === null ? "-" : payChip(r.pe_index, pool("pe_index"), payRatio(r.pe_index))}</td>`
case "expected_wins": return `<td class="num">${payWins(r.expected_wins, S)}</td>`
case "wins_above_expected": return `<td class="num">${mlbNum(r.wins_above_expected) === null ? "-" : payChip(r.wins_above_expected, pool("wins_above_expected"), paySigned(r.wins_above_expected))}</td>`
case "fo_score": return `<td class="num">${paySigned(mlbNum(r.fo_score) === null ? null : r.fo_score * S.games / 2)}</td>`
case "cash_flow": return `<td class="num">${paySigned(r.cash_flow, 2)}</td>`
}
}
const body = paySortRows(rows, sortKey, S).map(r => `<tr${r.franchise === selected ? ` class="paySelected"` : ""}>${cols.map(c => cell(c[0], r)).join("")}</tr>`).join("")
return `<div class="scrollbox statTableBox"><table class="statTable payTable"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`
}

// Headline tiles for a season: best value, most over expected, the price
// of a win, and how far off expected wins usually are.
function payHeroHtml(rows, league, season, modelMiss, S = PAY_MLB){
const played = rows.filter(r => mlbNum(r.pe_index) !== null)
const lg = (league || []).find(l => Number(l.season) === Number(season))
const tiles = []
const top = (key, sign = 1) => played.filter(r => mlbNum(r[key]) !== null).sort((a, b) => sign * (b[key] - a[key]))[0]
const best = top("pe_index"), over = top("wins_above_expected"), worst = top("pe_index", -1)
const tile = (label, value, sub, team) => `<div class="payTile">${team ? S.logo(team, "md") : ""}<div><span class="payTileLabel">${label}</span><b class="payTileValue">${value}</b><small>${sub}</small></div></div>`
if(best){ tiles.push(tile("Best value", payTeamName(best.franchise, season, S), `${payRatio(best.pe_index)} effectiveness · ${payRecord(best, S)} on ${payMoney(best.payroll)}`, best.franchise)) }
if(over){ tiles.push(tile("Most above expected", payTeamName(over.franchise, season, S), `${paySigned(over.wins_above_expected)} wins (expected ${payWins(over.expected_wins, S)})`, over.franchise)) }
if(worst){ tiles.push(tile("Least value", payTeamName(worst.franchise, season, S), `${payRatio(worst.pe_index)} effectiveness · ${payRecord(worst, S)} on ${payMoney(worst.payroll)}`, worst.franchise)) }
if(lg && mlbNum(lg.price_per_win_full_season) !== null){ tiles.push(tile("Price of a win", payMoney(lg.price_per_win_full_season), `average payroll ${payMoney(lg.average_payroll)} ÷ ${S.games / 2} wins`)) }
const note = modelMiss === null ? "" : `<p class="payNote">Expected wins miss by ${modelMiss.toFixed(1)} wins a season on average (every final season since ${S.since}).</p>`
return `<div class="payTiles">${tiles.join("")}</div>${note}`
}

// Mean absolute miss of expected wins per full season, final seasons.
function payModelMiss(rows, S = PAY_MLB){
const misses = (rows || []).filter(r => r.status === "final" && mlbNum(r.wins_above_expected) !== null)
.map(r => Math.abs(r.wins_above_expected) / r.games * S.games)
return misses.length ? misses.reduce((a, b) => a + b, 0) / misses.length : null
}

// Relative payroll across, wins per full season up. The curve is break-
// even value: effectiveness = (wins over average)^2 / relative payroll, so
// a team on an average season's wins x sqrt(relative payroll) got exactly
// the league's price per win.
function payScatterSvg(rows, selected = null, width = 720, S = PAY_MLB){
const avg = S.games / 2
const pts = rows.filter(r => mlbNum(r.relative_payroll) !== null && payPerSeason(r, "win_avg", S) !== null)
if(!pts.length){ return "" }
const H = 340, left = 44, right = 16, top = 14, bottom = 38
const W = Math.max(300, width)
const maxX = Math.max(2, ...pts.map(r => Number(r.relative_payroll))) * 1.05
const ys = pts.map(r => payPerSeason(r, "win_avg", S))
const pad = S.yStep * 0.3
const minY = Math.min(S.yRange[0], ...ys) - pad, maxY = Math.max(S.yRange[1], ...ys) + pad
const x = v => left + (W - left - right) * v / maxX
const y = v => top + (H - top - bottom) * (1 - (v - minY) / (maxY - minY))
const f = n => n.toFixed(1)
const ticks = []
for(let v = Math.ceil(minY / S.yStep) * S.yStep; v <= maxY; v += S.yStep){
ticks.push(`<line x1="${left}" x2="${W - right}" y1="${f(y(v))}" y2="${f(y(v))}" class="payGrid"/><text x="${left - 6}" y="${f(y(v) + 4)}" class="payTick" text-anchor="end">${v}</text>`)
}
for(let v = 0.5; v <= maxX; v += 0.5){
ticks.push(`<text x="${f(x(v))}" y="${H - 20}" class="payTick" text-anchor="middle">${v.toFixed(1)}×</text>`)
}
const curve = []
for(let i = 1; i <= 60; i++){
const v = maxX * i / 60, w = avg * Math.sqrt(v)
if(w >= minY && w <= maxY){ curve.push(`${curve.length ? "L" : "M"}${f(x(v))},${f(y(w))}`) }
}
const avgLine = `<line x1="${f(x(1))}" x2="${f(x(1))}" y1="${top}" y2="${H - bottom}" class="payAvgLine"/>`
const pool = pts.map(r => r.pe_index)
const ranked = pts.slice().sort((a, b) => b.pe_index - a.pe_index)
const labelled = new Set([...ranked.slice(0, 3), ...ranked.slice(-3)].map(r => r.franchise).concat(selected ? [selected] : []))
const dots = pts.map(r => {
const p = mlbPercentile(pool, r.pe_index)
const cx = x(Number(r.relative_payroll)), cy = y(payPerSeason(r, "win_avg", S))
const name = payTeamName(r.franchise, r.season, S)
const tip = `${name}: ${payRecord(r, S)} on ${payMoney(r.payroll)} (${payRatio(r.relative_payroll)}× the average payroll) - effectiveness ${payRatio(r.pe_index)}${mlbNum(r.expected_wins) === null ? "" : `, expected ${payWins(r.expected_wins, S)} wins`}`
const label = labelled.has(r.franchise) ? `<text x="${f(cx + 9)}" y="${f(cy + 4)}" class="payDotLabel">${mlbEsc(r.franchise)}</text>` : ""
return `<g data-tip="${mlbEsc(tip)}" data-pay-team="${mlbEsc(r.franchise)}" tabindex="0" class="payDot${r.franchise === selected ? " paySelectedDot" : ""}"><circle cx="${f(cx)}" cy="${f(cy)}" r="12" class="payHit"/><circle cx="${f(cx)}" cy="${f(cy)}" r="6" style="fill:${payColor(p)}"/>${label}</g>`
}).join("")
const summary = `${pts.length} teams' ${S.perLabel} against payroll next to the league average, with the break-even value curve`
return `<svg class="payChart" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="${mlbEsc(summary)}">
${ticks.join("")}
<text x="${W - right}" y="${H - 4}" class="payTick" text-anchor="end">payroll vs. the league average →</text>
<text x="4" y="${top - 2}" class="payTick">${S.perLabel}</text>
${avgLine}<path d="${curve.join("")}" class="payCurve"/>
<text x="${f(x(maxX * 0.98))}" y="${f(y(Math.min(maxY - pad, avg * Math.sqrt(maxX * 0.98))) - 6)}" class="payTick" text-anchor="end">break-even value</text>
${dots}
</svg>`
}

// One franchise's seasons: actual and expected wins per full season by year.
function payTeamChartSvg(rows, franchise, width = 720, S = PAY_MLB){
const avg = S.games / 2
const own = rows.filter(r => r.franchise === franchise).sort((a, b) => a.season - b.season)
if(!own.length){ return "" }
const H = 260, left = 44, right = 70, top = 14, bottom = 30
const W = Math.max(300, width)
const first = Number(own[0].season), last = Number(own[own.length - 1].season)
const actual = own.filter(r => payPerSeason(r, "win_avg", S) !== null).map(r => ({ season: Number(r.season), v: payPerSeason(r, "win_avg", S), r }))
const expected = own.filter(r => payPerSeason(r, "expected_win_avg", S) !== null).map(r => ({ season: Number(r.season), v: payPerSeason(r, "expected_win_avg", S), r }))
const all = actual.concat(expected).map(p => p.v)
const pad = S.yStep * 0.3
const minY = Math.min(S.yRange[0], ...all) - pad, maxY = Math.max(S.yRange[1], ...all) + pad
const x = s => left + (W - left - right) * (s - first) / Math.max(1, last - first)
const y = v => top + (H - top - bottom) * (1 - (v - minY) / (maxY - minY))
const f = n => n.toFixed(1)
const ticks = []
for(let v = Math.ceil(minY / S.yStep) * S.yStep; v <= maxY; v += S.yStep){
ticks.push(`<line x1="${left}" x2="${W - right}" y1="${f(y(v))}" y2="${f(y(v))}" class="payGrid"/><text x="${left - 6}" y="${f(y(v) + 4)}" class="payTick" text-anchor="end">${v}</text>`)
}
const step = last - first > 20 ? 5 : 2
for(let s = Math.ceil(first / step) * step; s <= last; s += step){
ticks.push(`<text x="${f(x(s))}" y="${H - 10}" class="payTick" text-anchor="middle">${s}</text>`)
}
const path = pts => {
let d = "", prev = null
pts.forEach(p => { d += `${prev !== null && p.season === prev + 1 ? "L" : "M"}${f(x(p.season))},${f(y(p.v))}`; prev = p.season })
return d
}
const dots = actual.map(p => {
const tip = `${p.season} ${payTeamName(franchise, p.season, S)}: ${payRecord(p.r, S)}${mlbNum(p.r.expected_wins) === null ? "" : `, expected ${payWins(p.r.expected_wins, S)}`} · ${payMoney(p.r.payroll)} (${mlbOrdinal(Number(p.r.payroll_rank))} in payroll) · effectiveness ${payRatio(p.r.pe_index)}`
return `<g data-tip="${mlbEsc(tip)}" data-pay-season="${p.season}" tabindex="0"><circle cx="${f(x(p.season))}" cy="${f(y(p.v))}" r="9" class="payHit"/><circle cx="${f(x(p.season))}" cy="${f(y(p.v))}" r="3.5" class="payActualDot"/></g>`
}).join("")
const endLabel = (pts, cls, text) => {
const p = pts[pts.length - 1]
return p ? `<text x="${f(x(p.season) + 6)}" y="${f(y(p.v) + 4)}" class="payTick ${cls}">${text}</text>` : ""
}
return `<svg class="payChart" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="${mlbEsc(`${S.name(franchise)} ${S.perLabel} and expected, by season`)}">
${ticks.join("")}
<line x1="${left}" x2="${W - right}" y1="${f(y(avg))}" y2="${f(y(avg))}" class="payAvgLine"/>
<path d="${path(expected)}" class="payExpectedLine"/><path d="${path(actual)}" class="payActualLine"/>
${dots}${endLabel(actual, "payActualLabel", "Wins")}${endLabel(expected, "", "Expected")}
</svg>`
}

// "The 10 seasons that spent most like it" - the Front Office model's
// comparison for one team-season.
function payNeighborsHtml(rows, r, S = PAY_MLB){
if(!r){ return "" }
const name = `${r.season} ${payTeamName(r.franchise, r.season, S)}`
if(!r.neighbors){ return `<p class="payNote">${S.noComparables} ${mlbEsc(name)} to compare with, so no expected wins.</p>` }
const bySeason = new Map(rows.map(x => [`${x.season} ${x.franchise}`, x]))
const chips = String(r.neighbors).split(";").map(key => {
const n = bySeason.get(key)
if(!n){ return "" }
const tip = `${n.season} ${payTeamName(n.franchise, n.season, S)}: ${payRecord(n, S)} on ${payRatio(n.relative_payroll)}× the average payroll`
const v = payPerSeason(n, "win_avg", S)
return `<span class="payNeighbor" data-tip="${mlbEsc(tip)}" tabindex="0">${S.logo(n.franchise, "xs")}${n.season} ${mlbEsc(n.franchise)} <b>${S.games > 100 ? Math.round(v) : v.toFixed(1)}</b></span>`
}).join("")
const base = mlbNum(r.baseline_win_avg) * S.games / 2
const fo = mlbNum(r.fo_score) * S.games / 2
return `<div class="payNeighbors"><p class="payNote">${mlbEsc(name)} spent ${payRatio(r.relative_payroll)}× the average payroll. The 10 seasons that spent most like that averaged ${base.toFixed(1)} ${S.perLabel} (each below)${
fo ? `; this front office's last three seasons add ${paySigned(fo)}` : ""} → <b>${payWins(r.expected_wins, S)} expected wins</b>${mlbNum(r.wins) === null ? "" : `, actual ${r.wins}`}.</p>${chips}</div>`
}

function payFranchiseTableHtml(rows, S = PAY_MLB){
if(!rows.length){ return "" }
const pool = key => rows.map(r => r[key])
const head = `<tr><th>Team</th><th class="num">Seasons</th><th class="num" data-tip="Average Payroll effectiveness index over every season">Avg. effectiveness</th><th class="num" data-tip="1 = the biggest payroll">Avg. payroll rank</th><th class="num" data-tip="1 = the most wins">Avg. wins rank</th><th class="num" data-tip="Average wins minus expected wins, seasons since ${S.since}">Avg. vs. expected</th><th class="num">Record</th></tr>`
const body = rows.map((r, i) => `<tr><td><button class="payTeamBtn" data-pay-team="${mlbEsc(r.franchise)}"><span class="rankNo">${i + 1}</span>${S.logo(r.franchise, "sm")}<b>${mlbEsc(S.name(r.franchise))}</b></button></td>
<td class="num">${r.seasons}</td>
<td class="num">${payChip(r.average_effectiveness, pool("average_effectiveness"), payRatio(r.average_effectiveness))}</td>
<td class="num">${payRatio(r.average_payroll_rank, 1)}</td>
<td class="num">${payRatio(r.average_wins_rank, 1)}</td>
<td class="num">${mlbNum(r.average_wins_above_expected) === null ? "-" : payChip(r.average_wins_above_expected, pool("average_wins_above_expected"), paySigned(r.average_wins_above_expected))}</td>
<td class="num">${payRecord(r, S)}</td></tr>`).join("")
return `<div class="scrollbox statTableBox"><table class="statTable payTable"><thead>${head}</thead><tbody>${body}</tbody></table></div>`
}

// The original workbook's twelve year grids: [key, rows by, measure,
// label, higher is better].
const PAY_GRIDS = [
["team_pe", "franchise", "pe_index", "Effectiveness", true],
["team_fo", "franchise", "fo_score", "Front office score", true],
["team_cash", "franchise", "cash_flow", "Spending change", true],
["team_wae", "franchise", "wins_above_expected", "Wins vs. expected", true],
["team_payrank", "franchise", "payroll_rank", "Payroll rank", true],
["team_winrank", "franchise", "wins_rank", "Wins rank", false],
["payrank_pe", "payroll_rank", "pe_index", "Effectiveness", true],
["payrank_wae", "payroll_rank", "wins_above_expected", "Wins vs. expected", true],
["payrank_fo", "payroll_rank", "fo_score", "Front office score", true],
["payrank_winrank", "payroll_rank", "wins_rank", "Wins rank", false],
["winrank_pe", "wins_rank", "pe_index", "Effectiveness", true],
["winrank_fo", "wins_rank", "fo_score", "Front office score", true],
]
const PAY_GRID_ROWS = [["franchise", "By team"], ["payroll_rank", "By payroll rank"], ["wins_rank", "By wins rank"]]

// {columns, years, cell(year, column) -> mean or null}: final seasons,
// tied ranks averaged.
function payGrid(rows, rowsBy, measure, S = PAY_MLB){
const final = (rows || []).filter(r => r.status === "final")
const sums = new Map()
final.forEach(r => {
const v = mlbNum(r[measure]), c = rowsBy === "franchise" ? r.franchise : mlbNum(r[rowsBy])
if(v === null || c === null){ return }
const key = `${r.season}|${c}`
const s = sums.get(key) || [0, 0]
s[0] += v; s[1] += 1
sums.set(key, s)
})
const columns = rowsBy === "franchise" ? S.teamOrder.slice() : Array.from({ length: S.teamOrder.length }, (_, i) => i + 1)
const years = [...new Set(final.map(r => Number(r.season)))].sort((a, b) => a - b)
const cell = (year, col) => { const s = sums.get(`${year}|${col}`); return s ? s[0] / s[1] : null }
return { columns, years, cell }
}

function payGridValue(v, measure, S = PAY_MLB){
if(v === null){ return "" }
if(measure === "payroll_rank" || measure === "wins_rank"){ return Number.isInteger(v) ? String(v) : v.toFixed(1) }
if(measure === "wins_above_expected"){ return paySigned(v, S.games > 100 ? 0 : 1) }
if(measure === "fo_score"){ return paySigned(v * S.games / 2, S.games > 100 ? 0 : 1) }
return v.toFixed(2)
}

function payGridHtml(rows, key, S = PAY_MLB){
const spec = PAY_GRIDS.find(g => g[0] === key) || PAY_GRIDS[0]
const [, rowsBy, measure, label, higher] = spec
const g = payGrid(rows, rowsBy, measure, S)
const values = []
g.years.forEach(y => g.columns.forEach(c => { const v = g.cell(y, c); if(v !== null){ values.push(v) } }))
const colName = c => rowsBy === "franchise" ? c : mlbOrdinal(c)
const head = `<tr><th>Year</th>${g.columns.map(c => `<th class="num">${rowsBy === "franchise" ? S.logo(c, "xs") : ""}<span>${mlbEsc(colName(c))}</span></th>`).join("")}</tr>`
const body = g.years.slice().reverse().map(y => `<tr><th>${y}</th>${g.columns.map(c => {
const v = g.cell(y, c)
if(v === null){ return `<td></td>` }
const p = mlbPercentile(values, v, higher ? 1 : -1)
const who = rowsBy === "franchise" ? `${y} ${payTeamName(c, y, S)}` : `${y}, ${mlbOrdinal(c)} in ${rowsBy === "payroll_rank" ? "payroll" : "wins"}`
return `<td class="payCell" style="background:${payColor(p)};color:${payInk(p)}" data-tip="${mlbEsc(`${who}: ${label.toLowerCase()} ${payGridValue(v, measure, S)}`)}">${payGridValue(v, measure, S)}</td>`
}).join("")}</tr>`).join("")
return `<div class="scrollbox payGridBox"><table class="payGridTable"><thead>${head}</thead><tbody>${body}</tbody></table></div>`
}

function payGridTabsHtml(key){
const spec = PAY_GRIDS.find(g => g[0] === key) || PAY_GRIDS[0]
const rowTabs = PAY_GRID_ROWS.map(([by, label]) => {
const first = PAY_GRIDS.find(g => g[1] === by)
return `<button class="tabButton${spec[1] === by ? " active" : ""}" data-pay-grid="${first[0]}">${label}</button>`
}).join("")
const measureTabs = PAY_GRIDS.filter(g => g[1] === spec[1]).map(g =>
`<button class="tabButton${g[0] === spec[0] ? " active" : ""}" data-pay-grid="${g[0]}">${g[3]}</button>`).join("")
return `<div class="tabGroup">${rowTabs}</div><div class="tabGroup payMeasureTabs">${measureTabs}</div>`
}

// League price of a win by season: the average payroll over an average
// team's wins in a full season (81 of 162, 8.5 of 17). Seasons with games
// played.
function payLeagueChartSvg(league, width = 720, S = PAY_MLB){
const pts = (league || []).filter(l => mlbNum(l.wins) !== null && mlbNum(l.price_per_win_full_season) !== null).map(l => ({ season: Number(l.season), v: Number(l.price_per_win_full_season) }))
if(pts.length < 2){ return "" }
const H = 220, left = 56, right = 16, top = 14, bottom = 30
const W = Math.max(300, width)
const first = pts[0].season, last = pts[pts.length - 1].season
const maxY = Math.max(...pts.map(p => p.v)) * 1.08
const x = s => left + (W - left - right) * (s - first) / Math.max(1, last - first)
const y = v => top + (H - top - bottom) * (1 - v / maxY)
const f = n => n.toFixed(1)
const ticks = []
const stepY = [5e5, 1e6, 2.5e6, 5e6, 1e7].find(s => maxY / s <= 6) || 2e7
for(let v = 0; v <= maxY; v += stepY){
ticks.push(`<line x1="${left}" x2="${W - right}" y1="${f(y(v))}" y2="${f(y(v))}" class="payGrid"/><text x="${left - 6}" y="${f(y(v) + 4)}" class="payTick" text-anchor="end">${v ? payMoney(v) : "$0"}</text>`)
}
for(let s = Math.ceil(first / S.yearStep) * S.yearStep; s <= last; s += S.yearStep){
ticks.push(`<text x="${f(x(s))}" y="${H - 10}" class="payTick" text-anchor="middle">${s}</text>`)
}
const d = pts.map((p, i) => `${i ? "L" : "M"}${f(x(p.season))},${f(y(p.v))}`).join("")
const dots = pts.map(p => `<g data-tip="${mlbEsc(`${p.season}: ${payMoney(p.v)} of payroll per win (average payroll ÷ ${S.games / 2})`)}" tabindex="0"><circle cx="${f(x(p.season))}" cy="${f(y(p.v))}" r="9" class="payHit"/><circle cx="${f(x(p.season))}" cy="${f(y(p.v))}" r="3" class="payActualDot"/></g>`).join("")
return `<svg class="payChart" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="League payroll per win by season, ${first} to ${last}">${ticks.join("")}<path d="${d}" class="payActualLine"/>${dots}</svg>`
}

// ---- The downloadable workbook ----
// Sheets as plain data: {name, columns: [[header, key, numFmt]], rows,
// scales: [{column index (0-based), higherIsGreen}], gridScale}. payroll.js
// writes them with ExcelJS.

const PAY_MONEY_FMT = '"$"#,##0'
const PAY_RATIO_FMT = "0.000"
const PAY_SIGNED_FMT = "+0.0;-0.0;0.0"

const PAY_SHEET_COLUMNS = [
["Franchise", "franchise", null], ["Year", "season", "0"], ["Status", "status", null],
["Team Payroll", "payroll", PAY_MONEY_FMT], ["W", "wins", "0"], ["L", "losses", "0"], ["WPCT", "wpct", ".000"],
["Price/Game", "price_per_game", PAY_MONEY_FMT], ["Price/Win", "price_per_win", PAY_MONEY_FMT],
["Price/Win/Average", "price_per_win_avg", PAY_RATIO_FMT], ["Win/Average", "win_avg", PAY_RATIO_FMT],
["Payroll Effectiveness Index", "pe_index", PAY_RATIO_FMT], ["Price/Game/Average", "relative_payroll", PAY_RATIO_FMT],
["FO Score", "fo_score", PAY_RATIO_FMT], ["Cash Flow", "cash_flow", PAY_RATIO_FMT], ["Expected Wins", "expected_wins", "0.0"],
["Wins Above Expected", "wins_above_expected", PAY_SIGNED_FMT], ["Payroll Rank", "payroll_rank", "0"],
["Wins Rank", "wins_rank", "0"], ["Normalized Effectiveness", "normalized_effectiveness", PAY_RATIO_FMT],
["Normalized WPC", "normalized_wpct", PAY_RATIO_FMT],
]
const PAY_SHEET_SCALED = ["pe_index", "fo_score", "cash_flow", "wins_above_expected"]

const PAY_GRID_SHEETS = {
team_pe: "Team By Year Effectiveness", team_fo: "Team by Year Front Office", team_cash: "Team by Year Cash Flow",
team_wae: "Team by Year Wins Above Expect", team_payrank: "Team By Year Payroll Rank", team_winrank: "Team By Year Win Rank",
payrank_pe: "Payroll By Year Effectiveness", payrank_wae: "Payroll by Year Above Expected", payrank_fo: "Payroll by Year Front Office",
payrank_winrank: "Payroll By Year Win Rank", winrank_pe: "Wins By Year Effectiveness", winrank_fo: "Wins by Year Front Office",
}

function payCellValue(v){
if(v === undefined || v === null || v === ""){ return null }
const n = Number(v)
return Number.isNaN(n) ? String(v) : n
}

function payWorkbookSheets(seasons, franchises, league, S = PAY_MLB){
const sheet = (name, columns, rows, scaledKeys = []) => ({
name, columns, rows: rows.map(r => columns.map(([, key]) => payCellValue(r[key]))),
scales: columns.map((c, i) => scaledKeys.includes(c[1]) ? { column: i, higherIsGreen: true } : null).filter(Boolean),
})
const sheets = [sheet("Payroll Effectiveness", PAY_SHEET_COLUMNS, seasons, PAY_SHEET_SCALED)]

const winAvg = new Map(seasons.map(r => [`${r.season} ${r.franchise}`, mlbNum(r.win_avg)]))
const nn = Array.from({ length: 10 }, (_, i) => [`NN${i + 1}`, `nn${i + 1}`, null])
const foRows = seasons.map(r => {
const out = { ...r }
String(r.neighbors || "").split(";").filter(Boolean).forEach((key, i) => {
const w = winAvg.get(key)
out[`nn${i + 1}`] = w === null || w === undefined ? key : `${key} (${w.toFixed(3)})`
})
return out
})
sheets.push(sheet("Front Office Score", [
["Franchise", "franchise", null], ["Year", "season", "0"], ["Price/Game/Average", "relative_payroll", PAY_RATIO_FMT],
["Win/Average", "win_avg", PAY_RATIO_FMT], ...nn, ["Neighbor Win/Average", "baseline_win_avg", PAY_RATIO_FMT],
["Year FO Score", "year_fo_score", PAY_RATIO_FMT], ["FO Score", "fo_score", PAY_RATIO_FMT],
["Expected Win/Average", "expected_win_avg", PAY_RATIO_FMT], ["FO Spending Habit", "spending_habit", PAY_RATIO_FMT],
["Cash Flow", "cash_flow", PAY_RATIO_FMT]], foRows))

sheets.push(sheet("Franchises", [
["Franchise", "franchise", null], ["Seasons", "seasons", "0"], ["Average Effectiveness", "average_effectiveness", PAY_RATIO_FMT],
["Average Payroll Rank", "average_payroll_rank", "0.0"], ["Average Wins Rank", "average_wins_rank", "0.0"],
["Average Wins Above Expected", "average_wins_above_expected", PAY_SIGNED_FMT],
["Normalized Effectiveness", "normalized_effectiveness", PAY_RATIO_FMT], ["Normalized Payroll", "normalized_payroll", PAY_RATIO_FMT],
["Normalized Wins", "normalized_wins", PAY_RATIO_FMT]], franchises, ["average_effectiveness"]))

sheets.push(sheet("League", [
["Year", "season", "0"], ["Teams", "teams", "0"], ["Total Payroll", "total_payroll", PAY_MONEY_FMT],
["Average Payroll", "average_payroll", PAY_MONEY_FMT], ["Average Price/Win", "price_per_win", PAY_MONEY_FMT],
[`Price/Win (${S.games / 2}-win average)`, "price_per_win_full_season", PAY_MONEY_FMT],
["Average Price/Game", "price_per_game", PAY_MONEY_FMT]], league))

PAY_GRIDS.forEach(([key, rowsBy, measure, , higher]) => {
const g = payGrid(seasons, rowsBy, measure, S)
const fmt = ["payroll_rank", "wins_rank", "wins_above_expected"].includes(measure) ? "0.0" : PAY_RATIO_FMT
sheets.push({
name: PAY_GRID_SHEETS[key],
columns: [["", "season", "0"], ...g.columns.map(c => [String(c), String(c), fmt])],
rows: g.years.map(y => [y, ...g.columns.map(c => g.cell(y, c))]),
scales: [], gridScale: { higherIsGreen: higher },
})
})
return sheets
}

if (typeof module !== "undefined" && module.exports) {
module.exports = {
PAY_MLB, PAY_NFL, payViewHtml, payColor, payInk, payTeamName, payMoney, payRatio, paySigned, payWins, payPerSeason, paySeasons, paySeasonRows, payDefaultSeason, paySortRows,
paySeasonTableHtml, payHeroHtml, payModelMiss, payScatterSvg, payTeamChartSvg, payNeighborsHtml, payFranchiseTableHtml,
payGrid, payGridHtml, payGridTabsHtml, payLeagueChartSvg, payWorkbookSheets, PAY_GRIDS, PAY_GRID_SHEETS,
}
}
