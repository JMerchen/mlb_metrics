// Age Curves page (age-curves.html): how a player's season compares with
// history at his age, and what players like him did the next year. Pure
// functions returning HTML/SVG strings so node tests can check them;
// age-curves.js does the DOM wiring. Shares mlb_render.js's pictures,
// helpers and CSS.
//
// Data (scripts/build_age_curves.py): age_curve_projections.csv (one row
// per player and stat), age_curve_comparables.csv (his 25 closest
// historical matches at the same age and what they did next season),
// age_curve_league.csv (average by age) and age_curve_player_history.csv.

// [key, label, hitter?, higher is better?, digits]
const AGE_METRICS = [
["OPS", "OPS", true, true, 3], ["AVG", "AVG", true, true, 3], ["OBP", "OBP", true, true, 3], ["SLG", "SLG", true, true, 3],
["K9", "K/9", false, true, 2], ["BB9", "BB/9", false, false, 2], ["HR9", "HR/9", false, false, 2], ["FIP", "FIP", false, false, 2],
]

// The league curve at ages with fewer seasons than this is mostly the few
// who lasted (survivor bias), so it isn't drawn there.
const AGE_LEAGUE_MIN_SEASONS = 50

function ageMetric(key){
const m = AGE_METRICS.find(x => x[0] === key) || AGE_METRICS[0]
return { key: m[0], label: m[1], hitter: m[2], higherBetter: m[3], digits: m[4] }
}

// ".845" for rate stats, "9.41" for per-nine stats.
function ageValue(value, metric){
const n = mlbNum(value)
if(n === null){ return "-" }
const text = n.toFixed(ageMetric(metric).digits)
return ageMetric(metric).digits === 3 ? text.replace(/^0(?=\.)/, "") : text
}

function ageDelta(delta, metric){
if(delta === null || delta === undefined || Number.isNaN(delta)){ return "-" }
const m = ageMetric(metric)
const text = Math.abs(delta).toFixed(m.digits)
return `${delta > 0 ? "+" : delta < 0 ? "−" : "±"}${m.digits === 3 ? text.replace(/^0(?=\.)/, "") : text}`
}

// Better for the player? (lower is better for BB/9, HR/9, FIP)
function ageIsBetter(delta, metric){
return ageMetric(metric).higherBetter ? delta > 0 : delta < 0
}

function ageMedian(values){
const list = values.filter(v => v !== null && !Number.isNaN(v)).sort((a, b) => a - b)
if(!list.length){ return null }
const mid = Math.floor(list.length / 2)
return list.length % 2 ? list[mid] : (list[mid - 1] + list[mid]) / 2
}

// His comparables with a next season: [{name, year, age, value, next, delta}].
function ageComparables(rows, key, metric){
return (rows || []).filter(r => String(r.key_mlbam) === String(key) && r.metric === metric).map(r => {
const value = mlbNum(r.value), next = mlbNum(r.next_value)
return { name: r.name, year: r.yearID, age: mlbNum(r.age), value, next, delta: value !== null && next !== null ? next - value : null }
})
}

function ageLeagueByAge(league, metric){
const out = {}
;(league || []).filter(r => r.metric === metric && (mlbNum(r.n_seasons) || 0) >= AGE_LEAGUE_MIN_SEASONS)
.forEach(r => { out[mlbNum(r.age)] = mlbNum(r.value) })
return out
}

// Header card: headshot, team, this season vs. league average at his age,
// and next season's projection as a range bar.
function ageHeroHtml(p, metric, league, comps, team){
const m = ageMetric(metric)
const value = mlbNum(p.value), age = mlbNum(p.age)
const avg = ageLeagueByAge(league, metric)[age]
const mean = mlbNum(p.projected_value_mean), lo = mlbNum(p.projected_value_p25), hi = mlbNum(p.projected_value_p75)
const withNext = comps.filter(c => c.delta !== null)
const medianDelta = ageMedian(withNext.map(c => c.delta))
const vsAvg = avg === undefined || value === null ? "" :
`<span class="factChip"><b>${ageDelta(value - avg, metric)}</b> vs. the average ${age}-year-old</span>`
const projection = mean === null ? `<p class="dim">Not enough comparable players with a next season to project.</p>` : `
<div class="ageProjection">
<div class="ageProjLabel">Next season, at ${age + 1}</div>
<div class="ageProjValue"><b>${ageValue(mean, metric)}</b> ${m.label} <span class="dim">likely between ${ageValue(lo, metric)} and ${ageValue(hi, metric)}</span></div>
${ageRangeBar(value, mean, lo, hi, metric)}
<div class="ageProjNote">Of his ${comps.length} closest matches at age ${age}, ${withNext.length} played the next season; their ${m.label} ${medianDelta === null ? "" : `changed by a median of <b>${ageDelta(medianDelta, metric)}</b>`}.</div>
</div>`
return `<div class="vizCard ageHero">
<div class="playerCardHead">${mlbAvatar(p.key_mlbam, p.name, "xl")}
<div><div class="teamHeadName">${mlbEsc(p.name)}</div>
<div class="teamHeadMeta">${team ? `${mlbLogo(team, "xs")} ${mlbEsc(mlbTeam(team))} · ` : ""}age ${age} · ${m.hitter ? "hitter" : "pitcher"}</div>
<div class="factChips"><span class="factChip">This season <b>${ageValue(value, metric)}</b> ${m.label}</span>${vsAvg}</div></div></div>
${projection}
</div>`
}

// This season's value and next season's likely range on one scale.
function ageRangeBar(value, mean, lo, hi, metric){
const points = [value, mean, lo, hi].filter(v => v !== null)
if(points.length < 2){ return "" }
const min = Math.min(...points), max = Math.max(...points)
const pad = (max - min) * 0.25 || Math.abs(max) * 0.05 || 1
const a = min - pad, b = max + pad
const x = v => (100 * (v - a) / (b - a)).toFixed(1)
const better = ageIsBetter(mean - value, metric)
return `<div class="ageRange" role="img" aria-label="${mlbEsc(`This season ${ageValue(value, metric)}, next season likely ${ageValue(lo, metric)} to ${ageValue(hi, metric)}, middle ${ageValue(mean, metric)}`)}">
<span class="ageRangeBand" style="left:${x(lo)}%;width:${(x(hi) - x(lo)).toFixed(1)}%"></span>
<span class="ageRangeMean" style="left:${x(mean)}%;background:${better ? "var(--mlb-better)" : "var(--mlb-worse)"}"></span>
<span class="ageRangeNow" style="left:${x(value)}%"></span>
<span class="ageRangeTag" style="left:${x(value)}%">now</span>
<span class="ageRangeTag ageRangeTagMean" style="left:${x(mean)}%">${ageValue(mean, metric)}</span>
</div>`
}

// The chart, drawn at the container's real width so text stays legible:
// the league's average by age (hairline), every comparable's step from his
// age to the next (faint), the player's own seasons (line and dots), and
// next season's likely range (band) with its middle (dashed).
function ageChartSvg(p, metric, league, history, comps, width = 720){
const m = ageMetric(metric)
const age = mlbNum(p.age), value = mlbNum(p.value)
const leagueByAge = ageLeagueByAge(league, metric)
const own = (history || []).filter(r => String(r.key_mlbam) === String(p.key_mlbam) && r.metric === metric)
.map(r => ({ age: mlbNum(r.age), value: mlbNum(r.value) })).filter(r => r.age !== null && r.value !== null)
if(!own.some(r => r.age === age) && value !== null){ own.push({ age, value }) }
own.sort((a, b) => a.age - b.age)
const mean = mlbNum(p.projected_value_mean), lo = mlbNum(p.projected_value_p25), hi = mlbNum(p.projected_value_p75)
const steps = comps.filter(c => c.delta !== null)

const firstAge = Math.min(age - 6, ...own.map(r => r.age)), lastAge = Math.max(age + 3, ...own.map(r => r.age))
const leagueAges = Object.keys(leagueByAge).map(Number).filter(a => a >= firstAge && a <= lastAge).sort((a, b) => a - b)
const ys = own.map(r => r.value).concat(leagueAges.map(a => leagueByAge[a]), steps.flatMap(c => [c.value, c.next]), [lo, hi, mean].filter(v => v !== null))
const sortedYs = ys.slice().sort((a, b) => a - b)
// Trim the most extreme comparable steps so one outlier doesn't flatten the chart.
const yLo = sortedYs[Math.floor(sortedYs.length * 0.03)], yHi = sortedYs[Math.ceil(sortedYs.length * 0.97) - 1]
const span = (yHi - yLo) || Math.abs(yHi) * 0.1 || 1
const minY = yLo - span * 0.12, maxY = yHi + span * 0.12
const H = 300, left = 44, right = 16, top = 14, bottom = 30
const W = Math.max(300, width)
const x = a => left + (W - left - right) * (a - firstAge) / Math.max(1, lastAge - firstAge)
const y = v => top + (H - top - bottom) * (1 - (Math.min(maxY, Math.max(minY, v)) - minY) / (maxY - minY))
const f = n => n.toFixed(1)

const ticks = []
for(let i = 0; i <= 4; i++){
const v = minY + (maxY - minY) * i / 4
ticks.push(`<line x1="${left}" x2="${W - right}" y1="${f(y(v))}" y2="${f(y(v))}" class="ageGrid"/><text x="${left - 6}" y="${f(y(v) + 4)}" class="ageTick" text-anchor="end">${ageValue(v, metric)}</text>`)
}
const ageTicks = []
for(let a = firstAge; a <= lastAge; a++){
if((lastAge - firstAge) > 12 && a % 2){ continue }
ageTicks.push(`<text x="${f(x(a))}" y="${H - 10}" class="ageTick" text-anchor="middle">${a}</text>`)
}
const leaguePath = leagueAges.map((a, i) => `${i ? "L" : "M"}${f(x(a))},${f(y(leagueByAge[a]))}`).join("")
const compLines = steps.map(c => {
const better = ageIsBetter(c.delta, metric)
return `<g class="ageComp" data-tip="${mlbEsc(`${c.name}, ${c.year} (age ${c.age}): ${ageValue(c.value, metric)} → ${ageValue(c.next, metric)} next season`)}"><line x1="${f(x(c.age))}" y1="${f(y(c.value))}" x2="${f(x(c.age + 1))}" y2="${f(y(c.next))}" class="${better ? "ageCompUp" : "ageCompDown"}"/><line x1="${f(x(c.age))}" y1="${f(y(c.value))}" x2="${f(x(c.age + 1))}" y2="${f(y(c.next))}" class="ageCompHit"/></g>`
}).join("")
const ownPath = own.map((r, i) => `${i ? "L" : "M"}${f(x(r.age))},${f(y(r.value))}`).join("")
const ownDots = own.map(r => `<g data-tip="${mlbEsc(`${p.name} at ${r.age}: ${ageValue(r.value, metric)} ${m.label}`)}" tabindex="0"><circle cx="${f(x(r.age))}" cy="${f(y(r.value))}" r="12" class="ageHitArea"/><circle cx="${f(x(r.age))}" cy="${f(y(r.value))}" r="5" class="ageOwnDot"/></g>`).join("")
const band = mean === null ? "" : `<rect x="${f(x(age + 1) - 9)}" y="${f(y(hi === null ? mean : Math.max(hi, lo)))}" width="18" height="${f(Math.abs(y(lo === null ? mean : lo) - y(hi === null ? mean : hi)))}" rx="4" class="ageBand"/>
<line x1="${f(x(age))}" y1="${f(y(value))}" x2="${f(x(age + 1))}" y2="${f(y(mean))}" class="ageProjLine"/>
<g data-tip="${mlbEsc(`Next season, at ${age + 1}: about ${ageValue(mean, metric)}, likely ${ageValue(lo, metric)} to ${ageValue(hi, metric)}`)}" tabindex="0"><circle cx="${f(x(age + 1))}" cy="${f(y(mean))}" r="12" class="ageHitArea"/><circle cx="${f(x(age + 1))}" cy="${f(y(mean))}" r="5.5" class="ageProjDot"/></g>`
const leagueLabelAge = leagueAges.filter(a => a <= lastAge).slice(-1)[0]
const leagueLabel = leagueLabelAge === undefined ? "" : `<text x="${f(Math.min(W - right - 2, x(leagueLabelAge)))}" y="${f(y(leagueByAge[leagueLabelAge]) - 7)}" class="ageTick" text-anchor="end">League average</text>`
const summary = `${p.name}'s ${m.label} by age, with ${steps.length} comparable players' next seasons and a projection of ${ageValue(mean, metric)} at ${age + 1}`
return `<svg class="ageChart" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="${mlbEsc(summary)}">
${ticks.join("")}${ageTicks.join("")}
<text x="4" y="${H - 10}" class="ageTick">age</text>
${compLines}
<path d="${leaguePath}" class="ageLeague"/>${leagueLabel}
${band}
<path d="${ownPath}" class="ageOwn"/>${ownDots}
</svg>`
}

// "Players who looked like him at his age", most similar first, each with
// where his stat went the next season.
function ageComparablesHtml(comps, metric){
if(!comps.length){ return "No comparable players" }
const m = ageMetric(metric)
return `<div class="rankList ageCompList">${comps.map((c, i) => {
const moved = c.delta === null ? `<span class="ageMove dim">no next year</span>`
: `<span class="ageMove ${ageIsBetter(c.delta, metric) ? "resultGood" : "resultBad"}">${ageDelta(c.delta, metric)}</span>`
return `<div class="rankRow ageCompRow" data-tip="${mlbEsc(`${c.name}, ${c.year} (age ${c.age}): ${ageValue(c.value, metric)} ${m.label}${c.next === null ? "" : `, then ${ageValue(c.next, metric)}`}`)}" tabindex="0">
<span class="rankNo">${i + 1}</span><span class="rankWho"><b>${mlbEsc(c.name)}</b><small>${mlbEsc(c.year)} · age ${mlbEsc(c.age)}</small></span>
<span class="ageCompValues">${ageValue(c.value, metric)} <span class="dim">→</span> ${c.next === null ? "-" : ageValue(c.next, metric)}</span>${moved}
</div>`
}).join("")}</div>`
}

// Search suggestions with headshots.
function ageSuggestionsHtml(matches){
if(!matches.length){ return `<span class="dim">No players found</span>` }
return matches.map((p, i) => `<button class="playerChoice" data-index="${i}">${mlbAvatar(p.key_mlbam, p.name, "xs")}<span>${mlbEsc(p.name)}</span><small>age ${mlbEsc(p.age)}</small></button>`).join("")
}

if (typeof module !== "undefined" && module.exports) {
module.exports = {
AGE_METRICS, ageMetric, ageValue, ageDelta, ageIsBetter, ageMedian, ageComparables, ageLeagueByAge,
ageHeroHtml, ageRangeBar, ageChartSvg, ageComparablesHtml, ageSuggestionsHtml,
}
}
