// Ranges with a stated hit rate (docs/data/nfl_ranges.json, built by
// scripts/build_nfl_ranges.py; the method - conformal prediction - is in
// src/mlb_metrics/conformal.py, and these functions mirror its range_for
// and chance_over_line).
//
// A calibration is the model's own past misses, on games it had not seen:
// {n, scale, edges, groups: [{n, scores}]}, each group's misses given at
// 201 evenly spaced shares (0%, 0.5%, ... 100%). A range is read straight
// off them, so "4 times in 5" has meant 4 times in 5.

function rgInterp(x, xs, ys){
if(x <= xs[0]){ return ys[0] }
const last = xs.length - 1
if(x >= xs[last]){ return ys[last] }
let lo = 0, hi = last
while(hi - lo > 1){
const mid = (lo + hi) >> 1
if(xs[mid] <= x){ lo = mid }else{ hi = mid }
}
const span = xs[hi] - xs[lo]
return span > 0 ? ys[lo] + (ys[hi] - ys[lo]) * (x - xs[lo]) / span : ys[hi]
}

const RG_GRID = Array.from({length: 201}, (_, i) => i / 200)

function rgGroup(cal, prediction){
const edges = cal.edges || []
let g = 0
while(g < edges.length && edges[g] < prediction){ g++ }
return cal.groups[g]
}

// The typical size of a miss at this prediction (1 with no scale).
function rgScale(cal, prediction){
const s = cal.scale
return s ? Math.max(s.intercept + s.slope * prediction, s.floor) : 1
}

// [low, high] for one prediction at `coverage` (0.5 or 0.8), or null.
// `lower` clips the low end (0 for yards and points).
function rangeFor(cal, prediction, coverage, lower){
if(!cal || !cal.groups || prediction === null || prediction === undefined || prediction === "" || Number.isNaN(Number(prediction))){ return null }
const p = Number(prediction)
const group = rgGroup(cal, p)
if(!group || !group.n){ return null }
const tail = (1 - coverage) / 2, n = group.n, size = rgScale(cal, p)
const at = share => rgInterp(Math.min(1, Math.max(0, share)), RG_GRID, group.scores)
// Rounded first, so 0.9 x 10 is 9 and not 9.000000000000002.
const low = p + at(Math.floor(Number((tail * (n + 1)).toFixed(9))) / n) * size
const high = p + at(Math.ceil(Number(((1 - tail) * (n + 1)).toFixed(9))) / n) * size
return [lower !== null && lower !== undefined ? Math.max(low, lower) : low, high]
}

// The chance the result is over `line`. Never exactly 0 or 1.
function chanceOver(cal, prediction, line){
if(!cal || !cal.groups){ return null }
const p = Number(prediction), x = Number(line)
if(line === "" || line === null || Number.isNaN(p) || Number.isNaN(x)){ return null }
const group = rgGroup(cal, p)
if(!group || !group.n){ return null }
const below = rgInterp((x - p) / rgScale(cal, p), group.scores, RG_GRID)
return (group.n * (1 - below) + 0.5) / (group.n + 1)
}

// The fair American price for a chance: 60% -> "−150", 40% -> "+150".
function fairPrice(chance){
const c = Number(chance)
if(!(c > 0 && c < 1)){ return "" }
return c >= 0.5 ? `−${Math.round(100 * c / (1 - c))}` : `+${Math.round(100 * (1 - c) / c)}`
}

// "Colts by 1 to Texans by 17" from home-minus-away margins.
function rgMarginSpan(low, high, home, away){
const side = x => { const r = Math.round(x); return r === 0 ? "even" : `${r > 0 ? home : away} by ${Math.abs(r)}` }
return `${side(low)} to ${side(high)}`
}

// The pick card's two range lines for a game: its total 4 times in 5 (the
// half-the-time range is already on the card) and its margin both ways.
function gameRangeLines(ranges, g, home, away){
if(!ranges){ return [] }
const out = []
const num = v => v === null || v === undefined || v === "" || Number.isNaN(Number(v)) ? null : Number(v)
const total = num(g.predicted_total), margin = num(g.predicted_margin)
const wide = total !== null && ranges.total ? rangeFor(ranges.total, total, 0.8, 0) : null
if(wide){ out.push(`4 times in 5 the total lands in ${Math.round(wide[0])}–${Math.round(wide[1])}`) }
const half = margin !== null && ranges.margin ? rangeFor(ranges.margin, margin, 0.5) : null
const most = margin !== null && ranges.margin ? rangeFor(ranges.margin, margin, 0.8) : null
if(half && most){
out.push(`Margin: half the time ${rgMarginSpan(half[0], half[1], home, away)}; 4 times in 5, ${rgMarginSpan(most[0], most[1], home, away)}`)
}
return out
}

function rgNumber(v, unit){
return unit === "receptions" || unit === "catches" ? (Math.round(v * 2) / 2).toString() : Math.round(v).toString()
}

// A player card's range line and line checker. `row` is a props row
// (category, projection, player_rate); `words` its stat words.
function propRangeHtml(ranges, row, unit){
const cal = ranges && ranges.props ? ranges.props[row.category] : null
const projection = row.projection === "" || row.projection === null || row.projection === undefined ? NaN : Number(row.projection)
if(!cal || Number.isNaN(projection)){ return "" }
const floor = cal.floor === undefined ? 0 : cal.floor
const half = rangeFor(cal, projection, 0.5, floor), most = rangeFor(cal, projection, 0.8, floor)
if(!half || !most){ return "" }
const average = Number(row.player_rate)
const line = Number.isNaN(average) ? Math.floor(projection) + 0.5 : Math.floor(average) + 0.5
return `<div class="propRange" data-prop-category="${String(row.category).replace(/"/g, "&quot;")}" data-prop-projection="${projection}">
<p class="propRangeLine">Likely ${unit || ""}: <b>${rgNumber(half[0], unit)}–${rgNumber(half[1], unit)}</b> half the time, <b>${rgNumber(most[0], unit)}–${rgNumber(most[1], unit)}</b> 4 times in 5.</p>
<label class="propLineCheck">Against a line of <input type="number" step="0.5" min="0" inputmode="decimal" value="${line}" aria-label="The sportsbook's line"> <output>${propLineText(cal, projection, line)}</output></label>
</div>`
}

// "over 43% (fair at +133) · under 57% (fair at −133)"
function propLineText(cal, projection, line){
const over = chanceOver(cal, projection, line)
if(over === null || line === "" || Number.isNaN(Number(line))){ return "" }
const pct = c => `${Math.round(100 * c)}%`
return `over <b>${pct(over)}</b> (fair at ${fairPrice(over)}) · under <b>${pct(1 - over)}</b> (fair at ${fairPrice(1 - over)})`
}

if (typeof module !== "undefined" && module.exports) {
module.exports = { rangeFor, chanceOver, fairPrice, rgMarginSpan, gameRangeLines, propRangeHtml, propLineText }
}
