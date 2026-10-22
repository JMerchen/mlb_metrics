// NFL player props, drawn instead of tabled (user request, 2026-10-22: "make
// the display for the player props section as interesting as possible").
//
// Two pieces, both pure functions returning HTML strings so node tests can
// check them; nfl.js does the DOM wiring and the table stays one click away.
//
//   leanChartHtml - the slate at a glance: one diverging bar per pick, how far
//     our projection sits from the player's own average (unders left, overs
//     right). For sacks, which have no projection, the opponent's rate vs the
//     league stands in.
//   propCardHtml - one card per pick: the projection against his average as
//     a sentence, his recent games as dots around the projection line (with a
//     "cleared it in X of Y" count), and the reasons - what the opponent
//     allows, his unit against theirs, teammates out, projected volume.
//
// Colour: over and under are the diverging pair (blue / orange, validated
// with the dataviz palette checker against the card surface); text always
// stays in the text colours and every coloured mark has a word beside it.

const PROP_STAT_WORDS = {
"Receptions": { stat: "receptions", unit: "catches" },
"Receiving Yards": { stat: "receiving yards", unit: "yards" },
"Rushing Yards": { stat: "rushing yards", unit: "yards" },
"Passing Yards": { stat: "passing yards", unit: "yards" },
"Sacks": { stat: "sacks", unit: "sacks" },
}

function propEsc(s){
return String(s === undefined || s === null ? "" : s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))
}

function propNumber(value){
return value === undefined || value === null || value === "" || Number.isNaN(Number(value)) ? null : Number(value)
}

function propTeam(code){
return typeof teamName === "function" ? teamName(code, "nfl") : (code || "")
}

// 72.53 -> "72.5", 6 -> "6", 0.739 with 2 digits -> "0.74"
function propFixed(n, digits){
if(n === null){ return "-" }
return n.toFixed(digits === undefined ? 1 : digits).replace(/\.0+$/, "")
}

function propSignedPct(gap){
const pct = Math.round(gap * 100)
return `${pct > 0 ? "+" : pct < 0 ? "−" : ""}${Math.abs(pct)}%`
}

// How far the lean goes: projection against his average where there is a
// projection, otherwise the opponent's rate against the league.
function propLeanGap(row){
const average = propNumber(row.player_rate)
const projection = propNumber(row.projection)
if(projection !== null && average){
return { gap: projection / average - 1, basis: "projection" }
}
const ratio = propNumber(row.ratio)
return ratio === null ? { gap: null, basis: "none" } : { gap: ratio - 1, basis: "opponent" }
}

function propIsOver(row){
return String(row.direction).toLowerCase() === "over"
}

// "45, 80, 12" + "2026 W4, 2026 W5, 2026 W6" -> [{value, label}], oldest first.
function propRecentGames(row){
const values = String(row.recent_values || "").split(",").map(s => s.trim()).filter(s => s !== "")
const weeks = String(row.recent_weeks || "").split(",").map(s => s.trim())
return values.map((v, i) => ({ value: Number(v), label: weeks[i] || "" })).filter(g => !Number.isNaN(g.value))
}

// The number each game is measured against: the projection, or his average
// when there is no projection (sacks).
function propLine(row){
const projection = propNumber(row.projection)
return projection !== null ? projection : propNumber(row.player_rate)
}

// How many of his recent games landed on the side we lean.
function propClearedCount(row){
const line = propLine(row)
const games = propRecentGames(row)
if(line === null || !games.length){ return null }
const over = propIsOver(row)
const hits = games.filter(g => over ? g.value > line : g.value < line).length
return { hits, games: games.length, line }
}

// ---- the slate at a glance -------------------------------------------------

function leanChartHtml(rows){
const picks = (rows || []).map(r => ({ row: r, lean: propLeanGap(r) })).filter(p => p.lean.gap !== null)
if(!picks.length){ return "" }
const reach = Math.max(0.25, ...picks.map(p => Math.abs(p.lean.gap)))
const lines = picks.map(({ row, lean }) => {
const over = propIsOver(row)
const width = Math.min(50, Math.abs(lean.gap) / reach * 50)
const words = PROP_STAT_WORDS[row.category] || { stat: String(row.category).toLowerCase() }
const label = `${propEsc(row.player_name)} <span class="leanStat">${propEsc(words.stat)}</span>`
const basis = lean.basis === "projection" ? "projection vs his average" : "opponent vs league"
return `<a class="leanRow" href="#prop-card-${propEsc(row.rank)}" title="${propEsc(row.player_name)}, ${propEsc(row.category)}: ${over ? "Over" : "Under"}, ${propSignedPct(lean.gap)} (${basis})">
<span class="leanName">${label}</span>
<span class="leanTrack"><span class="leanMid"></span><span class="leanBar ${over ? "leanOver" : "leanUnder"}" style="${over ? "left:50%" : `right:50%`};width:${width.toFixed(1)}%"></span></span>
<span class="leanValue">${over ? "Over" : "Under"} ${propSignedPct(lean.gap)}</span>
</a>`
}).join("")
return `<div class="leanChart">
<div class="leanAxis"><span></span><span class="leanAxisScale"><span>← under</span><span>his usual</span><span>over →</span></span><span></span></div>
${lines}
</div>`
}

// ---- one card per pick -------------------------------------------------------

function propHeadline(row){
const words = PROP_STAT_WORDS[row.category] || { stat: String(row.category).toLowerCase(), unit: "" }
const average = propNumber(row.player_rate)
const projection = propNumber(row.projection)
const lean = propLeanGap(row)
if(projection !== null && average !== null){
const direction = projection >= average ? "above" : "below"
return `We project <b>${propFixed(projection)}</b> ${propEsc(words.stat)}, ${Math.abs(Math.round(lean.gap * 100))}% ${direction} his ${propFixed(average)} average`
}
if(average !== null){
return `He averages <b>${propFixed(average, 2)}</b> ${propEsc(words.stat)} a game`
}
return ""
}

// His recent games as dots, the projection as a dashed line (it is a line
// we drew, not a measurement), his average as a hairline.
function propStripSvg(row){
const games = propRecentGames(row)
if(!games.length){ return "" }
const over = propIsOver(row)
const projection = propNumber(row.projection)
const average = propNumber(row.player_rate)
const line = propLine(row)
const W = 300, H = 104, top = 8, bottom = 22, left = 6, right = 6
const values = games.map(g => g.value).concat([projection, average].filter(v => v !== null))
const hi = Math.max(...values, 1) * 1.12
const lo = Math.min(0, ...values)
const y = v => top + (H - top - bottom) * (1 - (v - lo) / (hi - lo))
const step = (W - left - right) / games.length
const x = i => left + step * (i + 0.5)
const words = PROP_STAT_WORDS[row.category] || { unit: "" }
const marks = games.map((g, i) => {
const onLean = line === null ? false : (over ? g.value > line : g.value < line)
const tip = `${g.label}: ${propFixed(g.value)} ${words.unit}${onLean ? (over ? " (over)" : " (under)") : ""}`
return `<g class="stripDot" tabindex="0"><title>${propEsc(tip)}</title>
<circle cx="${x(i).toFixed(1)}" cy="${y(g.value).toFixed(1)}" r="12" fill="transparent"/>
<circle cx="${x(i).toFixed(1)}" cy="${y(g.value).toFixed(1)}" r="4.5" class="${onLean ? (over ? "dotOver" : "dotUnder") : "dotOther"}"/>
</g><text x="${x(i).toFixed(1)}" y="${H - 6}" class="stripTick" text-anchor="middle">${propEsc(String(g.label).replace(/^\d+ /, ""))}</text>`
}).join("")
const guide = (v, cls) => v === null ? "" : `<line x1="${left}" x2="${W - right}" y1="${y(v).toFixed(1)}" y2="${y(v).toFixed(1)}" class="${cls}"/>`
const summary = `${games.length} recent games of ${words.unit}; projection ${propFixed(projection)}, average ${propFixed(average)}`
return `<svg class="propStrip" viewBox="0 0 ${W} ${H}" role="img" aria-label="${propEsc(summary)}">
${guide(average, "stripAverage")}
${guide(projection, over ? "stripProjection stripProjectionOver" : "stripProjection stripProjectionUnder")}
${marks}
</svg>`
}

// Small diverging meter for a share above or below average (capped at ±50%).
function propMeter(share, overIsGood){
if(share === null){ return "" }
const width = Math.min(50, Math.abs(share) * 100)
const right = share >= 0
const cls = (right === overIsGood) ? "leanOver" : "leanUnder"
return `<span class="whyMeter"><span class="leanMid"></span><span class="leanBar ${cls}" style="${right ? "left:50%" : "right:50%"};width:${width.toFixed(1)}%"></span></span>`
}

function propOpponentReason(row){
const opponent = propTeam(row.opponent)
const allowed = propNumber(row.opponent_allowed_rate)
const league = propNumber(row.league_rate)
if(allowed === null || league === null || !league){ return null }
const share = allowed / league - 1
const pos = propEsc(row.position)
const more = share >= 0 ? "more" : "less"
const pct = `${Math.abs(Math.round(share * 100))}% ${more} than average`
let text
switch(row.category){
case "Receptions":
text = `${propEsc(opponent)} let ${pos}s catch ${Math.round(allowed * 100)}% of targets (league ${Math.round(league * 100)}%)`
break
case "Receiving Yards":
text = `${propEsc(opponent)} allow ${allowed.toFixed(1)} yards per target to ${pos}s (league ${league.toFixed(1)})`
break
case "Rushing Yards":
text = `${propEsc(opponent)} allow ${allowed.toFixed(1)} yards per carry to ${pos}s (league ${league.toFixed(1)})`
break
case "Passing Yards":
text = `${propEsc(opponent)} allow ${allowed.toFixed(1)} yards per pass attempt (league ${league.toFixed(1)})`
break
case "Sacks":
text = `${propEsc(opponent)} give up ${allowed.toFixed(1)} sacks a game (league ${league.toFixed(1)})`
break
default:
return null
}
// More allowed helps an over; the meter's colour says which way it pushes.
return { text: `${text}: ${pct}`, meter: propMeter(share, true) }
}

function propUnitReason(row, units){
if(!units || typeof PROP_UNIT_CONTEXT === "undefined"){ return null }
const context = PROP_UNIT_CONTEXT[row.category]
if(!context){ return null }
const values = context.map(([label, side, key]) => {
const unit = units[side === "own" ? row.team : row.opponent]
const n = unit ? propNumber(unit[key]) : null
return n === null ? null : { label, side, n }
})
if(values.some(v => v === null)){ return null }
if(values.length === 2){
const edge = values[0].n - values[1].n
const text = `${values[0].label} ${Math.round(values[0].n)} vs ${values[1].label} ${Math.round(values[1].n)}: `
+ `${edge >= 0 ? "+" : "−"}${Math.abs(Math.round(edge))} edge`
return { text, meter: propMeter(Math.max(-0.5, Math.min(0.5, edge / 120)), true) }
}
const only = values[0]
return { text: `${only.label} ${Math.round(only.n)}`, meter: propMeter(Math.max(-0.5, Math.min(0.5, (100 - only.n) / 120)), true) }
}

function propVolumeReason(row){
const targets = propNumber(row.projected_targets)
const carries = propNumber(row.projected_carries)
if(["Receptions", "Receiving Yards"].includes(row.category) && targets !== null){
return { text: `About ${targets.toFixed(1)} targets expected`, meter: "" }
}
if(row.category === "Rushing Yards" && carries !== null){
return { text: `About ${carries.toFixed(1)} carries expected`, meter: "" }
}
return null
}

function propCardHtml(row, units){
const over = propIsOver(row)
const words = PROP_STAT_WORDS[row.category] || { stat: String(row.category).toLowerCase(), unit: "" }
const cleared = propClearedCount(row)
const reasons = [
propOpponentReason(row),
propUnitReason(row, units),
row.context_note ? { text: propEsc(row.context_note), meter: "" } : null,
propVolumeReason(row),
].filter(Boolean)
const clearedLine = cleared
? `${over ? "Over" : "Under"} ${propFixed(cleared.line)} in <b>${cleared.hits} of his last ${cleared.games}</b> games`
: ""
return `<article class="propCard" id="prop-card-${propEsc(row.rank)}">
<header class="propCardHead">
<span class="propRank">#${propEsc(row.rank)}</span>
<span class="propWho"><b>${propEsc(row.player_name)}</b> <span class="propMeta">${propEsc(row.position)} · ${propEsc(propTeam(row.team))} vs ${propEsc(propTeam(row.opponent))}</span></span>
<span class="propLean ${over ? "propLeanOver" : "propLeanUnder"}"><i></i>${over ? "Over" : "Under"}</span>
</header>
<div class="propStat">${propEsc(words.stat.charAt(0).toUpperCase() + words.stat.slice(1))}</div>
<p class="propHeadline">${propHeadline(row)}</p>
${propStripSvg(row)}
${clearedLine ? `<p class="propCleared">${clearedLine}${propNumber(row.projection) !== null
? ` <span class="propKey"><i class="keyProjection ${over ? "keyOver" : "keyUnder"}"></i>our projection <i class="keyAverage"></i>his average</span>`
: ` <span class="propKey"><i class="keyAverage"></i>his average</span>`}</p>` : ""}
${reasons.length ? `<ul class="propWhy">${reasons.map(r => `<li>${r.meter}<span>${r.text}</span></li>`).join("")}</ul>` : ""}
</article>`
}

function propCardsHtml(rows, units){
if(!rows || !rows.length){ return "No player props yet - this board updates weekly." }
return rows.map(r => propCardHtml(r, units)).join("")
}

if (typeof module !== "undefined" && module.exports) {
module.exports = {
propLeanGap, propRecentGames, propClearedCount, propHeadline, propStripSvg,
leanChartHtml, propCardHtml, propCardsHtml, propOpponentReason, propUnitReason,
}
}
