// How strong is the evidence behind each bet? Renders
// docs/data/bet_evidence.json (scripts/build_bet_evidence.py; the method is
// in src/mlb_metrics/anytime_valid.py) for the NFL Picks tab.
//
// The evidence bankroll: 1 unit staked, a slice at a time, against "these
// bets have no edge". With no edge it rarely ever reaches the bar (20: 1
// chance in 20), however often it is checked - which a win rate with a
// plain margin of error cannot promise for numbers we look at every week.

function evEsc(value){
return String(value ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;")
}

// "×20.6", "×0.02", "×148".
function evTimes(v){
const n = Number(v)
if(!Number.isFinite(n)){ return "" }
return `×${n >= 100 ? Math.round(n).toLocaleString("en-US") : n >= 10 ? n.toFixed(1) : n >= 0.1 ? n.toFixed(2) : n.toFixed(3)}`
}

function evMonth(date){
const m = /^(\d{4})-(\d{2})/.exec(String(date || ""))
return m ? `${["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][Number(m[2]) - 1]} ${m[1]}` : ""
}

// One plain reading of a series' bankroll against the bar.
function evVerdict(s, bar){
const now = Number(s.wealth), peak = Number(s.peak)
if(s.bets < 30){ return "Too few bets to say" }
if(now >= bar * 5){ return "Strong" }
if(now >= bar){ return "At the bar" }
if(peak >= bar){ return "Reached the bar, since slipped" }
if(now >= 2){ return "Some, short of the bar" }
if(now > 0.5){ return "None yet" }
return "Evidence of no edge"
}

function evPct(v){
return `${(100 * Number(v)).toFixed(1)}%`
}

// The table: one row per series of bets.
function evidenceTableHtml(data){
const rows = (data && data.series) || []
if(!rows.length){ return "" }
const bar = Number(data.bar) || 20
const alarm = Number(data.alarm) || 1000
const body = rows.map(s => {
const record = s.win_rate === null || s.win_rate === undefined ? "" : `${s.won}–${s.lost} <small>${evPct(s.win_rate)}</small>`
const units = `${s.units > 0 ? "+" : ""}${Number(s.units).toFixed(1)}`
const peak = Number(s.peak) > Number(s.wealth) * 1.05 && Number(s.peak) > 1.5 ? ` <small>peak ${evTimes(s.peak)}${s.peak_date ? `, ${evMonth(s.peak_date)}` : ""}</small>` : ""
const rates = s.rate_low === undefined ? "<small>prices vary</small>" : `${evPct(s.rate_low)} to ${evPct(s.rate_high)}`
const fade = s.fade === undefined ? "" : s.fade_tripped ? `<b>Tripped ${evMonth(s.fade_tripped)}</b>` : `Quiet <small>${Math.round(s.fade)} of ${alarm.toLocaleString("en-US")}</small>`
return `<tr class="ev-${evEsc(s.kind)}"><td><b>${evEsc(s.label)}</b><small class="evNote">${evEsc(s.note)}</small></td><td>${s.bets}</td><td>${record}</td><td>${units}</td><td><b>${evTimes(s.wealth)}</b>${peak}</td><td>${evEsc(evVerdict(s, bar))}</td><td>${rates}</td><td>${fade}</td></tr>`
}).join("")
return `<table class="trackRecord evidenceTable"><tr><th>Bets</th><th>Made</th><th>Record</th><th>Units</th><th>1 unit of doubt became</th><th>Evidence</th><th>Win rates still possible</th><th>Fade alarm</th></tr>${body}</table>
<p class="evFoot">The bar is ×${bar}. At −110 a bet breaks even at ${evPct(data.break_even)}.</p>`
}

// The bankroll over time for one series, on a log scale (each gridline is
// ten times the one below), with the bar and the starting unit marked.
function evidenceChartSvg(s, bar, width = 720){
const pts = ((s && s.path) || []).map(p => ({at: String(p.at), v: Math.max(1e-6, Number(p.wealth))}))
if(pts.length < 2){ return "" }
const W = Math.max(300, width), H = 240, left = 46, right = 86, top = 16, bottom = 28
const logs = pts.map(p => Math.log10(p.v))
const minL = Math.floor(Math.min(0, ...logs) - 0.05), maxL = Math.ceil(Math.max(Math.log10(bar), ...logs) + 0.05)
const x = i => left + (W - left - right) * i / (pts.length - 1)
const y = v => top + (H - top - bottom) * (1 - (Math.log10(v) - minL) / (maxL - minL))
const f = n => n.toFixed(1)
const grid = []
for(let e = minL; e <= maxL; e++){
const v = 10 ** e
grid.push(`<line x1="${left}" x2="${W - right}" y1="${f(y(v))}" y2="${f(y(v))}" class="payGrid"/><text x="${left - 6}" y="${f(y(v) + 4)}" class="payTick" text-anchor="end">×${v >= 1 ? v.toLocaleString("en-US") : v}</text>`)
}
const every = Math.ceil(pts.length / (W < 500 ? 5 : 9))
const ticks = pts.map((p, i) => i % every === 0 || i === pts.length - 1 ? `<text x="${f(x(i))}" y="${H - 8}" class="payTick" text-anchor="middle">${evEsc(p.at.slice(0, 4) === p.at ? p.at : p.at.slice(5))}</text>` : "").join("")
const line = pts.map((p, i) => `${i ? "L" : "M"}${f(x(i))},${f(y(p.v))}`).join("")
const peakAt = pts.reduce((best, p, i) => p.v > pts[best].v ? i : best, 0)
const dots = pts.map((p, i) => `<g data-tip="${evEsc(`${p.at}: ${evTimes(p.v)}`)}" tabindex="0"><circle cx="${f(x(i))}" cy="${f(y(p.v))}" r="9" class="payHit"/><circle cx="${f(x(i))}" cy="${f(y(p.v))}" r="${i === peakAt || i === pts.length - 1 ? 4 : 2.5}" class="evDot"/></g>`).join("")
const last = pts[pts.length - 1]
const peakLabel = peakAt < pts.length - 1 ? `<text x="${f(x(peakAt))}" y="${f(y(pts[peakAt].v) - 9)}" class="payTick evLabel" text-anchor="middle">${evEsc(pts[peakAt].at)}: ${evTimes(pts[peakAt].v)}</text>` : ""
return `<svg class="payChart evChart" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="${evEsc(`${s.label}: the evidence bankroll over time, now ${evTimes(last.v)}`)}">
${grid.join("")}${ticks}
<line x1="${left}" x2="${W - right}" y1="${f(y(bar))}" y2="${f(y(bar))}" class="evBar"/><text x="${W - right + 6}" y="${f(y(bar) + 4)}" class="payTick">the bar, ×${bar}</text>
<line x1="${left}" x2="${W - right}" y1="${f(y(1))}" y2="${f(y(1))}" class="payAvgLine"/><text x="${W - right + 6}" y="${f(y(1) + 4)}" class="payTick">start, ×1</text>
<path d="${line}" class="evLine"/>${dots}${peakLabel}
<text x="${f(x(pts.length - 1) + 8)}" y="${f(y(last.v) + (Math.abs(y(last.v) - y(bar)) < 12 ? (last.v >= bar ? -10 : 16) : 4))}" class="payTick evLabel">now ${evTimes(last.v)}</text>
</svg>`
}

// The whole block: the table, and the spread rule's bankroll by season.
function evidenceHtml(data, chartKey = "nfl_spread_replay", width = 720){
const table = evidenceTableHtml(data)
if(!table){ return "" }
const s = data.series.find(x => x.key === chartKey)
const chart = s ? evidenceChartSvg(s, Number(data.bar) || 20, width) : ""
const card = chart ? `<div class="vizCard deep-only"><div class="vizHead"><b>${evEsc(s.label)}: the bankroll, season by season</b><span class="vizSub">Each line up the side is ten times the one below. Under ×1 the bets had lost the bankroll money; the dashed line is the bar.</span></div>${chart}</div>` : ""
return `<div class="tableScroll">${table}</div>${card}`
}

if (typeof module !== "undefined" && module.exports) {
module.exports = { evTimes, evMonth, evVerdict, evidenceTableHtml, evidenceChartSvg, evidenceHtml }
}
