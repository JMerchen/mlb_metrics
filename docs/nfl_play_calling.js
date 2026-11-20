// The Play-calling map: what an offense calls in each situation, given
// what came before, against what a typical team calls there - and what
// to expect from each call against the defense across from it.
//
// Data: docs/data/nfl_play_calling.json (scripts/build_nfl_play_calling.py;
// the method and what was tested are in
// src/mlb_metrics/nfl_play_sequencing.py). An offense's lean in a
// situation carries over within a season, so it is shown situation by
// situation. A defense's results in one situation do not, so its number
// there is the league's for the situation moved by how the defense does
// against the run or the pass overall; its raw record in the situation is
// in the hover text, as a record.

function pcEsc(value){
return String(value ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;")
}

function pcPct(v){
return `${Math.round(100 * Number(v))}%`
}

// What to expect when the offense calls a pass or a run in a situation
// against this defense: the league's rate there, moved by both teams'
// overall effects (mirrors nfl_play_sequencing.expected_success).
function pcExpected(data, offense, defense, key){
const out = {}
for(const kind of ["pass", "run"]){
const base = data.league[key][`${kind}_success`]
const v = base + data.teams[offense].offense_effect[kind] + data.teams[defense].defense_effect[kind]
out[kind] = Math.min(1, Math.max(0, v))
}
return out
}

// The situations where an offense's call is most unlike a typical team's:
// at least `minPlays` plays and a lean of 5 points or more, biggest first.
function pcTells(data, offense, count = 2, minPlays = 20){
const team = data.teams[offense]
const rows = []
for(const s of data.states){ rows.push({...s, ...team.offense[s.key], habit: false}) }
for(const h of data.habits){ rows.push({...h, ...team.habits[h.key], habit: true}) }
return rows.filter(r => r.plays >= minPlays && Math.abs(r.lean) >= 0.05).sort((a, b) => Math.abs(b.lean) - Math.abs(a.lean)).slice(0, count)
}

// A situation inside a sentence: "on 2nd & long, after a pass that
// failed", "on 1st down after a run moved the chains", "to open a drive".
function pcPhrase(row){
const lower = row.label.charAt(0).toLowerCase() + row.label.slice(1)
if(row.habit || row.key === "open"){ return lower }
return row.group === "1st down" ? `on 1st down ${lower}` : `on ${row.label}`
}

function pcOrdinal(n){
const s = ["th", "st", "nd", "rd"], v = n % 100
return n + (s[(v - 20) % 10] || s[v] || s[0])
}

// One situation's box: the pass/run split with a tick at the typical
// team's, the numbers, and what each call should do against the defense.
function pcBoxHtml(data, offense, defense, row, names){
if(!row.plays){ return `<div class="callBox callEmpty"><div class="callLabel">${pcEsc(row.label)}</div><div class="callNums">No plays yet</div></div>` }
const lean = Math.round(100 * row.lean)
// A tell needs enough plays behind it (as pcTells).
const tell = Math.abs(row.lean) >= 0.05 && row.plays >= 20
const leanText = tell ? `<span class="callLean">${lean > 0 ? `${lean} points more passing` : `${-lean} points more running`} than typical</span>` : ""
const record = (data.teams[defense].defense || {})[row.key]
const expected = data.league[row.key] && !row.habit ? pcExpected(data, offense, defense, row.key) : null
const tip = `${names(offense)}: ${row.passes} passes in ${row.plays} plays here (${pcPct(row.passes / row.plays)}); after allowing for the small count, ${pcPct(row.rate)}. A typical team in the same plays: ${pcPct(row.typical)}.`
+ (record ? ` ${names(defense)} defense here: passes worked ${record.pass_success} of ${record.passes} times, runs ${record.run_success} of ${record.runs}.` : "")
return `<div class="callBox${tell ? " callTell" : ""}" data-tip="${pcEsc(tip)}" tabindex="0">
<div class="callLabel">${pcEsc(row.label)}</div>
<div class="callBar" role="img" aria-label="${pcEsc(`Pass ${pcPct(row.rate)}, run ${pcPct(1 - row.rate)}; a typical team passes ${pcPct(row.typical)}`)}"><i class="callPass" style="width:${(100 * row.rate).toFixed(1)}%"></i><i class="callRun"></i><i class="callTick" style="left:${(100 * row.typical).toFixed(1)}%"></i></div>
<div class="callNums"><b>${pcPct(row.rate)} pass</b> <span>typical ${pcPct(row.typical)} · ${row.plays} plays</span></div>
${leanText}
${expected ? `<div class="callDef">Against ${pcEsc(names(defense))}: a pass works ${pcPct(expected.pass)}, a run ${pcPct(expected.run)}</div>` : ""}
</div>`
}

// The sentences above one offense's map.
function pcReadHtml(data, offense, defense, names){
const team = data.teams[offense]
const lines = []
const tells = pcTells(data, offense)
if(tells.length){
lines.push(`Biggest tells: ${tells.map(t => `${pcPhrase(t)}, they ${t.lean > 0 ? "throw" : "run"} <b>${pcPct(t.lean > 0 ? t.rate : 1 - t.rate)}</b> of the time (a typical team ${pcPct(t.lean > 0 ? t.typical : 1 - t.typical)})`).join("; ")}.`)
}else{
lines.push("No strong tells: in every situation with enough plays their calls sit within 5 points of a typical team's.")
}
const r = team.readable || {}
if(r.situation !== null && r.situation !== undefined){
lines.push(`Guessing run or pass from the situation alone would be right ${pcPct(r.situation)} of the time against them; knowing these habits, ${pcPct(r.habits)}${r.rank ? ` (the ${pcOrdinal(r.rank)} easiest offense of ${Object.keys(data.teams).length} to read)` : ""}.`)
}
const edge = kind => team.offense_effect[kind] + data.teams[defense].defense_effect[kind]
const points = v => `${Math.abs(Math.round(1000 * v) / 10).toFixed(1)} points ${v >= 0 ? "more" : "less"} often`
lines.push(`Against this defense, next to a typical matchup, their passes should work ${points(edge("pass"))} and their runs ${points(edge("run"))}.`)
return lines.map(l => `<p>${l}</p>`).join("")
}

// One offense's whole map against one defense.
function playCallingHtml(data, offense, defense, names){
if(!data || !data.teams || !data.teams[offense] || !data.teams[defense]){ return "" }
const team = data.teams[offense]
const groups = []
for(const s of data.states){
let g = groups.find(x => x.name === s.group)
if(!g){ g = {name: s.group, rows: []}; groups.push(g) }
g.rows.push({...s, ...team.offense[s.key], habit: false})
}
const tree = groups.map(g => `<div class="callCol"><h5>${pcEsc(g.name)}</h5>${g.rows.map(r => pcBoxHtml(data, offense, defense, r, names)).join("")}</div>`).join("")
const habits = data.habits.map(h => pcBoxHtml(data, offense, defense, {...h, ...team.habits[h.key], habit: true}, names)).join("")
return `<div class="vizCard callMap">
<div class="vizHead"><b>${pcEsc(names(offense))} offense</b> <span>against the ${pcEsc(names(defense))} defense</span>
<span class="vizKey"><i class="callKeyPass"></i>pass <i class="callKeyRun"></i>run <i class="keyTick"></i>a typical team</span></div>
<div class="callRead">${pcReadHtml(data, offense, defense, names)}</div>
<div class="callTree">${tree}</div>
<h5 class="callHabitsTitle">Habits that cut across downs</h5>
<div class="callHabits">${habits}</div>
</div>`
}

// Both offenses of a game, the road team first.
function playCallingMatchupHtml(data, home, away, names){
const name = names || (code => code)
const a = playCallingHtml(data, away, home, name), h = playCallingHtml(data, home, away, name)
return a || h ? `${a}${h}` : ""
}

if (typeof module !== "undefined" && module.exports) {
module.exports = { pcExpected, pcTells, pcPhrase, pcOrdinal, pcBoxHtml, pcReadHtml, playCallingHtml, playCallingMatchupHtml }
}
