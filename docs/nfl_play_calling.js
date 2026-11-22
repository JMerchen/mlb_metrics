// The play-calling tree (NFL Stats tab, under a team's schedule): how a
// team's drives open, one play branching to the next.
//
// Why a tree (user request, 2026-11-21): the first version, a box per
// situation for both offenses of a game, was too big for the Picks tab and
// did not inform picks. This one reads left to right from the start of a
// drive. Each layer is the next play of the drive ("Play by play") or, in
// the other view, alternately a play and whether it worked. No percentages
// on the drawing: a box's color is how typical that sequence is, the
// boldest in a column being the commonest way to get that far. The counts
// are in each box's hover text.
//
// Data: docs/data/nfl_play_calling.json (scripts/build_nfl_play_calling.py,
// src/mlb_metrics/nfl_play_sequencing.py): for each team, how many drives
// opened with each sequence - "P-R+R+" is an incompletion, then two runs
// that worked - on offense and against its defense, and the league's.
//
// The layout is fixed: every branch is always drawn, in the same place for
// every team, so two teams' trees can be read against each other. A
// sequence a team has never opened with is an empty outline.

// The color scale's two ends, on the cards' green: paper, and navy ink.
const PC_PALE = "#e6f2e6", PC_DEEP = "#0f2b57"
const PC_LAYERS = {
plays: ["call", "call", "call", "call"],
results: ["call", "result", "call", "result", "call"],
}
// Run above pass, worked above didn't: fixed, so a place means one thing.
const PC_BRANCHES = {call: ["R", "P"], result: ["+", "-"]}
const PC_WORD = {R: "run", P: "pass"}

function pcEsc(value){
return String(value ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;")
}

function pcOrdinal(n){
const s = ["th", "st", "nd", "rd"], v = n % 100
return n + (s[(v - 20) % 10] || s[v] || s[0])
}

// ---- color: one hue, pale to deep, mixed in OKLab so equal steps look equal ----

function pcToOklab(hex){
const lin = [1, 3, 5].map(i => { const c = parseInt(hex.slice(i, i + 2), 16) / 255; return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4 })
const l = Math.cbrt(0.4122214708 * lin[0] + 0.5363325363 * lin[1] + 0.0514459929 * lin[2])
const m = Math.cbrt(0.2119034982 * lin[0] + 0.6806995451 * lin[1] + 0.1073969566 * lin[2])
const s = Math.cbrt(0.0883024619 * lin[0] + 0.2817188376 * lin[1] + 0.6299787005 * lin[2])
return [0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s, 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s, 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s]
}

function pcFromOklab([L, a, b]){
const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3, m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3, s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3
const rgb = [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s, -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s, -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s]
return "#" + rgb.map(c => { const v = c <= 0.0031308 ? 12.92 * c : 1.055 * Math.max(c, 0) ** (1 / 2.4) - 0.055; return Math.round(255 * Math.min(1, Math.max(0, v))).toString(16).padStart(2, "0") }).join("")
}

// The color for a typicality t (0 = never, 1 = the commonest in its
// column) and whether text on it should be light.
function pcColor(t){
const p = pcToOklab(PC_PALE), d = pcToOklab(PC_DEEP), k = Math.min(1, Math.max(0, t))
const mixed = p.map((v, i) => v + (d[i] - v) * k)
return {fill: pcFromOklab(mixed), light: mixed[0] < 0.62}
}

// ---- the tree ----

// "P-R+" -> [{call: "P", ok: false}, {call: "R", ok: true}]
function pcParse(sequence){
const plays = []
for(let i = 0; i + 1 < sequence.length; i += 2){ plays.push({call: sequence[i], ok: sequence[i + 1] === "+"}) }
return plays
}

// The branches one drive takes through a view's layers.
function pcSteps(sequence, mode){
const plays = pcParse(sequence), layers = PC_LAYERS[mode], steps = []
let play = 0
for(const kind of layers){
if(!plays[play]){ break }
if(kind === "call"){ steps.push(plays[play].call) }else{ steps.push(plays[play].ok ? "+" : "-") }
// A result layer finishes its play; so does a call followed by a call.
if(kind === "result" || layers[steps.length] === "call"){ play += 1 }
}
return steps
}

// The whole fixed tree for a view, every node counting the drives that
// came through it: {path, step, kind, layer, play, drives, worked, share,
// children}. `worked` is how many of those drives saw this node's play
// work; `share` is drives over the busiest node of its layer.
function pcTree(counts, mode){
const layers = PC_LAYERS[mode]
const grow = (path, layer, play) => {
if(layer >= layers.length){ return [] }
const kind = layers[layer]
return PC_BRANCHES[kind].map(step => {
const nextPlay = kind === "result" || layers[layer + 1] === "call" ? play + 1 : play
return {path: path + step, step, kind, layer, play, drives: 0, worked: 0, share: 0, children: grow(path + step, layer + 1, nextPlay)}
})
}
const root = {path: "", step: "", kind: "root", layer: -1, play: 0, drives: 0, worked: 0, share: 1, children: grow("", 0, 1)}
for(const [sequence, n] of Object.entries(counts || {})){
const plays = pcParse(sequence), steps = pcSteps(sequence, mode)
root.drives += n
let node = root
for(const step of steps){
node = node.children.find(c => c.step === step)
node.drives += n
if(plays[node.play - 1] && plays[node.play - 1].ok){ node.worked += n }
}
}
const byLayer = pcByLayer(root)
for(const nodes of byLayer){
const most = Math.max(...nodes.map(n => n.drives))
for(const n of nodes){ n.share = most > 0 ? n.drives / most : 0 }
}
return root
}

// The tree's nodes layer by layer, each layer top to bottom.
function pcByLayer(root){
const out = []
const walk = node => { for(const c of node.children){ (out[c.layer] = out[c.layer] || []).push(c); walk(c) } }
walk(root)
return out
}

// A node in words: "run, then pass, then pass"; with results, "a run that
// worked, then a pass that didn't work, then a run".
function pcPhrase(path, mode){
if(mode !== "results"){ return [...path].map(c => PC_WORD[c]).join(", then ") }
const parts = []
for(let i = 0; i < path.length; i += 2){
const call = PC_WORD[path[i]], result = path[i + 1]
parts.push(`a ${call}${result === "+" ? " that worked" : result === "-" ? " that didn't work" : ""}`)
}
return parts.join(", then ")
}

// How many drives in 100 a count is, for the league's side of a tip.
function pcPer100(n, total){
return total > 0 ? Math.round(100 * n / total) : 0
}

// What the hover says about one node.
function pcTip(node, tree, league, mode, who){
const phrase = pcPhrase(node.path, mode)
const text = phrase.charAt(0).toUpperCase() + phrase.slice(1)
const lines = [`${text}: ${node.drives} of ${tree.drives} ${who} opened this way.`]
const same = league && pcFind(league, node.path)
if(same && league.drives){ lines.push(`Across the league, about ${pcPer100(same.drives, league.drives)} of every 100 do.`) }
if(mode !== "results" && node.drives){ lines.push(`The ${pcOrdinal(node.play)} play worked on ${node.worked} of those ${node.drives}.`) }
return lines.join(" ")
}

function pcFind(root, path){
let node = root
for(const step of path){ node = node && node.children.find(c => c.step === step) }
return node
}

// The commonest full sequence, and the one a team opens with most unlike
// the league (at least `minDrives` drives, two calls or more).
function pcStandouts(tree, league, mode, minDrives = 8){
const layers = pcByLayer(tree)
const last = layers[layers.length - 1] || []
const typical = last.slice().sort((a, b) => b.drives - a.drives)[0]
let signature = null
if(league && league.drives && tree.drives){
for(const nodes of layers.slice(1)){
for(const n of nodes){
if(n.kind !== "call" || n.drives < minDrives){ continue }
const usual = pcFind(league, n.path).drives / league.drives
const gap = n.drives / tree.drives - usual
if(gap > 0 && (!signature || gap > signature.gap)){ signature = {node: n, gap, usual: usual * tree.drives} }
}
}
}
return {typical: typical && typical.drives ? typical : null, signature}
}

// ---- drawing ----

// Where everything goes for a tree `width` wide: {height, columns: [x],
// nodes: [{node, x, y, w, h}], root: {x, y}}. Leaves sit on even rows;
// a parent sits midway between its children.
function pcLayout(tree, mode, width){
const layers = PC_LAYERS[mode]
const leaves = 2 ** layers.length
const rowH = mode === "results" ? 17 : 22
const top = 26, bottom = 8
const padL = 30, padR = 26
const gap = (width - padL - padR) / layers.length
const columns = layers.map((_, i) => padL + gap * (i + 1))
const placed = []
let leaf = 0
const place = node => {
const kids = node.children.map(place)
const y = kids.length ? (kids[0].y + kids[kids.length - 1].y) / 2 : top + rowH * (leaf++ + 0.5)
const spacing = rowH * 2 ** (layers.length - 1 - node.layer)
const h = Math.min(22, Math.max(11, Math.min(spacing - 4, rowH * 1.5)))
const w = node.kind === "result" ? h : Math.min(48, Math.max(30, gap * 0.6))
const box = {node, x: node.layer < 0 ? padL : columns[node.layer], y, w, h}
if(node.layer >= 0){ placed.push(box) }
return box
}
const root = place(tree)
return {height: top + rowH * leaves + bottom, columns, nodes: placed, root, rowH}
}

// A column's heading; where the columns are close together (`narrow`), a
// short one, and none over the worked / didn't marks the key explains.
function pcColumnTitle(mode, layer, narrow = false){
const layers = PC_LAYERS[mode]
if(layers[layer] === "result"){ return narrow ? "" : "Worked?" }
const play = layers.slice(0, layer + 1).filter(k => k === "call").length
return narrow ? pcOrdinal(play) : `${pcOrdinal(play)} play`
}

// The SVG for one tree.
function pcSvg(tree, league, mode, width, who, label){
const lay = pcLayout(tree, mode, width)
const at = new Map(lay.nodes.map(b => [b.node.path, b]))
const links = [], marks = []
const draw = (node, from) => {
for(const child of node.children){
const b = at.get(child.path)
const x1 = from.x + from.w / 2, x2 = b.x - b.w / 2, mid = (x1 + x2) / 2
const d = `M${x1.toFixed(1)},${from.y.toFixed(1)} C${mid.toFixed(1)},${from.y.toFixed(1)} ${mid.toFixed(1)},${b.y.toFixed(1)} ${x2.toFixed(1)},${b.y.toFixed(1)}`
links.push(child.drives
? `<path class="pcLink" d="${d}" stroke="${pcColor(0.22 + 0.78 * child.share).fill}" stroke-width="${(1.2 + 4.3 * child.share).toFixed(1)}"/>`
: `<path class="pcLink pcLinkNone" d="${d}"/>`)
const c = pcColor(child.share), ink = c.light ? "pcOnDeep" : "pcOnPale"
const tip = pcEsc(pcTip(child, tree, league, mode, who))
const x = (b.x - b.w / 2).toFixed(1), y = (b.y - b.h / 2).toFixed(1)
let inner
if(child.kind === "result"){
const r = b.h / 2, k = r * 0.42
const glyph = child.step === "+"
? `M${(b.x - k).toFixed(1)},${b.y.toFixed(1)} l${(k * 0.7).toFixed(1)},${(k * 0.75).toFixed(1)} l${(k * 1.3).toFixed(1)},${(-k * 1.5).toFixed(1)}`
: `M${(b.x - k).toFixed(1)},${(b.y - k).toFixed(1)} l${(2 * k).toFixed(1)},${(2 * k).toFixed(1)} M${(b.x + k).toFixed(1)},${(b.y - k).toFixed(1)} l${(-2 * k).toFixed(1)},${(2 * k).toFixed(1)}`
inner = `<circle cx="${b.x.toFixed(1)}" cy="${b.y.toFixed(1)}" r="${r.toFixed(1)}" fill="${c.fill}"/><path class="pcGlyph" d="${glyph}"/>`
}else{
inner = `<rect x="${x}" y="${y}" width="${b.w.toFixed(1)}" height="${b.h.toFixed(1)}" rx="${Math.min(6, b.h / 2).toFixed(1)}" fill="${c.fill}"/>`
+ `<text x="${b.x.toFixed(1)}" y="${(b.y + 0.5).toFixed(1)}" style="font-size:${b.h >= 16 ? 11 : 9.5}px">${child.step === "R" ? "Run" : "Pass"}</text>`
}
marks.push(`<g class="pcNode ${ink}${child.drives ? "" : " pcNone"}" data-tip="${tip}" tabindex="0">${inner}</g>`)
draw(child, b)
}
}
draw(tree, lay.root)
const narrow = lay.columns.length > 1 && lay.columns[1] - lay.columns[0] < 84
const heads = lay.columns.map((x, i) => `<text class="pcHead" x="${x.toFixed(1)}" y="12">${pcColumnTitle(mode, i, narrow)}</text>`).join("")
const r = lay.root
const start = `<g class="pcNode pcRoot"><rect x="${(r.x - 26).toFixed(1)}" y="${(r.y - 11).toFixed(1)}" width="52" height="22" rx="6"/><text x="${r.x.toFixed(1)}" y="${(r.y + 0.5).toFixed(1)}" style="font-size:11px">Drive</text></g>`
return `<svg class="pcTree" viewBox="0 0 ${width} ${lay.height}" width="${width}" height="${lay.height}" role="img" aria-label="${pcEsc(label)}">${heads}${links.join("")}${start}${marks.join("")}</svg>`
}

// One team's card: who and how many drives, the two standouts in words,
// the scale, and the tree. opts: {mode: "plays" | "results", side:
// "offense" | "defense", width, names}.
function playCallingTreeHtml(data, team, opts = {}){
const mode = opts.mode === "results" ? "results" : "plays"
const side = opts.side === "defense" ? "defense" : "offense"
const name = (opts.names || (code => code))(team)
if(!data || !data.teams || !data.teams[team]){ return "" }
const tree = pcTree(data.teams[team][side], mode)
if(!tree.drives){ return "" }
const league = pcTree(data.league, mode)
const who = side === "offense" ? `${name} drives` : `drives against the ${name}`
const subject = side === "offense" ? "They open" : "Offenses open against them"
const {typical, signature} = pcStandouts(tree, league, mode)
const lines = []
if(typical){ lines.push(`Most typical start: <b>${pcEsc(pcPhrase(typical.path, mode))}</b> (${typical.drives} of ${tree.drives} drives).`) }
if(signature){ lines.push(`Furthest from the league: ${subject.toLowerCase()} with <b>${pcEsc(pcPhrase(signature.node.path, mode))}</b> on ${signature.node.drives} drives, where a typical team would on about ${Math.round(signature.usual)}.`) }
const label = `${name}, ${side === "offense" ? "with the ball" : "on defense"}: how ${tree.drives} drives opened.` + (typical ? ` Most typical start: ${pcPhrase(typical.path, mode)}.` : "")
const ramp = [0, 0.25, 0.5, 0.75, 1].map(t => pcColor(t).fill).join(",")
const key = `<span class="vizKey pcKey">rarer <i class="pcRamp" style="background:linear-gradient(90deg,${ramp})"></i> most typical`
+ (mode === "results" ? ` <i class="pcKeyMark">&#10003;</i>worked <i class="pcKeyMark">&#10005;</i>didn't` : "") + `</span>`
return `<div class="vizCard pcCard">
<div class="vizHead"><b>${pcEsc(name)}</b> <span class="pcSide">${side === "offense" ? "with the ball" : "on defense: what offenses call against them"} · ${tree.drives} drives through week ${pcEsc(data.through_week)}</span>${key}</div>
<div class="pcRead">${lines.map(l => `<p>${l}</p>`).join("")}</div>
<div class="pcScroll">${pcSvg(tree, league, mode, Math.round(opts.width || 720), who, label)}</div>
</div>`
}

if (typeof module !== "undefined" && module.exports) {
module.exports = { PC_LAYERS, pcColor, pcParse, pcSteps, pcTree, pcByLayer, pcPhrase, pcTip, pcFind, pcStandouts, pcLayout, pcColumnTitle, pcSvg, playCallingTreeHtml }
}
