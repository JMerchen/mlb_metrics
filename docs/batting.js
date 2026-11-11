// Step In page: draws the at-bat on a canvas and listens for swings. The
// rules, the pitches and what a swing does are in batting_game.js; the
// pitchers come from docs/data/mlb_sim_profiles.json.

// The view is from behind the plate. Everything is drawn 540 units tall
// and 720 wide (480 on a narrow screen, cropped at the sides), then scaled
// to the canvas.
const BAT_VIEW_HEIGHT = 540
const BAT_FOCAL = 900          // with the camera 10 feet back, a foot at the plate is 90 units
const BAT_CAMERA_BACK = 10
const BAT_CAMERA_HEIGHT = 2.6
const BAT_EYE_LEVEL = 300      // where the horizon sits in the view
const BAT_WINDUP = 1.0         // seconds from the start of the delivery to the ball leaving the hand
const BAT_CATCH = 1.04         // the ball is caught just past the plate
const BAT_SPEEDS = [["0.5", "Half speed"], ["0.75", "Three-quarter speed"], ["1", "Real speed"]]
const BAT_BANNERS = {ball: "BALL", called_strike: "STRIKE", swinging_strike: "STRIKE", foul: "FOUL", out: "OUT",
single: "SINGLE", double: "DOUBLE", triple: "TRIPLE", homer: "HOME RUN", walk: "WALK", strikeout: "STRIKEOUT"}

const bat = {
profiles: null, teams: [], pitcher: null, side: "R", speed: 0.5, game: null,
phase: "idle", phaseStart: 0, until: 0, lastFrame: 0,
pitch: null, release: 0, swing: null, result: null, outcome: null, mark: null, hit: null, banner: null,
cursor: {x: 0, z: 2.5, shown: false}, spray: [], width: 720, crowd: [],
}

function batEl(id){ return document.getElementById(id) }
function batEsc(s){ return String(s === undefined || s === null ? "" : s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c])) }

// A point on the field as a point in the view, and how big a foot is there.
function batProject(x, y, z){
const scale = BAT_FOCAL / (y + BAT_CAMERA_BACK)
return {sx: bat.width / 2 + x * scale, sy: BAT_EYE_LEVEL - (z - BAT_CAMERA_HEIGHT) * scale, scale}
}

// ---- Drawing ----

function batDrawScene(ctx){
const w = bat.width, cx = w / 2
const sky = ctx.createLinearGradient(0, 0, 0, 200)
sky.addColorStop(0, "#16233f"); sky.addColorStop(1, "#3d5a8c")
ctx.fillStyle = sky; ctx.fillRect(0, 0, w, 200)
// The stands, with the crowd as specks.
ctx.fillStyle = "#232a38"; ctx.fillRect(0, 150, w, 136)
bat.crowd.forEach(c => { ctx.fillStyle = c.color; ctx.fillRect(c.x * w, 154 + c.y * 126, 2, 2) })
// Light towers.
;[0.12, 0.88].forEach(f => {
ctx.fillStyle = "#4a5263"; ctx.fillRect(w * f - 2, 60, 4, 92)
ctx.fillStyle = "#fff6d6"; ctx.fillRect(w * f - 22, 44, 44, 18)
const glow = ctx.createRadialGradient(w * f, 53, 4, w * f, 53, 70)
glow.addColorStop(0, "rgba(255,246,214,.35)"); glow.addColorStop(1, "rgba(255,246,214,0)")
ctx.fillStyle = glow; ctx.fillRect(w * f - 70, 0, 140, 130)
})
// The dark backdrop in center field that hitters pick the ball up against.
ctx.fillStyle = "#0e2619"; ctx.fillRect(cx - 105, 150, 210, 136)
// The outfield wall.
ctx.fillStyle = "#17472d"; ctx.fillRect(0, 284, w, 22)
ctx.fillStyle = "#e9c94a"; ctx.fillRect(0, 284, w, 2)
// Grass, the mound, and the dirt around the plate.
const grass = ctx.createLinearGradient(0, 306, 0, BAT_VIEW_HEIGHT)
grass.addColorStop(0, "#2f7a3f"); grass.addColorStop(1, "#3f9550")
ctx.fillStyle = grass; ctx.fillRect(0, 306, w, BAT_VIEW_HEIGHT - 306)
ctx.fillStyle = "#b98552"
ctx.beginPath(); ctx.ellipse(cx, 327, 74, 9, 0, 0, Math.PI * 2); ctx.fill()
ctx.beginPath(); ctx.ellipse(cx, 590, 470, 150, 0, 0, Math.PI * 2); ctx.fill()
// Home plate.
const corners = [[-0.708, 1.417], [0.708, 1.417], [0.708, 0.708], [0, 0], [-0.708, 0.708]].map(c => batProject(c[0], c[1], 0))
ctx.fillStyle = "#f4f1e8"
ctx.beginPath(); corners.forEach((c, i) => i ? ctx.lineTo(c.sx, c.sy) : ctx.moveTo(c.sx, c.sy)); ctx.closePath(); ctx.fill()
}

// The pitcher, a simple figure on the mound. `wind` is how far through his
// delivery he is (0 set, 1 the ball leaving his hand, past 1 following through).
function batDrawPitcher(ctx, wind){
const hand = bat.pitcher && bat.pitcher.throws === "L" ? -1 : 1
const foot = batProject(0, 58, 0.83), s = foot.scale * 1.2, x = foot.sx      // drawn a little larger than life, to be seen
const drive = Math.max(0, Math.min(1, (wind - 0.6) / 0.4))            // striding down the mound
const lift = wind < 0.6 ? Math.sin(Math.PI * Math.max(0, wind) / 0.6) : 0   // the leg kick
const drop = 0.45 * s * drive
const hip = foot.sy - 3.0 * s + drop, shoulder = foot.sy - 5.0 * s + drop, head = foot.sy - 5.75 * s + drop
ctx.lineCap = "round"
// Legs: the back leg stays on the rubber, the front one kicks and strides.
ctx.strokeStyle = "#d9dde6"; ctx.lineWidth = 0.42 * s
ctx.beginPath(); ctx.moveTo(x - hand * 0.2 * s, hip); ctx.lineTo(x - hand * 0.3 * s, foot.sy); ctx.stroke()
const kneeX = x + hand * (0.35 + 0.5 * drive) * s, kneeY = hip + (1.5 - 1.3 * lift) * s
ctx.beginPath(); ctx.moveTo(x + hand * 0.2 * s, hip); ctx.lineTo(kneeX, kneeY)
ctx.lineTo(x + hand * (0.45 + 1.0 * drive) * s, foot.sy + 0.15 * s * drive - 1.0 * s * lift); ctx.stroke()
// Body and head.
ctx.lineWidth = 0.95 * s
ctx.beginPath(); ctx.moveTo(x, hip); ctx.lineTo(x, shoulder + 0.35 * s); ctx.stroke()
ctx.fillStyle = "#e8c9a6"; ctx.beginPath(); ctx.arc(x, head, 0.36 * s, 0, Math.PI * 2); ctx.fill()
ctx.fillStyle = "#1d2a44"; ctx.beginPath(); ctx.arc(x, head - 0.08 * s, 0.38 * s, Math.PI, Math.PI * 2); ctx.fill()
// Glove arm, tucked.
ctx.strokeStyle = "#d9dde6"; ctx.lineWidth = 0.34 * s
ctx.beginPath(); ctx.moveTo(x + hand * 0.5 * s, shoulder + 0.3 * s); ctx.lineTo(x + hand * 0.75 * s, shoulder + 1.1 * s); ctx.stroke()
ctx.fillStyle = "#6b4423"; ctx.beginPath(); ctx.arc(x + hand * 0.75 * s, shoulder + 1.15 * s, 0.3 * s, 0, Math.PI * 2); ctx.fill()
// Throwing arm: hands together, back behind him, over the top to the release point, then across his body.
const from = {sx: x - hand * 0.5 * s, sy: shoulder + 0.3 * s}
const release = bat.pitch ? batProject(bat.pitch.x0, BAT_RELEASE_DISTANCE, bat.pitch.z0) : batProject(BAT_RELEASE[0] * hand, BAT_RELEASE_DISTANCE, BAT_RELEASE[1])
const keys = [[0, x + hand * 0.4 * s, shoulder + 1.0 * s], [0.5, x + hand * 0.4 * s, shoulder + 1.0 * s],
[0.82, x - hand * 1.9 * s, shoulder + 0.9 * s], [1, release.sx, release.sy], [1.25, x + hand * 0.9 * s, hip + 0.6 * s]]
const at = Math.max(0, Math.min(1.25, wind))
let k = 0
while(k < keys.length - 2 && at > keys[k + 1][0]){ k++ }
const f = (at - keys[k][0]) / (keys[k + 1][0] - keys[k][0])
const hx = keys[k][1] + (keys[k + 1][1] - keys[k][1]) * f, hy = keys[k][2] + (keys[k + 1][2] - keys[k][2]) * f
ctx.beginPath(); ctx.moveTo(from.sx, from.sy); ctx.lineTo(hx, hy); ctx.stroke()
if(wind < 1){ ctx.fillStyle = "#ffffff"; ctx.beginPath(); ctx.arc(hx, hy, 2.2, 0, Math.PI * 2); ctx.fill() }
}

function batDrawZone(ctx){
const a = batProject(-0.708, 0, 3.5), b = batProject(0.708, 0, 1.5)
ctx.fillStyle = "rgba(255,255,255,.05)"; ctx.fillRect(a.sx, a.sy, b.sx - a.sx, b.sy - a.sy)
ctx.strokeStyle = "rgba(255,255,255,.75)"; ctx.lineWidth = 2
ctx.strokeRect(a.sx, a.sy, b.sx - a.sx, b.sy - a.sy)
}

function batDrawBall(ctx, sx, sy, radius){
ctx.fillStyle = "rgba(0,0,0,.25)"; ctx.beginPath(); ctx.arc(sx + radius * 0.2, sy + radius * 0.25, radius, 0, Math.PI * 2); ctx.fill()
ctx.fillStyle = "#ffffff"; ctx.beginPath(); ctx.arc(sx, sy, radius, 0, Math.PI * 2); ctx.fill()
if(radius > 8){
ctx.strokeStyle = "#c8102e"; ctx.lineWidth = Math.max(1, radius * 0.14)
ctx.beginPath(); ctx.arc(sx - radius * 1.1, sy, radius, -0.7, 0.7); ctx.stroke()
ctx.beginPath(); ctx.arc(sx + radius * 1.1, sy, radius, Math.PI - 0.7, Math.PI + 0.7); ctx.stroke()
}
}

// Where the last pitch crossed (a ring: red for a strike, green for a
// ball, gold for one put in play) and where the bat was (a cross).
function batDrawMark(ctx){
const m = bat.mark
if(!m){ return }
const p = batProject(m.x, 0, m.z)
ctx.strokeStyle = m.kind === "ball" ? "#6fdc8c" : m.kind === "called_strike" || m.kind === "swinging_strike" ? "#ff6b6b" : "#ffd166"
ctx.lineWidth = 3
ctx.beginPath(); ctx.arc(p.sx, p.sy, 0.121 * 90, 0, Math.PI * 2); ctx.stroke()
if(m.swing){
const q = batProject(m.swing.x, 0, m.swing.z)
ctx.strokeStyle = "rgba(255,255,255,.9)"; ctx.lineWidth = 2
ctx.beginPath(); ctx.moveTo(q.sx - 7, q.sy - 7); ctx.lineTo(q.sx + 7, q.sy + 7); ctx.moveTo(q.sx + 7, q.sy - 7); ctx.lineTo(q.sx - 7, q.sy + 7); ctx.stroke()
}
}

// The bat's aim (mouse only): the ring is the part of the bat that hits it hard.
function batDrawCursor(ctx){
if(!bat.cursor.shown || bat.phase === "idle" || bat.phase === "over"){ return }
const p = batProject(bat.cursor.x, 0, bat.cursor.z)
ctx.strokeStyle = "rgba(255,209,102,.9)"; ctx.lineWidth = 2
ctx.beginPath(); ctx.arc(p.sx, p.sy, 0.3 * 90, 0, Math.PI * 2); ctx.stroke()
ctx.fillStyle = "rgba(255,209,102,.9)"; ctx.beginPath(); ctx.arc(p.sx, p.sy, 2.5, 0, Math.PI * 2); ctx.fill()
}

// The bat coming through, for a moment after a swing.
function batDrawSwing(ctx, now){
const s = bat.swing
if(!s || now - s.at > 0.18){ return }
const p = batProject(s.x, 0, s.z)
const f = (now - s.at) / 0.18, side = bat.side === "L" ? -1 : 1
const angle = (side > 0 ? Math.PI : 0) + side * (-1.1 + 2.0 * f)
ctx.save(); ctx.translate(p.sx - side * 150, p.sy + 4); ctx.rotate(angle - (side > 0 ? Math.PI : 0))
ctx.strokeStyle = "#d8a45e"; ctx.lineCap = "round"; ctx.lineWidth = 13
ctx.beginPath(); ctx.moveTo(side * 30, 0); ctx.lineTo(side * 190, 0); ctx.stroke()
ctx.restore()
}

// A batted ball leaving: off toward where it was hit, shrinking.
function batDrawHit(ctx, now){
const h = bat.hit
if(!h){ return }
const u = Math.min(1, (now - h.start) / 0.95)
const r = h.result, cx = bat.width / 2
let tx, ty, arc
if(r.kind === "foul"){ tx = cx + (r.spray < 0 ? -1 : 1) * (bat.width / 2 + 40); ty = 170; arc = 60 }
else{
tx = cx + r.spray / 45 * bat.width * 0.42
ty = r.launch < 8 ? 318 : r.launch < 24 ? 292 : r.kind === "homer" ? 120 : 215
arc = r.launch < 8 ? -6 : r.launch < 24 ? 40 : 190
}
const sx = h.sx + (tx - h.sx) * u, sy = h.sy + (ty - h.sy) * u - Math.sin(Math.PI * u) * arc
if(u < 1){ batDrawBall(ctx, sx, sy, Math.max(2, 10.5 * (1 - u) + 2 * u)) }
}

// Every ball put in play this game, on a small field seen from above.
function batDrawSpray(ctx){
if(!bat.spray.length){ return }
const size = 96, ox = 12 + size / 2, oy = 12 + size
ctx.fillStyle = "rgba(10,14,22,.62)"; ctx.fillRect(8, 8, size + 8, size + 10)
const k = (size - 6) / 440
const point = (spray, distance) => {
const a = (spray - 90) * Math.PI / 180
return [ox + Math.cos(a) * distance * k, oy + Math.sin(a) * distance * k]
}
ctx.fillStyle = "#2f7a3f"; ctx.beginPath(); ctx.moveTo(ox, oy)
for(let a = -45; a <= 45; a += 5){ const p = point(a, batFence(a)); ctx.lineTo(p[0], p[1]) }
ctx.closePath(); ctx.fill()
bat.spray.forEach(b => {
const p = point(b.spray, Math.min(b.distance, 440))
ctx.fillStyle = b.kind === "out" ? "#c9ced8" : b.kind === "homer" ? "#ffd166" : "#ffffff"
ctx.beginPath(); ctx.arc(p[0], p[1], b.kind === "out" ? 2.2 : 3, 0, Math.PI * 2); ctx.fill()
})
}

function batDrawBanner(ctx, now){
const b = bat.banner
if(!b || now > b.until){ return }
const fade = Math.min(1, (b.until - now) / 0.3)
ctx.globalAlpha = fade
ctx.font = `900 ${b.big ? 54 : 38}px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, Helvetica, Arial, sans-serif`
ctx.textAlign = "center"; ctx.lineJoin = "round"
ctx.lineWidth = 6; ctx.strokeStyle = "rgba(0,0,0,.55)"; ctx.strokeText(b.text, bat.width / 2, 112)
ctx.fillStyle = b.good ? "#ffd166" : "#ffffff"; ctx.fillText(b.text, bat.width / 2, 112)
ctx.globalAlpha = 1
}

function batDraw(now){
const canvas = batEl("batCanvas"), ctx = canvas.getContext("2d")
const k = canvas.width / bat.width
ctx.setTransform(k, 0, 0, k, 0, 0)
batDrawScene(ctx)
const wind = bat.phase === "windup" ? (now - bat.phaseStart) / BAT_WINDUP
: bat.phase === "flight" || bat.phase === "result" ? 1 + Math.min(0.25, now - bat.release) : 0
batDrawPitcher(ctx, wind)
batDrawZone(ctx)
if(bat.phase !== "flight"){ batDrawMark(ctx) }
if(bat.phase === "flight" && !(bat.result && bat.hit)){
const progress = Math.min(BAT_CATCH, (now - bat.release) * bat.speed / bat.pitch.flight)
// A short trail, so the path can be read.
for(let i = 3; i >= 0; i--){
const s = progress - i * 0.035
if(s <= 0){ continue }
const b = batBallAt(bat.pitch, s), p = batProject(b.x, b.y, b.z)
if(i){ ctx.fillStyle = `rgba(255,255,255,${0.22 - i * 0.05})`; ctx.beginPath(); ctx.arc(p.sx, p.sy, Math.max(2, 0.121 * p.scale), 0, Math.PI * 2); ctx.fill() }
else{ batDrawBall(ctx, p.sx, p.sy, Math.max(2.6, 0.121 * p.scale * 1.1)) }
}
}
batDrawHit(ctx, now)
batDrawSwing(ctx, now)
batDrawCursor(ctx)
batDrawSpray(ctx)
batDrawBanner(ctx, now)
}

// ---- The scoreboard and the words around the canvas ----

function batRenderBoard(){
const g = bat.game || batNewGame()
batEl("batInning").textContent = `${Math.min(g.inning, g.innings)} of ${g.innings}`
batEl("batOuts").textContent = "●".repeat(g.over ? 3 : g.outs) + "○".repeat(g.over ? 0 : 3 - g.outs)
batEl("batCount").textContent = `${g.balls}–${g.strikes}`
batEl("batRuns").textContent = g.runs
;["batFirst", "batSecond", "batThird"].forEach((id, i) => batEl(id).classList.toggle("on", Boolean(g.bases[i]) && !g.over))
}

function batRenderScout(){
const p = bat.pitcher
if(!p){ batEl("batScout").innerHTML = ""; return }
const arsenal = batArsenal(p, bat.side)
const fastest = arsenal.slice().sort((x, y) => y.velo - x.velo)[0]
const seconds = batFlightSeconds(fastest.velo)
batEl("batScout").innerHTML = `<div class="batScoutName"><b>${batEsc(p.name)}</b> <span class="dim">throws ${p.throws === "L" ? "left" : "right"}-handed</span></div>
<div class="batPitches">${arsenal.map(a => `<span class="batPitch"><b>${batEsc(a.name)}</b> ${Math.round(a.velo)} mph <span class="dim">${Math.round(a.share * 100)}%</span></span>`).join("")}</div>
<div class="batScoutLine dim">How often he throws each to a ${bat.side === "L" ? "left" : "right"}-handed batter. At real speed his ${batEsc(fastest.name.toLowerCase())} reaches the plate in ${seconds.toFixed(2)} seconds.</div>`
}

function batBest(){
try{ return JSON.parse(localStorage.getItem("srBatBest") || "{}") || {} }catch(e){ return {} }
}

function batRenderSummary(){
const g = bat.game, box = batEl("batSummary")
if(!g || !g.over){ box.style.display = "none"; return }
const best = batBest(), key = String(bat.speed)
const record = best[key] === undefined || g.runs > best[key]
if(record){
best[key] = g.runs
try{ localStorage.setItem("srBatBest", JSON.stringify(best)) }catch(e){ /* private window: the best score just isn't kept */ }
}
const speed = BAT_SPEEDS.find(s => s[0] === key)[1].toLowerCase()
const lines = [`${g.hits} hit${g.hits === 1 ? "" : "s"}`, `${g.homers} home run${g.homers === 1 ? "" : "s"}${g.longest ? ` (longest ${g.longest} feet)` : ""}`,
`${g.walks} walk${g.walks === 1 ? "" : "s"}`, `${g.strikeouts} strikeout${g.strikeouts === 1 ? "" : "s"}`]
if(g.hardest){ lines.push(`hardest hit ${g.hardest} mph`) }
box.style.display = ""
box.innerHTML = `<div class="batFinal"><b>${g.runs} run${g.runs === 1 ? "" : "s"}</b> off ${batEsc(bat.pitcher.name)} in ${g.innings} inning${g.innings === 1 ? "" : "s"} at ${speed}.</div>
<div class="batFinalLine">${lines.join(" &middot; ")}, on ${g.pitches} pitches.</div>
<div class="batFinalLine dim">${record && g.runs > 0 ? "That's your best at this speed." : `Your best at this speed: ${best[key]} run${best[key] === 1 ? "" : "s"}.`}</div>`
}

// ---- The game loop ----

function batSetPhase(phase, now, wait){
bat.phase = phase
bat.phaseStart = now
bat.until = now + (wait || 0)
}

function batStart(){
const now = performance.now() / 1000
bat.game = batNewGame()
bat.spray = []; bat.mark = null; bat.hit = null; bat.swing = null; bat.result = null; bat.pitch = null; bat.banner = null
batEl("batStart").style.display = "none"
batEl("batCall").textContent = `${bat.pitcher.name} is ready. Swing by clicking or tapping where the ball crosses, as it gets there.`
batRenderBoard(); batRenderSummary()
batSetPhase("wait", now, 1.0)
}

// Back to before the first pitch (a new pitcher, side or speed).
function batReset(){
bat.game = null; bat.phase = "idle"; bat.pitch = null; bat.swing = null; bat.result = null; bat.hit = null; bat.mark = null; bat.banner = null; bat.spray = []
const button = batEl("batStart")
button.style.display = ""; button.textContent = "Play ball"
batEl("batCall").textContent = "Three innings. Take the balls, swing at the strikes."
batRenderBoard(); batRenderSummary(); batRenderScout()
}

// The pitch is over: score it and say what happened.
function batSettle(result, now){
const pitch = bat.pitch
const outcome = batApply(bat.game, result)
bat.game = outcome.game
bat.result = result; bat.outcome = outcome
bat.mark = {x: pitch.x, z: pitch.z, kind: result.kind, swing: bat.swing ? {x: bat.swing.x, z: bat.swing.z} : null}
const inPlay = result.exitVelo !== undefined
if(inPlay){ bat.spray.push({spray: result.spray, distance: result.distance, kind: result.kind}) }
const key = outcome.ended === "walk" || outcome.ended === "strikeout" ? outcome.ended : result.kind
const good = ["single", "double", "triple", "homer", "walk"].includes(key)
bat.banner = {text: BAT_BANNERS[key] + (good && key !== "walk" ? "!" : ""), until: now + (inPlay ? 1.5 : 1.0), big: key === "homer", good}
let text = batCallText(pitch, result, outcome)
if(outcome.game.over){ text += " That's the game." }
else if(outcome.inningOver){ text += ` Three outs. On to inning ${outcome.game.inning}.` }
batEl("batCall").textContent = text
batRenderBoard()
batSetPhase("result", now, inPlay ? 1.8 : result.kind === "foul" ? 1.3 : 1.2)
}

function batSwingAt(x, z){
const now = performance.now() / 1000
if(bat.phase !== "flight" || bat.swing){ return }
const time = (now - bat.release) * bat.speed
bat.swing = {time, x, z, at: now}
const result = batSwingResult(bat.pitch, bat.swing, bat.side, Math.random)
if(result.kind === "swinging_strike"){ bat.result = result; return }      // the ball carries on into the glove
const b = batBallAt(bat.pitch, Math.min(BAT_CATCH, time / bat.pitch.flight)), p = batProject(b.x, b.y, b.z)
bat.hit = {start: now, sx: p.sx, sy: p.sy, result}
batSettle(result, now)
}

function batFrame(ms){
const now = ms / 1000
// A long gap between frames (the tab was in the background): throw the pitch again.
if(now - bat.lastFrame > 0.5 && (bat.phase === "windup" || bat.phase === "flight")){
bat.pitch = null; bat.swing = null; bat.result = null
batSetPhase("wait", now, 0.8)
}
bat.lastFrame = now
if(bat.phase === "wait" && now >= bat.until){
bat.pitch = batNextPitch(bat.pitcher, bat.side, bat.game.balls, bat.game.strikes, Math.random)
bat.swing = null; bat.result = null; bat.hit = null
batSetPhase("windup", now)
}
if(bat.phase === "windup" && now - bat.phaseStart >= BAT_WINDUP){
bat.release = now; bat.mark = null
batSetPhase("flight", now)
}
if(bat.phase === "flight" && (now - bat.release) * bat.speed / bat.pitch.flight >= BAT_CATCH){
batSettle(bat.result || batSwingResult(bat.pitch, null, bat.side, Math.random), now)
}
if(bat.phase === "result" && now >= bat.until){
bat.hit = null
if(bat.game.over){
bat.phase = "over"
const button = batEl("batStart")
button.style.display = ""; button.textContent = "Play again"
batRenderSummary()
}else{ batSetPhase("wait", now, 0.45 + Math.random() * 0.6) }
}
batDraw(now)
requestAnimationFrame(batFrame)
}

// ---- Setup ----

function batSizeCanvas(){
const canvas = batEl("batCanvas"), stage = batEl("batStage")
const shown = Math.min(stage.clientWidth || 720, 720)
bat.width = shown < 520 ? 480 : 720
const ratio = Math.min(window.devicePixelRatio || 1, 2)
canvas.style.width = `${shown}px`
canvas.style.height = `${Math.round(shown * BAT_VIEW_HEIGHT / bat.width)}px`
canvas.width = Math.round(shown * ratio)
canvas.height = Math.round(shown * BAT_VIEW_HEIGHT / bat.width * ratio)
}

// Where on the plate a point on the canvas is.
function batPlatePoint(event){
const rect = batEl("batCanvas").getBoundingClientRect()
const lx = (event.clientX - rect.left) * bat.width / rect.width, ly = (event.clientY - rect.top) * BAT_VIEW_HEIGHT / rect.height
return {x: (lx - bat.width / 2) / 90, z: BAT_CAMERA_HEIGHT - (ly - BAT_EYE_LEVEL) / 90}
}

function batFillPitchers(selectId){
const team = bat.teams.find(t => t.team === batEl("batTeam").value) || bat.teams[0]
const group = (label, role) => {
const rows = team.pitchers.filter(p => p.role === role)
return rows.length ? `<optgroup label="${label}">${rows.map(p => `<option value="${p.id}">${batEsc(p.name)}</option>`).join("")}</optgroup>` : ""
}
const select = batEl("batPitcher")
select.innerHTML = group("Starters", "starter") + group("Relievers", "reliever")
if(selectId && team.pitchers.some(p => p.id === selectId)){ select.value = selectId }
bat.pitcher = bat.profiles.pitchers[select.value]
}

function batChoose(team, pitcherId){
batEl("batTeam").value = team
batFillPitchers(pitcherId)
batReset()
}

async function loadBatting(){
bat.crowd = Array.from({length: 520}, (_, i) => {
const r = n => { const v = Math.sin(i * 12.9898 + n * 78.233) * 43758.5453; return v - Math.floor(v) }
return {x: r(1), y: r(2), color: ["#5b6477", "#7a8499", "#3c4454", "#9aa3b5", "#8a5a5a", "#5a6f8a"][Math.floor(r(3) * 6)]}
})
batSizeCanvas()
batRenderBoard()
batDraw(0)
try{
const response = await fetch(`./data/mlb_sim_profiles.json?t=${Date.now()}`, {cache: "no-store"})
if(!response.ok){ throw new Error(`mlb_sim_profiles.json ${response.status}`) }
bat.profiles = await response.json()
}catch(e){
console.log("no mlb_sim_profiles.json", e)
batEl("batCall").textContent = "The pitchers couldn't be loaded. Try again in a moment."
return
}
bat.teams = batPitchersByTeam(bat.profiles)
if(!bat.teams.length){ batEl("batCall").textContent = "No pitchers on file yet."; return }
batEl("batTeam").innerHTML = bat.teams.map(t => `<option value="${t.team}">${batEsc(typeof teamName === "function" ? teamName(t.team) : t.team)}</option>`).join("")
batEl("batSpeed").innerHTML = BAT_SPEEDS.map(s => `<option value="${s[0]}">${s[1]}</option>`).join("")
batEl("batSpeed").value = String(bat.speed)
// Opens on the hardest-throwing regular starter.
const regulars = [].concat(...bat.teams.map(t => t.pitchers.filter(p => p.role === "starter" && p.starts >= 15).map(p => ({...p, team: t.team}))))
const first = regulars.sort((x, y) => y.top - x.top)[0] || {...bat.teams[0].pitchers[0], team: bat.teams[0].team}
batChoose(first.team, first.id)
batEl("batTeam").onchange = () => { batFillPitchers(null); batReset() }
batEl("batPitcher").onchange = () => { bat.pitcher = bat.profiles.pitchers[batEl("batPitcher").value]; batReset() }
batEl("batSide").onchange = () => { bat.side = batEl("batSide").value; batReset() }
batEl("batSpeed").onchange = () => { bat.speed = Number(batEl("batSpeed").value); batReset() }
batEl("batRandom").onclick = () => {
const team = bat.teams[Math.floor(Math.random() * bat.teams.length)]
batChoose(team.team, team.pitchers[Math.floor(Math.random() * team.pitchers.length)].id)
}
batEl("batStart").onclick = batStart
const canvas = batEl("batCanvas")
canvas.addEventListener("pointermove", event => {
if(event.pointerType === "touch"){ return }
Object.assign(bat.cursor, batPlatePoint(event), {shown: true})
})
canvas.addEventListener("pointerleave", () => { bat.cursor.shown = false })
canvas.addEventListener("pointerdown", event => {
const point = batPlatePoint(event)
Object.assign(bat.cursor, point, {shown: event.pointerType !== "touch"})
if(bat.phase === "flight"){ event.preventDefault(); batSwingAt(point.x, point.z) }
})
document.addEventListener("keydown", event => {
if(event.code !== "Space" || ["SELECT", "INPUT", "BUTTON", "TEXTAREA", "A"].includes((event.target.tagName || ""))){ return }
if(bat.phase === "idle" || bat.phase === "over"){ event.preventDefault(); batStart() }
else{ event.preventDefault(); batSwingAt(bat.cursor.x, bat.cursor.z) }
})
window.addEventListener("resize", batSizeCanvas)
requestAnimationFrame(batFrame)
}

loadBatting()
