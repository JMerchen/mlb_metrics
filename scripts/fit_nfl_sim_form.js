// Fits the football simulation's game-day form (SIM_FORM in
// docs/nfl_sim.js) to real games by simulation-based inference.
//
// Why (2026-11-20): a simulator has settings no formula connects to the
// data - there is no equation for "how spread out are the final scores"
// in terms of how much a team's day varies. The newer answer (simulation-
// based or "likelihood-free" inference: Wood 2010's synthetic likelihood,
// Cranmer, Brehmer and Louppe 2020 for the field) is to stop looking for
// the formula: draw candidate settings, run the simulator at each, and
// keep the settings whose simulated games look like the real ones, each
// weighted by how close it came. What comes out is not one best guess but
// the whole range of settings the data cannot tell apart.
//
// The data: every game from 2018 on with a closing spread and total
// (data/backtest/nfl_game_misses_replay.csv). What "look like" means: six
// numbers about how far final scores land from the closing lines - the
// standard deviation of the margin's miss and of the total's, and the
// shares of each that are small (within 4 points) and large (over 20).
// For each candidate, 40 matchups spanning the real spreads and totals are
// calibrated to their lines (as the page does) and played 300 times.
//
// RESULT (2026-11-20). Round 1 (240 candidates, team 0-0.30, game 0-0.30):
// team 0.085 (90% 0.039-0.106), game 0.029 (0.005-0.070), but only 4.6
// candidates' worth of weight - the range was far wider than the answer.
// Round 2 searched what was left standing:
//
//   2127 real games (2018 on); 160 candidate settings (team 0.04-0.13, game 0-0.08), 12,000 simulated games each
//   posterior: team 0.079 (90% 0.051-0.097), game 0.040 (90% 0.010-0.066); 23.4 candidates' worth of weight
//   closest candidates: team 0.088 game 0.032 (16%); team 0.084 game 0.045 (5%); team 0.083 game 0.039 (5%); team 0.077 game 0.058 (4%); team 0.090 game 0.015 (4%)
//
//   checked on a fresh run (60 matchups x 500 games), fitted setting team 0.08, game 0.04:
//                                  real    no form   fitted
//     margin sd                      12.77    12.27    12.72
//     total sd                       13.13    12.21    13.15
//     margin within 4                29.7%    28.2%    26.7%
//     margin beyond 20               11.9%     9.9%    11.1%
//     total within 4                 25.9%    27.2%    25.6%
//     total beyond 20                11.7%     9.5%    12.1%
//     decided by 3 or fewer *        24.7%    19.3%    19.0%
//     decided by 8 or fewer *        53.3%    48.3%    47.6%
//     decided by 17 or more *        25.6%    24.8%    25.8%
//     favorite by 3+ wins *          70.3%    69.9%    68.6%
//     favorite by 7+ wins *          80.3%    79.9%    77.8%
//     total within 9 of the line *   52.8%    55.4%    52.2%
//     33 points or fewer *           18.9%    18.1%    19.5%
//     58 points or more *            19.8%    17.8%    19.9%
//     (* not used in the fit)
//
// So SIM_FORM was {team: 0.08, game: 0.04}: each offense's day varies by
// about 8%, with a smaller day both sides share. The spread of margins
// and totals matched real games, including the numbers the fit never saw
// about the total. What it did NOT fix, and no amount of form could: real
// games finish within a field goal a quarter of the time (24.7%) and
// simulated ones 19%, with or without form.
//
// ROUND 3 (2026-11-20, after the late-game changes in docs/nfl_sim.js -
// playing for the kick, taking the points, the two-minute drive when
// level, SIM_LEAD_DRAG). Those changes closed the field-goal gap (see
// below) but a leader that slows down also narrows margins, so the form
// was refitted (--draws 72 --team 0.07,0.16 --game 0,0.06):
//
//   2127 real games (2018 on); 72 candidate settings (team 0.07-0.16, game 0-0.06), 12,000 simulated games each
//   posterior: team 0.101 (90% 0.079-0.120), game 0.024 (90% 0.001-0.053); 18.5 candidates' worth of weight
//   closest candidates: team 0.104 game 0.024 (11%); team 0.109 game 0.001 (9%); team 0.088 game 0.040 (7%); team 0.108 game 0.019 (7%); team 0.096 game 0.030 (7%)
//
//   checked on a fresh run (60 matchups x 500 games), fitted setting team 0.1, game 0.02:
//                                  real    no form   fitted
//     margin sd                      12.77    11.73    12.49
//     total sd                       13.13    12.11    13.10
//     margin within 4                29.7%    29.8%    28.0%
//     margin beyond 20               11.9%     8.8%    10.7%
//     total within 4                 25.9%    27.4%    25.5%
//     total beyond 20                11.7%     9.4%    12.1%
//     decided by 3 or fewer *        24.7%    24.7%    23.7%
//     decided by 8 or fewer *        53.3%    50.9%    49.2%
//     decided by 17 or more *        25.6%    23.1%    25.3%
//     favorite by 3+ wins *          70.3%    70.4%    69.2%
//     favorite by 7+ wins *          80.3%    80.9%    79.2%
//     total within 9 of the line *   52.8%    55.8%    52.3%
//     33 points or fewer *           18.9%    18.1%    19.3%
//     58 points or more *            19.8%    17.7%    19.9%
//     (* not used in the fit)
//
// SIM_FORM = {team: 0.10, game: 0.02}. Games decided by 3 or fewer: 23.7%
// against a real 24.7% (it was 19.0%); by exactly 3, 14.7% against 14.4%
// (it was 9.4%); ties 0.3% against 0.4% (1.5%); overtime 4.2% against
// 5.5% (7.8%). Fourth-quarter drives by the team ahead: 1.52 points when
// up 1-8 and 1.72 when up 9+, against a real 1.57 and 1.70 (1.81, 1.83).
// Still open: decided by 8 or fewer 49.2% against 53.3%, and the margin's
// spread 12.5 against 12.8.
//
// Usage (the simulation part splits across processes):
//   for i in 0 1 2 3; do node scripts/fit_nfl_sim_form.js --simulate --shard $i/4 --out DIR & done; wait
//   node scripts/fit_nfl_sim_form.js --combine --out DIR
// Add --team LOW,HIGH --game LOW,HIGH --draws N (to both steps) to search
// another range; round 1 was --team 0,0.30 --game 0,0.30 --draws 240,
// round 2 --team 0.04,0.13 --game 0,0.08 --draws 160.

const fs = require("fs")
const path = require("path")
const sim = require("../docs/nfl_sim.js")

const ROOT = path.join(__dirname, "..")
const PROFILES = JSON.parse(fs.readFileSync(path.join(ROOT, "docs", "data", "nfl_sim_profiles.json"), "utf8"))
const MATCHUPS = 40, CALIBRATION_GAMES = 200, GAMES = 300
// Round 1 searched team 0-0.30 and game 0-0.30 with 240 candidates, round
// 2 team 0.04-0.13 and game 0-0.08 with 160; round 3 (the defaults here)
// followed the late-game changes.
const args = process.argv.slice(2)
const value = name => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : null }
const span = (name, fallback) => (value(name) || fallback).split(",").map(Number)
const DRAWS = Number(value("--draws")) || 72
const PRIOR = {team: span("--team", "0.07,0.16"), game: span("--game", "0,0.06")}
const SUMMARIES = ["margin sd", "total sd", "margin within 4", "margin beyond 20", "total within 4", "total beyond 20"]

function realGames(){
const lines = fs.readFileSync(path.join(ROOT, "data", "backtest", "nfl_game_misses_replay.csv"), "utf8").trim().split("\n")
const head = lines[0].split(",")
const col = name => head.indexOf(name)
const out = []
for(const line of lines.slice(1)){
const f = line.split(",")
const g = {season: Number(f[col("season")]), spread: Number(f[col("spread_line")]), margin: Number(f[col("margin")]),
total: Number(f[col("total")]), line: Number(f[col("total_line")])}
if(g.season >= 2018 && [g.spread, g.margin, g.total, g.line].every(v => f.length && Number.isFinite(v)) && f[col("spread_line")] !== "" && f[col("total_line")] !== "" && f[col("total")] !== ""){ out.push(g) }
}
return out
}

const mean = a => a.reduce((x, y) => x + y, 0) / a.length
const sd = a => { const m = mean(a); return Math.sqrt(mean(a.map(v => (v - m) * (v - m)))) }
const share = (a, test) => a.filter(test).length / a.length

// The six numbers, from each game's margin miss and total miss.
function summaries(marginMiss, totalMiss){
return [sd(marginMiss), sd(totalMiss), share(marginMiss, v => Math.abs(v) <= 4), share(marginMiss, v => Math.abs(v) > 20),
share(totalMiss, v => Math.abs(v) <= 4), share(totalMiss, v => Math.abs(v) > 20)]
}

// Matchups spanning the real spreads (each with a real game's total).
function matchups(games, count){
const sorted = games.slice().sort((a, b) => a.spread - b.spread || a.line - b.line)
const teams = Object.keys(PROFILES.teams)
return Array.from({length: count}, (_, i) => {
const g = sorted[Math.floor((i + 0.5) * sorted.length / count)]
return {home: teams[(i * 7) % teams.length], away: teams[(i * 7 + 13) % teams.length], spread: g.spread, line: g.line}
})
}

// Every game of `games` each matchup, at one setting: margins and totals.
function play(form, list, games, seed){
const out = []
for(const m of list){
const lineups = {home: sim.simLineup(PROFILES, m.home, new Set(), null), away: sim.simLineup(PROFILES, m.away, new Set(), null)}
const cal = sim.simCalibrate(PROFILES, m.home, m.away, m.spread, m.line, {lineups, form, games: CALIBRATION_GAMES})
for(let i = 0; i < games; i++){
const g = sim.simulateGame(PROFILES, m.home, m.away, {seed: seed + i, edge: cal.edge, scale: cal.scale, lineups, form})
out.push({spread: m.spread, line: m.line, margin: g.final.home - g.final.away, total: g.final.home + g.final.away})
}
}
return out
}

const missSummaries = rows => summaries(rows.map(r => r.margin - r.spread), rows.map(r => r.total - r.line))

function priorDraw(i){
const rng = sim.simRng(7919 * (i + 1))
return {team: PRIOR.team[0] + rng() * (PRIOR.team[1] - PRIOR.team[0]), game: PRIOR.game[0] + rng() * (PRIOR.game[1] - PRIOR.game[0])}
}

function simulate(shard, shards, outDir){
const list = matchups(realGames(), MATCHUPS)
const rows = []
for(let i = shard; i < DRAWS; i += shards){
const form = priorDraw(i)
rows.push({i, ...form, s: missSummaries(play(form, list, GAMES, 50000))})
}
fs.mkdirSync(outDir, {recursive: true})
fs.writeFileSync(path.join(outDir, `draws_${shard}.json`), JSON.stringify(rows))
}

function combine(outDir){
const real = realGames()
const target = missSummaries(real)
// How far the real numbers would move with a different set of seasons'
// games (a bootstrap), plus the simulation's own noise.
const rng = sim.simRng(11)
const boots = Array.from({length: 400}, () => missSummaries(Array.from({length: real.length}, () => real[Math.floor(rng() * real.length)])))
const simGames = MATCHUPS * GAMES
const variance = SUMMARIES.map((_, k) => { const v = sd(boots.map(b => b[k])) ** 2; return v + 2 * v * real.length / simGames })
const draws = fs.readdirSync(outDir).filter(f => /^draws_\d+\.json$/.test(f)).flatMap(f => JSON.parse(fs.readFileSync(path.join(outDir, f), "utf8")))
const logw = draws.map(d => -0.5 * d.s.reduce((sum, v, k) => sum + (v - target[k]) ** 2 / variance[k], 0))
const top = Math.max(...logw)
const w = logw.map(v => Math.exp(v - top)), total = w.reduce((a, b) => a + b, 0)
const post = name => {
const m = draws.reduce((sum, d, i) => sum + w[i] * d[name], 0) / total
const sorted = draws.map((d, i) => [d[name], w[i] / total]).sort((a, b) => a[0] - b[0])
const at = q => { let c = 0; for(const [v, p] of sorted){ c += p; if(c >= q){ return v } } return sorted[sorted.length - 1][0] }
return {mean: m, low: at(0.05), high: at(0.95)}
}
const team = post("team"), game = post("game")
const ess = total * total / w.reduce((a, b) => a + b * b, 0)
const fmt = v => v.toFixed(3)
console.log(`${real.length} real games (2018 on); ${draws.length} candidate settings (team ${PRIOR.team.join("-")}, game ${PRIOR.game.join("-")}), ${simGames.toLocaleString()} simulated games each`)
console.log(`posterior: team ${fmt(team.mean)} (90% ${fmt(team.low)}-${fmt(team.high)}), game ${fmt(game.mean)} (90% ${fmt(game.low)}-${fmt(game.high)}); ${ess.toFixed(1)} candidates' worth of weight`)
const best = draws.map((d, i) => ({...d, w: w[i] / total})).sort((a, b) => b.w - a.w).slice(0, 5)
console.log("closest candidates: " + best.map(d => `team ${fmt(d.team)} game ${fmt(d.game)} (${(100 * d.w).toFixed(0)}%)`).join("; "))

// A fresh, larger run at the fitted setting and at none, next to the real
// games - on the six fitted numbers and on ones the fit never saw.
const fitted = {team: Number(team.mean.toFixed(2)), game: Number(game.mean.toFixed(2))}
const list = matchups(real, 60)
const check = rows => {
const mm = rows.map(r => r.margin - r.spread), tm = rows.map(r => r.total - r.line)
const fav = rows.filter(r => r.margin !== 0 && Math.abs(r.spread) >= 3)
const big = rows.filter(r => r.margin !== 0 && Math.abs(r.spread) >= 7)
return [...summaries(mm, tm),
share(rows, r => Math.abs(r.margin) <= 3), share(rows, r => Math.abs(r.margin) <= 8), share(rows, r => Math.abs(r.margin) >= 17),
share(fav, r => (r.margin > 0) === (r.spread > 0)), share(big, r => (r.margin > 0) === (r.spread > 0)),
share(tm, v => Math.abs(v) <= 9), share(rows, r => r.total <= 33), share(rows, r => r.total >= 58)]
}
const names = [...SUMMARIES, "decided by 3 or fewer", "decided by 8 or fewer", "decided by 17 or more",
"favorite by 3+ wins", "favorite by 7+ wins", "total within 9 of the line", "33 points or fewer", "58 points or more"]
const columns = {real: check(real), none: check(play({team: 0, game: 0}, list, 500, 900000)), fitted: check(play(fitted, list, 500, 900000))}
console.log(`\nchecked on a fresh run (60 matchups x 500 games), fitted setting team ${fitted.team}, game ${fitted.game}:`)
console.log("                               real    no form   fitted")
names.forEach((name, k) => {
const show = v => (k < 2 ? v.toFixed(2) : (100 * v).toFixed(1) + "%").padStart(8)
console.log(`  ${(k < 6 ? name : name + " *").padEnd(28)}${show(columns.real[k])} ${show(columns.none[k])} ${show(columns.fitted[k])}`)
})
console.log("  (* not used in the fit)")
}

if(require.main === module){
const outDir = value("--out")
if(!outDir){ throw new Error("--out DIR is required") }
if(args.includes("--simulate")){
const [shard, shards] = (value("--shard") || "0/1").split("/").map(Number)
simulate(shard, shards, outDir)
}else if(args.includes("--combine")){
combine(outDir)
}
}

module.exports = { summaries, matchups, priorDraw, realGames }
