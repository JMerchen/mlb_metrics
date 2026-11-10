// College Hoops page: pure rendering and the matchup math (no DOM, so it
// is unit-tested in cbb_render.test.js). cbb.js loads docs/data/cbb.json
// (src/mlb_metrics/cbb_pipeline.py) and wires these to the page.
//
// A team's rating is points per 100 possessions better than the average
// Division I team: `offense` scored above average, `defense` allowed below
// it (so a good defense is positive), `net` the two added.

function cbbEsc(s){ return String(s === undefined || s === null ? "" : s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])) }

// Standard normal CDF (Abramowitz-Stegun 7.1.26; within 1e-7).
function cbbNormCdf(x){
const t = 1 / (1 + 0.3275911 * Math.abs(x) / Math.SQRT2)
const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x / 2)
return x >= 0 ? (1 + y) / 2 : (1 - y) / 2
}

// "61%", never 0% or 100% for something that can still happen. The
// result goes into HTML, so the "<" and ">" are entities.
function cbbPct(p){
if(p === null || p === undefined || Number.isNaN(Number(p))){ return "-" }
const v = Number(p)
if(v <= 0){ return "0%" }
if(v >= 1){ return "100%" }
if(v < 0.005){ return "&lt;1%" }
if(v > 0.995){ return "&gt;99%" }
return `${Math.round(v * 100)}%`
}

function cbbSigned(x, digits = 1){
const v = Number(x)
if(Number.isNaN(v)){ return "-" }
const shown = Math.abs(v).toFixed(digits)
return Number(shown) === 0 ? shown : `${v > 0 ? "+" : "−"}${shown}`
}

function cbbOrdinal(n){
const v = n % 100
return `${n}${v >= 11 && v <= 13 ? "th" : ({1: "st", 2: "nd", 3: "rd"}[n % 10] || "th")}`
}

// Team a against team b. `site` is "neutral", "a" (a at home) or "b".
// Returns the expected margin for a, each side's expected points, the pace
// and a's win chance - the same sums as cbb_ratings.predict_games.
function cbbMatchup(a, b, site, league){
const possessions = a.tempo + b.tempo - league.tempo
const edge = site === "a" ? league.home_edge : site === "b" ? -league.home_edge : 0
const pointsA = possessions / 100 * (league.efficiency + a.offense - b.defense + edge)
const pointsB = possessions / 100 * (league.efficiency + b.offense - a.defense - edge)
const margin = (a.net - b.net + 2 * edge) / 100 * possessions
return {margin, possessions, pointsA, pointsB, winA: cbbNormCdf(margin / league.sigma)}
}

function cbbTeamIndex(teams){
const byId = new Map()
;(teams || []).forEach(t => byId.set(Number(t.team_id), t))
return byId
}

function cbbDot(team){
const color = team && /^[0-9a-fA-F]{6}$/.test(team.color || "") ? `#${team.color}` : "var(--text-dim)"
return `<span class="cbbDot" style="background:${color}"></span>`
}

// A team's chance of beating an average Division I team on a neutral
// floor, from its rating and pace (the same idea as Bart Torvik's
// "Barthag", computed from our own numbers).
function cbbBeatsAverage(team, league){
return cbbNormCdf(team.net / 100 * team.tempo / league.sigma)
}

// The rankings table. `filter` = {search, conference, limit, league};
// with `league`, Deep dive adds each team's adjusted efficiency (points
// scored and allowed per 100 possessions against an average team) and its
// chance of beating an average team.
function cbbRankingsHtml(teams, filter = {}){
const league = filter.league
const search = (filter.search || "").trim().toLowerCase()
let rows = (teams || []).filter(t => (!filter.conference || t.conference === filter.conference)
&& (!search || t.team.toLowerCase().includes(search)))
const total = rows.length
if(filter.limit){ rows = rows.slice(0, filter.limit) }
if(!rows.length){ return '<div class="empty-note">No teams match.</div>' }
const body = rows.map(t => `<tr>
<td>${t.rank}</td>
<td class="cbbTeamCell">${cbbDot(t)}${cbbEsc(t.team)}</td>
<td>${cbbEsc(t.conference)}</td>
<td>${t.wins}–${t.losses}</td>
<td><b>${cbbSigned(t.net)}</b></td>
<td>${cbbSigned(t.offense)} <span class="dim">${cbbOrdinal(t.offense_rank)}</span></td>
<td>${cbbSigned(t.defense)} <span class="dim">${cbbOrdinal(t.defense_rank)}</span></td>
${league ? `<td class="deep-only">${(league.efficiency + t.offense).toFixed(1)}</td>
<td class="deep-only">${(league.efficiency - t.defense).toFixed(1)}</td>
<td class="deep-only">${cbbPct(cbbBeatsAverage(t, league))}</td>` : ""}
<td class="deep-only">${Number(t.tempo).toFixed(1)}</td>
<td class="deep-only">${t.last_season_net === null || t.last_season_net === undefined ? "-" : cbbSigned(t.last_season_net)}</td>
</tr>`).join("")
return `<table class="cbbTable">
<thead><tr><th>#</th><th>Team</th><th>Conference</th><th>Record</th><th>Rating</th><th>Offense</th><th>Defense</th>${league ? '<th class="deep-only" title="Points per 100 possessions against an average defense">Scores</th><th class="deep-only" title="Points allowed per 100 possessions against an average offense">Allows</th><th class="deep-only" title="Chance of beating an average Division I team on a neutral floor">Beats avg</th>' : ""}<th class="deep-only">Pace</th><th class="deep-only">Last season</th></tr></thead>
<tbody>${body}</tbody>
</table>${filter.limit && total > rows.length ? `<div class="cbbMore">Showing ${rows.length} of ${total}.</div>` : ""}`
}

// The matchup tool's answer for two teams.
function cbbMatchupHtml(a, b, site, league){
if(!a || !b || a.team_id === b.team_id){ return '<div class="empty-note">Pick two different teams.</div>' }
const m = cbbMatchup(a, b, site, league)
const fav = m.winA >= 0.5 ? a : b, dog = m.winA >= 0.5 ? b : a
const favWin = Math.max(m.winA, 1 - m.winA)
const favPts = m.winA >= 0.5 ? m.pointsA : m.pointsB, dogPts = m.winA >= 0.5 ? m.pointsB : m.pointsA
const where = site === "neutral" ? "on a neutral floor" : `at ${cbbEsc((site === "a" ? a : b).team)}`
const by = Math.abs(m.margin)
const edges = []
const versus = (off, def) => {
const gap = off.offense - def.defense
if(Math.abs(gap) >= 4){
edges.push(gap > 0
? `${cbbEsc(off.team)}'s offense (${cbbOrdinal(off.offense_rank)}) has the edge on ${cbbEsc(def.team)}'s defense (${cbbOrdinal(def.defense_rank)})`
: `${cbbEsc(def.team)}'s defense (${cbbOrdinal(def.defense_rank)}) has the edge on ${cbbEsc(off.team)}'s offense (${cbbOrdinal(off.offense_rank)})`)
}
}
versus(a, b); versus(b, a)
return `<div class="cbbMatchup">
<div class="cbbMatchupHead">${cbbEsc(fav.team)} <b>${cbbPct(favWin)}</b> <span class="dim">${where}</span></div>
<div class="pickBarTrack cbbBar"><span class="pickBarAway${m.winA >= 0.5 ? " picked" : ""}" style="width:${(m.winA * 100).toFixed(1)}%"></span><span class="pickBarHome${m.winA >= 0.5 ? "" : " picked"}"></span></div>
<div class="pickBarLabels"><span>${cbbEsc(a.team)} <b>${cbbPct(m.winA)}</b></span><span><b>${cbbPct(1 - m.winA)}</b> ${cbbEsc(b.team)}</span></div>
<div class="cbbMatchupLine">Expected score: ${cbbEsc(fav.team)} ${Math.round(favPts)}, ${cbbEsc(dog.team)} ${Math.round(dogPts)}${by < 0.5 ? " (a toss-up)" : ` (${cbbEsc(fav.team)} by ${by.toFixed(1)})`}</div>
<div class="cbbMatchupLine dim">About ${Math.round(m.possessions)} possessions each. Games land about ${Math.round(league.sigma)} points either side of the expected margin, which is where the win chance comes from.</div>
${edges.length ? `<ul class="cbbEdges">${edges.map(e => `<li>${e}</li>`).join("")}</ul>` : ""}
</div>`
}

// One game from the picks log: the pick, the expected score and, once
// played, the result.
function cbbGameRowHtml(g, byId){
const home = byId.get(Number(g.home_id)), away = byId.get(Number(g.away_id))
if(!home || !away){ return "" }
const p = Number(g.home_win_probability)
const pick = p >= 0.5 ? home : away
const done = g.home_score !== null && g.home_score !== undefined && g.away_score !== null && g.away_score !== undefined
const homeWon = done && Number(g.home_score) > Number(g.away_score)
const right = done && ((p >= 0.5) === homeWon)
const expected = g.predicted_home_points === null || g.predicted_home_points === undefined ? ""
: `${Math.round(p >= 0.5 ? g.predicted_home_points : g.predicted_away_points)}–${Math.round(p >= 0.5 ? g.predicted_away_points : g.predicted_home_points)}`
const rank = t => t.rank <= 25 ? `<span class="cbbRank">${t.rank}</span>` : ""
return `<div class="cbbGame${done ? (right ? " right" : " wrong") : ""}">
<div class="cbbGameTeams">${rank(away)}<span class="${pick === away ? "winner" : ""}">${cbbEsc(away.team)}</span> ${g.neutral ? "vs" : "at"} ${rank(home)}<span class="${pick === home ? "winner" : ""}">${cbbEsc(home.team)}</span></div>
<div class="cbbGamePick">We say <b>${cbbEsc(pick.team)} ${cbbPct(Math.max(p, 1 - p))}</b>${expected ? `<span class="dim">&nbsp;&middot; ${expected}</span>` : ""}</div>
${done ? `<div class="cbbGameFinal">Final: ${cbbEsc(away.team)} ${Math.round(g.away_score)}, ${cbbEsc(home.team)} ${Math.round(g.home_score)}&nbsp;&middot; <b>${right ? "right" : "wrong"}</b></div>` : ""}
</div>`
}

function cbbDateLabel(iso){
const d = new Date(`${iso}T12:00:00`)
return Number.isNaN(d.getTime()) ? iso : d.toLocaleDateString("en-US", {weekday: "long", month: "short", day: "numeric"})
}

// The games list for one day, best games (by the two ratings) first.
// `limit` caps how many show.
function cbbGamesHtml(games, byId, date, limit){
const day = (games || []).filter(g => g.date === date && byId.get(Number(g.home_id)) && byId.get(Number(g.away_id)))
if(!day.length){ return '<div class="empty-note">No games between Division I teams.</div>' }
const quality = g => byId.get(Number(g.home_id)).net + byId.get(Number(g.away_id)).net
day.sort((x, y) => quality(y) - quality(x))
const shown = limit ? day.slice(0, limit) : day
return shown.map(g => cbbGameRowHtml(g, byId)).join("")
+ (shown.length < day.length ? `<div class="cbbMore">The best ${shown.length} of ${day.length} games by rating.</div>` : "")
}

// "Right on 212 of 301 games (70%)", then by how sure we were.
function cbbRecordHtml(record){
if(!record || !record.games){ return "No picks graded yet this season." }
const head = `Right on <b>${record.right} of ${record.games}</b> games (${Math.round(100 * record.right / record.games)}%)`
+ (record.margin_miss !== null && record.margin_miss !== undefined ? `, margins off by ${record.margin_miss} points on average.` : ".")
const bands = (record.bands || []).filter(b => b.games)
if(!bands.length){ return head }
return `${head}<table class="trackRecord cbbRecord"><thead><tr><th>When we say</th><th>Right</th></tr></thead><tbody>${
bands.map(b => `<tr><td>${cbbEsc(b.label)}</td><td>${b.right} of ${b.games} <span class="dim">${Math.round(100 * b.right / b.games)}%</span></td></tr>`).join("")
}</tbody></table>`
}

const CBB_ROUND_LABELS = [["round_of_32", "Win a game"], ["sweet_16", "Sweet 16"], ["elite_8", "Elite 8"], ["final_four", "Final Four"], ["final", "Title game"], ["champion", "Champion"]]

// Each team's chance of reaching each round, title favorites first.
function cbbBracketOddsHtml(bracket, byId, limit){
const teams = ((bracket && bracket.teams) || []).slice().sort((x, y) => y.champion - x.champion || y.final_four - x.final_four || x.seed - y.seed)
if(!teams.length){ return '<div class="empty-note">No bracket yet.</div>' }
const shown = limit ? teams.slice(0, limit) : teams
const body = shown.map(t => {
const team = byId.get(Number(t.team_id)) || {team: `Team ${t.team_id}`}
return `<tr><td>${t.seed}</td><td class="cbbTeamCell">${cbbDot(team)}${cbbEsc(team.team)}${t.play_in ? ' <span class="dim">First Four</span>' : ""}</td><td>${cbbEsc(t.region)}</td>${
CBB_ROUND_LABELS.map(([key]) => `<td${key === "champion" ? ' class="cbbTitle"' : ""}>${cbbPct(t[key])}</td>`).join("")}</tr>`
}).join("")
return `<table class="cbbTable cbbOdds">
<thead><tr><th>Seed</th><th>Team</th><th>Region</th>${CBB_ROUND_LABELS.map(([, label]) => `<th>${label}</th>`).join("")}</tr></thead>
<tbody>${body}</tbody></table>${shown.length < teams.length ? `<div class="cbbMore">Showing ${shown.length} of ${teams.length} teams.</div>` : ""}`
}

const CBB_SEED_ORDER = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]

// One region's first-round games, top of the bracket to the bottom, with
// each side's chance. A First Four pair shows as "A or B".
function cbbRegionHtml(bracket, byId, region, league){
const inRegion = ((bracket && bracket.teams) || []).filter(t => t.region === region)
if(!inRegion.length){ return "" }
const slot = seed => inRegion.filter(t => t.seed === seed).sort((x, y) => y.round_of_64 - x.round_of_64)
const name = t => (byId.get(Number(t.team_id)) || {team: `Team ${t.team_id}`}).team
const games = []
for(let i = 0; i < CBB_SEED_ORDER.length; i += 2){
const top = slot(CBB_SEED_ORDER[i]), bottom = slot(CBB_SEED_ORDER[i + 1])
if(!top.length || !bottom.length){ continue }
// The chance each slot wins its first game: its teams' chances of reaching the round of 32.
const topWin = top.reduce((s, t) => s + t.round_of_32, 0), bottomWin = bottom.reduce((s, t) => s + t.round_of_32, 0)
const label = entry => entry.map(t => cbbEsc(name(t))).join(" or ")
games.push(`<div class="cbbSlot">
<div class="cbbSlotTeam${topWin >= bottomWin ? " winner" : ""}"><span class="cbbSeed">${CBB_SEED_ORDER[i]}</span><span class="cbbSlotName">${label(top)}</span><b>${cbbPct(topWin)}</b></div>
<div class="cbbSlotTeam${bottomWin > topWin ? " winner" : ""}"><span class="cbbSeed">${CBB_SEED_ORDER[i + 1]}</span><span class="cbbSlotName">${label(bottom)}</span><b>${cbbPct(bottomWin)}</b></div>
</div>`)
}
const favorite = inRegion.slice().sort((x, y) => y.final_four - x.final_four)[0]
return `<div class="cbbRegion">
<h4>${cbbEsc(region)}</h4>
<div class="cbbRegionNote">Most likely to come out: ${cbbEsc(name(favorite))} (${cbbPct(favorite.final_four)})</div>
${games.join("")}
</div>`
}

// Real bracket only: first-round games where the worse seed has a real
// chance (40% or better, seeds at least two lines apart), likeliest first.
function cbbUpsetsHtml(bracket, byId){
if(!bracket || bracket.kind !== "official"){ return "" }
const name = t => (byId.get(Number(t.team_id)) || {team: `Team ${t.team_id}`}).team
const picks = []
;[...new Set(bracket.teams.map(t => t.region))].forEach(region => {
const inRegion = bracket.teams.filter(t => t.region === region)
for(let i = 0; i < CBB_SEED_ORDER.length; i += 2){
const better = CBB_SEED_ORDER[i], worse = CBB_SEED_ORDER[i + 1]
const dogs = inRegion.filter(t => t.seed === worse), favorites = inRegion.filter(t => t.seed === better)
const chance = dogs.reduce((sum, t) => sum + t.round_of_32, 0)
if(worse - better >= 2 && chance >= 0.4 && dogs.length && favorites.length){
picks.push({chance, text: `${worse} ${dogs.map(t => cbbEsc(name(t))).join(" or ")} over ${better} ${favorites.map(t => cbbEsc(name(t))).join(" or ")}`, region})
}
}
})
if(!picks.length){ return "" }
picks.sort((x, y) => y.chance - x.chance)
return `<div class="cbbUpsets"><b>Upsets to consider:</b> ${picks.map(u => `${u.text} (${cbbPct(u.chance)}, ${cbbEsc(u.region)})`).join("; ")}.</div>`
}

// The headline: the four likeliest champions.
function cbbFavoritesHtml(bracket, byId){
const teams = ((bracket && bracket.teams) || []).slice().sort((x, y) => y.champion - x.champion).slice(0, 4)
return teams.map(t => {
const team = byId.get(Number(t.team_id)) || {team: `Team ${t.team_id}`}
return `<div class="streakStat"><div class="value">${cbbPct(t.champion)}</div><div class="label">${cbbEsc(team.team)}</div><div class="sub">${t.seed} seed, ${cbbEsc(t.region)}&nbsp;&middot; Final Four ${cbbPct(t.final_four)}</div></div>`
}).join("")
}

// What kind of bracket this is, in a sentence or two.
function cbbBracketNote(payload){
const b = payload.bracket || {}
const sims = Number(b.sims || 0).toLocaleString("en-US")
if(b.kind === "official"){
return `The real bracket, played out ${sims} times from our ratings; games already played are counted as they finished.`
+ (b.final_four_pairs_known ? "" : " Which regions meet in the Final Four isn't in our data yet, so that pairing is assumed.")
}
return `The real bracket comes out on Selection Sunday in March. Until then this is a stand-in: the 68 teams our ratings would send if the season ended today (each conference's best-rated team, then the best of the rest), seeded straight down the ratings and played out ${sims} times. The selection committee picks on results, not ratings, so expect the real field to differ by a dozen or so teams, mostly the small-conference champions.`
}

// Past tournaments: how the same model did with the ratings of Selection Sunday.
function cbbBacktestHtml(backtest){
const rows = (backtest || []).filter(r => r.games)
if(!rows.length){ return "" }
const games = rows.reduce((s, r) => s + r.games, 0), right = rows.reduce((s, r) => s + r.right, 0)
const seeded = rows.filter(r => r.seed_games)
const seedLine = seeded.length ? ` Where the data has seeds (${seeded[0].season}&ndash;${seeded[seeded.length - 1].season}), our favorite won ${seeded.reduce((s, r) => s + r.seed_model_right, 0)} of ${seeded.reduce((s, r) => s + r.seed_games, 0)} games between different seeds and the better seed won ${seeded.reduce((s, r) => s + r.seed_chalk_right, 0)}.` : ""
const body = rows.slice().reverse().map(r => `<tr><td>${r.season}</td><td>${r.right} of ${r.games} <span class="dim">${Math.round(100 * r.right / r.games)}%</span></td>
<td>${cbbEsc(r.champion || "-")}</td><td>${r.champion_rank ? cbbOrdinal(r.champion_rank) : "-"}</td><td>${r.champion_title_chance === null || r.champion_title_chance === undefined ? "-" : cbbPct(r.champion_title_chance)}</td></tr>`).join("")
return `<p class="sectionIntro">Each past tournament, picked before it started with only the games played by then: our favorite won <b>${right} of ${games}</b> games (${Math.round(100 * right / games)}%).${seedLine}</p>
<table class="cbbTable cbbBacktest">
<thead><tr><th>Year</th><th>Games right</th><th>Champion</th><th>Our rank for them</th><th>Our title chance</th></tr></thead>
<tbody>${body}</tbody></table>`
}

if (typeof module !== "undefined" && module.exports) {
module.exports = {
cbbNormCdf, cbbPct, cbbSigned, cbbOrdinal, cbbMatchup, cbbBeatsAverage, cbbTeamIndex, cbbRankingsHtml, cbbMatchupHtml, cbbGameRowHtml,
cbbGamesHtml, cbbRecordHtml, cbbBracketOddsHtml, cbbRegionHtml, cbbFavoritesHtml, cbbBracketNote, cbbBacktestHtml,
cbbUpsetsHtml,
cbbDateLabel,
}
}
