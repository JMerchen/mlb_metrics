// Today's Summary - the real site's landing page. A distilled, read-only
// snapshot (stats + Beat the Streak picks + MLB/NFL Game Picks) built
// from the same real, currently-live docs/data/*.csv files every other
// page reads - no embedded/curated data, no chat (that's the Claude
// Artifact's Assistant tab's job). The render functions and markup here
// intentionally mirror scripts/templates/bts_assistant_template.html's
// own renderStats/renderPicks/renderGamePicks (same DOM ids, same CSS
// classes, now shared via docs/style.css) so this page looks like that
// one, minus the chat panel at the bottom.

async function loadCSV(path){

const response = await fetch(`${path}?t=${Date.now()}`, {cache:"no-store"})

if(!response.ok){
throw new Error(`Failed to load ${path}: ${response.status}`)
}

const text = await response.text()

return Papa.parse(text, {header:true, skipEmptyLines:true}).data

}

function fmtPct(x){ return (Number(x) * 100).toFixed(1) + '%' }

function esc(s){ return String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])) }

// The one real date/slate column every picks CSV here has is "date" -
// simple lexicographic max works since it's always YYYY-MM-DD.
// "2026-10-07" -> "Wed, Oct 7"
function friendlyDate(iso){
const d = new Date(iso + 'T12:00:00')
return Number.isNaN(d.getTime()) ? iso : d.toLocaleDateString('en-US', {weekday: 'short', month: 'short', day: 'numeric'})
}

function latestDateStr(rows){
const dates = rows.map(r => r.date).filter(Boolean).sort()
return dates.length ? dates[dates.length - 1] : null
}

async function loadTodaysBeatTheStreak(){

const [picks, summaryRows, wave] = await Promise.all([
loadCSV("./data/beat_the_streak_picks.csv"),
loadCSV("./data/beat_the_streak_summary.csv"),
loadCSV("./data/wave.csv"),
])

const asOf = latestDateStr(picks)

const todaysPicks = picks
.filter(p => p.date === asOf)
.sort((a, b) => Number(a.rank) - Number(b.rank))

// beat_the_streak_summary.csv carries exactly one row (the all_time
// aggregate) - beat_the_streak_summary_by_version.csv is the one with a
// row per model version, not needed here.
const summary = summaryRows[0] || {}

// Same qualifier/derivation as build_bts_assistant_page.py's
// build_payload: PA_L+PA_R >= 30, name = first + last, ranked by
// Approach - "today's top approach" is just the #1 row.
const qualified = wave
.filter(w => (Number(w.PA_L) + Number(w.PA_R)) >= 30)
.map(w => ({
...w,
name: `${w.name_first} ${w.name_last}`,
Approach: Number(w.Approach),
Game_Hit_Probability: Number(w.Game_Hit_Probability),
}))
.sort((a, b) => b.Approach - a.Approach)

return {asOf, todaysPicks, summary, topApproach: qualified[0] || null}

}

// Mirrors build_bts_assistant_page.py's _game_picks_section - groupByWeek
// (NFL only) groups "upcoming" by the whole (season, week) slate instead
// of a single date, since real NFL weeks span Thu/Sun/Mon (see
// docs/nfl.js's own renderNflTodaysGamePicks, which groups the same way).
async function loadGamePicksSection(prefix, {groupByWeek = false} = {}){

const [picks, summaryRows, byVersion] = await Promise.all([
loadCSV(`./data/${prefix}_picks.csv`),
loadCSV(`./data/${prefix}_summary.csv`),
loadCSV(`./data/${prefix}_summary_by_version.csv`).catch(() => []),
])

const summary = summaryRows[0] || {}
// The row for the model making picks now (by-version holds "all_time"
// plus that one).
summary.current = byVersion.find(r => r.model_version && r.model_version !== 'all_time') || null

if(!picks.length){
return {asOf: null, upcomingPicks: [], summary}
}

let upcoming
const asOf = latestDateStr(picks)

if(groupByWeek){
const slateKey = p => `${p.season}_${String(p.week).padStart(2, "0")}`
const latestSlate = picks.map(slateKey).sort().slice(-1)[0]
upcoming = picks.filter(p => slateKey(p) === latestSlate)
} else {
upcoming = picks.filter(p => p.date === asOf)
}

upcoming = upcoming.slice().sort((a, b) => a.home_team.localeCompare(b.home_team))

return {asOf, upcomingPicks: upcoming, summary}

}

// Rendering lives in summary_render.js (shared with the full-site
// artifact's Assistant tab, so the two can't drift apart again).
function renderStats(all, top){
document.getElementById('stat-row').innerHTML = streakTilesHtml(all, top)
}

function renderPicks(todaysPicks){
document.getElementById('picks-list').innerHTML = btsPicksHtml(todaysPicks)
}

function renderGamePicks(sport, data){
const listEl = document.getElementById(`${sport}-gp-list`)
const statEl = document.getElementById(`${sport}-gp-stat-line`)
if(!data.asOf){
listEl.innerHTML = '<div class="empty-note">No games scheduled yet.</div>'
statEl.textContent = ''
return
}
statEl.innerHTML = gamePicksRecordHtml(data.summary, data.summary.current)
listEl.innerHTML = gamePicksListHtml(data.upcomingPicks, sport)
}

(async () => {

const [bts, mlb, nfl] = await Promise.all([
loadTodaysBeatTheStreak(),
loadGamePicksSection('game_picks'),
loadGamePicksSection('nfl_game_picks', {groupByWeek: true}),
])

const seasonOver = btsSeasonOver(bts.summary)
document.getElementById('asof-badge').textContent = seasonOver
? 'MLB season over'
: (bts.asOf ? 'Updated ' + friendlyDate(bts.asOf) : 'No data yet')
renderStats(bts.summary, bts.topApproach)
if(seasonOver){
document.getElementById('picks-list').innerHTML = btsSeasonOverHtml(bts.summary)
}else{
renderPicks(bts.todaysPicks)
}
renderGamePicks('mlb', mlb)
renderGamePicks('nfl', nfl)

})()
