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

const [picks, summaryRows] = await Promise.all([
loadCSV(`./data/${prefix}_picks.csv`),
loadCSV(`./data/${prefix}_summary.csv`),
])

const summary = summaryRows[0] || {}

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

function renderStats(all, top){

const statRow = document.getElementById('stat-row')

if(!all || !Object.keys(all).length){
statRow.innerHTML = '<div class="empty-note">No stats tracked yet.</div>'
return
}

const tiles = [
{ label: 'Current Streak', value: all.current_streak, sub: 'picks in a row' },
{ label: 'Longest Streak', value: all.longest_streak, sub: 'all-time best' },
{
label: 'Day Survival Rate',
value: fmtPct(all.day_survival_rate),
sub: `${(Number(all.day_survival_rate_ci_low) * 100).toFixed(0)}–${(Number(all.day_survival_rate_ci_high) * 100).toFixed(0)}% CI, n=${all.n_days_resolved}`,
},
{
label: "Today's Top Approach",
value: top ? top.name.split(' ').pop() : '—',
sub: top ? `${top.team} · ${fmtPct(top.Game_Hit_Probability)} GHP` : 'no qualified hitters',
},
]

statRow.innerHTML = tiles.map(t =>
`<div class="stat"><div class="btsLabel">${esc(t.label)}</div><div class="btsValue">${esc(t.value)}</div><div class="btsSub">${esc(t.sub)}</div></div>`
).join('')

}

function renderPicks(todaysPicks){

document.getElementById('picks-list').innerHTML = todaysPicks.map(p => `
<div class="pick">
<span class="rank">#${p.rank}</span>
<span class="name">${esc(p.name)}</span>
<span class="prob">${fmtPct(p.combined_probability)}</span>
<span class="grade ${p.grade === 'recommended' ? 'btsRecommended' : 'btsSpeculative'}">${p.grade === 'recommended' ? 'Recommended' : 'Speculative'}</span>
</div>
`).join('') || '<div class="pick"><span class="name">No picks logged yet today.</span></div>'

}

function renderGamePicks(sport, data){

const listEl = document.getElementById(`${sport}-gp-list`)
const statEl = document.getElementById(`${sport}-gp-stat-line`)
const s = data.summary

if(!data.asOf){
listEl.innerHTML = '<div class="empty-note">No games scheduled yet.</div>'
statEl.textContent = ''
return
}

const nBets = Number(s.n_bets_advised)
if(!nBets){
statEl.textContent = 'no bets resolved yet'
} else {
statEl.innerHTML = `<b>${fmtPct(s.win_rate_on_advised_bets)}</b> win rate · <b>${(Number(s.roi) * 100).toFixed(1)}%</b> ROI · ${nBets} bets`
}

// bet_units > 0 is the real "was a bet advised" signal (Kelly sizing/
// daily caps can zero it out even when the model's own above_threshold
// edge flag is true) - matches app.js/nfl.js's own
// renderTodaysGamePicks/renderNflTodaysGamePicks.
listEl.innerHTML = data.upcomingPicks.map(g => {
const homeWins = g.predicted_winner === g.home_team
const betUnits = Number(g.bet_units)
const betAdvised = betUnits > 0
const bet = betAdvised
? `<span class="bet-badge">Bet ${esc(g.bet_team)} ${betUnits.toFixed(2)}u</span>` : ''
return `<div class="game">
<div class="btsMatchup">
<span class="${!homeWins ? 'winner' : ''} away">${esc(g.away_team)}</span> @
<span class="${homeWins ? 'winner' : ''}">${esc(g.home_team)}</span>
</div>
<span class="prob">${fmtPct(g.predicted_probability)}</span>
${bet}
</div>`
}).join('') || '<div class="empty-note">No games scheduled yet.</div>'

}

(async () => {

const [bts, mlb, nfl] = await Promise.all([
loadTodaysBeatTheStreak(),
loadGamePicksSection('game_picks'),
loadGamePicksSection('nfl_game_picks', {groupByWeek: true}),
])

document.getElementById('asof-badge').textContent = 'as of ' + (bts.asOf || 'no data yet')
renderStats(bts.summary, bts.topApproach)
renderPicks(bts.todaysPicks)
renderGamePicks('mlb', mlb)
renderGamePicks('nfl', nfl)

})()
