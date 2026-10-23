// node --test docs/mlb_render.test.js
const test = require("node:test")
const assert = require("node:assert/strict")
global.teamName = require("./teams.js").teamName
const r = require("./mlb_render.js")

const text = html => html.replace(/<\/(div|span|td|button)>/g, " ").replace(/<[^>]+>/g, "").replace(/&middot;/g, "·").replace(/\s+/g, " ").trim()

test("pictures: MLB CDN urls by id / team, nothing for unknowns, text fallback underneath", () => {
  assert.match(r.mlbHeadshotUrl("666018.0"), /people\/666018\/headshot/)
  assert.equal(r.mlbHeadshotUrl(""), "")
  assert.match(r.mlbLogoUrl("tb"), /team-cap-on-dark\/139\.svg$/)
  assert.equal(r.mlbLogoUrl("XYZ"), "")
  const avatar = r.mlbAvatar(666018, "Jonathan Aranda")
  assert.ok(avatar.includes(">JA<") && avatar.includes('onerror="this.remove()"'))
  assert.ok(!r.mlbAvatar("", "Nobody Known").includes("<img"))
})

test("percentiles: ties count half, lower-is-better flips", () => {
  const pool = [0.1, 0.2, 0.3, 0.4]
  assert.equal(r.mlbPercentile(pool, 0.4), 88)
  assert.equal(r.mlbPercentile(pool, 0.4, -1), 13)
  assert.equal(r.mlbPercentile(pool, ""), null)
  assert.equal(r.mlbIndex(1.226), "+23%")
  assert.equal(r.mlbIndex(0.95), "−5%")
  assert.equal(r.mlbIndex(1.001), "avg")
})

test("streak chart: one column per day, best and final labelled, resets marked", () => {
  const progression = [
    { date: "2026-06-01", streak: "1", reset: "False" },
    { date: "2026-06-02", streak: "3", reset: "False" },
    { date: "2026-06-03", streak: "0", reset: "True" },
    { date: "2026-07-01", streak: "2", reset: "False" },
  ]
  const picks = [{ date: "2026-06-03", name: "Jake McCarthy", status: "miss", grade: "recommended" }]
  const html = r.streakChartHtml(progression, picks)
  assert.equal((html.match(/class="streakCol"/g) || []).length, 4)
  assert.equal((html.match(/class="streakReset"/g) || []).length, 1)
  assert.ok(html.includes("Best 3") && html.includes("Final 2"))
  assert.ok(html.includes("Jun 3: streak 0 - reset · Jake McCarthy (no hit)"))
  assert.ok(html.includes(">Jun<") && html.includes(">Jul<"))
  assert.equal(r.streakChartHtml([], picks), "")
})

test("day log: newest first, how each pick did, streak after the day", () => {
  const picks = [
    { date: "2026-09-23", rank: "1", name: "Gabriel Moreno", status: "hit", grade: "recommended", combined_probability: "0.7" },
    { date: "2026-09-24", rank: "1", name: "TJ Rumfield", status: "miss", grade: "recommended", combined_probability: "0.7" },
    { date: "2026-09-21", rank: "1", name: "Coby Mayo", status: "miss", grade: "speculative", combined_probability: "0.62" },
  ]
  const progression = [{ date: "2026-09-23", streak: "1", reset: "False" }, { date: "2026-09-24", streak: "0", reset: "True" }]
  const rows = text(r.btsLogHtml(picks, progression)).split(/(?=Sep \d)/).map(s => s.trim())
  assert.deepEqual(rows, [
    "Sep 24 TR TJ Rumfield ✗ Reset to 0",
    "Sep 23 GM Gabriel Moreno ✓ Streak 1",
    "Sep 21 CM Coby Mayo weaker ✗ Sat out",
  ])
})

test("pick cards: team from the player file, chance and result", () => {
  const html = text(r.btsPickCardsHtml(
    [{ key_mlbam: "681198", name: "TJ Rumfield", combined_probability: "0.7017", status: "hit", grade: "recommended" }],
    { 681198: { team: "COL" } },
  ))
  assert.equal(html, "TR TJ Rumfield COL Rockies 70% chance of a hit · Strong pick ✓ Got a hit")
})

test("starting pitchers: one card per game, away starter first, vs-average in words", () => {
  const html = text(r.pitcherDuelsHtml([
    { team: "MIL", opponent: "TB", is_home: "True", pitcher_name: "Jacob Misiorowski", Throws: "R", PAVE: "0.174", PAVE_PLUS: "0.774" },
    { team: "TB", opponent: "MIL", is_home: "False", pitcher_name: "Nick Martínez", Throws: "R", PAVE: "0.289", PAVE_PLUS: "1.283" },
  ]))
  assert.ok(html.startsWith("TB Rays at MIL Brewers"))
  assert.ok(html.indexOf("Nick Martínez") < html.indexOf("Jacob Misiorowski"))
  assert.ok(html.includes("Hits on 17% of at-bats · 23% fewer hits than average"))
  assert.ok(html.includes("28% more hits than average"))
  assert.equal(r.pitcherVsAverage(""), "")
})

test("game card: win split from our number, Vegas marked, result", () => {
  const html = r.gamePickCardHtml({
    home_team: "MIL", away_team: "TB", predicted_winner: "MIL", predicted_probability: "0.667",
    market_home_win_probability: "0.624", status: "pending", bet_units: "0",
  })
  assert.ok(text(html).includes("We say Brewers 67% · Vegas 62%"))
  assert.ok(html.includes('class="winVegas" style="left:37.6%"'))
  assert.ok(html.includes("width:33.3%") && html.includes("width:66.7%"))
  assert.ok(text(html).includes("Not played yet"))
})

test("versus Vegas: who picked the winner, running gap and verdict", () => {
  const game = (date, winner, status, homeProb) => ({ date, home_team: "A", away_team: "B", predicted_winner: winner, status, market_home_win_probability: homeProb })
  const picks = [
    game("2026-07-01", "A", "win", "0.6"),   // both right
    game("2026-07-02", "A", "loss", "0.4"),  // Vegas had B, B won: Vegas right
    game("2026-07-03", "B", "win", "0.6"),   // we had B, B won: we right
    game("2026-07-04", "A", "pending", "0.6"),
  ]
  assert.deepEqual(r.gamesVersusVegas(picks).map(g => [g.ours, g.vegas]), [[true, true], [false, true], [true, false]])
  const many = Array.from({ length: 12 }, (_, i) => game(`2026-07-${String(i + 1).padStart(2, "0")}`, "A", i < 3 ? "loss" : "win", "0.4"))
  // We picked A every game; Vegas picked B. A won 9 of 12.
  assert.ok(text(r.versusVegasChartHtml(many)).includes("we were right on 9 and Vegas on 3, so we finished 6 games ahead of Vegas"))
  assert.equal(r.versusVegasChartHtml(picks), "")
})

test("confidence buckets: how often right at each level we gave", () => {
  const picks = [["0.52", "win"], ["0.53", "loss"], ["0.62", "win"], ["0.7", "win"], ["0.7", "pending"]]
    .map(([p, status]) => ({ predicted_probability: p, status }))
  const rows = text(r.confidenceBucketsHtml(picks))
  assert.ok(rows.includes("We said 50–55% 50% of 2"))
  assert.ok(rows.includes("We said 60–65% 100% of 1"))
  assert.ok(rows.includes("We said 65% and up 100% of 1"))
  assert.ok(!rows.includes("55–60%"))
})

const TEAMS = [
  { team: "SD", Confidence: "1.23", win_rate: "0.56", games_played: "168", Bullpen_PAVE: "0.23", Park_Factor: "0.89" },
  { team: "LAA", Confidence: "0.73", win_rate: "0.38", games_played: "162", Bullpen_PAVE: "0.29", Park_Factor: "0.95" },
  { team: "NYY", Confidence: "1.11", win_rate: "0.52", games_played: "162", Bullpen_PAVE: "0.21", Park_Factor: "1.05" },
]

test("teams: rank by direction, record from win rate, power rankings in order", () => {
  assert.equal(r.teamRank(TEAMS, TEAMS[0], "Confidence", 1), 1)
  assert.equal(r.teamRank(TEAMS, TEAMS[2], "Bullpen_PAVE", -1), 1)   // fewest hits allowed
  assert.equal(r.teamRank(TEAMS, TEAMS[0], "Bullpen_PAVE", -1), 2)
  assert.equal(r.teamRecord(TEAMS[0]), "94-74")
  const order = [...r.powerRankingsHtml(TEAMS, ["SD"]).matchAll(/data-team="(\w+)"/g)].map(m => m[1])
  assert.deepEqual(order, ["SD", "NYY", "LAA"])
  assert.ok(r.powerRankingsHtml(TEAMS, ["SD"]).includes('class="powerRow powerPicked" data-team="SD"'))
})

test("team profile: rank in the dot for one team, two caps on each track to compare", () => {
  const one = r.teamProfileHtml(["SD"], TEAMS)
  assert.ok(text(one).includes("Padres 94-74 · 1st of 3 overall"))
  assert.ok(one.includes('class="pctDot"') && one.includes(">1</span>"))
  const two = r.teamProfileHtml(["SD", "LAA"], TEAMS)
  assert.ok(two.includes("teamCompare"))
  assert.equal((two.match(/class="pctMarker pctMarker0"/g) || []).length, (two.match(/class="pctMarker pctMarker1"/g) || []).length)
  assert.equal(r.teamProfileHtml([], TEAMS), "Select a team")
})

test("player card: lower-is-better stats flip, today's chance and streak, pitch mix shows the unclassified rest", () => {
  const pool = [
    { Whiff_Rate: "0.10", WAVE: "0.20" }, { Whiff_Rate: "0.20", WAVE: "0.25" }, { Whiff_Rate: "0.30", WAVE: "0.30" },
  ]
  const hitter = { key_mlbam: "1", name_first: "Steven", name_last: "Kwan", team: "CLE", PA_L: "212", PA_R: "438", Whiff_Rate: "0.10", WAVE: "0.30" }
  const html = r.playerCardHtml(hitter, "hitter", pool, { todayChance: "0.66", opponent: "TB", isHome: true, hitStreak: 4 })
  assert.ok(html.includes("Swing-and-miss rate: 10.0% - better than 83% of regular hitters"))
  assert.ok(text(html).includes("66% chance of a hit today vs. Rays"))
  assert.ok(text(html).includes("On a 4-game hit streak"))
  assert.ok(text(html).includes("650 plate appearances"))
  const pitcher = { key_mlbam: "2", name_first: "Shohei", name_last: "Ohtani", team: "LAD", Throws: "R", at_bats: "337",
    PAVE: "0.05", PAVE_PLUS: "0.24", Fastball_Rate: "0.15", Breaking_Rate: "0.12", Offspeed_Rate: "0.03" }
  const card = text(r.playerCardHtml(pitcher, "pitcher", [pitcher]))
  assert.ok(card.includes("Other / unclassified 70%"))
  assert.ok(card.includes("Allows 76% fewer hits than average"))
})

test("hitter table: WAVE, hit chance at 3.5 at-bats, CAVE, WOPS; season or vs. pitchers faced; sortable", () => {
  assert.equal(r.hitChanceFromWave("0.25").toFixed(4), (1 - Math.pow(0.75, 3.5)).toFixed(4))
  assert.equal(r.hitChanceFromWave(""), null)
  const wave = [{ key_mlbam: "1", name_first: "Steven", name_last: "Kwan", team: "CLE", WAVE: "0.24", Game_Hit_Probability: "0.6" },
    { key_mlbam: "2", name_first: "Yordan", name_last: "Álvarez", team: "HOU", WAVE: "0.23", Game_Hit_Probability: "0.6" }]
  const stats = [
    { key_mlbam: "1.0", WAVE: "0.259", WAVE_adj: "0.262", CAVE: "0.636", CAVE_adj: "0.643", WOPS: "0.795", WOPS_adj: "0.802" },
    { key_mlbam: "2", WAVE: "0.240", WAVE_adj: "0.236", CAVE: "0.622", CAVE_adj: "0.618", WOPS: "0.943", WOPS_adj: "0.930" },
  ]
  const season = r.hitterStatRows(wave, stats)
  assert.deepEqual([season[0].WAVE, season[0].CAVE, season[0].WOPS], [0.259, 0.636, 0.795])
  const faced = r.hitterStatRows(wave, stats, "faced")
  assert.deepEqual([faced[1].WAVE, faced[1].CAVE, faced[1].WOPS], [0.236, 0.618, 0.930])
  assert.deepEqual(r.sortHitterStatRows(season).map(x => x.name), ["Steven Kwan", "Yordan Álvarez"])
  assert.deepEqual(r.sortHitterStatRows(season, "WOPS").map(x => x.name), ["Yordan Álvarez", "Steven Kwan"])
  const plain = text(r.hitterStatsTableHtml(season, "WOPS"))
  assert.ok(plain.includes("Astros .240") && plain.includes("62.2%") && plain.includes(".943"))
  assert.ok(!plain.includes("+.") && !plain.includes("−."))
  const adjusted = text(r.hitterStatsTableHtml(faced, "WOPS", 40, "faced"))
  assert.ok(adjusted.includes(".930−.013"))
  assert.ok(adjusted.includes(".262+.003"))
  assert.ok(adjusted.includes("64.3%+0.7%"))
  assert.ok(r.hitterStatsTableHtml(season).includes("CAVE (Carried Average): share of games with a hit"))
})

test("schedule: series in order, ink deepens with the margin, four panels", () => {
  const game = (n, series, opponent, home, result, diff) => ({ team: "SD", game_no: String(n), series: String(series), opponent, home: String(home),
    result, run_diff: String(diff), runs_for: result === "W" ? "5" : "2", runs_against: result === "W" ? "4" : "11", date: `2026-04-0${n}` })
  const games = [game(1, 1, "DET", 1, "W", 1), game(2, 1, "DET", 1, "L", -9), game(3, 2, "SF", 0, "W", 12), { team: "LAD", game_no: "1", series: "1" }]
  const series = r.scheduleSeries(games, "SD")
  assert.deepEqual(series.map(s => [s.opponent, s.home, s.games.length]), [["DET", true, 2], ["SF", false, 1]])
  assert.equal(r.scheduleInk("W", 1), "rgb(112,192,132)")
  assert.equal(r.scheduleInk("W", 12), r.scheduleInk("W", 9))
  assert.equal(r.scheduleInk("L", -9), "rgb(160,14,18)")
  const sheet = r.scheduleSheetHtml(games, "SD")
  assert.equal((sheet.match(/class="paperPanel"/g) || []).length, 4)
  assert.equal((sheet.match(/class="schedGame"/g) || []).length, 3)
  assert.ok(sheet.includes("Apr 1 vs. Tigers: W 5-4") && sheet.includes("Apr 3 at Giants: W 5-4"))
  assert.ok(text(sheet).includes("2-1 · run differential +4 · won 1 of 2 series, split 1"))
  assert.ok(sheet.includes("Apr 1–2"))
  assert.ok(sheet.includes("team-cap-on-light"))
  assert.ok(r.scheduleSheetHtml(games, "NYY").includes("No games for this team yet"))
  const picker = r.scheduleTeamPickerHtml(["SD", "LAD"], "SD")
  assert.ok(picker.indexOf("LAD") < picker.indexOf('"SD"') && picker.includes('class="schedPick schedPicked" data-sched-team="SD"'))
})
