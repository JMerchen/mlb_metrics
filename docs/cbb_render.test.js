// node --test docs/cbb_render.test.js
const test = require("node:test")
const assert = require("node:assert/strict")
const r = require("./cbb_render.js")

const text = html => html.replace(/<[^>]+>/g, " ").replace(/&middot;/g, "·").replace(/&nbsp;/g, " ").replace(/&ndash;/g, "–").replace(/\s+/g, " ").trim().replace(/&lt;/g, "<").replace(/&gt;/g, ">")
const league = { efficiency: 105, home_edge: 2.5, tempo: 68, sigma: 11 }
const duke = { team_id: 150, team: "Duke", conference: "ACC", color: "00539b", rank: 1, net: 30, offense: 14, defense: 16, offense_rank: 4, defense_rank: 1, tempo: 70, wins: 20, losses: 2, last_season_net: 35 }
const yale = { team_id: 43, team: "Yale", conference: "Ivy", color: "zzz", rank: 80, net: 6, offense: 5, defense: 1, offense_rank: 70, defense_rank: 150, tempo: 66, wins: 15, losses: 7, last_season_net: null }
const byId = r.cbbTeamIndex([duke, yale])

test("normal curve and percent labels", () => {
  assert.ok(Math.abs(r.cbbNormCdf(0) - 0.5) < 1e-6)
  assert.ok(Math.abs(r.cbbNormCdf(1) - 0.841345) < 1e-5)
  assert.ok(Math.abs(r.cbbNormCdf(-2) - 0.022750) < 1e-5)
  assert.equal(r.cbbPct(0.613), "61%")
  assert.equal(r.cbbPct(0.002), "&lt;1%")
  assert.equal(r.cbbPct(0.999), "&gt;99%")
  assert.equal(r.cbbPct(0), "0%")
  assert.equal(r.cbbPct(1), "100%")
  assert.equal(r.cbbPct(null), "-")
  assert.equal(r.cbbSigned(3.24), "+3.2")
  assert.equal(r.cbbSigned(-0.02), "0.0")
  assert.equal(r.cbbSigned(-7), "−7.0")
})

test("a matchup: margin from the ratings and the pace, home edge on both ends", () => {
  const neutral = r.cbbMatchup(duke, yale, "neutral", league)
  assert.equal(neutral.possessions, 68)                       // 70 + 66 - 68
  assert.ok(Math.abs(neutral.margin - 24 * 0.68) < 1e-9)      // 24 points per 100 over 68 possessions
  assert.ok(Math.abs(neutral.pointsA - 0.68 * (105 + 14 - 1)) < 1e-9)
  assert.ok(Math.abs(neutral.pointsB - 0.68 * (105 + 5 - 16)) < 1e-9)
  assert.ok(Math.abs((neutral.pointsA - neutral.pointsB) - neutral.margin) < 1e-9)
  assert.ok(neutral.winA > 0.9)
  const away = r.cbbMatchup(duke, yale, "b", league)
  assert.ok(Math.abs(neutral.margin - away.margin - 5 * 0.68) < 1e-9)   // 2.5 on each end
  const home = r.cbbMatchup(duke, yale, "a", league)
  assert.ok(home.winA > neutral.winA && neutral.winA > away.winA)
  // The same game from the other side is the mirror image.
  const flipped = r.cbbMatchup(yale, duke, "a", league)
  assert.ok(Math.abs(flipped.winA + away.winA - 1) < 1e-9)
})

test("matchup card: favorite, expected score, and where the edges are", () => {
  const html = text(r.cbbMatchupHtml(duke, yale, "neutral", league))
  assert.match(html, /^Duke 93% on a neutral floor/)
  assert.match(html, /Expected score: Duke 80, Yale 64 \(Duke by 16\.3\)/)
  assert.match(html, /Duke's offense \(4th\) has the edge on Yale's defense \(150th\)/)
  assert.match(html, /Duke's defense \(1st\) has the edge on Yale's offense \(70th\)/)
  assert.match(text(r.cbbMatchupHtml(duke, yale, "b", league)), /at Yale/)
  assert.match(r.cbbMatchupHtml(duke, duke, "neutral", league), /two different teams/)
  assert.match(r.cbbMatchupHtml(undefined, yale, "neutral", league), /two different teams/)
})

test("rankings table: search, conference, limit", () => {
  const all = r.cbbRankingsHtml([duke, yale])
  assert.match(text(all), /1 Duke ACC 20–2 \+30\.0 \+14\.0 4th \+16\.0 1st 70\.0 \+35\.0/)
  assert.match(all, /background:#00539b/)
  assert.doesNotMatch(all, /background:#zzz/)                 // a bad color falls back
  assert.doesNotMatch(r.cbbRankingsHtml([duke, yale], { search: "yal" }), /Duke/)
  assert.doesNotMatch(r.cbbRankingsHtml([duke, yale], { conference: "ACC" }), /Yale/)
  assert.match(r.cbbRankingsHtml([duke, yale], { limit: 1 }), /Showing 1 of 2/)
  assert.match(r.cbbRankingsHtml([duke, yale], { search: "zzz" }), /No teams match/)
  // With the league's numbers, Deep dive gets adjusted efficiency and the chance of beating an average team.
  const deep = text(r.cbbRankingsHtml([duke, yale], { league }))
  assert.match(deep, /Scores Allows Beats avg Pace/)
  assert.match(deep, /1 Duke ACC 20–2 \+30\.0 \+14\.0 4th \+16\.0 1st 119\.0 89\.0 97% 70\.0/)
  assert.ok(Math.abs(r.cbbBeatsAverage(duke, league) - r.cbbNormCdf(30 * 0.70 / 11)) < 1e-9)
  assert.ok(Math.abs(r.cbbBeatsAverage({ net: 0, tempo: 68 }, league) - 0.5) < 1e-9)
})

test("game rows: the pick, the expected score, and the result once played", () => {
  const g = { date: "2026-11-10", game_id: "1", home_id: 43, away_id: 150, neutral: false, home_win_probability: 0.12,
    predicted_home_points: 64.2, predicted_away_points: 79.8, home_score: null, away_score: null }
  const row = r.cbbGameRowHtml(g, byId)
  assert.equal(text(row), "1 Duke at Yale We say Duke 88% · 80–64")
  assert.doesNotMatch(row, /right|wrong/)
  const won = r.cbbGameRowHtml({ ...g, home_score: 60, away_score: 75 }, byId)
  assert.match(won, /class="cbbGame right"/)
  assert.match(text(won), /Final: Duke 75, Yale 60 · right/)
  assert.match(r.cbbGameRowHtml({ ...g, home_score: 70, away_score: 68 }, byId), /class="cbbGame wrong"/)
  assert.match(text(r.cbbGameRowHtml({ ...g, neutral: true }, byId)), /Duke vs Yale/)
  assert.equal(r.cbbGameRowHtml({ ...g, home_id: 9999 }, byId), "")   // a team we don't rate
  const day = r.cbbGamesHtml([g, { ...g, game_id: "2", date: "2026-11-11" }], byId, "2026-11-10")
  assert.equal((day.match(/class="cbbGame"/g) || []).length, 1)
  assert.match(r.cbbGamesHtml([g], byId, "2026-12-01"), /No games/)
  assert.match(r.cbbGamesHtml([g, { ...g, game_id: "3" }], byId, "2026-11-10", 1), /The best 1 of 2 games/)
})

test("pick record: the headline and the bands that have games", () => {
  assert.equal(r.cbbRecordHtml({ games: 0 }), "No picks graded yet this season.")
  const html = text(r.cbbRecordHtml({ games: 300, right: 212, margin_miss: 8.7, bands: [
    { label: "50–60%", games: 80, right: 44 }, { label: "90–100%", games: 0, right: 0 }] }))
  assert.match(html, /^Right on 212 of 300 games \(71%\), margins off by 8\.7 points on average\./)
  assert.match(html, /50–60% 44 of 80 55%/)
  assert.doesNotMatch(html, /90–100%/)
})

const bracket = { kind: "projected", sims: 20000, final_four_pairs: [["East", "Midwest"], ["West", "South"]], final_four_pairs_known: true, teams: [
  { team_id: 150, seed: 1, region: "East", play_in: false, bid: "auto", round_of_64: 1, round_of_32: 0.98, sweet_16: 0.8, elite_8: 0.6, final_four: 0.45, final: 0.3, champion: 0.2 },
  { team_id: 43, seed: 16, region: "East", play_in: true, bid: "auto", round_of_64: 0.6, round_of_32: 0.015, sweet_16: 0.001, elite_8: 0, final_four: 0, final: 0, champion: 0 },
  { team_id: 77, seed: 16, region: "East", play_in: true, bid: "auto", round_of_64: 0.4, round_of_32: 0.005, sweet_16: 0, elite_8: 0, final_four: 0, final: 0, champion: 0 },
] }

test("bracket odds table and headline favorites", () => {
  const html = text(r.cbbBracketOddsHtml(bracket, byId))
  assert.match(html, /1 Duke East 98% 80% 60% 45% 30% 20%/)
  assert.match(html, /16 Yale First Four East 2% <1% 0% 0% 0% 0%/)
  assert.match(html, /Team 77/)                                // a team missing from the ratings still shows
  assert.match(r.cbbBracketOddsHtml(bracket, byId, 1), /Showing 1 of 3 teams/)
  assert.match(r.cbbBracketOddsHtml({ teams: [] }, byId), /No bracket yet/)
  assert.match(text(r.cbbFavoritesHtml(bracket, byId)), /^20% Duke 1 seed, East · Final Four 45%/)
})

test("a region's first round: slots in bracket order, a First Four pair as one slot", () => {
  const html = text(r.cbbRegionHtml(bracket, byId, "East", league))
  assert.match(html, /^East Most likely to come out: Duke \(45%\)/)
  assert.match(html, /1 Duke 98% 16 Yale or Team 77 2%/)
  assert.equal(r.cbbRegionHtml(bracket, byId, "West", league), "")
})

test("bracket note says whether it's the real bracket or a stand-in", () => {
  assert.match(r.cbbBracketNote({ bracket }), /Selection Sunday/)
  assert.match(r.cbbBracketNote({ bracket }), /20,000 times/)
  const real = r.cbbBracketNote({ bracket: { ...bracket, kind: "official", final_four_pairs_known: false } })
  assert.match(real, /^The real bracket/)
  assert.match(real, /pairing is assumed/)
  assert.doesNotMatch(r.cbbBracketNote({ bracket: { ...bracket, kind: "official" } }), /assumed/)
})

test("past tournaments table", () => {
  assert.equal(r.cbbBacktestHtml([]), "")
  const html = text(r.cbbBacktestHtml([
    { season: 2025, games: 67, right: 53, champion: "Florida", champion_rank: 4, champion_title_chance: 0.118, seed_games: 61, seed_model_right: 48, seed_chalk_right: 47 },
    { season: 2026, games: 67, right: 49, champion: "Michigan", champion_rank: 2, champion_title_chance: 0.196, seed_games: 62, seed_model_right: 46, seed_chalk_right: 45 },
    { season: 2012, games: 67, right: 46, champion: "Kentucky", champion_rank: 1, champion_title_chance: null, seed_games: 0 },
  ]))
  assert.match(html, /our favorite won 148 of 201 games \(74%\)/)
  assert.match(html, /our favorite won 94 of 123 games between different seeds and the better seed won 92/)
  assert.match(html, /2026 49 of 67 73% Michigan 2nd 20%/)
  assert.match(html, /2012 46 of 67 69% Kentucky 1st -/)
})

test("upsets to consider: only on the real bracket, worse seeds with a real chance", () => {
  const team = (id, seed, region, win) => ({ team_id: id, seed, region, play_in: false, round_of_64: 1, round_of_32: win })
  const teams = [team(150, 5, "East", 0.55), team(43, 12, "East", 0.45), team(1, 1, "East", 0.99), team(2, 16, "East", 0.01),
    team(3, 8, "East", 0.45), team(4, 9, "East", 0.55), team(5, 6, "West", 0.58), team(6, 11, "West", 0.42)]
  assert.equal(r.cbbUpsetsHtml({ kind: "projected", teams }, byId), "")
  const html = text(r.cbbUpsetsHtml({ kind: "official", teams }, byId))
  // 12 over 5 and 11 over 6 qualify; 9 over 8 is one line apart, 16 over 1 has no chance.
  assert.equal(html, "Upsets to consider: 12 Yale over 5 Duke (45%, East); 11 Team 6 over 6 Team 5 (42%, West).")
  assert.equal(r.cbbUpsetsHtml({ kind: "official", teams: teams.slice(2, 4) }, byId), "")
})

// A full 64-team bracket (no First Four): team id = 100 x region + seed, better seeds are better teams.
function fullBracket(){
  const regions = ["East", "Midwest", "West", "South"], teams = [], rated = []
  regions.forEach((region, r) => {
    for (let seed = 1; seed <= 16; seed++) {
      const id = 100 * (r + 1) + seed
      teams.push({ team_id: id, seed, region, play_in: false })
      rated.push({ team_id: id, team: `${region} ${seed}`, net: 34 - 3 * seed - r, offense: 0, defense: 0, tempo: 68 })
    }
  })
  return { bracket: { kind: "official", final_four_pairs: [["East", "Midwest"], ["West", "South"]], teams, results: [] }, ids: r.cbbTeamIndex(rated) }
}

test("playing a bracket: favorites only is all chalk; a random one has upsets and repeats with its seed", () => {
  const { bracket, ids } = fullBracket()
  const chalk = r.cbbPlayBracket(bracket, ids, league, null)
  assert.equal(chalk.upsets.length, 0)
  assert.equal(chalk.champion.team_id, 101)                         // the best 1 seed
  assert.deepEqual(chalk.regions.map(x => x.winner.seed), [1, 1, 1, 1])
  assert.deepEqual(chalk.regions[0].rounds.map(games => games.length), [8, 4, 2, 1])
  assert.equal(chalk.finalFour.length, 2)
  assert.equal(chalk.finalFour[0].loser.region, "Midwest")           // East plays Midwest, as paired
  assert.ok(chalk.regions[0].rounds[0][0].chance > 0.99)             // 1 over 16
  const a = r.cbbPlayBracket(bracket, ids, league, r.cbbRng(7)), b = r.cbbPlayBracket(bracket, ids, league, r.cbbRng(7))
  assert.equal(a.champion.team_id, b.champion.team_id)
  assert.deepEqual(a.upsets.map(g => g.winner.team_id), b.upsets.map(g => g.winner.team_id))
  // Over many brackets underdogs win about as often as their chance: an 8 beats a 9 a bit over half the time, and upsets happen.
  let eightWins = 0, upsets = 0
  const random = r.cbbRng(123)
  for (let i = 0; i < 400; i++) {
    const sim = r.cbbPlayBracket(bracket, ids, league, random)
    upsets += sim.upsets.length
    eightWins += sim.regions[0].rounds[0][1].winner.seed === 8 ? 1 : 0
  }
  const expected = r.cbbMatchup(ids.get(108), ids.get(109), "neutral", league).winA
  assert.ok(Math.abs(eightWins / 400 - expected) < 0.08)
  assert.ok(upsets / 400 > 8 && upsets / 400 < 25)
})

test("playing a bracket: First Four pairs play for their spot, and played games keep their real winner", () => {
  const { bracket, ids } = fullBracket()
  bracket.teams.push({ team_id: 117, seed: 16, region: "East", play_in: true })
  bracket.teams.find(t => t.team_id === 116).play_in = true
  ids.set(117, { team_id: 117, team: "East 16b", net: -40, offense: 0, defense: 0, tempo: 68 })
  const sim = r.cbbPlayBracket(bracket, ids, league, null)
  assert.equal(sim.regions[0].firstFour.length, 1)
  assert.equal(sim.regions[0].firstFour[0].winner.team_id, 116)
  // The real tournament had the 16 seed beat the 1: every bracket played from here has that result.
  bracket.results = [{ round: "Round of 64", home_id: 101, away_id: 116, home_score: 60, away_score: 61 }]
  const after = r.cbbPlayBracket(bracket, ids, league, null)
  const first = after.regions[0].rounds[0][0]
  assert.equal(first.winner.team_id, 116)
  assert.ok(first.played && first.upset && first.chance < 0.01)
  assert.notEqual(after.champion.team_id, 101)
  const html = text(r.cbbPlayedBracketHtml(after, ids))
  assert.match(html, /^East 2 wins it all|^Midwest 1 wins it all/)
  assert.match(html, /16 East 16 over 1 East 1 final/)
  assert.match(html, /upsets? by seed/)
  assert.match(r.cbbPlayedBracketHtml(r.cbbPlayBracket({ teams: [], final_four_pairs: [] }, ids, league, null), ids), /No bracket to play yet/)
})

// ---- Time Machine ----

// Three seasons of 60 teams. In 2011 the best teams stand further above the
// pack than in the other two, as the real ones have lately.
function eraHistory(){
  const season = (lift, champion) => ({ efficiency: 100, tempo: 70, tournament: champion !== null,
    teams: Array.from({ length: 60 }, (_, i) => {
      const net = (i < 50 ? 30 + lift : 0) - i * 0.5
      return [i + 1, net / 2, net / 2, i === 0 ? 74 : 70, 30 - Math.floor(i / 3), 5 + Math.floor(i / 3), i + 1 === champion ? 8 : i < 2 ? 5 : 0]
    }) })
  const names = {}
  for(let i = 1; i <= 60; i++){ names[i] = i === 1 ? "Alpha" : i === 2 ? "Beta" : `School ${i}` }
  return { sigma: 11.2, names, colors: { 1: "112233" }, seasons: { 2009: season(0, 1), 2010: season(0, null), 2011: season(6, 55) } }
}

test("era teams: every season on one scale, measured against its own best teams", () => {
  const teams = r.cbbEraTeams(eraHistory())
  assert.equal(teams.length, 180)
  assert.deepEqual(teams.slice(0, 5).map(t => t.rank), [1, 2, 3, 4, 5])
  assert.ok(teams.every((t, i) => t.team_id === i && (i === 0 || teams[i - 1].net >= t.net)))
  // 2011's top teams were 6 points further above average, but no further above their own peers.
  const best = season => teams.find(t => t.team === `${season} Alpha`)
  assert.ok(Math.abs(best(2011).net - best(2009).net) < 0.11)
  assert.ok(Math.abs(best(2009).net - 32) < 0.11)             // raw 30, lifted a third of 2011's extra 6
  assert.equal(best(2009).offense + best(2009).defense, best(2009).net)
  // Pace is carried as fast or slow for its season, on the shared floor.
  assert.equal(best(2009).tempo, 72)
  assert.equal(teams.find(t => t.team === "2009 Beta").tempo, 68)
  assert.equal(best(2009).color, "112233")
  assert.equal(r.cbbMarchLabel(best(2009)), "National champion")
  assert.equal(r.cbbMarchLabel(best(2011)), "Lost in the Elite 8")
  assert.equal(r.cbbMarchLabel(teams.find(t => t.team === "2009 School 9")), "Missed the tournament")
  assert.equal(r.cbbMarchLabel(best(2010)), "No tournament that year")
  assert.deepEqual(r.cbbEraTeams(null), [])
  const league = r.cbbEraLeague(eraHistory())
  assert.equal(league.sigma, 11.2)
  assert.equal(league.home_edge, 0)
})

test("the all-time bracket: champions get in, the rest by rating, seeded and snaked", () => {
  const teams = r.cbbEraTeams(eraHistory())
  const bracket = r.cbbAllTimeBracket(teams)
  assert.equal(bracket.teams.length, 64)
  const ids = new Set(bracket.teams.map(t => t.team_id))
  assert.equal(ids.size, 64)
  // 2011's champion was its 55th-best team: far below the cut, in anyway, as the last seed.
  const cinderella = teams.find(t => t.team === "2011 School 55")
  assert.ok(cinderella.rank > 64 && ids.has(cinderella.team_id))
  assert.equal(bracket.teams[63].team_id, cinderella.team_id)
  assert.equal(bracket.teams[63].seed, 16)
  // So the 63rd-best team is the one left out.
  assert.ok(ids.has(teams[62].team_id) && !ids.has(teams[63].team_id))
  // Four teams to a seed line, one per region; the top seed's region gets the worst 2 seed.
  for(let seed = 1; seed <= 16; seed++){
    assert.deepEqual(bracket.teams.filter(t => t.seed === seed).map(t => t.region).sort(), ["East", "Midwest", "South", "West"])
  }
  assert.equal(bracket.teams[0].region, "East")
  assert.equal(bracket.teams[7].region, "East")
  assert.deepEqual(bracket.final_four_pairs, [["East", "Midwest"], ["West", "South"]])
  assert.equal(r.cbbAllTimeBracket(teams, 16).teams.length, 16)
})

test("all-time bracket odds: shares of the runs, and the table that shows them", () => {
  const teams = r.cbbEraTeams(eraHistory())
  const byEra = r.cbbTeamIndex(teams), eraLeague = r.cbbEraLeague(eraHistory())
  const bracket = r.cbbAllTimeBracket(teams)
  const odds = r.cbbBracketOdds(bracket, byEra, eraLeague, 300, r.cbbRng(3))
  const sum = key => [...odds.values()].reduce((s, v) => s + v[key], 0)
  assert.ok(Math.abs(sum("champion") - 1) < 1e-9)
  assert.ok(Math.abs(sum("final_four") - 4) < 1e-9)
  assert.ok((odds.get(bracket.teams[0].team_id) || { champion: 0 }).champion > (odds.get(bracket.teams[63].team_id) || { champion: 0 }).champion)
  // Favorites only is one fixed bracket: the top seed wins every run.
  const chalk = r.cbbBracketOdds(bracket, byEra, eraLeague, 3, null)
  assert.ok(Math.abs(chalk.get(bracket.teams[0].team_id).champion - 1) < 1e-9)
  bracket.teams.forEach(t => Object.assign(t, odds.get(t.team_id) || { final_four: 0, champion: 0 }))
  const html = r.cbbAllTimeOddsHtml(bracket, byEra, 16)
  assert.match(text(html), /Seed Team Region Record That March Final Four here Wins it all/)
  assert.match(text(html), /Showing 16 of 64 teams/)
  assert.equal((r.cbbAllTimeOddsHtml(bracket, byEra, null).match(/<tr>/g) || []).length, 65)
  assert.match(text(r.cbbFavoritesHtml(bracket, byEra)), /\d+% (2009|2010|2011) Alpha 1 seed/)
})

test("playing one cross-era game: a real-looking score, no ties, and wins in line with the odds", () => {
  const teams = r.cbbEraTeams(eraHistory())
  const eraLeague = r.cbbEraLeague(eraHistory())
  const a = teams[0], b = teams[40]
  const expected = r.cbbMatchup(a, b, "neutral", eraLeague).winA
  const random = r.cbbRng(11)
  let wins = 0, overtime = 0, points = 0
  for(let i = 0; i < 2000; i++){
    const g = r.cbbPlayGame(a, b, eraLeague, random)
    assert.ok(Number.isInteger(g.scoreA) && Number.isInteger(g.scoreB) && g.scoreA !== g.scoreB && g.scoreB > 20)
    wins += g.scoreA > g.scoreB ? 1 : 0
    overtime += g.overtime ? 1 : 0
    points += g.scoreA + g.scoreB
  }
  assert.ok(Math.abs(wins / 2000 - expected) < 0.04)
  assert.ok(overtime > 0 && overtime < 200)
  assert.ok(points / 2000 > 125 && points / 2000 < 165)
  // The same seed plays the same game.
  assert.deepEqual(r.cbbPlayGame(a, b, eraLeague, r.cbbRng(5)), r.cbbPlayGame(a, b, eraLeague, r.cbbRng(5)))
})

test("the cross-era matchup card and the best-teams list", () => {
  const teams = r.cbbEraTeams(eraHistory())
  const eraLeague = r.cbbEraLeague(eraHistory())
  const a = teams.find(t => t.team === "2009 Alpha"), b = teams.find(t => t.team === "2011 School 55")
  const card = text(r.cbbEraMatchupHtml(a, b, eraLeague, null))
  assert.match(card, /2009 Alpha 30–5 · 🏆 National champion · rated \+32\.0, \d+(st|nd|rd|th) of the era/)
  assert.match(card, /2011 School 55 12–23 · 🏆 National champion/)
  assert.match(card, /On a neutral floor: 2009 Alpha \d+, 2011 School 55 \d+ on average/)
  assert.doesNotMatch(card, /Final/)
  const one = text(r.cbbEraMatchupHtml(a, b, eraLeague, { games: 1, winsA: 0, last: { scoreA: 70, scoreB: 75, overtime: true } }))
  assert.match(one, /Final \(overtime\): 2011 School 55 75 , 2009 Alpha 70/)
  assert.match(one, /One game played\./)
  const many = text(r.cbbEraMatchupHtml(a, b, eraLeague, { games: 100, winsA: 91, last: { scoreA: 80, scoreB: 61, overtime: false } }))
  assert.match(many, /Final: 2009 Alpha 80 , 2011 School 55 61/)
  assert.match(many, /100 games played: 2009 Alpha 91, 2011 School 55 9\./)
  assert.match(r.cbbEraMatchupHtml(a, a, eraLeague, null), /Pick two different teams/)
  assert.match(r.cbbEraMatchupHtml(a, undefined, eraLeague, null), /Pick two different teams/)

  const best = r.cbbBestEverHtml(teams, 25)
  assert.equal((best.match(/<tr>/g) || []).length, 26)
  assert.match(text(best), /# Team Record Rating Offense Defense That March 1 20(09|10|11) Alpha/)
  assert.match(text(best), /Showing 25 of 180\./)
  // A school search keeps each team's place in the full list.
  const beta = text(r.cbbBestEverHtml(teams, 60, "beta"))
  assert.equal((beta.match(/Beta/g) || []).length, 3)
  assert.doesNotMatch(beta, /Alpha|Showing/)
  assert.match(beta, new RegExp(`${teams.find(t => t.team === "2009 Beta").rank} 2009 Beta`))
  assert.match(r.cbbBestEverHtml(teams, 60, "no such school"), /No school by that name/)
})
