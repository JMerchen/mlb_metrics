// node --test docs/summary_render.test.js
const test = require("node:test")
const assert = require("node:assert/strict")
global.teamName = require("./teams.js").teamName
const {
  srPct, streakTilesHtml, btsPicksHtml, gamePicksRecordHtml, gamePickRowHtml, totalPointsText, teamEpaHtml,
  srWeather, pickBarHtml,
} = require("./summary_render.js")

const text = html => html.replace(/<[^>]+>/g, "").replace(/&middot;/g, "·").replace(/&nbsp;/g, " ").replace(/\s+/g, " ").trim()
// The card's face (what shows before it's clicked) and what opens below.
const face = html => text(html.match(/<summary>([\s\S]*)<\/summary>/)[1])
const detail = html => text(html.split("</summary>")[1])

test("teamName: nicknames per sport, unknown codes fall back", () => {
  const { teamName } = require("./teams.js")
  assert.equal(teamName("NYY", "mlb"), "Yankees")
  assert.equal(teamName("AZ", "mlb"), "Diamondbacks")
  assert.equal(teamName("ATH", "mlb"), "Athletics")
  assert.equal(teamName("ATL", "nfl"), "Falcons")
  assert.equal(teamName("ATL", "mlb"), "Braves")
  assert.equal(teamName("XYZ", "nfl"), "XYZ")
})

test("srPct: whole-number percentages", () => {
  assert.equal(srPct(0.589), "59%")
  assert.equal(srPct("0.5"), "50%")
  assert.equal(srPct(""), "-")
})

test("game row: nicknames, our number next to Vegas's", () => {
  const html = gamePickRowHtml({
    home_team: "NYY", away_team: "TB", predicted_winner: "NYY",
    predicted_probability: 0.589, market_predicted_winner_probability: 0.605, bet_units: 0,
    market_home_win_probability: 0.605,
  }, "mlb")
  assert.equal(face(html), "Rays at Yankees We say Yankees 59% · Vegas 61%")
  // Opened: each team's chance as a bar, and no weather for baseball.
  assert.equal(detail(html), "Rays 41%59% Yankees The tick is Vegas: Yankees 61%")
})

test("game row: a bet reads as a sentence, not units shorthand", () => {
  const html = gamePickRowHtml({
    home_team: "SEA", away_team: "LAC", predicted_winner: "SEA",
    predicted_probability: 0.82, market_predicted_winner_probability: "", bet_units: 1.73, bet_team: "SEA",
  }, "nfl")
  assert.equal(face(html), "Chargers at Seahawks We say Seahawks 82% Bet Seahawks · 1.7 units")
  assert.match(detail(html), /Before the final injury report$/)
})

test("NFL game row: outlined and labelled by the injury report; MLB rows aren't", () => {
  const after = gamePickRowHtml({
    home_team: "KC", away_team: "DEN", predicted_winner: "DEN", predicted_probability: 0.55,
    injury_report: "after", injury_note: "Patrick Mahomes (QB; Gardner Minshew starts)",
  }, "nfl")
  assert.match(after, /class="game report-after"/)
  assert.match(text(after), /After the final injury report/)
  assert.match(text(after), /Out: Patrick Mahomes \(QB; Gardner Minshew starts\)/)
  assert.match(gamePickRowHtml({ home_team: "KC", away_team: "DEN", predicted_winner: "KC" }, "nfl"), /class="game report-before"/)
  const mlb = gamePickRowHtml({ home_team: "NYY", away_team: "BOS", predicted_winner: "NYY" }, "mlb")
  assert.match(mlb, /class="game"/)
  assert.doesNotMatch(text(mlb), /injury report/)
})

test("record line while bets are paused: right N of M next to Vegas", () => {
  const line = text(gamePicksRecordHtml(
    { betting_enabled: false },
    { model_vs_market_n: 11, model_pick_accuracy: 0.4545, market_pick_accuracy: 0.5455 },
  ))
  assert.equal(line, "Right on 5 of 11 games so far (Vegas: 6 of 11). Not suggesting moneyline bets right now.")
})

test("record line while bets are on: won N of M, up or down units", () => {
  assert.equal(
    text(gamePicksRecordHtml({ betting_enabled: "True", n_bets_advised: 9, bets_won: 4, total_profit_units: 1.29 }, null)),
    "Our bets: won 4 of 9, up 1.3 units.",
  )
  assert.equal(
    text(gamePicksRecordHtml({ n_bets_advised: 9, bets_won: 2, total_profit_units: -3.4 }, null)),
    "Our bets: won 2 of 9, down 3.4 units.",
  )
  assert.equal(text(gamePicksRecordHtml({ n_bets_advised: 0 }, null)), "No bets settled yet.")
})

test("no statistician vocabulary on these panels", () => {
  const html = streakTilesHtml(
    { current_streak: 1, longest_streak: 10, day_survival_rate: 0.514, day_survival_rate_ci_low: 0.42,
      day_survival_rate_ci_high: 0.61, n_days_resolved: 105 },
    { name: "Jeremy Peña", team: "HOU", Game_Hit_Probability: 0.658 },
  ) + btsPicksHtml([{ name: "Chandler Simpson", combined_probability: 0.657, grade: "recommended" }])
  for (const jargon of ["CI", "n=", "GHP", "Approach", "Brier", "Speculative", "Recommended"]) {
    assert.ok(!text(html).includes(jargon), `"${jargon}" should not appear`)
  }
  assert.ok(text(html).includes("Astros · 66% to get a hit"))
  assert.ok(text(html).includes("1 day"))
})

test("NFL game row: predicted total next to Vegas's; nothing for MLB", () => {
  const html = gamePickRowHtml({
    home_team: "LA", away_team: "BUF", predicted_winner: "LA", predicted_probability: 0.51,
    market_predicted_winner_probability: 0.57, bet_units: 0, predicted_total: 51.37, market_total: 54.5,
  }, "nfl")
  assert.equal(face(html), "Bills at Rams We say Rams 51% · Vegas 57%")
  assert.equal(detail(html), "Bills 49%51% Rams For the RamsPlaying at homeVegas agrees: Rams 57% 🗓️Forecast comes the day before the game Total points: we say 51 · Vegas 54.5 Before the final injury report")
  assert.equal(text(totalPointsText({ predicted_total: "47.6", market_total: "" })), "Total points: we say 48")
  // With the range from the export: the range, not one number.
  assert.equal(text(totalPointsText({ predicted_total: 47.6, predicted_total_low: 39, predicted_total_high: 57, market_total: 47.5 })),
    "Total points: we say 39–57 · Vegas 47.5")
  const card = gamePickRowHtml({ home_team: "LA", away_team: "BUF", predicted_winner: "LA", predicted_probability: 0.51,
    predicted_total: 47.6, predicted_total_low: 39, predicted_total_high: 57 }, "nfl")
  assert.match(detail(card), /Total points: we say 39–57 About half of games finish inside our range\./)
  assert.equal(totalPointsText({ predicted_total: "" }), "")
  assert.equal(totalPointsText({}), "")
})

test("NFL EPA table: both teams, offense and defense with league rank; empty without ratings", () => {
  const html = teamEpaHtml({
    away_team: "SEA", home_team: "DEN",
    away_off_epa: 0.081, away_off_rank: 6, away_def_epa: -0.034, away_def_rank: 10,
    home_off_epa: "-0.02", home_off_rank: "21", home_def_epa: "0.05", home_def_rank: "28",
  }, "nfl")
  assert.equal(text(html.replace(/<\/t[dh]>/g, " ")), "EPA per play Offense Defense Seahawks +0.08 6th −0.03 10th Broncos −0.02 21st +0.05 28th")
  assert.ok(text(teamEpaHtml({ away_team: "SEA", home_team: "DEN", away_off_epa: -0.004, home_off_epa: 0.003 }, "nfl")).includes("Seahawks0.00"))
  assert.equal(teamEpaHtml({ away_team: "SEA", home_team: "DEN" }, "nfl"), "")
  assert.equal(teamEpaHtml({ away_team: "SEA", home_team: "DEN", home_off_epa: "", away_def_epa: "" }, "nfl"), "")
})

test("Beat the Streak after the regular season: final record, season over, no 'today' picks", () => {
  const { btsSeasonOver, btsSeasonOverHtml } = require("./summary_render.js")
  const summary = { current_streak: 1, longest_streak: 10, day_survival_rate: 0.52, n_days_resolved: 100,
                    season_over: "True", season_end: "2026-09-27" }
  assert.equal(btsSeasonOver(summary), true)
  assert.equal(btsSeasonOver({ season_over: "False" }), false)
  assert.equal(btsSeasonOver({}), false)
  const tiles = text(streakTilesHtml(summary, { name: "Jeremy Peña", team: "HOU", Game_Hit_Probability: 0.6 }).replace(/<\/div>/g, " "))
  assert.ok(tiles.includes("Season Over regular season ended Sep 27"))
  assert.ok(!tiles.includes("Best hitter today"))
  assert.ok(tiles.includes("at the end of the season"))
  assert.equal(text(btsSeasonOverHtml(summary)), "The regular season ended Sep 27. Beat the Streak picks return on Opening Day.")
})

test("NFL weather: a symbol and plain words, indoors, or not yet", () => {
  assert.equal(srWeather({}), null)
  assert.deepEqual(srWeather({ weather_indoor: 1, weather_temp: 30 }), { indoor: true, icon: "🏟️", label: "Indoors" })
  const wet = srWeather({ weather_indoor: 0, weather_temp: 51.4, weather_wind: 9, weather_rain: 0.8, weather_code: 63 })
  assert.equal(wet.icon, "🌧️")
  assert.equal(wet.label, "51°F, wind 9 mph, 80% chance of rain")
  assert.equal(srWeather({ weather_temp: 28, weather_wind: 6, weather_rain: 0, weather_code: 73 }).label, "28°F, wind 6 mph, snow")
  assert.equal(srWeather({ weather_temp: 60, weather_wind: 22, weather_rain: 0.1, weather_code: 3 }).icon, "💨")
  assert.equal(srWeather({ weather_temp: 70, weather_wind: 4, weather_rain: 0, weather_code: 0 }).icon, "☀️")
  const html = gamePickRowHtml({
    home_team: "TB", away_team: "ATL", predicted_winner: "TB", predicted_probability: 0.6,
    weather_indoor: 0, weather_temp: 78, weather_wind: 9, weather_rain: 0.9, weather_code: 63,
  }, "nfl")
  assert.match(face(html), /🌧️/)
  assert.match(detail(html), /🌧️78°F, wind 9 mph, 90% chance of rain/)
})

test("pick bar: away left, home right, our side filled, Vegas ticked", () => {
  const html = pickBarHtml({ home_team: "TB", away_team: "ATL", predicted_winner: "ATL", predicted_probability: 0.7,
    market_home_win_probability: 0.4 }, "nfl")
  assert.match(html, /pickBarAway picked" style="width:70.0%"/)
  assert.match(html, /pickBarVegas" style="left:60.0%"/)
  assert.equal(text(html), "Falcons 70%30% Buccaneers The tick is Vegas: Falcons 60%")
  assert.equal(pickBarHtml({ home_team: "TB", away_team: "ATL" }, "nfl"), "")
})

test("NFL totals: a lean when we're far from Vegas, and games outside our range marked", () => {
  const { totalLean, totalLeanBadgeHtml, totalRangeNote } = require("./summary_render.js")
  const g = { home_team: "BAL", away_team: "JAX", predicted_winner: "JAX", predicted_probability: 0.8,
    predicted_total: 44.6, predicted_total_low: 36, predicted_total_high: 54, market_total: 37.5,
    total_lean: "over", total_lean_line: 37.5 }
  assert.deepEqual(totalLean(g), { side: "over", line: 37.5, gap: 44.6 - 37.5, result: "", close: null, clv: null, weather: 0 })
  assert.equal(text(totalLeanBadgeHtml(g)), "Lean OVER 37.5")
  const html = gamePickRowHtml(g, "nfl")
  assert.match(face(html), /Lean OVER 37\.5/)
  assert.match(detail(html), /We lean over Vegas's 37\.5: our total is 7 points higher\./)
  assert.equal(text(totalLeanBadgeHtml({ ...g, total_lean_result: "loss" })), "Lean OVER 37.5 · lost")
  assert.equal(totalLean({ ...g, total_lean: "" }), null)
  assert.equal(totalRangeNote({ ...g, actual_total: 61, total_range_result: "over", total_range_miss: 7 }), "Finished 61 - 7 over our range")
  assert.equal(totalRangeNote({ ...g, actual_total: 40, total_range_result: "in", total_range_miss: 0 }), "Finished 40 - inside our range")
  assert.equal(totalRangeNote(g), "")
  // MLB cards never lean.
  assert.doesNotMatch(gamePickRowHtml({ ...g, home_team: "NYY", away_team: "BOS", predicted_winner: "NYY" }, "mlb"), /leanBadge/)
  assert.match(gamePicksRecordHtml({ betting_enabled: false, total_leans_won: 5, total_leans_lost: 3 }, null), /Over\/under leans: won <b>5 of 8<\/b>\./)
  assert.doesNotMatch(gamePicksRecordHtml({ betting_enabled: false }, null), /leans/)
})

test("NFL totals: where the line closed against our lean", () => {
  const { totalLeanCloseNote } = require("./summary_render.js")
  const g = { home_team: "ATL", away_team: "CIN", predicted_winner: "CIN", predicted_probability: 0.6,
    predicted_total: 47, market_total: 52.5, total_lean: "under", total_lean_line: 52.5 }
  assert.equal(totalLeanCloseNote(g), "")
  assert.equal(totalLeanCloseNote({ ...g, total_lean_close: 50.5, total_lean_clv: 2 }), "The line closed at 50.5 - 2 points our way")
  assert.equal(totalLeanCloseNote({ ...g, total_lean_close: 53, total_lean_clv: -0.5 }), "The line closed at 53 - 0.5 points against us")
  assert.equal(totalLeanCloseNote({ ...g, total_lean_close: 52.5, total_lean_clv: 0 }), "The line closed at 52.5 - where we got it")
  assert.match(detail(gamePickRowHtml({ ...g, total_lean_close: 50.5, total_lean_clv: 2 }, "nfl")), /The line closed at 50\.5 - 2 points our way/)
})

test("NFL totals: a lean's reason, with the weather's part", () => {
  const { totalLeanReason } = require("./summary_render.js")
  const g = { predicted_total: 42.4, total_lean: "under", total_lean_line: 46.5 }
  assert.equal(totalLeanReason(g), "We lean under Vegas's 46.5: our total is 4 points lower.")
  assert.equal(totalLeanReason({ ...g, total_lean_weather: -3.2 }),
    "We lean under Vegas's 46.5: our total is 4 points lower, and the forecast's rain and wind take off about 3 more.")
  // Our total right at the line, the weather alone makes the lean.
  assert.equal(totalLeanReason({ predicted_total: 46.5, total_lean: "under", total_lean_line: 46.5, total_lean_weather: -5.1 }),
    "We lean under Vegas's 46.5: the forecast's rain and wind take off about 5 points.")
  assert.equal(totalLeanReason({}), "")
})

test("NFL totals: a Tuesday lean on hold for the injury report", () => {
  const g = { home_team: "KC", away_team: "DEN", predicted_winner: "KC", predicted_probability: 0.6, predicted_total: 40,
    predicted_total_low: 31, predicted_total_high: 49, market_total: 47.5,
    total_lean_held: "KC QB Patrick Mahomes (questionable on the last report)" }
  const html = gamePickRowHtml(g, "nfl")
  assert.match(face(html), /Lean on hold/)
  assert.match(detail(html), /Lean on hold until this week's injury report: KC QB Patrick Mahomes \(questionable on the last report\)\./)
})

test("record line falls back to every pick on record right after a model change", () => {
  const line = text(gamePicksRecordHtml(
    { betting_enabled: false, model_vs_market_n: 120, model_pick_accuracy: 0.65, market_pick_accuracy: 0.7 },
    { model_vs_market_n: 0 },
  ))
  assert.equal(line, "Right on 78 of 120 games so far (Vegas: 84 of 120). Not suggesting moneyline bets right now.")
})

test("NFL spread: Vegas's number, ours, and the side that puts us on", () => {
  const { spreadText, spreadInfo } = require("./summary_render.js")
  const g = { home_team: "MIN", away_team: "BUF", predicted_winner: "BUF", predicted_probability: 0.58,
    spread_line: -3.5, predicted_margin: -2.6, spread_side: "MIN" }
  assert.equal(text(spreadText(g, "nfl")), "Spread: Vegas has Bills by 3.5 · we have Bills by 2.5 (our side: Vikings +3.5)")
  assert.equal(text(spreadText({ ...g, predicted_margin: -6.1, spread_side: "BUF", spread_result: "win" }, "nfl")),
    "Spread: Vegas has Bills by 3.5 · we have Bills by 6 (our side: Bills −3.5, covered)")
  assert.equal(text(spreadText({ ...g, spread_line: 0, predicted_margin: 0.1, spread_side: "MIN" }, "nfl")),
    "Spread: Vegas has a pick'em · we have it even (our side: Vikings pick'em)")
  assert.equal(spreadInfo({ ...g, spread_line: "" }, "nfl"), null)
  assert.equal(spreadText({ home_team: "MIN", away_team: "BUF" }, "nfl"), "")
  const card = detail(gamePickRowHtml(g, "nfl"))
  assert.match(card, /Spread: Vegas has Bills by 3\.5/)
  assert.match(card, /coin flip in testing/)
  assert.doesNotMatch(gamePickRowHtml({ home_team: "NYY", away_team: "BOS", predicted_winner: "NYY", spread_line: 1, predicted_margin: 2 }, "mlb"), /Spread/)
})

test("NFL card: the case for the pick and what cuts against it", () => {
  const { pickReasons, pickReasonsHtml } = require("./summary_render.js")
  const g = { home_team: "MIN", away_team: "BUF", predicted_winner: "BUF", predicted_probability: 0.58,
    market_predicted_winner_probability: 0.62,
    away_off_rank: 1, home_off_rank: 28, away_def_rank: 23, home_def_rank: 2,
    away_qb: "Josh Allen", home_qb: "Carson Wentz", away_qb_rank: 2, home_qb_rank: 25,
    injury_out_home: "Kyler Murray (QB), Jauan Jennings (WR), Byron Murphy (CB)", injury_out_away: "" }
  const r = pickReasons(g, "nfl")
  assert.equal(r.pick, "Bills")
  assert.deepEqual(r.for, [
    "Better offense: 1st to the Vikings' 28th",
    "The Vikings are without Kyler Murray at quarterback; Carson Wentz starts",
    "Better quarterback play lately: Josh Allen ranks 2nd among this week's starters, Carson Wentz 25th",
    "The Vikings are missing 2 regulars",
    "Vegas agrees: Bills 62%",
  ])
  assert.deepEqual(r.against, ["The Vikings have the better defense: 2nd to 23rd", "Playing on the road"])
  const html = pickReasonsHtml(g, "nfl")
  assert.match(html, /For the Bills/)
  assert.match(html, /class="whyAgainst"/)
  // Close ranks aren't an edge either way; a pick against Vegas says so.
  const even = pickReasons({ home_team: "KC", away_team: "DEN", predicted_winner: "KC", home_off_rank: 10, away_off_rank: 12,
    market_predicted_winner_probability: 0.44 }, "nfl")
  assert.deepEqual(even.for, ["Playing at home"])
  assert.deepEqual(even.against, ["Vegas favors the Broncos (56%)"])
})

test("NFL card: who's out, or in doubt before the final report", () => {
  const { injuryLineHtml } = require("./summary_render.js")
  const g = { home_team: "MIN", away_team: "BUF", injury_report: "after" }
  assert.equal(text(injuryLineHtml({ ...g, injury_out_home: "Kyler Murray (QB), Jauan Jennings (WR)", injury_out_away: "Josh Palmer (WR)" }, "nfl")),
    "Out: Bills - Josh Palmer (WR) · Vikings - Kyler Murray (QB), Jauan Jennings (WR)")
  // A repriced pick from before the by-team lists keeps its one note.
  assert.equal(text(injuryLineHtml({ ...g, injury_note: "Jauan Jennings, Josh Palmer" }, "nfl")), "Out: Jauan Jennings, Josh Palmer")
  assert.equal(text(injuryLineHtml(g, "nfl")), "The final injury report took no regulars out.")
  assert.equal(text(injuryLineHtml({ home_team: "MIN", away_team: "BUF", injury_report: "before",
    starters_in_doubt: "MIN QB Kyler Murray (questionable on the last report)" }, "nfl")),
    "In doubt: MIN QB Kyler Murray (questionable on the last report)")
  assert.equal(injuryLineHtml({ home_team: "MIN", away_team: "BUF", injury_report: "before" }, "nfl"), "")
})

test("NFL card and page: the track record behind a pick", () => {
  const { trackRecordNote, trackRecordTableHtml } = require("./summary_render.js")
  const table = [
    { group: "confidence", label: "55–60%", low: "0.55", high: "0.6", season_games: "23", season_right: "16", backtest_games: "421", backtest_right: "240" },
    { group: "confidence", label: "75%+", low: "0.75", high: "1.0", season_games: "0", season_right: "0", backtest_games: "", backtest_right: "" },
    { group: "vegas", label: "with", low: "", high: "", season_games: "104", season_right: "74", backtest_games: "1650", backtest_right: "1122" },
    { group: "vegas", label: "against", low: "", high: "", season_games: "31", season_right: "17", backtest_games: "310", backtest_right: "136" },
  ]
  const g = { home_team: "MIN", away_team: "BUF", predicted_winner: "BUF", predicted_probability: 0.58, market_predicted_winner_probability: 0.62 }
  assert.equal(trackRecordNote(g, table), "Picks this sure (55–60%) have been right 57% of the time in testing (421 games), 16 of 23 this season.")
  assert.equal(trackRecordNote({ ...g, market_predicted_winner_probability: 0.45 }, table),
    "Picks this sure (55–60%) have been right 57% of the time in testing (421 games), 16 of 23 this season."
    + " This one goes against Vegas's favorite; those picks have been right 44% of the time in testing (310 games), 17 of 31 this season.")
  assert.equal(trackRecordNote({ ...g, predicted_probability: 0.9 }, table), "")   // a band with no games yet
  assert.equal(trackRecordNote(g, []), "")
  assert.match(detail(gamePickRowHtml(g, "nfl", { confidence: table })), /Picks this sure \(55–60%\)/)
  const html = text(trackRecordTableHtml(table, { total_leans_won: 5, total_leans_lost: 3, total_lean_backtest_games: 237,
    total_lean_backtest_won_rate: 0.62, spread_sides_won: 0, spread_sides_lost: 0, spread_backtest_games: 1923, spread_backtest_won_rate: 0.5 }))
  assert.match(html, /We say 55–60%16 of 23 70%57% of 421/)
  assert.match(html, /We pick against Vegas's favorite17 of 31 55%44% of 310/)
  assert.match(html, /Over\/under leans5 of 8 63%62% of 237/)
  assert.match(html, /Our side of the spread-50% of 1,923/)
  assert.equal(trackRecordTableHtml([], {}), "")
})
