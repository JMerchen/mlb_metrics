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
  // Who's out is left to the What if? panel; the pick just says which side of the report it's on.
  assert.match(text(after), /After the final injury report/)
  assert.doesNotMatch(text(after), /Out:|Mahomes/)
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
  assert.equal(detail(html), "Bills 49%51% Rams 🗓️Forecast comes the day before the game Total points: we say 51 · Vegas 54.5 Before the final injury report")
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
  assert.deepEqual(totalLean(g), { side: "over", line: 37.5, gap: 44.6 - 37.5, result: "", close: null, clv: null })
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
