// node --test docs/summary_render.test.js
const test = require("node:test")
const assert = require("node:assert/strict")
global.teamName = require("./teams.js").teamName
const {
  srPct, streakTilesHtml, btsPicksHtml, gamePicksRecordHtml, gamePickRowHtml, totalPointsText, teamEpaHtml,
} = require("./summary_render.js")

const text = html => html.replace(/<[^>]+>/g, "").replace(/&middot;/g, "·").replace(/&nbsp;/g, " ").replace(/\s+/g, " ").trim()

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
  const row = text(gamePickRowHtml({
    home_team: "NYY", away_team: "TB", predicted_winner: "NYY",
    predicted_probability: 0.589, market_predicted_winner_probability: 0.605, bet_units: 0,
  }, "mlb"))
  assert.equal(row, "Rays at Yankees We say Yankees 59% · Vegas 61%")
})

test("game row: a bet reads as a sentence, not units shorthand", () => {
  const row = text(gamePickRowHtml({
    home_team: "SEA", away_team: "LAC", predicted_winner: "SEA",
    predicted_probability: 0.82, market_predicted_winner_probability: "", bet_units: 1.73, bet_team: "SEA",
  }, "nfl"))
  assert.equal(row, "Chargers at Seahawks We say Seahawks 82% Bet Seahawks · 1.7 units Before the final injury report")
})

test("NFL game row: outlined and labelled by the injury report; MLB rows aren't", () => {
  const after = gamePickRowHtml({
    home_team: "KC", away_team: "DEN", predicted_winner: "DEN", predicted_probability: 0.55,
    injury_report: "after", injury_note: "Patrick Mahomes (QB; Gardner Minshew starts)",
  }, "nfl")
  assert.match(after, /class="game report-after"/)
  assert.match(text(after), /After the final injury report · Out: Patrick Mahomes \(QB; Gardner Minshew starts\)/)
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
  assert.equal(line, "Right on 5 of 11 games so far (Vegas: 6 of 11). Not suggesting bets right now.")
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
  const row = text(gamePickRowHtml({
    home_team: "LA", away_team: "BUF", predicted_winner: "LA", predicted_probability: 0.51,
    market_predicted_winner_probability: 0.57, bet_units: 0, predicted_total: 51.37, market_total: 54.5,
  }, "nfl"))
  assert.equal(row, "Bills at Rams We say Rams 51% · Vegas 57%Total points: we say 51 · Vegas 54.5 Before the final injury report")
  assert.equal(text(totalPointsText({ predicted_total: "47.6", market_total: "" })), "Total points: we say 48")
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
