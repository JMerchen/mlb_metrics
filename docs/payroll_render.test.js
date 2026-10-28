// node --test docs/payroll_render.test.js
const test = require("node:test")
const assert = require("node:assert/strict")
global.teamName = require("./teams.js").teamName
Object.assign(global, require("./mlb_render.js"))
const p = require("./payroll_render.js")

const text = html => html.replace(/<\/(div|span|td|th|button|b|small)>/g, " ").replace(/<[^>]+>/g, "").replace(/\s+/g, " ").trim()

const row = (season, franchise, extra) => ({
  season: String(season), franchise, payroll: "100000000", wins: "81", losses: "81", games: "162", status: "final",
  win_avg: "1", pe_index: "1", relative_payroll: "1", payroll_rank: "15", wins_rank: "15", expected_win_avg: "1",
  expected_wins: "81", wins_above_expected: "0", fo_score: "0", cash_flow: "0", baseline_win_avg: "1", neighbors: "", ...extra,
})
const SEASONS = [
  row(2025, "TB", { payroll: "80000000", wins: "95", losses: "67", win_avg: "1.17284", pe_index: "2.2", relative_payroll: "0.6",
    payroll_rank: "28", wins_rank: "3", expected_wins: "85.2", expected_win_avg: "1.05185", wins_above_expected: "9.8", fo_score: "0.04", cash_flow: "-0.1",
    neighbors: "2004 MIA;2003 KC" }),
  row(2025, "NYY", { payroll: "300000000", wins: "70", losses: "92", win_avg: "0.864", pe_index: "0.4", relative_payroll: "2.5",
    payroll_rank: "1", wins_rank: "25", expected_wins: "95", expected_win_avg: "1.17284", wins_above_expected: "-25", fo_score: "-0.02", cash_flow: "0.3" }),
  row(2025, "WSH", {}),
  row(2004, "MIA", { win_avg: "1.1", relative_payroll: "0.62" }),
  row(2003, "KC", { win_avg: "0.95", relative_payroll: "0.58" }),
  row(2003, "WSH", { wins: "83", losses: "79" }),
  row(2026, "TB", { status: "forecast", wins: "", losses: "", win_avg: "", pe_index: "", wins_rank: "", wins_above_expected: "", expected_wins: "88" }),
]
const LEAGUE = [
  { season: "2003", teams: "30", average_payroll: "70000000", wins: "2430", price_per_win: "864198", price_per_81_wins: "864198" },
  // 2020: 60 games, so per actual win it's 2.7x dearer; per 81 wins it isn't.
  { season: "2020", teams: "30", average_payroll: "140000000", wins: "900", price_per_win: "4666667", price_per_81_wins: "1728395" },
  { season: "2025", teams: "30", average_payroll: "160000000", wins: "2430", price_per_win: "1975309", price_per_81_wins: "1975309" },
  { season: "2026", teams: "30", average_payroll: "170000000", wins: "", price_per_win: "", price_per_81_wins: "2098765" },
]

test("formats money, ratios and signed numbers", () => {
  assert.equal(p.payMoney(121063657), "$121.1M")
  assert.equal(p.payMoney(4.2e9), "$4.20B")
  assert.equal(p.paySigned(9.84), "+9.8")
  assert.equal(p.paySigned(-25), "−25.0")
  assert.equal(p.paySigned(0.01), "0.0")
  assert.equal(p.payRatio(""), "-")
})

test("old franchise names by season", () => {
  assert.equal(p.payTeamName("WSH", 2003), "Expos")
  assert.equal(p.payTeamName("WSH", 2005), "Nationals")
  assert.equal(p.payTeamName("TB", 2007), "Devil Rays")
  assert.equal(p.payTeamName("CLE", 2021), "Indians")
  assert.equal(p.payTeamName("CLE", 2022), "Guardians")
})

test("opens on the latest season with games, not a forecast", () => {
  assert.equal(p.payDefaultSeason(SEASONS), 2025)
  assert.deepEqual(p.paySeasons(SEASONS), [2026, 2025, 2004, 2003])
})

test("season table sorts by effectiveness and shows the record and chips", () => {
  const rows = p.paySeasonRows(SEASONS, 2025)
  const html = p.paySeasonTableHtml(rows, "pe_index")
  const out = text(html)
  assert.ok(out.indexOf("Rays") < out.indexOf("Nationals") && out.indexOf("Nationals") < out.indexOf("Yankees"))
  assert.match(out, /Rays \$80\.0M 28th 95-67 3rd 2\.20 85 \+9\.8 \+3\.2 −0\.10/)
  assert.match(html, /data-pay-sort="wins_above_expected"/)
  assert.match(html, /data-pay-team="TB"/)
  const byPayroll = text(p.paySeasonTableHtml(rows, "payroll_rank"))
  assert.ok(byPayroll.indexOf("Yankees") < byPayroll.indexOf("Rays"))
})

test("a forecast season shows payroll and expected wins only", () => {
  const out = text(p.paySeasonTableHtml(p.paySeasonRows(SEASONS, 2026), "expected_wins"))
  assert.match(out, /Rays \$100\.0M 15th 88/)
  assert.doesNotMatch(out, /Record|effectiveness/)
})

test("hero tiles name the best and worst value and the price of a win", () => {
  const out = text(p.payHeroHtml(p.paySeasonRows(SEASONS, 2025), LEAGUE, 2025, 9.4))
  assert.match(out, /Best value Rays 2\.20 effectiveness · 95-67 on \$80\.0M/)
  assert.match(out, /Most above expected Rays \+9\.8 wins \(expected 85\)/)
  assert.match(out, /Least value Yankees 0\.40/)
  assert.match(out, /Price of a win \$2\.0M/)
  assert.match(out, /miss by 9\.4 wins/)
  assert.match(out, /average payroll \$160\.0M ÷ 81 wins/)
  assert.match(text(p.payHeroHtml(p.paySeasonRows(SEASONS, 2026), LEAGUE, 2026, null)), /Price of a win \$2\.1M average payroll \$170\.0M ÷ 81 wins/)
})

test("model miss is the mean absolute wins vs. expected per 162", () => {
  assert.equal(p.payModelMiss(SEASONS.slice(0, 3)), (9.8 + 25 + 0) / 3)
})

test("scatter: one dot per team with a tip, labels the extremes, break-even curve", () => {
  const svg = p.payScatterSvg(p.paySeasonRows(SEASONS, 2025), "WSH", 720)
  assert.equal((svg.match(/<g [^>]*class="payDot/g) || []).length, 3)
  assert.match(svg, /Rays: 95-67 on \$80\.0M \(0\.60× the average payroll\) - effectiveness 2\.20, expected 85 wins/)
  assert.match(svg, /class="payCurve"/)
  assert.match(svg, /paySelectedDot/)
  assert.equal(p.payScatterSvg(p.paySeasonRows(SEASONS, 2026)), "")
})

test("team chart and the seasons its expected wins came from", () => {
  const svg = p.payTeamChartSvg(SEASONS, "TB", 720)
  assert.match(svg, /payActualLine/)
  assert.match(svg, /2025 Rays: 95-67, expected 85/)
  const tb = SEASONS[0]
  const out = text(p.payNeighborsHtml(SEASONS, tb))
  assert.match(out, /2025 Rays spent 0\.60× the average payroll/)
  assert.match(out, /averaged 81\.0 wins per 162/)
  assert.match(out, /add \+3\.2 → 85 expected wins ?, actual 95/)
  assert.match(out, /2004 MIA 89/)
  assert.match(text(p.payNeighborsHtml(SEASONS, SEASONS[5])), /No seasons 11 or more years before 2003 Expos/)
})

test("grids: years by team with tied ranks averaged; final seasons only", () => {
  const g = p.payGrid(SEASONS, "franchise", "pe_index")
  assert.equal(g.columns[0], "NYY")
  assert.deepEqual(g.years, [2003, 2004, 2025])
  assert.equal(g.cell(2025, "TB"), 2.2)
  assert.equal(g.cell(2026, "TB"), null)
  const ranks = p.payGrid(SEASONS, "wins_rank", "pe_index")
  assert.equal(ranks.columns.length, 30)
  assert.equal(ranks.cell(2025, 15), 1)
  const html = p.payGridHtml(SEASONS, "team_wae")
  assert.match(html, /2025 Rays: wins vs\. expected \+10/)
  assert.match(text(p.payGridTabsHtml("payrank_wae")), /By team By payroll rank By wins rank Effectiveness Wins vs\. expected Front office score Wins rank/)
})

test("league chart plots each played season's price of a win over 81 wins", () => {
  const svg = p.payLeagueChartSvg(LEAGUE)
  assert.match(svg, /2025: \$2\.0M of payroll per win \(average payroll ÷ 81\)/)
  assert.match(svg, /2020: \$1\.7M of payroll per win/)
  assert.doesNotMatch(svg, /2026:/)
})

test("workbook: the original's sheets, neighbours spelled out, grid colour scales", () => {
  const sheets = p.payWorkbookSheets(SEASONS, [{ franchise: "TB", seasons: "29", average_effectiveness: "1.9" }], LEAGUE)
  const names = sheets.map(s => s.name)
  assert.deepEqual(names.slice(0, 4), ["Payroll Effectiveness", "Front Office Score", "Franchises", "League"])
  assert.equal(names.length, 16)
  assert.ok(names.includes("Team by Year Wins Above Expect") && names.includes("Wins by Year Front Office"))
  assert.ok(names.every(n => n.length <= 31))
  const main = sheets[0]
  assert.equal(main.rows.length, SEASONS.length)
  assert.equal(main.rows[0][main.columns.findIndex(c => c[0] === "Team Payroll")], 80000000)
  assert.deepEqual(main.scales.map(s => main.columns[s.column][0]), ["Payroll Effectiveness Index", "FO Score", "Cash Flow", "Wins Above Expected"])
  const fo = sheets[1]
  assert.equal(fo.rows[0][fo.columns.findIndex(c => c[0] === "NN1")], "2004 MIA (1.100)")
  const winRank = sheets.find(s => s.name === "Team By Year Win Rank")
  assert.equal(winRank.gridScale.higherIsGreen, false)
  assert.deepEqual(winRank.rows.map(r => r[0]), [2003, 2004, 2025])
})

test("colour runs green (value) through grey to red (overspending), inked to read", () => {
  assert.equal(p.payColor(100), "rgb(71,171,79)")
  assert.equal(p.payColor(50), "rgb(91,98,112)")
  assert.equal(p.payColor(0), "rgb(165,50,52)")
  assert.equal(p.payInk(100), "#0b0d10")
  assert.equal(p.payInk(0), "#fff")
  assert.equal(p.payInk(50), "#fff")
  const chip = p.paySeasonTableHtml(p.paySeasonRows(SEASONS, 2025), "pe_index")
  // Three teams: the best sits on the green side, the worst on the red.
  assert.match(chip, /background:rgb\(78,146,90\);color:#0b0d10">2\.20/)
  assert.match(chip, /background:rgb\(140,66,72\);color:#fff">0\.40/)
})
