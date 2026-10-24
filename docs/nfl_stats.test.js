// node --test docs/nfl_stats.test.js
const test = require("node:test")
const assert = require("node:assert/strict")
global.teamName = require("./teams.js").teamName
Object.assign(global, require("./mlb_render.js"))
const n = require("./nfl_stats.js")

const text = html => html.replace(/<\/(div|span|td|button)>/g, " ").replace(/<[^>]+>/g, "").replace(/\s+/g, " ").trim()

test("pictures: ESPN logos with its two odd codes; headshots as given", () => {
  assert.match(n.nflLogoUrl("KC"), /nfl\/500\/kc\.png$/)
  assert.match(n.nflLogoUrl("LA"), /\/lar\.png$/)
  assert.match(n.nflLogoUrl("WAS"), /\/wsh\.png$/)
  assert.equal(n.nflLogoUrl(""), "")
  assert.ok(n.nflAvatar("https://x/y", "Patrick Mahomes").includes(">PM<"))
  assert.ok(!n.nflAvatar("", "Patrick Mahomes").includes("<img"))
})

const SCHEDULE = [
  { team: "KC", week: "1", date: "2026-09-14", opponent: "DEN", home: "1.0", points_for: "31", points_against: "10", result: "W", margin: "21" },
  { team: "KC", week: "2", date: "2026-09-20", opponent: "BUF", home: "0.0", points_for: "20", points_against: "20", result: "T", margin: "0" },
  { team: "KC", week: "3", date: "2026-09-27", opponent: "MIA", home: "0.0", points_for: "17", points_against: "24", result: "L", margin: "-7" },
  { team: "KC", week: "4", date: "", opponent: "", home: "", result: "BYE" },
  { team: "KC", week: "5", date: "2026-10-11", opponent: "LV", home: "1.0", result: "" },
]

test("schedule sheet: three panels, marks by result, byes, upcoming, record", () => {
  const sheet = n.nflScheduleSheetHtml(SCHEDULE, "KC")
  assert.equal((sheet.match(/class="paperPanel"/g) || []).length, 3)
  assert.ok(text(sheet).includes("1-1-1 · point differential +14 · next: vs. Raiders, Oct 11"))
  assert.ok(sheet.includes("Week 1, Sep 14: vs. Broncos - W 31–10"))
  assert.ok(sheet.includes("Week 2, Sep 20: at Bills - T 20–20"))
  assert.ok(text(sheet).includes("Bye week"))
  assert.ok(sheet.includes("Week 5, Oct 11: vs. Raiders - not played yet"))
  assert.equal(n.nflScheduleInk("W", 21), n.nflScheduleInk("W", 35))
  assert.equal(n.nflScheduleInk("L", -1), "rgb(232,132,132)")
  assert.ok(n.nflScheduleSheetHtml(SCHEDULE, "NYJ").includes("No schedule"))
})

const STATS = [
  { team: "SF", games: "6", wins: "6", losses: "0", ties: "0", net_epa_play: "0.243", point_diff_pg: "11.2", off_epa_play: "0.18", def_epa_play: "-0.06" },
  { team: "MIA", games: "5", wins: "0", losses: "5", ties: "0", net_epa_play: "-0.223", point_diff_pg: "-12.4", off_epa_play: "-0.12", def_epa_play: "0.10" },
  { team: "KC", games: "5", wins: "5", losses: "0", ties: "0", net_epa_play: "0.224", point_diff_pg: "11.4", off_epa_play: "0.14", def_epa_play: "-0.08" },
]
const UNITS = [{ team: "SF", qb_plus: "150", qb_plus_season: "150" }, { team: "MIA", qb_plus: "53", qb_plus_season: "53" }]

test("teams: power rankings by net EPA; lower-is-better ranks; profiles one or two", () => {
  const rows = n.nflTeamRows(STATS, UNITS)
  assert.equal(rows[0].win_pct, 1)
  assert.equal(rows[0].qb_plus, "150")
  const order = [...n.nflPowerRankingsHtml(rows, ["SF"]).matchAll(/data-team="(\w+)"/g)].map(m => m[1])
  assert.deepEqual(order, ["SF", "KC", "MIA"])
  assert.equal(n.nflTeamRank(rows, rows[0], "def_epa_play", -1), 2)   // KC allows less
  const one = n.nflTeamProfileHtml(["SF"], rows)
  assert.ok(text(one).includes("49ers 6-0 · 1st of 3 in net EPA"))
  const two = n.nflTeamProfileHtml(["SF", "MIA"], rows)
  assert.ok(two.includes("teamCompare") && two.includes("pctMarker1"))
  assert.equal(n.nflSigned(-0.061, 3), "−.061")
})

const PLAYERS = [
  { player_id: "1", player_name: "Jaxon Smith-Njigba", team: "SEA", position: "WR", games: "6", headshot_url: "",
    targets_pg: "11.7", targets_pg_adj: "12.4", rec_ypg: "114.75", rec_ypg_adj: "118.7", epa_target: "0.58", epa_target_adj: "0.58", target_share: "0.335", target_share_adj: "0.351" },
  { player_id: "2", player_name: "Drake London", team: "ATL", position: "WR", games: "6", headshot_url: "",
    targets_pg: "11.0", targets_pg_adj: "10.9", rec_ypg: "104.6", rec_ypg_adj: "100.75", epa_target: "0.20", epa_target_adj: "0.21", target_share: "0.315", target_share_adj: "0.302" },
  { player_id: "3", player_name: "Jared Goff", team: "DET", position: "QB", games: "5", int_rate: "0.0", int_rate_adj: "0.0", pass_ypg: "308" },
  { player_id: "4", player_name: "Josh Allen", team: "BUF", position: "QB", games: "6", int_rate: "0.03", int_rate_adj: "0.021", pass_ypg: "250" },
]

test("players: by position, sortable, season or vs. defenses faced with how far it moved", () => {
  const table = n.nflPlayerTableHtml(PLAYERS, "WR", "rec_ypg")
  assert.ok(text(table).indexOf("Jaxon Smith-Njigba") < text(table).indexOf("Drake London"))
  assert.ok(!text(table).includes("Goff"))
  const faced = text(n.nflPlayerTableHtml(PLAYERS, "WR", "rec_ypg", "faced"))
  assert.ok(faced.includes("118.7+4.0") && faced.includes("100.8−3.8"))
  assert.ok(faced.includes("+.58no change"))
  // INT rate: lower is better, so it sorts ascending.
  const qbs = text(n.nflPlayerTableHtml(PLAYERS, "QB", "int_rate"))
  assert.ok(qbs.indexOf("Jared Goff") < qbs.indexOf("Josh Allen"))
  const card = n.nflPlayerCardHtml(PLAYERS[0], PLAYERS)
  assert.ok(card.includes("Rec yds/g: 114.8 - better than 75% of WRs"))
  assert.ok(card.includes("Catches/g: no data yet"))
  assert.ok(text(n.nflPlayerSuggestionsHtml([PLAYERS[1]])).includes("Drake London WR · ATL"))
})
