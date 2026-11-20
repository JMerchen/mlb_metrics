const test = require("node:test")
const assert = require("node:assert")
const { evTimes, evMonth, evVerdict, evidenceTableHtml, evidenceChartSvg, evidenceHtml } = require("./evidence_render.js")

const text = html => html.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim()
const SPREAD = {
  key: "nfl_spread_replay", label: "Spread bets, every one since 2002", kind: "replay", note: "Chosen with these in view.",
  bets: 868, won: 487, lost: 362, win_rate: 0.5736, units: 80.73, wealth: 20.644, peak: 64.803, peak_date: "2021-11-21",
  rate_low: 0.524, rate_high: 0.618, fade: 63.2, fade_peak: 207, fade_tripped: null,
  path: [{ at: "2002", wealth: 0.4 }, { at: "2014", wealth: 1.7 }, { at: "2021", wealth: 39.3 }, { at: "2026", wealth: 20.6 }],
}
const WINNERS = { key: "nfl_winner_replay", label: "Our winner in every game", kind: "replay", note: "Picks, not bets.", bets: 5445, won: 3594, lost: 1851, win_rate: 0.66, units: -88.7, wealth: 0.016, peak: 2.8, peak_date: "2006-10-01", path: [] }
const LIVE = { key: "nfl_total_live", label: "Over/under leans posted here", kind: "live", note: "The clean test.", bets: 7, won: 5, lost: 2, win_rate: 0.714, units: 2.5, wealth: 1.36, peak: 1.65, rate_low: 0.194, rate_high: 0.982, fade: 5.1, fade_tripped: null, path: [] }
const DATA = { bar: 20, alarm: 1000, confidence: 0.95, break_even: 0.5238, series: [SPREAD, WINNERS, LIVE] }

test("a bankroll is written as a multiple, to the precision that matters", () => {
  assert.equal(evTimes(20.644), "×20.6")
  assert.equal(evTimes(1.363), "×1.36")
  assert.equal(evTimes(0.016), "×0.016")
  assert.equal(evTimes(1482.3), "×1,482")
  assert.equal(evMonth("2021-11-21"), "Nov 2021")
})

test("the verdict reads the bankroll against the bar, and says when there are too few bets", () => {
  assert.equal(evVerdict(SPREAD, 20), "At the bar")
  assert.equal(evVerdict({ ...SPREAD, wealth: 14.4 }, 20), "Reached the bar, since slipped")
  assert.equal(evVerdict({ ...SPREAD, wealth: 150 }, 20), "Strong")
  assert.equal(evVerdict({ ...SPREAD, wealth: 2.75, peak: 3 }, 20), "Some, short of the bar")
  assert.equal(evVerdict(WINNERS, 20), "Evidence of no edge")
  assert.equal(evVerdict(LIVE, 20), "Too few bets to say")
})

test("the table gives each series its record, bankroll with its peak, the rates still possible and the alarm", () => {
  const html = evidenceTableHtml(DATA)
  const t = text(html)
  assert.match(t, /Spread bets, every one since 2002 Chosen with these in view\. 868 487–362 57\.4% \+80\.7 ×20\.6 peak ×64\.8, Nov 2021 At the bar 52\.4% to 61\.8% Quiet 63 of 1,000/)
  // Bets at varying prices have no single win rate to bound.
  assert.match(t, /Our winner in every game .* -88\.7 ×0\.016 .*Evidence of no edge prices vary/)
  assert.match(html, /class="ev-live"/)
  assert.match(t, /The bar is ×20\. At −110 a bet breaks even at 52\.4%\./)
  assert.match(text(evidenceTableHtml({ ...DATA, series: [{ ...SPREAD, fade_tripped: "2025-12-07" }] })), /Tripped Dec 2025/)
  assert.equal(evidenceTableHtml({ series: [] }), "")
})

test("the chart is a log-scale line with the bar, the start, the peak and the latest value labeled", () => {
  const svg = evidenceChartSvg(SPREAD, 20, 720)
  assert.match(svg, /aria-label="Spread bets, every one since 2002: the evidence bankroll over time, now ×20\.6"/)
  assert.equal((svg.match(/data-tip=/g) || []).length, 4)
  assert.match(svg, /data-tip="2021: ×39\.3"/)
  const t = text(svg)
  assert.match(t, /the bar, ×20/)
  assert.match(t, /start, ×1/)
  assert.match(t, /2021: ×39\.3/)
  assert.match(t, /now ×20\.6/)
  // Gridlines a factor of ten apart, covering below ×1 and above the bar.
  assert.match(t, /×0\.1 ×1 ×10 ×100/)
  assert.equal(evidenceChartSvg({ path: [{ at: "x", wealth: 1 }] }, 20), "")
})

test("the block holds the table and, for Deep dive, the spread rule's chart", () => {
  const html = evidenceHtml(DATA)
  assert.match(html, /<table class="trackRecord evidenceTable">/)
  assert.match(html, /<div class="vizCard deep-only">/)
  assert.equal(evidenceHtml({ series: [] }), "")
})
