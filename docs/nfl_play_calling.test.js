const test = require("node:test")
const assert = require("node:assert")
const { PC_LAYERS, pcColor, pcParse, pcSteps, pcTree, pcByLayer, pcPhrase, pcTip, pcFind, pcStandouts, pcLayout, pcColumnTitle, playCallingTreeHtml } = require("./nfl_play_calling.js")

const text = html => html.replace(/<[^>]+>/g, " ").replace(/&amp;/g, "&").replace(/\s+/g, " ").trim()
// Kansas City: 10 drives. Six open run, run; five of those go on run, run, pass.
const KC = { "R+R+P+P-": 3, "R+R-P+R+": 2, "R-R+R-": 1, "R-P+": 1, "P+P+P+P+": 2, "P-": 1 }
const LEAGUE = { "R+R+P+P-": 10, "R-P+R+P+": 30, "P+P+P+P+": 30, "P-R+P-R-": 30 }
const DATA = { season: 2026, through_week: 10, depth: 4, league: LEAGUE, teams: { KC: { offense: KC, defense: { "P-R-": 4, "R+": 1 } }, BUF: { offense: {}, defense: {} } } }
const NAMES = code => ({ KC: "Chiefs", BUF: "Bills" }[code] || code)

test("a sequence is read as calls and whether each worked", () => {
  assert.deepStrictEqual(pcParse("P-R+"), [{ call: "P", ok: false }, { call: "R", ok: true }])
  assert.deepStrictEqual(pcSteps("P-R+R+P-", "plays"), ["P", "R", "R", "P"])
  // With results the layers alternate, and stop at the third call.
  assert.deepStrictEqual(pcSteps("P-R+R+P-", "results"), ["P", "-", "R", "+", "R"])
  assert.deepStrictEqual(pcSteps("P-", "results"), ["P", "-"])
  assert.deepStrictEqual(pcSteps("", "plays"), [])
})

test("the tree always has every branch, in a fixed order, and counts the drives through each", () => {
  const tree = pcTree(KC, "plays")
  assert.strictEqual(tree.drives, 10)
  const layers = pcByLayer(tree)
  assert.deepStrictEqual(layers.map(l => l.length), [2, 4, 8, 16])
  assert.deepStrictEqual(layers[1].map(n => n.path), ["RR", "RP", "PR", "PP"])        // run above pass, everywhere
  assert.deepStrictEqual(layers[0].map(n => n.drives), [7, 3])
  assert.deepStrictEqual(layers[1].map(n => n.drives), [6, 1, 0, 2])
  assert.strictEqual(pcFind(tree, "RRP").drives, 5)
  assert.strictEqual(pcFind(tree, "RRPP").drives, 3)
  assert.strictEqual(pcFind(tree, "PRRR").drives, 0)                                   // never opened with it: still there
  // How often each node's own play worked.
  assert.strictEqual(pcFind(tree, "R").worked, 5)
  assert.strictEqual(pcFind(tree, "RR").worked, 4)
  assert.strictEqual(pcFind(tree, "RRPP").worked, 0)
})

test("a box's color share is its drives over the busiest box of its column", () => {
  const tree = pcTree(KC, "plays")
  assert.strictEqual(pcFind(tree, "R").share, 1)
  assert.ok(Math.abs(pcFind(tree, "P").share - 3 / 7) < 1e-9)
  assert.strictEqual(pcFind(tree, "RR").share, 1)
  assert.ok(Math.abs(pcFind(tree, "PP").share - 2 / 6) < 1e-9)
  assert.strictEqual(pcFind(tree, "PR").share, 0)
  assert.strictEqual(pcTree({}, "plays").children[0].share, 0)
})

test("with results, a layer after each play splits it by whether it worked", () => {
  const tree = pcTree(KC, "results")
  assert.deepStrictEqual(pcByLayer(tree).map(l => l.length), [2, 4, 8, 16, 32])
  assert.deepStrictEqual(PC_LAYERS.results, ["call", "result", "call", "result", "call"])
  assert.strictEqual(pcFind(tree, "R+").drives, 5)
  assert.strictEqual(pcFind(tree, "R-").drives, 2)
  assert.strictEqual(pcFind(tree, "R+R").drives, 5)
  assert.strictEqual(pcFind(tree, "R+R-P").drives, 2)
  assert.strictEqual(pcFind(tree, "R+R").play, 2)
  assert.strictEqual(pcFind(tree, "R+R-").play, 2)
  assert.strictEqual(pcFind(tree, "R+R-P").play, 3)
})

test("the scale runs pale to deep, and says when text on it must be light", () => {
  const none = pcColor(0), half = pcColor(0.5), most = pcColor(1)
  assert.strictEqual(none.fill, "#e6f2e6")
  assert.strictEqual(most.fill, "#0f2b57")
  assert.ok(!none.light && most.light)
  assert.match(half.fill, /^#[0-9a-f]{6}$/)
  assert.deepStrictEqual(pcColor(7), most)                      // clamped
})

test("a sequence in words, with and without how each play went", () => {
  assert.strictEqual(pcPhrase("RPP", "plays"), "run, then pass, then pass")
  assert.strictEqual(pcPhrase("R+P-R", "results"), "a run that worked, then a pass that didn't work, then a run")
  assert.strictEqual(pcPhrase("P-", "results"), "a pass that didn't work")
})

test("the hover gives the counts, the league's rate and how often the play worked", () => {
  const tree = pcTree(KC, "plays"), league = pcTree(LEAGUE, "plays")
  const tip = pcTip(pcFind(tree, "RR"), tree, league, "plays", "Chiefs drives")
  assert.strictEqual(tip, "Run, then run: 6 of 10 Chiefs drives opened this way. Across the league, about 10 of every 100 do. The 2nd play worked on 4 of those 6.")
  const results = pcTree(KC, "results")
  assert.strictEqual(pcTip(pcFind(results, "R+"), results, pcTree(LEAGUE, "results"), "results", "Chiefs drives"),
    "A run that worked: 5 of 10 Chiefs drives opened this way. Across the league, about 10 of every 100 do.")
})

test("the standouts are the commonest full sequence and the one furthest above the league", () => {
  const tree = pcTree(KC, "plays"), league = pcTree(LEAGUE, "plays")
  const { typical, signature } = pcStandouts(tree, league, "plays", 4)
  assert.strictEqual(typical.path, "RRPP")
  assert.strictEqual(signature.node.path, "RR")                 // 6 of 10 against the league's 1 in 10
  assert.ok(Math.abs(signature.usual - 1) < 1e-9)
  // Nothing has enough drives behind it at the default bar.
  assert.strictEqual(pcStandouts(tree, league, "plays").signature, null)
  assert.strictEqual(pcStandouts(pcTree({}, "plays"), league, "plays").typical, null)
})

test("the layout puts leaves on even rows, parents between their children, columns left to right", () => {
  const tree = pcTree(KC, "plays")
  const lay = pcLayout(tree, "plays", 720)
  assert.strictEqual(lay.nodes.length, 30)
  const at = path => lay.nodes.find(b => b.node.path === path)
  assert.ok(Math.abs(at("R").y - (at("RR").y + at("RP").y) / 2) < 1e-9)
  assert.ok(at("R").y < at("P").y && at("RRRR").y < at("RRRP").y)
  assert.ok(lay.root.x < at("R").x && at("R").x < at("RR").x && at("RRR").x < at("RRRR").x)
  assert.ok(at("RRRR").x + at("RRRR").w / 2 <= 720 && at("PPPP").y + at("PPPP").h / 2 <= lay.height)
  // Neighbouring leaves do not touch.
  assert.ok(at("RRRP").y - at("RRRR").y >= at("RRRR").h + 2)
  const results = pcLayout(pcTree(KC, "results"), "results", 340)
  assert.strictEqual(results.nodes.length, 62)
  const leaves = results.nodes.filter(b => b.node.layer === 4)
  assert.ok(leaves[1].y - leaves[0].y >= leaves[0].h + 2)
  assert.ok(Math.max(...results.nodes.map(b => b.x + b.w / 2)) <= 340)
})

test("column headings name the play, and shorten when the columns are close", () => {
  assert.deepStrictEqual([0, 1, 2, 3].map(i => pcColumnTitle("plays", i)), ["1st play", "2nd play", "3rd play", "4th play"])
  assert.deepStrictEqual([0, 1, 2, 3, 4].map(i => pcColumnTitle("results", i)), ["1st play", "Worked?", "2nd play", "Worked?", "3rd play"])
  assert.deepStrictEqual([0, 1, 2].map(i => pcColumnTitle("results", i, true)), ["1st", "", "2nd"])
})

test("the card names the team and the side, says the standouts in words and shows no percentages", () => {
  const html = playCallingTreeHtml(DATA, "KC", { names: NAMES, width: 720 })
  const t = text(html)
  assert.ok(t.includes("Chiefs with the ball · 10 drives through week 10"))
  assert.ok(t.includes("Most typical start: run, then run, then pass, then pass (3 of 10 drives)."))
  assert.ok(t.includes("rarer") && t.includes("most typical"))
  assert.ok(!/\d%/.test(t))
  assert.strictEqual((html.match(/class="pcNode /g) || []).length, 31)     // 30 boxes and the start
  assert.ok(html.includes('viewBox="0 0 720 '))
  // Boxes nobody has opened with are drawn empty.
  assert.ok(html.includes("pcNone"))
  const results = playCallingTreeHtml(DATA, "KC", { names: NAMES, mode: "results", width: 340 })
  assert.ok(text(results).includes("worked") && results.includes("<circle"))
  assert.ok(results.includes('viewBox="0 0 340 '))
})

test("the defense's tree is the drives it faced; a team with no drives gets no card", () => {
  const html = playCallingTreeHtml(DATA, "KC", { names: NAMES, side: "defense" })
  const t = text(html)
  assert.ok(t.includes("Chiefs on defense: what offenses call against them · 5 drives through week 10"))
  assert.ok(html.includes("drives against the Chiefs opened this way"))
  assert.strictEqual(playCallingTreeHtml(DATA, "BUF", { names: NAMES }), "")
  assert.strictEqual(playCallingTreeHtml(DATA, "NYJ", { names: NAMES }), "")
  assert.strictEqual(playCallingTreeHtml(null, "KC"), "")
})
