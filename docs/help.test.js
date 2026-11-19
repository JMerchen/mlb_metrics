const test = require("node:test")
const assert = require("node:assert")
const fs = require("node:fs")
const path = require("node:path")
const { helpPopPosition } = require("./help.js")

const VIEW = { width: 390, height: 800 }

test("the box sits under the bubble, lined up a little to its left", () => {
  const spot = helpPopPosition({ left: 120, top: 100, bottom: 120 }, { width: 200, height: 150 }, VIEW)
  assert.deepEqual(spot, { left: 106, top: 128 })
})

test("the box stays inside the window", () => {
  // A bubble at the right edge: the box slides left to fit.
  assert.equal(helpPopPosition({ left: 360, top: 100, bottom: 120 }, { width: 300, height: 150 }, VIEW).left, 78)
  // A bubble at the left edge: it keeps the margin.
  assert.equal(helpPopPosition({ left: 4, top: 100, bottom: 120 }, { width: 300, height: 150 }, VIEW).left, 12)
})

test("with no room below, the box goes above the bubble - unless there is no room there either", () => {
  assert.equal(helpPopPosition({ left: 50, top: 700, bottom: 720 }, { width: 200, height: 150 }, VIEW).top, 542)
  assert.equal(helpPopPosition({ left: 50, top: 100, bottom: 120 }, { width: 200, height: 750 }, VIEW).top, 128)
})

test("every page but How it works loads the script in its head, and hides the paragraphs only when it runs", () => {
  const pages = fs.readdirSync(__dirname).filter(f => f.endsWith(".html"))
  assert.ok(pages.length >= 10)
  for (const page of pages) {
    const head = fs.readFileSync(path.join(__dirname, page), "utf8").split("</head>")[0]
    assert.equal(head.includes('<script src="./help.js"></script>'), page !== "methodology.html", page)
  }
  const css = fs.readFileSync(path.join(__dirname, "style.css"), "utf8")
  assert.match(css, /html\.helpOn p\.sectionIntro/)
})
