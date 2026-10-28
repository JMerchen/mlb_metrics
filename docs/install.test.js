// node --test docs/install.test.js
const test = require("node:test")
const assert = require("node:assert/strict")
const { installMode, installTipHtml } = require("./install.js")
const { swStrategy, swCacheKey, SW_PRECACHE } = require("./sw.js")
const fs = require("node:fs")
const path = require("node:path")

const win = ({ ua = "Mozilla/5.0 (X11; Linux x86_64) Chrome/124", standalone = false, iosStandalone, touch = 0 } = {}) => ({
  matchMedia: q => ({ matches: standalone && q.includes("standalone") }),
  navigator: { userAgent: ua, standalone: iosStandalone, maxTouchPoints: touch },
})

test("install mode: installed app, iPhone/iPad Safari, or wait for the browser's prompt", () => {
  assert.equal(installMode(win()), "prompt")
  assert.equal(installMode(win({ standalone: true })), "installed")
  assert.equal(installMode(win({ ua: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X)" })), "ios")
  assert.equal(installMode(win({ ua: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X)", iosStandalone: true })), "installed")
  // iPadOS reports a Mac, with touch.
  assert.equal(installMode(win({ ua: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)", touch: 5 })), "ios")
  assert.equal(installMode(win({ ua: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)" })), "prompt")
  assert.match(installTipHtml(), /Add to Home Screen/)
})

test("offline: site files network-first, icons and CDN libraries cache-first, others left alone", () => {
  const origin = "https://jmerchen.github.io"
  assert.equal(swStrategy(`${origin}/mlb_metrics/data/wave.csv?t=1`, origin), "network-first")
  assert.equal(swStrategy(`${origin}/mlb_metrics/nfl.html`, origin), "network-first")
  assert.equal(swStrategy(`${origin}/mlb_metrics/icons/icon-192.png`, origin), "cache-first")
  assert.equal(swStrategy("https://cdn.jsdelivr.net/npm/papaparse@5.4.1/papaparse.min.js", origin), "cache-first")
  assert.equal(swStrategy("https://img.mlbstatic.com/x.png", origin), "skip")
  assert.equal(swCacheKey(`${origin}/mlb_metrics/data/wave.csv?t=123`), `${origin}/mlb_metrics/data/wave.csv`)
})

test("every precached file and manifest icon exists; every page links the manifest and install.js", () => {
  const docs = __dirname
  SW_PRECACHE.filter(f => f !== "./").forEach(f => assert.ok(fs.existsSync(path.join(docs, f)), f))
  const manifest = JSON.parse(fs.readFileSync(path.join(docs, "manifest.webmanifest"), "utf8"))
  assert.equal(manifest.display, "standalone")
  manifest.icons.forEach(i => assert.ok(fs.existsSync(path.join(docs, i.src)), i.src))
  fs.readdirSync(docs).filter(f => f.endsWith(".html")).forEach(f => {
    const html = fs.readFileSync(path.join(docs, f), "utf8")
    assert.match(html, /rel="manifest" href="\.\/manifest\.webmanifest"/, f)
    assert.match(html, /name="viewport"/, f)
    assert.match(html, /<script src="\.\/install\.js"><\/script>/, f)
  })
})
