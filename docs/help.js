// Explanations behind a "?" (user request, 2026-11-19).
//
// The paragraph under a section heading that says what the section is
// (class="sectionIntro", or class="helpText" on any other paragraph) no
// longer sits on the page. A small "?" bubble goes at the end of the
// heading before it; clicking the bubble shows the paragraph in a box next
// to it, and the box goes away when the reader scrolls, clicks anywhere
// else, or presses Escape. A paragraph with no heading before it gets the
// bubble on a line of its own, labeled by its data-help-label ("What is
// this?" if it has none). Paragraphs in a row share one bubble.
//
// The paragraphs stay in the page as written - only hidden (style.css,
// html.helpOn) - so the scripts that fill them in keep working, the box
// always shows their current text, and without this script they are simply
// shown as before. "How it works" does not load this: that page is the
// explanation.
//
// Loaded in every other page's <head>, so the paragraphs never flash.

// Where the box goes: under the bubble, kept inside the window; above it
// when there is no room below and there is above.
function helpPopPosition(anchor, pop, view, gap = 8, pad = 12){
const left = Math.round(Math.max(pad, Math.min(anchor.left - 14, view.width - pop.width - pad)))
const below = anchor.bottom + gap
const above = anchor.top - gap - pop.height
const fitsBelow = below + pop.height <= view.height - pad
return {left, top: Math.round(fitsBelow || above < pad ? below : above)}
}

if(typeof document !== "undefined"){ (function(){
const root = document.documentElement
root.classList.add("helpOn")
const SOURCES = "p.sectionIntro, p.helpText"
const bubbles = new Map()
let nextId = 1
let pop = null
let openFor = null

function bubble(id, label){
const b = document.createElement("button")
b.type = "button"
b.className = "helpBubble"
b.dataset.helpFor = id
b.setAttribute("aria-expanded", "false")
b.setAttribute("aria-label", label)
b.textContent = "?"
return b
}

function attach(p){
if(p.dataset.help && bubbles.has(p.dataset.help) && bubbles.get(p.dataset.help).isConnected){ return }
const prev = p.previousElementSibling
// The next paragraph of an explanation already behind a bubble.
if(prev && prev.matches(SOURCES) && prev.dataset.help){
p.dataset.help = prev.dataset.help
return
}
const id = p.dataset.help || String(nextId++)
p.dataset.help = id
let b
if(prev && prev.matches("h1, h2, h3, h4, h5, summary")){
b = bubble(id, `Explain: ${prev.textContent.trim().replace(/\s+/g, " ")}`)
prev.appendChild(b)
}else{
const text = p.dataset.helpLabel || "What is this?"
const line = document.createElement("div")
line.className = "helpLine"
b = bubble(id, text)
b.classList.add("helpLone")
b.innerHTML = `<span class="helpMark" aria-hidden="true">?</span>${text.replace(/&/g, "&amp;").replace(/</g, "&lt;")}`
line.appendChild(b)
p.parentNode.insertBefore(line, p)
}
if(p.classList.contains("deep-only")){ (b.closest(".helpLine") || b).classList.add("deep-only") }
bubbles.set(id, b)
}

function scan(){
document.querySelectorAll(SOURCES).forEach(p => { if(!p.closest(".helpPop")){ attach(p) } })
}

function close(){
if(!pop || pop.hidden){ return }
pop.hidden = true
if(openFor){ openFor.setAttribute("aria-expanded", "false") }
openFor = null
}

function open(b){
const id = b.dataset.helpFor
const parts = [...document.querySelectorAll(SOURCES)].filter(p => p.dataset.help === id)
if(!parts.length){ return }
if(!pop){
pop = document.createElement("div")
pop.className = "helpPop"
pop.setAttribute("role", "note")
pop.id = "helpPop"
document.body.appendChild(pop)
}
pop.innerHTML = parts.map(p => `<p>${p.innerHTML}</p>`).join("")
pop.hidden = false
pop.style.left = "0px"
pop.style.top = "0px"
const view = {width: root.clientWidth, height: window.innerHeight}
const spot = helpPopPosition(b.getBoundingClientRect(), {width: pop.offsetWidth, height: pop.offsetHeight}, view)
pop.style.left = `${spot.left}px`
pop.style.top = `${spot.top}px`
openFor = b
b.setAttribute("aria-expanded", "true")
b.setAttribute("aria-controls", "helpPop")
}

document.addEventListener("click", event => {
const b = event.target.closest ? event.target.closest(".helpBubble") : null
if(b){
// A bubble in a fold's heading must not open or shut the fold.
event.preventDefault()
event.stopPropagation()
const same = openFor === b
close()
if(!same){ open(b) }
return
}
if(!(event.target.closest && event.target.closest(".helpPop"))){ close() }
}, true)
window.addEventListener("scroll", event => { if(!(pop && event.target instanceof Node && pop.contains(event.target))){ close() } }, {capture: true, passive: true})
window.addEventListener("resize", close)
document.addEventListener("keydown", event => { if(event.key === "Escape"){ close() } })

function start(){
scan()
// Sections drawn later, and headings a script rewrites (which drops
// their bubble), get theirs on the next frame.
let queued = false
new MutationObserver(() => {
if(queued){ return }
queued = true
requestAnimationFrame(() => { queued = false; scan() })
}).observe(document.body, {childList: true, subtree: true})
}
if(document.readyState === "loading"){ document.addEventListener("DOMContentLoaded", start) }else{ start() }
})() }

if (typeof module !== "undefined" && module.exports) {
module.exports = { helpPopPosition }
}
