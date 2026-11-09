// Site-wide Basic / Deep dive switch (user request, 2026-11-09).
//
// Basic (the default) is a light, clean site with the parts that give an
// answer without digging - picks, records, results, schedules, rankings.
// Deep dive shows everything (What if?, simulations, the big stat tables,
// unit ratings, Stuff+, floors, Age Curves, Payroll) in a soft dark scheme
// meant for longer reading. Anything marked class="deep-only" is hidden in
// Basic (style.css). The choice is remembered per browser.
//
// Loaded in every page's <head>, before the stylesheet paints, so a Deep
// dive visitor never sees a flash of the light page.
(function(){
const KEY = "srMode"
const root = document.documentElement
const THEME_COLOR = {basic: "#f6f7f9", deep: "#15181d"}

function read(){
try{
const v = window.localStorage.getItem(KEY)
return v === "deep" || v === "basic" ? v : "basic"
}catch(e){ return "basic" }
}

function apply(mode){
root.dataset.mode = mode
root.dataset.theme = mode === "deep" ? "dark" : "light"
const meta = document.querySelector('meta[name="theme-color"]')
if(meta){ meta.setAttribute("content", THEME_COLOR[mode]) }
document.querySelectorAll("[data-sr-mode]").forEach(b=>{
const on = b.getAttribute("data-sr-mode") === mode
b.classList.toggle("active", on)
b.setAttribute("aria-pressed", on ? "true" : "false")
})
}

// A tab hidden by Basic can't stay the open one: open its group's first
// tab that's still showing.
function leaveHiddenTabs(){
document.querySelectorAll(".tabGroup").forEach(group=>{
const active = group.querySelector(".tabButton.active.deep-only")
if(!active){ return }
const visible = [...group.querySelectorAll(".tabButton")].find(b=>!b.classList.contains("deep-only"))
if(visible){ visible.click() }
})
}

function set(mode){
try{ window.localStorage.setItem(KEY, mode) }catch(e){}
apply(mode)
if(mode === "basic"){ leaveHiddenTabs() }
window.dispatchEvent(new CustomEvent("srmodechange", {detail: {mode}}))
}

function addSwitch(){
const header = document.querySelector(".siteHeader")
if(!header || header.querySelector(".modeSwitch")){ return }
const el = document.createElement("div")
el.className = "modeSwitch"
el.setAttribute("role", "group")
el.setAttribute("aria-label", "How much to show")
el.innerHTML = `<button type="button" data-sr-mode="basic" title="The essentials, in a light theme">Basic</button>`
+ `<button type="button" data-sr-mode="deep" title="Everything, in a dark theme for longer reading">Deep dive</button>`
el.querySelectorAll("button").forEach(b=>{ b.onclick = ()=>set(b.getAttribute("data-sr-mode")) })
header.appendChild(el)
apply(root.dataset.mode || read())
}

apply(read())
if(document.readyState === "loading"){
document.addEventListener("DOMContentLoaded", ()=>{ addSwitch(); if(read() === "basic"){ leaveHiddenTabs() } })
}else{
addSwitch()
}
window.srMode = {get: read, set}
})()
