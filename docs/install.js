// The site as an installable app: registers the service worker (sw.js, for
// offline use) and adds an "Install app" button to the header. Chrome, Edge
// and Android install straight from the button; iPhone and iPad have no
// install prompt, so the button explains Share → Add to Home Screen. The
// button hides once the site is running as the installed app.

// "installed" (running as the app), "ios" (Safari's Add to Home Screen) or
// "prompt" (wait for the browser's install prompt).
function installMode(win){
const standalone = (win.matchMedia && win.matchMedia("(display-mode: standalone)").matches) || win.navigator.standalone === true
if(standalone){ return "installed" }
const ua = win.navigator.userAgent || ""
const ios = /iPad|iPhone|iPod/.test(ua) || (/Macintosh/.test(ua) && win.navigator.maxTouchPoints > 1)
return ios ? "ios" : "prompt"
}

function installTipHtml(){
return `<div class="installTip" role="dialog" aria-label="Install the app">
<b>Install on iPhone or iPad</b>
<ol><li>Tap the <b>Share</b> button <span aria-hidden="true">(the square with an arrow)</span>.</li>
<li>Choose <b>Add to Home Screen</b>, then <b>Add</b>.</li></ol>
<p>It opens full-screen from its own icon, and the latest picks stay readable offline.</p>
<button class="tabButton" data-install-close>Got it</button>
</div>`
}

function setUpInstall(win){
const doc = win.document
if("serviceWorker" in win.navigator && /^https?:$/.test(win.location.protocol)){
win.navigator.serviceWorker.register("./sw.js").catch(e => console.log("no offline support", e))
}
const mode = installMode(win)
const tabs = doc.querySelector(".siteHeader .pageTabs")
if(mode === "installed" || !tabs){ return }
const button = doc.createElement("button")
button.className = "tabButton installButton"
button.type = "button"
button.textContent = "Install app"
button.hidden = mode !== "ios"
tabs.appendChild(button)
let deferred = null
win.addEventListener("beforeinstallprompt", e => {
e.preventDefault()
deferred = e
button.hidden = false
})
win.addEventListener("appinstalled", () => { button.hidden = true })
button.addEventListener("click", async () => {
if(deferred){
deferred.prompt()
const choice = await deferred.userChoice
if(choice && choice.outcome === "accepted"){ button.hidden = true }
deferred = null
return
}
if(doc.querySelector(".installTip")){ return }
const holder = doc.createElement("div")
holder.innerHTML = installTipHtml()
const tip = holder.firstElementChild
doc.querySelector(".siteHeader").after(tip)
tip.querySelector("[data-install-close]").addEventListener("click", () => tip.remove())
})
}

if(typeof window !== "undefined" && window.document){
setUpInstall(window)
}

if(typeof module !== "undefined" && module.exports){
module.exports = { installMode, installTipHtml }
}
