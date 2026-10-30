// Service worker for the site installed as an app (install.js registers
// it; manifest.webmanifest describes the app). Online, pages, scripts and
// data always come from the network, so the app is never staler than the
// site; each response is kept, and offline the last copy is served.
// Icons and the pinned-version CDN libraries (Papa Parse, ExcelJS) come
// from the cache first. Bump SW_VERSION to drop everything kept.

const SW_VERSION = "v2"
const SW_CACHE = `dg-metrics-${SW_VERSION}`
const SW_PRECACHE = ["./", "./index.html", "./style.css", "./install.js", "./manifest.webmanifest", "./icons/icon-192.png"]

// "network-first", "cache-first" or "skip" (left to the browser).
function swStrategy(url, origin){
const u = new URL(url)
if(u.origin !== origin){ return u.hostname === "cdn.jsdelivr.net" ? "cache-first" : "skip" }
return u.pathname.includes("/icons/") ? "cache-first" : "network-first"
}

// Pages fetch data with a cache-busting ?t=..., so keep one copy per path.
function swCacheKey(url){
const u = new URL(url)
if(u.origin === (typeof self !== "undefined" && self.location ? self.location.origin : u.origin)){ u.search = "" }
return u.toString()
}

if(typeof self !== "undefined" && typeof self.addEventListener === "function" && typeof window === "undefined"){
self.addEventListener("install", event => {
event.waitUntil(caches.open(SW_CACHE).then(cache => cache.addAll(SW_PRECACHE)).then(() => self.skipWaiting()))
})

self.addEventListener("activate", event => {
event.waitUntil(caches.keys()
.then(keys => Promise.all(keys.filter(k => k.startsWith("dg-metrics-") && k !== SW_CACHE).map(k => caches.delete(k))))
.then(() => self.clients.claim()))
})

self.addEventListener("fetch", event => {
const request = event.request
if(request.method !== "GET"){ return }
const strategy = swStrategy(request.url, self.location.origin)
if(strategy === "skip"){ return }
const key = swCacheKey(request.url)
const keep = response => {
if(response && (response.ok || response.type === "opaque")){
const copy = response.clone()
caches.open(SW_CACHE).then(cache => cache.put(key, copy))
}
return response
}
if(strategy === "cache-first"){
event.respondWith(caches.match(key).then(hit => hit || fetch(request).then(keep)))
return
}
event.respondWith(fetch(request).then(keep).catch(() => caches.match(key).then(hit =>
hit || (request.mode === "navigate" ? caches.match("./index.html") : Response.error()))))
})
}

if(typeof module !== "undefined" && module.exports){
module.exports = { swStrategy, swCacheKey, SW_PRECACHE }
}
