// Payroll value views: the MLB Payroll page (payroll.html) and the NFL
// page's Payroll tab (started by nfl.js). payrollView(sport, prefix, rootId)
// draws payroll_render.js's sections into the root, loads the sport's CSVs
// and wires the DOM; element ids are prefix + a name, so both views can
// share one page (the artifact). The workbook download is built in the
// browser with ExcelJS (loaded on first click).

async function loadCSV(path){
const response = await fetch(`${path}?t=${Date.now()}`, {cache:"no-store"})
if(!response.ok){ throw new Error(`Failed to load ${path}: ${response.status}`) }
const text = await response.text()
return Papa.parse(text, {header:true, skipEmptyLines:true}).data
}

const EXCELJS_URL = "https://cdn.jsdelivr.net/npm/exceljs@4.4.0/dist/exceljs.min.js"

async function payOptionalCSV(path){
try{ return await loadCSV(path) }catch(e){ console.log(`no ${path} yet`, e); return [] }
}

function payColumnLetter(n){
let s = ""
for(n += 1; n > 0; n = Math.floor((n - 1) / 26)){ s = String.fromCharCode(65 + (n - 1) % 26) + s }
return s
}

function payLoadScript(url){
return new Promise((resolve, reject) => {
const script = document.createElement("script")
script.src = url
script.onload = resolve
script.onerror = () => reject(new Error(`Couldn't load ${url}`))
document.head.appendChild(script)
})
}

function payScaleRule(higherIsGreen){
const [low, high] = higherIsGreen ? ["FFF8696B", "FF63BE7B"] : ["FF63BE7B", "FFF8696B"]
return { type: "colorScale", cfvo: [{ type: "min" }, { type: "percentile", value: 50 }, { type: "max" }],
color: [{ argb: low }, { argb: "FFFFEB84" }, { argb: high }] }
}

async function payWriteWorkbook(sheets, fileName){
if(typeof ExcelJS === "undefined"){ await payLoadScript(EXCELJS_URL) }
const wb = new ExcelJS.Workbook()
sheets.forEach(spec => {
const ws = wb.addWorksheet(spec.name.slice(0, 31), { views: [{ state: "frozen", xSplit: 1, ySplit: 1 }] })
ws.addRow(spec.columns.map(c => c[0])).font = { bold: true }
spec.rows.forEach(row => ws.addRow(row))
spec.columns.forEach(([header, , fmt], i) => {
const col = ws.getColumn(i + 1)
col.width = Math.max(8, Math.min(header.length + 2, 28))
if(fmt){ col.eachCell((cell, n) => { if(n > 1){ cell.numFmt = fmt } }) }
})
const last = spec.rows.length + 1
if(last < 2){ return }
spec.scales.forEach(s => {
const letter = payColumnLetter(s.column)
ws.addConditionalFormatting({ ref: `${letter}2:${letter}${last}`, rules: [payScaleRule(s.higherIsGreen)] })
})
if(spec.gridScale){
ws.addConditionalFormatting({ ref: `B2:${payColumnLetter(spec.columns.length - 1)}${last}`, rules: [payScaleRule(spec.gridScale.higherIsGreen)] })
}
})
const blob = new Blob([await wb.xlsx.writeBuffer()], { type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" })
const link = document.createElement("a")
link.href = URL.createObjectURL(blob)
link.download = fileName
document.body.appendChild(link)
link.click()
link.remove()
setTimeout(() => URL.revokeObjectURL(link.href), 1000)
}

function payrollView(S, P, rootId){
const root = document.getElementById(rootId)
if(!root){ return }
root.innerHTML = payViewHtml(S, P)
const $ = name => document.getElementById(P + name)
const loaded = { seasons: [], franchises: [], league: [] }
const state = { measure: S.measures ? S.measures[0][0] : null, season: null, team: null, teamSeason: null, sortKey: "pe_index", gridKey: "team_pe" }
// The rows for the chosen measure (NFL: cash or cap); MLB has one.
const rows = key => S.measures ? loaded[key].filter(r => r.measure === state.measure) : loaded[key]
const width = name => { const w = $(name).clientWidth; return w > 100 ? w - 32 : 720 }

function renderSeason(){
const seasonRows = paySeasonRows(rows("seasons"), state.season)
const status = seasonRows[0] && seasonRows[0].status
const label = status === "forecast" ? " (preseason forecast)" : status === "in progress" ? " (in progress)" : ""
$("Hero").innerHTML = payHeroHtml(seasonRows, rows("league"), state.season, payModelMiss(rows("seasons"), S), S)
const scatter = payScatterSvg(seasonRows, state.team, width("Scatter"), S)
$("Scatter").innerHTML = scatter ? `<div class="vizHead"><b>${state.season}${label}: wins against spending</b>
<span class="vizKey"><i class="payKeyGood"></i>More value <i class="payKeyMid"></i>Average <i class="payKeyBad"></i>Less value</span>
<span class="vizSub">Each dot is a team. Further right spent more than the league average; higher won more. Teams above the curve got more wins per dollar than the league did.</span></div>${scatter}`
: `<div class="vizHead"><b>${state.season}${label}</b><span class="vizSub">No games played yet - the table shows expected wins from spending alone.</span></div>`
$("SeasonTitle").textContent = `${state.season} season${label}`
$("Table").innerHTML = paySeasonTableHtml(seasonRows, state.sortKey, state.team, S)
}

function renderTeam(){
if(!state.team){ return }
const own = rows("seasons").filter(r => r.franchise === state.team)
if(!own.some(r => Number(r.season) === state.teamSeason)){ state.teamSeason = Number(state.season) }
const pick = own.find(r => Number(r.season) === state.teamSeason) || own[own.length - 1]
$("TeamPick").value = state.team
$("TeamChart").innerHTML = `<div class="vizHead"><b>${mlbEsc(S.name(state.team))}: ${S.perLabel}, actual and expected</b>
<span class="vizKey"><i class="payKeyActual"></i>Wins <i class="payKeyExpected"></i>Expected from spending</span>
<span class="vizSub">Click a season to see the 10 seasons its expected wins came from.</span></div>${payTeamChartSvg(rows("seasons"), state.team, width("TeamChart"), S)}`
$("Neighbors").innerHTML = payNeighborsHtml(rows("seasons"), pick, S)
}

function selectTeam(team, season = null){
state.team = team
state.teamSeason = season === null ? Number(state.season) : season
renderSeason()
renderTeam()
}

function renderGrid(){
$("GridTabs").innerHTML = payGridTabsHtml(state.gridKey)
$("Grid").innerHTML = payGridHtml(rows("seasons"), state.gridKey, S)
}

function renderAll(){
$("Franchises").innerHTML = payFranchiseTableHtml(rows("franchises"), S)
$("League").innerHTML = payLeagueChartSvg(rows("league"), width("League"), S)
renderGrid()
renderSeason()
renderTeam()
}

async function download(){
const button = $("Download")
const label = button.textContent
button.disabled = true
button.textContent = "Building..."
try{
const sheets = payWorkbookSheets(rows("seasons"), rows("franchises"), rows("league"), S)
const name = S.measures ? S.workbook.replace(".xlsx", `_${state.measure}.xlsx`) : S.workbook
await payWriteWorkbook(sheets, name)
button.textContent = label
}catch(e){
console.log(e)
button.textContent = "Download unavailable here"
}
button.disabled = false
}

const toHistory = () => $("TeamSection").scrollIntoView({ behavior: "smooth" })
$("SeasonPick").addEventListener("change", e => {
state.season = Number(e.target.value)
state.teamSeason = state.season
renderSeason()
renderTeam()
})
if(S.measures){
$("Measures").addEventListener("click", e => {
const button = e.target.closest("[data-pay-measure]")
if(!button){ return }
state.measure = button.dataset.payMeasure
$("Measures").querySelectorAll("[data-pay-measure]").forEach(b => b.classList.toggle("active", b === button))
renderAll()
})
}
$("TeamPick").addEventListener("change", e => selectTeam(e.target.value))
$("Table").addEventListener("click", e => {
const sort = e.target.closest("[data-pay-sort]")
if(sort){ state.sortKey = sort.dataset.paySort; renderSeason(); return }
const team = e.target.closest("[data-pay-team]")
if(team){ selectTeam(team.dataset.payTeam); toHistory() }
})
$("Scatter").addEventListener("click", e => {
const team = e.target.closest("[data-pay-team]")
if(team){ selectTeam(team.dataset.payTeam) }
})
$("TeamChart").addEventListener("click", e => {
const season = e.target.closest("[data-pay-season]")
if(season){ state.teamSeason = Number(season.dataset.paySeason); renderTeam() }
})
$("Franchises").addEventListener("click", e => {
const team = e.target.closest("[data-pay-team]")
if(team){ selectTeam(team.dataset.payTeam); toHistory() }
})
$("GridTabs").addEventListener("click", e => {
const tab = e.target.closest("[data-pay-grid]")
if(tab){ state.gridKey = tab.dataset.payGrid; renderGrid() }
})
$("Download").addEventListener("click", download)
let resizeTimer = null
window.addEventListener("resize", () => {
clearTimeout(resizeTimer)
resizeTimer = setTimeout(() => {
if(state.season === null || !root.offsetParent){ return }
renderSeason()
renderTeam()
$("League").innerHTML = payLeagueChartSvg(rows("league"), width("League"), S)
}, 150)
})

Promise.all([payOptionalCSV(S.files.seasons), payOptionalCSV(S.files.franchises), payOptionalCSV(S.files.league)])
.then(([seasons, franchises, league]) => {
Object.assign(loaded, { seasons, franchises, league })
if(!seasons.length){
$("Hero").innerHTML = "No payroll data published yet."
return
}
state.season = payDefaultSeason(rows("seasons"))
$("SeasonPick").innerHTML = paySeasons(rows("seasons")).map(s =>
`<option value="${s}"${s === state.season ? " selected" : ""}>${s}</option>`).join("")
const teams = [...new Set(rows("seasons").map(r => r.franchise))].sort((a, b) => S.name(a).localeCompare(S.name(b)))
$("TeamPick").innerHTML = teams.map(t => `<option value="${t}">${mlbEsc(S.name(t))}</option>`).join("")
// Open the team view on the season's best value.
const best = paySeasonRows(rows("seasons"), state.season).filter(r => mlbNum(r.pe_index) !== null).sort((a, b) => b.pe_index - a.pe_index)[0]
state.team = best ? best.franchise : teams[0]
renderAll()
})
}

installVizTips()
payrollView(PAY_MLB, "pay", "payrollView")
