// Payroll page (payroll.html): loads the CSVs and wires the DOM; the
// rendering lives in payroll_render.js. The workbook download is built
// here in the browser with ExcelJS (loaded on first click).

async function loadCSV(path){
const response = await fetch(`${path}?t=${Date.now()}`, {cache:"no-store"})
if(!response.ok){ throw new Error(`Failed to load ${path}: ${response.status}`) }
const text = await response.text()
return Papa.parse(text, {header:true, skipEmptyLines:true}).data
}

const EXCELJS_URL = "https://cdn.jsdelivr.net/npm/exceljs@4.4.0/dist/exceljs.min.js"

let paySeasonsAll = []
let payFranchises = []
let payLeague = []
let paySeason = null
let payTeam = null
let payTeamSeason = null
let paySortKey = "pe_index"
let payGridKey = "team_pe"

async function payOptionalCSV(path){
try{ return await loadCSV(path) }catch(e){ console.log(`no ${path} yet`, e); return [] }
}

function payWidth(id){
const width = document.getElementById(id).clientWidth
return width > 100 ? width - 32 : 720
}

function renderPaySeason(){
const rows = paySeasonRows(paySeasonsAll, paySeason)
const status = rows[0] && rows[0].status
const label = status === "forecast" ? " (preseason forecast)" : status === "in progress" ? " (in progress)" : ""
document.getElementById("payHero").innerHTML = payHeroHtml(rows, payLeague, paySeason, payModelMiss(paySeasonsAll))
const scatter = payScatterSvg(rows, payTeam, payWidth("payScatter"))
document.getElementById("payScatter").innerHTML = scatter ? `<div class="vizHead"><b>${paySeason}${label}: wins against payroll</b>
<span class="vizKey"><i class="payKeyGood"></i>More value <i class="payKeyMid"></i>Average <i class="payKeyBad"></i>Less value</span>
<span class="vizSub">Each dot is a team. Further right spent more than the league average; higher won more. Teams above the curve got more wins per dollar than the league did.</span></div>${scatter}`
: `<div class="vizHead"><b>${paySeason}${label}</b><span class="vizSub">No games played yet - the table shows expected wins from payroll alone.</span></div>`
document.getElementById("paySeasonTitle").textContent = `${paySeason} season${label}`
document.getElementById("payTable").innerHTML = paySeasonTableHtml(rows, paySortKey, payTeam)
}

function renderPayTeam(){
if(!payTeam){ return }
const own = paySeasonsAll.filter(r => r.franchise === payTeam)
if(!own.some(r => Number(r.season) === payTeamSeason)){ payTeamSeason = Number(paySeason) }
const pick = own.find(r => Number(r.season) === payTeamSeason) || own[own.length - 1]
document.getElementById("payTeamPick").value = payTeam
document.getElementById("payTeamChart").innerHTML = `<div class="vizHead"><b>${mlbEsc(mlbTeam(payTeam))}: wins and expected wins per 162 games</b>
<span class="vizKey"><i class="payKeyActual"></i>Wins <i class="payKeyExpected"></i>Expected from payroll</span>
<span class="vizSub">Click a season to see the 10 seasons its expected wins came from.</span></div>${payTeamChartSvg(paySeasonsAll, payTeam, payWidth("payTeamChart"))}`
document.getElementById("payNeighbors").innerHTML = payNeighborsHtml(paySeasonsAll, pick)
}

function selectPayTeam(team, season = null){
payTeam = team
payTeamSeason = season === null ? Number(paySeason) : season
renderPaySeason()
renderPayTeam()
}

function renderPayGrid(){
document.getElementById("payGridTabs").innerHTML = payGridTabsHtml(payGridKey)
document.getElementById("payGrid").innerHTML = payGridHtml(paySeasonsAll, payGridKey)
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

async function downloadPayWorkbook(){
const button = document.getElementById("payDownload")
const label = button.textContent
button.disabled = true
button.textContent = "Building..."
try{
if(typeof ExcelJS === "undefined"){ await payLoadScript(EXCELJS_URL) }
const wb = new ExcelJS.Workbook()
payWorkbookSheets(paySeasonsAll, payFranchises, payLeague).forEach(spec => {
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
link.download = "Payroll_Effectiveness.xlsx"
document.body.appendChild(link)
link.click()
link.remove()
setTimeout(() => URL.revokeObjectURL(link.href), 1000)
button.textContent = label
}catch(e){
console.log(e)
button.textContent = "Download unavailable here"
}
button.disabled = false
}

function wirePayroll(){
document.getElementById("paySeasonPick").addEventListener("change", e => {
paySeason = Number(e.target.value)
payTeamSeason = paySeason
renderPaySeason()
renderPayTeam()
})
document.getElementById("payTeamPick").addEventListener("change", e => selectPayTeam(e.target.value))
document.getElementById("payTable").addEventListener("click", e => {
const sort = e.target.closest("[data-pay-sort]")
if(sort){ paySortKey = sort.dataset.paySort; renderPaySeason(); return }
const team = e.target.closest("[data-pay-team]")
if(team){ selectPayTeam(team.dataset.payTeam); document.getElementById("payTeamSection").scrollIntoView({ behavior: "smooth" }) }
})
document.getElementById("payScatter").addEventListener("click", e => {
const team = e.target.closest("[data-pay-team]")
if(team){ selectPayTeam(team.dataset.payTeam) }
})
document.getElementById("payTeamChart").addEventListener("click", e => {
const season = e.target.closest("[data-pay-season]")
if(season){ payTeamSeason = Number(season.dataset.paySeason); renderPayTeam() }
})
document.getElementById("payFranchises").addEventListener("click", e => {
const team = e.target.closest("[data-pay-team]")
if(team){ selectPayTeam(team.dataset.payTeam); document.getElementById("payTeamSection").scrollIntoView({ behavior: "smooth" }) }
})
document.getElementById("payGridTabs").addEventListener("click", e => {
const tab = e.target.closest("[data-pay-grid]")
if(tab){ payGridKey = tab.dataset.payGrid; renderPayGrid() }
})
document.getElementById("payDownload").addEventListener("click", downloadPayWorkbook)
let resizeTimer = null
window.addEventListener("resize", () => {
clearTimeout(resizeTimer)
resizeTimer = setTimeout(() => {
if(paySeason === null){ return }
renderPaySeason()
renderPayTeam()
document.getElementById("payLeague").innerHTML = payLeagueChartSvg(payLeague, payWidth("payLeague"))
}, 150)
})
}

async function loadPayroll(){
const [seasons, franchises, league] = await Promise.all([
payOptionalCSV("./data/payroll_seasons.csv"), payOptionalCSV("./data/payroll_franchises.csv"), payOptionalCSV("./data/payroll_league.csv"),
])
paySeasonsAll = seasons
payFranchises = franchises
payLeague = league
if(!seasons.length){
document.getElementById("payHero").innerHTML = "No payroll data published yet."
return
}
paySeason = payDefaultSeason(seasons)
document.getElementById("paySeasonPick").innerHTML = paySeasons(seasons).map(s =>
`<option value="${s}"${s === paySeason ? " selected" : ""}>${s}</option>`).join("")
const teams = [...new Set(seasons.map(r => r.franchise))].sort((a, b) => mlbTeam(a).localeCompare(mlbTeam(b)))
document.getElementById("payTeamPick").innerHTML = teams.map(t => `<option value="${t}">${mlbEsc(mlbTeam(t))}</option>`).join("")
// Open the team view on the season's best value.
const best = paySeasonRows(seasons, paySeason).filter(r => mlbNum(r.pe_index) !== null).sort((a, b) => b.pe_index - a.pe_index)[0]
document.getElementById("payFranchises").innerHTML = payFranchiseTableHtml(franchises)
document.getElementById("payLeague").innerHTML = payLeagueChartSvg(league, payWidth("payLeague"))
renderPayGrid()
selectPayTeam(best ? best.franchise : teams[0])
}

installVizTips()
wirePayroll()
loadPayroll()
