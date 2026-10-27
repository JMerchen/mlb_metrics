"""Temporary: how far back Spotrac's and FanGraphs' team payrolls go."""
import re
import time
from playwright.sync_api import sync_playwright

urls = {f"spotrac_{y}": f"https://www.spotrac.com/mlb/payroll/_/year/{y}/" for y in (1990, 1995, 2000, 2005, 2010, 2011, 2012, 2023)}
urls["fangraphs_payroll"] = "https://www.fangraphs.com/roster-resource/payroll"
urls["fangraphs_payroll_2015"] = "https://www.fangraphs.com/roster-resource/payroll?season=2015"
with sync_playwright() as p, open("payroll_probe/coverage.txt", "w") as out:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    for name, url in urls.items():
        try:
            page.goto(url, timeout=60000)
            for _ in range(15):
                if "Just a moment" not in page.title():
                    break
                time.sleep(2)
            page.wait_for_timeout(4000)
            text = re.sub(r"\s+", " ", page.inner_text("body"))
            i = text.find("Yankees")
            out.write(f"== {name} {page.url}\ntitle: {page.title()}\n{text[max(0, i - 600):i + 900] if i >= 0 else text[:1500]}\n\n")
            print(name, page.title(), flush=True)
        except Exception as e:
            out.write(f"== {name} ERROR {e}\n\n")
    browser.close()
