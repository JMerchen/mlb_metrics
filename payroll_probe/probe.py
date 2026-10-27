"""Temporary: fetch candidate payroll pages with a real browser."""
import sys
import time
from playwright.sync_api import sync_playwright

urls = {
    "cube_1988": "https://thebaseballcube.com/content/payroll_year/1988",
    "cube_2023": "https://thebaseballcube.com/content/payroll_year/2023",
    "cube_2026": "https://thebaseballcube.com/content/payroll_year/2026",
    "wb_cube_1988": "https://web.archive.org/web/2025/https://thebaseballcube.com/content/payroll_year/1988/",
    "wb_cube_2023": "https://web.archive.org/web/2025/https://thebaseballcube.com/content/payroll_year/2023/",
    "ump": "https://stevetheump.com/Payrolls.htm",
}
with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    for name, url in urls.items():
        try:
            page.goto(url, timeout=60000)
            for _ in range(20):
                if "Just a moment" not in page.title():
                    break
                time.sleep(2)
            page.wait_for_timeout(3000)
            html = page.content()
            open(f"payroll_probe/{name}.html", "w").write(html)
            print(name, page.title(), len(html), flush=True)
        except Exception as e:
            print(name, "ERROR", e, flush=True)
    browser.close()
