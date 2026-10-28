"""Temporary: 2024-2026 payrolls from the Wayback Machine's copies of The
Baseball Cube, loaded in a browser (a handful of page loads)."""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from playwright.sync_api import sync_playwright

from mlb_metrics import payroll_sources, payroll_value

PATH = "data/payroll/team_payroll.csv"
payrolls = payroll_value.load_payrolls(PATH)
before = payrolls.set_index(["season", "franchise"])["payroll"]
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    for season in (2024, 2025, 2026):
        for near in ("2026", "2025"):
            url = f"https://web.archive.org/web/{near}/https://www.thebaseballcube.com/content/payroll_year/{season}/"
            try:
                page.goto(url, timeout=90000)
                page.wait_for_timeout(3000)
            except Exception as e:
                print(season, near, "load failed", type(e).__name__, flush=True)
                continue
            table = payroll_sources.parse_cube_payrolls(page.content(), season)
            stamp = (re.search(r"/web/(\d{14})", page.url) or [None, near])[1]
            print(season, near, page.url, len(table), "teams", flush=True)
            if len(table) == payroll_sources.expected_teams(season) and table["franchise"].nunique() == len(table):
                payrolls = payroll_sources.apply_payrolls(payrolls, table, stamp)
                break
    browser.close()
after = payrolls.set_index(["season", "franchise"])["payroll"]
changes = pd.DataFrame({"old": before.reindex(after.index), "new": after})
changes = changes[changes["old"] != changes["new"]].reset_index()
changes["change"] = changes["new"] - changes["old"]
changes.to_csv("payroll_probe/changes.csv", index=False)
print(len(changes), "payrolls changed")
out = payrolls.copy()
for col in ("wins", "losses"):
    out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
out.to_csv(PATH, index=False)
