"""Temporary: The Baseball Cube's payroll pages, 1988-2026, from the
Wayback Machine's latest good snapshot of each."""
import csv
import html
import re
import sys
import time

import requests

S = requests.Session()
S.headers["User-Agent"] = "mlb_metrics payroll research (github.com/JMerchen/mlb_metrics)"


def parse(page):
    rows = []
    for tr in re.findall(r'<tr class="data-row.*?</tr>', page, re.S):
        cells = re.findall(r"<td[^>]*?(?:sortkey=\"(\d+)\")?[^>]*>(.*?)</td>", tr, re.S)
        text = [html.unescape(re.sub(r"<[^>]+>", "", c[1])).strip() for c in cells]
        if len(cells) < 8 or not text[0].isdigit():
            continue
        pay = cells[5][0] or text[5].replace(",", "")
        rows.append({"season": int(text[0]), "team": text[1], "payroll": int(pay), "w": text[6], "l": text[7]})
    return rows


def snapshots(year):
    out = []
    for host in ("www.thebaseballcube.com", "thebaseballcube.com"):
        url = f"{host}/content/payroll_year/{year}"
        for attempt in range(4):
            try:
                r = S.get("https://web.archive.org/cdx/search/cdx", params={
                    "url": url, "matchType": "prefix", "output": "json", "filter": "statuscode:200",
                    "fl": "timestamp,original"}, timeout=60)
                data = r.json()
                out += [tuple(d) for d in data[1:]]
                break
            except Exception as e:
                print("cdx retry", year, host, e, flush=True)
                time.sleep(5 * (attempt + 1))
    return sorted(set(out), reverse=True)


def main(out_path):
    found = []
    for year in range(1988, 2027):
        snaps = snapshots(year)
        got = None
        for ts, original in snaps[:6]:
            try:
                page = S.get(f"https://web.archive.org/web/{ts}id_/{original}", timeout=60).text
            except Exception as e:
                print("fetch fail", year, ts, e, flush=True)
                continue
            rows = [r for r in parse(page) if r["season"] == year]
            if len(rows) >= 26:
                got = (ts, rows)
                break
            time.sleep(1)
        print(year, "snapshots", len(snaps), "->", got[0] if got else None, len(got[1]) if got else 0, flush=True)
        if got:
            found += [{**r, "snapshot": got[0]} for r in got[1]]
        time.sleep(1)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["season", "team", "payroll", "w", "l", "snapshot"])
        w.writeheader()
        w.writerows(found)


if __name__ == "__main__":
    if sys.argv[1] == "--test":
        rows = parse(open(sys.argv[2]).read())
        print(len(rows), rows[:2])
    else:
        main(sys.argv[1])
