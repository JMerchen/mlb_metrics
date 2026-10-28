"""Team payrolls from The Baseball Cube - the Payroll page's one source.

thebaseballcube.com/content/payroll_year/<season> lists every team's
opening-day payroll (players under contract on opening day; later call-ups
and trades don't count), 1988 on - the source the original workbook was
built from. The site turns automated requests away (a Cloudflare check),
so the pages come from the Wayback Machine's copies: the newest capture
with a complete table, asking the archive for a fresh capture when it has
none for a season yet.
"""

import html
import re
import time

import pandas as pd
import requests

from mlb_metrics import payroll_value

CUBE_URL = "https://www.thebaseballcube.com/content/payroll_year/{season}/"
CDX_URL = "https://web.archive.org/cdx/search/cdx"
SNAPSHOT_URL = "https://web.archive.org/web/{timestamp}id_/{original}"
SAVE_URL = "https://web.archive.org/save/{url}"
USER_AGENT = "mlb_metrics payroll page (github.com/JMerchen/mlb_metrics)"
TRIES = 6


def expected_teams(season: int) -> int:
    return 26 if season <= 1992 else 28 if season <= 1997 else 30


def parse_cube_payrolls(page: str, season: int) -> pd.DataFrame:
    """A payroll_year page -> [season, team, franchise, payroll, wins,
    losses], one row per team in its table."""
    rows = []
    for tr in re.findall(r'<tr class="data-row.*?</tr>', page, re.S):
        cells = re.findall(r'<td[^>]*?(?:sortkey="(\d+)")?[^>]*>(.*?)</td>', tr, re.S)
        text = [html.unescape(re.sub(r"<[^>]+>", "", c[1])).strip() for c in cells]
        if len(cells) < 8 or text[0] != str(season):
            continue
        payroll = cells[5][0] or text[5].replace(",", "")
        rows.append({"season": season, "team": text[1], "payroll": int(payroll),
                     "wins": pd.to_numeric(text[6], errors="coerce"), "losses": pd.to_numeric(text[7], errors="coerce")})
    out = pd.DataFrame(rows, columns=["season", "team", "payroll", "wins", "losses"])
    out.insert(2, "franchise", out["team"].map(payroll_value.franchise_code))
    return out


def _get(session, url, **kwargs):
    for attempt in range(3):
        try:
            response = session.get(url, timeout=60, **kwargs)
            if response.status_code == 200:
                return response
            print(f"  {response.status_code} from {url}", flush=True)
            if response.status_code == 429:  # the archive's rate limit: back off once, then give up
                if attempt:
                    return None
                time.sleep(30)
                continue
        except requests.RequestException as e:
            print(f"  {type(e).__name__} from {url}", flush=True)
        time.sleep(5 * (attempt + 1))
    return None


def snapshots(season: int, session) -> list:
    """(timestamp, original URL) of the archive's good captures of the
    season's page, newest first."""
    found = set()
    for host in ("www.thebaseballcube.com", "thebaseballcube.com"):
        response = _get(session, CDX_URL, params={
            "url": f"{host}/content/payroll_year/{season}", "matchType": "prefix", "output": "json",
            "filter": "statuscode:200", "fl": "timestamp,original"})
        if response is not None and response.text.strip():
            found.update(tuple(row) for row in response.json()[1:])
    return sorted(found, reverse=True)


def request_capture(season: int, session) -> bool:
    """Ask the Wayback Machine to capture the season's page now."""
    response = _get(session, SAVE_URL.format(url=CUBE_URL.format(season=season)))
    return response is not None


def fetch_season(season: int, session=None, capture: bool = False):
    """(payrolls, snapshot timestamp) from the newest complete capture, or
    (None, None). `capture`: with no complete capture, request one first."""
    session = session or requests.Session()
    session.headers.setdefault("User-Agent", USER_AGENT)
    for attempt in range(2):
        for timestamp, original in snapshots(season, session)[:TRIES]:
            response = _get(session, SNAPSHOT_URL.format(timestamp=timestamp, original=original))
            if response is None:
                continue
            table = parse_cube_payrolls(response.text, season)
            if len(table) == expected_teams(season) and table["franchise"].nunique() == len(table):
                return table, timestamp
            print(f"  {season} capture {timestamp}: {len(table)} teams - trying an older one")
        if not capture or attempt:
            break
        print(f"  {season}: asking the Wayback Machine for a fresh capture")
        if not request_capture(season, session):
            break
        time.sleep(30)
    return None, None


def apply_payrolls(payrolls: pd.DataFrame, fetched: pd.DataFrame, snapshot: str) -> pd.DataFrame:
    """Replace the season's payrolls with `fetched` (adding any team-season
    not there yet, wins and losses blank), recording the capture each came
    from in `payroll_source`."""
    season = int(fetched["season"].iloc[0])
    out = payrolls.copy()
    if "payroll_source" not in out.columns:
        out["payroll_source"] = pd.NA
    out["payroll_source"] = out["payroll_source"].astype(object)
    source = f"thebaseballcube.com via web.archive.org/web/{snapshot}"
    keyed = fetched.set_index("franchise")["payroll"]
    mask = out["season"] == season
    known = out.loc[mask, "franchise"]
    out.loc[mask, "payroll"] = known.map(keyed).fillna(out.loc[mask, "payroll"]).astype("int64").values
    out.loc[mask & out["franchise"].isin(keyed.index), "payroll_source"] = source
    new = fetched[~fetched["franchise"].isin(known)]
    if not new.empty:
        add = pd.DataFrame({"season": season, "franchise": new["franchise"], "payroll": new["payroll"],
                            "wins": pd.NA, "losses": pd.NA, "payroll_source": source})
        out = pd.concat([out, add], ignore_index=True)
    return out.sort_values(["season", "franchise"]).reset_index(drop=True)
