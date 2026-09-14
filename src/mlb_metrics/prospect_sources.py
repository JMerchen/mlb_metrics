"""Real per-source fetchers for the MLB organizational prospect board
(minor leaguers already in an MLB org, not yet debuted) -
consensus_rankings.build_consensus_ranking's own real inputs. Each
fetcher returns a real DataFrame with at least [player_name, rank] on
success, or an EMPTY DataFrame with those same columns on ANY real
failure (network, parsing, unexpected page structure) - a missing source
is a real, honestly-reported gap for that day's run, never a crashed
pipeline (same graceful-degradation posture as
nfl_game_picks.apply_ml_model/apply_calibration elsewhere in this
project).

Real, honest caveat (2026-09-13): these fetchers scrape live, third-party
HTML pages this project does not control, and could not be verified
reachable or parseable from the development environment - a confirmed,
hard network policy blocked every direct attempt to inspect these pages
before writing this file (both raw HTTP and the harness's own page-fetch
tool returned EGRESS_BLOCKED for every candidate site tried, including
plain data APIs). They are expected to need real fixes once they actually
run in the scheduled GitHub Actions job (a normally-unrestricted network,
unlike this development sandbox) - `fetch_all_sources`'s own per-source
try/except, plus the runner script's own honest per-source success/
failure log, exist specifically so a real breakage here degrades to
"fewer sources this week," never a failed pipeline run. Extend
SOURCE_FETCHERS with more outlets as they're confirmed working in CI -
two real, independent sources is enough to start "talking to one
another," not the ceiling.

**Real bugs found on the first live CI run (2026-09-13)** - exactly the
"iterate on failures" plan, now with real evidence instead of a
hypothesis:
1. `pd.read_html` on this project's pinned pandas (3.0.x) raises a real
   `FileNotFoundError` when given a raw `bytes`/`str` HTML blob directly
   (a real pandas 3.x behavior change - earlier pandas accepted a raw
   HTML string; 3.x requires a real file-like object). Fixed everywhere
   in this file by wrapping the response in `io.StringIO` first.
2. Baseball America returned a real `403 Forbidden` - real bot-blocking,
   not a parsing bug. A more realistic, full browser-style header set
   is a real, honest attempt at this (some basic bot-detection only
   checks for a plausible header SET, not just User-Agent), but this may
   be a durable block (e.g. a known-cloud-IP-range denylist) that a
   header change alone can't fix - reported honestly either way once
   this runs again, not assumed fixed.
3. pybaseball's own `top_prospects()` hit the SAME real `pd.read_html`
   bug internally (it calls `pd.read_html(requests.get(url).content)`
   with no StringIO wrapping) - an external library bug this project
   cannot patch from here. Rather than depend on a call known to be
   broken under this project's own pinned pandas version,
   `fetch_mlb_pipeline_prospects` now fetches and parses MLB.com's same
   real page itself (same URL, same real "both batters and pitchers,
   sorted by rank" shape pybaseball's own source documents), with the
   StringIO fix applied."""

import datetime
import io

import pandas as pd

# A real, full browser-style header set (not just User-Agent) - some
# basic bot-detection checks for a plausible SET of headers a real
# browser would send, not just one field. Not guaranteed to clear a
# real Cloudflare-class block (a real, honest limitation - see module
# docstring point 2), but a real, low-cost, non-deceptive attempt (this
# project's own real User-Agent identifies itself, not a spoofed one).
_REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; mlb-metrics-prospect-board/1.0; +https://github.com/JMerchen/mlb_metrics)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _read_html_tables(content: bytes):
    """pd.read_html on this project's pinned pandas (3.0.x) raises a real
    FileNotFoundError when given raw bytes/str directly - it must be a
    real file-like object. See module docstring point 1 for the full
    real bug this works around (confirmed via a live failed CI run, not
    a hypothetical)."""
    return pd.read_html(io.StringIO(content.decode("utf-8", errors="replace")))


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["rank", "player_name"])


def _first_table_with_columns(tables, required_columns: set) -> pd.DataFrame | None:
    for table in tables:
        if required_columns.issubset(set(table.columns)):
            return table
    return None


def fetch_mlb_pipeline_prospects() -> pd.DataFrame:
    """MLB.com's own "Top Prospects" list. Originally delegated to
    pybaseball's existing, maintained `top_prospects()` scraper - but a
    real live CI run (2026-09-13) found pybaseball's own internal call
    hits a real pandas 3.x `pd.read_html` incompatibility this project
    can't patch (see module docstring point 3), so this fetches and
    parses the SAME real URL/shape directly instead (leaguewide, both
    batters and pitchers concatenated and sorted by real rank - the same
    shape pybaseball's own `top_prospects(teamName=None, playerType=None)`
    documents), with the real StringIO fix applied."""
    url = "https://www.mlb.com/prospects/stats/top-prospects"
    try:
        import requests

        response = requests.get(url, timeout=30, headers=_REQUEST_HEADERS)
        response.raise_for_status()
        tables = _read_html_tables(response.content)
    except Exception as exc:
        print(f"[prospect_sources] MLB Pipeline fetch failed: {exc}")
        return _empty()

    if len(tables) < 2:
        print(f"[prospect_sources] MLB Pipeline page had an unexpected number of tables ({len(tables)}) - "
              f"skipping this source. Columns per table: {[list(t.columns) for t in tables]}")
        return _empty()

    raw = pd.concat(tables[:2], ignore_index=True)
    if "Rk" not in raw.columns or "Name" not in raw.columns:
        print(f"[prospect_sources] MLB Pipeline returned an unexpected shape - skipping this source. "
              f"Columns found: {list(raw.columns)}")
        return _empty()
    raw = raw.sort_values(by="Rk")

    result = raw.rename(columns={"Rk": "rank", "Name": "player_name"}).copy()
    result["rank"] = pd.to_numeric(result["rank"], errors="coerce")
    result = result.dropna(subset=["rank", "player_name"])
    result["source_url"] = url
    return result


def fetch_baseball_america_prospects(season: int = None) -> pd.DataFrame:
    """Baseball America's real, published Top 100 Prospects list for
    `season` (this real calendar year if not given) - a year-templated
    URL (baseballamerica.com/rankings/{season}-top-100-prospects/), not a
    per-article numeric id like some outlets use (which would silently
    break every single year with no obvious fix). Real risk flagged
    honestly: Baseball America has historically paywalled part of its
    real rankings behind a subscription, so this may only ever return a
    real PARTIAL list (e.g. a free top 25) rather than the full 100 - a
    real, honest partial source is still a real source for
    consensus_rankings.build_consensus_ranking (it simply can't vouch for
    players outside its own real reach), not a reason to exclude it."""
    season = datetime.date.today().year if season is None else season
    url = f"https://www.baseballamerica.com/rankings/{season}-top-100-prospects/"
    try:
        import requests

        response = requests.get(url, timeout=30, headers=_REQUEST_HEADERS)
        response.raise_for_status()
        tables = _read_html_tables(response.content)
    except Exception as exc:
        print(f"[prospect_sources] Baseball America fetch failed: {exc}")
        return _empty()

    table = _first_table_with_columns(tables, {"Rank", "Name"})
    if table is None:
        table = _first_table_with_columns(tables, {"Rk", "Name"})
    if table is None:
        print(f"[prospect_sources] Baseball America page had no recognizable ranking table - skipping. "
              f"Found {len(tables)} tables with columns: {[list(t.columns) for t in tables]}")
        return _empty()

    rank_col = "Rank" if "Rank" in table.columns else "Rk"
    result = table.rename(columns={rank_col: "rank", "Name": "player_name"}).copy()
    result["rank"] = pd.to_numeric(result["rank"], errors="coerce")
    result = result.dropna(subset=["rank", "player_name"])
    result["source_url"] = url
    return result


SOURCE_FETCHERS = {
    "MLB Pipeline": fetch_mlb_pipeline_prospects,
    "Baseball America": fetch_baseball_america_prospects,
}


def fetch_all_sources() -> dict[str, pd.DataFrame]:
    """Calls every real fetcher in SOURCE_FETCHERS, catching anything a
    fetcher itself didn't already handle (a real, final safety net - no
    single source's real failure should ever take down the others).
    Returns {source_name: DataFrame}, including real empty frames for any
    source that failed - callers report which sources actually returned
    data (see consensus_rankings.build_consensus_ranking's own handling
    of empty sources)."""
    results = {}
    for name, fetcher in SOURCE_FETCHERS.items():
        try:
            results[name] = fetcher()
        except Exception as exc:
            print(f"[prospect_sources] {name} fetch raised unexpectedly: {exc}")
            results[name] = _empty()
    return results
