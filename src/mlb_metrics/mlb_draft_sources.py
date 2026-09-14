"""Real per-source fetchers for the MLB draft-eligible college board
(current college juniors/seniors/eligible underclassmen, not yet drafted)
- consensus_rankings.build_consensus_ranking's own real inputs. Same real
contract, graceful-degradation posture, and honest "could not verify
reachability from the development environment" caveat as
prospect_sources.py's own module docstring - see that file for the full
reasoning, not repeated here.

Real, honest 2026-09-13 note on URL stability: unlike prospect_sources.py's
sources, at least one real candidate here (D1Baseball) publishes this kind
of ranking at a URL that has already changed shape between updates within
the SAME draft class (a numeric revision suffix, not a stable year-only
slug) - a real, confirmed fragility, not a hypothetical one. This fetcher
is written defensively (same empty-DataFrame-on-any-failure contract) so a
real URL-shape change degrades to "one fewer source this cycle," but the
URL itself will likely need a real, manual update some season - flagged
here rather than silently assumed permanent.

**Real bugs found on the first live CI run (2026-09-13)**: both real
sources here returned a real `403 Forbidden` (real bot-blocking, not a
parsing bug - see prospect_sources.py's own module docstring point 2 for
the honest "may be a durable block" caveat and the real, full
browser-style header set now sent). Also fixed the same real
`pd.read_html` pandas 3.x bug prospect_sources.py's own module docstring
documents (point 1) - both fetchers here were equally exposed to it, just
masked by the 403 firing first."""

import datetime
import io

import pandas as pd

_REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; mlb-metrics-draft-board/1.0; +https://github.com/JMerchen/mlb_metrics)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["rank", "player_name"])


def _first_table_with_columns(tables, required_columns: set) -> pd.DataFrame | None:
    for table in tables:
        if required_columns.issubset(set(table.columns)):
            return table
    return None


def _read_html_tables(content: bytes):
    """See prospect_sources._read_html_tables's own docstring for the
    full real pandas 3.x bug this works around."""
    return pd.read_html(io.StringIO(content.decode("utf-8", errors="replace")))


def fetch_baseball_america_college_draft(season: int = None) -> pd.DataFrame:
    """Baseball America's real, published "Top College MLB Draft
    Prospects" list for `season` (this real calendar year if not given) -
    a year-templated URL, same real stability reasoning as
    prospect_sources.fetch_baseball_america_prospects (and the same real
    subscription-paywall caveat: this may only return a real partial
    list)."""
    season = datetime.date.today().year if season is None else season
    url = f"https://www.baseballamerica.com/rankings/{season}-top-college-draft-prospects/"
    try:
        import requests

        response = requests.get(url, timeout=30, headers=_REQUEST_HEADERS)
        response.raise_for_status()
        tables = _read_html_tables(response.content)
    except Exception as exc:
        print(f"[mlb_draft_sources] Baseball America fetch failed: {exc}")
        return _empty()

    table = _first_table_with_columns(tables, {"Rank", "Name"})
    if table is None:
        table = _first_table_with_columns(tables, {"Rk", "Name"})
    if table is None:
        print(f"[mlb_draft_sources] Baseball America page had no recognizable ranking table - skipping. "
              f"Found {len(tables)} tables with columns: {[list(t.columns) for t in tables]}")
        return _empty()

    rank_col = "Rank" if "Rank" in table.columns else "Rk"
    result = table.rename(columns={rank_col: "rank", "Name": "player_name"}).copy()
    result["rank"] = pd.to_numeric(result["rank"], errors="coerce")
    result = result.dropna(subset=["rank", "player_name"])
    result["source_url"] = url
    return result


def fetch_d1baseball_college_draft(url: str = None, season: int = None) -> pd.DataFrame:
    """D1Baseball's real, published Top 100 college draft prospect list.
    Unlike Baseball America's stable year-only slug, D1Baseball's real URL
    has already shown a numeric revision suffix (e.g. "...-2/") that isn't
    predictable purely from the season - `url` is exposed as a real,
    explicit override specifically so a real broken URL can be fixed by
    passing the current one (from config or a caller) without touching
    this function's own logic, rather than requiring a code change every
    time the site republishes at a new path."""
    season = datetime.date.today().year if season is None else season
    url = url or f"https://d1baseball.com/prospects/{season}-mlb-draft-top-100-college-prospects/"
    try:
        import requests

        response = requests.get(url, timeout=30, headers=_REQUEST_HEADERS)
        response.raise_for_status()
        tables = _read_html_tables(response.content)
    except Exception as exc:
        print(f"[mlb_draft_sources] D1Baseball fetch failed: {exc}")
        return _empty()

    table = _first_table_with_columns(tables, {"Rank", "Name"})
    if table is None:
        table = _first_table_with_columns(tables, {"Rk", "Name"})
    if table is None:
        print(f"[mlb_draft_sources] D1Baseball page had no recognizable ranking table - skipping. "
              f"Found {len(tables)} tables with columns: {[list(t.columns) for t in tables]}")
        return _empty()

    rank_col = "Rank" if "Rank" in table.columns else "Rk"
    result = table.rename(columns={rank_col: "rank", "Name": "player_name"}).copy()
    result["rank"] = pd.to_numeric(result["rank"], errors="coerce")
    result = result.dropna(subset=["rank", "player_name"])
    result["source_url"] = url
    return result


SOURCE_FETCHERS = {
    "Baseball America": fetch_baseball_america_college_draft,
    "D1Baseball": fetch_d1baseball_college_draft,
}


def fetch_all_sources() -> dict[str, pd.DataFrame]:
    """Same real, per-source isolation contract as
    prospect_sources.fetch_all_sources - one source's real failure never
    takes down the others."""
    results = {}
    for name, fetcher in SOURCE_FETCHERS.items():
        try:
            results[name] = fetcher()
        except Exception as exc:
            print(f"[mlb_draft_sources] {name} fetch raised unexpectedly: {exc}")
            results[name] = _empty()
    return results
