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
another," not the ceiling."""

import datetime

import pandas as pd

_REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; mlb-metrics-prospect-board/1.0)"}


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["rank", "player_name"])


def _first_table_with_columns(tables, required_columns: set) -> pd.DataFrame | None:
    for table in tables:
        if required_columns.issubset(set(table.columns)):
            return table
    return None


def fetch_mlb_pipeline_prospects() -> pd.DataFrame:
    """MLB.com's own "Top Prospects" list, via pybaseball's existing,
    maintained `top_prospects()` scraper (already a real dependency of
    this project) - the lowest-risk of these fetchers, since pybaseball's
    own maintainers are responsible for keeping it working against
    MLB.com's real page structure, not this project."""
    try:
        from pybaseball import top_prospects

        raw = top_prospects()
    except Exception as exc:
        print(f"[prospect_sources] MLB Pipeline fetch failed: {exc}")
        return _empty()

    if raw is None or raw.empty or "Rk" not in raw.columns or "Name" not in raw.columns:
        print("[prospect_sources] MLB Pipeline returned an unexpected shape - skipping this source.")
        return _empty()

    result = raw.rename(columns={"Rk": "rank", "Name": "player_name"}).copy()
    result["rank"] = pd.to_numeric(result["rank"], errors="coerce")
    result = result.dropna(subset=["rank", "player_name"])
    result["source_url"] = "https://www.mlb.com/prospects/stats/top-prospects"
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
        tables = pd.read_html(response.content)
    except Exception as exc:
        print(f"[prospect_sources] Baseball America fetch failed: {exc}")
        return _empty()

    table = _first_table_with_columns(tables, {"Rank", "Name"})
    if table is None:
        table = _first_table_with_columns(tables, {"Rk", "Name"})
    if table is None:
        print("[prospect_sources] Baseball America page had no recognizable ranking table - skipping.")
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
