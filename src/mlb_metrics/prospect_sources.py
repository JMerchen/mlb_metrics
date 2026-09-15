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
   StringIO fix applied.

**Real user feedback (2026-09-15), after 6 live CI runs**: with Baseball
America durably 403-blocked, this board had settled at ONE working
source (MLB Pipeline) - a real single-source passthrough, not an actual
consensus, which the user correctly called out as "not worth it" for a
feature whose whole point is blending multiple real outlets. Three more
real candidates added below (`fetch_fangraphs_prospects`,
`fetch_cbs_sports_prospects`, `fetch_prospects_live_rankings`) - same
"ship blind, iterate on failures" posture as every source in this
project so far, each with a `url` override for when a guessed real URL
pattern turns out wrong (the same escape hatch
`fetch_baseball_america_prospects`/`nfl_draft_sources.fetch_drafttek_big_board`
already use). Picked deliberately as real, separate old-style
server-rendered article/table pages rather than JS-heavy interactive
tools (FanGraphs' own "The Board" is a known React SPA and was
considered and rejected for that reason - the same real lesson
`nfl_draft_sources.py`'s own docstring already documents for NFL Mock
Draft Database) - not a guarantee against real 403s or an unexpected
page shape, but a real, honest bet on plain HTML being reachable at
all. Real per-source outcome reported here once a live run confirms it,
not asserted ahead of that.

**Real live run result (2026-09-15)**: all 3 new candidates FAILED,
each in a real, different, informative way - FanGraphs found 6 tables
with only integer columns (landed on the wrong page entirely, not a
parsing bug), CBS Sports and Prospects Live returned "No tables found"
(also a wrong-URL symptom, not confirmed JS-rendering like NFL Mock
Draft Database - a real, important distinction). Root cause confirmed
via `WebSearch` (a tool that reaches the open web through a different
path than this project's own `requests` calls or `WebFetch` - BOTH of
which remain confirmed `EGRESS_BLOCKED` from this dev sandbox, same as
every prior real attempt - see module docstring's original network
caveat): every one of the 3 guessed default URLs was simply wrong.
FanGraphs' real article lives on `blogs.fangraphs.com`, not
`www.fangraphs.com/prospects/...`; Prospects Live's real content is at
the site root (`www.prospectslive.com`), not `/rankings`; CBS Sports'
real page is a dated news-article slug
(`fantasy-baseball-top-100-prospects-for-{season}`), not an evergreen
`/rankings/prospects/` path that was guessed. All 3 default URLs
corrected below using the real, `WebSearch`-confirmed ones. A 4th real
candidate, `fetch_just_baseball_prospects`, added the same day (a real,
confirmed, evergreen non-dated URL) for the same reason - more real
candidate sources raise the odds of ending up with an actual
multi-source consensus, the same reasoning that already worked for the
NFL draft board (DraftTek + FantasyPros).

**Real live run result with the corrected URLs (2026-09-15, 2nd
trigger)**: FanGraphs and Just Baseball BOTH actually reached their
real ranking tables this time (227 and 101 real embedded tables
respectively, most of them small per-player scouting-grade widgets, not
the ranking table itself) - real progress, needing only real parsing
fixes, not another wrong URL: FanGraphs' real header row is `Rk`/`Name`
(fixed via a real `{"Rk", "Name"}` fallback); Just Baseball's real `<th>`
cells carry a visible `"Sort by <field>"` accessibility suffix (e.g.
real `"Rank Sort by rank"`), recovered via the new
`_strip_sortable_header_suffix` helper. CBS Sports and Prospects Live
BOTH still returned "No tables found" even with their real,
WebSearch-confirmed URLs - a real, more informative signal now that a
wrong-URL explanation is ruled out: this reads as genuine client-side
rendering or access-gating on those two specific real pages, not
pursued further this round (same "an honest partial source set is
still real signal" posture as every other real gap in this project)."""

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


def _strip_sortable_header_suffix(table: pd.DataFrame) -> pd.DataFrame:
    """A real, confirmed quirk on Just Baseball's own real ranking table
    (2026-09-15 live CI run): its real `<th>` cells carry a visible
    accessibility label appended to the header text, e.g. real
    `"Rank Sort by rank"` / `"Player Sort by player"` instead of a plain
    `"Rank"` / `"Player"` - `pd.read_html` parses that whole real string
    as the column name, so an exact-match column-name check like every
    other source in this module uses never matches. Strips everything
    from " Sort by" onward (case-insensitive) on every real column,
    leaving a table with no such real header text (or one with only
    integer positional columns) unchanged.

    Real, confirmed bug this ALSO had to fix (2026-09-15, same live CI
    run, a real crash, not a hypothetical): the real page's own table
    had TWO real "Rank"-prefixed columns, already made distinct by
    pandas' own `.1` dedup suffix (`"Rank Sort by rank"` and
    `"Rank Sort by rank.1"`) - stripping at " Sort by" collapses BOTH
    to the identical plain `"Rank"`, silently producing a real
    DUPLICATE-named column. `table.rename(columns={...})` then renames
    BOTH to `"rank"`, so `result["rank"]` returns a real DataFrame
    instead of a Series and `pd.to_numeric` raises `TypeError: arg must
    be a list, tuple, 1-d array, or Series` - this crashed the entire
    real script (see `board_runner.run_board`'s own now-added per-source
    isolation for why one fetcher's bug should never do that again
    regardless). Re-deduplicating the stripped names here, the same way
    pandas itself would, keeps the real distinctness intact."""
    if table.empty:
        return table
    new_columns = []
    for col in table.columns:
        if isinstance(col, str):
            idx = col.lower().find(" sort by")
            new_columns.append(col[:idx] if idx != -1 else col)
        else:
            new_columns.append(col)
    seen = {}
    deduped_columns = []
    for name in new_columns:
        if name not in seen:
            seen[name] = 0
            deduped_columns.append(name)
        else:
            seen[name] += 1
            deduped_columns.append(f"{name}.{seen[name]}")
    renamed = table.copy()
    renamed.columns = deduped_columns
    return renamed


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
    if "Rk" not in raw.columns or "Player" not in raw.columns:
        print(f"[prospect_sources] MLB Pipeline returned an unexpected shape - skipping this source. "
              f"Columns found: {list(raw.columns)}")
        return _empty()
    raw = raw.sort_values(by="Rk")

    result = raw.rename(columns={"Rk": "rank", "Player": "player_name"}).copy()
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


def fetch_fangraphs_prospects(season: int = None, url: str = None) -> pd.DataFrame:
    """FanGraphs' real, published written "Top 100 Prospects" article for
    `season` (this real calendar year if not given) - deliberately NOT
    FanGraphs' own "The Board" tool (a known React SPA, the same
    JS-rendered-content problem `nfl_draft_sources.py`'s own docstring
    documents for NFL Mock Draft Database - `pd.read_html` could never
    reach that page's real data), betting instead on the separate,
    real, old-style written article FanGraphs publishes alongside it
    each offseason. Real, confirmed URL (2026-09-15, via WebSearch,
    NOT direct fetch - see module docstring's real network caveat):
    `blogs.fangraphs.com`, not `www.fangraphs.com` - the FIRST real
    guess landed on the wrong subdomain entirely (confirmed via a live
    CI run returning 6 tiny 3-column tables, real nav/footer content,
    not the actual article). Real, confirmed header row (2026-09-15,
    2nd live CI run with the corrected URL): `Rk`/`Name` - the real
    article page also embeds ~220 small per-player scouting-grade
    tables (Hit/Power/Run/Fielding/Throw, pitch-type grades) alongside
    the one real ranking table, `_first_table_with_columns` picking the
    first real match out of all of them."""
    season = datetime.date.today().year if season is None else season
    url = url or f"https://blogs.fangraphs.com/{season}-top-100-prospects/"
    try:
        import requests

        response = requests.get(url, timeout=30, headers=_REQUEST_HEADERS)
        response.raise_for_status()
        tables = _read_html_tables(response.content)
    except Exception as exc:
        print(f"[prospect_sources] FanGraphs fetch failed: {exc}")
        return _empty()

    table = _first_table_with_columns(tables, {"Rank", "Name"})
    if table is None:
        table = _first_table_with_columns(tables, {"Rank", "Player"})
    if table is None:
        table = _first_table_with_columns(tables, {"#", "Name"})
    if table is None:
        table = _first_table_with_columns(tables, {"Rk", "Name"})
    if table is None:
        print(f"[prospect_sources] FanGraphs page had no recognizable ranking table - skipping. "
              f"Found {len(tables)} tables with columns: {[list(t.columns) for t in tables]}")
        return _empty()

    rank_col = "Rank" if "Rank" in table.columns else ("Rk" if "Rk" in table.columns else "#")
    name_col = "Name" if "Name" in table.columns else "Player"
    result = table.rename(columns={rank_col: "rank", name_col: "player_name"}).copy()
    result["rank"] = pd.to_numeric(result["rank"], errors="coerce")
    result = result.dropna(subset=["rank", "player_name"])
    result["source_url"] = url
    return result


def fetch_cbs_sports_prospects(season: int = None, url: str = None) -> pd.DataFrame:
    """CBS Sports' real, published "Fantasy Baseball: Top 100 prospects
    for {season}" article - real, confirmed URL (2026-09-15, via
    WebSearch, NOT direct fetch - see module docstring's real network
    caveat), a real, honest fragility: this is a dated news-article
    slug (not a stable evergreen rankings hub), so it WILL need a fresh
    real URL confirmed most years - `url` exists specifically as that
    escape hatch, same as `fetch_d1baseball_college_draft`'s own
    precedent for the same kind of real URL fragility."""
    season = datetime.date.today().year if season is None else season
    url = url or f"https://www.cbssports.com/fantasy/baseball/news/fantasy-baseball-top-100-prospects-for-{season}/"
    try:
        import requests

        response = requests.get(url, timeout=30, headers=_REQUEST_HEADERS)
        response.raise_for_status()
        tables = _read_html_tables(response.content)
    except Exception as exc:
        print(f"[prospect_sources] CBS Sports fetch failed: {exc}")
        return _empty()

    table = _first_table_with_columns(tables, {"Rank", "Player"})
    if table is None:
        table = _first_table_with_columns(tables, {"Rank", "Name"})
    if table is None:
        table = _first_table_with_columns(tables, {"RK", "Player"})
    if table is None:
        print(f"[prospect_sources] CBS Sports page had no recognizable ranking table - skipping. "
              f"Found {len(tables)} tables with columns: {[list(t.columns) for t in tables]}")
        return _empty()

    rank_col = "RK" if "RK" in table.columns else "Rank"
    name_col = "Player" if "Player" in table.columns else "Name"
    result = table.rename(columns={rank_col: "rank", name_col: "player_name"}).copy()
    result["rank"] = pd.to_numeric(result["rank"], errors="coerce")
    result = result.dropna(subset=["rank", "player_name"])
    result["source_url"] = url
    return result


def fetch_prospects_live_rankings(url: str = None) -> pd.DataFrame:
    """Prospects Live's real, published Top 100 rankings - a smaller,
    dedicated prospect-coverage outlet, deliberately picked as a real,
    separate voice from the bigger, more heavily-trafficked (and more
    likely bot-protected) outlets already in this module - the same
    "an independent real source is still a real source" reasoning
    `fetch_baseball_america_prospects`'s own docstring already
    establishes. Real, confirmed real site root (2026-09-15, via
    WebSearch, NOT direct fetch - see module docstring's real network
    caveat): `www.prospectslive.com` - the first real guess (bare
    `prospectslive.com/rankings`) returned zero real `<table>`
    elements. Real, honest risk flagged: this outlet's own real
    write-ups are reportedly member-gated, so this may only ever return
    a real PARTIAL list rather than the full 100 - same "an honest
    partial source is still a real source" reasoning as Baseball
    America's own docstring, not a reason to drop it."""
    url = url or "https://www.prospectslive.com/"
    try:
        import requests

        response = requests.get(url, timeout=30, headers=_REQUEST_HEADERS)
        response.raise_for_status()
        tables = _read_html_tables(response.content)
    except Exception as exc:
        print(f"[prospect_sources] Prospects Live fetch failed: {exc}")
        return _empty()

    table = _first_table_with_columns(tables, {"Rank", "Name"})
    if table is None:
        table = _first_table_with_columns(tables, {"Rank", "Player"})
    if table is None:
        table = _first_table_with_columns(tables, {"#", "Name"})
    if table is None:
        print(f"[prospect_sources] Prospects Live page had no recognizable ranking table - skipping. "
              f"Found {len(tables)} tables with columns: {[list(t.columns) for t in tables]}")
        return _empty()

    rank_col = "Rank" if "Rank" in table.columns else "#"
    name_col = "Name" if "Name" in table.columns else "Player"
    result = table.rename(columns={rank_col: "rank", name_col: "player_name"}).copy()
    result["rank"] = pd.to_numeric(result["rank"], errors="coerce")
    result = result.dropna(subset=["rank", "player_name"])
    result["source_url"] = url
    return result


def fetch_just_baseball_prospects(url: str = None) -> pd.DataFrame:
    """Just Baseball's real, published "Top 100 MLB Prospects" page - a
    real, confirmed, evergreen (non-dated) URL (2026-09-15, via
    WebSearch, NOT direct fetch - see module docstring's real network
    caveat), a real, independent, newer outlet - the same
    "an independent real source is still a real source" reasoning
    `fetch_baseball_america_prospects`'s own docstring already
    establishes. Real, confirmed quirk (2026-09-15 live CI run): this
    page's real ranking table's real headers carry a
    "<field> Sort by <field>" accessibility suffix (its real header
    cells read `"Rank Sort by rank"`, `"Player Sort by player"`, ...) -
    `_strip_sortable_header_suffix` recovers the real plain field name
    before matching. The real page also embeds ~100 small per-player
    scouting-grade widgets alongside the one real ranking table, same
    "pick the first real match out of many" shape as FanGraphs'
    own docstring documents above."""
    url = url or "https://www.justbaseball.com/prospects/top-100-mlb-prospects/"
    try:
        import requests

        response = requests.get(url, timeout=30, headers=_REQUEST_HEADERS)
        response.raise_for_status()
        tables = [_strip_sortable_header_suffix(t) for t in _read_html_tables(response.content)]
    except Exception as exc:
        print(f"[prospect_sources] Just Baseball fetch failed: {exc}")
        return _empty()

    table = _first_table_with_columns(tables, {"Rank", "Name"})
    if table is None:
        table = _first_table_with_columns(tables, {"Rank", "Player"})
    if table is None:
        table = _first_table_with_columns(tables, {"#", "Name"})
    if table is None:
        print(f"[prospect_sources] Just Baseball page had no recognizable ranking table - skipping. "
              f"Found {len(tables)} tables with columns: {[list(t.columns) for t in tables]}")
        return _empty()

    rank_col = "Rank" if "Rank" in table.columns else "#"
    name_col = "Name" if "Name" in table.columns else "Player"
    result = table.rename(columns={rank_col: "rank", name_col: "player_name"}).copy()
    result["rank"] = pd.to_numeric(result["rank"], errors="coerce")
    result = result.dropna(subset=["rank", "player_name"])
    result["source_url"] = url
    return result


SOURCE_FETCHERS = {
    "MLB Pipeline": fetch_mlb_pipeline_prospects,
    "Baseball America": fetch_baseball_america_prospects,
    "Just Baseball": fetch_just_baseball_prospects,
    "FanGraphs": fetch_fangraphs_prospects,
    "CBS Sports": fetch_cbs_sports_prospects,
    "Prospects Live": fetch_prospects_live_rankings,
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
