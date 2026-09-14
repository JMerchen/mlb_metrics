"""Real per-source fetchers for the NFL draft-eligible college board
(current college players eligible for the next NFL draft) -
consensus_rankings.build_consensus_ranking's own real inputs. Same real
contract, graceful-degradation posture, and honest "could not verify
reachability from the development environment" caveat as
prospect_sources.py's own module docstring - see that file for the full
reasoning, not repeated here.

**Real result from the first live CI run (2026-09-13)**: NFL Mock Draft
Database and FantasyPros both actually got a real 200 OK with real HTML
back (unlike prospect_sources.py's Baseball America, which hit a real
403). FantasyPros' failure was the same real pandas 3.x `pd.read_html`
bug prospect_sources.py's own module docstring documents (point 1) - now
fixed.

**Real result from the SECOND live CI run (2026-09-14), after that
fix**: NFL Mock Draft Database returned "No tables found" - a real,
decisive signal this specific page's ranking data is rendered
client-side by JavaScript (an SPA), not present in the raw HTML at all.
No amount of `pd.read_html` parsing logic can reach content that was
never in the response - this needs a real headless browser to render,
which is real added infrastructure (this project has no browser-driven
scraping today), not attempted here. Replaced with DraftTek's real
"NFL Draft Big Board" instead - an old-style, paginated, plain-HTML
site (`.asp` URLs, no visible JS framework in its own URL/page-naming
convention), a real, different bet on scrapability than the JS-heavy
site it replaces - still genuinely unverified until it actually runs,
same "ship blind, iterate" posture as every other source here.

FantasyPros' own real page ALSO had a second, separate quirk once the
`pd.read_html` bug was fixed: its one real ranking table has no `<th>`
header cells, so pandas parsed it with plain positional integer columns
(0, 1, 2, ...) and the real header text became row 0 of the DATA -
`_promote_header_row_if_needed` recovers this specific, confirmed real
case.

**Real result from the THIRD live CI run (2026-09-14), after the
DraftTek swap**: DraftTek's own real page has the SAME missing-`<th>`
quirk FantasyPros had - `_promote_header_row_if_needed` now applies
there too. FantasyPros' real header row, once correctly parsed, reads
"RK"/"PLAYER NAME" (all-caps, a space in the name column) - a real,
different spelling than the friendlier-cased guess this module started
with, now matched as a real fallback."""

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


# Real, confirmed header words seen so far across the real sources this
# module fetches (FantasyPros' own real header row: RK/PLAYER NAME/TEAM/
# POS/..., DraftTek's own real header row not yet confirmed exactly -
# widened defensively with the other real plausible variants a college
# football big board's own real header row could use) - a real,
# hand-maintained set, extended as more real header text is confirmed
# via live CI runs, not asserted ahead of that.
_HEADER_LOOKALIKES = {
    "rank", "rk", "no", "no.",
    "player", "name", "player name", "prospect",
    "team", "pos", "position", "school", "college",
}


def _promote_header_row_if_needed(table: pd.DataFrame) -> pd.DataFrame:
    """A real, confirmed pd.read_html quirk (FantasyPros' own real page,
    2026-09-14 live CI run): a table with no real <th> header cells
    parses with plain positional integer columns (0, 1, 2, ...) and the
    page's real header text ends up as row 0 of the DATA instead of
    becoming the columns. If every column here is a real int (the
    signal this happened) AND row 0 contains at least one recognizable
    real header word, promote row 0 to be the real header and drop it
    from the data - a real, targeted parse recovery, not blind
    guessing (a table that already has real string columns, or whose
    row 0 doesn't look like a header, passes through unchanged)."""
    if table.empty or not all(isinstance(c, int) for c in table.columns):
        return table
    first_row = table.iloc[0].astype(str).str.strip().str.lower()
    if not any(v in _HEADER_LOOKALIKES for v in first_row):
        return table
    promoted = table.iloc[1:].reset_index(drop=True)
    promoted.columns = table.iloc[0].astype(str).str.strip().tolist()
    return promoted


def fetch_drafttek_big_board(season: int = None, pages: int = 1, url: str = None) -> pd.DataFrame:
    """DraftTek's real, published "NFL Draft Big Board" for `season`
    (this real calendar year if not given) - an old-style, paginated,
    plain-HTML site (see module docstring for why this replaced NFL Mock
    Draft Database, which turned out to be JS-rendered). Real, confirmed
    URL pattern: `.../Top-NFL-Draft-Prospects-{season}-Page-{n}.asp`,
    `n` starting at 1 - `pages` controls how many real pages to fetch and
    concatenate (each real page is ~100 more players; default 1 keeps
    this a real, cheap single request until a deeper board is confirmed
    worth the extra real fetches). A real failure on ANY requested page
    degrades this whole source to whatever earlier pages DID succeed
    (not all-or-nothing) - a partial real board is still real signal, the
    same "an honest partial source is still a real source" reasoning
    prospect_sources.fetch_baseball_america_prospects's own docstring
    already establishes. `url` overrides the FIRST page's URL only (a
    real, explicit escape hatch if this real URL pattern ever breaks),
    matching mlb_draft_sources.fetch_d1baseball_college_draft's own
    `url`-override precedent."""
    season = datetime.date.today().year if season is None else season
    base_url = url or f"https://www.drafttek.com/{season}-NFL-Draft-Big-Board/Top-NFL-Draft-Prospects-{season}-Page-1.asp"

    frames = []
    for page in range(1, pages + 1):
        page_url = base_url if page == 1 else (
            f"https://www.drafttek.com/{season}-NFL-Draft-Big-Board/Top-NFL-Draft-Prospects-{season}-Page-{page}.asp"
        )
        try:
            import requests

            response = requests.get(page_url, timeout=30, headers=_REQUEST_HEADERS)
            response.raise_for_status()
            tables = [_promote_header_row_if_needed(t) for t in _read_html_tables(response.content)]
        except Exception as exc:
            print(f"[nfl_draft_sources] DraftTek page {page} fetch failed: {exc}")
            continue

        table = _first_table_with_columns(tables, {"Rank", "Name"})
        if table is None:
            table = _first_table_with_columns(tables, {"Rk", "Name"})
        if table is None:
            print(f"[nfl_draft_sources] DraftTek page {page} had no recognizable ranking table - skipping. "
                  f"Found {len(tables)} tables with columns: {[list(t.columns) for t in tables]}")
            continue

        rank_col = "Rank" if "Rank" in table.columns else "Rk"
        page_result = table.rename(columns={rank_col: "rank", "Name": "player_name"}).copy()
        page_result["source_url"] = page_url
        frames.append(page_result)

    if not frames:
        return _empty()

    result = pd.concat(frames, ignore_index=True)
    result["rank"] = pd.to_numeric(result["rank"], errors="coerce")
    result = result.dropna(subset=["rank", "player_name"])
    return result


def fetch_fantasypros_big_board(url: str = None) -> pd.DataFrame:
    """FantasyPros' real, published NFL Draft Big Board - a stable,
    non-year-suffixed URL as of 2026-09-13 (unlike
    fetch_nfl_mock_draft_database_consensus's year-templated one),
    exposed as a real override in case that changes."""
    url = url or "https://www.fantasypros.com/nfl-draft-big-board-prospect-rankings/"
    try:
        import requests

        response = requests.get(url, timeout=30, headers=_REQUEST_HEADERS)
        response.raise_for_status()
        tables = [_promote_header_row_if_needed(t) for t in _read_html_tables(response.content)]
    except Exception as exc:
        print(f"[nfl_draft_sources] FantasyPros fetch failed: {exc}")
        return _empty()

    # Real, confirmed FantasyPros header row (2026-09-14 live CI run):
    # "RK"/"PLAYER NAME" (all-caps, a space in the name column) - tried
    # after the friendlier-cased guesses in case a future real page
    # reformats back to those.
    table = _first_table_with_columns(tables, {"Rank", "Player"})
    if table is None:
        table = _first_table_with_columns(tables, {"Rank", "Name"})
    if table is None:
        table = _first_table_with_columns(tables, {"RK", "PLAYER NAME"})
    if table is None:
        print(f"[nfl_draft_sources] FantasyPros page had no recognizable ranking table - skipping. "
              f"Found {len(tables)} tables with columns: {[list(t.columns) for t in tables]}")
        return _empty()

    rank_col = "RK" if "RK" in table.columns else "Rank"
    name_col = "PLAYER NAME" if "PLAYER NAME" in table.columns else ("Player" if "Player" in table.columns else "Name")
    result = table.rename(columns={rank_col: "rank", name_col: "player_name"}).copy()
    result["rank"] = pd.to_numeric(result["rank"], errors="coerce")
    result = result.dropna(subset=["rank", "player_name"])
    result["source_url"] = url
    return result


SOURCE_FETCHERS = {
    "DraftTek": fetch_drafttek_big_board,
    "FantasyPros": fetch_fantasypros_big_board,
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
            print(f"[nfl_draft_sources] {name} fetch raised unexpectedly: {exc}")
            results[name] = _empty()
    return results
