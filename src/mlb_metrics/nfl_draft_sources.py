"""Real per-source fetchers for the NFL draft-eligible college board
(current college players eligible for the next NFL draft) -
consensus_rankings.build_consensus_ranking's own real inputs. Same real
contract, graceful-degradation posture, and honest "could not verify
reachability from the development environment" caveat as
prospect_sources.py's own module docstring - see that file for the full
reasoning, not repeated here.

Real, honest 2026-09-13 note: NFL Mock Draft Database's own real
"Consensus Big Board" is ITSELF already an aggregation of 100+ other real
big boards/mock drafts - pulling it as one of two sources here means this
board is a real "consensus of (a consensus + one more independent
board)," not two fully independent primary sources. That's a real,
disclosed methodology choice (matching how the user's own original
request was framed - aggregating already-published rankings, not
re-deriving every underlying board from scratch ourselves), not a hidden
one - the board's own `source_ranked_by`/`source_ranks` output columns
make this visible rather than presenting it as more independent than it
is. These pages are also real, JS-heavy sites more likely than
prospect_sources.py's targets to need real fixes once they actually run
in CI (a static `pd.read_html` may not see content a browser renders
client-side) - flagged honestly, not assumed to already work."""

import datetime

import pandas as pd

_REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; mlb-metrics-draft-board/1.0)"}


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["rank", "player_name"])


def _first_table_with_columns(tables, required_columns: set) -> pd.DataFrame | None:
    for table in tables:
        if required_columns.issubset(set(table.columns)):
            return table
    return None


def fetch_nfl_mock_draft_database_consensus(season: int = None) -> pd.DataFrame:
    """NFL Mock Draft Database's real, published "Consensus Big Board" for
    `season` (this real calendar year if not given) - a year-templated
    URL. See module docstring for the "this source is itself already an
    aggregate" disclosure."""
    season = datetime.date.today().year if season is None else season
    url = f"https://www.nflmockdraftdatabase.com/big-boards/{season}/consensus-big-board-{season}"
    try:
        import requests

        response = requests.get(url, timeout=30, headers=_REQUEST_HEADERS)
        response.raise_for_status()
        tables = pd.read_html(response.content)
    except Exception as exc:
        print(f"[nfl_draft_sources] NFL Mock Draft Database fetch failed: {exc}")
        return _empty()

    table = _first_table_with_columns(tables, {"Rank", "Name"})
    if table is None:
        table = _first_table_with_columns(tables, {"Rk", "Name"})
    if table is None:
        print("[nfl_draft_sources] NFL Mock Draft Database page had no recognizable ranking table - skipping.")
        return _empty()

    rank_col = "Rank" if "Rank" in table.columns else "Rk"
    result = table.rename(columns={rank_col: "rank", "Name": "player_name"}).copy()
    result["rank"] = pd.to_numeric(result["rank"], errors="coerce")
    result = result.dropna(subset=["rank", "player_name"])
    result["source_url"] = url
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
        tables = pd.read_html(response.content)
    except Exception as exc:
        print(f"[nfl_draft_sources] FantasyPros fetch failed: {exc}")
        return _empty()

    table = _first_table_with_columns(tables, {"Rank", "Player"})
    if table is None:
        table = _first_table_with_columns(tables, {"Rank", "Name"})
    if table is None:
        print("[nfl_draft_sources] FantasyPros page had no recognizable ranking table - skipping.")
        return _empty()

    name_col = "Player" if "Player" in table.columns else "Name"
    result = table.rename(columns={"Rank": "rank", name_col: "player_name"}).copy()
    result["rank"] = pd.to_numeric(result["rank"], errors="coerce")
    result = result.dropna(subset=["rank", "player_name"])
    result["source_url"] = url
    return result


SOURCE_FETCHERS = {
    "NFL Mock Draft Database": fetch_nfl_mock_draft_database_consensus,
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
