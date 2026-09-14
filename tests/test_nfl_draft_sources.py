import sys
import types

import pandas as pd

from mlb_metrics import nfl_draft_sources


def _fake_requests_module(get_fn):
    fake = types.ModuleType("requests")
    fake.get = get_fn
    return fake


class _FakeResponse:
    content = b"<html></html>"

    def raise_for_status(self):
        pass


def test_fetch_drafttek_big_board_normalizes_real_shape(monkeypatch):
    table = pd.DataFrame([
        {"Rank": 1, "Name": "Fernando Mendoza"},
        {"Rank": 2, "Name": "Jeremiyah Love"},
    ])
    seen_urls = []

    def _get(url, timeout, headers):
        seen_urls.append(url)
        return _FakeResponse()

    monkeypatch.setitem(sys.modules, "requests", _fake_requests_module(_get))
    monkeypatch.setattr(pd, "read_html", lambda content: [table])

    result = nfl_draft_sources.fetch_drafttek_big_board(season=2026)

    assert seen_urls == ["https://www.drafttek.com/2026-NFL-Draft-Big-Board/Top-NFL-Draft-Prospects-2026-Page-1.asp"]
    assert list(result["player_name"]) == ["Fernando Mendoza", "Jeremiyah Love"]
    assert list(result["rank"]) == [1.0, 2.0]
    assert (result["source_url"] == seen_urls[0]).all()


def test_fetch_drafttek_big_board_fetches_multiple_pages_and_concatenates(monkeypatch):
    page1 = pd.DataFrame([{"Rank": 1, "Name": "Player One"}])
    page2 = pd.DataFrame([{"Rank": 101, "Name": "Player Two"}])
    seen_urls = []

    def _get(url, timeout, headers):
        seen_urls.append(url)
        return _FakeResponse()

    def _read_html(content):
        # First call returns page1's table, second returns page2's.
        return [page1] if len(seen_urls) == 1 else [page2]

    monkeypatch.setitem(sys.modules, "requests", _fake_requests_module(_get))
    monkeypatch.setattr(pd, "read_html", _read_html)

    result = nfl_draft_sources.fetch_drafttek_big_board(season=2026, pages=2)

    assert seen_urls == [
        "https://www.drafttek.com/2026-NFL-Draft-Big-Board/Top-NFL-Draft-Prospects-2026-Page-1.asp",
        "https://www.drafttek.com/2026-NFL-Draft-Big-Board/Top-NFL-Draft-Prospects-2026-Page-2.asp",
    ]
    assert list(result["player_name"]) == ["Player One", "Player Two"]
    assert list(result["rank"]) == [1.0, 101.0]


def test_fetch_drafttek_big_board_keeps_pages_that_succeeded_when_another_page_fails(monkeypatch):
    page1 = pd.DataFrame([{"Rank": 1, "Name": "Player One"}])
    call_count = {"n": 0}

    def _get(url, timeout, headers):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise ConnectionError("real network failure")
        return _FakeResponse()

    monkeypatch.setitem(sys.modules, "requests", _fake_requests_module(_get))
    monkeypatch.setattr(pd, "read_html", lambda content: [page1])

    result = nfl_draft_sources.fetch_drafttek_big_board(season=2026, pages=2)

    # Page 1 succeeded, page 2 failed - a real partial board, not empty.
    assert list(result["player_name"]) == ["Player One"]


def test_fetch_drafttek_big_board_accepts_url_override_for_page_1(monkeypatch):
    table = pd.DataFrame([{"Rank": 1, "Name": "Fernando Mendoza"}])
    seen_urls = []

    def _get(url, timeout, headers):
        seen_urls.append(url)
        return _FakeResponse()

    monkeypatch.setitem(sys.modules, "requests", _fake_requests_module(_get))
    monkeypatch.setattr(pd, "read_html", lambda content: [table])

    override_url = "https://www.drafttek.com/some-other-real-path.asp"
    result = nfl_draft_sources.fetch_drafttek_big_board(url=override_url)

    assert seen_urls == [override_url]
    assert (result["source_url"] == override_url).all()


def test_fetch_drafttek_big_board_returns_empty_on_request_failure(monkeypatch):
    def _raise(url, timeout, headers):
        raise ConnectionError("real network failure")

    monkeypatch.setitem(sys.modules, "requests", _fake_requests_module(_raise))

    result = nfl_draft_sources.fetch_drafttek_big_board(season=2026)

    assert result.empty
    assert list(result.columns) == ["rank", "player_name"]


def test_fetch_fantasypros_big_board_normalizes_player_column_variant(monkeypatch):
    # Real, plausible variant: FantasyPros uses "Player" rather than "Name".
    table = pd.DataFrame([{"Rank": 1, "Player": "Fernando Mendoza"}])
    monkeypatch.setitem(
        sys.modules, "requests", _fake_requests_module(lambda url, timeout, headers: _FakeResponse())
    )
    monkeypatch.setattr(pd, "read_html", lambda content: [table])

    result = nfl_draft_sources.fetch_fantasypros_big_board()

    assert list(result["player_name"]) == ["Fernando Mendoza"]
    assert list(result["rank"]) == [1.0]


def test_fetch_fantasypros_big_board_accepts_url_override(monkeypatch):
    table = pd.DataFrame([{"Rank": 1, "Player": "Fernando Mendoza"}])
    seen_urls = []

    def _get(url, timeout, headers):
        seen_urls.append(url)
        return _FakeResponse()

    monkeypatch.setitem(sys.modules, "requests", _fake_requests_module(_get))
    monkeypatch.setattr(pd, "read_html", lambda content: [table])

    override_url = "https://www.fantasypros.com/nfl/rankings/prospects-overall.php"
    result = nfl_draft_sources.fetch_fantasypros_big_board(url=override_url)

    assert seen_urls == [override_url]
    assert (result["source_url"] == override_url).all()


def test_fetch_fantasypros_big_board_returns_empty_when_table_not_found(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "requests", _fake_requests_module(lambda url, timeout, headers: _FakeResponse())
    )
    monkeypatch.setattr(pd, "read_html", lambda content: [pd.DataFrame([{"Unrelated": 1}])])

    result = nfl_draft_sources.fetch_fantasypros_big_board()

    assert result.empty


def test_fetch_fantasypros_big_board_recovers_a_real_missing_header_row(monkeypatch):
    # Real, confirmed live behavior (2026-09-14 CI run): FantasyPros' one
    # real ranking table has no <th> cells, so pandas parses it with
    # plain integer columns and the real header text ends up as row 0.
    headerless = pd.DataFrame(
        [["Rank", "Player"], [1, "Fernando Mendoza"], [2, "Jeremiyah Love"]]
    )
    monkeypatch.setitem(
        sys.modules, "requests", _fake_requests_module(lambda url, timeout, headers: _FakeResponse())
    )
    monkeypatch.setattr(pd, "read_html", lambda content: [headerless])

    result = nfl_draft_sources.fetch_fantasypros_big_board()

    assert list(result["player_name"]) == ["Fernando Mendoza", "Jeremiyah Love"]
    assert list(result["rank"]) == [1.0, 2.0]


def test_promote_header_row_if_needed_leaves_a_real_headered_table_unchanged():
    table = pd.DataFrame([{"Rank": 1, "Player": "Real Player"}])

    result = nfl_draft_sources._promote_header_row_if_needed(table)

    pd.testing.assert_frame_equal(result, table)


def test_promote_header_row_if_needed_leaves_a_real_non_header_looking_row_unchanged():
    # All-integer columns, but row 0 doesn't look like a real header -
    # must not be mistaken for one.
    table = pd.DataFrame([[1, "Real Player"], [2, "Another Player"]])

    result = nfl_draft_sources._promote_header_row_if_needed(table)

    pd.testing.assert_frame_equal(result, table)


def test_read_html_tables_handles_a_real_raw_html_bytes_response():
    # Real, confirmed bug (2026-09-13 CI run): pd.read_html on this
    # project's pinned pandas raises FileNotFoundError when given raw
    # bytes/str directly - _read_html_tables must wrap it in a real
    # file-like object first.
    html = b"<!DOCTYPE html><html><body><table><tr><th>Rank</th><th>Name</th></tr>" \
           b"<tr><td>1</td><td>Real Player</td></tr></table></body></html>"

    tables = nfl_draft_sources._read_html_tables(html)

    assert len(tables) == 1
    assert list(tables[0]["Name"]) == ["Real Player"]


def test_fetch_all_sources_isolates_one_sources_failure_from_the_others(monkeypatch):
    def _working():
        return pd.DataFrame([{"rank": 1, "player_name": "Real Player"}])

    def _broken():
        raise RuntimeError("real failure")

    monkeypatch.setattr(nfl_draft_sources, "SOURCE_FETCHERS", {"Working": _working, "Broken": _broken})

    results = nfl_draft_sources.fetch_all_sources()

    assert list(results["Working"]["player_name"]) == ["Real Player"]
    assert results["Broken"].empty
