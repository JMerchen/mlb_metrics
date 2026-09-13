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


def test_fetch_nfl_mock_draft_database_consensus_normalizes_real_shape(monkeypatch):
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

    result = nfl_draft_sources.fetch_nfl_mock_draft_database_consensus(season=2026)

    assert seen_urls == ["https://www.nflmockdraftdatabase.com/big-boards/2026/consensus-big-board-2026"]
    assert list(result["player_name"]) == ["Fernando Mendoza", "Jeremiyah Love"]
    assert list(result["rank"]) == [1.0, 2.0]


def test_fetch_nfl_mock_draft_database_consensus_returns_empty_on_request_failure(monkeypatch):
    def _raise(url, timeout, headers):
        raise ConnectionError("real network failure")

    monkeypatch.setitem(sys.modules, "requests", _fake_requests_module(_raise))

    result = nfl_draft_sources.fetch_nfl_mock_draft_database_consensus(season=2026)

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


def test_fetch_all_sources_isolates_one_sources_failure_from_the_others(monkeypatch):
    def _working():
        return pd.DataFrame([{"rank": 1, "player_name": "Real Player"}])

    def _broken():
        raise RuntimeError("real failure")

    monkeypatch.setattr(nfl_draft_sources, "SOURCE_FETCHERS", {"Working": _working, "Broken": _broken})

    results = nfl_draft_sources.fetch_all_sources()

    assert list(results["Working"]["player_name"]) == ["Real Player"]
    assert results["Broken"].empty
