import sys
import types

import pandas as pd

from mlb_metrics import prospect_sources


def _fake_requests_module(get_fn):
    fake = types.ModuleType("requests")
    fake.get = get_fn
    return fake


class _FakeResponse:
    content = b"<!DOCTYPE html><html></html>"

    def raise_for_status(self):
        pass


def test_fetch_mlb_pipeline_prospects_normalizes_real_shape(monkeypatch):
    batters = pd.DataFrame([{"Rk": "1", "Name": "Roman Anthony", "Pos": "OF"}])
    pitchers = pd.DataFrame([{"Rk": "1", "Name": "Andrew Painter", "Pos": "RHP"}])
    monkeypatch.setitem(
        sys.modules, "requests", _fake_requests_module(lambda url, timeout, headers: _FakeResponse())
    )
    monkeypatch.setattr(pd, "read_html", lambda content: [batters, pitchers])

    result = prospect_sources.fetch_mlb_pipeline_prospects()

    assert set(result["player_name"]) == {"Roman Anthony", "Andrew Painter"}
    assert list(result["rank"]) == [1.0, 1.0]
    assert (result["source_url"] == "https://www.mlb.com/prospects/stats/top-prospects").all()


def test_fetch_mlb_pipeline_prospects_returns_empty_on_request_failure(monkeypatch):
    def _raise(url, timeout, headers):
        raise ConnectionError("real network failure")

    monkeypatch.setitem(sys.modules, "requests", _fake_requests_module(_raise))

    result = prospect_sources.fetch_mlb_pipeline_prospects()

    assert result.empty
    assert list(result.columns) == ["rank", "player_name"]


def test_fetch_mlb_pipeline_prospects_returns_empty_on_too_few_tables(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "requests", _fake_requests_module(lambda url, timeout, headers: _FakeResponse())
    )
    monkeypatch.setattr(pd, "read_html", lambda content: [pd.DataFrame([{"Rk": 1, "Name": "Solo Table"}])])

    result = prospect_sources.fetch_mlb_pipeline_prospects()

    assert result.empty


def test_fetch_mlb_pipeline_prospects_returns_empty_on_unexpected_shape(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "requests", _fake_requests_module(lambda url, timeout, headers: _FakeResponse())
    )
    monkeypatch.setattr(
        pd, "read_html", lambda content: [pd.DataFrame([{"Unexpected": "shape"}]), pd.DataFrame([{"Unexpected": "shape"}])]
    )

    result = prospect_sources.fetch_mlb_pipeline_prospects()

    assert result.empty


def test_fetch_baseball_america_prospects_normalizes_real_shape(monkeypatch):
    table = pd.DataFrame([
        {"Rank": 1, "Name": "Ethan Salas"},
        {"Rank": 2, "Name": "Konnor Griffin"},
    ])
    monkeypatch.setitem(
        sys.modules, "requests", _fake_requests_module(lambda url, timeout, headers: _FakeResponse())
    )
    monkeypatch.setattr(pd, "read_html", lambda content: [table])

    result = prospect_sources.fetch_baseball_america_prospects(season=2026)

    assert list(result["player_name"]) == ["Ethan Salas", "Konnor Griffin"]
    assert list(result["rank"]) == [1.0, 2.0]
    assert (result["source_url"] == "https://www.baseballamerica.com/rankings/2026-top-100-prospects/").all()


def test_fetch_baseball_america_prospects_returns_empty_when_table_not_found(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "requests", _fake_requests_module(lambda url, timeout, headers: _FakeResponse())
    )
    monkeypatch.setattr(pd, "read_html", lambda content: [pd.DataFrame([{"Unrelated": 1}])])

    result = prospect_sources.fetch_baseball_america_prospects(season=2026)

    assert result.empty


def test_fetch_baseball_america_prospects_returns_empty_on_request_failure(monkeypatch):
    def _raise(url, timeout, headers):
        raise ConnectionError("real network failure")

    monkeypatch.setitem(sys.modules, "requests", _fake_requests_module(_raise))

    result = prospect_sources.fetch_baseball_america_prospects(season=2026)

    assert result.empty


def test_fetch_baseball_america_prospects_returns_empty_on_http_error(monkeypatch):
    # Real, confirmed live behavior (2026-09-13 CI run): Baseball America
    # returns a real 403 Forbidden - raise_for_status must surface that
    # as a real failure, not silently proceed to parse an error page.
    class _ForbiddenResponse:
        content = b"<!DOCTYPE html><html>Forbidden</html>"

        def raise_for_status(self):
            raise RuntimeError("403 Client Error: Forbidden")

    monkeypatch.setitem(
        sys.modules, "requests", _fake_requests_module(lambda url, timeout, headers: _ForbiddenResponse())
    )

    result = prospect_sources.fetch_baseball_america_prospects(season=2026)

    assert result.empty


def test_read_html_tables_handles_a_real_raw_html_bytes_response():
    # Real, confirmed bug (2026-09-13 CI run): pd.read_html on this
    # project's pinned pandas raises FileNotFoundError when given raw
    # bytes/str directly - _read_html_tables must wrap it in a real
    # file-like object first.
    html = b"<!DOCTYPE html><html><body><table><tr><th>Rank</th><th>Name</th></tr>" \
           b"<tr><td>1</td><td>Real Player</td></tr></table></body></html>"

    tables = prospect_sources._read_html_tables(html)

    assert len(tables) == 1
    assert list(tables[0]["Name"]) == ["Real Player"]


def test_fetch_all_sources_isolates_one_sources_failure_from_the_others(monkeypatch):
    def _working():
        return pd.DataFrame([{"rank": 1, "player_name": "Real Player"}])

    def _broken():
        raise RuntimeError("real failure")

    monkeypatch.setattr(prospect_sources, "SOURCE_FETCHERS", {"Working": _working, "Broken": _broken})

    results = prospect_sources.fetch_all_sources()

    assert list(results["Working"]["player_name"]) == ["Real Player"]
    assert results["Broken"].empty
