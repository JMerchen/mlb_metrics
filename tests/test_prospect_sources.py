import sys
import types

import pandas as pd

from mlb_metrics import prospect_sources


def test_fetch_mlb_pipeline_prospects_normalizes_real_shape(monkeypatch):
    raw = pd.DataFrame([
        {"Rk": "1", "Name": "Roman Anthony", "Pos": "OF"},
        {"Rk": "2", "Name": "Jackson Holliday", "Pos": "2B"},
    ])
    fake_module = types.ModuleType("pybaseball")
    fake_module.top_prospects = lambda: raw
    monkeypatch.setitem(sys.modules, "pybaseball", fake_module)

    result = prospect_sources.fetch_mlb_pipeline_prospects()

    assert list(result["player_name"]) == ["Roman Anthony", "Jackson Holliday"]
    assert list(result["rank"]) == [1.0, 2.0]
    assert (result["source_url"] == "https://www.mlb.com/prospects/stats/top-prospects").all()


def test_fetch_mlb_pipeline_prospects_returns_empty_on_exception(monkeypatch):
    fake_module = types.ModuleType("pybaseball")

    def _raise():
        raise RuntimeError("real network failure")

    fake_module.top_prospects = _raise
    monkeypatch.setitem(sys.modules, "pybaseball", fake_module)

    result = prospect_sources.fetch_mlb_pipeline_prospects()

    assert result.empty
    assert list(result.columns) == ["rank", "player_name"]


def test_fetch_mlb_pipeline_prospects_returns_empty_on_unexpected_shape(monkeypatch):
    fake_module = types.ModuleType("pybaseball")
    fake_module.top_prospects = lambda: pd.DataFrame([{"Unexpected": "shape"}])
    monkeypatch.setitem(sys.modules, "pybaseball", fake_module)

    result = prospect_sources.fetch_mlb_pipeline_prospects()

    assert result.empty


def test_fetch_baseball_america_prospects_normalizes_real_shape(monkeypatch):
    table = pd.DataFrame([
        {"Rank": 1, "Name": "Ethan Salas"},
        {"Rank": 2, "Name": "Konnor Griffin"},
    ])

    class _FakeResponse:
        content = b"<html></html>"

        def raise_for_status(self):
            pass

    fake_requests = types.ModuleType("requests")
    fake_requests.get = lambda url, timeout, headers: _FakeResponse()
    monkeypatch.setitem(sys.modules, "requests", fake_requests)
    monkeypatch.setattr(pd, "read_html", lambda content: [table])

    result = prospect_sources.fetch_baseball_america_prospects(season=2026)

    assert list(result["player_name"]) == ["Ethan Salas", "Konnor Griffin"]
    assert list(result["rank"]) == [1.0, 2.0]
    assert (result["source_url"] == "https://www.baseballamerica.com/rankings/2026-top-100-prospects/").all()


def test_fetch_baseball_america_prospects_returns_empty_when_table_not_found(monkeypatch):
    class _FakeResponse:
        content = b"<html></html>"

        def raise_for_status(self):
            pass

    fake_requests = types.ModuleType("requests")
    fake_requests.get = lambda url, timeout, headers: _FakeResponse()
    monkeypatch.setitem(sys.modules, "requests", fake_requests)
    monkeypatch.setattr(pd, "read_html", lambda content: [pd.DataFrame([{"Unrelated": 1}])])

    result = prospect_sources.fetch_baseball_america_prospects(season=2026)

    assert result.empty


def test_fetch_baseball_america_prospects_returns_empty_on_request_failure(monkeypatch):
    fake_requests = types.ModuleType("requests")

    def _raise(url, timeout, headers):
        raise ConnectionError("real network failure")

    fake_requests.get = _raise
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    result = prospect_sources.fetch_baseball_america_prospects(season=2026)

    assert result.empty


def test_fetch_all_sources_isolates_one_sources_failure_from_the_others(monkeypatch):
    def _working():
        return pd.DataFrame([{"rank": 1, "player_name": "Real Player"}])

    def _broken():
        raise RuntimeError("real failure")

    monkeypatch.setattr(prospect_sources, "SOURCE_FETCHERS", {"Working": _working, "Broken": _broken})

    results = prospect_sources.fetch_all_sources()

    assert list(results["Working"]["player_name"]) == ["Real Player"]
    assert results["Broken"].empty
