import sys
import types

import pandas as pd

from mlb_metrics import mlb_draft_sources


def _fake_requests_module(get_fn):
    fake = types.ModuleType("requests")
    fake.get = get_fn
    return fake


class _FakeResponse:
    content = b"<html></html>"

    def raise_for_status(self):
        pass


def test_fetch_baseball_america_college_draft_normalizes_real_shape(monkeypatch):
    table = pd.DataFrame([
        {"Rank": 1, "Name": "Roch Cholowsky"},
        {"Rank": 2, "Name": "Grady Emerson"},
    ])
    monkeypatch.setitem(
        sys.modules, "requests", _fake_requests_module(lambda url, timeout, headers: _FakeResponse())
    )
    monkeypatch.setattr(pd, "read_html", lambda content: [table])

    result = mlb_draft_sources.fetch_baseball_america_college_draft(season=2026)

    assert list(result["player_name"]) == ["Roch Cholowsky", "Grady Emerson"]
    assert list(result["rank"]) == [1.0, 2.0]
    assert (result["source_url"] == "https://www.baseballamerica.com/rankings/2026-top-college-draft-prospects/").all()


def test_fetch_baseball_america_college_draft_returns_empty_on_request_failure(monkeypatch):
    def _raise(url, timeout, headers):
        raise ConnectionError("real network failure")

    monkeypatch.setitem(sys.modules, "requests", _fake_requests_module(_raise))

    result = mlb_draft_sources.fetch_baseball_america_college_draft(season=2026)

    assert result.empty
    assert list(result.columns) == ["rank", "player_name"]


def test_fetch_d1baseball_college_draft_uses_default_url_when_none_given(monkeypatch):
    table = pd.DataFrame([{"Rk": 1, "Name": "Roch Cholowsky"}])
    seen_urls = []

    def _get(url, timeout, headers):
        seen_urls.append(url)
        return _FakeResponse()

    monkeypatch.setitem(sys.modules, "requests", _fake_requests_module(_get))
    monkeypatch.setattr(pd, "read_html", lambda content: [table])

    result = mlb_draft_sources.fetch_d1baseball_college_draft(season=2026)

    assert seen_urls == ["https://d1baseball.com/prospects/2026-mlb-draft-top-100-college-prospects/"]
    assert list(result["player_name"]) == ["Roch Cholowsky"]


def test_fetch_d1baseball_college_draft_accepts_url_override_for_a_real_broken_default(monkeypatch):
    # Real, confirmed fragility (see module docstring): D1Baseball's real
    # URL has already carried a numeric revision suffix - `url` exists so
    # a caller can point at the CURRENT real page without a code change.
    table = pd.DataFrame([{"Rk": 1, "Name": "Roch Cholowsky"}])
    seen_urls = []

    def _get(url, timeout, headers):
        seen_urls.append(url)
        return _FakeResponse()

    monkeypatch.setitem(sys.modules, "requests", _fake_requests_module(_get))
    monkeypatch.setattr(pd, "read_html", lambda content: [table])

    override_url = "https://d1baseball.com/prospects/2026-mlb-draft-top-100-college-prospects-2/"
    result = mlb_draft_sources.fetch_d1baseball_college_draft(url=override_url)

    assert seen_urls == [override_url]
    assert (result["source_url"] == override_url).all()


def test_fetch_all_sources_isolates_one_sources_failure_from_the_others(monkeypatch):
    def _working():
        return pd.DataFrame([{"rank": 1, "player_name": "Real Player"}])

    def _broken():
        raise RuntimeError("real failure")

    monkeypatch.setattr(mlb_draft_sources, "SOURCE_FETCHERS", {"Working": _working, "Broken": _broken})

    results = mlb_draft_sources.fetch_all_sources()

    assert list(results["Working"]["player_name"]) == ["Real Player"]
    assert results["Broken"].empty
