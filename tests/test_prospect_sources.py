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
    # Real, confirmed column name (2026-09-14 CI run): MLB.com's own real
    # page uses "Player", not "Name".
    batters = pd.DataFrame([{"Rk": "1", "Player": "Roman Anthony", "Pos": "OF"}])
    pitchers = pd.DataFrame([{"Rk": "1", "Player": "Andrew Painter", "Pos": "RHP"}])
    monkeypatch.setitem(
        sys.modules, "requests", _fake_requests_module(lambda url, timeout, headers: _FakeResponse())
    )
    monkeypatch.setattr(pd, "read_html", lambda content: [batters, pitchers])

    result = prospect_sources.fetch_mlb_pipeline_prospects()

    assert set(result["player_name"]) == {"Roman Anthony", "Andrew Painter"}
    assert list(result["rank"]) == [1.0, 1.0]
    assert (result["source_url"] == "https://www.mlb.com/prospects/stats/top-prospects").all()


def test_fetch_mlb_pipeline_prospects_extracts_team_and_level_and_drops_raw_stat_columns(monkeypatch):
    # Real, confirmed bug (2026-09-16 user report): MLB Pipeline's real
    # page is a full stats table (PA/AB/HR/ERA/WHIP/...), and it used to
    # carry ALL of those raw columns straight through - once merged with
    # other sources by consensus_rankings.build_consensus_ranking, this
    # produced a huge, sparse, confusing column list. Only a fixed,
    # curated set of real scouting/roster fields should survive.
    batters = pd.DataFrame([
        {"Rk": "1", "Player": "Roman Anthony", "Tm": "BOS", "Level": "AAA", "PA": 500, "HR": 20},
    ])
    pitchers = pd.DataFrame([{"Rk": "2", "Player": "Andrew Painter", "Tm": "PHI", "Level": "AA", "ERA": 2.5, "L": 3}])
    monkeypatch.setitem(
        sys.modules, "requests", _fake_requests_module(lambda url, timeout, headers: _FakeResponse())
    )
    monkeypatch.setattr(pd, "read_html", lambda content: [batters, pitchers])

    result = prospect_sources.fetch_mlb_pipeline_prospects()

    assert set(result.columns) == {"rank", "player_name", "team", "position", "highest_level", "age", "eta", "fv", "source_url"}
    by_name = result.set_index("player_name")
    assert by_name.loc["Roman Anthony", "team"] == "BOS"
    assert by_name.loc["Roman Anthony", "highest_level"] == "AAA"
    assert by_name.loc["Andrew Painter", "team"] == "PHI"


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


def test_fetch_fangraphs_prospects_normalizes_rank_name_shape(monkeypatch):
    table = pd.DataFrame([
        {"Rank": 1, "Name": "Roman Anthony"},
        {"Rank": 2, "Name": "Andrew Painter"},
    ])
    monkeypatch.setitem(
        sys.modules, "requests", _fake_requests_module(lambda url, timeout, headers: _FakeResponse())
    )
    monkeypatch.setattr(pd, "read_html", lambda content: [table])

    result = prospect_sources.fetch_fangraphs_prospects(season=2026)

    assert list(result["player_name"]) == ["Roman Anthony", "Andrew Painter"]
    assert list(result["rank"]) == [1.0, 2.0]
    assert (result["source_url"] == "https://blogs.fangraphs.com/2026-top-100-prospects/").all()


def test_fetch_fangraphs_prospects_accepts_url_override(monkeypatch):
    table = pd.DataFrame([{"Rank": 1, "Name": "Roman Anthony"}])
    seen_urls = []

    def _get(url, timeout, headers):
        seen_urls.append(url)
        return _FakeResponse()

    monkeypatch.setitem(sys.modules, "requests", _fake_requests_module(_get))
    monkeypatch.setattr(pd, "read_html", lambda content: [table])

    override_url = "https://www.fangraphs.com/some-other-real-path/"
    result = prospect_sources.fetch_fangraphs_prospects(url=override_url)

    assert seen_urls == [override_url]
    assert (result["source_url"] == override_url).all()


def test_fetch_fangraphs_prospects_returns_empty_when_table_not_found(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "requests", _fake_requests_module(lambda url, timeout, headers: _FakeResponse())
    )
    monkeypatch.setattr(pd, "read_html", lambda content: [pd.DataFrame([{"Unrelated": 1}])])

    result = prospect_sources.fetch_fangraphs_prospects(season=2026)

    assert result.empty


def test_fetch_fangraphs_prospects_returns_empty_on_request_failure(monkeypatch):
    def _raise(url, timeout, headers):
        raise ConnectionError("real network failure")

    monkeypatch.setitem(sys.modules, "requests", _fake_requests_module(_raise))

    result = prospect_sources.fetch_fangraphs_prospects(season=2026)

    assert result.empty


def test_fetch_fangraphs_prospects_normalizes_real_confirmed_rk_name_columns(monkeypatch):
    # Real, confirmed live behavior (2026-09-15 CI run, 2nd trigger
    # with the corrected blogs.fangraphs.com URL): the real ranking
    # table's real header row is "Rk"/"Name", not the friendlier-cased
    # "Rank" guessed first.
    table = pd.DataFrame([
        {"Rk": 1, "Name": "Jesus Made", "Team": "MIL"},
        {"Rk": 2, "Name": "Franklin Arias", "Team": "SEA"},
    ])
    monkeypatch.setitem(
        sys.modules, "requests", _fake_requests_module(lambda url, timeout, headers: _FakeResponse())
    )
    monkeypatch.setattr(pd, "read_html", lambda content: [table])

    result = prospect_sources.fetch_fangraphs_prospects(season=2026)

    assert list(result["player_name"]) == ["Jesus Made", "Franklin Arias"]
    assert list(result["rank"]) == [1.0, 2.0]


def test_strip_sortable_header_suffix_recovers_real_just_baseball_headers():
    # Real, confirmed live behavior (2026-09-15 CI run): Just Baseball's
    # real ranking table's real <th> cells carry a visible
    # "Sort by <field>" accessibility label appended, so pd.read_html
    # parses the whole real string as the column name.
    table = pd.DataFrame([
        {
            "Rank Sort by rank": 1,
            "Player Sort by player": "Jesus Made",
            "Team Sort by team": "MIL",
            "Level": "AA",
        }
    ])

    result = prospect_sources._strip_sortable_header_suffix(table)

    assert list(result.columns) == ["Rank", "Player", "Team", "Level"]


def test_strip_sortable_header_suffix_rededuplicates_a_real_collapsed_duplicate_rank_column():
    # Real, confirmed CRASH (2026-09-15 live CI run, exit code 1, took
    # down the entire script - not a hypothetical): Just Baseball's real
    # table has TWO real "Rank"-prefixed columns, already made distinct
    # by pandas' own ".1" dedup suffix ("Rank Sort by rank" and
    # "Rank Sort by rank.1"). Stripping at " Sort by" collapsed BOTH to
    # the identical plain "Rank", so table.rename(columns={"Rank": ...})
    # renamed BOTH to "rank", making result["rank"] a real DataFrame
    # instead of a Series - pd.to_numeric then raised
    # "TypeError: arg must be a list, tuple, 1-d array, or Series".
    table = pd.DataFrame(
        [[1, 1, "Jesus Made"]],
        columns=["Rank Sort by rank", "Rank Sort by rank.1", "Player Sort by player"],
    )

    result = prospect_sources._strip_sortable_header_suffix(table)

    assert list(result.columns) == ["Rank", "Rank.1", "Player"]


def test_fetch_just_baseball_prospects_recovers_real_sort_by_header_suffix(monkeypatch):
    # Real, confirmed live behavior (2026-09-15 CI run): same real
    # "Sort by <field>" header quirk, exercised end to end through the
    # real fetcher.
    table = pd.DataFrame([
        {"Rank Sort by rank": 1, "Player Sort by player": "Jesus Made"},
        {"Rank Sort by rank": 2, "Player Sort by player": "Franklin Arias"},
    ])
    monkeypatch.setitem(
        sys.modules, "requests", _fake_requests_module(lambda url, timeout, headers: _FakeResponse())
    )
    monkeypatch.setattr(pd, "read_html", lambda content: [table])

    result = prospect_sources.fetch_just_baseball_prospects()

    assert list(result["player_name"]) == ["Jesus Made", "Franklin Arias"]


def test_fetch_just_baseball_prospects_does_not_crash_on_a_real_duplicate_rank_column(monkeypatch):
    # Real, confirmed CRASH reproduced end to end through the real
    # fetcher (2026-09-15 live CI run, exit code 1) - see
    # test_strip_sortable_header_suffix_rededuplicates_a_real_collapsed_duplicate_rank_column
    # for the full real root cause.
    table = pd.DataFrame(
        [
            [1, 1, "Jesus Made"],
            [2, 2, "Franklin Arias"],
        ],
        columns=["Rank Sort by rank", "Rank Sort by rank.1", "Player Sort by player"],
    )
    monkeypatch.setitem(
        sys.modules, "requests", _fake_requests_module(lambda url, timeout, headers: _FakeResponse())
    )
    monkeypatch.setattr(pd, "read_html", lambda content: [table])

    result = prospect_sources.fetch_just_baseball_prospects()

    assert list(result["player_name"]) == ["Jesus Made", "Franklin Arias"]
    assert list(result["rank"]) == [1.0, 2.0]
    assert list(result["rank"]) == [1.0, 2.0]


def test_fetch_cbs_sports_prospects_normalizes_rank_player_shape(monkeypatch):
    table = pd.DataFrame([
        {"Rank": 1, "Player": "Roman Anthony"},
        {"Rank": 2, "Player": "Andrew Painter"},
    ])
    monkeypatch.setitem(
        sys.modules, "requests", _fake_requests_module(lambda url, timeout, headers: _FakeResponse())
    )
    monkeypatch.setattr(pd, "read_html", lambda content: [table])

    result = prospect_sources.fetch_cbs_sports_prospects(season=2026)

    assert list(result["player_name"]) == ["Roman Anthony", "Andrew Painter"]
    assert list(result["rank"]) == [1.0, 2.0]
    assert (
        result["source_url"]
        == "https://www.cbssports.com/fantasy/baseball/news/fantasy-baseball-top-100-prospects-for-2026/"
    ).all()


def test_fetch_cbs_sports_prospects_accepts_url_override(monkeypatch):
    table = pd.DataFrame([{"Rank": 1, "Player": "Roman Anthony"}])
    seen_urls = []

    def _get(url, timeout, headers):
        seen_urls.append(url)
        return _FakeResponse()

    monkeypatch.setitem(sys.modules, "requests", _fake_requests_module(_get))
    monkeypatch.setattr(pd, "read_html", lambda content: [table])

    override_url = "https://www.cbssports.com/some-other-real-path/"
    result = prospect_sources.fetch_cbs_sports_prospects(url=override_url)

    assert seen_urls == [override_url]
    assert (result["source_url"] == override_url).all()


def test_fetch_cbs_sports_prospects_returns_empty_when_table_not_found(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "requests", _fake_requests_module(lambda url, timeout, headers: _FakeResponse())
    )
    monkeypatch.setattr(pd, "read_html", lambda content: [pd.DataFrame([{"Unrelated": 1}])])

    result = prospect_sources.fetch_cbs_sports_prospects()

    assert result.empty


def test_fetch_cbs_sports_prospects_returns_empty_on_request_failure(monkeypatch):
    def _raise(url, timeout, headers):
        raise ConnectionError("real network failure")

    monkeypatch.setitem(sys.modules, "requests", _fake_requests_module(_raise))

    result = prospect_sources.fetch_cbs_sports_prospects()

    assert result.empty


def test_fetch_prospects_live_rankings_normalizes_rank_name_shape(monkeypatch):
    table = pd.DataFrame([
        {"Rank": 1, "Name": "Roman Anthony"},
        {"Rank": 2, "Name": "Andrew Painter"},
    ])
    monkeypatch.setitem(
        sys.modules, "requests", _fake_requests_module(lambda url, timeout, headers: _FakeResponse())
    )
    monkeypatch.setattr(pd, "read_html", lambda content: [table])

    result = prospect_sources.fetch_prospects_live_rankings(season=2026)

    assert list(result["player_name"]) == ["Roman Anthony", "Andrew Painter"]
    assert list(result["rank"]) == [1.0, 2.0]
    assert (result["source_url"] == "https://www.prospectslive.com/2026-top-100-prospects/").all()


def test_fetch_prospects_live_rankings_accepts_url_override(monkeypatch):
    table = pd.DataFrame([{"Rank": 1, "Name": "Roman Anthony"}])
    seen_urls = []

    def _get(url, timeout, headers):
        seen_urls.append(url)
        return _FakeResponse()

    monkeypatch.setitem(sys.modules, "requests", _fake_requests_module(_get))
    monkeypatch.setattr(pd, "read_html", lambda content: [table])

    override_url = "https://prospectslive.com/some-other-real-path"
    result = prospect_sources.fetch_prospects_live_rankings(url=override_url)

    assert seen_urls == [override_url]
    assert (result["source_url"] == override_url).all()


def test_fetch_prospects_live_rankings_returns_empty_when_table_not_found(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "requests", _fake_requests_module(lambda url, timeout, headers: _FakeResponse())
    )
    monkeypatch.setattr(pd, "read_html", lambda content: [pd.DataFrame([{"Unrelated": 1}])])

    result = prospect_sources.fetch_prospects_live_rankings()

    assert result.empty


def test_fetch_prospects_live_rankings_returns_empty_on_request_failure(monkeypatch):
    def _raise(url, timeout, headers):
        raise ConnectionError("real network failure")

    monkeypatch.setitem(sys.modules, "requests", _fake_requests_module(_raise))

    result = prospect_sources.fetch_prospects_live_rankings()

    assert result.empty


def test_fetch_just_baseball_prospects_normalizes_rank_name_shape(monkeypatch):
    table = pd.DataFrame([
        {"Rank": 1, "Name": "Roman Anthony"},
        {"Rank": 2, "Name": "Andrew Painter"},
    ])
    monkeypatch.setitem(
        sys.modules, "requests", _fake_requests_module(lambda url, timeout, headers: _FakeResponse())
    )
    monkeypatch.setattr(pd, "read_html", lambda content: [table])

    result = prospect_sources.fetch_just_baseball_prospects()

    assert list(result["player_name"]) == ["Roman Anthony", "Andrew Painter"]
    assert list(result["rank"]) == [1.0, 2.0]
    assert (
        result["source_url"] == "https://www.justbaseball.com/prospects/top-100-mlb-prospects/"
    ).all()


def test_fetch_just_baseball_prospects_accepts_url_override(monkeypatch):
    table = pd.DataFrame([{"Rank": 1, "Name": "Roman Anthony"}])
    seen_urls = []

    def _get(url, timeout, headers):
        seen_urls.append(url)
        return _FakeResponse()

    monkeypatch.setitem(sys.modules, "requests", _fake_requests_module(_get))
    monkeypatch.setattr(pd, "read_html", lambda content: [table])

    override_url = "https://www.justbaseball.com/some-other-real-path/"
    result = prospect_sources.fetch_just_baseball_prospects(url=override_url)

    assert seen_urls == [override_url]
    assert (result["source_url"] == override_url).all()


def test_fetch_just_baseball_prospects_extracts_enrichment_fields_and_drops_the_duplicate_rank_column(monkeypatch):
    # Real, confirmed bug (2026-09-16 user report): Just Baseball's real
    # page shape is `['Rank', 'Rank.1', 'Player', 'Team', 'Age', 'Level',
    # 'Position', 'ETA', 'FV']` (see _strip_sortable_header_suffix's own
    # docstring for the real duplicate-Rank-column root cause) - the
    # genuinely redundant `Rank.1` column used to leak straight into the
    # committed board as a second, confusing rank-looking column.
    table = pd.DataFrame([
        {
            "Rank": 1, "Rank.1": 1, "Player": "Jesus Made", "Team": "Milwaukee Brewers",
            "Age": 19, "Level": "AA", "Position": "SS", "ETA": 2026, "FV": "70",
        }
    ])
    monkeypatch.setitem(
        sys.modules, "requests", _fake_requests_module(lambda url, timeout, headers: _FakeResponse())
    )
    monkeypatch.setattr(pd, "read_html", lambda content: [table])

    result = prospect_sources.fetch_just_baseball_prospects()

    assert "Rank.1" not in result.columns
    row = result.iloc[0]
    assert row["team"] == "Milwaukee Brewers"
    assert row["age"] == 19
    assert row["highest_level"] == "AA"
    assert row["position"] == "SS"
    assert row["eta"] == 2026
    assert row["fv"] == "70"


def test_fetch_just_baseball_prospects_returns_empty_when_table_not_found(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "requests", _fake_requests_module(lambda url, timeout, headers: _FakeResponse())
    )
    monkeypatch.setattr(pd, "read_html", lambda content: [pd.DataFrame([{"Unrelated": 1}])])

    result = prospect_sources.fetch_just_baseball_prospects()

    assert result.empty


def test_fetch_just_baseball_prospects_returns_empty_on_request_failure(monkeypatch):
    def _raise(url, timeout, headers):
        raise ConnectionError("real network failure")

    monkeypatch.setitem(sys.modules, "requests", _fake_requests_module(_raise))

    result = prospect_sources.fetch_just_baseball_prospects()

    assert result.empty


def test_fetch_tjstats_prospects_normalizes_rank_name_shape(monkeypatch):
    table = pd.DataFrame([
        {"Rank": 1, "Name": "Roman Anthony"},
        {"Rank": 2, "Name": "Andrew Painter"},
    ])
    monkeypatch.setitem(
        sys.modules, "requests", _fake_requests_module(lambda url, timeout, headers: _FakeResponse())
    )
    monkeypatch.setattr(pd, "read_html", lambda content: [table])

    result = prospect_sources.fetch_tjstats_prospects()

    assert list(result["player_name"]) == ["Roman Anthony", "Andrew Painter"]
    assert list(result["rank"]) == [1.0, 2.0]


def test_fetch_tjstats_prospects_accepts_url_override(monkeypatch):
    table = pd.DataFrame([{"Rank": 1, "Name": "Roman Anthony"}])
    seen_urls = []

    def _get(url, timeout, headers):
        seen_urls.append(url)
        return _FakeResponse()

    monkeypatch.setitem(sys.modules, "requests", _fake_requests_module(_get))
    monkeypatch.setattr(pd, "read_html", lambda content: [table])

    override_url = "https://tjstats.ca/some-other-real-path/"
    result = prospect_sources.fetch_tjstats_prospects(url=override_url)

    assert seen_urls == [override_url]
    assert (result["source_url"] == override_url).all()


def test_fetch_tjstats_prospects_returns_empty_when_table_not_found(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "requests", _fake_requests_module(lambda url, timeout, headers: _FakeResponse())
    )
    monkeypatch.setattr(pd, "read_html", lambda content: [pd.DataFrame([{"Unrelated": 1}])])

    result = prospect_sources.fetch_tjstats_prospects()

    assert result.empty


def test_fetch_tjstats_prospects_returns_empty_on_request_failure(monkeypatch):
    def _raise(url, timeout, headers):
        raise ConnectionError("real network failure")

    monkeypatch.setitem(sys.modules, "requests", _fake_requests_module(_raise))

    result = prospect_sources.fetch_tjstats_prospects()

    assert result.empty


def test_filter_undebuted_drops_rows_with_mlb_highest_level():
    # Real, confirmed user report (2026-09-16): "MLB prospects are meant
    # to be those undebuted, but some have already reached the majors" -
    # confirmed live via docs/data/prospect_rankings.csv (31 of 173 real
    # rows had highest_level == "MLB").
    df = pd.DataFrame([
        {"player_name": "Still Minors", "rank": 1, "highest_level": "AAA"},
        {"player_name": "Called Up", "rank": 2, "highest_level": "MLB"},
        {"player_name": "Called Up Lowercase", "rank": 3, "highest_level": "mlb"},
    ])

    result = prospect_sources.filter_undebuted(df)

    assert list(result["player_name"]) == ["Still Minors"]


def test_filter_undebuted_keeps_rows_with_unknown_level():
    # A source that doesn't publish highest_level at all (e.g. MLB
    # Pipeline's own real shape for some players) is real, honest
    # "unknown" - not evidence of having debuted, so it must be kept.
    df = pd.DataFrame([
        {"player_name": "Unknown Level", "rank": 1, "highest_level": None},
    ])

    result = prospect_sources.filter_undebuted(df)

    assert list(result["player_name"]) == ["Unknown Level"]


def test_filter_undebuted_is_a_noop_when_the_column_is_missing_entirely():
    df = pd.DataFrame([{"player_name": "No Level Column", "rank": 1}])

    result = prospect_sources.filter_undebuted(df)

    assert list(result["player_name"]) == ["No Level Column"]


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
