"""Unit tests for market_odds.py's real ingestion logic - quant-analytics
item #6, slice 2. fetch_scoreboard/fetch_summary do real network and are
deliberately not tested here (same split schedule.py uses for
_fetch_raw_schedule); everything else is pure parsing/math and gets
hand-computed/fixture-based coverage. Fixtures mirror the real JSON shape
quant-analytics item #6 slice 1's confirmation dispatch actually printed
(GitHub Actions run 32516808493, 2026-08-21), and the de-vigged
probabilities below are the REAL values that same dispatch printed, not
independently re-derived - an integration-style check that the real
ingestion logic reproduces the real already-confirmed output."""

import pytest

from mlb_metrics.market_odds import (
    ESPN_TEAM_ABBREV_FIXUPS,
    _apply_abbrev_fixup,
    _extract_market_row,
    _parse_pickcenter_row,
    devig,
    moneyline_to_implied_probability,
)


def _pickcenter_row(provider, home_ml, away_ml):
    return {
        "provider": {"id": "100", "name": provider},
        "details": f"{provider} line",
        "awayTeamOdds": {"moneyLine": away_ml},
        "homeTeamOdds": {"moneyLine": home_ml},
    }


def _summary(pickcenter=None):
    return {"pickcenter": pickcenter if pickcenter is not None else []}


def _event(home_abbrev, away_abbrev, event_id="401695000"):
    return {
        "id": event_id,
        "competitions": [
            {
                "competitors": [
                    {"homeAway": "home", "team": {"abbreviation": home_abbrev}},
                    {"homeAway": "away", "team": {"abbreviation": away_abbrev}},
                ]
            }
        ],
    }


def test_moneyline_to_implied_probability_negative_favorite():
    # A clean, hand-verifiable case: -150 -> 150/250 == 0.6 exactly.
    assert moneyline_to_implied_probability(-150) == pytest.approx(0.6)


def test_moneyline_to_implied_probability_positive_underdog():
    # +200 -> 100/300 == 1/3 exactly.
    assert moneyline_to_implied_probability(200) == pytest.approx(1 / 3)


def test_devig_removes_the_bookmakers_overround():
    # Two-sided vig example: implied probabilities sum to > 1.0 before
    # de-vigging (the real vig), exactly 1.0 after.
    home_implied = moneyline_to_implied_probability(-150)
    away_implied = moneyline_to_implied_probability(130)
    assert home_implied + away_implied > 1.0
    home_devigged = devig(home_implied, away_implied)
    away_devigged = devig(away_implied, home_implied)
    assert home_devigged + away_devigged == pytest.approx(1.0)


def test_espn_team_abbrev_fixups_match_the_real_confirmed_crosswalk():
    # The exact real mismatches quant-analytics item #6 slice 1's
    # dispatch found - all other 28 real ESPN abbreviations matched
    # schedule.TEAM_ID_TO_ABBREV's values exactly, so those need no fixup.
    assert ESPN_TEAM_ABBREV_FIXUPS == {"ARI": "AZ", "CHW": "CWS"}
    assert _apply_abbrev_fixup("ARI") == "AZ"
    assert _apply_abbrev_fixup("CHW") == "CWS"
    assert _apply_abbrev_fixup("NYY") == "NYY"


def test_parse_pickcenter_row_prefers_the_configured_provider():
    summary = _summary([_pickcenter_row("FanDuel", -110, -110), _pickcenter_row("DraftKings", -193, 179)])
    provider, home_ml, away_ml = _parse_pickcenter_row(summary)
    assert provider == "DraftKings"
    assert home_ml == -193
    assert away_ml == 179


def test_parse_pickcenter_row_falls_back_to_first_row_if_preferred_provider_absent():
    summary = _summary([_pickcenter_row("FanDuel", -110, -110)])
    provider, home_ml, away_ml = _parse_pickcenter_row(summary)
    assert provider == "FanDuel"
    assert home_ml == -110
    assert away_ml == -110


def test_parse_pickcenter_row_returns_none_when_pickcenter_missing():
    assert _parse_pickcenter_row(_summary(pickcenter=[])) is None
    assert _parse_pickcenter_row({}) is None


def test_extract_market_row_reproduces_a_real_dispatch_confirmed_devig():
    # Real Blue Jays at Yankees line the slice-1 dispatch actually printed:
    # away_ml=179 home_ml=-193 -> de-vigged home=0.6476.
    event = _event(home_abbrev="NYY", away_abbrev="TOR")
    summary = _summary([_pickcenter_row("DraftKings", home_ml=-193, away_ml=179)])

    row = _extract_market_row(event, summary)

    assert row["home_team"] == "NYY"
    assert row["away_team"] == "TOR"
    assert row["market_provider"] == "DraftKings"
    assert row["market_home_win_probability"] == pytest.approx(0.6476, abs=1e-4)
    # Raw moneylines carried alongside the de-vigged probability - needed
    # for real Kelly bet sizing (kelly.py), which must be evaluated
    # against the real, vigged price, not the de-vigged one.
    assert row["home_moneyline"] == -193
    assert row["away_moneyline"] == 179


def test_extract_market_row_applies_the_real_abbrev_fixup():
    # Real Diamondbacks at Braves line the slice-1 dispatch actually
    # printed: away_ml=112 home_ml=-120 -> de-vigged home=0.5363. ESPN's
    # real abbreviation for the Diamondbacks is "ARI" - must come back as
    # this project's "AZ".
    event = _event(home_abbrev="ATL", away_abbrev="ARI")
    summary = _summary([_pickcenter_row("DraftKings", home_ml=-120, away_ml=112)])

    row = _extract_market_row(event, summary)

    assert row["home_team"] == "ATL"
    assert row["away_team"] == "AZ"
    assert row["market_home_win_probability"] == pytest.approx(0.5363, abs=1e-4)
    assert row["home_moneyline"] == -120
    assert row["away_moneyline"] == 112


def test_extract_market_row_returns_none_without_a_real_home_away_split():
    event = {"id": "1", "competitions": [{"competitors": [{"team": {"abbreviation": "NYY"}}]}]}
    summary = _summary([_pickcenter_row("DraftKings", -193, 179)])
    assert _extract_market_row(event, summary) is None


def test_extract_market_row_returns_none_without_real_pickcenter_data():
    event = _event(home_abbrev="NYY", away_abbrev="TOR")
    assert _extract_market_row(event, _summary(pickcenter=[])) is None
