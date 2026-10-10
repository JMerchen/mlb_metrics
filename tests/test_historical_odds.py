import pandas as pd
import pytest

from mlb_metrics import historical_odds


def _event(event_id="1", home="DET", away="CIN", status="STATUS_FINAL"):
    return {
        "id": event_id,
        "date": "2025-06-15T17:40Z",
        "status": {"type": {"name": status}},
        "competitions": [{"competitors": [
            {"homeAway": "home", "team": {"abbreviation": home}},
            {"homeAway": "away", "team": {"abbreviation": away}},
        ]}],
    }


def _side(ml, open_=None, close=None):
    """An odds block shaped like the core API's (seen in the 2026-10-10 probe)."""
    side = {"moneyLine": ml, "current": {"moneyLine": {"american": f"{ml:+d}" if ml else None}}}
    if open_ is not None:
        side["open"] = {"moneyLine": {"american": open_}}
    if close is not None:
        side["close"] = {"moneyLine": {"american": close}}
    return side


def test_parse_american_handles_strings_even_and_placeholders():
    assert historical_odds.parse_american("+125") == 125
    assert historical_odds.parse_american("-115") == -115
    assert historical_odds.parse_american("EVEN") == 100
    assert historical_odds.parse_american(140) == 140
    # ESPN uses 0 for "no line"; anything under 100 is not an American price.
    assert historical_odds.parse_american(0) is None
    assert historical_odds.parse_american(None) is None
    assert historical_odds.parse_american("n/a") is None


def test_parse_event_odds_reads_open_and_close_and_skips_live_feeds():
    odds = {"items": [
        {"provider": {"name": "ESPN BET"},
         "homeTeamOdds": _side(-185, open_="-175", close="-180"),
         "awayTeamOdds": _side(155, open_="+145", close="+150")},
        {"provider": {"name": "ESPN BET - Live Odds"},
         "homeTeamOdds": _side(-6000), "awayTeamOdds": _side(2000)},
    ]}
    rows = historical_odds.parse_event_odds("2025-06-15", _event(), odds)
    assert len(rows) == 1
    row = rows[0]
    assert (row["home_team"], row["away_team"], row["provider"]) == ("DET", "CIN", "ESPN BET")
    assert (row["home_ml_open"], row["home_ml_close"], row["away_ml_close"]) == (-175, -180, 150)


def test_consensus_uses_close_falls_back_to_last_line_and_takes_the_median():
    rows = pd.DataFrame([
        # Book with a close snapshot.
        {"date": "2023-06-15", "event_id": "1", "home_team": "BAL", "away_team": "TOR", "start_time": "t",
         "status": "STATUS_FINAL", "provider": "A", "home_ml": -118, "away_ml": 100,
         "home_ml_open": None, "away_ml_open": None, "home_ml_close": -115, "away_ml_close": -105},
        # Books without one: their last pre-game line stands in.
        {"date": "2023-06-15", "event_id": "1", "home_team": "BAL", "away_team": "TOR", "start_time": "t",
         "status": "STATUS_FINAL", "provider": "B", "home_ml": -125, "away_ml": 105,
         "home_ml_open": None, "away_ml_open": None, "home_ml_close": None, "away_ml_close": None},
        {"date": "2023-06-15", "event_id": "1", "home_team": "BAL", "away_team": "TOR", "start_time": "t",
         "status": "STATUS_FINAL", "provider": "C", "home_ml": -120, "away_ml": 102,
         "home_ml_open": None, "away_ml_open": None, "home_ml_close": None, "away_ml_close": None},
        # A postponed game is left out.
        {"date": "2023-06-15", "event_id": "2", "home_team": "OAK", "away_team": "TB", "start_time": "t",
         "status": "STATUS_POSTPONED", "provider": "A", "home_ml": 170, "away_ml": -200,
         "home_ml_open": None, "away_ml_open": None, "home_ml_close": 170, "away_ml_close": -200},
    ])
    games = historical_odds.consensus_closing_lines(rows)
    assert list(games["event_id"]) == ["1"]
    game = games.iloc[0]
    assert game["n_books"] == 3
    # Medians in implied-probability space: -115/-125/-120 -> -120, and
    # -105/+105/+102 -> +102 (taking the median of the raw American numbers
    # only agrees here by luck - the scale jumps from -100 to +100).
    assert game["home_ml"] == pytest.approx(-120)
    assert game["away_ml"] == pytest.approx(102)
    home = 120 / 220
    away = 100 / 202
    assert game["market_home_win_probability"] == pytest.approx(home / (home + away))


def test_to_american_round_trips_through_implied():
    for ml in (-250, -110, 100, 102, 180):
        assert float(historical_odds.to_american(historical_odds.implied(ml))) == pytest.approx(ml if ml != 100 else -100)
