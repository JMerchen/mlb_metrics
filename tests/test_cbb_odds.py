import pandas as pd
import pytest

from mlb_metrics import cbb_odds


def _item(provider="ESPN BET", spread=-7.5, total=145.5, home_ml=-320, away_ml=250, snapshots=True):
    item = {"provider": {"name": provider}, "spread": spread, "overUnder": total,
            "homeTeamOdds": {"moneyLine": home_ml}, "awayTeamOdds": {"moneyLine": away_ml}}
    if snapshots:
        item["homeTeamOdds"].update({"open": {"pointSpread": {"american": "-6.5"}, "moneyLine": {"american": "-280"}},
                                     "close": {"pointSpread": {"american": "-7.5"}, "moneyLine": {"american": "-320"}}})
        item["awayTeamOdds"].update({"open": {"pointSpread": {"american": "+6.5"}, "moneyLine": {"american": "+225"}},
                                     "close": {"pointSpread": {"american": "+7.5"}, "moneyLine": {"american": "+250"}}})
        item["open"] = {"total": {"american": "143.5"}}
        item["close"] = {"total": {"american": "145.5"}}
    return item


def test_parse_points_handles_text_signs_and_pick():
    assert cbb_odds.parse_points("-7.5") == -7.5
    assert cbb_odds.parse_points("+3") == 3.0
    assert cbb_odds.parse_points("PK") == 0.0
    assert cbb_odds.parse_points("o145.5") == 145.5
    assert cbb_odds.parse_points(None) is None
    assert cbb_odds.parse_points("n/a") is None


def test_parse_game_odds_reads_lines_and_snapshots_and_skips_live_feeds():
    rows = cbb_odds.parse_game_odds("401", {"items": [_item(), _item(provider="ESPN Bet - Live Odds")]})
    assert len(rows) == 1
    row = rows[0]
    assert row["game_id"] == 401 and row["provider"] == "ESPN BET"
    assert row["spread"] == -7.5 and row["total"] == 145.5
    assert row["spread_open"] == -6.5 and row["spread_close"] == -7.5
    assert row["total_open"] == 143.5 and row["total_close"] == 145.5
    assert row["home_ml_close"] == -320 and row["away_ml_open"] == 225


def test_home_spread_falls_back_to_the_away_sides_number():
    item = _item()
    del item["homeTeamOdds"]["open"]
    row = cbb_odds.parse_game_odds(1, {"items": [item]})[0]
    assert row["spread_open"] == -6.5


def test_parse_game_odds_with_nothing_returns_no_rows():
    assert cbb_odds.parse_game_odds(1, None) == []
    assert cbb_odds.parse_game_odds(1, {"items": []}) == []


def test_consensus_takes_the_median_across_books_and_drops_empty_games():
    rows = (cbb_odds.parse_game_odds(1, {"items": [_item(provider="A", spread=-7, snapshots=False),
                                                   _item(provider="B", spread=-8, snapshots=False),
                                                   _item(provider="C", spread=-7.5, snapshots=False)]})
            + [{**{c: None for c in cbb_odds.ROW_COLUMNS}, "game_id": 2, "provider": ""}])
    lines = cbb_odds.consensus_lines(pd.DataFrame(rows, columns=cbb_odds.ROW_COLUMNS))
    assert list(lines["game_id"]) == [1]
    game = lines.iloc[0]
    assert game["n_books"] == 3 and game["spread"] == -7.5 and game["total"] == 145.5
    assert game["home_ml"] == pytest.approx(-320) and game["away_ml"] == pytest.approx(250)
    assert pd.isna(game["spread_open"])


def test_consensus_prefers_the_close_snapshot():
    item = _item(spread=-9.0)   # the headline number disagrees with the close snapshot
    lines = cbb_odds.consensus_lines(pd.DataFrame(cbb_odds.parse_game_odds(1, {"items": [item]}), columns=cbb_odds.ROW_COLUMNS))
    assert lines.iloc[0]["spread"] == -7.5 and lines.iloc[0]["spread_open"] == -6.5


def test_consensus_of_nothing_is_empty():
    assert cbb_odds.consensus_lines(pd.DataFrame(columns=cbb_odds.ROW_COLUMNS)).empty
