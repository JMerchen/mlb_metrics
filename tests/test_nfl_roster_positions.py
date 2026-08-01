import pandas as pd

from mlb_metrics import nfl_roster_positions


def test_build_eligibility_table_qb_gets_exactly_one_row():
    players_df = pd.DataFrame([{"player_id": "qb1", "position": "QB"}])

    result = nfl_roster_positions.build_eligibility_table(players_df)

    assert len(result) == 1
    assert result.iloc[0]["dk_slot"] == "QB"


def test_build_eligibility_table_rb_wr_te_get_own_slot_and_flex():
    players_df = pd.DataFrame(
        [
            {"player_id": "rb1", "position": "RB"},
            {"player_id": "wr1", "position": "WR"},
            {"player_id": "te1", "position": "TE"},
        ]
    )

    result = nfl_roster_positions.build_eligibility_table(players_df)

    assert len(result) == 6
    for player_id, own_slot in [("rb1", "RB"), ("wr1", "WR"), ("te1", "TE")]:
        slots = set(result[result["player_id"] == player_id]["dk_slot"])
        assert slots == {own_slot, "FLEX"}


def test_build_eligibility_table_excludes_positions_with_no_dk_slot():
    players_df = pd.DataFrame(
        [
            {"player_id": "rb1", "position": "RB"},
            {"player_id": "k1", "position": "K"},
            {"player_id": "ol1", "position": "OL"},
        ]
    )

    result = nfl_roster_positions.build_eligibility_table(players_df)

    assert "k1" not in result["player_id"].values
    assert "ol1" not in result["player_id"].values
    assert "rb1" in result["player_id"].values


def test_nfl_flex_eligible_positions_matches_rb_wr_te():
    assert nfl_roster_positions.NFL_FLEX_ELIGIBLE_POSITIONS == ("RB", "WR", "TE")
