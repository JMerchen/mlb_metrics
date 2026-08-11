import pandas as pd
import pytest

from mlb_metrics import nfl_ff_rankings


def _rankings_row(id, pos, ecr, sd=1.0, best=1, worst=1, ecr_type="bo", scrape_date="2026-08-11"):
    return {
        "id": id, "pos": pos, "ecr": ecr, "sd": sd, "best": best, "worst": worst,
        "ecr_type": ecr_type, "scrape_date": scrape_date, "player": f"Player {id}", "team": "NYJ",
    }


def _playerid_row(fantasypros_id, gsis_id):
    return {"fantasypros_id": fantasypros_id, "gsis_id": gsis_id}


def test_compute_ff_rankings_export_crosswalks_to_player_id():
    rankings = pd.DataFrame([_rankings_row(101, "QB", ecr=5.0, sd=1.2, best=1, worst=10)])
    playerids = pd.DataFrame([_playerid_row(101.0, "00-0012345")])

    result = nfl_ff_rankings.compute_ff_rankings_export(rankings, playerids)

    assert list(result.columns) == ["player_id", "ecr", "ecr_sd", "ecr_best", "ecr_worst"]
    row = result.iloc[0]
    assert row["player_id"] == "00-0012345"
    assert row["ecr"] == pytest.approx(5.0)
    assert row["ecr_sd"] == pytest.approx(1.2)
    assert row["ecr_best"] == 1
    assert row["ecr_worst"] == 10


def test_compute_ff_rankings_export_filters_to_best_ball_overall_only():
    rankings = pd.DataFrame([
        _rankings_row(101, "QB", ecr=5.0, ecr_type="bo"),
        _rankings_row(101, "QB", ecr=3.0, ecr_type="ro"),  # real redraft-overall row, different real rank
    ])
    playerids = pd.DataFrame([_playerid_row(101.0, "00-0012345")])

    result = nfl_ff_rankings.compute_ff_rankings_export(rankings, playerids).set_index("player_id")

    # Only the real "bo" (best-ball overall) row should survive.
    assert result.loc["00-0012345", "ecr"] == pytest.approx(5.0)


def test_compute_ff_rankings_export_excludes_dst():
    rankings = pd.DataFrame([
        _rankings_row(101, "QB", ecr=5.0),
        _rankings_row(202, "DST", ecr=1.0),
    ])
    playerids = pd.DataFrame([_playerid_row(101.0, "00-0012345"), _playerid_row(202.0, "00-0099999")])

    result = nfl_ff_rankings.compute_ff_rankings_export(rankings, playerids)

    assert "00-0099999" not in set(result["player_id"])
    assert "00-0012345" in set(result["player_id"])


def test_compute_ff_rankings_export_missing_crosswalk_is_absent_not_fabricated():
    rankings = pd.DataFrame([_rankings_row(101, "QB", ecr=5.0)])
    playerids = pd.DataFrame([_playerid_row(999.0, "00-0099999")])  # no real match for id 101

    result = nfl_ff_rankings.compute_ff_rankings_export(rankings, playerids)

    assert result.empty


def test_compute_ff_rankings_export_keeps_only_latest_scrape_date():
    rankings = pd.DataFrame([
        _rankings_row(101, "QB", ecr=8.0, scrape_date="2026-08-01"),  # stale real snapshot
        _rankings_row(101, "QB", ecr=5.0, scrape_date="2026-08-11"),  # real latest snapshot
    ])
    playerids = pd.DataFrame([_playerid_row(101.0, "00-0012345")])

    result = nfl_ff_rankings.compute_ff_rankings_export(rankings, playerids)

    assert len(result) == 1
    assert result.iloc[0]["ecr"] == pytest.approx(5.0)
