import pandas as pd
import pytest

from mlb_metrics import roster_positions


def _person(key_mlbam, abbreviation=None, missing_primary_position=False):
    person = {"id": key_mlbam}
    if not missing_primary_position:
        person["primaryPosition"] = {"abbreviation": abbreviation}
    return person


def test_normalize_position_eligibility_maps_outfield_variants_to_of():
    people = [_person(1, "LF"), _person(2, "CF"), _person(3, "RF"), _person(4, "OF")]
    result = roster_positions.normalize_position_eligibility(people).set_index("key_mlbam")

    assert result.loc[1, "dk_slot"] == "OF"
    assert result.loc[2, "dk_slot"] == "OF"
    assert result.loc[3, "dk_slot"] == "OF"
    assert result.loc[4, "dk_slot"] == "OF"


def test_normalize_position_eligibility_infield_positions_pass_through():
    people = [_person(1, "C"), _person(2, "1B"), _person(3, "2B"), _person(4, "3B"), _person(5, "SS")]
    result = roster_positions.normalize_position_eligibility(people).set_index("key_mlbam")

    assert result.loc[1, "dk_slot"] == "C"
    assert result.loc[2, "dk_slot"] == "1B"
    assert result.loc[3, "dk_slot"] == "2B"
    assert result.loc[4, "dk_slot"] == "3B"
    assert result.loc[5, "dk_slot"] == "SS"


def test_normalize_position_eligibility_pitcher_variants_map_to_p():
    people = [_person(1, "P"), _person(2, "SP"), _person(3, "RP")]
    result = roster_positions.normalize_position_eligibility(people).set_index("key_mlbam")

    assert (result["dk_slot"] == "P").all()


def test_normalize_position_eligibility_excludes_dh_and_unrecognized():
    people = [_person(1, "DH"), _person(2, "UNKNOWN_POS")]
    result = roster_positions.normalize_position_eligibility(people)

    assert result.empty


def test_normalize_position_eligibility_excludes_missing_primary_position():
    people = [_person(1, missing_primary_position=True)]
    result = roster_positions.normalize_position_eligibility(people)

    assert result.empty


def test_fetch_position_eligibility_chunks_and_dedupes(monkeypatch):
    calls = []

    def fake_get(endpoint, params):
        calls.append(params["personIds"])
        ids = [int(x) for x in params["personIds"].split(",")]
        return {"people": [_person(i, "SS") for i in ids]}

    import sys
    monkeypatch.setitem(sys.modules, "statsapi", type("FakeStatsapi", (), {"get": staticmethod(fake_get)})())

    result = roster_positions.fetch_position_eligibility([1, 2, 2, 3])

    assert set(result["key_mlbam"]) == {1, 2, 3}
    assert len(calls) == 1  # well under CHUNK_SIZE, one call
    assert calls[0] == "1,2,3"  # de-duped, order preserved
