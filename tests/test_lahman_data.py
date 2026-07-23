import pandas as pd
import pytest

from mlb_metrics import lahman_data


def test_build_crosswalk_joins_via_bbref_and_excludes_unmatched():
    chadwick = pd.DataFrame([
        {"key_mlbam": 1, "key_bbref": "aaronha01"},
        {"key_mlbam": 2, "key_bbref": "unknownxx01"},  # no Lahman match
    ])
    people = pd.DataFrame([
        {"playerID": "aaronha01", "bbrefID": "aaronha01"},
        {"playerID": "mayswi01", "bbrefID": "mayswi01"},
    ])

    crosswalk = lahman_data.build_crosswalk(chadwick, people)

    assert list(crosswalk["key_mlbam"]) == [1]
    assert crosswalk.iloc[0]["playerID"] == "aaronha01"


def test_attach_age_exact_arithmetic_just_after_birthday():
    seasons = pd.DataFrame([{"playerID": "p1", "yearID": 2000}])
    people = pd.DataFrame([{"playerID": "p1", "birthYear": 1975, "birthMonth": 6, "birthDay": 29}])

    result = lahman_data.attach_age(seasons, people)

    # Born 1975-06-29, reference 2000-06-30 - just turned 25 the day before.
    assert result.iloc[0]["age"] == 25


def test_attach_age_before_birthday_that_year():
    seasons = pd.DataFrame([{"playerID": "p1", "yearID": 2000}])
    people = pd.DataFrame([{"playerID": "p1", "birthYear": 1975, "birthMonth": 7, "birthDay": 1}])

    result = lahman_data.attach_age(seasons, people)

    # Born 1975-07-01, reference 2000-06-30 - hasn't turned 25 yet that year.
    assert result.iloc[0]["age"] == 24


def test_attach_age_defaults_missing_month_day_to_july_first():
    seasons = pd.DataFrame([{"playerID": "p1", "yearID": 2000}])
    people = pd.DataFrame([{"playerID": "p1", "birthYear": 1975, "birthMonth": None, "birthDay": None}])

    result = lahman_data.attach_age(seasons, people)

    assert result.iloc[0]["age"] == 24  # same as the explicit July 1 case above


def test_attach_age_drops_players_with_no_birth_year():
    seasons = pd.DataFrame([{"playerID": "p1", "yearID": 2000}, {"playerID": "p2", "yearID": 2000}])
    people = pd.DataFrame([
        {"playerID": "p1", "birthYear": 1975, "birthMonth": 6, "birthDay": 1},
        {"playerID": "p2", "birthYear": None, "birthMonth": None, "birthDay": None},
    ])

    result = lahman_data.attach_age(seasons, people)

    assert list(result["playerID"]) == ["p1"]


def test_attach_age_uses_each_rows_own_year_column():
    seasons = pd.DataFrame([{"playerID": "p1", "yearID": 2000}, {"playerID": "p1", "yearID": 2010}])
    people = pd.DataFrame([{"playerID": "p1", "birthYear": 1975, "birthMonth": 1, "birthDay": 1}])

    result = lahman_data.attach_age(seasons, people).set_index("yearID")

    assert result.loc[2000, "age"] == 25
    assert result.loc[2010, "age"] == 35


def test_persist_and_load_lahman_table_round_trips(tmp_path):
    df = pd.DataFrame([{"playerID": "p1", "birthYear": 1975}])
    raw_dir = str(tmp_path / "raw")

    assert lahman_data.load_persisted_lahman_table(raw_dir, "people") is None

    lahman_data.persist_lahman_table(df, raw_dir, "people")
    loaded = lahman_data.load_persisted_lahman_table(raw_dir, "people")

    assert loaded is not None
    assert loaded.iloc[0]["playerID"] == "p1"
