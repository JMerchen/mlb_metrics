import pandas as pd

from mlb_metrics import data


def test_persist_raw_statcast_appends_and_dedupes_by_pitch(tmp_path):
    raw_dir = str(tmp_path / "raw")
    season = 2026

    day1 = pd.DataFrame({
        "game_pk": [1, 1],
        "at_bat_number": [1, 2],
        "pitch_number": [1, 1],
        "game_date": pd.to_datetime(["2026-04-01", "2026-04-01"]),
        "events": ["single", "field_out"],
    })
    combined1 = data.persist_raw_statcast(day1, raw_dir, season)
    assert len(combined1) == 2

    day2 = pd.DataFrame({
        "game_pk": [1, 2],
        "at_bat_number": [2, 1],  # first row shares day1's pitch key
        "pitch_number": [1, 1],
        "game_date": pd.to_datetime(["2026-04-01", "2026-04-02"]),
        "events": ["walk", "single"],
    })
    combined2 = data.persist_raw_statcast(day2, raw_dir, season)

    assert len(combined2) == 3
    dup_row = combined2[(combined2["game_pk"] == 1) & (combined2["at_bat_number"] == 2)]
    assert dup_row.iloc[0]["events"] == "walk"  # keep="last" -> day2's value wins on conflict

    reloaded = data.load_persisted_statcast(raw_dir, season)
    assert len(reloaded) == 3


def test_load_persisted_statcast_returns_none_when_absent(tmp_path):
    assert data.load_persisted_statcast(str(tmp_path / "raw"), 2026) is None


def test_label_pitcher_roles_picks_lowest_at_bat_number_as_starter():
    # Top half = home_team pitching to away_team's batters; Bot half =
    # away_team pitching to home_team's batters (matches the convention
    # used throughout data.py's other team-lookup helpers).
    rows = [
        {"game_id": 1, "inning_topbot": "Top", "home_team": "A", "away_team": "B", "pitcher": 101, "at_bat_number": 1},
        {"game_id": 1, "inning_topbot": "Bot", "home_team": "A", "away_team": "B", "pitcher": 201, "at_bat_number": 2},
        {"game_id": 1, "inning_topbot": "Top", "home_team": "A", "away_team": "B", "pitcher": 102, "at_bat_number": 3},
        {"game_id": 1, "inning_topbot": "Bot", "home_team": "A", "away_team": "B", "pitcher": 202, "at_bat_number": 4},
    ]
    data_with_game_id = pd.DataFrame(rows)

    roles = data.label_pitcher_roles(data_with_game_id).set_index(["team", "pitcher"])

    assert roles.loc[("A", 101), "is_starter"]
    assert not roles.loc[("A", 102), "is_starter"]
    assert roles.loc[("B", 201), "is_starter"]
    assert not roles.loc[("B", 202), "is_starter"]
