import pandas as pd

from mlb_metrics import nfl_meetings


def _game(game_id, season, week, gameday, home, away, home_score, away_score, spread=3.0, total=44.5, location="Home",
          home_ml=-160.0, away_ml=140.0, game_type="REG"):
    return {"game_id": game_id, "season": season, "game_type": game_type, "week": week, "gameday": gameday,
            "home_team": home, "away_team": away, "home_score": home_score, "away_score": away_score, "location": location,
            "spread_line": spread, "total_line": total, "home_moneyline": home_ml, "away_moneyline": away_ml}


def _schedules():
    return pd.DataFrame([
        _game("2019_03_IND_HOU", 2019, 3, "2019-09-22", "HOU", "IND", 20, 23),
        _game("2020_13_IND_HOU", 2020, 13, "2020-12-06", "HOU", "IND", 20, 26, spread=-3.5),
        _game("2021_13_IND_HOU", 2021, 13, "2021-12-05", "HOU", "IND", 0, 31),
        _game("2022_01_IND_HOU", 2022, 1, "2022-09-11", "HOU", "IND", 20, 20),
        _game("2023_02_IND_HOU", 2023, 2, "2023-09-17", "HOU", "IND", 20, 31),
        _game("2024_08_IND_HOU", 2024, 8, "2024-10-27", "HOU", "IND", 23, 20),
        _game("2025_18_IND_HOU", 2025, 18, "2026-01-04", "HOU", "IND", 38, 30),
        _game("2025_05_HOU_IND", 2025, 5, "2025-10-05", "IND", "HOU", 17, 27),                    # the other building
        _game("2024_05_HOU_IND", 2024, 5, "2024-10-06", "HOU", "IND", 30, 10, location="Neutral"),  # a neutral field
        _game("2026_11_IND_HOU", 2026, 11, "2026-11-19", "HOU", "IND", None, None),                # the game itself, unplayed
        _game("2015_10_OAK_SD", 2015, 10, "2015-10-25", "SD", "OAK", 29, 37),                      # before both teams moved
    ])


def test_past_meetings_are_the_latest_five_in_the_same_building():
    games = pd.DataFrame([{"game_id": "2026_11_IND_HOU", "home_team": "HOU", "away_team": "IND"}])
    meetings = nfl_meetings.past_meetings(games, _schedules())
    assert list(meetings.columns) == nfl_meetings.MEETING_COLUMNS
    assert meetings["meeting_game_id"].tolist() == ["2025_18_IND_HOU", "2024_08_IND_HOU", "2023_02_IND_HOU",
                                                    "2022_01_IND_HOU", "2021_13_IND_HOU"]
    assert (meetings["game_id"] == "2026_11_IND_HOU").all()
    first = meetings.iloc[0]
    assert (first["date"], first["home_score"], first["away_score"]) == ("2026-01-04", 38, 30)
    assert (first["spread_line"], first["total_line"], first["home_moneyline"], first["away_moneyline"]) == (3.0, 44.5, -160.0, 140.0)
    assert nfl_meetings.past_meetings(games, _schedules(), limit=2)["meeting_game_id"].tolist() == ["2025_18_IND_HOU", "2024_08_IND_HOU"]


def test_a_franchise_that_moved_keeps_its_history_and_empty_inputs_give_an_empty_table():
    games = pd.DataFrame([{"game_id": "2026_12_LV_LAC", "home_team": "LAC", "away_team": "LV"}])
    meetings = nfl_meetings.past_meetings(games, _schedules())
    assert meetings["meeting_game_id"].tolist() == ["2015_10_OAK_SD"]
    assert (meetings.iloc[0]["home_team"], meetings.iloc[0]["away_team"]) == ("LAC", "LV")
    assert nfl_meetings.past_meetings(games.iloc[0:0], _schedules()).empty
    assert nfl_meetings.past_meetings(games, pd.DataFrame()).empty
    never = pd.DataFrame([{"game_id": "x", "home_team": "KC", "away_team": "HOU"}])
    assert nfl_meetings.past_meetings(never, _schedules()).empty


def test_write_matchup_history_adds_the_older_seasons_file(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    old = pd.DataFrame([_game("2010_08_HOU_IND", 2010, 8, "2010-11-01", "IND", "HOU", 30, 17),
                        _game("2011_01_IND_HOU", 2011, 1, "2011-09-11", "HOU", "IND", 34, 7)])
    old.to_csv(raw / nfl_meetings.OLD_GAMES_FILE, index=False)
    games = pd.DataFrame([{"game_id": "2026_11_IND_HOU", "home_team": "HOU", "away_team": "IND"}])
    recent = _schedules()[lambda s: s["season"] >= 2024]
    meetings = nfl_meetings.write_matchup_history(games, recent, str(tmp_path / "out"), str(raw))
    assert meetings["meeting_game_id"].tolist() == ["2025_18_IND_HOU", "2024_08_IND_HOU", "2011_01_IND_HOU"]
    written = pd.read_csv(tmp_path / "out" / "nfl_matchup_history.csv")
    assert written["meeting_game_id"].tolist() == meetings["meeting_game_id"].tolist()
    # No older file and no schedules: an empty file, not an error.
    assert nfl_meetings.write_matchup_history(games, pd.DataFrame(), str(tmp_path / "out2"), str(tmp_path)).empty
