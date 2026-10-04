import numpy as np
import pandas as pd
import pytest

from mlb_metrics import mlb_game_model


def _game(game_pk, date, home, away, home_score, away_score, home_sp=1, away_sp=2,
          home_pen=11, away_pen=12, home_sp_events=None, away_sp_events=None, game_type="R"):
    """Pitch rows for one game: the starter faces the first batters, a
    reliever the rest. Events are what the PITCHING side allowed."""
    rows = []
    at_bat = 0
    for half, pitching_sp, pitching_pen, events in (
        ("Top", home_sp, home_pen, home_sp_events or ["strikeout", "single"]),
        ("Bot", away_sp, away_pen, away_sp_events or ["strikeout", "single"]),
    ):
        for i, event in enumerate(list(events) + ["field_out"]):
            at_bat += 1
            pitcher = pitching_sp if i < len(events) else pitching_pen
            rows.append({
                "game_pk": game_pk, "game_date": pd.Timestamp(date), "game_type": game_type,
                "home_team": home, "away_team": away, "inning_topbot": half,
                "at_bat_number": at_bat, "pitch_number": 1, "pitcher": pitcher,
                "post_home_score": 0, "post_away_score": 0, "events": event,
                "woba_value": {"single": 0.9, "walk": 0.7}.get(event, 0.0), "woba_denom": 1,
                "estimated_woba_using_speedangle": {"single": 0.8, "field_out": 0.1}.get(event, np.nan),
            })
    rows[-1]["post_home_score"] = home_score
    rows[-1]["post_away_score"] = away_score
    return rows


def _statcast(*games):
    return pd.DataFrame([row for game in games for row in game])


def test_game_table_reads_final_score_starters_and_winner():
    games, _ = mlb_game_model.prepare_history(_statcast(_game(1, "2026-04-01", "NYY", "BOS", 5, 3, home_sp=7, away_sp=8)))
    row = games.iloc[0]
    assert (row["home_score"], row["away_score"], row["home_win"]) == (5, 3, 1)
    assert (row["home_sp"], row["away_sp"]) == (7, 8)


def test_spring_training_and_postseason_are_left_out():
    sc = _statcast(_game(1, "2026-04-01", "NYY", "BOS", 5, 3), _game(2, "2026-10-05", "NYY", "BOS", 1, 2, game_type="D"))
    games, _ = mlb_game_model.prepare_history(sc)
    assert list(games["game_pk"]) == [1]


def test_team_aliases_are_normalized():
    games, _ = mlb_game_model.prepare_history(_statcast(_game(1, "2026-04-01", "AZ", "ATH", 5, 3)))
    assert (games.iloc[0]["home_team"], games.iloc[0]["away_team"]) == ("ARI", "OAK")


def test_features_use_only_games_strictly_before_the_game_date():
    """A doubleheader's second game must not see the first, and a game's
    own result never feeds its own features."""
    sc = _statcast(
        _game(1, "2026-04-01", "NYY", "BOS", 10, 0),
        _game(2, "2026-04-02", "NYY", "BOS", 10, 0),
        _game(3, "2026-04-02", "NYY", "BOS", 10, 0),
    )
    games, pa = mlb_game_model.prepare_history(sc)
    features = mlb_game_model.build_features(games, games, pa)
    # Game 1 has no history at all.
    assert features.loc[0, "rd_diff"] == 0
    # Both games on 04-02 see only 04-01, so they are identical.
    assert features.loc[1, "rd_diff"] == pytest.approx(features.loc[2, "rd_diff"])
    assert features.loc[1, "rd_diff"] > 0


def test_unknown_starters_fall_back_to_the_league_rate():
    """A missing probable pitcher (or a debut) is rated as league average,
    not as a zero-strikeout pitcher."""
    history = _statcast(_game(1, "2026-04-01", "NYY", "BOS", 3, 2))
    games, pa = mlb_game_model.prepare_history(history)
    slate = pd.DataFrame([{"game_date": pd.Timestamp("2026-04-02"), "home_team": "NYY", "away_team": "BOS",
                           "home_sp": np.nan, "away_sp": 999}])
    features = mlb_game_model.build_features(slate, games, pa)
    assert features.loc[0, "sp_kbb_diff"] == pytest.approx(0.0)


def test_better_starter_and_run_differential_favor_that_side():
    games_rows = []
    for i in range(10):
        games_rows.append(_game(i, f"2026-04-{i + 1:02d}", "NYY", "BOS", 6, 1, home_sp=1, away_sp=2,
                                home_sp_events=["strikeout"] * 4, away_sp_events=["walk"] * 4))
    games, pa = mlb_game_model.prepare_history(_statcast(*games_rows))
    slate = pd.DataFrame([{"game_date": pd.Timestamp("2026-04-20"), "home_team": "NYY", "away_team": "BOS",
                           "home_sp": 1, "away_sp": 2}])
    features = mlb_game_model.build_features(slate, games, pa)
    assert features.loc[0, "rd_diff"] > 0
    assert features.loc[0, "sp_kbb_diff"] > 0


def _season(n_games=160):
    """GOOD beats BAD every time, with a strikeout starter against a
    walk-prone one; home field alternates."""
    rows = []
    start = pd.Timestamp("2026-04-01")
    for i in range(n_games):
        date = start + pd.Timedelta(days=i)
        good_home = i % 2 == 0
        strikeouts, walks = ["strikeout"] * 3, ["walk"] * 3
        if good_home:
            rows.append(_game(i, date, "GOOD", "BAD", 5, 1, home_sp=1, away_sp=2,
                              home_sp_events=strikeouts, away_sp_events=walks))
        else:
            rows.append(_game(i, date, "BAD", "GOOD", 1, 5, home_sp=2, away_sp=1,
                              home_sp_events=walks, away_sp_events=strikeouts))
    # Noise games so both outcomes occur from either side.
    for j in range(40):
        date = start + pd.Timedelta(days=j)
        rows.append(_game(1000 + j, date, "MID1", "MID2", 3 if j % 2 else 2, 2 if j % 2 else 3, home_sp=3, away_sp=4))
    return _statcast(*rows)


def test_compute_game_win_probabilities_favors_the_stronger_team_and_keeps_schedule_codes():
    schedule = pd.DataFrame([
        {"game_pk": 9001, "date": pd.Timestamp("2026-10-01"), "home_team": "GOOD", "away_team": "BAD",
         "home_probable_pitcher_key_mlbam": 1, "away_probable_pitcher_key_mlbam": 2},
        {"game_pk": 9002, "date": pd.Timestamp("2026-10-01"), "home_team": "BAD", "away_team": "GOOD",
         "home_probable_pitcher_key_mlbam": 2, "away_probable_pitcher_key_mlbam": None},
    ])
    result = mlb_game_model.compute_game_win_probabilities(_season(), schedule)
    assert list(result.columns) == ["game_pk", "date", "home_team", "away_team", "home_win_probability"]
    assert result.loc[0, "home_win_probability"] > 0.5
    assert result.loc[1, "home_win_probability"] < 0.5
    assert list(result["home_team"]) == ["GOOD", "BAD"]


def test_too_little_history_returns_none_so_the_caller_falls_back():
    schedule = pd.DataFrame([{"game_pk": 1, "date": pd.Timestamp("2026-04-03"), "home_team": "NYY",
                              "away_team": "BOS", "home_probable_pitcher_key_mlbam": 1,
                              "away_probable_pitcher_key_mlbam": 2}])
    sc = _statcast(_game(1, "2026-04-01", "NYY", "BOS", 5, 3))
    assert mlb_game_model.compute_game_win_probabilities(sc, schedule) is None
    empty = pd.DataFrame(columns=mlb_game_model.STATCAST_COLUMNS)
    assert mlb_game_model.compute_game_win_probabilities(empty, schedule) is None
