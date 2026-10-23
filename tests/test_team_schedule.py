import pandas as pd

from mlb_metrics import team_schedule


def _pitch(game_pk, date, home, away, hs, as_, game_type="R"):
    return {"game_pk": game_pk, "game_date": pd.Timestamp(date), "game_type": game_type, "home_team": home,
            "away_team": away, "post_home_score": hs, "post_away_score": as_}


def _statcast():
    return pd.DataFrame([
        _pitch(1, "2026-04-01", "NYY", "BOS", 0, 0), _pitch(1, "2026-04-01", "NYY", "BOS", 3, 2),   # NYY win 3-2
        _pitch(2, "2026-04-02", "NYY", "BOS", 1, 9),                                               # BOS win 9-1
        _pitch(3, "2026-04-03", "TB", "NYY", 4, 5),                                                # new opponent and park
        _pitch(4, "2026-04-10", "TB", "NYY", 2, 1),                                                # same park, a week later
        _pitch(5, "2026-10-05", "NYY", "TB", 7, 0, game_type="F"),                                 # playoffs: left out
    ])


def test_final_scores_take_the_last_score_of_regular_season_games():
    scores = team_schedule.final_scores(_statcast()).set_index("game_pk")
    assert scores.loc[1, ["home_score", "away_score"]].tolist() == [3, 2]
    assert 5 not in scores.index


def test_team_games_from_each_side_with_results_and_series():
    games = team_schedule.team_games(_statcast())
    nyy = games[games["team"] == "NYY"]
    assert nyy["opponent"].tolist() == ["BOS", "BOS", "TB", "TB"]
    assert nyy["result"].tolist() == ["W", "L", "W", "L"]
    assert nyy["run_diff"].tolist() == [1, -8, 1, -1]
    assert nyy["home"].tolist() == [1, 1, 0, 0]
    assert nyy["series"].tolist() == [1, 1, 2, 3]       # a week apart: a separate visit
    bos = games[games["team"] == "BOS"]
    assert bos[["result", "run_diff", "series"]].values.tolist() == [["L", -1, 1], ["W", 8, 1]]
