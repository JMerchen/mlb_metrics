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


def _odds(date, event, home, away, home_ml, away_ml, start="T17:05Z", provider="ESPN BET", status="STATUS_FINAL"):
    return {"date": date, "event_id": str(event), "home_team": home, "away_team": away, "start_time": f"{date}{start}",
            "status": status, "provider": provider, "home_ml": home_ml, "away_ml": away_ml,
            "home_ml_open": None, "away_ml_open": None, "home_ml_close": home_ml, "away_ml_close": away_ml}


def test_team_games_carry_each_sides_closing_moneyline():
    odds = pd.DataFrame([
        _odds("2026-04-01", 11, "NYY", "BOS", -150, 130),
        _odds("2026-04-02", 12, "NYY", "BOS", -125, 105),
        _odds("2026-04-03", 13, "TB", "NYY", 110, -130, status="STATUS_POSTPONED"),     # not played: no line
    ])
    games = team_schedule.team_games(_statcast(), odds)
    nyy = games[games["team"] == "NYY"].set_index("game_pk")
    assert nyy.loc[1, ["moneyline", "opponent_moneyline"]].tolist() == [-150, 130]
    assert nyy.loc[2, ["moneyline", "opponent_moneyline"]].tolist() == [-125, 105]
    assert nyy.loc[[3, 4], "moneyline"].isna().all()
    bos = games[games["team"] == "BOS"].set_index("game_pk")
    assert bos.loc[1, ["moneyline", "opponent_moneyline"]].tolist() == [130, -150]
    # Without odds the columns are there and blank.
    plain = team_schedule.team_games(_statcast())
    assert plain["moneyline"].isna().all() and plain["opponent_moneyline"].isna().all()
    assert list(plain.columns)[-2:] == ["moneyline", "opponent_moneyline"]


def test_game_moneylines_match_espn_codes_and_doubleheaders_in_order():
    statcast = pd.DataFrame([
        _pitch(21, "2026-05-01", "AZ", "CWS", 3, 2), _pitch(22, "2026-05-01", "AZ", "CWS", 1, 5),      # a doubleheader
        _pitch(23, "2026-05-02", "AZ", "CWS", 4, 0), _pitch(24, "2026-05-02", "AZ", "CWS", 2, 6),      # another, with one game's odds missing
    ])
    odds = pd.DataFrame([
        _odds("2026-05-01", 902, "ARI", "CHW", -200, 170, start="T23:40Z"),        # the nightcap, listed first
        _odds("2026-05-01", 901, "ARI", "CHW", -140, 120, start="T19:10Z"),
        _odds("2026-05-02", 903, "ARI", "CHW", -110, -110),
    ])
    lines = team_schedule.game_moneylines(team_schedule.final_scores(statcast), odds).set_index("game_pk")
    assert lines.loc[21].tolist() == [-140, 120]
    assert lines.loc[22].tolist() == [-200, 170]
    assert 23 not in lines.index and 24 not in lines.index       # two games, one line: not guessed at
    assert team_schedule.game_moneylines(team_schedule.final_scores(statcast), None).empty
