import numpy as np
import pandas as pd
import pytest

from mlb_metrics import wops


def _pa(game_pk, date, top, ab_no, batter, pitcher, event, stand="L", p_throws="R"):
    return {"game_pk": game_pk, "game_date": pd.Timestamp(date), "game_type": "R", "home_team": "NYY", "away_team": "BOS",
            "inning_topbot": "Top" if top else "Bot", "at_bat_number": ab_no, "batter": batter, "pitcher": pitcher,
            "stand": stand, "p_throws": p_throws, "events": event}


def _statcast():
    rows = []
    # Game 1: BOS batter 10 vs NYY starter 900 (then reliever 901); NYY batter 20 vs BOS starter 800.
    rows += [_pa(1, "2026-04-01", True, 1, 10, 900, "single"), _pa(1, "2026-04-01", True, 5, 10, 900, "walk"),
             _pa(1, "2026-04-01", True, 9, 10, 901, "field_out"), _pa(1, "2026-04-01", False, 2, 20, 800, "home_run", "R", "L"),
             _pa(1, "2026-04-01", False, 6, 20, 800, "strikeout", "R", "L")]
    # Game 2: batter 10 hitless, batter 20 a double.
    rows += [_pa(2, "2026-04-02", True, 1, 10, 900, "strikeout"), _pa(2, "2026-04-02", True, 5, 10, 900, "field_out"),
             _pa(2, "2026-04-02", False, 2, 20, 800, "double", "R", "L")]
    # A pitch that isn't the end of a PA, and a playoff game: both ignored.
    rows.append(dict(_pa(2, "2026-04-02", True, 7, 10, 900, None)))
    playoff = _pa(3, "2026-10-05", True, 1, 10, 900, "home_run")
    playoff["game_type"] = "F"
    rows.append(playoff)
    return pd.DataFrame(rows)


def test_plate_appearances_count_obp_and_slg_pieces():
    pas = wops.plate_appearances(_statcast())
    assert len(pas) == 8
    b10 = pas[pas["batter"] == 10]
    assert b10[["PA", "AB", "H", "TB", "OB"]].sum().tolist() == [5, 4, 1, 1, 2]   # walk is a PA, not an AB
    hr = pas[(pas["batter"] == 20) & (pas["events"] == "home_run")].iloc[0]
    assert (hr["TB"], hr["bat_team"], hr["fld_team"], hr["is_home"]) == (4, "NYY", "BOS", 1)


def test_starters_are_each_teams_first_pitcher():
    starters = wops.game_starters(wops.plate_appearances(_statcast()))
    by_team = starters[starters["game_pk"] == 1].set_index("fld_team")["starter"].to_dict()
    assert by_team == {"NYY": 900, "BOS": 800}
    starts = wops.starter_games(wops.plate_appearances(_statcast()))
    assert starts.set_index(["pitcher", "game_pk"]).loc[(900, 1), "BF"] == 2    # the reliever's PA isn't his


def test_training_features_only_use_earlier_games():
    frame = wops.training_frame(_statcast()).set_index(["batter", "game_pk"])
    assert frame.loc[(10, 1), "hit_pa_season"] == 0          # nothing before his first game
    assert frame.loc[(10, 2), "hit_pa_season"] == 3          # game 1 only
    assert frame.loc[(20, 2), "hit_slg_season"] > frame.loc[(10, 2), "hit_slg_season"]
    assert frame.loc[(10, 1), "platoon_edge"] == 1           # lefty vs a righty starter
    assert frame.loc[(20, 1), "platoon_edge"] == 1           # righty vs a lefty: also opposite hands


def test_games_with_hit_counts_games_not_hits():
    recent = wops.games_with_hit(_statcast(), last_n=10).set_index("key_mlbam")
    assert recent.loc[10].tolist() == [1, 2]     # a hit in game 1, none in game 2
    assert recent.loc[20].tolist() == [2, 2]
    assert wops.games_with_hit(_statcast(), last_n=1).set_index("key_mlbam").loc[10].tolist() == [0, 1]


def test_current_features_face_todays_probable_or_an_average_starter():
    probables = pd.DataFrame([
        {"team": "BOS", "opponent": "NYY", "is_home": "False", "key_mlbam": 800, "Throws": "L"},
        {"team": "NYY", "opponent": "BOS", "is_home": "True", "key_mlbam": 900, "Throws": "R"},
    ])
    now = wops.current_features(_statcast(), probables).set_index("batter")
    assert now.loc[10, "opponent"] == "NYY" and now.loc[10, "starter"] == 900
    assert now.loc[10, "hit_pa_season"] == 5                # through his latest game
    assert now.loc[10, "sp_bf_season"] == 4                 # 900's two starts
    assert now.loc[20, "is_home"] == 1
    idle = wops.current_features(_statcast()).set_index("batter")
    assert idle.loc[10, "opponent"] == "" and idle.loc[10, "sp_bf_season"] == 0


def test_fit_and_predict_round_trip():
    frame = wops.training_frame(pd.concat([_statcast()] * 3, ignore_index=True))
    models = wops.fit(pd.concat([frame] * 40, ignore_index=True))
    predictions = wops.predict(models, frame)
    assert np.isfinite(predictions).all()
    assert predictions.between(0, 4).all()
