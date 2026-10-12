import math

import pandas as pd
import pytest

from mlb_metrics import nfl_totals


def _play(game_id, week, offense, defense, home_team, epa, kind="pass", **flags):
    return {
        "game_id": game_id, "season": 2025, "week": week, "season_type": flags.get("season_type", "REG"),
        "posteam": offense, "defteam": defense, "home_team": home_team, "epa": epa,
        "pass": 1 if kind == "pass" else 0, "rush": 1 if kind == "rush" else 0,
        "qb_kneel": flags.get("qb_kneel", 0), "qb_spike": flags.get("qb_spike", 0),
    }


def _team_game(week, team, opp, epa_sum, plays, points=20.0, home=1.0):
    return {"game_id": f"g{week}_{min(team, opp)}", "season": 2025, "week": week, "team": team, "opp": opp,
            "home": home, "epa_sum": epa_sum, "plays": plays, "points": points}


def test_team_game_epa_sums_real_plays_only_and_attaches_points_across_renamed_codes():
    pbp = pd.DataFrame([
        _play("g1", 1, "LV", "KC", "LV", 0.5),
        _play("g1", 1, "LV", "KC", "LV", -0.1, kind="rush"),
        _play("g1", 1, "LV", "KC", "LV", 3.0, qb_kneel=1),     # kneel: dropped
        _play("g1", 1, "LV", "KC", "LV", 3.0, kind="other"),   # punt etc.: dropped
        _play("g1", 1, "KC", "LV", "LV", 0.2),
        _play("p1", 20, "KC", "LV", "LV", 9.0, season_type="POST"),
    ])
    # Old schedules keep the franchise's code at the time (OAK for Las Vegas).
    schedules = pd.DataFrame([{"game_id": "g1", "home_team": "OAK", "away_team": "KC", "home_score": 27, "away_score": 20}])

    games = nfl_totals.team_game_epa(pbp, schedules).set_index("team")

    assert set(games.index) == {"LV", "KC"}
    assert games.loc["LV", "epa_sum"] == pytest.approx(0.4)
    assert games.loc["LV", "plays"] == 2
    assert games.loc["LV", "home"] == 1.0 and games.loc["KC", "home"] == 0.0
    assert games.loc["LV", "points"] == 27 and games.loc["KC", "points"] == 20


def test_team_game_epa_without_the_needed_columns_is_empty():
    assert nfl_totals.team_game_epa(pd.DataFrame({"game_id": ["g1"]})).empty


def test_wave_ratings_blends_windows_by_weight():
    # KC's offense: 0.0 EPA/play in weeks 1-2, 0.4 in weeks 3-4 (50 plays each).
    games = pd.DataFrame([
        _team_game(1, "KC", "DEN", 0.0, 50), _team_game(2, "KC", "DEN", 0.0, 50),
        _team_game(3, "KC", "DEN", 20.0, 50), _team_game(4, "KC", "DEN", 20.0, 50),
    ])
    ratings = nfl_totals.wave_ratings(games, windows=[(None, 0.5), (2, 0.5)], shrink_plays=0.0).set_index("team")

    # all four games 0.2, last two 0.4 -> 0.5 * 0.2 + 0.5 * 0.4
    assert ratings.loc["KC", "off_epa"] == pytest.approx(0.3)
    # DEN's defense allowed exactly what KC's offense produced.
    assert ratings.loc["DEN", "def_epa"] == pytest.approx(0.3)
    assert math.isnan(ratings.loc["KC", "def_epa"]) and math.isnan(ratings.loc["DEN", "off_epa"])


def test_wave_ratings_shrinks_a_small_sample_toward_the_league_rate():
    games = pd.DataFrame([
        _team_game(1, "KC", "DEN", 10.0, 50),   # 0.2 per play
        _team_game(1, "DEN", "KC", -10.0, 50),  # -0.2 per play; league rate 0.0
    ])
    raw = nfl_totals.wave_ratings(games, windows=[(None, 1.0)], shrink_plays=0.0).set_index("team")
    shrunk = nfl_totals.wave_ratings(games, windows=[(None, 1.0)], shrink_plays=50.0).set_index("team")
    assert raw.loc["KC", "off_epa"] == pytest.approx(0.2)
    assert shrunk.loc["KC", "off_epa"] == pytest.approx(0.1)


def test_predict_game_points_adds_both_sides_into_the_total():
    history = pd.DataFrame([
        _team_game(1, "KC", "DEN", 10.0, 50), _team_game(1, "DEN", "KC", -10.0, 50, home=0.0),
    ])
    games = pd.DataFrame([{"game_id": "w2", "home_team": "KC", "away_team": "DEN"}])
    coefficients = {"const": 20.0, "off_epa": 10.0, "opp_def_epa": 5.0, "off_plays": 0.0,
                    "opp_def_plays": 0.0, "home": 2.0}
    ratings = nfl_totals.wave_ratings(history).set_index("team")

    out = nfl_totals.predict_game_points(history, games, coefficients).iloc[0]

    home = 20 + 10 * ratings.loc["KC", "off_epa"] + 5 * ratings.loc["DEN", "def_epa"] + 2
    away = 20 + 10 * ratings.loc["DEN", "off_epa"] + 5 * ratings.loc["KC", "def_epa"]
    assert out["predicted_home_points"] == pytest.approx(home)
    assert out["predicted_away_points"] == pytest.approx(away)
    assert out["predicted_total"] == pytest.approx(home + away)
    assert out["predicted_home_points"] > out["predicted_away_points"]


def test_predict_game_points_is_empty_for_a_team_with_no_history():
    history = pd.DataFrame([_team_game(1, "KC", "DEN", 10.0, 50), _team_game(1, "DEN", "KC", -10.0, 50)])
    games = pd.DataFrame([{"game_id": "w2", "home_team": "KC", "away_team": "NYJ"}])
    out = nfl_totals.predict_game_points(history, games).iloc[0]
    assert math.isnan(out["predicted_total"])
