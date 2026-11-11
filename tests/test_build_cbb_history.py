"""scripts/build_cbb_history.py: how each team's March is coded for the
Time Machine file."""

import importlib.util
from pathlib import Path

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "build_cbb_history.py"


def _load():
    spec = importlib.util.spec_from_file_location("build_cbb_history", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _game(date, round_name, home, away, home_score, away_score, tournament=True):
    return {"date": pd.Timestamp(date), "round": round_name, "tournament": tournament,
            "home_id": home, "away_id": away, "home_score": home_score, "away_score": away_score}


def test_march_results_codes_the_round_each_team_lost_in():
    history = _load()
    games = pd.DataFrame([
        _game("2026-11-20", None, 9, 1, 60, 70, tournament=False),     # a regular-season loss is not a March result
        _game("2026-03-17", "First Four", 1, 2, 70, 60),
        _game("2026-03-19", "Round of 64", 3, 1, 80, 60),
        _game("2026-03-21", "Round of 32", 3, 4, 71, 70),
        _game("2026-03-26", "Sweet 16", 5, 3, 60, 65),
        _game("2026-03-28", "Elite 8", 3, 6, 77, 70),
        _game("2026-04-04", "Final Four", 7, 3, 60, 61),
        _game("2026-04-06", "Championship", 3, 8, 59, 66),
    ])
    assert history.march_results(games) == {2: 1, 1: 2, 4: 3, 5: 4, 6: 5, 7: 6, 3: 7, 8: 8}
    assert history.CHAMPION == 8
    assert 9 not in history.march_results(games)


def test_march_results_is_empty_without_a_tournament():
    history = _load()
    games = pd.DataFrame([_game("2020-03-01", None, 1, 2, 70, 60, tournament=False)])
    assert history.march_results(games) == {}
