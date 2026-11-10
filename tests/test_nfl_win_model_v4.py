import numpy as np
import pandas as pd
import pytest

from mlb_metrics import config, ml_models, nfl_game_picks, nfl_team_strength


def _games(rows):
    return pd.DataFrame(rows, columns=["game_id", "season", "week", "game_type", "home_team", "away_team",
                                       "home_score", "away_score"])


def test_elo_moves_toward_the_winner_and_more_for_a_blowout():
    close = nfl_team_strength.compute_elo_ratings(_games([("g1", 2025, 1, "REG", "KC", "DEN", 21, 20)]))
    rout = nfl_team_strength.compute_elo_ratings(_games([("g1", 2025, 1, "REG", "KC", "DEN", 45, 10)]))
    start = config.NFL_ELO_START
    assert close["KC"] > start > close["DEN"]
    assert close["KC"] + close["DEN"] == pytest.approx(2 * start)  # zero-sum
    assert rout["KC"] - start > close["KC"] - start
    # A road win is worth more than a home win by the same score: the home side was spotted points.
    road = nfl_team_strength.compute_elo_ratings(_games([("g1", 2025, 1, "REG", "KC", "DEN", 20, 21)]))
    assert road["DEN"] - start > close["KC"] - start


def test_elo_skips_unplayed_and_playoff_games_and_uses_todays_team_codes():
    games = _games([
        ("g1", 2019, 1, "REG", "OAK", "DEN", 24, 16),
        ("g2", 2019, 2, "REG", "DEN", "OAK", None, None),      # not played
        ("g3", 2019, 20, "POST", "DEN", "OAK", 40, 0),          # playoffs don't count
    ])
    elo = nfl_team_strength.compute_elo_ratings(games)
    assert set(elo.index) == {"LV", "DEN"} and elo["LV"] > elo["DEN"]


def test_elo_is_pulled_a_third_of_the_way_back_between_seasons():
    games = _games([("g1", 2025, 1, "REG", "KC", "DEN", 30, 10)])
    same_season = nfl_team_strength.compute_elo_ratings(games, current_season=2025)
    next_season = nfl_team_strength.compute_elo_ratings(games, current_season=2026)
    start = config.NFL_ELO_START
    assert next_season["KC"] - start == pytest.approx((same_season["KC"] - start) * config.NFL_ELO_SEASON_KEEP)
    # The pull also happens inside the history, when the season changes.
    two = _games([("g1", 2024, 1, "REG", "KC", "DEN", 30, 10), ("g2", 2025, 1, "REG", "BUF", "MIA", 20, 17)])
    assert nfl_team_strength.compute_elo_ratings(two)["KC"] == pytest.approx(next_season["KC"])


def _plays(team, opp, game, week, pass_epa, rush_epa, n=30):
    rows = []
    for kind, epa in (("pass", pass_epa), ("rush", rush_epa)):
        rows += [{"game_id": game, "season": 2026, "week": week, "season_type": "REG", "posteam": team, "defteam": opp,
                  "epa": epa, "pass": int(kind == "pass"), "rush": int(kind == "rush"), "qb_kneel": 0, "qb_spike": 0}] * n
    return rows


def test_pass_rush_epa_rates_each_side_against_the_league():
    rows = []
    for week in range(1, 5):
        rows += _plays("KC", "DEN", f"g{week}", week, pass_epa=0.4, rush_epa=-0.1)
        rows += _plays("DEN", "KC", f"g{week}", week, pass_epa=-0.2, rush_epa=0.1)
    out = nfl_team_strength.compute_pass_rush_epa(pd.DataFrame(rows)).set_index("team")
    assert list(out.columns) == nfl_team_strength.PASS_RUSH_COLUMNS
    # KC throws better than the league (0.1 a play) and DEN worse; pulled toward it, so inside the raw gap of 0.3.
    assert 0 < out.loc["KC", "pass_off_epa"] < 0.3 and out.loc["DEN", "pass_off_epa"] == pytest.approx(-out.loc["KC", "pass_off_epa"])
    # What KC's passing gains is what DEN's pass defense allows.
    assert out.loc["DEN", "pass_def_epa"] == pytest.approx(out.loc["KC", "pass_off_epa"])
    assert out.loc["KC", "rush_off_epa"] < 0 < out.loc["DEN", "rush_off_epa"]
    assert nfl_team_strength.compute_pass_rush_epa(pd.DataFrame()).empty
    assert nfl_team_strength.compute_pass_rush_epa(pd.DataFrame({"game_id": ["g"]})).empty


def test_kneels_and_spikes_stay_out_of_pass_rush_epa():
    rows = _plays("KC", "DEN", "g1", 1, 0.2, 0.0) + _plays("DEN", "KC", "g1", 1, 0.0, 0.0)
    base = nfl_team_strength.compute_pass_rush_epa(pd.DataFrame(rows)).set_index("team")
    kneel = {**rows[0], "epa": -5.0, "pass": 0, "rush": 1, "qb_kneel": 1}
    with_kneel = nfl_team_strength.compute_pass_rush_epa(pd.DataFrame(rows + [kneel] * 10)).set_index("team")
    assert with_kneel.loc["KC", "rush_off_epa"] == pytest.approx(base.loc["KC", "rush_off_epa"])


def test_margin_model_turns_a_predicted_margin_into_a_win_chance():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, 2))
    margin = 3 + 6 * x[:, 0] - 2 * x[:, 1] + rng.normal(scale=13, size=600)
    model = ml_models.MarginWinModel(alpha=1.0).fit(pd.DataFrame(x, columns=["a", "b"]), margin)
    assert 11 < model.sigma_ < 15
    proba = model.predict_proba(pd.DataFrame([[0.0, 0.0], [3.0, 0.0], [-3.0, 0.0]], columns=["a", "b"]))
    assert proba.shape == (3, 2) and np.allclose(proba.sum(axis=1), 1.0)
    even, strong, weak = proba[:, 1]
    assert 0.5 < even < 0.65          # the intercept is the home edge
    assert strong > 0.85 and weak < 0.2
    # Never 0 or 1, however lopsided.
    extreme = model.predict_proba(pd.DataFrame([[100.0, 0.0]], columns=["a", "b"]))[0, 1]
    assert extreme == pytest.approx(0.99)
    assert model.get_params() == {"alpha": 1.0, "clip": 0.01}


def test_v4_features_are_home_minus_away_and_keep_each_starters_qb_plus(monkeypatch):
    master = pd.DataFrame({"team": ["KC", "DEN"], "elo": [1600.0, 1450.0], "pass_off_epa": [0.10, -0.05],
                           "pass_def_epa": [-0.02, 0.04], "rush_off_epa": [0.01, 0.03], "rush_def_epa": [0.00, -0.02]})
    games = pd.DataFrame({"game_id": ["g1"], "season": [2026], "week": [10], "home_team": ["KC"], "away_team": ["DEN"]})
    base = games.assign(home_qb_plus=0.15, away_qb_plus=-0.05,
                        **{c: 1.0 for c in nfl_game_picks.QB_PLUS_FEATURE_COLUMNS if "qb_plus" not in c})
    monkeypatch.setattr(nfl_game_picks, "build_game_features_qb_plus", lambda *a, **k: base.copy())
    f = nfl_game_picks.build_game_features_elo(master, None, None, games).iloc[0]
    assert f["elo_diff"] == 150.0
    assert f["pass_off_diff"] == pytest.approx(0.15) and f["pass_def_diff"] == pytest.approx(-0.06)
    assert f["rush_off_diff"] == pytest.approx(-0.02) and f["rush_def_diff"] == pytest.approx(0.02)
    assert (f["home_qb_plus"], f["away_qb_plus"]) == (0.15, -0.05)
    assert nfl_game_picks.resolve_feature_builder(nfl_game_picks.ELO_FEATURE_COLUMNS) is nfl_game_picks.build_game_features_elo
    # A master from before these columns existed: blank differences, so the game keeps its fallback probability.
    old = nfl_game_picks.build_game_features_elo(master[["team"]], None, None, games)
    assert old["elo_diff"].isna().all()


def test_the_team_table_carries_elo_and_pass_rush_epa():
    """assemble_team_metrics returns a fixed column list - the v4 inputs have to be on it."""
    import inspect

    source = inspect.getsource(nfl_team_strength.assemble_team_metrics)
    listed = source[source.index("output_columns = ["):]
    assert '"elo"' in listed and "PASS_RUSH_COLUMNS" in listed


def test_blended_model_averages_its_members_each_on_its_own_columns():
    class Fixed:
        def __init__(self, column):
            self.column = column

        def predict_proba(self, X):
            assert list(X.columns) == [self.column]
            p = X[self.column].to_numpy()
            return np.column_stack([1 - p, p])

    X = pd.DataFrame({"a": [0.2, 0.9], "b": [0.6, 0.5], "unused": [1.0, 1.0]})
    blend = ml_models.BlendedWinModel([(Fixed("a"), ["a"]), (Fixed("b"), ["b"])])
    assert blend.predict_proba(X)[:, 1] == pytest.approx([0.4, 0.7])
    assert blend.predict_proba(X).sum(axis=1) == pytest.approx([1.0, 1.0])
    assert blend.get_params() == {}


def test_blend_feature_columns_are_both_models_inputs_with_their_own_builder():
    columns = nfl_game_picks.BLEND_FEATURE_COLUMNS
    assert set(columns) == set(nfl_game_picks.QB_PLUS_FEATURE_COLUMNS) | set(nfl_game_picks.ELO_FEATURE_COLUMNS)
    assert len(columns) == len(set(columns))
    assert nfl_game_picks.resolve_feature_builder(columns) is nfl_game_picks.build_game_features_blend
