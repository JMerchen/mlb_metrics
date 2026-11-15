import numpy as np
import pandas as pd
import pytest

from mlb_metrics import config, ml_models, nfl_game_picks, nfl_qb_elo


def _line(season, week, team, player, attempts=30, completions=20, yards=250, tds=2, ints=0, sacks=2, carries=3, rush=10, rush_tds=0):
    return {"season": season, "week": week, "season_type": "REG", "team": team, "player_id": player, "attempts": attempts,
            "completions": completions, "passing_yards": yards, "passing_tds": tds, "passing_interceptions": ints,
            "sacks_suffered": sacks, "carries": carries, "rushing_yards": rush, "rushing_tds": rush_tds}


def _game(season, week, home, away, home_score, away_score, home_qb="H", away_qb="A", game_id=None):
    return {"season": season, "week": week, "game_type": "REG", "game_id": game_id or f"{season}_{week:02d}_{away}_{home}",
            "home_team": home, "away_team": away, "home_score": home_score, "away_score": away_score,
            "home_qb_id": home_qb, "away_qb_id": away_qb, "location": "Home"}


def test_qb_game_value_is_fivethirtyeights_formula():
    weekly = pd.DataFrame([_line(2026, 1, "KC", "QB1"), _line(2026, 1, "KC", "RB1", attempts=0), _line(2026, 1, "SD", "QB2", ints=2)])
    values = nfl_qb_elo.qb_game_values(weekly).set_index("player_id")
    expected = -2.2 * 30 + 3.7 * 20 + 250 / 5 + 11.3 * 2 - 8 * 2 - 1.1 * 3 + 0.6 * 10
    assert values.loc["QB1", "value"] == pytest.approx(expected)
    assert values.loc["QB2", "value"] == pytest.approx(expected - 14.1 * 2)
    assert values.loc["QB2", "team"] == "LAC"                    # an old team code maps to today's
    assert "RB1" not in values.index                              # no pass attempts: not a quarterback game
    assert nfl_qb_elo.qb_game_values(pd.DataFrame()).empty


def test_ratings_move_only_on_games_already_played_and_record_the_pregame_gap():
    schedules = pd.DataFrame([
        _game(2025, 1, "KC", "DEN", 31, 10, "MAHOMES", "NIX"),
        _game(2025, 2, "DEN", "KC", 17, 20, "NIX", "MAHOMES"),
        _game(2025, 3, "KC", "DEN", None, None, "MAHOMES", "NIX"),      # not played: ignored
    ])
    weekly = pd.DataFrame([
        _line(2025, 1, "KC", "MAHOMES", yards=350, tds=4), _line(2025, 1, "DEN", "NIX", yards=150, tds=0, ints=2),
        _line(2025, 2, "KC", "MAHOMES", yards=300, tds=2), _line(2025, 2, "DEN", "NIX", yards=200, tds=1),
    ])
    state, gaps = nfl_qb_elo.rate(schedules, weekly, record=True)
    assert len(gaps) == 2
    # Before anything was played both teams are average and both QBs are new: only home field separates them.
    assert gaps.iloc[0]["qb_elo_gap"] == pytest.approx(config.NFL_QB_ELO_HOME_EDGE)
    assert state.elo["KC"] > config.NFL_ELO_START > state.elo["DEN"]
    assert state.elo["KC"] + state.elo["DEN"] == pytest.approx(2 * config.NFL_ELO_START)   # what one gains the other loses
    assert state.qb_value["MAHOMES"] > state.new_qb_value() and state.qb_value["MAHOMES"] > state.qb_value["NIX"]
    # A backup the ratings have never seen costs Kansas City; its own starter is worth what he's been above the team's level.
    assert state.qb_adjustment("KC", "BACKUP") < state.qb_adjustment("KC", "MAHOMES")
    assert state.qb_adjustment("KC", None) == 0.0
    assert state.gap("KC", "DEN", "BACKUP", "NIX") < state.gap("KC", "DEN", "MAHOMES", "NIX")
    assert state.gap("KC", "DEN", neutral=True) == pytest.approx(state.gap("KC", "DEN") - config.NFL_QB_ELO_HOME_EDGE)
    table = state.team_table().set_index("team")
    assert table.loc["KC", "qb_elo"] == pytest.approx(state.elo["KC"]) and table.loc["KC", "qb_team_value"] > table.loc["DEN", "qb_team_value"]


def test_a_new_season_pulls_ratings_back_toward_average():
    schedules = pd.DataFrame([_game(2025, 1, "KC", "DEN", 40, 3)])
    during = nfl_qb_elo.rate(schedules, pd.DataFrame())
    after = nfl_qb_elo.rate(schedules, pd.DataFrame(), current_season=2026)
    gap_during = during.elo["KC"] - config.NFL_ELO_START
    assert after.elo["KC"] - config.NFL_ELO_START == pytest.approx(gap_during * config.NFL_ELO_SEASON_KEEP)
    assert nfl_qb_elo.rate(pd.DataFrame(), pd.DataFrame()).elo == {}
    # Without weekly stats the Elo still runs; nobody gets a quarterback adjustment.
    assert during.qb_value == {} and during.qb_adjustment("KC", None) == 0.0


def test_gap_to_probability_is_the_elo_curve_and_the_model_member_adds_its_columns():
    assert nfl_qb_elo.gap_to_probability(0) == pytest.approx(0.5)
    assert nfl_qb_elo.gap_to_probability(400) == pytest.approx(10 / 11)
    X = pd.DataFrame({"qb_elo_base": [100.0, 0.0], "home_qb_elo_adj": [20.0, 0.0], "away_qb_elo_adj": [-30.0, 50.0]})
    p = ml_models.EloGapModel().predict_proba(X)[:, 1]
    assert p[0] == pytest.approx(nfl_qb_elo.gap_to_probability(150.0))      # 100 + 20 - (-30)
    assert p[1] == pytest.approx(nfl_qb_elo.gap_to_probability(-50.0))
    assert ml_models.EloGapModel().predict_proba(X).sum(axis=1) == pytest.approx([1.0, 1.0])


def test_the_closing_line_pulls_the_gap_toward_it_after_the_game():
    # Same result, but the line had the home team a 10-point favorite: the gap ends up wider.
    plain = nfl_qb_elo.rate(pd.DataFrame([_game(2025, 1, "KC", "DEN", 24, 21)]), pd.DataFrame())
    lined = nfl_qb_elo.rate(pd.DataFrame([{**_game(2025, 1, "KC", "DEN", 24, 21), "spread_line": 10.0}]), pd.DataFrame())
    gap = lambda state: state.elo["KC"] - state.elo["DEN"]
    pull = config.NFL_QB_ELO_MARKET_PULL * (10.0 * config.NFL_QB_ELO_POINTS - config.NFL_QB_ELO_HOME_EDGE)
    assert gap(lined) == pytest.approx(gap(plain) + pull)
    # A game with no line on file just skips the pull.
    blank = nfl_qb_elo.rate(pd.DataFrame([{**_game(2025, 1, "KC", "DEN", 24, 21), "spread_line": None}]), pd.DataFrame())
    assert gap(blank) == pytest.approx(gap(plain))


def test_v5_features_are_the_elo_gap_and_each_starters_adjustment():
    assert nfl_game_picks.V5_FEATURE_COLUMNS == nfl_game_picks.QB_ELO_FEATURE_COLUMNS
    assert nfl_game_picks.resolve_feature_builder(nfl_game_picks.V5_FEATURE_COLUMNS) is nfl_game_picks.build_game_features_v5
    schedules = pd.DataFrame([_game(2025, 1, "KC", "DEN", 31, 10, "MAHOMES", "NIX"), _game(2025, 2, "DEN", "KC", 17, 20, "NIX", "MAHOMES")])
    weekly = pd.DataFrame([_line(2025, 1, "KC", "MAHOMES", yards=350, tds=4), _line(2025, 1, "DEN", "NIX", yards=150, tds=0, ints=2),
                           _line(2025, 2, "KC", "MAHOMES", yards=300, tds=2), _line(2025, 2, "DEN", "NIX", yards=200, tds=1)])
    state = nfl_qb_elo.rate(schedules, weekly)
    master = state.team_table()
    master.attrs["qb_elo_state"] = state
    games = pd.DataFrame([
        {"game_id": "g1", "season": 2025, "week": 3, "home_team": "KC", "away_team": "DEN", "home_qb_id": "MAHOMES", "away_qb_id": "NIX"},
        {"game_id": "g2", "season": 2025, "week": 3, "home_team": "KC", "away_team": "DEN", "home_qb_id": "BACKUP", "away_qb_id": nfl_qb_elo.AVERAGE_QB_ID},
    ])
    features = nfl_game_picks.build_game_features_v5(master, None, None, games).set_index("game_id")
    assert features.loc["g1", "qb_elo_base"] == pytest.approx(state.gap("KC", "DEN"))
    assert features.loc["g1", "home_qb_elo_adj"] == pytest.approx(state.qb_adjustment("KC", "MAHOMES"))
    assert features.loc["g2", "home_qb_elo_adj"] == pytest.approx(state.qb_adjustment("KC", "BACKUP"))
    assert features.loc["g2", "home_qb_elo_adj"] < features.loc["g1", "home_qb_elo_adj"]
    assert features.loc["g2", "away_qb_elo_adj"] == pytest.approx(state.average_qb_adjustment("DEN"))
    p = ml_models.EloGapModel().predict_proba(features[nfl_game_picks.V5_FEATURE_COLUMNS])[:, 1]
    assert p[0] == pytest.approx(nfl_qb_elo.gap_to_probability(state.gap("KC", "DEN", "MAHOMES", "NIX")))
    assert p[1] < p[0]                                         # the backup against an average quarterback
    # No rating on the team table: blank features, and the pick falls back to the heuristic.
    assert nfl_game_picks.qb_elo_state(pd.DataFrame({"team": ["KC"]})) is None
    blank = nfl_game_picks.build_game_features_v5(pd.DataFrame({"team": ["KC", "DEN"]}), None, None, games)
    assert blank[nfl_game_picks.V5_FEATURE_COLUMNS].isna().all().all()


def test_the_saved_model_is_v5_and_its_walk_forward_file_shows_the_gain():
    artifact = ml_models.load_model(config.NFL_GAME_PICK_WIN_PROBABILITY_MODEL_PATH)
    assert artifact["feature_columns"] == nfl_game_picks.V5_FEATURE_COLUMNS
    members = artifact["model"].members
    assert len(members) == 1 and isinstance(members[0][0], ml_models.EloGapModel)
    assert config.NFL_GAME_PICK_MODEL_VERSION == "v5"
    scored = pd.read_csv("data/backtest/nfl_win_model_v4_walk_forward.csv").dropna(subset=["v5", "market"])
    scored = scored[scored["home_won"].isin([0, 1])]
    loss = lambda column: float(-(scored["home_won"] * np.log(scored[column]) + (1 - scored["home_won"]) * np.log(1 - scored[column])).mean())
    assert loss("v5") < loss("new") - 0.008          # the recorded result: 0.6197 against 0.6307
    assert loss("market") < loss("v5")               # and Vegas is still ahead


def _season_of_starts(team="KC", opponent="DEN", starter="MAHOMES", games=5, season=2025):
    """`team` wins `games` home games with `starter`, the opponent using one quarterback throughout."""
    return pd.DataFrame([_game(season, week, team, opponent, 30, 10, starter, "NIX") for week in range(1, games + 1)])


def test_a_backup_for_the_regular_starter_is_marked_down_and_an_average_starter_is_not():
    state = nfl_qb_elo.rate(_season_of_starts(), pd.DataFrame())
    assert state.regular_starter("KC") == "MAHOMES" and state.wins["KC"] == 5 and state.wins["DEN"] == 0
    assert state.backup_penalty("KC", "MAHOMES") == 0.0
    assert state.backup_penalty("KC", "BACKUP") == config.NFL_QB_ELO_BACKUP_PENALTY
    # With no box scores every quarterback is rated the same, so the penalty is the whole difference.
    assert state.qb_adjustment("KC", "MAHOMES") - state.qb_adjustment("KC", "BACKUP") == pytest.approx(config.NFL_QB_ELO_BACKUP_PENALTY)
    assert state.qb_adjustment("KC", nfl_qb_elo.AVERAGE_QB_ID) == pytest.approx(state.average_qb_adjustment("KC"))
    # Before anyone has config.NFL_QB_ELO_STARTER_MIN_STARTS starts there is no regular starter to stand in for.
    early = nfl_qb_elo.rate(_season_of_starts(games=config.NFL_QB_ELO_STARTER_MIN_STARTS - 1), pd.DataFrame())
    assert early.regular_starter("KC") is None and early.backup_penalty("KC", "BACKUP") == 0.0
    # A new season starts the count again.
    rolled = nfl_qb_elo.rate(_season_of_starts(), pd.DataFrame(), current_season=2026)
    assert rolled.regular_starter("KC") is None and rolled.wins == {}


def test_a_winning_team_sitting_its_starter_in_the_final_week_is_resting_everyone():
    wins = config.NFL_QB_ELO_SITTING_MIN_WINS
    state = nfl_qb_elo.rate(_season_of_starts(games=wins), pd.DataFrame())
    both = config.NFL_QB_ELO_BACKUP_PENALTY + config.NFL_QB_ELO_SITTING_PENALTY
    assert state.backup_penalty("KC", "BACKUP", final_week=True) == both
    assert state.backup_penalty("KC", "BACKUP", final_week=False) == config.NFL_QB_ELO_BACKUP_PENALTY
    assert state.backup_penalty("KC", "MAHOMES", final_week=True) == 0.0                 # the starter plays: nothing
    assert state.gap("KC", "DEN", "MAHOMES", "NIX", final_week=True) - state.gap("KC", "DEN", "BACKUP", "NIX", final_week=True) \
        == pytest.approx(both)
    # A team short of the wins is only starting a backup. (Denver lost them all.)
    short = nfl_qb_elo.rate(_season_of_starts(games=wins - 1), pd.DataFrame())
    assert short.backup_penalty("KC", "BACKUP", final_week=True) == config.NFL_QB_ELO_BACKUP_PENALTY
    assert state.backup_penalty("DEN", "STIDHAM", final_week=True) == config.NFL_QB_ELO_BACKUP_PENALTY


def test_final_week_is_the_last_of_the_regular_season():
    assert nfl_qb_elo.final_regular_week(2020) == 17 and nfl_qb_elo.final_regular_week(2021) == 18
    assert nfl_qb_elo.is_final_week(2026, 18) and not nfl_qb_elo.is_final_week(2026, 17)
    assert nfl_qb_elo.is_final_week(2019, 17) and not nfl_qb_elo.is_final_week(2019, 17, "WC")
    assert not nfl_qb_elo.is_final_week(float("nan"), 18)


def test_rest_moves_the_gap_and_is_capped():
    state = nfl_qb_elo.QbEloState()
    level = state.gap("KC", "DEN")
    assert state.gap("KC", "DEN", rest_diff=3) == pytest.approx(level + 3 * config.NFL_QB_ELO_REST_PER_DAY)
    assert state.gap("KC", "DEN", rest_diff=-3) == pytest.approx(level - 3 * config.NFL_QB_ELO_REST_PER_DAY)
    cap = config.NFL_QB_ELO_REST_CAP_DAYS
    assert state.gap("KC", "DEN", rest_diff=30) == pytest.approx(level + cap * config.NFL_QB_ELO_REST_PER_DAY)
    assert state.gap("KC", "DEN", rest_diff=float("nan")) == pytest.approx(level)
    # rate() reads the schedule's rest days when they are there.
    games = pd.DataFrame([{**_game(2025, 1, "KC", "DEN", 20, 17), "home_rest": 14, "away_rest": 7}])
    _, gaps = nfl_qb_elo.rate(games, pd.DataFrame(), record=True)
    assert gaps.iloc[0]["qb_elo_gap"] == pytest.approx(config.NFL_QB_ELO_HOME_EDGE + 7 * config.NFL_QB_ELO_REST_PER_DAY)


def test_live_features_carry_rest_the_neutral_field_and_the_final_week():
    wins = config.NFL_QB_ELO_SITTING_MIN_WINS
    state = nfl_qb_elo.rate(_season_of_starts(games=wins, season=2026), pd.DataFrame())
    master = pd.DataFrame({"team": ["KC", "DEN"]})
    master.attrs["qb_elo_state"] = state
    games = pd.DataFrame([
        {"game_id": "final", "season": 2026, "week": 18, "game_type": "REG", "home_team": "KC", "away_team": "DEN",
         "home_qb_id": "BACKUP", "away_qb_id": "NIX", "location": "Home", "home_rest": 7, "away_rest": 10},
        {"game_id": "earlier", "season": 2026, "week": 12, "game_type": "REG", "home_team": "KC", "away_team": "DEN",
         "home_qb_id": "BACKUP", "away_qb_id": "NIX", "location": "Neutral", "home_rest": 7, "away_rest": 7},
    ])
    features = nfl_game_picks.build_game_features_v5(master, pd.DataFrame(), pd.DataFrame(), games).set_index("game_id")
    assert features.loc["final", "qb_elo_base"] == pytest.approx(state.gap("KC", "DEN", rest_diff=-3))
    assert features.loc["earlier", "qb_elo_base"] == pytest.approx(state.gap("KC", "DEN", neutral=True))
    assert features.loc["final", "home_qb_elo_adj"] == pytest.approx(state.qb_adjustment("KC", "BACKUP", final_week=True))
    assert features.loc["earlier", "home_qb_elo_adj"] == pytest.approx(state.qb_adjustment("KC", "BACKUP"))
    assert features.loc["earlier", "home_qb_elo_adj"] - features.loc["final", "home_qb_elo_adj"] \
        == pytest.approx(config.NFL_QB_ELO_SITTING_PENALTY)


def test_a_listed_starter_who_never_threw_a_pass_gives_way_to_the_one_who_did():
    # The feed lists LOCK for weeks 2-5, but DARNOLD threw every pass.
    schedules = pd.DataFrame([_game(2026, week, "SEA", "LA", 24, 20, "LOCK" if week > 1 else "DARNOLD", "STAFFORD") for week in range(1, 6)])
    weekly = pd.DataFrame([_line(2026, week, "SEA", "DARNOLD", yards=320, tds=3) for week in range(1, 6)]
                          + [_line(2026, week, "LA", "STAFFORD") for week in range(1, 6)])
    state = nfl_qb_elo.rate(schedules, weekly)
    assert state.starts["SEA"] == {"DARNOLD": 5} and state.regular_starter("SEA") == "DARNOLD"
    assert "LOCK" not in state.qb_value and state.qb_value["DARNOLD"] > state.new_qb_value()
    # A starter knocked out early still started: he threw a pass, so he keeps the start.
    hurt = pd.DataFrame([_line(2026, 1, "SEA", "DARNOLD", attempts=3, completions=2, yards=20, tds=0),
                         _line(2026, 1, "SEA", "LOCK", attempts=30)])
    state = nfl_qb_elo.rate(schedules.iloc[:1], hurt)
    assert state.starts["SEA"] == {"DARNOLD": 1}
    # With no box score for the game the listed starter stands.
    assert nfl_qb_elo.rate(schedules.iloc[:1], pd.DataFrame()).starts["SEA"] == {"DARNOLD": 1}
