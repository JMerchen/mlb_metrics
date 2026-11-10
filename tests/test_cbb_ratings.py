import numpy as np
import pandas as pd
import pytest

from tests.cbb_helpers import box_for, game, round_robin
from mlb_metrics import cbb_ratings, config

STRENGTH = {1: (10.0, 8.0), 2: (4.0, 2.0), 3: (0.0, 0.0), 4: (-3.0, -5.0), 5: (-11.0, -5.0)}


def test_tournament_round_reads_round_and_region_from_the_note():
    assert cbb_ratings.tournament_round("NCAA Men's Basketball Championship - East Region - 1st Round") == ("Round of 64", "East")
    # "Midwest Region" is not the West.
    assert cbb_ratings.tournament_round("Men's Basketball Championship - Midwest Region - Sweet 16") == ("Sweet 16", "Midwest")
    assert cbb_ratings.tournament_round("MEN'S BASKETBALL CHAMPIONSHIP - SOUTH REGION - ELITE 8 AT HOUSTON TX") == ("Elite 8", "South")
    assert cbb_ratings.tournament_round("Men's Basketball Championship - West Region - First Four") == ("First Four", "West")
    assert cbb_ratings.tournament_round("Men's Basketball Championship - Final Four") == ("Final Four", None)
    assert cbb_ratings.tournament_round("Men's Basketball Championship - National Championship") == ("Championship", None)
    for other in ("NIT - 1st Round", "SEC Tournament", "NCAA Women's Basketball Championship - 1st Round", None, float("nan")):
        assert cbb_ratings.tournament_round(other) == (None, None)


def test_prepare_games_takes_pace_from_the_box_score_and_marks_division_one():
    schedules = pd.DataFrame([
        game(1, "2026-11-10", 1, 2, 80, 70),
        game(2, "2026-11-11", 1, 9, 95, 50, away_conf=None),           # a non-Division I visitor
        game(3, "2026-11-12", 2, 3, 0, 0, done=False),                  # not played yet
        game(4, "2026-11-13", 2, 3, 66, 60, neutral=True),              # no box score
    ])
    box = box_for(schedules[schedules["game_id"].isin([1, 2])], possessions=70)
    games = cbb_ratings.prepare_games(schedules, box)
    assert list(games.columns) == cbb_ratings.GAME_COLUMNS
    assert games["game_id"].tolist() == ["1", "2", "4"]
    by_id = games.set_index("game_id")
    # 50 attempts - 9 offensive rebounds + 10 turnovers + 0.475 x 40 free throws
    assert by_id.at["1", "possessions"] == pytest.approx(70.0)
    assert by_id.at["4", "possessions"] == pytest.approx(63 / config.CBB_POINTS_PER_POSSESSION)
    assert by_id.at["2", "away_d1"] == False and by_id.at["1", "away_d1"] == True  # noqa: E712
    assert by_id.at["4", "neutral"] == True  # noqa: E712
    assert not games["tournament"].any() and games["home_seed"].isna().all()   # the AP rank isn't a seed


def test_prepare_games_names_tournament_rounds_including_the_2011_to_2015_wording():
    note = "MEN'S BASKETBALL CHAMPIONSHIP - EAST REGION - "
    modern = pd.DataFrame([
        game(1, "2026-03-17", 1, 2, 70, 60, note=note + "First Four", season_type=3, home_seed=16, away_seed=16),
        game(2, "2026-03-19", 3, 1, 80, 60, note=note + "1st Round", season_type=3, home_seed=1, away_seed=16),
        game(3, "2026-03-21", 3, 4, 80, 60, note=note + "2nd Round", season_type=3, home_seed=1, away_seed=8),
        game(4, "2026-03-25", 3, 5, 80, 60, note="NIT - 2nd Round", season_type=3),
    ])
    games = cbb_ratings.prepare_games(modern).set_index("game_id")
    assert games["round"].tolist()[:3] == ["First Four", "Round of 64", "Round of 32"] and pd.isna(games["round"].iloc[3])
    assert games.at["2", "home_seed"] == 1 and games.at["2", "away_seed"] == 16 and games.at["2", "region"] == "East"
    assert pd.isna(games.at["4", "home_seed"]) and not games.at["4", "tournament"]
    # 2011-15: the play-in games were the "1st Round", the round of 64 the "2nd" and the round of 32 the "3rd".
    old = pd.DataFrame([
        game(1, "2013-03-19", 1, 2, 70, 60, note=note + "1ST ROUND", season_type=3, season=2013),
        game(2, "2013-03-21", 3, 1, 80, 60, note=note + "2ND ROUND", season_type=3, season=2013),
        game(3, "2013-03-23", 3, 4, 80, 60, note=note + "3RD ROUND", season_type=3, season=2013),
    ])
    assert cbb_ratings.prepare_games(old)["round"].tolist() == ["First Four", "Round of 64", "Round of 32"]


def test_fit_ratings_recovers_each_teams_offense_defense_and_the_home_edge():
    schedules = round_robin(STRENGTH, home_edge=2.0)
    games = cbb_ratings.prepare_games(schedules, box_for(schedules))
    ratings, league = cbb_ratings.fit_ratings(games, ridge=1e-9, home_edge_games=0.0)
    r = ratings.set_index("team_id")
    assert league["home_edge"] == pytest.approx(2.0, abs=1e-4)
    assert league["tempo"] == pytest.approx(70.0, abs=1e-4)
    # Ratings are relative to the league: differences between teams match.
    for team, (offense, defense) in STRENGTH.items():
        assert r.at[team, "offense"] - r.at[3, "offense"] == pytest.approx(offense, abs=1e-3)
        assert r.at[team, "defense"] - r.at[3, "defense"] == pytest.approx(defense, abs=1e-3)
        assert r.at[team, "net"] == pytest.approx(r.at[team, "offense"] + r.at[team, "defense"])
    assert r["games"].tolist() == [8] * 5 and (r["wins"] + r["losses"]).tolist() == [8] * 5
    assert r.at[1, "wins"] == 8 and r.at[5, "losses"] == 8


def test_fit_ratings_counts_only_division_one_games_before_the_date():
    schedules = round_robin(STRENGTH)
    extra = pd.DataFrame([game(9001, "2026-12-30", 5, 99, 150, 20, away_conf=None),      # a rout of a lower-division team
                          game(9002, "2027-02-01", 5, 1, 140, 40)])                       # after the date
    games = cbb_ratings.prepare_games(pd.concat([schedules, extra], ignore_index=True))
    base, _ = cbb_ratings.fit_ratings(cbb_ratings.prepare_games(schedules))
    cut, league = cbb_ratings.fit_ratings(games, as_of="2027-01-15")
    assert league["games"] == 20
    pd.testing.assert_frame_equal(cut, base)
    later, _ = cbb_ratings.fit_ratings(games)
    assert later.set_index("team_id").at[5, "net"] > base.set_index("team_id").at[5, "net"]


def test_fit_ratings_starts_from_last_seasons_rating_and_lets_games_take_over():
    prior = pd.DataFrame({"team_id": [1, 2, 7], "offense": [10.0, -10.0, 6.0], "defense": [10.0, -10.0, 4.0],
                          "net": [20.0, -20.0, 10.0], "tempo": [72.0, 64.0, 70.0]})
    empty = cbb_ratings.prepare_games(round_robin(STRENGTH)).iloc[0:0]
    preseason, league = cbb_ratings.fit_ratings(empty, prior=prior, carryover=0.5)
    p = preseason.set_index("team_id")
    assert p.at[1, "net"] == pytest.approx(10.0) and p.at[2, "net"] == pytest.approx(-10.0)
    assert p.at[1, "tempo"] > league["tempo"] > p.at[2, "tempo"]
    assert league["home_edge"] == config.CBB_HOME_EDGE_PRIOR and (p["games"] == 0).all()
    # A team that plays no games this season keeps its prior; one with games moves toward them.
    games = cbb_ratings.prepare_games(round_robin(STRENGTH))
    light, _ = cbb_ratings.fit_ratings(games, prior=prior, carryover=0.5, prior_weight=1.0)
    heavy, _ = cbb_ratings.fit_ratings(games, prior=prior, carryover=0.5, prior_weight=50.0)
    light, heavy = light.set_index("team_id"), heavy.set_index("team_id")
    assert light.at[7, "net"] == pytest.approx(5.0) and light.at[7, "games"] == 0
    # Team 2 is good on the floor (+6 net vs team 3) but its prior says bad: the heavier the prior, the lower it sits.
    assert light.at[2, "net"] > heavy.at[2, "net"]


def test_recent_games_can_count_more_and_one_blowout_can_be_capped():
    schedules = round_robin(STRENGTH)
    late = game(9001, "2026-12-31", 5, 1, 120, 50)               # the worst team routs the best, at the end
    games = cbb_ratings.prepare_games(pd.concat([schedules, pd.DataFrame([late])], ignore_index=True))
    net = lambda **kw: cbb_ratings.fit_ratings(games, **kw)[0].set_index("team_id").at[5, "net"]
    assert net(half_life_days=5.0) > net(half_life_days=None)
    assert net(margin_cap=10.0) < net(margin_cap=None)


def test_predict_games_turns_ratings_and_pace_into_a_margin_and_a_win_chance():
    league = {"efficiency": 105.0, "home_edge": 2.5, "tempo": 68.0, "games": 100}
    p = cbb_ratings.predict_games([20.0, 20.0, 0.0], [0.0, 0.0, 20.0], [70.0, 70.0, 66.0], [66.0, 66.0, 70.0],
                                  [True, False, True], league, sigma=11.0)
    assert p["possessions"].tolist() == [68.0, 68.0, 68.0]
    assert p["margin"].tolist() == pytest.approx([13.6, 13.6 + 5 * 0.68, -13.6])
    assert p["home_win_probability"][0] + p["home_win_probability"][2] == pytest.approx(1.0)
    assert p["home_win_probability"][1] > p["home_win_probability"][0] > 0.85
    even = cbb_ratings.predict_games(5.0, 5.0, 68.0, 68.0, True, league)
    assert even["home_win_probability"][0] == pytest.approx(0.5)


def test_team_names_lists_division_one_teams_once():
    schedules = pd.DataFrame([game(1, "2026-11-10", 1, 2, 80, 70, home_conf=7, away_conf=2),
                              game(2, "2026-11-11", 1, 9, 95, 50, home_conf=7, away_conf=None)])
    teams = cbb_ratings.team_names(schedules)
    assert sorted(teams["team_id"]) == [1, 2]
    assert teams.set_index("team_id").at[1, "conference_id"] == 7 and teams.set_index("team_id").at[1, "team"] == "Team 1"


def test_preseason_ranked_is_the_ap_top_25_at_each_teams_first_game():
    schedules = pd.DataFrame([
        game(1, "2026-11-03", 1, 2, 80, 70, home_seed=3, away_seed=99),
        game(2, "2026-11-20", 2, 3, 80, 70, home_seed=18, away_seed=25),     # team 2 climbed into the poll later
        game(3, "2026-11-25", 4, -1, 0, 0, done=False, home_seed=12, away_seed=99),
    ])
    assert cbb_ratings.preseason_ranked(schedules) == {1, 3, 4}
    assert cbb_ratings.preseason_ranked(schedules.drop(columns=["home_current_rank"])) == set()


def test_a_preseason_ranked_team_starts_above_its_carryover():
    prior = pd.DataFrame({"team_id": [1, 2], "offense": [5.0, 5.0], "defense": [5.0, 5.0], "net": [10.0, 10.0], "tempo": [68.0, 68.0]})
    empty = cbb_ratings.prepare_games(round_robin(STRENGTH)).iloc[0:0]
    start, _ = cbb_ratings.fit_ratings(empty, prior=prior, carryover=0.8, ranked={1, 9}, ranked_bonus=4.0)
    s = start.set_index("team_id")
    assert s.at[1, "net"] == pytest.approx(12.0) and s.at[2, "net"] == pytest.approx(8.0)
    assert s.at[1, "offense"] == pytest.approx(6.0) and s.at[1, "defense"] == pytest.approx(6.0)
    assert s.at[9, "net"] == pytest.approx(4.0)          # ranked with no rating from last season
    none, _ = cbb_ratings.fit_ratings(empty, prior=prior, carryover=0.8)
    assert none.set_index("team_id").at[1, "net"] == pytest.approx(8.0)
