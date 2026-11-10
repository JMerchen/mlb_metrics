import numpy as np
import pandas as pd
import pytest

from tests.cbb_helpers import REGIONS, bracket_schedule, game
from mlb_metrics import cbb_bracket, cbb_ratings, config

LEAGUE = {"efficiency": 105.0, "home_edge": 2.5, "tempo": 68.0, "games": 100}


def _ratings(team_ids, net):
    return pd.DataFrame({"team_id": team_ids, "offense": np.asarray(net) / 2, "defense": np.asarray(net) / 2,
                         "net": net, "tempo": 68.0, "games": 10, "wins": 5, "losses": 5})


def _league_of(n_conferences=31, per_conference=6):
    """Conferences 1..N from strongest to weakest, six teams each: team
    100*conference + k is the conference's k-th best. Conference 1's worst
    team is about as good as conference 5's best."""
    rows = [(100 * c + k, c, 32.0 - 2.0 * c - 1.5 * k) for c in range(1, n_conferences + 1) for k in range(1, per_conference + 1)]
    teams = pd.DataFrame({"team_id": [r[0] for r in rows], "conference_id": [r[1] for r in rows]})
    return _ratings([r[0] for r in rows], [r[2] for r in rows]), teams


def test_projected_field_is_each_conferences_best_team_plus_the_best_of_the_rest():
    ratings, teams = _league_of()
    field = cbb_bracket.projected_field(ratings, teams)
    assert list(field.columns) == cbb_bracket.FIELD_COLUMNS and len(field) == 68 and field["team_id"].is_unique
    auto, at_large = field[field["bid"] == "auto"], field[field["bid"] == "at-large"]
    assert sorted(auto["team_id"]) == [100 * c + 1 for c in range(1, 32)] and len(at_large) == 37
    # At-large bids go to the best teams left, whatever the conference: the weakest one in is better than the best one out.
    net = ratings.set_index("team_id")["net"]
    left_out = net.drop(field["team_id"])
    assert net[at_large["team_id"]].min() >= left_out.max()
    # First Four: the four weakest at-large teams and the four weakest conference champions.
    play_in = field[field["play_in"]]
    assert len(play_in) == 8
    assert set(play_in.loc[play_in["bid"] == "auto", "team_id"]) == {2801, 2901, 3001, 3101}
    assert set(play_in.loc[play_in["bid"] == "auto", "seed"]) == {16}
    assert net[play_in.loc[play_in["bid"] == "at-large", "team_id"]].max() <= net[at_large.loc[~at_large["play_in"], "team_id"]].min()
    # 64 spots: four to a seed line in four regions, a First Four pair sharing one.
    assert field.groupby(["region", "seed"]).size().index.nunique() == 64
    assert field.groupby("region").size().tolist() == [17, 17, 17, 17]
    assert field.groupby(["region", "seed"]).size().max() == 2
    # The four best teams are the 1 seeds, one a region, and the snake gives the best of them the weakest 2 seed.
    assert field["team_id"].iloc[0] == 101
    ones, twos = field[field["seed"] == 1], field[field["seed"] == 2]
    assert ones["region"].nunique() == 4 and net[ones["team_id"]].min() >= net[twos["team_id"]].max()
    best_region = ones.loc[ones["team_id"] == 101, "region"].iloc[0]
    assert net[twos.loc[twos["region"] == best_region, "team_id"]].iloc[0] == net[twos["team_id"]].min()


def test_projected_field_leaves_an_independent_to_earn_an_at_large_bid():
    ratings, teams = _league_of()
    teams.loc[teams["team_id"] == 3101, "conference_id"] = cbb_bracket.INDEPENDENTS_CONFERENCE_ID
    field = cbb_bracket.projected_field(ratings, teams)
    # The weakest conference's best team is now an independent: too weak for an at-large bid, and its old
    # conference sends its next-best team.
    assert 3101 not in set(field["team_id"]) and 3102 in set(field["team_id"]) and len(field) == 68
    assert (field["bid"] == "auto").sum() == 31


def test_official_field_reads_the_posted_bracket_with_its_first_four():
    schedule, where = bracket_schedule()
    field = cbb_bracket.official_field(schedule)
    assert len(field) == 68 and field["play_in"].sum() == 8
    for _, row in field.iterrows():
        assert (row["region"], row["seed"]) == where[row["team_id"]]
    assert set(field.loc[field["play_in"], "seed"]) == {16}
    # Once the First Four is played the winner appears in the round of 64 too: still one row a team.
    played, _ = bracket_schedule(first_four_done=True)
    again = cbb_bracket.official_field(played)
    assert len(again) == 68 and again["team_id"].is_unique and again["play_in"].sum() == 8


def test_official_field_waits_for_the_whole_bracket():
    schedule, _ = bracket_schedule()
    assert cbb_bracket.official_field(schedule[~schedule["notes_headline"].str.contains("Midwest")]) is None
    assert cbb_bracket.official_field(schedule.iloc[:20]) is None
    regular = pd.DataFrame([game(1, "2026-11-10", 1, 2, 80, 70)])
    assert cbb_bracket.official_field(regular) is None
    assert cbb_bracket.official_field(regular.drop(columns=["home_current_rank"])) is None


def test_final_four_pairs_for_a_projection_from_config_from_the_games_or_assumed(monkeypatch):
    assert cbb_bracket.final_four_pairs(2027, "projected") == ([["East", "Midwest"], ["West", "South"]], True)
    assert cbb_bracket.final_four_pairs(2027, "official") == ([["East", "West"], ["South", "Midwest"]], False)
    monkeypatch.setattr(config, "CBB_FINAL_FOUR_PAIRS", {2027: [["East", "South"], ["West", "Midwest"]]})
    assert cbb_bracket.final_four_pairs(2027, "official") == ([["East", "South"], ["West", "Midwest"]], True)
    monkeypatch.setattr(config, "CBB_FINAL_FOUR_PAIRS", {})
    schedule, _ = bracket_schedule(first_four_done=True)
    field = cbb_bracket.official_field(schedule)
    semifinal = pd.DataFrame([game(900, "2027-04-03", 101, 401, 0, 0, done=False, season_type=3,
                                   note="NCAA Men's Basketball Championship - Final Four")])
    pairs, known = cbb_bracket.final_four_pairs(2027, "official", pd.concat([schedule, semifinal], ignore_index=True), field)
    assert known and pairs == [["East", "Midwest"], ["West", "South"]]


def _official(strong=None):
    schedule, where = bracket_schedule()
    field = cbb_bracket.official_field(schedule)
    net = {team: 20.0 - 2.5 * seed for team, (region, seed) in where.items()}      # better seeds are better teams
    net.update(strong or {})
    return schedule, field, _ratings(list(net), list(net.values()))


def test_simulate_counts_how_often_each_team_reaches_each_round():
    _, field, ratings = _official()
    pairs = [["East", "West"], ["South", "Midwest"]]
    odds = cbb_bracket.simulate(field, ratings, LEAGUE, pairs, sims=4000, seed=1).set_index("team_id")
    assert list(odds.columns) == cbb_bracket.ROUND_COLUMNS
    assert odds["champion"].sum() == pytest.approx(1.0) and odds["final_four"].sum() == pytest.approx(4.0)
    assert odds["round_of_32"].sum() == pytest.approx(32.0)
    play_in = field.set_index("team_id")["play_in"]
    assert (odds.loc[~play_in.reindex(odds.index), "round_of_64"] == 1.0).all()
    assert odds.at[116, "round_of_64"] + odds.at[117, "round_of_64"] == pytest.approx(1.0)     # a First Four pair
    assert (odds.diff(axis=1).iloc[:, 1:] <= 1e-12).all().all()                                 # nobody gets likelier by round
    # The four 1 seeds are equal here, so each wins about a quarter of the time, and a 1 beats a 16 nearly always.
    assert odds.loc[[101, 201, 301, 401], "champion"].sum() > 0.5
    assert odds.at[101, "round_of_32"] > 0.97 and odds.at[102, "sweet_16"] > odds.at[107, "sweet_16"]
    # Same seed, same result.
    again = cbb_bracket.simulate(field, ratings, LEAGUE, pairs, sims=4000, seed=1).set_index("team_id")
    pd.testing.assert_frame_equal(odds, again)


def test_simulate_takes_played_games_as_they_finished_and_follows_the_semifinal_pairing():
    _, field, ratings = _official(strong={101: 80.0, 201: 60.0})     # East's and West's 1 seeds tower over the field
    upset = {(101, 116): 116, (116, 101): 116, (101, 117): 117, (117, 101): 117}      # ...but East's lost to its 16 seed
    pairs = [["East", "West"], ["South", "Midwest"]]
    odds = cbb_bracket.simulate(field, ratings, LEAGUE, pairs, results=upset, sims=2000).set_index("team_id")
    assert odds.at[101, "round_of_32"] == 0.0 and odds.at[101, "champion"] == 0.0
    assert odds.at[116, "round_of_32"] + odds.at[117, "round_of_32"] == pytest.approx(1.0)
    assert odds.at[201, "champion"] > 0.9
    # With no upset the two giants meet in a semifinal when their regions are paired, in the final when they aren't.
    paired = cbb_bracket.simulate(field, ratings, LEAGUE, pairs, sims=2000).set_index("team_id")
    apart = cbb_bracket.simulate(field, ratings, LEAGUE, [["East", "South"], ["West", "Midwest"]], sims=2000).set_index("team_id")
    assert paired.at[201, "final"] < 0.2 and apart.at[201, "final"] > 0.9


def test_played_results_and_win_matrix():
    done = cbb_ratings.prepare_games(pd.DataFrame([
        game(1, "2027-03-18", 101, 116, 60, 70, neutral=True, season_type=3, home_seed=1, away_seed=16,
             note="NCAA Men's Basketball Championship - East Region - 1st Round"),
        game(2, "2027-01-05", 101, 116, 90, 50),
    ]))
    assert cbb_bracket.played_results(done) == {(101, 116): 116, (116, 101): 116}
    ratings = _ratings([1, 2, 3], [20.0, 0.0, 0.0])
    P = cbb_bracket.win_matrix([1, 2, 3, 4], ratings, LEAGUE, results={(2, 3): 3, (3, 2): 3})
    assert P[0, 1] > 0.85 and P[0, 1] + P[1, 0] == pytest.approx(1.0)
    assert P[1, 2] == 0.0 and P[2, 1] == 1.0
    assert P[1, 3] == pytest.approx(0.5)             # a team the ratings don't know counts as the weakest rated
