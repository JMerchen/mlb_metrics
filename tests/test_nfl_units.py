import pandas as pd
import pytest

from mlb_metrics import config, nfl_units


def _play(game_id, offense, defense, epa, kind, week=1, **extra):
    row = {
        "game_id": game_id, "season": 2025, "week": week, "season_type": "REG", "posteam": offense, "defteam": defense,
        "epa": epa, "pass_attempt": 0, "sack": 0, "receiver_player_id": None, "rush": 0, "qb_scramble": 0, "qb_dropback": 0,
    }
    if kind == "target":
        row.update(pass_attempt=1, qb_dropback=1, receiver_player_id="wr")
    elif kind == "sack":
        row.update(pass_attempt=1, qb_dropback=1, sack=1)
    elif kind == "scramble":
        row.update(rush=1, qb_scramble=1, qb_dropback=1)
    elif kind == "run":
        row.update(rush=1)
    row.update(extra)
    return row


def test_unit_game_epa_splits_targets_designed_runs_and_dropbacks():
    pbp = pd.DataFrame([
        _play("g1", "KC", "DEN", 0.5, "target"),
        _play("g1", "KC", "DEN", -2.0, "sack"),
        _play("g1", "KC", "DEN", 1.0, "scramble"),
        _play("g1", "KC", "DEN", 0.2, "run"),
        _play("g1", "KC", "DEN", -0.4, "run"),
    ])
    row = nfl_units.unit_game_epa(pbp).iloc[0]
    assert (row["team"], row["opp"]) == ("KC", "DEN")
    assert row["targets"] == 1 and row["target_epa"] == pytest.approx(0.5)
    assert row["runs"] == 2 and row["run_epa"] == pytest.approx(-0.2)   # the scramble is a dropback, not a run
    assert row["dropbacks"] == 3 and row["dropback_epa"] == pytest.approx(-0.5)


def test_team_unit_ratings_put_average_at_100_and_flip_defenses(monkeypatch):
    monkeypatch.setattr(config, "NFL_UNIT_SHRINK_PLAYS", 0.0)
    monkeypatch.setattr(config, "NFL_TOTALS_WINDOWS", [(None, 1.0)])
    games = pd.DataFrame([
        {"game_id": "g1", "season": 2025, "week": 1, "team": "KC", "opp": "DEN", "target_epa": 3.0, "targets": 10,
         "run_epa": 1.0, "runs": 10, "dropback_epa": 3.0, "dropbacks": 10},
        {"game_id": "g1", "season": 2025, "week": 1, "team": "DEN", "opp": "KC", "target_epa": -1.0, "targets": 10,
         "run_epa": -1.0, "runs": 10, "dropback_epa": -1.0, "dropbacks": 10},
        {"game_id": "g2", "season": 2025, "week": 1, "team": "BUF", "opp": "NYJ", "target_epa": 1.0, "targets": 10,
         "run_epa": 0.0, "runs": 10, "dropback_epa": 1.0, "dropbacks": 10},
        {"game_id": "g2", "season": 2025, "week": 1, "team": "NYJ", "opp": "BUF", "target_epa": 1.0, "targets": 10,
         "run_epa": 0.0, "runs": 10, "dropback_epa": 1.0, "dropbacks": 10},
    ])
    qb_games = pd.DataFrame([{"game_id": "g1", "season": 2025, "week": 1, "qb": "mahomes", "epa_sum": 3.0, "dropbacks": 10}])
    starters = pd.DataFrame({"team": ["KC", "DEN", "BUF", "NYJ"], "opponent": ["DEN", "KC", "NYJ", "BUF"],
                             "qb_id": ["mahomes", None, None, None], "qb_name": ["Patrick Mahomes", None, None, None]})

    ratings = nfl_units.team_unit_ratings(games, qb_games, starters).set_index("team")

    # The unadjusted season view (one game each can't separate an offense
    # from the defense it faced, so the schedule-adjusted views can't either).
    assert ratings["receiving_plus_season"].mean() == pytest.approx(100, abs=0.5)
    assert ratings["receiving_plus_season"].std() == pytest.approx(config.NFL_UNIT_PLUS_POINTS_PER_SD, abs=1)
    assert ratings.loc["KC", "receiving_plus_season"] > 100 > ratings.loc["DEN", "receiving_plus_season"]
    # KC's offense put up the most per dropback against DEN, so DEN's pass defense rates worst
    assert ratings.loc["DEN", "pass_defense_plus_season"] == ratings["pass_defense_plus_season"].min()
    assert ratings.loc["KC", "qb_plus_season"] == ratings["qb_plus_season"].max()
    assert ratings.loc["KC", "qb_name"] == "Patrick Mahomes"


def test_listed_starters_fill_byes_from_each_teams_latest_game():
    week = pd.DataFrame([{"season": 2026, "week": 7, "home_team": "KC", "away_team": "DEN",
                          "home_qb_id": "m", "away_qb_id": "n", "home_qb_name": "Patrick Mahomes", "away_qb_name": "Bo Nix"}])
    history = pd.DataFrame([
        {"season": 2026, "week": 5, "home_team": "BUF", "away_team": "OAK", "home_qb_id": "a", "away_qb_id": "x",
         "home_qb_name": "Josh Allen", "away_qb_name": "Old QB"},
        {"season": 2026, "week": 6, "home_team": "LV", "away_team": "BUF", "home_qb_id": "c", "away_qb_id": "a2",
         "home_qb_name": "Kirk Cousins", "away_qb_name": "Mitchell Trubisky"},
    ])
    listed = nfl_units.listed_starters(week, history).set_index("team")
    assert listed.loc["KC", "opponent"] == "DEN" and listed.loc["DEN", "qb_name"] == "Bo Nix"
    assert listed.loc["KC", "home"] == 1.0 and listed.loc["DEN", "home"] == 0.0 and pd.isna(listed.loc["BUF", "home"])
    assert listed.loc["BUF", "qb_name"] == "Mitchell Trubisky" and pd.isna(listed.loc["BUF", "opponent"])
    assert listed.loc["LV", "qb_name"] == "Kirk Cousins"   # OAK in week 5 is the same franchise, latest wins


def _unit_game(team, opp, rate, season=2026, week=1, plays=10):
    return {"game_id": f"{season}_{week}_{team}_{opp}", "season": season, "week": week, "team": team, "opp": opp,
            "target_epa": rate * plays, "targets": plays, "run_epa": rate * plays, "runs": plays,
            "dropback_epa": rate * plays, "dropbacks": plays}


def _starters(teams):
    return pd.DataFrame({"team": teams, "opponent": None, "home": None, "qb_id": None, "qb_name": None})


def test_strength_of_schedule_credits_production_against_a_strong_defense(monkeypatch):
    monkeypatch.setattr(config, "NFL_UNIT_SHRINK_PLAYS", 0.0)
    # C faced both defenses: X gave it 0.30 a target, Y nothing - X is weak, Y strong.
    # A only played X (0.30), B only played Y (0.10).
    games = pd.DataFrame([
        _unit_game("A", "X", 0.30, week=1), _unit_game("C", "X", 0.30, week=1),
        _unit_game("B", "Y", 0.10, week=2), _unit_game("C", "Y", 0.00, week=2),
    ])
    ratings = nfl_units.team_unit_ratings(games, pd.DataFrame(), _starters(["A", "B", "C", "X", "Y"])).set_index("team")

    assert ratings.loc["A", "receiving_plus_season"] > ratings.loc["B", "receiving_plus_season"]
    assert ratings.loc["B", "receiving_plus_season_sos"] > ratings.loc["A", "receiving_plus_season_sos"]
    # and the defenses: Y allowed less than X either way
    assert ratings.loc["Y", "pass_defense_plus_season_sos"] > ratings.loc["X", "pass_defense_plus_season_sos"]
    # the plain columns are the default view
    assert ratings["receiving_plus"].equals(ratings[f"receiving_plus_{nfl_units.DEFAULT_VIEW}"])


def test_season_view_uses_this_season_only_and_recency_reaches_back(monkeypatch):
    monkeypatch.setattr(config, "NFL_UNIT_SHRINK_PLAYS", 0.0)
    # A was great last season and poor this season; B the reverse.
    games = pd.DataFrame(
        [_unit_game("A", "Z", 0.40, season=2025, week=w) for w in range(1, 18)]
        + [_unit_game("B", "Z", -0.20, season=2025, week=w) for w in range(1, 18)]
        + [_unit_game("A", "Z", 0.00, season=2026, week=1), _unit_game("B", "Z", 0.10, season=2026, week=1)]
    )
    ratings = nfl_units.team_unit_ratings(games, pd.DataFrame(), _starters(["A", "B", "Z"])).set_index("team")

    assert ratings.loc["B", "receiving_plus_season"] > ratings.loc["A", "receiving_plus_season"]
    assert ratings.loc["A", "receiving_plus_recency"] > ratings.loc["B", "receiving_plus_recency"]
