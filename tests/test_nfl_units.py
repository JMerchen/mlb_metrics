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

    assert ratings["receiving_plus"].mean() == pytest.approx(100, abs=0.5)
    assert ratings.loc["KC", "receiving_plus"] > 100 > ratings.loc["DEN", "receiving_plus"]
    # KC's offense put up the most per dropback against DEN, so DEN's pass defense rates worst
    assert ratings.loc["DEN", "pass_defense_plus"] == ratings["pass_defense_plus"].min()
    assert ratings.loc["KC", "qb_plus"] == ratings["qb_plus"].max()
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
    assert listed.loc["BUF", "qb_name"] == "Mitchell Trubisky" and pd.isna(listed.loc["BUF", "opponent"])
    assert listed.loc["LV", "qb_name"] == "Kirk Cousins"   # OAK in week 5 is the same franchise, latest wins
