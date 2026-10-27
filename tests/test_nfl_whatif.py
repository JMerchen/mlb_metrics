import json

import pandas as pd
import pytest

from mlb_metrics import nfl_whatif


def _snap(game, week, team, pid, name, position, offense=0.0, defense=0.0):
    return {"game_id": game, "season": 2026, "game_type": "REG", "week": week, "team": team,
            "pfr_player_id": pid, "player": name, "position": position,
            "offense_pct": offense, "defense_pct": defense}


def test_regular_players_without_rosters_uses_the_teams_last_four_games():
    snaps = []
    for week in range(1, 6):
        game = f"2026_0{week}_GB"
        snaps.append(_snap(game, week, "GB", "lt", "Left Tackle", "T", offense=1.0))
        # The receiver played every snap in weeks 1-3, then missed weeks 4-5.
        if week <= 3:
            snaps.append(_snap(game, week, "GB", "wr", "Receiver", "WR", offense=1.0))
        snaps.append(_snap(game, week, "GB", "cb", "Corner", "CB", defense=0.4))
        snaps.append(_snap(game, week, "GB", "lb", "Backup", "LB", defense=0.2))
    regulars = nfl_whatif.regular_players(pd.DataFrame(snaps)).set_index("pfr_player_id")

    assert regulars.loc["lt", "group"] == "OL"
    assert regulars.loc["lt", "share"] == pytest.approx(1.0)
    # Weeks 2-5: played 2 of 4 at 100% -> 0.5.
    assert regulars.loc["wr", "share"] == pytest.approx(0.5)
    assert regulars.loc["cb", "share"] == pytest.approx(0.4)  # rotational, still listed
    assert "lb" not in regulars.index  # under LISTED_SHARE
    assert "cb" not in nfl_whatif.regular_players(pd.DataFrame(snaps), min_share=0.5)["pfr_player_id"].values


def _roster(week, team, pfr_id, name, status="ACT", position="CB"):
    return {"season": 2026, "week": week, "team": team, "pfr_id": pfr_id, "full_name": name,
            "status": status, "position": position}


def test_regular_players_follow_a_traded_player_and_fill_missing_lineman_ids():
    snaps, roster = [], []
    for week in range(1, 6):
        for team in ("PIT", "DAL"):
            snaps.append(_snap(f"2026_0{week}_{team}", week, team, f"qb_{team}", "Filler", "QB", offense=1.0))
        # Porter: every snap for Pittsburgh through week 3, traded, inactive
        # week 4, then 60% for Dallas in week 5.
        if week <= 3:
            snaps.append(_snap(f"2026_0{week}_PIT", week, "PIT", "porter", "Joey Porter Jr.", "CB", defense=1.0))
        if week == 5:
            snaps.append(_snap("2026_05_DAL", week, "DAL", "porter", "Joey Porter Jr.", "CB", defense=0.6))
        snaps.append(_snap(f"2026_0{week}_DAL", week, "DAL", "smith", "Tyler Smith", "G", offense=1.0))
        snaps.append(_snap(f"2026_0{week}_DAL", week, "DAL", "cut", "Released Guy", "WR", offense=0.9))
        team = "PIT" if week <= 3 else "DAL"
        roster.append(_roster(week, team, "porter", "Joey Porter Jr.", status="INA" if week == 4 else "ACT"))
        # Rosters leave linemen's pfr_id blank.
        roster.append(_roster(week, "DAL", None, "Tyler Smith", position="OL"))
        roster.append(_roster(week, "DAL", "cut", "Released Guy", status="CUT" if week == 5 else "ACT", position="WR"))
    regulars = nfl_whatif.regular_players(pd.DataFrame(snaps), pd.DataFrame(roster)).set_index("pfr_player_id")

    # His last 4 active games: PIT weeks 1-3 at 100% and DAL week 5 at 60%.
    assert regulars.loc["porter", "team"] == "DAL"
    assert regulars.loc["porter", "share"] == pytest.approx(0.9)
    assert regulars.loc["smith", "team"] == "DAL"
    assert regulars.loc["smith", "group"] == "OL"
    assert "cut" not in regulars.index


def _weekly(pid, name, team, week, attempts, epa, season=2026):
    return {"player_id": pid, "player_display_name": name, "position": "QB", "team": team, "season": season, "week": week,
            "season_type": "REG", "attempts": attempts, "sacks_suffered": 0, "passing_epa": epa,
            "game_id": f"{season}_{week:02d}_{team}", "completions": 20, "passing_yards": 200, "passing_tds": 1,
            "passing_interceptions": 0, "carries": 2, "rushing_yards": 5, "rushing_tds": 0}


def _games():
    return pd.DataFrame([{
        "game_id": "2026_08_CAR_GB", "season": 2026, "week": 8, "home_team": "GB", "away_team": "CAR",
        "home_qb_id": "love", "away_qb_id": "young", "home_qb_name": "Jordan Love", "away_qb_name": "Bryce Young",
        "gameday": "2026-10-25",
    }])


def test_qb_options_lists_the_starter_first_then_teammates_with_qb_plus():
    weekly = pd.DataFrame([
        _weekly("love", "Jordan Love", "GB", 1, 30, 6.0),
        _weekly("taylor", "Tyrod Taylor", "GB", 2, 10, -2.0),
        _weekly("young", "Bryce Young", "CAR", 1, 30, 0.0),
    ])
    rosters = pd.DataFrame([
        {"season": 2026, "week": 7, "team": "CAR", "position": "QB", "full_name": "Kenny Pickett",
         "gsis_id": "pickett", "status": "ACT"},
        {"season": 2026, "week": 7, "team": "CAR", "position": "QB", "full_name": "Cut Guy",
         "gsis_id": "cut", "status": "CUT"},
    ])

    options = nfl_whatif.qb_options(weekly, rosters, _games())

    assert [q["id"] for q in options["GB"]] == ["love", "taylor", "average"]
    assert options["GB"][0]["listed"] is True
    assert [q["name"] for q in options["CAR"]] == ["Bryce Young", "Kenny Pickett", "Average starter"]
    # League 4.0 EPA / 70 dropbacks; replacement = that - 0.10; Pickett has no dropbacks.
    replacement = 4.0 / 70 - 0.10
    assert options["CAR"][1]["qb_plus"] == pytest.approx(replacement, abs=1e-4)
    assert options["GB"][0]["qb_plus"] == pytest.approx((6.0 + 150 * replacement) / 180, abs=1e-4)
    # The average starter: the mean of each team's most-used passer (Love, Young).
    young = (0.0 + 150 * replacement) / 180
    assert options["GB"][-1]["qb_plus"] == pytest.approx((options["GB"][0]["qb_plus"] + young) / 2, abs=1e-4)


def test_write_whatif_json_carries_the_shown_probability_pairings_and_regulars(monkeypatch, tmp_path):
    # Stand-in model: home log-odds rise with the home QB's id being "love".
    def fake_probs(master, qb_continuity, weekly, games):
        return pd.DataFrame({"game_id": games["game_id"],
                             "home_win_probability": [0.6 if h == "love" else 0.4 for h in games["home_qb_id"]]})

    # No saved model: QBs are swapped by id, and an average starter can't
    # be priced, so it isn't offered.
    monkeypatch.setattr(nfl_whatif.config, "NFL_GAME_PICK_WIN_PROBABILITY_MODEL_PATH", str(tmp_path / "none.joblib"))
    monkeypatch.setattr(nfl_whatif.nfl_game_picks, "compute_game_win_probabilities", fake_probs)
    monkeypatch.setattr(nfl_whatif.nfl_game_picks, "apply_ml_model", lambda probs, *a: probs)
    weekly = pd.DataFrame([_weekly("love", "Jordan Love", "GB", 1, 30, 6.0),
                           _weekly("taylor", "Tyrod Taylor", "GB", 2, 10, -2.0),
                           _weekly("young", "Bryce Young", "CAR", 1, 30, 0.0)])
    snaps = pd.DataFrame([_snap("2026_01_GB", 1, "GB", "lt", "Left Tackle", "T", offense=1.0)])
    shown = pd.DataFrame({"game_id": ["2026_08_CAR_GB"], "home_win_probability": [0.62]})
    path = tmp_path / "nfl_whatif.json"

    nfl_whatif.write_whatif_json(str(path), shown, None, None, weekly, None, snaps, _games())

    data = json.loads(path.read_text())
    assert data["group_units"]["OL"] == ["pass_offense", "run_offense"]
    assert data["logodds_per_point"] == pytest.approx(nfl_whatif.LOGODDS_PER_POINT)
    assert data["group_default_pct"] == {}  # no play-by-play given
    assert data["average_regular_share"] == {"OL": 1.0}
    game = data["games"][0]
    assert game["home_win_probability"] == pytest.approx(0.62)
    assert game["pairing_log_odds"]["love|young"] == pytest.approx(0.4055, abs=1e-4)
    assert game["pairing_log_odds"]["taylor|young"] == pytest.approx(-0.4055, abs=1e-4)
    assert game["home"]["regulars"] == [{"id": "lt", "name": "Left Tackle", "group": "OL", "share": 1.0, "record": {}}]
    assert game["home"]["units"] == {}
    assert game["away"]["regulars"] == []
    assert [q["id"] for q in game["home"]["qbs"]] == ["love", "taylor"]


def test_pairings_with_a_qb_plus_model_price_an_average_starter(monkeypatch, tmp_path):
    from sklearn.linear_model import LogisticRegression

    columns = nfl_whatif.nfl_game_picks.QB_PLUS_FEATURE_COLUMNS
    X = pd.DataFrame(0.0, index=range(4), columns=columns)
    X["home_qb_plus"] = [-0.2, -0.1, 0.1, 0.2]
    model = LogisticRegression().fit(X, [0, 0, 1, 1])
    path = tmp_path / "model.joblib"
    nfl_whatif.ml_models.save_model({"model": model, "feature_columns": columns}, str(path))
    monkeypatch.setattr(nfl_whatif.config, "NFL_GAME_PICK_WIN_PROBABILITY_MODEL_PATH", str(path))
    master = pd.DataFrame([{"team": t, "pyth_Strength": 1.0, "pyth_Confidence": 1.0, "offensive_edge": 1.0,
                            "defensive_edge": 1.0, "turnover_margin": 1.0, "points_per_drive": 1.0} for t in ("GB", "CAR")])
    options = {"GB": [{"id": "love", "qb_plus": 0.12}, {"id": "average", "qb_plus": 0.03}],
               "CAR": [{"id": "young", "qb_plus": 0.05}]}
    weekly = pd.DataFrame([_weekly("love", "Jordan Love", "GB", 1, 30, 6.0), _weekly("young", "Bryce Young", "CAR", 1, 30, 0.0)])

    qb_continuity = pd.DataFrame([{"team": t, "recent_primary_qb_id": q, "recent_primary_qb_epa": 0.0,
                                   "recent_primary_qb_games": 8} for t, q in (("GB", "love"), ("CAR", "young"))])

    pairs = nfl_whatif.pairing_log_odds(master, qb_continuity, weekly, _games(), options)["2026_08_CAR_GB"]

    assert set(pairs) == {"love|young", "average|young"}
    assert pairs["love|young"] > pairs["average|young"]


def test_add_results_marks_finals_starters_and_who_played():
    data = {"games": [{
        "game_id": "2026_07_DAL_PHI",
        "home": {"qbs": [{"id": "hurts"}, {"id": "mckee"}], "regulars": [{"id": "lane"}, {"id": "brown"}]},
        "away": {"qbs": [{"id": "dak"}], "regulars": [{"id": "lamb"}]},
    }]}
    schedules = pd.DataFrame([{"game_id": "2026_07_DAL_PHI", "home_score": 20.0, "away_score": 24.0,
                               "home_qb_id": "mckee", "away_qb_id": "dak"}])
    snaps = pd.DataFrame([
        {"pfr_player_id": "lane", "offense_pct": 1.0, "defense_pct": 0.0},
        {"pfr_player_id": "brown", "offense_pct": 0.0, "defense_pct": 0.0},  # special teams only
        {"pfr_player_id": "lamb", "offense_pct": 0.8, "defense_pct": 0.0},
    ])

    game = nfl_whatif.add_results(data, schedules, snaps)["games"][0]

    assert game["final"] == {"home_score": 20, "away_score": 24}
    assert [q["started"] for q in game["home"]["qbs"]] == [False, True]
    assert [r["played"] for r in game["home"]["regulars"]] == [True, False]
    assert game["away"]["regulars"][0]["played"] is True


def _play(game, week, off, deff, yards, run=0, dropback=0):
    return {"game_id": game, "season": 2026, "week": week, "season_type": "REG", "posteam": off, "defteam": deff,
            "yards_gained": yards, "rush": run, "qb_scramble": 0, "qb_dropback": dropback}


def test_unit_yards_team_units_and_position_defaults():
    plays = []
    for week in range(1, 4):
        game = f"2026_0{week}_DAL_X"
        plays += [_play(game, week, "DAL", "X", 4, run=1)] * 25 + [_play(game, week, "DAL", "X", 6, dropback=1)] * 35
        plays += [_play(game, week, "X", "DAL", 5, run=1)] * 20 + [_play(game, week, "X", "DAL", 7, dropback=1)] * 30
    yards = nfl_whatif.unit_game_yards(pd.DataFrame(plays))
    assert yards.loc[yards["team"] == "DAL", "run_yards"].tolist() == [100.0] * 3

    units = nfl_whatif.team_units(yards)["DAL"]
    assert units["run_offense"] == {"ypp": 4.0, "plays": 25.0}
    assert units["pass_defense"] == {"ypp": 7.0, "plays": 30.0}

    defaults = nfl_whatif.group_default_pct(yards)
    # A running back touches the run game only: league run yards a team-game
    # = (100 + 100) / 2; the drop costs |-0.13| log-odds.
    rb_drop = 0.13 / (nfl_whatif.LOGODDS_PER_POINT * nfl_whatif.POINTS_PER_YARD * 100)
    assert defaults["RB"] == pytest.approx(1 - rb_drop, abs=1e-3)
    # The line touches both offense units: the same drop costs more, so it's smaller.
    assert 1 - defaults["OL"] < 1 - defaults["WR"]


def test_games_missed_record_needs_two_missed_games():
    plays, snaps = [], []
    for week in range(1, 10):
        game = f"2026_{week:02d}_DAL_X"
        runs = 30 if week not in (8, 9) else 15  # half the run yards in the two games he missed
        plays += [_play(game, week, "DAL", "X", 4 if runs == 30 else 2, run=1)] * 25
        plays += [_play(game, week, "DAL", "X", 6, dropback=1)] * 35
        plays += [_play(game, week, "X", "DAL", 4, run=1)] * 25 + [_play(game, week, "X", "DAL", 6, dropback=1)] * 35
        if week < 8:
            snaps.append(_snap(game, week, "DAL", "rb1", "Javonte Williams", "RB", offense=0.8))
    yards = nfl_whatif.unit_game_yards(pd.DataFrame(plays))
    regulars = pd.DataFrame([{"team": "DAL", "group": "RB", "player": "Javonte Williams", "pfr_player_id": "rb1", "share": 0.8}])

    record = nfl_whatif.games_missed_record(pd.DataFrame(snaps), yards, regulars)

    assert record["rb1"]["run_offense"]["games"] == 2
    # About half (the second missed game's expectation already includes the first).
    assert record["rb1"]["run_offense"]["pct"] == pytest.approx(0.5, abs=0.05)
