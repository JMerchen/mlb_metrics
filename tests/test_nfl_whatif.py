import json

import pandas as pd
import pytest

from mlb_metrics import nfl_whatif


def _snap(game, week, team, pid, name, position, offense=0.0, defense=0.0):
    return {"game_id": game, "season": 2026, "game_type": "REG", "week": week, "team": team,
            "pfr_player_id": pid, "player": name, "position": position,
            "offense_pct": offense, "defense_pct": defense}


def test_regular_players_uses_the_last_four_games_counting_a_missed_game_as_zero():
    snaps = []
    for week in range(1, 6):
        game = f"2026_0{week}_GB"
        snaps.append(_snap(game, week, "GB", "lt", "Left Tackle", "T", offense=1.0))
        # The receiver played every snap in weeks 1-3, then missed weeks 4-5.
        if week <= 3:
            snaps.append(_snap(game, week, "GB", "wr", "Receiver", "WR", offense=1.0))
        snaps.append(_snap(game, week, "GB", "cb", "Corner", "CB", defense=0.4))
    regulars = nfl_whatif.regular_players(pd.DataFrame(snaps)).set_index("pfr_player_id")

    assert regulars.loc["lt", "group"] == "OL"
    assert regulars.loc["lt", "share"] == pytest.approx(1.0)
    # Weeks 2-5: played 2 of 4 at 100% -> 0.5, still a regular.
    assert regulars.loc["wr", "share"] == pytest.approx(0.5)
    assert "cb" not in regulars.index  # 40% of snaps isn't a regular


def _weekly(pid, name, team, week, attempts, epa, season=2026):
    return {"player_id": pid, "player_display_name": name, "team": team, "season": season, "week": week,
            "season_type": "REG", "attempts": attempts, "sacks_suffered": 0, "passing_epa": epa}


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

    assert [q["id"] for q in options["GB"]] == ["love", "taylor"]
    assert options["GB"][0]["listed"] is True
    assert [q["name"] for q in options["CAR"]] == ["Bryce Young", "Kenny Pickett"]
    # League 4.0 EPA / 70 dropbacks; replacement = that - 0.10; Pickett has no dropbacks.
    replacement = 4.0 / 70 - 0.10
    assert options["CAR"][1]["qb_plus"] == pytest.approx(replacement, abs=1e-4)
    assert options["GB"][0]["qb_plus"] == pytest.approx((6.0 + 150 * replacement) / 180, abs=1e-4)


def test_write_whatif_json_carries_the_shown_probability_pairings_and_regulars(monkeypatch, tmp_path):
    # Stand-in model: home log-odds rise with the home QB's id being "love".
    def fake_probs(master, qb_continuity, weekly, games):
        return pd.DataFrame({"game_id": games["game_id"],
                             "home_win_probability": [0.6 if h == "love" else 0.4 for h in games["home_qb_id"]]})

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
    assert data["injury_log_odds"]["OL"] == pytest.approx(-0.85)
    game = data["games"][0]
    assert game["home_win_probability"] == pytest.approx(0.62)
    assert game["pairing_log_odds"]["love|young"] == pytest.approx(0.4055, abs=1e-4)
    assert game["pairing_log_odds"]["taylor|young"] == pytest.approx(-0.4055, abs=1e-4)
    assert game["home"]["regulars"] == [{"id": "lt", "name": "Left Tackle", "group": "OL", "share": 1.0}]
    assert game["away"]["regulars"] == []
