import numpy as np
import pandas as pd
import pytest

from mlb_metrics import nfl_season_views as views


def _schedules():
    return pd.DataFrame([
        {"season": 2026, "game_type": "REG", "week": 1, "gameday": "2026-09-13", "home_team": "KC", "away_team": "DEN", "home_score": 31, "away_score": 10},
        {"season": 2026, "game_type": "REG", "week": 2, "gameday": "2026-09-20", "home_team": "BUF", "away_team": "KC", "home_score": 20, "away_score": 20},
        {"season": 2026, "game_type": "REG", "week": 2, "gameday": "2026-09-20", "home_team": "DEN", "away_team": "MIA", "home_score": 17, "away_score": 24},
        {"season": 2026, "game_type": "REG", "week": 3, "gameday": "2026-09-27", "home_team": "MIA", "away_team": "KC", "home_score": None, "away_score": None},
        {"season": 2026, "game_type": "POST", "week": 19, "gameday": "2027-01-10", "home_team": "KC", "away_team": "BUF", "home_score": 30, "away_score": 3},
    ])


def test_team_schedule_results_byes_and_upcoming():
    rows = views.team_schedule(_schedules())
    kc = rows[rows["team"] == "KC"].set_index("week")
    assert kc.loc[1, ["opponent", "home", "result", "margin"]].tolist() == ["DEN", 1, "W", 21]
    assert kc.loc[2, ["opponent", "home", "result"]].tolist() == ["BUF", 0, "T"]
    assert kc.loc[3, "result"] == ""                       # not played yet
    assert 19 not in kc.index                               # playoffs left out
    assert rows[(rows["team"] == "BUF") & (rows["week"] == 1)]["result"].tolist() == ["BYE"]


def test_team_stats_record_points_and_epa():
    rows = views.team_schedule(_schedules())
    pbp = pd.DataFrame([
        {"season_type": "REG", "posteam": "KC", "defteam": "DEN", "epa": 0.5, "pass": 1, "rush": 0, "interception": 0, "fumble_lost": 0, "success": 1},
        {"season_type": "REG", "posteam": "KC", "defteam": "DEN", "epa": -0.1, "pass": 0, "rush": 1, "interception": 0, "fumble_lost": 1, "success": 0},
        {"season_type": "REG", "posteam": "DEN", "defteam": "KC", "epa": -0.4, "pass": 1, "rush": 0, "interception": 1, "fumble_lost": 0, "success": 0},
    ])
    stats = views.team_stats(rows, pbp).set_index("team")
    assert stats.loc["KC", ["games", "wins", "losses", "ties"]].tolist() == [2, 1, 0, 1]
    assert stats.loc["KC", "point_diff_pg"] == pytest.approx((51 - 30) / 2)
    assert stats.loc["KC", "off_epa_play"] == pytest.approx(0.2)
    assert stats.loc["KC", "def_epa_play"] == pytest.approx(-0.4)
    assert stats.loc["KC", "net_epa_play"] == pytest.approx(0.6)
    assert stats.loc["KC", "takeaways_pg"] == pytest.approx(0.5)


def _weekly():
    """Two receivers with identical lines; A faced stingy defense D1, B
    generous D2 (which also gave 150 yards to another receiver)."""
    rows = []
    for week in range(1, 6):
        for pid, opp in (("A", "D1"), ("B", "D2")):
            rows.append({"player_id": pid, "player_display_name": pid, "position": "WR", "season_type": "REG",
                         "team": "T" + pid, "opponent_team": opp, "game_id": f"{week}{opp}", "week": week,
                         "headshot_url": "", "targets": 8, "receptions": 5, "receiving_yards": 60,
                         "receiving_epa": 1.0, "target_share": 0.25, "attempts": 0, "sacks_suffered": 0})
        rows.append({"player_id": "C", "player_display_name": "C", "position": "WR", "season_type": "REG",
                     "team": "TC", "opponent_team": "D2", "game_id": f"{week}D2", "week": week, "headshot_url": "",
                     "targets": 10, "receptions": 9, "receiving_yards": 150, "receiving_epa": 6.0,
                     "target_share": 0.3, "attempts": 0, "sacks_suffered": 0})
    return pd.DataFrame(rows)


def test_player_views_blend_and_credit_tough_defenses():
    players = views.player_views(_weekly()).set_index("player_id")
    assert players.loc["A", "rec_ypg"] == pytest.approx(60)
    assert players.loc["A", "rec_ypg"] == pytest.approx(players.loc["B", "rec_ypg"])
    assert players.loc["A", "rec_ypg_adj"] > players.loc["B", "rec_ypg_adj"]
    assert players.loc["A", "epa_target_adj"] > players.loc["B", "epa_target_adj"]
    assert set(players["position"]) == {"WR"}
    assert players.loc["A", "games"] == 5


def test_player_views_need_games_and_volume():
    few = _weekly()
    few = few[~((few["player_id"] == "A") & (few["week"] > 2))]
    assert "A" not in set(views.player_views(few)["player_id"])
