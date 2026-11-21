"""The play-calling tree's data and the situations behind its research (src/mlb_metrics/nfl_play_sequencing.py)."""

import numpy as np
import pandas as pd

from mlb_metrics import nfl_play_sequencing as seq


def _play(play_id, play_type, down, togo, **kw):
    row = {"game_id": "g1", "play_id": play_id, "season": 2026, "week": 1, "season_type": "REG", "posteam": "KC", "defteam": "BUF",
           "play_type": play_type, "down": down, "ydstogo": togo, "yardline_100": 60, "xpass": 0.5, "success": 0, "yards_gained": 0,
           "sack": 0, "complete_pass": 0, "first_down": 0, "fixed_drive": 1, "qb_kneel": 0, "qb_spike": 0}
    row.update(kw)
    return row


def _drive():
    return pd.DataFrame([
        _play(1, "pass", 1, 10),                                                   # opens the drive, incomplete
        _play(2, "run", 2, 10, yards_gained=1),                                    # 2nd & long after a failed pass; stuffed
        _play(3, "pass", 3, 9, complete_pass=1, yards_gained=18, first_down=1, success=1),   # 3rd & long, converts with a big gain
        _play(4, "run", 1, 10, yards_gained=6, success=1),                         # 1st after a pass moved the chains; after a 15+ gain
        _play(5, "run", 2, 4, yards_gained=5, first_down=1, success=1),            # 2nd & medium after a run
        _play(6, "run", 1, 10, yards_gained=8, success=1, yardline_100=18),        # 1st after a run moved the chains; two runs in a row; red zone
        _play(7, "pass", 2, 2, sack=1, yards_gained=-7, yardline_100=10),          # 2nd & short
        _play(8, "pass", 3, 9, complete_pass=1, yards_gained=4, yardline_100=17),  # 3rd & long, after a loss
        _play(9, "field_goal", 4, 5),
        _play(10, "run", 1, 10, fixed_drive=2, posteam="BUF", defteam="KC"),       # the other team opens its drive
        _play(11, "pass", 2, 10, fixed_drive=2, posteam="BUF", defteam="KC", qb_spike=1),
        _play(12, "run", 1, 10, season_type="POST", fixed_drive=3),
    ])


def test_each_play_gets_the_situation_its_down_distance_and_previous_play_put_it_in():
    p = seq.scrimmage(_drive())
    assert len(p) == 9                                           # no field goal, spike or playoff play
    assert list(seq.situations(p)) == [
        "open", "second_long_after_pass", "third_long", "first_after_pass", "second_medium_after_run", "first_after_run",
        "second_short", "third_long", "open"]
    flags = seq.habit_flags(p)
    assert list(flags["after_incompletion"]) == [False, True, False, False, False, False, False, False, False]
    assert list(flags["after_stuffed_run"]) == [False, False, True, False, False, False, False, False, False]
    assert list(flags["after_big_gain"]) == [False, False, False, True, False, False, False, False, False]
    assert list(np.flatnonzero(flags["after_two_runs"])) == [5, 6]
    assert list(flags["red_zone"]) == [False] * 5 + [True, True, True, False]
    assert flags["after_loss"].iloc[7] and not flags["after_loss"].iloc[8]      # a new drive remembers nothing


def test_a_drive_is_its_first_calls_in_order_and_whether_each_worked():
    drives = seq.drive_sequences(seq.scrimmage(_drive()))
    # Kansas City: incomplete, a stuffed run, a pass that converted, a run
    # that worked - and nothing past the fourth play. Buffalo's drive is
    # one run (the spike is not a call).
    assert list(drives["sequence"]) == ["P-R-P+R+", "R-"]
    assert list(drives["posteam"]) == ["KC", "BUF"] and list(drives["defteam"]) == ["BUF", "KC"]
    assert list(seq.drive_sequences(seq.scrimmage(_drive()), depth=2)["sequence"]) == ["P-R-", "R-"]


def test_the_tree_counts_each_teams_drives_and_the_drives_it_faced():
    plays = _drive()
    again = plays[plays["fixed_drive"] == 1].assign(game_id="g2", week=3)
    data = seq.build(pd.concat([plays, again], ignore_index=True), 2026)
    assert data["season"] == 2026 and data["through_week"] == 3 and data["depth"] == seq.TREE_DEPTH
    assert data["teams"]["KC"] == {"offense": {"P-R-P+R+": 2}, "defense": {"R-": 1}}
    assert data["teams"]["BUF"] == {"offense": {"R-": 1}, "defense": {"P-R-P+R+": 2}}
    assert data["league"] == {"P-R-P+R+": 2, "R-": 1}
    assert seq.build(plays, 2030) == {"season": 2030, "through_week": 0, "depth": seq.TREE_DEPTH, "league": {}, "teams": {}}
