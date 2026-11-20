"""The Play-calling map's data (src/mlb_metrics/nfl_play_sequencing.py)."""

import numpy as np
import pandas as pd
import pytest

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


def _season(rng, team_pass, n=600):
    """Plays where every situation is 2nd & long after a failed pass: a
    first-down incompletion, then the play counted."""
    rows = []
    for i in range(n):
        offense = "KC" if i % 2 else "BUF"
        defense = "BUF" if offense == "KC" else "KC"
        is_pass = rng.random() < team_pass[offense]
        rows.append(_play(2 * i, "pass", 1, 10, fixed_drive=i, posteam=offense, defteam=defense, xpass=0.55))
        rows.append(_play(2 * i + 1, "pass" if is_pass else "run", 2, 10, fixed_drive=i, posteam=offense, defteam=defense, xpass=0.7,
                          success=int(rng.random() < (0.45 if is_pass else 0.3)), complete_pass=int(is_pass)))
    return pd.DataFrame(rows)


def test_a_teams_lean_is_measured_against_a_typical_team_in_the_same_plays_and_pulled_toward_it():
    rng = np.random.default_rng(0)
    data = seq.build(_season(rng, {"KC": 0.9, "BUF": 0.5}), 2026)
    kc, buf = (data["teams"][t]["offense"]["second_long_after_pass"] for t in ("KC", "BUF"))
    league = data["league"]["second_long_after_pass"]
    assert kc["plays"] == 300 and league["plays"] == 600
    # Typical is the expected pass rate (0.7) plus the league's own lean here.
    assert kc["typical"] == pytest.approx(0.7 + league["lean"], abs=1e-3)
    raw = kc["passes"] / kc["plays"]
    assert kc["typical"] < kc["rate"] < raw                     # pulled toward typical, not all the way
    assert kc["rate"] == pytest.approx((kc["passes"] + seq.LEAN_PRIOR * kc["typical"]) / (300 + seq.LEAN_PRIOR), abs=1e-3)
    assert kc["lean"] > 0.1 > 0 > buf["lean"]
    # Knowing the habit makes Kansas City easier to guess than the situation alone.
    assert data["teams"]["KC"]["readable"]["habits"] >= data["teams"]["KC"]["readable"]["situation"]
    assert {data["teams"][t]["readable"]["rank"] for t in ("KC", "BUF")} == {1, 2}
    assert [s["key"] for s in data["states"]] == [key for key, _, _ in seq.STATES]


def test_a_defenses_record_is_kept_raw_and_its_effect_is_pulled_hard_toward_the_league():
    rng = np.random.default_rng(1)
    plays = _season(rng, {"KC": 0.7, "BUF": 0.7})
    # Buffalo's defense is perfect against the pass on the counted plays.
    counted = (plays["down"] == 2) & (plays["defteam"] == "BUF") & (plays["play_type"] == "pass")
    plays.loc[counted, "success"] = 0
    data = seq.build(plays, 2026)
    record = data["teams"]["BUF"]["defense"]["second_long_after_pass"]
    assert record["pass_success"] == 0 and record["passes"] == int(counted.sum())
    effect = data["teams"]["BUF"]["defense_effect"]["pass"]
    raw_gap = 0.0 - data["overall_success"]["pass"]
    assert raw_gap < effect < 0 and abs(effect) < 0.6 * abs(raw_gap)        # a few hundred plays move it only part way
    expected = seq.expected_success(data, "KC", "BUF", "second_long_after_pass")
    league = data["league"]["second_long_after_pass"]
    assert expected["pass"] == pytest.approx(league["pass_success"] + data["teams"]["KC"]["offense_effect"]["pass"] + effect, abs=1e-3)
    assert seq.build(plays, 2030) == {"season": 2030, "through_week": 0, "teams": {}}
