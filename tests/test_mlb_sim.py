"""Tests for mlb_sim (the MLB game simulation's player profiles) and
scripts/build_mlb_sim_profiles.py."""

import numpy as np
import pandas as pd

from mlb_metrics import mlb_sim


def _pitch(**kw):
    row = {c: np.nan for c in mlb_sim.STATCAST_COLUMNS}
    row.update({"game_pk": 1, "game_date": pd.Timestamp("2026-06-01"), "game_type": "R", "pitch_type": "FF",
                "release_speed": 95.0, "batter": 10, "pitcher": 20, "player_name": "Arm, Big", "stand": "R",
                "p_throws": "R", "home_team": "HOM", "away_team": "AWY", "inning": 1, "inning_topbot": "Top",
                "balls": 0, "strikes": 0, "outs_when_up": 0, "bat_score": 0, "fld_score": 0, "post_bat_score": 0,
                "n_thruorder_pitcher": 1, "plate_x": 0.1, "plate_z": 2.5, "sz_top": 3.4, "sz_bot": 1.6,
                "description": "ball"})
    row.update(kw)
    return row


def test_pitch_results_and_contact_results_map_statcast():
    p = mlb_sim.prepare(pd.DataFrame([
        _pitch(description="foul_tip"), _pitch(description="automatic_ball"), _pitch(description="hit_into_play"),
        _pitch(description="pitchout", pitch_type="FA"), _pitch(description="ball", pitch_type="PO"),
    ]))
    assert list(p["result"]) == ["S", "B", "X", "B"]
    assert list(p["ptype"]) == ["FF", "FF", "FF", "FF"]
    events = pd.Series(["single", "field_out", "grounded_into_double_play", "home_run", "sac_fly"])
    bb = pd.Series(["line_drive", "popup", "ground_ball", "fly_ball", "fly_ball"])
    assert list(mlb_sim.contact_result(events, bb)) == ["1B", "FO", "GO", "HR", "FO"]


def test_ratios_pull_toward_one():
    expected = np.array([50.0, 50.0])
    assert np.allclose(mlb_sim._ratio(np.array([50.0, 50.0]), expected, 100), [1, 1])
    small = mlb_sim._ratio(np.array([10.0, 0.0]), np.array([5.0, 5.0]), 100)
    assert 1 < small[0] < 2 and 0 < small[1] < 1


def test_transitions_follow_runners_to_the_next_plate_appearance():
    rows = [
        # 0 outs, nobody on: a single.
        _pitch(at_bat_number=1, pitch_number=1, description="hit_into_play", events="single", bb_type="line_drive"),
        # Runner on 1st: a groundball double play.
        _pitch(at_bat_number=2, pitch_number=1, on_1b=10, description="hit_into_play", events="grounded_into_double_play",
               bb_type="ground_ball"),
        # 2 outs, nobody on: a home run.
        _pitch(at_bat_number=3, pitch_number=1, outs_when_up=2, description="hit_into_play", events="home_run",
               bb_type="fly_ball", post_bat_score=1),
        # A strikeout ends the half.
        _pitch(at_bat_number=4, pitch_number=1, outs_when_up=2, bat_score=1, post_bat_score=1, description="swinging_strike",
               events="strikeout", strikes=2),
        # Bottom of the 1st starts.
        _pitch(at_bat_number=5, pitch_number=1, inning_topbot="Bot"),
    ]
    t = mlb_sim.transitions(mlb_sim.prepare(pd.DataFrame(rows)))
    assert t["1B|0|0"] == [[1.0, 1, 0, 0]]
    assert t["GO|1|0"] == [[1.0, 0, 2, 0]]
    assert t["HR|0|2"] == [[1.0, 0, 0, 1]]
    assert t["K|0|2"] == [[1.0, 0, 1, 0]]


def test_the_regular_lineup_is_the_most_used_nine_in_their_usual_order():
    rows = []
    for g in range(1, 6):
        order = list(range(1, 10)) if g < 5 else [99] + list(range(2, 10))
        for i, b in enumerate(order, start=1):
            rows.append(_pitch(game_pk=g, at_bat_number=i, pitch_number=1, batter=b))
    p = mlb_sim.prepare(pd.DataFrame(rows))
    bat_team = p.groupby("batter")["bat_team"].last()
    assert mlb_sim._regular_lineup(p, bat_team, "AWY") == list(range(1, 10))


def test_build_profiles_end_to_end():
    rng = np.random.default_rng(1)
    rows = []
    ab = 0
    for g in range(1, 41):
        for half, bat_team_hitters, pitcher in (("Top", range(100, 109), 200), ("Bot", range(300, 309), 400)):
            for i in range(27):
                ab += 1
                batter = list(bat_team_hitters)[i % 9]
                n = int(rng.integers(1, 5))
                for k in range(n):
                    last = k == n - 1
                    desc = rng.choice(["ball", "called_strike", "foul", "swinging_strike"]) if not last else "hit_into_play"
                    rows.append(_pitch(game_pk=g, game_date=pd.Timestamp("2026-06-01") + pd.Timedelta(days=g),
                                       inning=1 + i // 3, inning_topbot=half, outs_when_up=i % 3, at_bat_number=ab,
                                       pitch_number=k + 1, batter=batter, pitcher=pitcher, description=desc,
                                       pitch_type=rng.choice(["FF", "SL", "CH"]),
                                       events=rng.choice(["field_out", "single", "home_run"]) if last else np.nan,
                                       bb_type=rng.choice(["ground_ball", "fly_ball"]) if last else np.nan,
                                       player_name="Arm, Top" if pitcher == 200 else "Arm, Bottom"))
    profiles = mlb_sim.build_profiles(pd.DataFrame(rows), 2026)
    assert set(profiles["teams"]) == {"AWY", "HOM"}
    assert profiles["teams"]["AWY"]["lineup"] == list(range(100, 109))
    assert profiles["teams"]["HOM"]["starters"][0]["id"] == 200
    top = profiles["pitchers"]["200"]
    assert top["name"] == "Top Arm" and set(top["types"]) == {"FF", "SL", "CH"}
    assert len(top["mix"]["R"]["first"]) == 3
    assert len(profiles["league"]["calibration"]) == len(mlb_sim.PITCH_RESULTS)
    assert all(len(v) == 6 for v in profiles["league"]["pitch_table"]["FF"].values())
