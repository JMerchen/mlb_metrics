"""Our MLB wins above replacement (src/mlb_metrics/mlb_war.py)."""

import numpy as np
import pandas as pd
import pytest

from mlb_metrics import mlb_war as w


def _pitch(game, at_bat, number, **kw):
    row = {"game_pk": game, "game_date": pd.Timestamp("2026-04-01"), "game_type": "R", "at_bat_number": at_bat, "pitch_number": number,
           "batter": 10, "pitcher": 90, "events": None, "description": "ball", "delta_run_exp": 0.0, "inning": 1, "inning_topbot": "Top",
           "home_team": "NYY", "away_team": "BOS", "stand": "R", "p_throws": "R", "hit_location": None, "bb_type": None,
           "launch_speed": np.nan, "launch_angle": np.nan, "hc_x": np.nan, "hc_y": np.nan, "on_1b": None, "on_2b": None, "on_3b": None,
           "outs_when_up": 0, "balls": 0, "strikes": 0, "post_home_score": 0, "post_away_score": 0, "age_bat": 27, "age_pit": 30,
           "player_name": "Sale, Chris", "des": None, "bat_score": 0, "post_bat_score": 0}
    row.update({f"fielder_{k}": 100 + k for k in range(2, 10)})
    row.update(kw)
    return row


def test_a_plate_appearance_is_worth_the_sum_of_its_pitches_and_belongs_to_the_side_batting():
    pitches = pd.DataFrame([
        _pitch(1, 1, 1, delta_run_exp=0.03),
        _pitch(1, 1, 2, delta_run_exp=-0.04, description="called_strike"),
        _pitch(1, 1, 3, delta_run_exp=0.45, events="single", description="hit_into_play", bb_type="ground_ball", hit_location=7,
               launch_speed=98.0, launch_angle=4.0, hc_x=100.0, hc_y=120.0, des="Willy Adames singles on a ground ball."),
        _pitch(1, 2, 1, inning_topbot="Bot", batter=20, pitcher=91, delta_run_exp=-0.25, events="strikeout", description="swinging_strike"),
        _pitch(1, 3, 1, delta_run_exp=-0.1, events="truncated_pa"),                       # cut short by a runner's out: nobody's
        _pitch(2, 1, 1, game_type="S", delta_run_exp=1.0, events="home_run"),             # spring training
    ])
    pa = w.plate_appearances(pitches)
    assert len(pa) == 2
    single = pa.iloc[0]
    assert single["value"] == pytest.approx(0.44) and single["bat_team"] == "BOS" and single["field_team"] == "NYY"
    assert single["in_play"] and not single["out"] and single["fielder_1"] == 90
    assert single["spray"] < 0                                                           # pulled toward left field
    strikeout = pa.iloc[1]
    assert strikeout["bat_team"] == "NYY" and strikeout["field_team"] == "BOS" and not strikeout["in_play"]


def _balls(n, rng):
    """Balls in play where the chance of an out falls with how hard the
    ball is hit, and the out is made by the shortstop (soft) or the
    center fielder (in the air)."""
    speed = rng.uniform(60, 110, n)
    air = rng.random(n) < 0.5
    chance = np.clip(1.25 - speed / 100, 0.05, 0.95)
    out = rng.random(n) < chance
    rows = []
    for i in range(n):
        position = 8 if air[i] else 6
        rows.append({"game_pk": i // 20, "at_bat_number": i % 20, "events": "field_out" if out[i] else "single", "out": bool(out[i]), "in_play": True,
                     "bb_type": "fly_ball" if air[i] else "ground_ball", "launch_speed": speed[i], "launch_angle": 30.0 if air[i] else -5.0,
                     "spray": rng.normal(0, 15), "stand": "R", "hit_location": position if out[i] else 8, "value": -0.25 if out[i] else 0.5,
                     "field_team": "NYY", "fielder_1": 90, **{f"fielder_{k}": 100 + k for k in range(2, 10)}})
    return pd.DataFrame(rows)


def test_the_out_model_learns_the_chance_and_whose_ball_it_was_from_other_games():
    rng = np.random.default_rng(0)
    pa = _balls(4000, rng)
    balls = w.fielding_chances(pa)
    soft, hard = balls[balls["launch_speed"] < 70], balls[balls["launch_speed"] > 100]
    assert soft["chance"].mean() > 0.5 > hard["chance"].mean()
    assert abs(balls["chance"].mean() - balls["out"].mean()) < 0.03
    ground, air = balls[balls["bb_type"] == "ground_ball"], balls[balls["bb_type"] == "fly_ball"]
    assert ground["whose_6"].mean() > 0.9 and air["whose_8"].mean() > 0.9
    assert balls["gap"].iloc[0] == pytest.approx(0.75)                                    # 0.5 for a hit, -0.25 for an out


def test_a_fielder_gets_what_the_out_was_worth_and_shares_the_blame_for_a_hit():
    balls = pd.DataFrame([
        {"game_pk": 1, "events": "field_out", "out": True, "hit_location": 6, "chance": 0.3, "gap": 0.7},      # a hard play made
        {"game_pk": 1, "events": "single", "out": False, "hit_location": 8, "chance": 0.8, "gap": 0.7},        # an easy one missed
        {"game_pk": 1, "events": "field_error", "out": False, "hit_location": 5, "chance": 0.9, "gap": 0.7},   # an error: his alone
    ])
    for k in range(1, 10):
        balls[f"whose_{k}"] = 0.75 if k == 6 else 0.25 if k == 4 else 0.0
        balls[f"fielder_{k}"] = 100 + k
    balls["field_team"] = "NYY"
    credits = w.fielding_credits(balls)
    by = credits.groupby(["player_id", "position"])[["plays", "outs_above", "runs"]].sum()
    shortstop = by.loc[(106, "SS")]
    assert shortstop["outs_above"] == pytest.approx(0.7 - 0.8 * 0.75) and shortstop["plays"] == pytest.approx(1.75)
    assert by.loc[(104, "2B"), "outs_above"] == pytest.approx(-0.8 * 0.25)
    assert by.loc[(105, "3B"), "outs_above"] == pytest.approx(-0.9) and by.loc[(105, "3B"), "runs"] == pytest.approx(-0.63)
    assert (108, "CF") not in by.index                                                     # he picked the single up; it was not his ball


def test_fielding_luck_comes_out_of_a_plate_appearances_value():
    pa = pd.DataFrame({"value": [-0.25, 0.5, -0.2]}, index=[0, 1, 2])
    balls = pd.DataFrame({"out": [True, False], "chance": [0.3, 0.8], "gap": [0.7, 0.7]}, index=[0, 1])
    neutral = w.neutral_value(pa, balls)
    assert neutral[0] == pytest.approx(-0.25 + 0.7 * 0.7)        # robbed: worth more than the out it became
    assert neutral[1] == pytest.approx(0.5 - 0.8 * 0.7)          # a gift: worth less than the hit
    assert neutral[2] == pytest.approx(-0.2)                     # not in play: as it was


def _season(rng, n=30000):
    """Two teams. Hitter 1 is truly +0.08 runs a plate appearance; pitcher
    901 truly saves 0.06; everyone else is average."""
    rows = []
    for i in range(n):
        home = i % 2 == 0
        bat_team, field_team = ("BOS", "NYY") if home else ("NYY", "BOS")
        base = 1 if bat_team == "BOS" else 51
        batter = base + (rng.integers(0, 9) if rng.random() < 0.9 else rng.integers(9, 16))
        # Each game's first 44 plate appearances belong to that day's starters (five take turns), the rest to relievers.
        game, at_bat = i // 70, i % 70
        pitch_base = 900 if field_team == "NYY" else 950
        pitcher = pitch_base + (game % 5 + 1 if at_bat < 44 else rng.integers(10, 22))
        value = rng.normal(0, 0.4) + (0.08 if batter == 1 else 0) - (0.06 if pitcher == 901 else 0)
        rows.append({"game_pk": i // 70, "at_bat_number": i % 70, "game_date": pd.Timestamp("2026-04-01") + pd.Timedelta(days=i // 700),
                     "batter": batter, "pitcher": pitcher, "bat_team": bat_team, "field_team": field_team, "home_team": "NYY" if home else "BOS",
                     "stand": "R", "p_throws": "R" if pitcher % 2 else "L", "value": value, "age_bat": 28, "age_pit": 29,
                     **{f"fielder_{k}": (50 if field_team == "NYY" else 0) + k for k in range(2, 10)}})
    return pd.DataFrame(rows)


def test_a_pitcher_who_opens_most_of_his_games_is_a_starter():
    pa = pd.DataFrame({"game_pk": [1, 1, 1, 2, 2, 3], "at_bat_number": [1, 2, 3, 1, 2, 1], "field_team": ["NYY"] * 6,
                       "pitcher": [5, 5, 6, 5, 6, 6]})
    roles = w.pitcher_roles(pa)
    assert roles[5] == "SP" and roles[6] == "RP"        # 6 opened one of his three games


def test_the_fit_finds_the_hitter_and_the_pitcher_and_keeps_their_seasons_whole():
    rng = np.random.default_rng(1)
    pa = _season(rng)
    fit = w.fit_plate_appearances(pa, pa["value"])
    batters, pitchers = fit["batters"].set_index("player_id"), fit["pitchers"].set_index("player_id")
    assert batters["runs"].idxmax() == 1 and pitchers["runs"].idxmax() == 901
    best = batters.loc[1]
    # About +0.08 a plate appearance, not pulled toward the average.
    assert best["runs"] / best["pa"] == pytest.approx(0.08, abs=0.03)
    assert 0 < best["talent"] < best["runs"] / best["pa"] + 0.02          # the pulled estimate sits nearer his tier
    assert pitchers.loc[901, "runs"] / pitchers.loc[901, "pa"] == pytest.approx(0.06, abs=0.03)
    assert abs(batters["runs"].sum()) < 1e-6 and abs(pitchers["runs"].sum()) < 1e-6      # against the league's average
    assert set(batters["tier"]) == {"roster", "replacement"} or set(batters["tier"]) == {"roster"}
    assert pitchers.loc[901, "tier"] == "SP" and pitchers.loc[912, "tier"].startswith("RP")


def test_the_table_measures_each_player_against_his_position_and_a_replacement():
    rng = np.random.default_rng(2)
    pa = _season(rng, 12000)
    fit = w.fit_plate_appearances(pa, pa["value"])
    credits = pd.DataFrame({"game_pk": [1, 1, 2], "player_id": [56, 6, 56], "team": ["NYY", "BOS", "NYY"], "position": ["SS", "SS", "SS"],
                            "plays": [1.0, 1.0, 1.0], "outs_above": [0.5, -0.5, 0.5], "runs": [0.4, -0.4, 0.4]})
    table = w.player_table(pa, fit, credits).set_index("player_id")
    hitter, pitcher = table.loc[1], table.loc[901]
    assert hitter["bf"] == 0 and hitter["pitching"] == 0 and hitter["pa"] > 500
    assert hitter["replacement"] == pytest.approx(w.REPLACEMENT["bat"] * hitter["pa"])
    assert hitter["war"] == pytest.approx((hitter["batting"] + hitter["fielding"] + hitter["replacement"]) / w.RUNS_PER_WIN)
    assert pitcher["position"] == "SP" and pitcher["pa"] == 0
    assert pitcher["replacement"] == pytest.approx(w.REPLACEMENT["SP"] * pitcher["bf"])
    # Fielding is against the position's average: two shortstops, one up, one down.
    assert table.loc[56, "fielding"] == pytest.approx(0.8 - 2 * (0.4 / 3)) and table.loc[6, "fielding"] == pytest.approx(-0.4 - 0.4 / 3)
    # Batting against the position's average sums to zero within a position.
    hitters = table[table["pa"] > table["bf"]]
    assert abs(hitters.groupby("position")["batting"].sum()).max() < 1e-6
    assert hitter["team"] == "BOS" and hitter["age"] == 28


def test_names_come_from_the_pitches_themselves():
    rows = [_pitch(1, i, 1, batter=10, events=e, des=d) for i, (e, d) in enumerate([
        ("single", "Elly De La Cruz singles on a line drive."), ("strikeout", "Elly De La Cruz strikes out swinging."),
        ("walk", "Elly De La Cruz walks."), ("field_out", "Elly De La Cruz flies out to left fielder Juan Soto.")])]
    rows += [_pitch(1, 9, 1, batter=11, pitcher=91, player_name="Leiter Jr., Mark", events="strikeout", des="Bobby Witt Jr. strikes out swinging."),
             _pitch(1, 10, 1, batter=11, pitcher=91, player_name="Leiter Jr., Mark", events="strikeout", des="Bobby Witt Jr. called out on strikes.")]
    # A pitcher who batted once keeps the name on his pitching line.
    rows += [_pitch(1, 11, 1, batter=90, pitcher=91, player_name="Leiter Jr., Mark", events="sac_bunt", des="Chris Sale hits a sacrifice bunt. Runner scores.")]
    names = w.player_names(pd.DataFrame(rows))
    assert names[10] == "Elly De La Cruz" and names[11] == "Bobby Witt Jr."
    assert names[90] == "Chris Sale" and names[91] == "Mark Leiter Jr."


def test_every_team_is_in_a_division_and_an_empty_season_builds_nothing(tmp_path):
    assert len(w.TEAM_DIVISION) == 30 and all(len(teams) == 5 for teams in w.DIVISIONS.values())
    assert w.TEAM_DIVISION["ATH"] == "AL West" and w.TEAM_DIVISION["AZ"] == "NL West"
    assert w.build(2031, str(tmp_path)) == {"season": 2031, "through": None, "runs_per_win": w.RUNS_PER_WIN, "players": [], "teams": []}


def test_team_records_come_from_the_final_scores():
    pitches = pd.DataFrame([
        _pitch(1, 1, 1, post_home_score=0, post_away_score=1), _pitch(1, 60, 1, post_home_score=3, post_away_score=5),
        _pitch(2, 1, 1, post_home_score=2, post_away_score=0), _pitch(2, 61, 1, post_home_score=4, post_away_score=1),
    ])
    records = w.team_records(pitches)
    assert records.loc["NYY"].to_dict() == {"games": 2, "wins": 1, "scored": 7, "allowed": 6}
    assert records.loc["BOS"].to_dict() == {"games": 2, "wins": 1, "scored": 6, "allowed": 7}


def _inning(game, plays, start=1):
    """One half-inning as pitches: each play is (runners before as a dict
    of base -> id, outs, runs in so far, event or None, extra columns)."""
    rows = []
    for i, (on, outs, score, event, extra) in enumerate(plays):
        last = i == len(plays) - 1
        rows.append(_pitch(game, start + (extra.get("at_bat", i)), extra.get("pitch", 1), on_1b=on.get(1), on_2b=on.get(2), on_3b=on.get(3),
                           outs_when_up=outs, bat_score=score, post_bat_score=extra.get("post", score), events=event,
                           hit_location=extra.get("field"), bb_type="ground_ball" if event in ("single", "field_out") else None))
    return rows


def test_run_expectancy_is_the_runs_that_follow_each_state():
    rows = []
    for game in range(40):
        # Nobody on, nobody out: half the innings score one run later.
        scored = game % 2
        rows += _inning(game, [({}, 0, 0, "single", {}), ({1: 7}, 0, 0, "field_out", {}), ({1: 7}, 1, 0, "field_out", {}),
                               ({1: 7}, 2, 0, "field_out", {"post": scored})])
    table = w.run_expectancy(pd.DataFrame(rows))
    assert table[(0, 0)] == pytest.approx(0.5) and table[(0, 1)] == pytest.approx(0.5) and table[(2, 1)] == pytest.approx(0.5)


def _running_season():
    """Innings where a runner on first either steals second during the
    next plate appearance or stays; and singles with a man on first where
    he ends on third (runner 7) or second (runner 8)."""
    rows = []
    for game in range(120):
        fast = game % 2 == 0
        runner = 7 if fast else 8
        scores = game % 3 == 0
        # A plate appearance of two pitches: after the first, the fast runner is on second.
        rows += _inning(game, [
            ({}, 0, 0, "single", {"at_bat": 0}),
            ({1: runner}, 0, 0, None, {"at_bat": 1, "pitch": 1}),
            ({2: runner} if fast else {1: runner}, 0, 0, "strikeout", {"at_bat": 1, "pitch": 2}),
            ({2: runner} if fast else {1: runner}, 1, 0, "field_out", {"at_bat": 2}),
            ({2: runner} if fast else {1: runner}, 2, 0, "field_out", {"at_bat": 3, "post": int(scores and fast)}),
        ])
        # Another inning: man on first, a single to right; the fast one takes third.
        rows += [dict(r, inning=2) for r in _inning(game, [
            ({1: runner}, 0, 0, "single", {"at_bat": 10, "field": 9}),
            ({1: 30, 3: runner} if fast else {1: 30, 2: runner}, 0, 0, "field_out", {"at_bat": 11}),
            ({1: 30, 3: runner} if fast else {1: 30, 2: runner}, 1, 0, "field_out", {"at_bat": 12}),
            ({1: 30, 3: runner} if fast else {1: 30, 2: runner}, 2, 0, "field_out", {"at_bat": 13, "post": int(fast)}),
        ])]
        # A third inning that opens with a man on second and nobody out and always scores,
        # so that state has a value to steal into.
        rows += [dict(r, inning=3) for r in _inning(game, [
            ({2: 40}, 0, 0, "field_out", {"at_bat": 20}), ({2: 40}, 1, 0, "field_out", {"at_bat": 21}),
            ({2: 40}, 2, 0, "field_out", {"at_bat": 22, "post": 1})])]
    return pd.DataFrame(rows)


def test_baserunning_credits_the_steal_and_the_extra_base_and_nets_to_zero():
    running = w.baserunning(_running_season()).set_index("player_id")
    fast, slow = running.loc[7], running.loc[8]
    # Only the fast runner steals: he is ahead of the average runner on first, and everyone else there is behind.
    assert fast["steals"] > 0 > slow["steals"] and running.loc[30, "steals"] < 0
    # First to third on the single, against first to second.
    assert fast["advances"] > 0 > slow["advances"]
    assert fast["running"] == pytest.approx(fast["steals"] + fast["advances"])
    assert abs(running["steals"].sum()) < 1e-9 and abs(running["advances"].sum()) < 1e-9
    assert fast["team"] == "BOS" and fast["times_on"] > 0


def test_running_goes_into_a_players_runs_above_average():
    rng = np.random.default_rng(4)
    pa = _season(rng, 6000)
    fit = w.fit_plate_appearances(pa, pa["value"])
    credits = pd.DataFrame({"game_pk": [1], "player_id": [56], "team": ["NYY"], "position": ["SS"], "plays": [1.0], "outs_above": [0.0], "runs": [0.0]})
    running = pd.DataFrame({"player_id": [1, 1], "team": ["BOS", "NYY"], "times_on": [100, 20], "steals": [2.0, 0.5], "advances": [1.0, 0.0], "running": [3.0, 0.5]})
    with_running = w.player_table(pa, fit, credits, running).set_index("player_id")
    without = w.player_table(pa, fit, credits).set_index("player_id")
    assert with_running.loc[1, "running"] == pytest.approx(3.5) and with_running.loc[1, "times_on"] == 120      # both stints
    assert with_running.loc[1, "raa"] == pytest.approx(without.loc[1, "raa"] + 3.5)
    assert without.loc[1, "running"] == 0


def test_the_team_table_turns_players_runs_into_wins_above_a_replacement_team():
    records = pd.DataFrame({"games": [162, 162, 162], "wins": [100, 81, 62], "scored": [850, 720, 600], "allowed": [620, 720, 830]}, index=["A", "B", "C"])
    runs = pd.DataFrame({"batting": [90.0, 0.0, -90.0], "running": [10.0, 0.0, -10.0], "pitching": [80.0, 0.0, -80.0], "fielding": [20.0, 0.0, -20.0]}, index=["A", "B", "C"])
    time = pd.DataFrame({"pa": [6100.0] * 3, "bf_SP": [3400.0] * 3, "bf_RP": [2700.0] * 3}, index=["A", "B", "C"])
    t = w.team_table(records, runs, time)
    league = (850 + 720 + 600) / 486
    # An average team's players add up to an even record.
    assert t.loc["B", "expected"] == pytest.approx(81.0) and t.loc["B", "offense"] == 0
    assert t.loc["A", "expected"] == pytest.approx(162 * w.pythagorean(league * 162 + 100, league * 162 - 100))
    # The replacement team: a replacement hitter's runs off every plate appearance, a replacement pitcher's on every batter.
    off = w.REPLACEMENT["bat"] * 6100 / 162
    on = (w.REPLACEMENT["SP"] * 3400 + w.REPLACEMENT["RP"] * 2700) / 162
    assert t.loc["B", "baseline"] == pytest.approx(162 * w.pythagorean(league - off, league + on))
    assert 40 < t.loc["B", "baseline"] < 50
    assert (t["baseline"] + t["war"] == t["expected"]).all()
    # What is left: the real runs' wins less expected, and the record less the real runs' wins.
    real = 162 * w.pythagorean(850, 620)
    assert t.loc["A", "uncredited"] == pytest.approx(real - t.loc["A", "expected"]) and t.loc["A", "close"] == pytest.approx(100 - real)
    assert (t["expected"] + t["uncredited"] + t["close"]).round(6).tolist() == [100.0, 81.0, 62.0]
    # Through the curve, the best team is nearer the middle than adding runs at a flat rate would put it.
    assert t.loc["A", "expected"] - 81 < 200 / w.RUNS_PER_WIN
