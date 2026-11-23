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
           "player_name": "Sale, Chris", "des": None}
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
    assert w.build(2031, str(tmp_path)) == {"season": 2031, "through": None, "runs_per_win": w.RUNS_PER_WIN, "players": []}


def test_team_records_come_from_the_final_scores():
    pitches = pd.DataFrame([
        _pitch(1, 1, 1, post_home_score=0, post_away_score=1), _pitch(1, 60, 1, post_home_score=3, post_away_score=5),
        _pitch(2, 1, 1, post_home_score=2, post_away_score=0), _pitch(2, 61, 1, post_home_score=4, post_away_score=1),
    ])
    records = w.team_records(pitches)
    assert records.loc["NYY"].to_dict() == {"games": 2, "wins": 1, "scored": 7, "allowed": 6}
    assert records.loc["BOS"].to_dict() == {"games": 2, "wins": 1, "scored": 6, "allowed": 7}
