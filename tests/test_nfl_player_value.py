"""Our NFL wins above replacement (src/mlb_metrics/nfl_player_value.py)."""

import numpy as np
import pandas as pd
import pytest

from mlb_metrics import nfl_player_value as v


def test_the_mixed_model_recovers_big_effects_and_pulls_small_samples_toward_zero():
    rng = np.random.default_rng(0)
    # 40 players; true effects spread 0.3; players 0-19 get 400 rows, 20-39 get 8.
    true = rng.normal(0, 0.3, 40)
    codes = np.concatenate([np.repeat(np.arange(20), 400), np.repeat(np.arange(20, 40), 8)])
    y = 1.0 + true[codes] + rng.normal(0, 1.0, len(codes))
    fit = v.mixed_fit(y, [("intercept", np.zeros(len(codes), dtype=int), 1)], [("player", codes, 40)])
    assert fit["fixed"]["intercept"][0] == pytest.approx(1.0, abs=0.12)
    effects = fit["random"]["player"]
    assert np.corrcoef(effects[:20], true[:20])[0, 1] > 0.95
    # The raw mean of 8 rows is mostly noise; the fit keeps little of it.
    raw_small = np.array([y[codes == i].mean() - 1.0 for i in range(20, 40)])
    assert np.abs(effects[20:]).mean() < 0.6 * np.abs(raw_small).mean()
    assert fit["variance"]["player"] == pytest.approx(0.09, abs=0.05) and fit["noise"] == pytest.approx(1.0, abs=0.05)


def test_a_factor_that_does_not_apply_to_a_row_is_left_out_of_it():
    y = np.array([1.0, 1.0, 3.0, 3.0] * 30)
    receiver = np.array([0, 0, -1, -1] * 30)          # only the first two rows have one
    fit = v.mixed_fit(y, [("kind", np.array([0, 0, 1, 1] * 30), 2)], [("receiver", receiver, 1)])
    assert fit["fixed"]["kind"] == pytest.approx([1.0, 3.0], abs=1e-6)
    assert abs(fit["random"]["receiver"][0]) < 1e-6


def _play(play_id, **kw):
    row = {"game_id": "g1", "play_id": play_id, "season": 2026, "week": 1, "season_type": "REG", "posteam": "KC", "defteam": "BUF",
           "play_type": "pass", "down": 1, "epa": 0.0, "qb_dropback": 1, "qb_scramble": 0, "qb_kneel": 0, "qb_spike": 0,
           "complete_pass": 0, "sack": 0, "interception": 0, "air_epa": np.nan, "yac_epa": np.nan, "xyac_epa": np.nan,
           "passer_id": "QB1", "receiver_player_id": None, "rusher_player_id": None, "xpass": 0.6, "wp": 0.5}
    row.update(kw)
    return row


def test_a_plays_parts_add_back_to_its_epa():
    pbp = pd.DataFrame([
        _play(1, epa=1.2, complete_pass=1, receiver_player_id="WR1", air_epa=0.5, yac_epa=0.7, xyac_epa=0.4),   # a catch
        _play(2, epa=-0.8, receiver_player_id="WR1"),                                                           # incomplete
        _play(3, epa=-1.5, sack=1),                                                                             # a sack: no target
        _play(4, epa=0.4, play_type="run", qb_dropback=0, passer_id=None, rusher_player_id="RB1"),              # a run
        _play(5, epa=0.9, play_type="run", qb_scramble=1, rusher_player_id="QB1"),                              # a scramble is the QB's dropback
        _play(6, epa=-0.5, play_type="run", qb_dropback=0, qb_kneel=1, passer_id=None, rusher_player_id="QB1"),  # a kneel is not a play
        _play(7, epa=2.0, season_type="POST", complete_pass=1, receiver_player_id="WR1", air_epa=1, yac_epa=1, xyac_epa=1),
    ])
    p = v.offense_plays(pbp)
    assert list(p["kind"]) == ["target", "target", "no_target", "run", "no_target"]
    catch = p.iloc[0]
    assert catch["throw"] == pytest.approx(0.9) and catch["after_catch"] == pytest.approx(0.3)      # air + expected after catch; the rest
    assert catch["throw"] + catch["after_catch"] == pytest.approx(1.2)
    assert p.iloc[1]["throw"] == pytest.approx(-0.8) and p.iloc[1]["after_catch"] == 0
    # No target: the play's EPA less what an average throw (here 0.05) is worth.
    average_throw = (0.9 - 0.8) / 2
    assert p.iloc[2]["no_target"] == pytest.approx(-1.5 - average_throw)
    assert p.iloc[4]["no_target"] == pytest.approx(0.9 - average_throw) and p.iloc[4]["qb"] == "QB1" and pd.isna(p.iloc[4]["rusher"])
    assert p.iloc[3]["run"] == pytest.approx(0.4) and p.iloc[3]["rusher"] == "RB1" and pd.isna(p.iloc[3]["qb"])


def _season(rng, n=6000):
    """Two teams, each with a quarterback, a starter and reserves at every
    spot. KC's WR_A is truly +0.5 a target; everyone else is average."""
    rows = []
    for i in range(n):
        team, other = ("KC", "BUF") if i % 2 else ("BUF", "KC")
        if rng.random() < 0.55:
            receiver = rng.choice(["WR_A", "WR_B", "WR_C", "WR_D", "TE_A", "TE_B", "RB_A"], p=[0.25, 0.2, 0.15, 0.05, 0.15, 0.05, 0.15])
            receiver = f"{team}_{receiver}"
            complete = rng.random() < 0.65
            bonus = 0.5 if receiver == "KC_WR_A" else 0.0
            epa = rng.normal(0.9 if complete else -0.9, 1.0) + bonus
            rows.append(_play(i, posteam=team, defteam=other, passer_id=f"{team}_QB", receiver_player_id=receiver, epa=epa,
                              complete_pass=int(complete), air_epa=epa * 0.6 if complete else np.nan,
                              yac_epa=epa * 0.4 if complete else np.nan, xyac_epa=0.3 if complete else np.nan))
        elif rng.random() < 0.15:
            rows.append(_play(i, posteam=team, defteam=other, passer_id=f"{team}_QB", epa=rng.normal(-1.2, 0.8), sack=1))
        else:
            rusher = f"{team}_{rng.choice(['RB_A', 'RB_B', 'RB_C'], p=[0.6, 0.3, 0.1])}"
            rows.append(_play(i, posteam=team, defteam=other, play_type="run", qb_dropback=0, passer_id=None, rusher_player_id=rusher,
                              epa=rng.normal(0.0, 0.9)))
    return pd.DataFrame(rows)


def _positions():
    spots = {"QB": "QB", "WR_A": "WR", "WR_B": "WR", "WR_C": "WR", "WR_D": "WR", "TE_A": "TE", "TE_B": "TE",
             "RB_A": "RB", "RB_B": "RB", "RB_C": "RB"}
    return pd.Series({f"{team}_{spot}": position for team in ("KC", "BUF") for spot, position in spots.items()})


def test_tiers_are_starters_and_reserves_by_plays_at_the_position():
    plays = v.offense_plays(_season(np.random.default_rng(1), 3000))
    catch = v._tiers(plays, "receiver", "catch", _positions())
    by_player = pd.Series(catch.to_numpy(), index=plays["receiver"]).groupby(level=0).first()
    assert by_player["KC_WR_A"] == "WR1" and by_player["KC_WR_C"] == "WR1" and by_player["KC_WR_D"] == "WRx"   # three starting receivers
    assert by_player["KC_TE_A"] == "TE1" and by_player["KC_TE_B"] == "TEx" and by_player["KC_RB_A"] == "RB1"
    rush = v._tiers(plays, "rusher", "rush", _positions())
    by_rusher = pd.Series(rush.to_numpy(), index=plays["rusher"]).groupby(level=0).first()
    assert by_rusher["BUF_RB_B"] == "RB1" and by_rusher["BUF_RB_C"] == "RBx"                                    # two starting backs
    assert (catch[plays["receiver"].isna()] == "").all()


def test_the_offense_model_gives_the_credit_to_the_player_who_earned_it():
    rng = np.random.default_rng(2)
    plays = v.offense_plays(_season(rng))
    effects = v.fit_offense(plays, _positions())
    value = v.offense_value(effects).set_index("player_id")
    receivers = value[value["receiving_plays"] > 100]
    assert receivers["receiving_paa"].idxmax() == "KC_WR_A"
    best = value.loc["KC_WR_A"]
    # About +0.5 a target on some 400 targets, less the pull toward average.
    assert 100 < best["receiving_paa"] < 0.5 * best["receiving_plays"] * 1.3
    # His quarterback threw the same passes and gets much less of it.
    assert value.loc["KC_QB", "passing_paa"] < 0.5 * best["receiving_paa"]
    # Above replacement is above average plus the reserve gap on his plays.
    assert best["receiving_par"] > best["receiving_paa"]
    assert best["war"] == pytest.approx(best["par"] / v.POINTS_PER_WIN)
    # A reserve is measured against reserves: about zero above replacement.
    assert abs(value.loc["KC_WR_D", "receiving_par"]) < 12


def test_free_tiers_measure_the_gap_the_constants_replace():
    rng = np.random.default_rng(3)
    season = _season(rng)
    # Make every third-string back 0.4 a carry worse than the others.
    third = season["rusher_player_id"].fillna("").str.endswith("RB_C")
    season.loc[third, "epa"] -= 0.4
    effects = v.fit_offense(v.offense_plays(season), _positions(), gaps=None)
    run = effects[(effects["part"] == "run")]
    starter = np.average(run[run["tier"] == "RB1"]["effect"], weights=run[run["tier"] == "RB1"]["plays"])
    reserve = np.average(run[run["tier"] == "RBx"]["effect"], weights=run[run["tier"] == "RBx"]["plays"])
    assert starter - reserve == pytest.approx(0.4, abs=0.12)
    # With free tiers a starter's replacement is the reserve tier's level.
    assert run[run["tier"] == "RB1"]["replacement"].iloc[0] == pytest.approx(run[run["tier"] == "RBx"]["replacement"].iloc[0])


def test_a_defender_is_credited_only_for_plays_that_cost_the_offense_and_shares_them():
    base = dict(game_id="g1", season=2026, week=1, season_type="REG", posteam="KC", defteam="BUF", play_type="pass", down=1,
                qb_kneel=0, qb_spike=0)
    rows = [
        dict(base, play_id=1, epa=-2.0, sack_player_id="D1"),                                             # a sack, alone
        dict(base, play_id=2, epa=-1.0, half_sack_1_player_id="D1", half_sack_2_player_id="D2"),          # a shared sack
        dict(base, play_id=3, epa=1.5, solo_tackle_1_player_id="D2"),                                     # a tackle after a gain: nothing
        dict(base, play_id=4, epa=-0.6, pass_defense_1_player_id="D3", solo_tackle_1_player_id=None),
        dict(base, play_id=5, epa=-0.9, play_type="run", tackle_for_loss_1_player_id="D2", solo_tackle_1_player_id="D2"),  # named twice, counted once
    ]
    pbp = pd.DataFrame(rows).reindex(columns=["game_id", "play_id", "season", "week", "season_type", "posteam", "defteam", "play_type",
                                              "down", "epa", "qb_kneel", "qb_spike"] + v.DEFENSE_COLUMNS)
    credits = v.defense_credits(pbp)
    by = credits.groupby("player_id")[["points", "sacks", "tackles", "tackles_for_loss", "passes_defended"]].sum()
    assert by.loc["D1", "points"] == pytest.approx(2.0 + 0.5) and by.loc["D1", "sacks"] == 1.5
    assert by.loc["D2", "points"] == pytest.approx(0.5 + 0.9) and by.loc["D2", "sacks"] == 0.5
    assert by.loc["D2", "tackles"] == 1 and by.loc["D2", "tackles_for_loss"] == 1
    assert by.loc["D3", "points"] == pytest.approx(0.6) and by.loc["D3", "passes_defended"] == 1


def test_a_defenders_value_is_his_rate_against_his_position_pulled_toward_it():
    table = pd.DataFrame({
        "player_id": ["a", "b", "c", "d"], "position": ["ED"] * 4, "team": ["KC", "KC", "KC", "BUF"],
        "snaps": [800.0, 800.0, 100.0, 800.0], "points": [56.0, 24.0, 7.0, 32.0], "starter": [True, True, False, True],
    })
    out = v.defense_value(table).set_index("player_id")
    average = 119.0 / 2500
    rate_a = (56 + v.DEFENSE_SHRINK_SNAPS * average) / (800 + v.DEFENSE_SHRINK_SNAPS)
    assert out.loc["a", "paa"] == pytest.approx(v.DEFENSE_SCALE * (rate_a - average) * 800)
    starters = (56 + 24 + 32) / 2400
    assert out.loc["a", "par"] == pytest.approx(v.DEFENSE_SCALE * (rate_a - (starters - v.DEFENSE_REPLACEMENT_GAP["ED"])) * 800)
    assert out.loc["a", "war"] == pytest.approx(out.loc["a", "par"] / v.POINTS_PER_WIN)
    # 100 snaps at the same raw rate as the best starter counts for far less.
    assert 0 < out.loc["c", "paa"] < 0.1 * out.loc["a", "paa"]
    assert out.loc["b", "paa"] < 0 < out.loc["a", "paa"]


def test_a_kicker_is_measured_against_the_chance_of_each_kick():
    base = dict(season_type="REG", posteam="KC", kicker_player_id="K1", extra_point_result=None, field_goal_result=None, fg_prob=np.nan)
    rows = [dict(base, play_type="field_goal", field_goal_result="made", fg_prob=0.6),          # +1.2
            dict(base, play_type="field_goal", field_goal_result="missed", fg_prob=0.9),        # -2.7
            dict(base, play_type="extra_point", extra_point_result="good"),
            dict(base, play_type="extra_point", extra_point_result="failed"),
            dict(base, play_type="pass")]
    out = v.kicker_value(pd.DataFrame(rows)).set_index("player_id")
    assert out.loc["K1", "kicks"] == 4 and out.loc["K1", "fg_made"] == 1 and out.loc["K1", "fg_tried"] == 2
    raw = 1.2 - 2.7 + 0.5 - 0.5
    assert out.loc["K1", "paa"] == pytest.approx(raw * 4 / (4 + v.KICKER_SHRINK_KICKS))
    assert out.loc["K1", "par"] == pytest.approx(out.loc["K1", "paa"] + v.KICKER_REPLACEMENT_GAP * 4)


def test_every_team_is_in_a_division_and_a_missing_season_builds_nothing(tmp_path):
    assert len(v.TEAM_DIVISION) == 32 and all(len(teams) == 4 for teams in v.DIVISIONS.values())
    assert v.TEAM_DIVISION["LA"] == "NFC West" and v.TEAM_ALIASES["LAR"] == "LA"
    assert v.build(2031, str(tmp_path)) == {"season": 2031, "through_week": 0, "points_per_win": v.POINTS_PER_WIN, "players": [], "teams": []}
    assert v.position_group("FB") == "RB" and v.position_group("OL") == "OTH"


def test_team_records_count_a_tie_as_half_a_win():
    schedules = pd.DataFrame({"game_type": ["REG", "REG", "REG", "POST", "REG"], "home_team": ["KC", "KC", "BUF", "KC", "KC"],
                              "away_team": ["BUF", "LAR", "KC", "BUF", "BUF"], "home_score": [27, 20, 17, 30, np.nan],
                              "away_score": [20, 20, 24, 10, np.nan]})
    records = v.team_records(schedules)
    assert records.loc["KC"].to_dict() == {"games": 3, "wins": 2.5, "scored": 71, "allowed": 57}       # the playoff game and the unplayed one are left out
    assert records.loc["LA", "wins"] == 0.5 and records.loc["BUF", "wins"] == 0


def test_the_team_table_turns_players_points_into_wins_above_a_replacement_team():
    records = pd.DataFrame({"games": [17, 17, 17], "wins": [13.0, 8.5, 4.0], "scored": [480, 380, 280], "allowed": [300, 380, 460]}, index=["A", "B", "C"])
    points = pd.DataFrame({"offense": [260.0, 165.0, 70.0], "defense": [90.0, 50.0, 10.0]}, index=["A", "B", "C"])
    t = v.team_table(records, points)
    league = 1140 / 51
    assert t.loc["B", "offense"] == 0 and t.loc["B", "defense"] == 0 and t.loc["B", "expected"] == pytest.approx(8.5)
    assert t.loc["A", "offense"] == pytest.approx(95) and t.loc["A", "defense"] == pytest.approx(40)
    assert t.loc["A", "expected"] == pytest.approx(17 * v.pythagorean(league * 17 + 95, league * 17 - 40))
    # A replacement team: the average team's points above replacement taken away on both sides.
    assert t.loc["B", "baseline"] == pytest.approx(17 * v.pythagorean(league - 165 / 17, league + 50 / 17))
    assert 1.5 < t.loc["B", "baseline"] < 4
    assert (t["baseline"] + t["war"] == t["expected"]).all()
    assert (t["expected"] + t["uncredited"] + t["close"]).round(6).tolist() == [13.0, 8.5, 4.0]
    assert t.loc["A", "close"] == pytest.approx(13 - 17 * v.pythagorean(480, 300))
