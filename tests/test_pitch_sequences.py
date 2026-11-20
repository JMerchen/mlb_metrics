"""Pitches as sequences (src/mlb_metrics/pitch_sequences.py)."""

import numpy as np
import pandas as pd
import pytest

from mlb_metrics import pitch_sequences as ps


def _pitch(game, at_bat, number, pitch_type, description, **kw):
    row = {"game_date": "2026-05-01", "game_pk": game, "at_bat_number": at_bat, "pitch_number": number, "pitcher": 10, "batter": 20,
           "pitch_type": pitch_type, "description": description, "balls": 0, "strikes": 0, "stand": "R", "p_throws": "R",
           "game_type": "R", "release_speed": 94.0, "pfx_x": -0.6, "pfx_z": 1.3, "plate_x": 0.4, "plate_z": 2.5}
    row.update(kw)
    return row


def _raw():
    return pd.DataFrame([
        # Given out of order on purpose; a second at-bat by the same hitter; a lefty; a spring game.
        _pitch(1, 1, 2, "SL", "swinging_strike", release_speed=85.0, plate_x=0.9, plate_z=1.4),
        _pitch(1, 1, 1, "FF", "called_strike"),
        _pitch(1, 1, 3, "SL", "hit_into_play", release_speed=86.0),
        _pitch(1, 2, 1, "CH", "ball", batter=21, p_throws="L", pfx_x=1.2, plate_x=-0.5),
        _pitch(1, 3, 1, "FF", "foul_tip"),
        _pitch(1, 3, 2, "XX", "foul"),
        _pitch(1, 3, 3, "FF", "some_new_description"),
        _pitch(2, 1, 1, "FF", "ball", game_type="S"),
        _pitch(1, 4, 1, "FF", "ball", release_speed=np.nan),
    ])


def test_clean_orders_pitches_names_outcomes_and_mirrors_left_handers():
    d = ps.clean(_raw())
    # The spring game, the unknown description and the unmeasured pitch are dropped.
    assert len(d) == 6
    assert list(d["pitch_number"][:3]) == [1, 2, 3]
    assert [ps.OUTCOMES[o] for o in d["outcome"]] == ["called_strike", "whiff", "in_play", "ball", "whiff", "foul"]
    assert ps.PITCH_TYPES[d["type"].iloc[5]] == "other"
    lefty = d[d["p_throws"] == "L"].iloc[0]
    assert (lefty["pfx_x"], lefty["plate_x"]) == (-1.2, 0.5)          # arm side means the same for everyone
    assert list(d["game_pitch"]) == [0, 1, 2, 3, 4, 5]
    # The second time this hitter faces him today.
    assert list(d["times_faced"]) == [0, 0, 0, 0, 1, 1]
    assert list(d["family"]) == ["fast", "breaking", "breaking", "offspeed", "fast", "other"]


def test_player_rates_are_pulled_toward_the_league_and_default_to_it():
    d = ps.clean(_raw())
    rates = ps.player_rates(d)
    assert rates["league"]["swing"] == pytest.approx(4 / 6)
    assert rates["league"]["whiff"] == pytest.approx(2 / 4)
    # One pitcher threw all six pitches: with 200 pitches of prior he sits at the league's rate.
    assert rates["pitcher"].loc[10, "pitcher_whiff"] == pytest.approx(0.5)
    # His slider: 2 swings, 1 whiff, against a league slider rate of 0.5.
    slider = ps.PITCH_TYPES.index("SL")
    assert rates["pitcher_type"].loc[(10, slider), "type_usage"] == pytest.approx(2 / 6)
    with_rates = ps.add_player_rates(d.assign(pitcher=[10, 10, 10, 10, 10, 99]), rates)
    assert with_rates["pitcher_whiff"].iloc[5] == pytest.approx(rates["league"]["whiff"])
    assert with_rates["type_usage"].iloc[5] == 0.0
    assert set(ps.STATIC) <= set(with_rates.columns)


def test_previous_pitch_features_stay_inside_the_at_bat():
    d = ps.clean(_raw())
    prev = ps.previous_pitch_features(d)
    assert list(prev["first_pitch"]) == [1, 0, 0, 1, 1, 0]
    assert prev["speed_change"].iloc[1] == pytest.approx(-9.0)
    assert list(prev["same_type"]) == [0, 0, 1, 0, 0, 0]
    assert prev["prev_outcome"].iloc[2] == ps.OUTCOMES.index("whiff") and prev["prev_type"].iloc[3] == -1


def test_sequences_show_only_as_much_history_as_asked_and_never_the_answer():
    d = ps.add_player_rates(ps.clean(_raw()), ps.player_rates(ps.clean(_raw())))
    scale = ps.fit_scale(d)
    known, results = len(ps.PITCH_TYPES) + len(ps.NUMERIC), len(ps.OUTCOMES)
    steps, static, y = ps.sequences(d, scale, "results")
    assert steps.shape == (6, ps.MAX_LEN, known + 1 + results) and static.shape == (6, len(ps.STATIC))
    assert list(y) == list(d["outcome"])
    third = steps[2]                                   # the at-bat's third pitch
    assert third[-1, known] == 1.0 and third[-1, -results:].sum() == 0.0        # flagged, result hidden
    assert third[-2, -results:].argmax() == ps.OUTCOMES.index("whiff")          # the pitch before it ended in a whiff
    assert third[-3, ps.PITCH_TYPES.index("FF")] == 1.0 and third[:-3].sum() == 0.0
    # The first pitch of the next at-bat sees nothing of the last one.
    assert steps[3, :-1].sum() == 0.0
    pitches_only, _, _ = ps.sequences(d, scale, "pitches")
    assert pitches_only[2, -2, ps.PITCH_TYPES.index("SL")] == 1.0 and pitches_only[2, :-1, -results:].sum() == 0.0
    alone, _, _ = ps.sequences(d, scale, "none")
    assert alone[:, :-1].sum() == 0.0
    assert np.array_equal(alone[:, -1], steps[:, -1])


def test_training_pitches_get_their_rates_from_other_games():
    """A rate that includes the pitch it is attached to leaks the answer."""
    rows = []
    for game in range(1, 11):
        # In game 1 the pitcher gets a whiff on every swing; in the rest, never.
        for number in range(1, 9):
            rows.append(_pitch(game, 1, number, "SL", "swinging_strike" if game == 1 else "foul"))
    d = ps.clean(pd.DataFrame(rows))
    leaky = ps.add_player_rates(d, ps.player_rates(d, strength=1.0, type_strength=1.0))
    held_out = ps.add_training_rates(d, folds=5)
    assert len(held_out) == len(d) and list(held_out["game_pk"]) == list(d["game_pk"])
    first = held_out["game_pk"] == 1
    # Game 1 shares a fold with game 6; the other eight games had no whiffs.
    assert held_out.loc[first, "type_whiff"].max() < 0.02
    assert leaky.loc[leaky["game_pk"] == 1, "type_whiff"].min() > 0.09
