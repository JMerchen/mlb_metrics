"""Tests for pitch_metrics (Stuff+ and Decision+)."""

import numpy as np
import pandas as pd

from mlb_metrics import pitch_metrics as pm


def _pitches(n=6000, seed=0):
    """Synthetic pitches: pitcher 1 throws harder (97 mph) than 2 (90), and
    harder pitches are worth fewer runs to the hitter; hitter 10 swings at
    strikes and takes balls, hitter 20 the other way round."""
    rng = np.random.default_rng(seed)
    pitcher = rng.choice([1, 2, 3, 4, 5, 6], n)
    velo = np.where(pitcher == 1, 97.0, np.where(pitcher == 2, 90.0, 93.0)) + rng.normal(0, 1, n)
    ptype = rng.choice(["FF", "SL", "CH"], n, p=[0.6, 0.25, 0.15])
    plate_x = rng.uniform(-1.6, 1.6, n)
    plate_z = rng.uniform(1.0, 4.0, n)
    in_zone = (np.abs(plate_x) <= 0.83) & (plate_z >= 1.6) & (plate_z <= 3.4)
    batter = rng.choice([10, 20, 30, 40], n)
    swing = np.where(batter == 10, in_zone, np.where(batter == 20, ~in_zone, rng.random(n) < 0.47))
    desc = np.where(swing, np.where(rng.random(n) < 0.5, "foul", "hit_into_play"), np.where(in_zone, "called_strike", "ball"))
    # Swinging at strikes and taking balls pays; the reverse costs.
    rv = np.where(swing, np.where(in_zone, 0.02, -0.04), np.where(in_zone, -0.03, 0.03)) - 0.004 * (velo - 93) + rng.normal(0, 0.02, n)
    return pd.DataFrame({
        "game_pk": rng.integers(1, 200, n), "game_date": pd.Timestamp("2026-05-01") + pd.to_timedelta(rng.integers(0, 100, n), "D"),
        "pitch_type": ptype, "release_speed": velo, "pfx_x": rng.normal(-0.5, 0.3, n), "pfx_z": rng.normal(1.2, 0.3, n),
        "release_spin_rate": rng.normal(2300, 100, n), "spin_axis": rng.normal(200, 20, n), "release_extension": rng.normal(6.4, 0.2, n),
        "release_pos_x": rng.normal(-2, 0.2, n), "release_pos_z": rng.normal(5.8, 0.2, n), "arm_angle": rng.normal(45, 5, n),
        "vx0": rng.normal(6, 1, n), "vy0": -velo * 1.46, "vz0": rng.normal(-5, 1, n), "ax": rng.normal(-10, 2, n), "ay": rng.normal(28, 2, n),
        "az": rng.normal(-15, 3, n), "p_throws": np.where(pitcher % 2 == 0, "L", "R"), "stand": np.where(batter == 20, "L", "R"),
        "pitcher": pitcher, "batter": batter, "delta_run_exp": rv, "description": desc, "plate_x": plate_x, "plate_z": plate_z,
        "sz_top": 3.4, "sz_bot": 1.6, "balls": rng.integers(0, 4, n), "strikes": rng.integers(0, 3, n), "outs_when_up": rng.integers(0, 3, n),
        "on_1b": np.nan, "on_2b": np.nan, "on_3b": np.nan,
    })


def test_features_mirror_lefties_and_measure_off_the_fastball():
    p = _pitches(600)
    f = pm.stuff_features(p)
    lefty = (p["p_throws"] == "L").to_numpy()
    assert np.allclose(f.loc[lefty, "hb"], -p.loc[lefty, "pfx_x"] * 12)
    assert np.allclose(f.loc[~lefty, "rel_x"], p.loc[~lefty, "release_pos_x"])
    ff = (p["pitch_type"] == "FF").to_numpy()
    # A fastball's velocity gap from its own pitcher's fastball average is small.
    assert abs(np.nanmean(f.loc[ff, "velo_diff"])) < 0.5
    assert np.isfinite(f["vaa"]).all()


def test_stuff_plus_rewards_the_harder_thrower_without_seeing_him():
    p = _pitches()
    pred = pm.stuff_predictions(p, pm.stuff_features(p), folds=3)
    assert np.isfinite(pred).all()
    _, by_pitcher = pm.stuff_plus(p, pred)
    grade = by_pitcher.set_index("pitcher")["stuff_plus"]
    assert grade[1] > 100 > grade[2]
    assert abs(np.average(by_pitcher["stuff_plus"], weights=by_pitcher["pitches"]) - 100) < 0.5


def test_decision_plus_rewards_swinging_at_strikes_and_taking_balls():
    p = _pitches()
    f = pm.decision_frame(p)
    assert set(f["swing"].unique()) == {0, 1}
    values = pm.decision_values(f, pm.decision_models(f))
    grades = pm.decision_plus(values).set_index("batter")
    assert grades.loc[10, "decision_plus"] > 100 > grades.loc[20, "decision_plus"]
    assert grades.loc[20, "chase"] > grades.loc[10, "chase"]


def test_plus_scale_and_shrinkage():
    values = pd.Series([1.0, 2.0, 3.0])
    weights = pd.Series([100, 100, 100])
    plus = pm._plus(values, weights, lower_is_better=False)
    assert np.allclose(plus, [100 - 10 * np.sqrt(1.5), 100, 100 + 10 * np.sqrt(1.5)])
    # A small sample is pulled toward the middle more than a big one.
    uneven = pd.Series([2000, 2000, 50])
    raw = pm._plus(values, uneven, lower_is_better=False)
    shrunk = pm._plus(values, uneven, lower_is_better=False, shrink=300)
    assert abs(shrunk[2] - shrunk[1]) < abs(raw[2] - raw[1])
