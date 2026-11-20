"""Ranges with a stated hit rate (src/mlb_metrics/conformal.py)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlb_metrics import config, conformal

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_a_range_read_off_past_misses_hits_as_often_as_it_says_whatever_their_shape():
    """The point of the method: no bell curve assumed. Misses here are
    lopsided (most a little under, a few far over), as a receiver's are."""
    rng = np.random.default_rng(0)
    miss = lambda n: rng.exponential(20, n) - 14
    past_pred = rng.uniform(20, 80, 2000)
    sorted_scores = conformal.scores(past_pred, past_pred + miss(2000))
    new_pred = rng.uniform(20, 80, 20000)
    actual = new_pred + miss(20000)
    for coverage in (0.5, 0.8, 0.95):
        low, high = conformal.interval(new_pred, sorted_scores, coverage)
        assert np.mean((actual >= low) & (actual <= high)) == pytest.approx(coverage, abs=0.02)
    # The range is lopsided like the misses: more room above than below.
    low, high = conformal.band(sorted_scores, 0.8)
    assert high > -low > 0


def test_too_few_past_misses_promise_nothing_on_that_side():
    assert conformal.band(np.array([-1.0, 0.0, 2.0]), 0.8) == (-np.inf, np.inf)
    low, high = conformal.band(np.arange(-4.0, 5.0), 0.8)       # nine misses: ranks 1 and 9
    assert (low, high) == (-4.0, 4.0)


def test_grouping_and_scaling_keep_the_hit_rate_for_small_and_large_predictions():
    """Why (2026-11-20): one range for every player projection was right
    on average and wrong for almost everyone - 80% ranges caught 92% of
    small receiving-yard projections and 59% of large ones."""
    rng = np.random.default_rng(1)

    def sample(n):
        pred = rng.uniform(5, 100, n)
        return pred, np.maximum(0.0, pred + rng.normal(0, 0.45 * pred))

    past_pred, past_actual = sample(6000)
    new_pred, new_actual = sample(6000)
    small, large = new_pred < 30, new_pred > 75

    def hit(cal, rows):
        ranges = [conformal.range_for(cal, p, 0.8, 0.0) for p in new_pred[rows]]
        return np.mean([(lo <= a <= hi) for (lo, hi), a in zip(ranges, new_actual[rows])])

    plain = conformal.calibrate(past_pred, past_actual)
    assert hit(plain, small) > 0.95 and hit(plain, large) < 0.70
    used = conformal.calibrate(past_pred, past_actual, groups=5, scaled=True)
    assert hit(used, small) == pytest.approx(0.8, abs=0.04) and hit(used, large) == pytest.approx(0.8, abs=0.04)
    assert len(used["groups"]) == 5 and len(used["edges"]) == 4 and used["scale"]["slope"] > 0


def test_the_chance_over_a_line_is_honest_and_never_certain():
    rng = np.random.default_rng(2)
    pred = rng.uniform(30, 70, 4000)
    cal = conformal.calibrate(pred, pred + rng.normal(0, 15, 4000))
    assert conformal.chance_over_line(cal, 50.0, 50.0) == pytest.approx(0.5, abs=0.03)
    assert conformal.chance_over_line(cal, 50.0, 65.0) == pytest.approx(0.16, abs=0.03)
    assert 0 < conformal.chance_over_line(cal, 50.0, 500.0) < 0.001
    assert 0.999 < conformal.chance_over_line(cal, 50.0, -500.0) < 1
    # The exact count agrees with the shipped summary of the misses.
    exact = conformal.chance_over(50.0, 60.0, conformal.scores(pred, pred + rng.normal(0, 15, 4000)))
    assert conformal.chance_over_line(cal, 50.0, 60.0) == pytest.approx(exact, abs=0.02)


def test_the_site_file_matches_the_hand_set_total_range_and_serves_every_projected_category():
    """The pick card shows the total as plus or minus
    config.NFL_TOTAL_BAND_HALF_WIDTH for "half the time"; the misses say
    the same, or one of them needs changing."""
    ranges = json.loads((REPO_ROOT / "docs" / "data" / "nfl_ranges.json").read_text())
    low, high = conformal.range_for(ranges["total"], 45.0, 0.5)
    assert (high - low) / 2 == pytest.approx(config.NFL_TOTAL_BAND_HALF_WIDTH, abs=0.75)
    wide_low, wide_high = conformal.range_for(ranges["total"], 45.0, 0.8)
    assert wide_low < low and wide_high > high
    assert set(ranges["props"]) == {"Passing Yards", "Receiving Yards", "Receptions", "Rushing Yards"}
    assert ranges["props"]["Rushing Yards"]["floor"] is None and ranges["props"]["Receptions"]["floor"] == 0.0


def test_build_puts_game_and_player_calibrations_in_one_file():
    spec = importlib.util.spec_from_file_location("build_nfl_ranges", REPO_ROOT / "scripts" / "build_nfl_ranges.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    rng = np.random.default_rng(6)
    misses = pd.DataFrame({"our_total": rng.uniform(35, 55, 400), "our_margin": rng.uniform(-10, 10, 400)})
    misses["total"] = misses["our_total"] + rng.normal(0, 13, 400)
    misses["margin"] = misses["our_margin"] + rng.normal(0, 13, 400)
    misses.loc[:50, ["our_total", "total"]] = np.nan          # margins go back further than totals
    props = pd.DataFrame({"category": ["Receptions"] * 300 + ["Rushing Yards"] * 300, "season": 2026,
                          "week": np.tile(np.arange(4, 10), 100), "projection": rng.uniform(1, 80, 600)})
    props["actual"] = props["projection"] + rng.normal(0, 5, 600)
    out = script.build(misses, props)
    assert out["total"]["n"] == 349 and out["margin"]["n"] == 400
    assert out["through"] == {"season": 2026, "week": 9}
    assert len(out["props"]["Receptions"]["groups"]) == script.PROP_GROUPS
    assert out["props"]["Rushing Yards"]["floor"] is None
    assert script.build(None, pd.DataFrame()) == {"levels": [0.5, 0.8], "props": {}}
