import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "backtest_game_pick_discrimination.py")
spec = importlib.util.spec_from_file_location("backtest_game_pick_discrimination", SCRIPT)
bt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bt)

from mlb_metrics import game_picks  # noqa: E402


def _features(n=200, seed=0):
    """Real-shaped feature rows: every column game_picks.build_game_features
    produces, so the helpers under test see the same shape they would live."""
    rng = np.random.default_rng(seed)
    rows = {
        "game_pk": np.arange(n),
        "date": pd.to_datetime("2026-04-01") + pd.to_timedelta(rng.integers(0, 30, n), unit="D"),
        "home_team": ["NYY"] * n,
        "away_team": ["BOS"] * n,
        "home_composite": rng.normal(1.0, 0.08, n),
        "away_composite": rng.normal(1.0, 0.08, n),
    }
    for side in ("home", "away"):
        for kind in ("bullpen", "starter"):
            rows[f"{side}_{kind}_pave_plus"] = rng.normal(100, 10, n)
            rows[f"{side}_{kind}_power_a_plus"] = rng.normal(100, 10, n)
    return pd.DataFrame(rows)


def test_murphy_decomposition_satisfies_the_brier_identity():
    # Brier = Reliability - Resolution + Uncertainty. If this identity
    # doesn't hold, every resolution number this backtest reports - the
    # whole basis for "is a wider spread honest" - is meaningless.
    rng = np.random.default_rng(1)
    p = rng.uniform(0.35, 0.65, 3000)
    y = (rng.uniform(size=3000) < p).astype(int)

    d = bt.murphy_decomposition(p, y, bins=10)
    brier = ((p - y) ** 2).mean()

    assert d["reliability"] - d["resolution"] + d["uncertainty"] == pytest.approx(brier, abs=2e-3)


def test_murphy_decomposition_resolution_is_zero_for_a_constant_forecast():
    # A real forecast that says the same thing about every game separates
    # nothing - resolution must be exactly 0, which is the degenerate end
    # of the scale this backtest measures against.
    y = np.array([1, 0, 1, 0, 1, 0, 1, 1, 0, 0] * 20)
    p = np.full(len(y), 0.5)

    assert bt.murphy_decomposition(p, y, bins=10)["resolution"] == pytest.approx(0.0, abs=1e-12)


def test_murphy_decomposition_resolution_is_high_for_a_perfect_forecast():
    y = np.array([1, 0] * 100)
    p = y.astype(float) * 0.98 + 0.01  # near-perfect, correctly ordered

    d = bt.murphy_decomposition(p, y, bins=2)

    assert d["resolution"] == pytest.approx(d["uncertainty"], abs=1e-6)


def test_probability_with_home_field_zero_reproduces_the_live_formula_exactly():
    # The whole point of rebuilding ratings from stored features is to
    # compare candidates against the REAL live model, not an approximation
    # of it - so at hfa=0 this must match compute_game_win_probabilities
    # exactly, not merely closely.
    df = _features()
    schedule = df[["game_pk", "home_team", "away_team"]].copy()
    schedule["date"] = df["date"]

    rebuilt = bt.probability_with_home_field(df, 0.0)

    home, away = bt._ratings(df)
    from mlb_metrics import config
    floor = config.GAME_PICK_RATING_FLOOR
    expected = np.clip(home, floor, None) / (np.clip(home, floor, None) + np.clip(away, floor, None))
    assert np.abs(rebuilt - expected).max() == 0.0


def test_probability_with_home_field_raises_the_home_side():
    df = _features()

    assert bt.probability_with_home_field(df, 0.15).mean() > bt.probability_with_home_field(df, 0.0).mean()


def test_walk_forward_logistic_never_predicts_without_enough_real_history(monkeypatch):
    # No-lookahead discipline: the earliest real dates must come back NaN
    # rather than being scored by a model fit on games that hadn't
    # happened yet.
    monkeypatch.setattr(bt, "MIN_TRAIN_GAMES", 50)
    df = _features(n=120).sort_values("date").reset_index(drop=True)
    y = (np.arange(len(df)) % 2).astype(int)

    preds = bt.walk_forward_logistic(df, y)

    assert np.isnan(preds[0])  # first real date can never be predicted
    assert not np.isnan(preds).all()  # but later real dates are


def test_score_reports_auc_and_the_decomposition_together():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.4, 0.6, 500)
    y = (rng.uniform(size=500) < p).astype(int)

    row = bt.score(p, y, "candidate")

    assert row["model"] == "candidate"
    assert set(["AUC", "Brier", "logloss", "bias", "std", "reliability", "resolution", "uncertainty"]) <= set(row)
    assert row["bias"] == pytest.approx(p.mean() - y.mean())
