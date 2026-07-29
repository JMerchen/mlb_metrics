import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from mlb_metrics import ml_models


def test_walk_forward_date_split_never_puts_future_in_train():
    dates = pd.Series([f"2026-01-{d:02d}" for d in range(1, 21) for _ in range(3)])  # 20 dates, 3 rows each
    splitter = ml_models.WalkForwardDateSplit(dates, min_train_dates=10, test_block_dates=5)

    folds = list(splitter.split())
    assert len(folds) == splitter.get_n_splits()
    assert len(folds) == 2  # cursor 10 -> test [10:15]; cursor 15 -> test [15:20]; cursor 20 stops

    unique_dates = sorted(dates.unique())
    for train_idx, test_idx in folds:
        train_dates = set(dates.iloc[train_idx])
        test_dates = set(dates.iloc[test_idx])
        assert train_dates.isdisjoint(test_dates)
        max_train_date = max(train_dates)
        min_test_date = min(test_dates)
        assert unique_dates.index(max_train_date) < unique_dates.index(min_test_date)

    # First fold trains on the first 10 dates only.
    first_train_dates = set(dates.iloc[folds[0][0]])
    assert first_train_dates == set(unique_dates[:10])
    first_test_dates = set(dates.iloc[folds[0][1]])
    assert first_test_dates == set(unique_dates[10:15])


def test_walk_forward_date_split_too_few_dates_yields_no_folds():
    dates = pd.Series([f"2026-01-{d:02d}" for d in range(1, 6)])  # 5 dates
    splitter = ml_models.WalkForwardDateSplit(dates, min_train_dates=10, test_block_dates=5)

    assert list(splitter.split()) == []
    assert splitter.get_n_splits() == 0


def test_grid_search_walk_forward_selects_best_alpha_and_respects_split():
    rng = np.random.RandomState(0)
    n_dates = 20
    rows_per_date = 20
    dates = np.repeat([f"2026-01-{d:02d}" for d in range(1, n_dates + 1)], rows_per_date)
    x = rng.normal(size=len(dates))
    y = 3 * x + rng.normal(scale=0.01, size=len(dates))  # near-noiseless linear signal
    X = pd.DataFrame({"x": x})

    search = ml_models.grid_search_walk_forward(
        X, y, dates, Ridge(), {"alpha": [0.001, 1000.0]},
        min_train_dates=10, test_block_dates=5,
    )
    # A near-noiseless linear relationship should strongly prefer the
    # unregularized (tiny-alpha) fit over one crushed toward 0 by alpha=1000.
    assert search.best_params_["alpha"] == 0.001
    assert search.n_splits_ == ml_models.WalkForwardDateSplit(dates, 10, 5).get_n_splits()


def test_evaluate_predictions_exact_arithmetic():
    actual = pd.Series([1.0, 2.0, 3.0, 4.0])
    predicted = pd.Series([1.0, 2.0, 3.0, 6.0])  # last prediction off by 2

    result = ml_models.evaluate_predictions(actual, predicted)

    assert result["n"] == 4
    assert result["mae"] == pytest.approx(0.5)  # (0+0+0+2)/4
    assert result["baseline_mae"] == pytest.approx(1.0)  # mean=2.5, |1-2.5|+|2-2.5|+|3-2.5|+|4-2.5| = 1+0.5+0.5+1.5=3.5/4
    assert result["correlation"] == pytest.approx(actual.corr(predicted))


def test_evaluate_predictions_empty_returns_nan_not_crash():
    result = ml_models.evaluate_predictions(pd.Series([], dtype=float), pd.Series([], dtype=float))
    assert result["n"] == 0
    assert pd.isna(result["mae"])
    assert pd.isna(result["baseline_mae"])
    assert pd.isna(result["correlation"])


def test_evaluate_classifier_predictions_exact_arithmetic():
    import math

    actual = pd.Series([0, 0, 1, 1])
    predicted_proba = pd.Series([0.1, 0.4, 0.6, 0.9])  # every prediction on the correct side of 0.5

    result = ml_models.evaluate_classifier_predictions(actual, predicted_proba)

    assert result["n"] == 4
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["roc_auc"] == pytest.approx(1.0)  # perfect rank separation

    expected_log_loss = -sum([
        math.log(1 - 0.1), math.log(1 - 0.4), math.log(0.6), math.log(0.9),
    ]) / 4
    assert result["log_loss"] == pytest.approx(expected_log_loss)

    expected_brier = ((0.1 - 0) ** 2 + (0.4 - 0) ** 2 + (0.6 - 1) ** 2 + (0.9 - 1) ** 2) / 4
    assert result["brier_score"] == pytest.approx(expected_brier)

    # base rate = 0.5 -> always predicting 0.5 gives -log(0.5) per row regardless of actual
    assert result["baseline_log_loss"] == pytest.approx(-math.log(0.5))


def test_evaluate_classifier_predictions_single_class_auc_is_nan_not_raise():
    actual = pd.Series([1, 1, 1])
    predicted_proba = pd.Series([0.6, 0.7, 0.8])

    result = ml_models.evaluate_classifier_predictions(actual, predicted_proba)

    assert result["n"] == 3
    assert pd.isna(result["roc_auc"])
    assert result["accuracy"] == pytest.approx(1.0)


def test_evaluate_classifier_predictions_empty_returns_nan_not_crash():
    result = ml_models.evaluate_classifier_predictions(pd.Series([], dtype=float), pd.Series([], dtype=float))
    assert result["n"] == 0
    assert pd.isna(result["accuracy"])
    assert pd.isna(result["log_loss"])
    assert pd.isna(result["brier_score"])
    assert pd.isna(result["roc_auc"])
    assert pd.isna(result["baseline_log_loss"])


def test_save_and_load_model_round_trips(tmp_path):
    model = Ridge(alpha=5.0).fit([[1], [2], [3]], [2, 4, 6])
    path = str(tmp_path / "nested" / "model.joblib")

    ml_models.save_model(model, path)
    loaded = ml_models.load_model(path)

    assert loaded.alpha == 5.0
    assert loaded.predict([[4]])[0] == pytest.approx(model.predict([[4]])[0])


def test_load_model_missing_file_returns_none(tmp_path):
    assert ml_models.load_model(str(tmp_path / "does_not_exist.joblib")) is None


def test_load_model_corrupt_file_returns_none(tmp_path):
    path = tmp_path / "corrupt.joblib"
    path.write_text("not a real joblib file")
    assert ml_models.load_model(str(path)) is None
