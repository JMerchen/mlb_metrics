import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from mlb_metrics import age_curve_ml, ml_models


def test_year_blocked_split_never_puts_future_year_in_train():
    years = pd.Series([y for y in range(1900, 1960) for _ in range(2)])  # 60 distinct years, 2 rows each
    splitter = age_curve_ml.build_year_blocked_splits(years, min_train_years=40, test_block_years=10)

    folds = list(splitter.split())
    assert len(folds) == splitter.get_n_splits()
    assert len(folds) == 2  # cursor 40 -> [40:50]; cursor 50 -> [50:60]; cursor 60 stops

    unique_years = sorted(years.unique())
    for train_idx, test_idx in folds:
        train_years = set(years.iloc[train_idx])
        test_years = set(years.iloc[test_idx])
        assert train_years.isdisjoint(test_years)
        assert max(train_years) < min(test_years)

    assert set(years.iloc[folds[0][0]]) == set(unique_years[:40])
    assert set(years.iloc[folds[0][1]]) == set(unique_years[40:50])


def test_year_blocked_split_too_few_years_yields_no_folds():
    years = pd.Series(range(1900, 1905))  # 5 years
    splitter = age_curve_ml.build_year_blocked_splits(years, min_train_years=40, test_block_years=10)

    assert list(splitter.split()) == []
    assert splitter.get_n_splits() == 0


def _pitcher_season(player, year, age, ip, k9, bb9, hr9, fip):
    return {"playerID": player, "yearID": year, "age": age, "IP": ip, "K9": k9, "BB9": bb9, "HR9": hr9, "FIP": fip}


def test_build_hr9_training_table_joins_next_season_and_excludes_unresolved():
    historical = pd.DataFrame([
        _pitcher_season("p1", 2000, 25, 180, 8.0, 3.0, 1.0, 4.0),
        _pitcher_season("p1", 2001, 26, 190, 8.5, 2.8, 0.9, 3.8),  # p1's real next season
        _pitcher_season("p2", 2000, 30, 150, 7.0, 3.5, 1.3, 4.5),  # no 2001 row for p2 - excluded
    ])

    result = age_curve_ml.build_hr9_training_table(historical)

    assert set(zip(result["playerID"], result["yearID"])) == {("p1", 2000)}
    row = result[result["playerID"] == "p1"].iloc[0]
    assert row["next_HR9"] == pytest.approx(0.9)
    assert list(result.columns) == ["playerID", "yearID"] + age_curve_ml.FEATURE_COLUMNS + ["next_HR9"]


def test_grid_search_year_blocked_selects_best_alpha():
    rng_years = [y for y in range(1900, 1960) for _ in range(20)]
    import numpy as np
    rng = np.random.RandomState(0)
    x = rng.normal(size=len(rng_years))
    y = 2 * x + rng.normal(scale=0.01, size=len(rng_years))
    X = pd.DataFrame({"x": x})

    search = age_curve_ml.grid_search_year_blocked(
        X, y, rng_years, Ridge(), {"alpha": [0.001, 1000.0]}, min_train_years=40, test_block_years=10,
    )

    assert search.best_params_["alpha"] == 0.001


class _ConstantModel:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return pd.Series([self.value] * len(X)).to_numpy()


def test_project_next_season_ml_returns_point_estimate_in_knn_shaped_dict():
    player_row = pd.Series({"age": 28, "IP": 180, "K9": 9.0, "BB9": 3.0, "HR9": 1.1, "FIP": 3.9})

    result = age_curve_ml.project_next_season_ml(_ConstantModel(1.05), player_row)

    assert result["projected_value_mean"] == pytest.approx(1.05)
    assert result["projected_value_p25"] == pytest.approx(1.05)
    assert result["projected_value_p75"] == pytest.approx(1.05)
    assert pd.isna(result["n_comparables"])
    assert pd.isna(result["n_with_next_season"])
