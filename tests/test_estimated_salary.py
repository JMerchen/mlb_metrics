import pandas as pd
import pytest

from mlb_metrics import estimated_salary


def test_compute_estimated_salary_exact_linear_scaling():
    # reference range 0-10, floor 2000, ceiling 6000, round to 100.
    points = pd.Series([0.0, 5.0, 10.0])
    result = estimated_salary.compute_estimated_salary(points, 0.0, 10.0, floor=2000, ceiling=6000, round_to=100)

    assert result.tolist() == [2000, 4000, 6000]


def test_compute_estimated_salary_clips_below_floor():
    points = pd.Series([-5.0])
    result = estimated_salary.compute_estimated_salary(points, 0.0, 10.0, floor=2000, ceiling=6000, round_to=100)

    assert result.iloc[0] == 2000


def test_compute_estimated_salary_clips_above_ceiling():
    points = pd.Series([100.0])
    result = estimated_salary.compute_estimated_salary(points, 0.0, 10.0, floor=2000, ceiling=6000, round_to=100)

    assert result.iloc[0] == 6000


def test_compute_estimated_salary_rounds_to_nearest_increment():
    # fraction = 0.33 -> raw = 2000 + 0.33*4000 = 3320 -> rounds to 3300.
    points = pd.Series([3.3])
    result = estimated_salary.compute_estimated_salary(points, 0.0, 10.0, floor=2000, ceiling=6000, round_to=100)

    assert result.iloc[0] == pytest.approx(3300)


def test_compute_estimated_salary_degenerate_reference_range_maps_to_floor():
    # reference_max == reference_min - must not divide by zero.
    points = pd.Series([5.0, 5.0])
    result = estimated_salary.compute_estimated_salary(points, 5.0, 5.0, floor=2000, ceiling=6000, round_to=100)

    assert result.tolist() == [2000, 2000]


def test_compute_hitter_estimated_salary_uses_configured_hitter_bounds():
    from mlb_metrics import config
    points = pd.Series([config.DFS_REFERENCE_MIN_POINTS_HITTER, config.DFS_REFERENCE_MAX_POINTS_HITTER])

    result = estimated_salary.compute_hitter_estimated_salary(points)

    assert result.iloc[0] == config.DFS_ESTIMATED_SALARY_FLOOR
    assert result.iloc[1] == config.DFS_ESTIMATED_SALARY_CEILING_HITTER


def test_compute_pitcher_estimated_salary_uses_configured_pitcher_bounds():
    from mlb_metrics import config
    points = pd.Series([config.DFS_REFERENCE_MIN_POINTS_PITCHER, config.DFS_REFERENCE_MAX_POINTS_PITCHER])

    result = estimated_salary.compute_pitcher_estimated_salary(points)

    assert result.iloc[0] == config.DFS_ESTIMATED_SALARY_FLOOR
    assert result.iloc[1] == config.DFS_ESTIMATED_SALARY_CEILING_PITCHER
