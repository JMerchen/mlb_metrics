import pandas as pd
import pytest

from mlb_metrics import config, nfl_estimated_salary


def test_compute_estimated_salary_exact_linear_scaling():
    points = pd.Series([0.0, 5.0, 10.0])
    result = nfl_estimated_salary.compute_estimated_salary(points, 0.0, 10.0, floor=2000, ceiling=6000, round_to=100)

    assert result.tolist() == [2000, 4000, 6000]


def test_compute_estimated_salary_clips_below_floor():
    points = pd.Series([-50.0])
    result = nfl_estimated_salary.compute_estimated_salary(points, 0.0, 10.0, floor=2000, ceiling=6000, round_to=100)

    assert result.iloc[0] == 2000


def test_compute_estimated_salary_clips_above_ceiling():
    points = pd.Series([100.0])
    result = nfl_estimated_salary.compute_estimated_salary(points, 0.0, 10.0, floor=2000, ceiling=6000, round_to=100)

    assert result.iloc[0] == 6000


def test_compute_estimated_salary_degenerate_reference_range_maps_to_floor():
    points = pd.Series([5.0, 5.0])
    result = nfl_estimated_salary.compute_estimated_salary(points, 5.0, 5.0, floor=2000, ceiling=6000, round_to=100)

    assert result.tolist() == [2000, 2000]


def test_qb_skill_dst_salaries_share_same_dollar_per_point_rate():
    # The real bug MLB's own salary-parity fix addressed: a QB, a skill
    # player, and a DST projecting the SAME DK_Points must get the SAME
    # Estimated_Salary - one DK point is worth the same dollar amount
    # regardless of position.
    midpoint = (config.NFL_DFS_REFERENCE_MIN_POINTS + config.NFL_DFS_REFERENCE_MAX_POINTS) / 2
    points = pd.Series([midpoint])

    qb_salary = nfl_estimated_salary.compute_qb_estimated_salary(points)
    skill_salary = nfl_estimated_salary.compute_skill_estimated_salary(points)
    dst_salary = nfl_estimated_salary.compute_dst_estimated_salary(points)

    assert qb_salary.iloc[0] == skill_salary.iloc[0] == dst_salary.iloc[0]


def test_compute_qb_estimated_salary_uses_shared_reference_bounds():
    points = pd.Series([config.NFL_DFS_REFERENCE_MIN_POINTS, config.NFL_DFS_REFERENCE_MAX_POINTS])
    result = nfl_estimated_salary.compute_qb_estimated_salary(points)

    assert result.iloc[0] == config.NFL_DFS_ESTIMATED_SALARY_FLOOR
    assert result.iloc[1] == config.NFL_DFS_ESTIMATED_SALARY_CEILING
