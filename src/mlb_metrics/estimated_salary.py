"""Estimated_Salary: a MODELED DraftKings-style price, derived from this
project's own DK_Points_Hitter/DK_Points_Pitcher projections - explicitly
NOT a real DraftKings salary. DraftKings has no public API for contest
salaries (there is no free, ToS-compliant way to fetch them), so a real
price cannot be sourced automatically. Rather than scrape DraftKings
(fragile, likely against their ToS) or require a manual daily CSV upload,
this builds a defensible estimate from data this project already computes -
an explicit user decision ("build your best guess at pricing based on
performance"), not a fallback landed on by default.

**Never treat this as a real price.** A real DFS player could lose real
money mistaking this for what DraftKings will actually charge, or building
a lineup here that isn't a valid budget on the real platform. Every place
this number surfaces uses the name `Estimated_Salary`, never bare `Salary`
or `DK_Salary` (config.py, this module, dfs_optimizer.py,
scripts/build_optimal_lineup.py, the CSV column header, docs/dfs.html's
warning box, docs/dfs.js's rendered table header, README).

## The formula

Linear min-max scaling of DK_Points_Hitter/DK_Points_Pitcher into
[config.DFS_ESTIMATED_SALARY_FLOOR, config.DFS_ESTIMATED_SALARY_CEILING_HITTER
or _PITCHER], clipped at both ends, then rounded to the nearest
config.DFS_ESTIMATED_SALARY_ROUND_TO ($100 - real DK salaries are always
$100 increments, confirmed via web search). The reference min/max points
each scale maps FROM (config.DFS_REFERENCE_MIN/MAX_POINTS_HITTER/PITCHER)
are computed from this project's own first real day of production DFS
output, not guessed - see those constants' docstring in config.py for the
exact numbers and their v1-calibration caveat.

This is a single-signal (DK_Points only) linear model - real DK pricing
also reflects season-long track record, popularity/ownership effects, and
other signals this project doesn't compute. A deliberate v1 simplification,
documented here rather than hidden."""

import pandas as pd

from mlb_metrics import config


def compute_estimated_salary(
    dk_points: pd.Series,
    reference_min_points: float,
    reference_max_points: float,
    floor: int = config.DFS_ESTIMATED_SALARY_FLOOR,
    ceiling: int = config.DFS_ESTIMATED_SALARY_CEILING_HITTER,
    round_to: int = config.DFS_ESTIMATED_SALARY_ROUND_TO,
) -> pd.Series:
    """Linear min-max scaling of `dk_points` into [floor, ceiling], clipped
    at both ends (a point value outside the reference range still produces
    a valid, clipped salary rather than an out-of-range one), rounded to
    the nearest `round_to`. `reference_max_points == reference_min_points`
    (a degenerate single-point reference range) maps everything to `floor`
    rather than dividing by zero."""
    span = reference_max_points - reference_min_points
    if span <= 0:
        fraction = pd.Series(0.0, index=dk_points.index)
    else:
        fraction = ((dk_points - reference_min_points) / span).clip(0, 1)

    raw_salary = floor + fraction * (ceiling - floor)
    rounded = (raw_salary / round_to).round() * round_to
    return rounded.clip(lower=floor, upper=ceiling)


def compute_hitter_estimated_salary(dk_points_hitter: pd.Series) -> pd.Series:
    return compute_estimated_salary(
        dk_points_hitter,
        config.DFS_REFERENCE_MIN_POINTS_HITTER,
        config.DFS_REFERENCE_MAX_POINTS_HITTER,
        ceiling=config.DFS_ESTIMATED_SALARY_CEILING_HITTER,
    )


def compute_pitcher_estimated_salary(dk_points_pitcher: pd.Series) -> pd.Series:
    return compute_estimated_salary(
        dk_points_pitcher,
        config.DFS_REFERENCE_MIN_POINTS_PITCHER,
        config.DFS_REFERENCE_MAX_POINTS_PITCHER,
        ceiling=config.DFS_ESTIMATED_SALARY_CEILING_PITCHER,
    )
