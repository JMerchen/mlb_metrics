"""Estimated_Salary for NFL DFS - direct structural port of
estimated_salary.py, applied to DK_Points_QB/DK_Points_Skill/
DK_Points_DST instead of DK_Points_Hitter/DK_Points_Pitcher. See that
module's docstring for the full "never a real DraftKings price" warning
and the "why ONE shared reference range across position groups, not a
per-group one" reasoning - both apply identically here: QB, skill, and
DST DK scoring each span very different raw point ranges (a DST rarely
scores double digits; a strong QB week regularly does), so a
per-position-group scale would misprice a point the same way the old MLB
hitter/pitcher split did before its fix.

**Never treat this as a real price.** Always surfaced as
`Estimated_Salary`, never bare `Salary`/`DK_Salary` - same disclaimer
requirement as the MLB module (config.py, this module,
nfl_dfs_optimizer.py, the CSV column header, docs/nfl.html's warning
box, README).
"""

import pandas as pd

from mlb_metrics import config


def compute_estimated_salary(
    dk_points: pd.Series,
    reference_min_points: float,
    reference_max_points: float,
    floor: int = config.NFL_DFS_ESTIMATED_SALARY_FLOOR,
    ceiling: int = config.NFL_DFS_ESTIMATED_SALARY_CEILING,
    round_to: int = config.NFL_DFS_ESTIMATED_SALARY_ROUND_TO,
) -> pd.Series:
    """Linear min-max scaling of `dk_points` into [floor, ceiling],
    clipped at both ends, rounded to the nearest `round_to`.
    `reference_max_points == reference_min_points` (a degenerate
    single-point reference range) maps everything to `floor` rather than
    dividing by zero - same edge-case handling as
    estimated_salary.compute_estimated_salary."""
    span = reference_max_points - reference_min_points
    if span <= 0:
        fraction = pd.Series(0.0, index=dk_points.index)
    else:
        fraction = ((dk_points - reference_min_points) / span).clip(0, 1)

    raw_salary = floor + fraction * (ceiling - floor)
    rounded = (raw_salary / round_to).round() * round_to
    return rounded.clip(lower=floor, upper=ceiling)


def compute_qb_estimated_salary(dk_points_qb: pd.Series) -> pd.Series:
    """Delegates to the SHARED reference range/ceiling (not a
    QB-specific one) - see module docstring."""
    return compute_estimated_salary(dk_points_qb, config.NFL_DFS_REFERENCE_MIN_POINTS, config.NFL_DFS_REFERENCE_MAX_POINTS)


def compute_skill_estimated_salary(dk_points_skill: pd.Series) -> pd.Series:
    """Delegates to the SAME shared reference range/ceiling
    compute_qb_estimated_salary uses - see module docstring."""
    return compute_estimated_salary(dk_points_skill, config.NFL_DFS_REFERENCE_MIN_POINTS, config.NFL_DFS_REFERENCE_MAX_POINTS)


def compute_dst_estimated_salary(dk_points_dst: pd.Series) -> pd.Series:
    """Delegates to the SAME shared reference range/ceiling - see module
    docstring. DST DK scoring rarely reaches the top of the shared range
    (a real DST's points-allowed-bucket ceiling caps its typical upside
    well below a strong QB week), so DSTs naturally price toward the
    lower end - a real, not artifactual, difference, same as MLB
    pitchers genuinely projecting more points than hitters."""
    return compute_estimated_salary(dk_points_dst, config.NFL_DFS_REFERENCE_MIN_POINTS, config.NFL_DFS_REFERENCE_MAX_POINTS)
