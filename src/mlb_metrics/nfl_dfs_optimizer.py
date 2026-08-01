"""NFL DK Classic optimal-lineup solver - needs almost no new logic.
`dfs_optimizer.solve_optimal_lineup` is REUSED UNMODIFIED (imported
directly, not re-implemented) - its existing "cap any duplicated
`key_mlbam` group at <= 1 selected row" MILP constraint
(dfs_optimizer.py:199), originally built for the rare two-way-MLB-player
edge case, now handles FLEX eligibility for real, not just an edge case:
`nfl_roster_positions.build_eligibility_table` gives every RB/WR/TE TWO
pool rows (own slot + `"FLEX"`, same `key_mlbam`), so that same
constraint naturally caps them at one selected slot between the two.

Only `build_player_pool` is NFL-specific, since the INPUT shape differs -
three separate DK-points sources (QB/skill/DST) instead of MLB's hitter/
pitcher split, three separate Estimated_Salary functions
(nfl_estimated_salary.py), and DST is a TEAM-level unit with no
`player_id` at all - its pool rows use `team` as the shared `key_mlbam`
identity column instead (solve_optimal_lineup's dedupe constraint works
off whatever the caller puts under that column name; it never assumes
the value is actually an MLBAM id).
"""

import pandas as pd

from mlb_metrics import nfl_estimated_salary
from mlb_metrics.dfs_optimizer import solve_optimal_lineup  # noqa: F401 - re-exported for callers

POOL_COLUMNS = ["key_mlbam", "dk_slot", "DK_Points", "Estimated_Salary"]


def build_player_pool(
    qb_dk: pd.DataFrame, skill_dk: pd.DataFrame, dst_dk: pd.DataFrame, eligibility: pd.DataFrame
) -> pd.DataFrame:
    """One row per player/DST eligible for today's optimizer, in
    POOL_COLUMNS shape. `qb_dk`/`skill_dk`/`dst_dk` are
    nfl_dfs.compute_qb_dk_points/compute_skill_dk_points/
    nfl_dst.compute_dst_dk_points's own output. `eligibility` is
    nfl_roster_positions.build_eligibility_table's output - QBs need no
    eligibility lookup (no FLEX eligibility, mapped to dk_slot="QB"
    directly); only skill players are joined against it (inner join - a
    skill player with no resolvable DK slot is excluded, not defaulted,
    though in practice every real RB/WR/TE resolves). DST rows get
    dk_slot="DST" directly, with `team` copied into `key_mlbam` as their
    pool identity (see module docstring)."""
    qb = qb_dk.copy()
    qb["dk_slot"] = "QB"
    qb["DK_Points"] = qb["DK_Points_QB"]
    qb["Estimated_Salary"] = nfl_estimated_salary.compute_qb_estimated_salary(qb["DK_Points_QB"])
    qb = qb.rename(columns={"player_id": "key_mlbam"})

    skill = skill_dk.merge(eligibility[["player_id", "dk_slot"]], on="player_id", how="inner").copy()
    skill["DK_Points"] = skill["DK_Points_Skill"]
    skill["Estimated_Salary"] = nfl_estimated_salary.compute_skill_estimated_salary(skill["DK_Points_Skill"])
    skill = skill.rename(columns={"player_id": "key_mlbam"})

    dst = dst_dk.copy()
    dst["dk_slot"] = "DST"
    dst["DK_Points"] = dst["DK_Points_DST"]
    dst["Estimated_Salary"] = nfl_estimated_salary.compute_dst_estimated_salary(dst["DK_Points_DST"])
    dst["key_mlbam"] = dst["team"]

    return pd.concat([qb[POOL_COLUMNS], skill[POOL_COLUMNS], dst[POOL_COLUMNS]], ignore_index=True)
