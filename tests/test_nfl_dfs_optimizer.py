import pandas as pd
import pytest

from mlb_metrics import config, nfl_dfs_optimizer


def test_build_player_pool_assembles_qb_skill_dst_with_shared_key_mlbam_column():
    qb_dk = pd.DataFrame([{"player_id": "qb1", "DK_Points_QB": 20.0}])
    skill_dk = pd.DataFrame([{"player_id": "rb1", "DK_Points_Skill": 15.0}])
    dst_dk = pd.DataFrame([{"team": "SEA", "DK_Points_DST": 8.0}])
    eligibility = pd.DataFrame([{"player_id": "rb1", "position": "RB", "dk_slot": "RB"}])

    pool = nfl_dfs_optimizer.build_player_pool(qb_dk, skill_dk, dst_dk, eligibility)

    assert set(pool.columns) == set(nfl_dfs_optimizer.POOL_COLUMNS)
    qb_row = pool[pool["dk_slot"] == "QB"].iloc[0]
    assert qb_row["key_mlbam"] == "qb1"
    assert qb_row["DK_Points"] == 20.0

    dst_row = pool[pool["dk_slot"] == "DST"].iloc[0]
    assert dst_row["key_mlbam"] == "SEA"  # DST pool identity is the team code
    assert dst_row["DK_Points"] == 8.0


def test_build_player_pool_skill_player_gets_own_slot_and_flex_rows():
    qb_dk = pd.DataFrame(columns=["player_id", "DK_Points_QB"])
    skill_dk = pd.DataFrame([{"player_id": "rb1", "DK_Points_Skill": 15.0}])
    dst_dk = pd.DataFrame(columns=["team", "DK_Points_DST"])
    eligibility = pd.DataFrame(
        [
            {"player_id": "rb1", "position": "RB", "dk_slot": "RB"},
            {"player_id": "rb1", "position": "RB", "dk_slot": "FLEX"},
        ]
    )

    pool = nfl_dfs_optimizer.build_player_pool(qb_dk, skill_dk, dst_dk, eligibility)

    slots = set(pool[pool["key_mlbam"] == "rb1"]["dk_slot"])
    assert slots == {"RB", "FLEX"}


def _row(key, slot, points, salary=3000.0):
    return {"key_mlbam": key, "dk_slot": slot, "DK_Points": points, "Estimated_Salary": salary}


def test_solve_optimal_lineup_flex_picks_the_best_value_across_positions():
    # Real go/no-go check for FLEX: three strong "overflow" candidates
    # (rb_c, wr_d, te_b) each clearly beat every other option at their
    # own position, and the pool has more RB/WR/TE-eligible candidates
    # (9) than there are RB+WR+TE+FLEX slots to fill (7) - the true
    # optimum must include all three strong candidates SOMEWHERE (own
    # slot or FLEX, doesn't matter which), not just greedily fill each
    # position's minimum with locally-available options.
    pool = pd.DataFrame(
        [
            _row("qb1", "QB", 20.0),
            _row("dst1", "DST", 8.0),
            # RB: 3 candidates for 2 RB + possible FLEX.
            _row("rb_a", "RB", 1.0), _row("rb_a", "FLEX", 1.0),
            _row("rb_b", "RB", 1.0), _row("rb_b", "FLEX", 1.0),
            _row("rb_c", "RB", 15.0), _row("rb_c", "FLEX", 15.0),
            # WR: 4 candidates for 3 WR + possible FLEX.
            _row("wr_a", "WR", 1.0), _row("wr_a", "FLEX", 1.0),
            _row("wr_b", "WR", 1.0), _row("wr_b", "FLEX", 1.0),
            _row("wr_c", "WR", 1.0), _row("wr_c", "FLEX", 1.0),
            _row("wr_d", "WR", 12.0), _row("wr_d", "FLEX", 12.0),
            # TE: 2 candidates for 1 TE + possible FLEX.
            _row("te_a", "TE", 1.0), _row("te_a", "FLEX", 1.0),
            _row("te_b", "TE", 8.0), _row("te_b", "FLEX", 8.0),
        ]
    )

    result = nfl_dfs_optimizer.solve_optimal_lineup(pool, salary_cap=config.NFL_DFS_SALARY_CAP, roster_slots=config.NFL_DFS_ROSTER_SLOTS)

    assert result is not None
    assert len(result) == 9
    counts = result["dk_slot"].value_counts().to_dict()
    assert counts == config.NFL_DFS_ROSTER_SLOTS

    selected = set(result["key_mlbam"])
    assert {"qb1", "dst1", "rb_c", "wr_d", "te_b"} <= selected
    # Total points is deterministic even though which specific "weak"
    # filler candidates round out the roster is a tie (39 skill points +
    # 20 QB + 8 DST = 67).
    assert result["DK_Points"].sum() == pytest.approx(67.0)
