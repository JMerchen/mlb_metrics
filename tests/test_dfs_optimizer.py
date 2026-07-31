import pandas as pd
import pytest

from mlb_metrics import dfs_optimizer


def _hitter_row(key_mlbam, dk_points_hitter):
    return {
        "key_mlbam": key_mlbam, "name_first": "H", "name_last": str(key_mlbam), "team": "BOS", "opponent": "NYY",
        "is_home": True, "PA_L": 20, "PA_R": 20, "Expected_Bases": 1.5, "Game_Hit_Probability": 0.7,
        "Matchup_Hit_Probability": 0.7, "Matchup_Ratio": 1.0, "Adjusted_Expected_Bases": 1.5,
        "DK_Points_Hitter": dk_points_hitter,
    }


def _pitcher_row(key_mlbam, dk_points_pitcher):
    return {
        "key_mlbam": key_mlbam, "name_first": "P", "name_last": str(key_mlbam), "team": "NYY", "opponent": "BOS",
        "is_home": False, "starts": 5, "K9": 9.0, "BB9": 3.0, "HR9": 1.0, "IP_per_start": 6.0,
        "Expected_IP": 6.0, "Expected_K": 6.0, "Expected_BB": 2.0, "Expected_H_Allowed": 5.0,
        "FIP_Windowed": 4.0, "Expected_ER": 2.0, "DK_Points_Pitcher": dk_points_pitcher,
    }


def test_build_player_pool_joins_eligibility_and_computes_salary():
    hitters = pd.DataFrame([_hitter_row(1, 4.0)])
    pitchers = pd.DataFrame([_pitcher_row(99, 15.0)])
    eligibility = pd.DataFrame([{"key_mlbam": 1, "primary_position": "SS", "dk_slot": "SS"}])

    pool = dfs_optimizer.build_player_pool(hitters, pitchers, eligibility).set_index("key_mlbam")

    assert pool.loc[1, "dk_slot"] == "SS"
    assert pool.loc[1, "DK_Points"] == 4.0
    assert pool.loc[99, "dk_slot"] == "P"
    assert pool.loc[99, "DK_Points"] == 15.0
    assert list(pool.reset_index().columns) == dfs_optimizer.POOL_COLUMNS


def test_build_player_pool_carries_game_pk_and_game_datetime():
    hitters = pd.DataFrame([{**_hitter_row(1, 4.0), "game_pk": 824651, "game_datetime": "2026-07-31T18:20:00Z"}])
    pitchers = pd.DataFrame([{**_pitcher_row(99, 15.0), "game_pk": 824651, "game_datetime": "2026-07-31T18:20:00Z"}])
    eligibility = pd.DataFrame([{"key_mlbam": 1, "primary_position": "SS", "dk_slot": "SS"}])

    pool = dfs_optimizer.build_player_pool(hitters, pitchers, eligibility).set_index("key_mlbam")

    assert pool.loc[1, "game_pk"] == 824651
    assert pool.loc[1, "game_datetime"] == "2026-07-31T18:20:00Z"
    assert pool.loc[99, "game_pk"] == 824651
    assert bool(pool.loc[1, "is_home"]) is True
    assert bool(pool.loc[99, "is_home"]) is False


def test_build_player_pool_missing_game_pk_column_is_na_not_a_crash():
    # An older dfs_hitters.csv/dfs_pitchers.csv snapshot (or a hand-built
    # test frame) that predates game_pk/game_datetime must degrade to NA
    # on the POOL_COLUMNS selection, not KeyError.
    hitters = pd.DataFrame([_hitter_row(1, 4.0)])
    pitchers = pd.DataFrame([_pitcher_row(99, 15.0)])
    eligibility = pd.DataFrame([{"key_mlbam": 1, "primary_position": "SS", "dk_slot": "SS"}])

    pool = dfs_optimizer.build_player_pool(hitters, pitchers, eligibility).set_index("key_mlbam")

    assert pd.isna(pool.loc[1, "game_pk"])
    assert pd.isna(pool.loc[99, "game_datetime"])


def test_build_player_pool_ceiling_falls_back_to_dk_points_when_missing():
    # No Ceiling_DK_Points column at all in the input (an older CSV, or a
    # player with no real scored history yet) - must default to the mean
    # projection rather than leaving NaN, which would make the player
    # silently unselectable under a ceiling-objective solve.
    hitters = pd.DataFrame([_hitter_row(1, 4.0)])
    pitchers = pd.DataFrame([_pitcher_row(99, 15.0)])
    eligibility = pd.DataFrame([{"key_mlbam": 1, "primary_position": "SS", "dk_slot": "SS"}])

    pool = dfs_optimizer.build_player_pool(hitters, pitchers, eligibility).set_index("key_mlbam")

    assert pool.loc[1, "Ceiling_DK_Points"] == 4.0
    assert pool.loc[99, "Ceiling_DK_Points"] == 15.0


def test_build_player_pool_uses_real_ceiling_when_present():
    hitter = _hitter_row(1, 4.0)
    hitter["Ceiling_DK_Points"] = 9.5
    pitcher = _pitcher_row(99, 15.0)
    pitcher["Ceiling_DK_Points"] = 25.0
    hitters = pd.DataFrame([hitter])
    pitchers = pd.DataFrame([pitcher])
    eligibility = pd.DataFrame([{"key_mlbam": 1, "primary_position": "SS", "dk_slot": "SS"}])

    pool = dfs_optimizer.build_player_pool(hitters, pitchers, eligibility).set_index("key_mlbam")

    assert pool.loc[1, "Ceiling_DK_Points"] == 9.5
    assert pool.loc[99, "Ceiling_DK_Points"] == 25.0


def test_build_player_pool_boom_adjusted_falls_back_to_dk_points_when_missing():
    hitters = pd.DataFrame([_hitter_row(1, 4.0)])
    pitchers = pd.DataFrame([_pitcher_row(99, 15.0)])
    eligibility = pd.DataFrame([{"key_mlbam": 1, "primary_position": "SS", "dk_slot": "SS"}])

    pool = dfs_optimizer.build_player_pool(hitters, pitchers, eligibility).set_index("key_mlbam")

    assert pool.loc[1, "Boom_Adjusted_DK_Points"] == 4.0
    assert pool.loc[99, "Boom_Adjusted_DK_Points"] == 15.0


def test_build_player_pool_uses_real_boom_adjusted_when_present():
    hitter = _hitter_row(1, 4.0)
    hitter["Boom_Adjusted_DK_Points"] = 6.2
    pitcher = _pitcher_row(99, 15.0)
    pitcher["Boom_Adjusted_DK_Points"] = 18.0
    hitters = pd.DataFrame([hitter])
    pitchers = pd.DataFrame([pitcher])
    eligibility = pd.DataFrame([{"key_mlbam": 1, "primary_position": "SS", "dk_slot": "SS"}])

    pool = dfs_optimizer.build_player_pool(hitters, pitchers, eligibility).set_index("key_mlbam")

    assert pool.loc[1, "Boom_Adjusted_DK_Points"] == 6.2
    assert pool.loc[99, "Boom_Adjusted_DK_Points"] == 18.0


def test_build_player_pool_value_score_hitter_uses_boom_adjusted_over_salary():
    from mlb_metrics import estimated_salary

    hitter = _hitter_row(1, 4.0)
    hitter["Boom_Adjusted_DK_Points"] = 6.2
    hitters = pd.DataFrame([hitter])
    pitchers = pd.DataFrame([_pitcher_row(99, 15.0)])
    eligibility = pd.DataFrame([{"key_mlbam": 1, "primary_position": "SS", "dk_slot": "SS"}])

    pool = dfs_optimizer.build_player_pool(hitters, pitchers, eligibility).set_index("key_mlbam")

    expected_salary = estimated_salary.compute_hitter_estimated_salary(pd.Series([4.0])).iloc[0]
    expected_value = 6.2 / (expected_salary / 1000)
    assert pool.loc[1, "Value_Score"] == pytest.approx(expected_value)


def test_build_player_pool_value_score_pitcher_uses_upside_deviation_and_configured_k():
    from mlb_metrics import config, estimated_salary

    pitcher = _pitcher_row(99, 15.0)
    pitcher["Upside_Deviation"] = 3.0
    hitters = pd.DataFrame([_hitter_row(1, 4.0)])
    pitchers = pd.DataFrame([pitcher])
    eligibility = pd.DataFrame([{"key_mlbam": 1, "primary_position": "SS", "dk_slot": "SS"}])

    pool = dfs_optimizer.build_player_pool(hitters, pitchers, eligibility).set_index("key_mlbam")

    expected_salary = estimated_salary.compute_pitcher_estimated_salary(pd.Series([15.0])).iloc[0]
    expected_value_points = 15.0 + config.DFS_VALUE_BOOM_K_PITCHER * 3.0
    expected_value = expected_value_points / (expected_salary / 1000)
    assert pool.loc[99, "Value_Score"] == pytest.approx(expected_value)


def test_build_player_pool_value_score_pitcher_falls_back_to_zero_deviation_when_missing():
    # No Upside_Deviation column at all - must default to 0 (no boom
    # credit), not error or leave NaN.
    from mlb_metrics import estimated_salary

    hitters = pd.DataFrame([_hitter_row(1, 4.0)])
    pitchers = pd.DataFrame([_pitcher_row(99, 15.0)])
    eligibility = pd.DataFrame([{"key_mlbam": 1, "primary_position": "SS", "dk_slot": "SS"}])

    pool = dfs_optimizer.build_player_pool(hitters, pitchers, eligibility).set_index("key_mlbam")

    expected_salary = estimated_salary.compute_pitcher_estimated_salary(pd.Series([15.0])).iloc[0]
    expected_value = 15.0 / (expected_salary / 1000)
    assert pool.loc[99, "Value_Score"] == pytest.approx(expected_value)


def test_build_player_pool_excludes_hitter_with_no_eligibility():
    hitters = pd.DataFrame([_hitter_row(1, 4.0), _hitter_row(2, 3.0)])
    pitchers = pd.DataFrame(columns=list(_pitcher_row(0, 0).keys()))
    eligibility = pd.DataFrame([{"key_mlbam": 1, "primary_position": "SS", "dk_slot": "SS"}])  # no row for 2

    pool = dfs_optimizer.build_player_pool(hitters, pitchers, eligibility)

    assert set(pool["key_mlbam"]) == {1}


def _small_full_pool():
    """One cheap and one expensive-but-better candidate per slot, sized so
    the salary cap is actually binding - forces a real choice, not just
    "pick everyone's best player" (which would trivially pass even a buggy
    optimizer if the cap were never tight)."""
    rows = []
    key = 1
    for slot, count in [("P", 2), ("C", 1), ("1B", 1), ("2B", 1), ("3B", 1), ("SS", 1), ("OF", 3)]:
        for _ in range(count + 1):  # exactly one spare candidate per slot
            rows.append({
                "key_mlbam": key, "name_first": slot, "name_last": str(key), "team": "BOS", "opponent": "NYY",
                "dk_slot": slot, "DK_Points": 5.0, "Estimated_Salary": 4000,
            })
            key += 1
    return pd.DataFrame(rows)


def test_solve_optimal_lineup_respects_cap_and_slot_counts():
    pool = _small_full_pool()
    result = dfs_optimizer.solve_optimal_lineup(pool, salary_cap=50000)

    assert result is not None
    assert len(result) == 10
    assert result["Estimated_Salary"].sum() <= 50000
    counts = result["dk_slot"].value_counts().to_dict()
    assert counts == {"P": 2, "OF": 3, "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1}


def test_solve_optimal_lineup_picks_higher_points_under_binding_cap():
    # Two OF candidates share a slot; the higher-points one costs more but
    # the cap comfortably affords it - optimizer must pick it, not the
    # cheaper/worse one.
    pool = pd.DataFrame([
        {"key_mlbam": 1, "name_first": "A", "name_last": "A", "team": "BOS", "opponent": "NYY",
         "dk_slot": "OF", "DK_Points": 10.0, "Estimated_Salary": 5000},
        {"key_mlbam": 2, "name_first": "B", "name_last": "B", "team": "BOS", "opponent": "NYY",
         "dk_slot": "OF", "DK_Points": 3.0, "Estimated_Salary": 2000},
    ])
    result = dfs_optimizer.solve_optimal_lineup(pool, salary_cap=50000, roster_slots={"OF": 1})

    assert result["key_mlbam"].tolist() == [1]


def test_solve_optimal_lineup_infeasible_slot_returns_none():
    # Only 2 OF candidates but the slot needs 3 - can't be filled.
    pool = pd.DataFrame([
        {"key_mlbam": 1, "name_first": "A", "name_last": "A", "team": "BOS", "opponent": "NYY",
         "dk_slot": "OF", "DK_Points": 5.0, "Estimated_Salary": 3000},
        {"key_mlbam": 2, "name_first": "B", "name_last": "B", "team": "BOS", "opponent": "NYY",
         "dk_slot": "OF", "DK_Points": 5.0, "Estimated_Salary": 3000},
    ])
    result = dfs_optimizer.solve_optimal_lineup(pool, salary_cap=50000, roster_slots={"OF": 3})

    assert result is None


def test_solve_optimal_lineup_infeasible_cap_returns_none():
    pool = pd.DataFrame([
        {"key_mlbam": 1, "name_first": "A", "name_last": "A", "team": "BOS", "opponent": "NYY",
         "dk_slot": "OF", "DK_Points": 5.0, "Estimated_Salary": 60000},  # exceeds cap alone
    ])
    result = dfs_optimizer.solve_optimal_lineup(pool, salary_cap=50000, roster_slots={"OF": 1})

    assert result is None


def test_solve_optimal_lineup_empty_pool_returns_none():
    pool = pd.DataFrame(columns=dfs_optimizer.POOL_COLUMNS)
    result = dfs_optimizer.solve_optimal_lineup(pool, salary_cap=50000, roster_slots={"OF": 3})

    assert result is None


def test_solve_optimal_lineup_min_salary_forces_the_pricier_pick():
    # Same two-candidate OF slot as
    # test_solve_optimal_lineup_picks_higher_points_under_binding_cap, but
    # with DK_Points reversed so the objective alone would pick the
    # cheaper/worse one - only a min_salary floor should force the pricier
    # pick.
    pool = pd.DataFrame([
        {"key_mlbam": 1, "name_first": "A", "name_last": "A", "team": "BOS", "opponent": "NYY",
         "dk_slot": "OF", "DK_Points": 3.0, "Estimated_Salary": 5000},
        {"key_mlbam": 2, "name_first": "B", "name_last": "B", "team": "BOS", "opponent": "NYY",
         "dk_slot": "OF", "DK_Points": 10.0, "Estimated_Salary": 2000},
    ])
    unconstrained = dfs_optimizer.solve_optimal_lineup(pool, salary_cap=50000, roster_slots={"OF": 1})
    assert unconstrained["key_mlbam"].tolist() == [2]

    floored = dfs_optimizer.solve_optimal_lineup(pool, salary_cap=50000, roster_slots={"OF": 1}, min_salary=3000)
    assert floored["key_mlbam"].tolist() == [1]


def test_solve_optimal_lineup_min_salary_infeasible_returns_none():
    pool = pd.DataFrame([
        {"key_mlbam": 1, "name_first": "A", "name_last": "A", "team": "BOS", "opponent": "NYY",
         "dk_slot": "OF", "DK_Points": 5.0, "Estimated_Salary": 3000},
    ])
    result = dfs_optimizer.solve_optimal_lineup(
        pool, salary_cap=50000, roster_slots={"OF": 1}, min_salary=4000
    )

    assert result is None


def test_solve_optimal_lineup_two_way_player_selected_at_most_once():
    # Same key_mlbam appears twice (once as a hitter-eligible OF row, once
    # as the pitcher-eligible P row) - a real two-way player. The optimizer
    # must not select both rows (using one real person for two roster
    # slots at once).
    pool = pd.DataFrame([
        {"key_mlbam": 1, "name_first": "Two", "name_last": "Way", "team": "BOS", "opponent": "NYY",
         "dk_slot": "OF", "DK_Points": 20.0, "Estimated_Salary": 500},
        {"key_mlbam": 1, "name_first": "Two", "name_last": "Way", "team": "BOS", "opponent": "NYY",
         "dk_slot": "P", "DK_Points": 20.0, "Estimated_Salary": 500},
    ])
    result = dfs_optimizer.solve_optimal_lineup(pool, salary_cap=50000, roster_slots={"OF": 1, "P": 1})

    # Can't fill both slots from the same one real player - infeasible.
    assert result is None
