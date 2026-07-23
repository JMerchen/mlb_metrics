import math

import pandas as pd
import pytest

from mlb_metrics import age_curve


def _stint(player, year, stint, ab, h, doubles, triples, hr, bb, hbp, sf):
    return {
        "playerID": player, "yearID": year, "stint": stint, "AB": ab, "H": h,
        "2B": doubles, "3B": triples, "HR": hr, "BB": bb, "HBP": hbp, "SF": sf,
    }


def test_build_historical_seasons_aggregates_stints_and_computes_ops():
    # Player p1, year 2000: two stints (a mid-season trade).
    batting = pd.DataFrame([
        _stint("p1", 2000, 1, ab=200, h=60, doubles=10, triples=2, hr=5, bb=20, hbp=2, sf=3),
        _stint("p1", 2000, 2, ab=100, h=30, doubles=5, triples=0, hr=3, bb=10, hbp=1, sf=1),
    ])
    people = pd.DataFrame([{"playerID": "p1", "birthYear": 1975, "birthMonth": 1, "birthDay": 1}])

    result = age_curve.build_historical_seasons(batting, people, min_at_bats=0).set_index("playerID")

    # Summed across stints: AB=300, H=90, 2B=15, 3B=2, HR=8, BB=30, HBP=3, SF=4.
    tb = 90 + 15 + 2 * 2 + 3 * 8
    expected_slg = tb / 300
    expected_obp = (90 + 30 + 3) / (300 + 30 + 3 + 4)
    assert result.loc["p1", "AB"] == 300
    assert result.loc["p1", "OPS"] == pytest.approx(expected_slg + expected_obp)
    assert result.loc["p1", "age"] == 25  # born 1975-01-01, as of 2000-06-30


def test_build_historical_seasons_filters_by_min_at_bats():
    batting = pd.DataFrame([
        _stint("p1", 2000, 1, ab=500, h=150, doubles=0, triples=0, hr=0, bb=0, hbp=0, sf=0),
        _stint("p2", 2000, 1, ab=10, h=5, doubles=0, triples=0, hr=0, bb=0, hbp=0, sf=0),
    ])
    people = pd.DataFrame([
        {"playerID": "p1", "birthYear": 1975, "birthMonth": 1, "birthDay": 1},
        {"playerID": "p2", "birthYear": 1975, "birthMonth": 1, "birthDay": 1},
    ])

    result = age_curve.build_historical_seasons(batting, people, min_at_bats=200)

    assert set(result["playerID"]) == {"p1"}


def _season(player, year, age, ops, ab=400):
    return {"playerID": player, "yearID": year, "age": age, "AB": ab, "OPS": ops}


def test_find_comparables_ranks_by_ops_distance_within_age_window():
    historical = pd.DataFrame([
        _season("a", 2000, 27, 0.800),
        _season("b", 2001, 27, 0.750),
        _season("c", 2002, 26, 0.760),  # within age_window=1 of 27
        _season("d", 2003, 29, 0.760),  # outside age_window=1 of 27
    ])

    result = age_curve.find_comparables(current_age=27, current_ops=0.760, historical_seasons=historical, k=2, age_window=1)

    # c: |.760-.760|=0, b: |.750-.760|=.01, a: |.800-.760|=.04 (excluded, k=2), d: age out of window.
    assert list(result["playerID"]) == ["c", "b"]


def test_find_comparables_respects_k_limit():
    historical = pd.DataFrame([_season(str(i), 2000, 27, 0.700 + i * 0.001) for i in range(10)])

    result = age_curve.find_comparables(current_age=27, current_ops=0.700, historical_seasons=historical, k=3, age_window=0)

    assert len(result) == 3


def test_project_next_season_looks_up_actual_following_year_and_averages():
    comparables = pd.DataFrame([_season("a", 2000, 27, 0.800), _season("b", 2001, 27, 0.750)])
    historical_seasons = pd.DataFrame([
        _season("a", 2001, 28, 0.820),  # a's actual next season
        _season("b", 2002, 28, 0.700),  # b's actual next season
    ])

    result = age_curve.project_next_season(comparables, historical_seasons)

    assert result["n_comparables"] == 2
    assert result["n_with_next_season"] == 2
    assert result["projected_ops_mean"] == pytest.approx((0.820 + 0.700) / 2)


def test_project_next_season_excludes_comparables_with_no_next_season():
    comparables = pd.DataFrame([_season("a", 2000, 27, 0.800), _season("b", 2001, 27, 0.750)])
    # Only "a" has a resolvable next season - "b" retired/fell below the AB
    # threshold/etc, and must not be silently treated as continuing at 0.750.
    historical_seasons = pd.DataFrame([_season("a", 2001, 28, 0.820)])

    result = age_curve.project_next_season(comparables, historical_seasons)

    assert result["n_comparables"] == 2
    assert result["n_with_next_season"] == 1
    assert result["projected_ops_mean"] == pytest.approx(0.820)


def test_project_next_season_all_missing_returns_nan_not_zero():
    comparables = pd.DataFrame([_season("a", 2000, 27, 0.800)])
    historical_seasons = pd.DataFrame(columns=["playerID", "yearID", "age", "AB", "OPS"])

    result = age_curve.project_next_season(comparables, historical_seasons)

    assert result["n_with_next_season"] == 0
    assert math.isnan(result["projected_ops_mean"])
    assert math.isnan(result["projected_ops_p25"])


def test_backtest_projection_accuracy_scores_against_real_next_season():
    historical = pd.DataFrame([
        _season("a", 2000, 27, 0.700),  # test season - has a real next season below
        _season("a", 2001, 28, 0.750),  # a's actual next season
        _season("b", 2000, 27, 0.705),  # a's nearest same-year comparable
        _season("b", 2001, 28, 0.650),  # b's actual next season
    ])
    test_seasons = historical[(historical["playerID"] == "a") & (historical["yearID"] == 2000)]

    result = age_curve.backtest_projection_accuracy(historical, test_seasons, k=1, age_window=0)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["actual_next_OPS"] == pytest.approx(0.750)
    # Only comparable available at/before yearID=2000 excluding "a" itself is "b" (k=1).
    assert row["projected_ops_mean"] == pytest.approx(0.650)
    assert row["abs_error"] == pytest.approx(abs(0.650 - 0.750))


def test_backtest_projection_accuracy_excludes_future_seasons_from_the_pool():
    historical = pd.DataFrame([
        _season("a", 2000, 27, 0.700),  # test season
        _season("a", 2001, 28, 0.750),  # a's actual next season
        _season("b", 2005, 27, 0.700),  # exact OPS match, but AFTER the test season - must not be used
        _season("c", 1999, 27, 0.690),  # available before the test season - correct comparable
        _season("c", 2000, 28, 0.900),  # c's actual next season
    ])
    test_seasons = historical[(historical["playerID"] == "a") & (historical["yearID"] == 2000)]

    result = age_curve.backtest_projection_accuracy(historical, test_seasons, k=1, age_window=0)

    # If "b" (a perfect OPS match but from the future) leaked into the
    # pool, the projection would come from "b" instead - it has no
    # resolvable next season in this fixture, so the projection would be
    # NaN. Getting c's actual next season (0.900) instead proves the
    # future row was correctly excluded.
    assert result.iloc[0]["projected_ops_mean"] == pytest.approx(0.900)


def test_backtest_projection_accuracy_skips_unscorable_test_seasons():
    historical = pd.DataFrame([
        _season("a", 2000, 27, 0.700),  # no real 2001 row for "a" - career ended here
        _season("b", 2000, 27, 0.700),
        _season("b", 2001, 28, 0.750),
    ])
    test_seasons = historical[["playerID", "yearID", "age", "OPS", "AB"]]

    result = age_curve.backtest_projection_accuracy(historical, test_seasons, k=1, age_window=0)

    assert list(result["playerID"]) == ["b"]


def test_league_age_curve_averages_by_age():
    historical = pd.DataFrame([
        _season("a", 2000, 27, 0.800),
        _season("b", 2000, 27, 0.700),
        _season("c", 2000, 30, 0.900),
    ])

    curve = age_curve.league_age_curve(historical).set_index("age")

    assert curve.loc[27, "OPS"] == pytest.approx(0.750)
    assert curve.loc[27, "n_seasons"] == 2
    assert curve.loc[30, "OPS"] == pytest.approx(0.900)
