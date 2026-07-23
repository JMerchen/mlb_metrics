import math

import pandas as pd
import pytest

from mlb_metrics import age_curve


def _stint(player, year, stint, ab, h, doubles, triples, hr, bb, hbp, sf):
    return {
        "playerID": player, "yearID": year, "stint": stint, "AB": ab, "H": h,
        "2B": doubles, "3B": triples, "HR": hr, "BB": bb, "HBP": hbp, "SF": sf,
    }


def test_build_historical_hitter_seasons_aggregates_stints_and_computes_every_metric():
    # Player p1, year 2000: two stints (a mid-season trade).
    batting = pd.DataFrame([
        _stint("p1", 2000, 1, ab=200, h=60, doubles=10, triples=2, hr=5, bb=20, hbp=2, sf=3),
        _stint("p1", 2000, 2, ab=100, h=30, doubles=5, triples=0, hr=3, bb=10, hbp=1, sf=1),
    ])
    people = pd.DataFrame([{"playerID": "p1", "birthYear": 1975, "birthMonth": 1, "birthDay": 1}])

    result = age_curve.build_historical_hitter_seasons(batting, people, min_at_bats=0).set_index("playerID")

    # Summed across stints: AB=300, H=90, 2B=15, 3B=2, HR=8, BB=30, HBP=3, SF=4.
    tb = 90 + 15 + 2 * 2 + 3 * 8
    expected_avg = 90 / 300
    expected_slg = tb / 300
    expected_obp = (90 + 30 + 3) / (300 + 30 + 3 + 4)
    assert result.loc["p1", "AB"] == 300
    assert result.loc["p1", "AVG"] == pytest.approx(expected_avg)
    assert result.loc["p1", "OBP"] == pytest.approx(expected_obp)
    assert result.loc["p1", "SLG"] == pytest.approx(expected_slg)
    assert result.loc["p1", "OPS"] == pytest.approx(expected_slg + expected_obp)
    assert result.loc["p1", "age"] == 25  # born 1975-01-01, as of 2000-06-30


def test_build_historical_hitter_seasons_filters_by_min_at_bats():
    batting = pd.DataFrame([
        _stint("p1", 2000, 1, ab=500, h=150, doubles=0, triples=0, hr=0, bb=0, hbp=0, sf=0),
        _stint("p2", 2000, 1, ab=10, h=5, doubles=0, triples=0, hr=0, bb=0, hbp=0, sf=0),
    ])
    people = pd.DataFrame([
        {"playerID": "p1", "birthYear": 1975, "birthMonth": 1, "birthDay": 1},
        {"playerID": "p2", "birthYear": 1975, "birthMonth": 1, "birthDay": 1},
    ])

    result = age_curve.build_historical_hitter_seasons(batting, people, min_at_bats=200)

    assert set(result["playerID"]) == {"p1"}


def _pitcher_stint(player, year, stint, ipouts, so, bb, hbp, hr):
    return {
        "playerID": player, "yearID": year, "stint": stint, "IPouts": ipouts,
        "SO": so, "BB": bb, "HBP": hbp, "HR": hr,
    }


def test_build_historical_pitcher_seasons_aggregates_stints_and_computes_every_metric():
    # Player p1, year 2000: two stints (a mid-season trade).
    pitching = pd.DataFrame([
        _pitcher_stint("p1", 2000, 1, ipouts=200, so=60, bb=20, hbp=2, hr=5),
        _pitcher_stint("p1", 2000, 2, ipouts=100, so=30, bb=10, hbp=1, hr=3),
    ])
    people = pd.DataFrame([{"playerID": "p1", "birthYear": 1975, "birthMonth": 1, "birthDay": 1}])

    result = age_curve.build_historical_pitcher_seasons(pitching, people, min_ip=0).set_index("playerID")

    # Summed across stints: IPouts=300 (IP=100), SO=90, BB=30, HBP=3, HR=8.
    expected_ip = 300 / 3
    expected_k9 = 90 * 9 / expected_ip
    expected_bb9 = 30 * 9 / expected_ip
    expected_hr9 = 8 * 9 / expected_ip
    expected_fip = (13 * 8 + 3 * (30 + 3) - 2 * 90) / expected_ip + 3.10
    assert result.loc["p1", "IP"] == pytest.approx(expected_ip)
    assert result.loc["p1", "K9"] == pytest.approx(expected_k9)
    assert result.loc["p1", "BB9"] == pytest.approx(expected_bb9)
    assert result.loc["p1", "HR9"] == pytest.approx(expected_hr9)
    assert result.loc["p1", "FIP"] == pytest.approx(expected_fip)
    assert result.loc["p1", "age"] == 25  # born 1975-01-01, as of 2000-06-30


def test_build_historical_pitcher_seasons_filters_by_min_ip():
    pitching = pd.DataFrame([
        _pitcher_stint("p1", 2000, 1, ipouts=450, so=150, bb=50, hbp=5, hr=10),  # IP=150
        _pitcher_stint("p2", 2000, 1, ipouts=30, so=10, bb=5, hbp=0, hr=1),  # IP=10
    ])
    people = pd.DataFrame([
        {"playerID": "p1", "birthYear": 1975, "birthMonth": 1, "birthDay": 1},
        {"playerID": "p2", "birthYear": 1975, "birthMonth": 1, "birthDay": 1},
    ])

    result = age_curve.build_historical_pitcher_seasons(pitching, people, min_ip=50)

    assert set(result["playerID"]) == {"p1"}


def test_build_historical_pitcher_seasons_missing_hbp_treated_as_zero():
    pitching = pd.DataFrame([
        {"playerID": "p1", "yearID": 1900, "stint": 1, "IPouts": 300, "SO": 90, "BB": 30, "HR": 8}
    ])  # no HBP column at all, matching Lahman's earliest seasons
    people = pd.DataFrame([{"playerID": "p1", "birthYear": 1875, "birthMonth": 1, "birthDay": 1}])

    result = age_curve.build_historical_pitcher_seasons(pitching, people, min_ip=0).set_index("playerID")

    expected_fip = (13 * 8 + 3 * (30 + 0) - 2 * 90) / 100 + 3.10
    assert result.loc["p1", "FIP"] == pytest.approx(expected_fip)


def _season(player, year, age, avg, obp, slg, ab=400):
    return {"playerID": player, "yearID": year, "age": age, "AB": ab, "AVG": avg, "OBP": obp, "SLG": slg, "OPS": obp + slg}


def test_find_comparables_ranks_by_metric_distance_within_age_window():
    historical = pd.DataFrame([
        _season("a", 2000, 27, avg=0.300, obp=0.400, slg=0.400),  # OPS .800
        _season("b", 2001, 27, avg=0.300, obp=0.350, slg=0.400),  # OPS .750
        _season("c", 2002, 26, avg=0.300, obp=0.360, slg=0.400),  # OPS .760, within age_window=1
        _season("d", 2003, 29, avg=0.300, obp=0.360, slg=0.400),  # OPS .760, outside age_window=1
    ])

    result = age_curve.find_comparables(current_age=27, current_value=0.760, historical_seasons=historical, metric="OPS", k=2, age_window=1)

    # c: |.760-.760|=0, b: |.750-.760|=.01, a: |.800-.760|=.04 (excluded, k=2), d: age out of window.
    assert list(result["playerID"]) == ["c", "b"]


def test_find_comparables_respects_k_limit():
    historical = pd.DataFrame([_season(str(i), 2000, 27, avg=0.300, obp=0.350 + i * 0.001, slg=0.350) for i in range(10)])

    result = age_curve.find_comparables(current_age=27, current_value=0.700, historical_seasons=historical, metric="OPS", k=3, age_window=0)

    assert len(result) == 3


def test_find_comparables_different_metrics_pick_different_nearest_neighbors():
    # "a" is the current player's stand-in target: AVG=.300 but a much
    # lower OBP (.310, low walk rate) than "b" (AVG=.250, OBP=.400 - a
    # patient low-average hitter). Ranking by AVG should prefer "b" is
    # farther and something AVG-close should win; ranking by OBP should
    # flip which historical player is "nearest."
    historical = pd.DataFrame([
        _season("close_avg", 2000, 27, avg=0.298, obp=0.310, slg=0.400),   # nearest by AVG to .300
        _season("close_obp", 2001, 27, avg=0.250, obp=0.395, slg=0.400),   # nearest by OBP to .400
    ])

    by_avg = age_curve.find_comparables(current_age=27, current_value=0.300, historical_seasons=historical, metric="AVG", k=1, age_window=0)
    by_obp = age_curve.find_comparables(current_age=27, current_value=0.400, historical_seasons=historical, metric="OBP", k=1, age_window=0)

    assert by_avg.iloc[0]["playerID"] == "close_avg"
    assert by_obp.iloc[0]["playerID"] == "close_obp"


def test_project_next_season_looks_up_actual_following_year_and_averages():
    comparables = pd.DataFrame([_season("a", 2000, 27, 0.300, 0.400, 0.400), _season("b", 2001, 27, 0.280, 0.350, 0.400)])
    historical_seasons = pd.DataFrame([
        _season("a", 2001, 28, 0.310, 0.410, 0.410),  # a's actual next season, OPS = .820
        _season("b", 2002, 28, 0.260, 0.330, 0.370),  # b's actual next season, OPS = .700
    ])

    result = age_curve.project_next_season(comparables, historical_seasons, metric="OPS")

    assert result["n_comparables"] == 2
    assert result["n_with_next_season"] == 2
    assert result["projected_value_mean"] == pytest.approx((0.820 + 0.700) / 2)


def test_project_next_season_uses_the_given_metric_not_ops():
    comparables = pd.DataFrame([_season("a", 2000, 27, 0.300, 0.400, 0.400)])
    historical_seasons = pd.DataFrame([_season("a", 2001, 28, 0.320, 0.410, 0.410)])  # a's next-season AVG = .320

    result = age_curve.project_next_season(comparables, historical_seasons, metric="AVG")

    assert result["projected_value_mean"] == pytest.approx(0.320)


def test_project_next_season_excludes_comparables_with_no_next_season():
    comparables = pd.DataFrame([_season("a", 2000, 27, 0.300, 0.400, 0.400), _season("b", 2001, 27, 0.280, 0.350, 0.400)])
    # Only "a" has a resolvable next season - "b" retired/fell below the AB
    # threshold/etc, and must not be silently treated as continuing at its prior level.
    historical_seasons = pd.DataFrame([_season("a", 2001, 28, 0.310, 0.410, 0.410)])

    result = age_curve.project_next_season(comparables, historical_seasons, metric="OPS")

    assert result["n_comparables"] == 2
    assert result["n_with_next_season"] == 1
    assert result["projected_value_mean"] == pytest.approx(0.820)


def test_project_next_season_all_missing_returns_nan_not_zero():
    comparables = pd.DataFrame([_season("a", 2000, 27, 0.300, 0.400, 0.400)])
    historical_seasons = pd.DataFrame(columns=["playerID", "yearID", "age", "AB", "AVG", "OBP", "SLG", "OPS"])

    result = age_curve.project_next_season(comparables, historical_seasons, metric="OPS")

    assert result["n_with_next_season"] == 0
    assert math.isnan(result["projected_value_mean"])
    assert math.isnan(result["projected_value_p25"])


def test_backtest_projection_accuracy_scores_against_real_next_season():
    historical = pd.DataFrame([
        _season("a", 2000, 27, 0.300, 0.350, 0.350),  # test season - OPS .700, has a real next season below
        _season("a", 2001, 28, 0.320, 0.380, 0.370),  # a's actual next season, OPS = .750
        _season("b", 2000, 27, 0.300, 0.352, 0.353),  # a's nearest same-year comparable, OPS ~.705
        _season("b", 2001, 28, 0.280, 0.330, 0.320),  # b's actual next season, OPS = .650
    ])
    test_seasons = historical[(historical["playerID"] == "a") & (historical["yearID"] == 2000)]

    result = age_curve.backtest_projection_accuracy(historical, test_seasons, metric="OPS", k=1, age_window=0)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["actual_next_value"] == pytest.approx(0.750)
    # Only comparable available at/before yearID=2000 excluding "a" itself is "b" (k=1).
    assert row["projected_value_mean"] == pytest.approx(0.650)
    assert row["abs_error"] == pytest.approx(abs(0.650 - 0.750))


def test_backtest_projection_accuracy_excludes_future_seasons_from_the_pool():
    historical = pd.DataFrame([
        _season("a", 2000, 27, 0.300, 0.350, 0.350),  # test season, OPS .700
        _season("a", 2001, 28, 0.320, 0.380, 0.370),  # a's actual next season, OPS .750
        _season("b", 2005, 27, 0.300, 0.350, 0.350),  # exact OPS match, but AFTER the test season - must not be used
        _season("c", 1999, 27, 0.290, 0.345, 0.345),  # available before the test season - correct comparable
        _season("c", 2000, 28, 0.400, 0.450, 0.450),  # c's actual next season, OPS = .900
    ])
    test_seasons = historical[(historical["playerID"] == "a") & (historical["yearID"] == 2000)]

    result = age_curve.backtest_projection_accuracy(historical, test_seasons, metric="OPS", k=1, age_window=0)

    # If "b" (a perfect OPS match but from the future) leaked into the
    # pool, the projection would come from "b" instead - it has no
    # resolvable next season in this fixture, so the projection would be
    # NaN. Getting c's actual next season (.900) instead proves the
    # future row was correctly excluded.
    assert result.iloc[0]["projected_value_mean"] == pytest.approx(0.900)


def test_backtest_projection_accuracy_skips_unscorable_test_seasons():
    historical = pd.DataFrame([
        _season("a", 2000, 27, 0.300, 0.350, 0.350),  # no real 2001 row for "a" - career ended here
        _season("b", 2000, 27, 0.300, 0.350, 0.350),
        _season("b", 2001, 28, 0.320, 0.380, 0.370),
    ])
    test_seasons = historical[["playerID", "yearID", "age", "AVG", "OBP", "SLG", "OPS", "AB"]]

    result = age_curve.backtest_projection_accuracy(historical, test_seasons, metric="OPS", k=1, age_window=0)

    assert list(result["playerID"]) == ["b"]


def test_league_age_curve_averages_by_age_for_the_given_metric():
    historical = pd.DataFrame([
        _season("a", 2000, 27, 0.320, 0.400, 0.400),  # OPS .800
        _season("b", 2000, 27, 0.280, 0.350, 0.350),  # OPS .700
        _season("c", 2000, 30, 0.360, 0.450, 0.450),  # OPS .900
    ])

    ops_curve = age_curve.league_age_curve(historical, metric="OPS").set_index("age")
    avg_curve = age_curve.league_age_curve(historical, metric="AVG").set_index("age")

    assert ops_curve.loc[27, "value"] == pytest.approx(0.750)
    assert ops_curve.loc[27, "n_seasons"] == 2
    assert ops_curve.loc[30, "value"] == pytest.approx(0.900)
    assert avg_curve.loc[27, "value"] == pytest.approx((0.320 + 0.280) / 2)
