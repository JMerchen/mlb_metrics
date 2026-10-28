"""Tests for payroll_value (the MLB Payroll page's numbers) and
scripts/build_payroll_value.py."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlb_metrics import payroll_value as pv

REPO_ROOT = Path(__file__).resolve().parents[1]


def _league(seasons, teams=("AAA", "BBB", "CCC"), wins=None):
    """Synthetic team-seasons: payroll 100/200/300 (x season index + 1),
    records 90-72 / 81-81 / 72-90 unless given."""
    rows = []
    for k, season in enumerate(seasons):
        for i, team in enumerate(teams):
            w = (wins or {}).get((season, team), [90, 81, 72][i % 3])
            rows.append({"season": season, "franchise": team, "payroll": 100.0 * (i + 1) * (k + 1), "wins": w, "losses": 162 - w})
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _three_neighbors(monkeypatch):
    monkeypatch.setattr(pv, "NEIGHBORS", 3)
    monkeypatch.setattr(pv, "_ALIAS", {**pv._ALIAS, "AAA": "AAA", "BBB": "BBB", "CCC": "CCC", "DDD": "DDD"})


def test_franchise_code_maps_every_old_name():
    assert pv.franchise_code("Montreal Expos") == "WSH"
    assert pv.franchise_code("\xa0OAK") == "ATH"
    assert pv.franchise_code("CHW") == "CWS"
    assert pv.franchise_code("Tampa Bay Devil Rays") == "TB"
    assert pv.franchise_code("arizona diamondbacks") == "AZ"
    with pytest.raises(ValueError):
        pv.franchise_code("Brooklyn Dodgers")


def test_value_metrics_against_the_league():
    s = pv.season_metrics(_league([2000]))
    a = s.set_index("franchise").loc["AAA"]
    # League: 600 payroll over 243 wins and 486 games.
    assert a["price_per_win_avg"] == pytest.approx((100 / 90) / (600 / 243))
    assert a["win_avg"] == pytest.approx(90 / 81)
    assert a["pe_index"] == pytest.approx(a["win_avg"] / a["price_per_win_avg"])
    # Effectiveness is Win/Average squared over relative payroll.
    assert a["pe_index"] == pytest.approx(a["win_avg"] ** 2 / a["relative_payroll"])
    assert a["relative_payroll"] == pytest.approx(0.5)
    assert list(s["payroll_rank"]) == [3, 2, 1]
    assert list(s["wins_rank"]) == [1, 2, 3]
    assert set(s["status"]) == {"final"}
    assert s["normalized_effectiveness"].min() == -1 and s["normalized_effectiveness"].max() == 1


def test_neighbors_are_other_franchises_eleven_or_more_seasons_earlier():
    seasons = [1990, 1995, 2001, 2002]
    s = pv.season_metrics(_league(seasons, teams=("AAA", "BBB", "CCC", "DDD")))
    row = s[(s["season"] == 2001) & (s["franchise"] == "AAA")].iloc[0]
    # Only 1990 is 11+ years back; AAA's own 1990 is excluded.
    assert sorted(n.split()[1] for n in row["neighbors"].split(";")) == ["BBB", "CCC", "DDD"]
    assert all(n.startswith("1990") for n in row["neighbors"].split(";"))
    pool = s[(s["season"] == 1990) & (s["franchise"] != "AAA")]
    assert row["baseline_win_avg"] == pytest.approx(pool["win_avg"].mean())
    # 1990 and 1995 have nothing that far back: no expected wins.
    early = s[s["season"] < 2001]
    assert early["expected_wins"].isna().all() and (early["neighbors"] == "").all()


def test_nearest_by_relative_payroll(monkeypatch):
    monkeypatch.setattr(pv, "NEIGHBORS", 1)
    s = pv.season_metrics(_league([1990, 2001], teams=("AAA", "BBB", "CCC", "DDD")))
    row = s[(s["season"] == 2001) & (s["franchise"] == "DDD")].iloc[0]
    # DDD spends 1.6x the average; CCC 1990 (1.2x) is the closest.
    assert row["neighbors"] == "1990 CCC"


def test_front_office_score_is_the_last_three_seasons_edge():
    seasons = [1979, 1980, 1991, 1992, 1993, 1994, 1995]
    wins = {(y, "AAA"): w for y, w in zip([1991, 1992, 1993, 1994], [95, 100, 90, 85])}
    s = pv.season_metrics(_league(seasons, wins=wins)).set_index(["franchise", "season"])
    a = s.loc["AAA"]
    assert a.loc[1991, "fo_score"] == 0  # no earlier season with a baseline
    assert a.loc[1992, "fo_score"] == pytest.approx(a.loc[1991, "year_fo_score"])
    expect = a.loc[[1992, 1993, 1994], "year_fo_score"].mean()
    assert a.loc[1995, "fo_score"] == pytest.approx(expect)
    assert a.loc[1995, "expected_win_avg"] == pytest.approx(a.loc[1995, "baseline_win_avg"] + expect)
    assert a.loc[1995, "expected_wins"] == pytest.approx(a.loc[1995, "expected_win_avg"] * 81)
    assert a.loc[1995, "wins_above_expected"] == pytest.approx(a.loc[1995, "wins"] - a.loc[1995, "expected_wins"])


def test_cash_flow_against_the_previous_three_seasons():
    df = _league([2000, 2001, 2002, 2003])
    df.loc[(df["season"] == 2003) & (df["franchise"] == "AAA"), "payroll"] = 4000.0
    s = pv.season_metrics(df).set_index(["franchise", "season"])
    a = s.loc["AAA"]
    assert np.isnan(a.loc[2000, "cash_flow"])
    habit = a.loc[[2000, 2001, 2002], "relative_payroll"].mean()
    assert a.loc[2003, "cash_flow"] == pytest.approx(a.loc[2003, "relative_payroll"] - habit)
    assert a.loc[2003, "cash_flow"] > 0


def test_a_season_without_records_is_a_forecast():
    df = _league([1980, 1981, 1995])
    df.loc[df["season"] == 1995, ["wins", "losses"]] = np.nan
    s = pv.season_metrics(df)
    f = s[s["season"] == 1995]
    assert set(f["status"]) == {"forecast"}
    assert (f["games"] == 162).all()
    assert f["relative_payroll"].tolist() == pytest.approx([0.5, 1.0, 1.5])
    assert f["expected_wins"].notna().all() and f["pe_index"].isna().all() and f["wins_rank"].isna().all()
    s = pv.season_metrics(_league([1980, 1995]), final_seasons=[1980])
    assert set(s.loc[s["season"] == 1995, "status"]) == {"in progress"}


def test_league_price_of_a_win_over_81_wins():
    df = _league([2019, 2020])
    df.loc[df["season"] == 2020, "wins"] = [33, 30, 27]  # a 60-game season
    df.loc[df["season"] == 2020, "losses"] = [27, 30, 33]
    league = pv.league_seasons(pv.season_metrics(df)).set_index("season")
    assert league.loc[2019, "price_per_81_wins"] == pytest.approx(league.loc[2019, "price_per_win"])
    assert league.loc[2020, "price_per_win"] == pytest.approx(1200 / 90)
    assert league.loc[2020, "price_per_81_wins"] == pytest.approx(400 / 81)


def test_franchise_summary_and_model_error():
    s = pv.season_metrics(_league([1980, 1981, 1992, 1993]))
    f = pv.franchise_summary(s).set_index("franchise")
    assert f.loc["AAA", "seasons"] == 4
    assert f.loc["AAA", "average_payroll_rank"] == 3
    assert f.loc["AAA", "normalized_payroll"] == 1 and f.loc["CCC", "normalized_payroll"] == -1
    assert f["average_effectiveness"].is_monotonic_decreasing
    final = s[s["expected_wins"].notna()]
    assert pv.model_error_wins(s) == pytest.approx((final["wins_above_expected"].abs() / final["games"]).mean() * 162)


def test_records_from_the_standings_and_the_game_log():
    raw = {"records": [{"teamRecords": [
        {"team": {"id": 147, "name": "New York Yankees"}, "wins": 94, "losses": 68},
        {"team": {"id": 133, "name": "Athletics"}, "wins": 70, "losses": 92},
        {"team": {"id": 999, "name": "Somebody"}, "wins": 1, "losses": 1},
    ]}]}
    rec = pv.standings_records(raw, 2026)
    assert rec.to_dict("records") == [
        {"season": 2026, "franchise": "NYY", "wins": 94, "losses": 68},
        {"season": 2026, "franchise": "ATH", "wins": 70, "losses": 92},
    ]
    games = pd.DataFrame({"team": ["ATH", "ATH", "ATH", "CHW"], "result": ["W", "L", "W", "L"]})
    log = pv.schedule_records(games, 2026).set_index("franchise")
    assert log.loc["ATH", "wins"] == 2 and log.loc["CWS", "losses"] == 1


def test_update_records_fills_only_matching_team_seasons():
    df = _league([2026])
    df[["wins", "losses"]] = np.nan
    rec = pd.DataFrame([{"season": 2026, "franchise": "AAA", "wins": 10, "losses": 5},
                        {"season": 2025, "franchise": "AAA", "wins": 99, "losses": 63}])
    out = pv.update_records(df, rec).set_index("franchise")
    assert out.loc["AAA", "wins"] == 10 and np.isnan(out.loc["BBB", "wins"])
    assert len(out) == 3


def test_committed_payrolls_cover_every_team_season():
    df = pv.load_payrolls(str(REPO_ROOT / "data" / "payroll" / "team_payroll.csv"))
    counts = df.groupby("season").size()
    assert counts.loc[1988] == 26 and counts.loc[1993] == 28 and counts.loc[1998] == 30
    assert (counts.loc[1998:] == 30).all()
    assert df["franchise"].nunique() == 30
    assert (df["payroll"] > 0).all()


def _load_script():
    path = REPO_ROOT / "scripts" / "build_payroll_value.py"
    spec = importlib.util.spec_from_file_location("build_payroll_value", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_inputs(tmp_path, schedule_rows):
    payrolls = _league([1980, 1981, 1995, 2026])
    payrolls.loc[payrolls["season"] == 2026, ["wins", "losses"]] = np.nan
    payrolls.loc[payrolls["season"] == 1995, ["wins", "losses"]] = np.nan
    payrolls["wins"] = payrolls["wins"].astype("Int64")
    payrolls["losses"] = payrolls["losses"].astype("Int64")
    payrolls.to_csv(tmp_path / "team_payroll.csv", index=False)
    (tmp_path / "dates.json").write_text(json.dumps(
        {"season": 2026, "regular_season_start": "2026-03-25", "regular_season_end": "2026-09-27"}))
    pd.DataFrame(schedule_rows, columns=["team", "date", "result"]).to_csv(tmp_path / "schedule.csv", index=False)


def _run(module, tmp_path, monkeypatch, today, standings):
    monkeypatch.setattr(module, "fetch_standings", standings)
    monkeypatch.setattr("sys.argv", ["build", "--payrolls", str(tmp_path / "team_payroll.csv"),
                                     "--season-dates", str(tmp_path / "dates.json"), "--schedule", str(tmp_path / "schedule.csv"),
                                     "--data-dir", str(tmp_path / "out"), "--today", today])
    module.main()
    return pd.read_csv(tmp_path / "team_payroll.csv"), pd.read_csv(tmp_path / "out" / "payroll_seasons.csv")


def test_build_fills_records_from_the_standings(tmp_path, monkeypatch):
    module = _load_script()
    _write_inputs(tmp_path, [])

    def standings(season):
        teams = ["AAA", "BBB", "CCC"] + [f"X{i}" for i in range(27)]
        return pd.DataFrame([{"season": season, "franchise": t, "wins": 80 + i % 3, "losses": 82 - i % 3}
                             for i, t in enumerate(teams)])

    payrolls, seasons = _run(module, tmp_path, monkeypatch, "2026-10-27", standings)
    assert payrolls.loc[payrolls["season"] == 1995, "wins"].notna().all()  # a blank past season filled
    assert set(seasons.loc[seasons["season"] == 2026, "status"]) == {"final"}
    for name in ("payroll_seasons.csv", "payroll_franchises.csv", "payroll_league.csv"):
        assert (tmp_path / "out" / name).exists()


def test_build_falls_back_to_the_game_log_without_losing_games(tmp_path, monkeypatch):
    module = _load_script()
    rows = [["AAA", "2026-05-01", "W"], ["AAA", "2026-05-02", "L"], ["BBB", "2026-05-01", "L"], ["AAA", "2025-09-01", "W"]]
    _write_inputs(tmp_path, rows)

    def down(season):
        raise ConnectionError("no network")

    payrolls, seasons = _run(module, tmp_path, monkeypatch, "2026-05-03", down)
    now = payrolls[payrolls["season"] == 2026].set_index("franchise")
    assert (now.loc["AAA", "wins"], now.loc["AAA", "losses"]) == (1, 1)
    assert set(seasons.loc[seasons["season"] == 2026, "status"]) >= {"in progress"}
    # A later run whose log is missing a game keeps the fuller record.
    payrolls.loc[(payrolls["season"] == 2026) & (payrolls["franchise"] == "AAA"), ["wins", "losses"]] = [2, 1]
    payrolls.to_csv(tmp_path / "team_payroll.csv", index=False)
    payrolls, _ = _run(module, tmp_path, monkeypatch, "2026-05-03", down)
    now = payrolls[payrolls["season"] == 2026].set_index("franchise")
    assert (now.loc["AAA", "wins"], now.loc["AAA", "losses"]) == (2, 1)
