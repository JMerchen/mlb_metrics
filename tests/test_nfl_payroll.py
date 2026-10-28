"""Tests for nfl_payroll (the NFL page's Payroll tab) and
scripts/build_nfl_payroll_value.py."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlb_metrics import nfl_payroll as npay

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_team_codes():
    assert npay.team_code("Redskins") == "WAS" and npay.team_code("Washington") == "WAS"
    assert npay.team_code("Raiders") == "LV" and npay.team_code("OAK") == "LV" and npay.team_code("sd") == "LAC"
    assert len(npay.TEAMS) == 32
    with pytest.raises(ValueError):
        npay.team_code("Oilers")


def test_spending_counts_a_player_season_once_per_team():
    season = lambda year, team, cash, cap: {"year": str(year), "team": team, "cash_paid": cash, "cap_number": cap}  # noqa: E731
    history = [season(2023, "Chiefs", 10.0, 5.0), season(2024, "Chiefs", 1.0, 6.0)]
    contracts = pd.DataFrame({
        "otc_id": [1, 1, 2, 3],
        # Player 1's two contracts both carry his whole history.
        "season_history": [history, history, [season(2024, "Chiefs", 2.5, 2.0), season(2024, "Raiders", 0.5, 0.5)], None],
    })
    out = npay.team_spending(contracts).set_index(["season", "franchise"])
    assert out.loc[(2023, "KC"), "cash"] == 10e6
    assert out.loc[(2024, "KC"), "cash"] == 3.5e6 and out.loc[(2024, "KC"), "cap"] == 8e6
    assert out.loc[(2024, "KC"), "players"] == 2
    assert out.loc[(2024, "LV"), "cap"] == 0.5e6


def _schedule(rows):
    return pd.DataFrame(rows, columns=["season", "game_type", "home_team", "away_team", "home_score", "away_score"])


def test_records_count_ties_as_half_and_mark_final_seasons():
    sched = _schedule([
        [2024, "REG", "KC", "OAK", 27, 20], [2024, "REG", "OAK", "KC", 17, 17], [2024, "POST", "KC", "BUF", 30, 0],
        [2025, "REG", "KC", "LV", 21, 14], [2025, "REG", "LV", "KC", np.nan, np.nan],
    ])
    rec = npay.team_records(sched).set_index(["season", "franchise"])
    kc = rec.loc[(2024, "KC")]
    assert (kc["wins"], kc["losses"], kc["ties"]) == (1.5, 0.5, 1)
    assert rec.loc[(2024, "LV"), "losses"] == 1.5
    assert rec.loc[(2024, "KC"), "final"] and not rec.loc[(2025, "KC"), "final"]
    assert npay.last_season(sched) == 2025


def _table(seasons=(2014, 2015, 2016)):
    rows = []
    for k, season in enumerate(seasons):
        for i, team in enumerate(npay.TEAMS):
            w = 3 + (i % 11)
            rows.append({"season": season, "franchise": team, "cash": 1e8 * (1 + i / 10 + k / 5), "cap": 1.5e8 + 1e6 * i,
                         "players": 70, "wins": float(w), "losses": float(17 - w), "ties": 0, "final": True})
    return pd.DataFrame(rows)


def test_metrics_for_each_measure_compare_any_earlier_season():
    m = npay.season_metrics(_table())
    assert set(m["measure"]) == {"cash", "cap"} and len(m) == 2 * 3 * 32
    cash = m[m["measure"] == "cash"]
    assert cash.loc[cash["season"] == 2014, "expected_wins"].isna().all()
    later = cash[cash["season"] == 2015].iloc[0]
    assert later["neighbors"] and all(n.startswith("2014 ") for n in later["neighbors"].split(";"))
    # Cap spending barely varies, so relative payroll sits near 1.
    cap = m[m["measure"] == "cap"]
    assert cap["relative_payroll"].between(0.85, 1.15).all()
    franchises, league = npay.summaries(m)
    assert set(franchises["measure"]) == {"cash", "cap"} and "ties" in franchises
    assert league.loc[league["measure"] == "cash", "price_per_win_full_season"].notna().all()
    assert np.isfinite(npay.model_error_wins(m, "cash"))


def _load_script():
    path = REPO_ROOT / "scripts" / "build_nfl_payroll_value.py"
    spec = importlib.util.spec_from_file_location("build_nfl_payroll_value", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_writes_the_table_and_page_data(tmp_path, monkeypatch):
    module = _load_script()
    otc = {code: name for name, code in npay.OTC_TEAMS.items() if name not in ("Redskins", "Washington")}
    contracts = pd.DataFrame({"otc_id": range(32), "season_history": [
        [{"year": str(y), "team": otc[t], "cash_paid": 100 + i, "cap_number": 150 + i} for y in (2014, 2015)]
        for i, t in enumerate(npay.TEAMS)]})
    games = [[y, "REG", npay.TEAMS[i], npay.TEAMS[i + 1], 20 + i % 3, 17] for y in (2014, 2015) for i in range(0, 32, 2)]
    monkeypatch.setattr(module, "fetch_contracts", lambda: contracts)
    monkeypatch.setattr(module, "fetch_schedules", lambda seasons: _schedule(games))
    monkeypatch.setattr("sys.argv", ["build", "--table", str(tmp_path / "t.csv"), "--data-dir", str(tmp_path / "out")])
    module.main()
    table = pd.read_csv(tmp_path / "t.csv")
    assert len(table) == 64 and table["cash"].iloc[0] == 100e6
    seasons = pd.read_csv(tmp_path / "out" / "nfl_payroll_seasons.csv")
    assert set(seasons["measure"]) == {"cash", "cap"} and set(seasons["status"]) == {"final"}
    for name in ("nfl_payroll_franchises.csv", "nfl_payroll_league.csv"):
        assert (tmp_path / "out" / name).exists()
