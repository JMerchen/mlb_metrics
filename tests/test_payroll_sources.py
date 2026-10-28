"""Tests for payroll_sources (The Baseball Cube via the Wayback Machine)
and scripts/fetch_payrolls.py."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from mlb_metrics import payroll_sources as ps

REPO_ROOT = Path(__file__).resolve().parents[1]
PAGE = (REPO_ROOT / "tests" / "fixtures" / "cube_payroll_2023.html").read_text()


def test_parses_every_team_with_its_franchise():
    t = ps.parse_cube_payrolls(PAGE, 2023)
    assert len(t) == 30 and t["franchise"].nunique() == 30
    nyy = t.set_index("franchise").loc["NYY"]
    assert (nyy["payroll"], nyy["wins"], nyy["losses"]) == (268954047, 82, 80)
    assert ps.parse_cube_payrolls(PAGE, 2024).empty


def test_expected_team_counts():
    assert [ps.expected_teams(y) for y in (1988, 1993, 1998, 2026)] == [26, 28, 30, 30]


class FakeResponse:
    def __init__(self, text="", status=200, data=None):
        self.text, self.status_code, self._data = text, status, data

    def json(self):
        return self._data


class FakeSession:
    """CDX lists two captures; the newer is broken (a partial table)."""

    def __init__(self, pages):
        self.headers, self.pages, self.calls = {}, pages, []

    def get(self, url, timeout=None, params=None):
        self.calls.append(url)
        if "cdx" in url:
            if params["url"].startswith("www."):
                return FakeResponse("x", data=[["timestamp", "original"], ["20251204", "https://www.thebaseballcube.com/content/payroll_year/2023/"],
                                               ["20240101", "https://www.thebaseballcube.com/content/payroll_year/2023/"]])
            return FakeResponse("")
        return FakeResponse(self.pages[url.split("/web/")[1].split("id_")[0]])


def test_fetch_takes_the_newest_complete_capture():
    broken = PAGE[:PAGE.index("New York Yankees")]
    session = FakeSession({"20251204": broken, "20240101": PAGE})
    table, snapshot = ps.fetch_season(2023, session)
    assert snapshot == "20240101" and len(table) == 30
    assert ps.fetch_season(2023, FakeSession({"20251204": broken, "20240101": broken})) == (None, None)


def test_apply_replaces_payrolls_and_adds_missing_teams():
    fetched = ps.parse_cube_payrolls(PAGE, 2023)
    payrolls = pd.DataFrame({"season": [2022, 2023, 2023], "franchise": ["NYY", "NYY", "BOS"],
                             "payroll": [1, 2, 3], "wins": [99, 82, 78], "losses": [63, 80, 84]})
    out = ps.apply_payrolls(payrolls, fetched, "20251025063344")
    nyy = out[(out["season"] == 2023) & (out["franchise"] == "NYY")].iloc[0]
    assert nyy["payroll"] == 268954047 and nyy["wins"] == 82
    assert nyy["payroll_source"] == "thebaseballcube.com via web.archive.org/web/20251025063344"
    assert out[out["season"] == 2022]["payroll"].tolist() == [1]
    assert pd.isna(out[out["season"] == 2022]["payroll_source"]).all()
    assert (out["season"] == 2023).sum() == 30
    added = out[(out["season"] == 2023) & (out["franchise"] == "LAD")].iloc[0]
    assert pd.isna(added["wins"]) and added["payroll"] == 217612634


def _load_script():
    path = REPO_ROOT / "scripts" / "fetch_payrolls.py"
    spec = importlib.util.spec_from_file_location("fetch_payrolls", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_updates_the_file_and_reports_changes(tmp_path, monkeypatch):
    module = _load_script()
    assert module.parse_seasons("1988-1990,2026") == [1988, 1989, 1990, 2026]
    path = tmp_path / "team_payroll.csv"
    pd.DataFrame({"season": [2022, 2023], "franchise": ["NYY", "NYY"], "payroll": [1, 2],
                  "wins": [99, 82], "losses": [63, 80]}).to_csv(path, index=False)
    fetched = ps.parse_cube_payrolls(PAGE, 2023)
    monkeypatch.setattr(module.payroll_sources, "fetch_season",
                        lambda season, capture=False: (fetched, "20251025063344") if season == 2023 else (None, None))
    monkeypatch.setattr("sys.argv", ["fetch", "--payrolls", str(path), "--seasons", "2022-2023", "--report", str(tmp_path / "r.csv")])
    module.main()
    out = pd.read_csv(path)
    assert len(out) == 31 and out.loc[out["season"] == 2022, "payroll"].item() == 1
    report = pd.read_csv(tmp_path / "r.csv")
    assert len(report) == 30
    assert report.set_index("franchise").loc["NYY", "change"] == 268954047 - 2


CSV_EXPORT = '''year,team,roster,league,division,team payroll,w,l,rank,lg rk,mlb rk,last year payroll,top salary
2026,Los Angeles Dodgers,team roster,NL,West,"415,212,461",100,62,1,2,2,"331,013,580",Kyle Tucker (1)
2026,Athletics,team roster,AL,West,"136,029,542",64,98,4,14,28,"74,036,474",Luis Severino (58)
2 record(s),,,,,,,,,,,,
'''


def test_parses_the_csv_export():
    t = ps.parse_cube_csv(CSV_EXPORT)
    assert t[["season", "franchise", "payroll", "wins", "losses"]].values.tolist() == [
        [2026, "LAD", 415212461, 100, 62], [2026, "ATH", 136029542, 64, 98]]


def test_script_imports_a_csv_export(tmp_path, monkeypatch):
    module = _load_script()
    table = ps.parse_cube_payrolls(PAGE, 2023)
    lines = ["year,team,roster,league,division,team payroll,w,l"] + [
        f'2023,{r.team},team roster,NL,East,"{r.payroll:,}",{r.wins},{r.losses}' for r in table.itertuples()]
    export = tmp_path / "payroll_2023.csv"
    export.write_text("\n".join(lines + ["30 record(s),,,,,,,"]) + "\n")
    path = tmp_path / "team_payroll.csv"
    pd.DataFrame({"season": [2023], "franchise": ["NYY"], "payroll": [2], "wins": [82], "losses": [80]}).to_csv(path, index=False)
    monkeypatch.setattr(module.payroll_sources, "fetch_season", lambda *a, **k: pytest.fail("no fetching with --csv"))
    monkeypatch.setattr("sys.argv", ["fetch", "--payrolls", str(path), "--csv", str(export)])
    module.main()
    out = pd.read_csv(path).set_index("franchise")
    assert len(out) == 30 and out.loc["NYY", "payroll"] == 268954047 and out.loc["NYY", "wins"] == 82
    assert out.loc["NYY", "payroll_source"] == "thebaseballcube.com CSV export (payroll_2023.csv)"
