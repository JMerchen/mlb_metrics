import pandas as pd

from mlb_metrics import nfl_data


def test_table_path_includes_table_and_season():
    path = nfl_data.table_path("data/raw/nfl", "weekly", 2025)
    assert path == "data/raw/nfl/weekly_2025.parquet"


def test_load_persisted_table_returns_none_when_absent(tmp_path):
    assert nfl_data.load_persisted_table(str(tmp_path), "weekly", 2025) is None


def test_persist_and_load_table_round_trips(tmp_path):
    df = pd.DataFrame([{"player_id": "00-0023459", "season": 2025, "week": 1}])
    raw_dir = str(tmp_path / "raw")

    nfl_data.persist_table(df, raw_dir, "weekly", 2025)
    loaded = nfl_data.load_persisted_table(raw_dir, "weekly", 2025)

    assert loaded is not None
    assert loaded.iloc[0]["player_id"] == "00-0023459"


def test_persist_table_overwrites_not_appends(tmp_path):
    # Real behavior this project depends on: nflverse retroactively
    # corrects published stats, so a fresh persist must fully replace the
    # prior file, not merge into it (unlike data.py's Statcast pattern) -
    # see nfl_data.py's module docstring for the full reasoning.
    raw_dir = str(tmp_path / "raw")
    first = pd.DataFrame([{"player_id": "a", "passing_yards": 200}, {"player_id": "b", "passing_yards": 100}])
    nfl_data.persist_table(first, raw_dir, "weekly", 2025)

    corrected = pd.DataFrame([{"player_id": "a", "passing_yards": 250}])
    nfl_data.persist_table(corrected, raw_dir, "weekly", 2025)

    loaded = nfl_data.load_persisted_table(raw_dir, "weekly", 2025)
    assert len(loaded) == 1
    assert loaded.iloc[0]["passing_yards"] == 250


def _roster_rows():
    def row(gsis, name, team, week, position="QB", status="ACT"):
        return {"season": 2026, "week": week, "team": team, "gsis_id": gsis, "full_name": name, "position": position,
                "status": status, "pfr_id": gsis.upper()}
    return pd.DataFrame([
        row("cousins", "Kirk Cousins", "LV", 9), row("cousins", "Kirk Cousins", "LV", 10),     # traded, old team plays this week
        row("henley", "Daiyan Henley", "LAC", 9, "LB"),                                         # traded, no row this week
        row("wentz", "Carson Wentz", "MIN", 9), row("wentz", "Carson Wentz", "MIN", 10),       # stays
        row("allen", "Josh Allen", "BUF", 10),                                                  # not on the depth chart
    ])


def _chart(rows, dt="2026-11-11T07:02:21Z"):
    return pd.DataFrame([{"dt": dt, "team": team, "gsis_id": gsis, "player_name": gsis, "pos_abb": "QB", "pos_rank": 1}
                         for gsis, team in rows])


def test_latest_depth_chart_keeps_the_newest_snapshot():
    charts = pd.concat([_chart([("cousins", "LV")], dt="2026-11-09T07:00:00Z"),
                        _chart([("cousins", "MIN"), (None, "MIN")], dt="2026-11-11T07:02:21Z")], ignore_index=True)
    latest = nfl_data.latest_depth_chart(charts)
    assert list(latest.columns) == nfl_data.DEPTH_CHART_COLUMNS
    assert latest[["team", "gsis_id"]].values.tolist() == [["MIN", "cousins"]]
    assert nfl_data.latest_depth_chart(pd.DataFrame()).empty


def test_apply_depth_chart_teams_moves_a_traded_player_for_the_current_week():
    chart = _chart([("cousins", "MIN"), ("henley", "DEN"), ("wentz", "MIN"), ("rookie", "MIN")])
    rosters, moves = nfl_data.apply_depth_chart_teams(_roster_rows(), chart, 2026, 10, now="2026-11-11T15:00:00Z")
    assert sorted(moves[["full_name", "from_team", "to_team"]].values.tolist()) == [
        ["Daiyan Henley", "LAC", "DEN"], ["Kirk Cousins", "LV", "MIN"]]
    week = rosters[rosters["week"] == 10].set_index("gsis_id")
    assert week.at["cousins", "team"] == "MIN" and week.at["wentz", "team"] == "MIN" and week.at["allen", "team"] == "BUF"
    # His old team is off this week, so he had no row: his latest one is carried into the week on the new team.
    assert week.at["henley", "team"] == "DEN" and week.at["henley", "position"] == "LB" and week.at["henley", "status"] == "ACT"
    assert "rookie" not in week.index                      # someone the roster file has never listed isn't invented
    earlier = rosters[rosters["week"] == 9].set_index("gsis_id")
    assert earlier.at["cousins", "team"] == "LV" and earlier.at["henley", "team"] == "LAC"   # history is left alone
    assert len(rosters) == len(_roster_rows()) + 1 and list(rosters.columns) == list(_roster_rows().columns)


def test_apply_depth_chart_teams_leaves_rosters_alone_when_it_shouldnt_be_trusted(capsys):
    rosters = _roster_rows()
    chart = _chart([("cousins", "MIN")])
    stale, moves = nfl_data.apply_depth_chart_teams(rosters, chart, 2026, 10, now="2026-11-20T00:00:00Z")
    assert stale.equals(rosters) and moves.empty                                # a snapshot more than a few days old
    same, moves = nfl_data.apply_depth_chart_teams(rosters, _chart([("cousins", "LV"), ("wentz", "MIN")]), 2026, 10,
                                                   now="2026-11-11T15:00:00Z")
    assert same.equals(rosters) and moves.empty                                 # nothing differs
    assert nfl_data.apply_depth_chart_teams(rosters, pd.DataFrame(), 2026, 10)[1].empty
    # Dozens of "moves" at once is two files disagreeing, not trades.
    many = pd.DataFrame([{"season": 2026, "week": 10, "team": "LV", "gsis_id": f"p{i}", "full_name": f"P {i}",
                          "position": "WR", "status": "ACT", "pfr_id": f"P{i}"} for i in range(nfl_data.DEPTH_CHART_MAX_MOVES + 1)])
    untouched, moves = nfl_data.apply_depth_chart_teams(many, _chart([(f"p{i}", "MIN") for i in range(len(many))]), 2026, 10,
                                                        now="2026-11-11T15:00:00Z")
    assert untouched.equals(many) and moves.empty and "too many" in capsys.readouterr().out
