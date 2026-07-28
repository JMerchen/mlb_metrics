import pandas as pd
import pytest

from mlb_metrics import config, data, dfs_backtest, dfs_ml


@pytest.fixture(autouse=True)
def _no_network_name_register(monkeypatch):
    # chadwick_register() (data.get_name_register's underlying fetch) hits
    # a network URL this sandbox blocks - names aren't used in any numeric
    # computation pipeline.compute_outputs does, so an empty-but-correctly-
    # shaped register is a safe stand-in for every test in this file that
    # exercises the real pipeline (assemble_ml_training_rows).
    monkeypatch.setattr(
        data, "get_name_register",
        lambda: pd.DataFrame(columns=["key_mlbam", "key_bbref", "name_first", "name_last"]),
    )


def _statcast_game(game_pk, date, home_team, away_team, home_pitcher, away_pitcher, home_score, away_score):
    """A minimal 4-row single-game Statcast fixture - same shape
    test_game_picks_backtest.py's fixture uses, since
    derive_historical_team_schedule reuses derive_historical_schedule_games
    directly."""
    return [
        {"game_pk": game_pk, "game_date": date, "home_team": home_team, "away_team": away_team,
         "pitcher": home_pitcher, "inning_topbot": "Top", "at_bat_number": 1,
         "post_home_score": 0, "post_away_score": 0},
        {"game_pk": game_pk, "game_date": date, "home_team": home_team, "away_team": away_team,
         "pitcher": away_pitcher, "inning_topbot": "Bot", "at_bat_number": 2,
         "post_home_score": min(1, home_score), "post_away_score": 0},
        {"game_pk": game_pk, "game_date": date, "home_team": home_team, "away_team": away_team,
         "pitcher": home_pitcher, "inning_topbot": "Top", "at_bat_number": 3,
         "post_home_score": min(1, home_score), "post_away_score": min(1, away_score)},
        {"game_pk": game_pk, "game_date": date, "home_team": home_team, "away_team": away_team,
         "pitcher": away_pitcher, "inning_topbot": "Bot", "at_bat_number": 4,
         "post_home_score": home_score, "post_away_score": away_score},
    ]


def test_derive_historical_team_schedule_widens_to_one_row_per_team():
    rows = _statcast_game(100, pd.Timestamp("2026-06-01"), "NYY", "BOS", 101, 201, 5, 2)
    result = dfs_backtest.derive_historical_team_schedule(pd.DataFrame(rows))

    assert len(result) == 2
    nyy = result[result["team"] == "NYY"].iloc[0]
    bos = result[result["team"] == "BOS"].iloc[0]

    assert nyy["opponent"] == "BOS"
    assert nyy["is_home"] == True  # noqa: E712
    assert nyy["probable_pitcher_key_mlbam"] == 101
    assert bos["opponent"] == "NYY"
    assert bos["is_home"] == False  # noqa: E712
    assert bos["probable_pitcher_key_mlbam"] == 201


def _batter_events(batter, events, date="2026-06-01"):
    """`events` items are either a bare event string (bat_score/post_bat_score
    default to 0, i.e. no RBI) or an (event, bat_score, post_bat_score)
    3-tuple for a row that needs a real RBI delta."""
    rows = []
    for item in events:
        if isinstance(item, tuple):
            event, bat_score, post_bat_score = item
        else:
            event, bat_score, post_bat_score = item, 0, 0
        rows.append({
            "game_date": pd.Timestamp(date), "batter": batter, "events": event,
            "bat_score": bat_score, "post_bat_score": post_bat_score,
        })
    return rows


def test_compute_actual_hitter_dk_points_hit_type_plus_bb_hbp_rbi():
    rows = _batter_events(1, [
        "single", "double", "triple", "home_run",
        "walk", "intent_walk", "hit_by_pitch",
        ("field_out", 2, 3),  # scores a run -> 1 RBI, no hit-type/BB/HBP points
        "field_out",  # 0-0, no RBI - confirms field_out itself is never scored
    ])
    result = dfs_backtest.compute_actual_hitter_dk_points(pd.DataFrame(rows)).set_index("key_mlbam")

    expected = (
        config.DFS_DK_HITTER_SINGLE_POINTS + config.DFS_DK_HITTER_DOUBLE_POINTS
        + config.DFS_DK_HITTER_TRIPLE_POINTS + config.DFS_DK_HITTER_HR_POINTS
        + 2 * config.DFS_DK_HITTER_BB_POINTS  # walk + intent_walk
        + config.DFS_DK_HITTER_HBP_POINTS
        + 1 * config.DFS_DK_HITTER_RBI_POINTS
    )
    assert result.loc[1, "Actual_DK_Points_Modeled"] == pytest.approx(expected)


def test_compute_actual_hitter_dk_points_still_excludes_runs_and_steals():
    # No column here can carry a "run scored" or "stolen base" signal at
    # all - a field_out with no score change contributes exactly 0,
    # confirming those two categories are structurally impossible to
    # accidentally credit, not just zero-weighted.
    rows = _batter_events(1, ["field_out"])
    result = dfs_backtest.compute_actual_hitter_dk_points(pd.DataFrame(rows)).set_index("key_mlbam")
    assert result.loc[1, "Actual_DK_Points_Modeled"] == 0


def test_compute_actual_hitter_got_hit_is_max_not_sum():
    # Two hits in the date's game(s) still yields Got_Hit=1, not 2 - one
    # hit already clears the bar (compute_game_hit_probability's own
    # per-game label convention), and this also covers a double-header's
    # two games correctly pooling to the same date/label.
    rows = _batter_events(1, ["single", "home_run", "field_out"])
    result = dfs_backtest.compute_actual_hitter_got_hit(pd.DataFrame(rows)).set_index("key_mlbam")
    assert result.loc[1, "Got_Hit"] == 1


def test_compute_actual_hitter_got_hit_zero_hits():
    rows = _batter_events(1, ["field_out", "strikeout", "walk"])
    result = dfs_backtest.compute_actual_hitter_got_hit(pd.DataFrame(rows)).set_index("key_mlbam")
    assert result.loc[1, "Got_Hit"] == 0


def _pitcher_events(pitcher, events, date="2026-06-01"):
    return [{"game_date": pd.Timestamp(date), "pitcher": pitcher, "events": e} for e in events]


def test_compute_actual_pitcher_dk_points_exact_arithmetic():
    # 2 K (2 outs), 1 field_out (1 out) = 3 outs = 1 IP. 1 BB, 1 single, 1 HR.
    rows = _pitcher_events(99, ["strikeout", "strikeout", "field_out", "walk", "single", "home_run"])
    result = dfs_backtest.compute_actual_pitcher_dk_points(pd.DataFrame(rows)).set_index("key_mlbam")

    assert result.loc[99, "Actual_IP"] == pytest.approx(1.0)
    assert result.loc[99, "Actual_K"] == 2
    assert result.loc[99, "Actual_BB"] == 1
    assert result.loc[99, "Actual_H"] == 2  # single + home_run both count as hits

    fip = (13 * 1 + 3 * 1 - 2 * 2) / 1.0 + config.FIP_CONSTANT
    er = max(fip * 1.0 / 9, 0)
    expected_points = (
        1.0 * config.DFS_DK_PITCHER_IP_POINTS
        + 2 * config.DFS_DK_PITCHER_K_POINTS
        + 1 * config.DFS_DK_PITCHER_BB_POINTS
        + 2 * config.DFS_DK_PITCHER_H_POINTS
        + er * config.DFS_DK_PITCHER_ER_POINTS
    )
    assert result.loc[99, "Actual_DK_Points_Modeled"] == pytest.approx(expected_points)


def test_compute_actual_pitcher_dk_points_zero_ip_does_not_divide_by_zero():
    # 0 outs recorded, and no BB/H/K/HR either - isolates the zero-IP FIP
    # division from any other scoring component (a walk/hit allowed with 0
    # outs would correctly still score negative via the BB/H penalty terms,
    # which is not what this test is checking).
    rows = _pitcher_events(99, ["hit_by_pitch"])  # not modeled as a hit/BB/K, 0 outs
    result = dfs_backtest.compute_actual_pitcher_dk_points(pd.DataFrame(rows)).set_index("key_mlbam")

    assert result.loc[99, "Actual_IP"] == 0
    assert result.loc[99, "Actual_DK_Points_Modeled"] == 0


def test_ml_feature_columns_contain_no_leaked_outcome_column():
    # dfs_ml's feature schema must never include a same-day-outcome column
    # (anything Actual_*-prefixed) - assemble_ml_training_rows below relies
    # on this staying true to avoid training on the answer.
    assert not any(col.startswith("Actual_") for col in dfs_ml.HITTER_FEATURE_COLUMNS)
    assert not any(col.startswith("Actual_") for col in dfs_ml.PITCHER_FEATURE_COLUMNS)


def _game_rows(game_pk, date, events, pitcher=99, batter=1, home_team="NYY", away_team="BOS"):
    # teams.compute_home_run_stats needs both a "homer" and a "non_homer"
    # at-bat where the score actually changes (pre_score != post_score) -
    # both home_run and single bump away_score by 1 run here (a
    # simplification; other hit types don't drive in runs in this
    # fixture, which is fine for this test).
    rows = []
    away_runs = 0
    for i, e in enumerate(events):
        pre = away_runs
        if e in ("home_run", "single"):
            away_runs += 1
        rows.append({
            "game_pk": game_pk, "game_date": date, "pitcher": pitcher, "batter": batter,
            "events": e, "p_throws": "R", "inning_topbot": "Top",
            "home_team": home_team, "away_team": away_team,
            "at_bat_number": i + 1, "pitch_number": 1,
            "home_score": 0, "away_score": pre,
            "post_home_score": 0, "post_away_score": away_runs,
            # bat_score/post_bat_score are the BATTING team's own score
            # (helpers.estimate_rbi) - the away team bats on every "Top"
            # row in this fixture, so these mirror away_score/post_away_score.
            "bat_score": pre, "post_bat_score": away_runs,
        })
    return rows


def _multi_game_statcast(n_games=6, gap_days=5):
    # teams.compute_home_run_stats needs at least one "home_run" event to
    # appear so its homer/non_homer pivot has both columns.
    events = ["strikeout"] * 5 + ["field_out"] * 6 + ["walk"] * 3 + ["single"] * 4 + ["double"] * 1 + ["home_run"] * 1  # 20/game
    rows = []
    for i in range(n_games):
        date = pd.Timestamp("2026-05-01") + pd.Timedelta(days=i * gap_days)
        rows.extend(_game_rows(i + 1, date, events))
    return pd.DataFrame(rows)


def test_assemble_ml_training_rows_full_history_has_expected_schema_and_no_leakage(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _multi_game_statcast(n_games=6).to_parquet(raw_dir / "statcast_2026.parquet", index=False)

    result = dfs_backtest.assemble_ml_training_rows(str(raw_dir), season=2026, days=None)

    assert not result["hitters"].empty
    expected_hitter_cols = {"date", "key_mlbam", *dfs_ml.HITTER_FEATURE_COLUMNS, "Actual_DK_Points_Modeled"}
    assert set(result["hitters"].columns) == expected_hitter_cols

    assert not result["pitchers"].empty
    expected_pitcher_cols = {
        "date", "key_mlbam", *dfs_ml.PITCHER_FEATURE_COLUMNS,
        "Actual_IP", "Actual_K", "Actual_BB", "Actual_H",
    }
    assert set(result["pitchers"].columns) == expected_pitcher_cols

    # Every scored date is strictly after the earliest game date - the
    # first date can never be scored since it has no prior history at all.
    assert (result["hitters"]["date"] > pd.Timestamp("2026-05-01")).all()
    assert (result["pitchers"]["date"] > pd.Timestamp("2026-05-01")).all()


def test_assemble_ml_training_rows_days_truncates_to_most_recent_dates(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _multi_game_statcast(n_games=6).to_parquet(raw_dir / "statcast_2026.parquet", index=False)

    full = dfs_backtest.assemble_ml_training_rows(str(raw_dir), season=2026, days=None)
    truncated = dfs_backtest.assemble_ml_training_rows(str(raw_dir), season=2026, days=1)

    assert truncated["hitters"]["date"].nunique() <= 1
    assert full["hitters"]["date"].nunique() >= truncated["hitters"]["date"].nunique()


def test_assemble_ml_training_rows_no_persisted_data_returns_empty(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    result = dfs_backtest.assemble_ml_training_rows(str(raw_dir), season=2026, days=None)

    assert result["hitters"].empty
    assert result["pitchers"].empty


def test_assemble_hitter_hit_log_has_expected_schema_and_no_lookahead(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _multi_game_statcast(n_games=6).to_parquet(raw_dir / "statcast_2026.parquet", index=False)

    result = dfs_backtest.assemble_hitter_hit_log(str(raw_dir), season=2026, days=None)

    assert not result.empty
    expected_cols = {
        "date", "key_mlbam", "name_first", "name_last", "team",
        *dfs_ml.HITTER_FEATURE_COLUMNS, "Total_PA", "Got_Hit",
    }
    assert set(result.columns) == expected_cols

    # Every scored date is strictly after the earliest game date - the
    # first date can never be scored since it has no prior history at all
    # (same no-lookahead guarantee assemble_ml_training_rows's test checks).
    assert (result["date"] > pd.Timestamp("2026-05-01")).all()

    assert (result["Total_PA"] == result["PA_L"] + result["PA_R"]).all()
    assert result["Got_Hit"].isin([0, 1]).all()


def test_assemble_hitter_hit_log_no_leaked_outcome_in_feature_columns():
    # Got_Hit itself (and any Actual_*-prefixed column) must never appear
    # among the feature columns this log's rows are built from - guards
    # against a future refactor accidentally training on the answer.
    assert "Got_Hit" not in dfs_ml.HITTER_FEATURE_COLUMNS
    assert not any(col.startswith("Actual_") for col in dfs_ml.HITTER_FEATURE_COLUMNS)


def test_assemble_hitter_hit_log_days_truncates_to_most_recent_dates(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _multi_game_statcast(n_games=6).to_parquet(raw_dir / "statcast_2026.parquet", index=False)

    full = dfs_backtest.assemble_hitter_hit_log(str(raw_dir), season=2026, days=None)
    truncated = dfs_backtest.assemble_hitter_hit_log(str(raw_dir), season=2026, days=1)

    assert truncated["date"].nunique() <= 1
    assert full["date"].nunique() >= truncated["date"].nunique()


def test_assemble_hitter_hit_log_doubleheader_produces_one_row_per_hitter(tmp_path):
    # A real doubleheader (two game_pks, same team, same game_date) makes
    # derive_historical_team_schedule legitimately emit two schedule rows
    # for that team/date - build_hitter_features's team/key_mlbam-only
    # merges then fan those out into a same-key_mlbam cartesian duplicate.
    # assemble_hitter_hit_log must still emit exactly one row per
    # (date, key_mlbam) despite that.
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    rows = _multi_game_statcast(n_games=5).to_dict("records")
    doubleheader_date = pd.Timestamp("2026-05-01") + pd.Timedelta(days=5 * 5)
    events = ["strikeout"] * 5 + ["field_out"] * 6 + ["walk"] * 3 + ["single"] * 4 + ["double"] * 1 + ["home_run"] * 1
    rows.extend(_game_rows(101, doubleheader_date, events))
    rows.extend(_game_rows(102, doubleheader_date, events))
    pd.DataFrame(rows).to_parquet(raw_dir / "statcast_2026.parquet", index=False)

    result = dfs_backtest.assemble_hitter_hit_log(str(raw_dir), season=2026, days=None)

    doubleheader_rows = result[result["date"] == doubleheader_date]
    assert not doubleheader_rows.empty
    assert not doubleheader_rows.duplicated(subset="key_mlbam").any()


def test_assemble_hitter_hit_log_no_persisted_data_returns_empty(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    result = dfs_backtest.assemble_hitter_hit_log(str(raw_dir), season=2026, days=None)

    assert result.empty
