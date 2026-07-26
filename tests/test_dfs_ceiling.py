import pandas as pd
import pytest

from mlb_metrics import config, data, dfs_ceiling


@pytest.fixture(autouse=True)
def _no_network_name_register(monkeypatch):
    # Same rationale as test_dfs_backtest.py's identical fixture:
    # chadwick_register() hits a network URL this sandbox blocks, and names
    # aren't used in any numeric computation backtest_ceiling_signal's
    # underlying pipeline.compute_outputs call needs.
    monkeypatch.setattr(
        data, "get_name_register",
        lambda: pd.DataFrame(columns=["key_mlbam", "key_bbref", "name_first", "name_last"]),
    )


def _row(date, batter=1, pitcher=99, events="single", bat_score=0, post_bat_score=0):
    return {
        "game_date": pd.Timestamp(date), "batter": batter, "pitcher": pitcher,
        "events": events, "bat_score": bat_score, "post_bat_score": post_bat_score,
    }


def test_compute_player_dk_points_history_scores_each_real_date_separately():
    rows = [
        _row("2026-06-01", events="single"),   # 3 pts
        _row("2026-06-05", events="home_run"),  # 10 pts
    ]
    persisted = pd.DataFrame(rows)

    history = dfs_ceiling.compute_player_dk_points_history(persisted)
    hitters = history["hitters"].sort_values("game_date").reset_index(drop=True)

    assert len(hitters) == 2
    assert hitters.loc[0, "Actual_DK_Points_Modeled"] == pytest.approx(config.DFS_DK_HITTER_SINGLE_POINTS)
    assert hitters.loc[1, "Actual_DK_Points_Modeled"] == pytest.approx(config.DFS_DK_HITTER_HR_POINTS)


def test_compute_player_dk_points_history_as_of_date_excludes_future():
    rows = [
        _row("2026-06-01", events="single"),
        _row("2026-06-05", events="home_run"),
    ]
    persisted = pd.DataFrame(rows)

    history = dfs_ceiling.compute_player_dk_points_history(persisted, as_of_date=pd.Timestamp("2026-06-05"))
    hitters = history["hitters"]

    assert len(hitters) == 1
    assert hitters.iloc[0]["Actual_DK_Points_Modeled"] == pytest.approx(config.DFS_DK_HITTER_SINGLE_POINTS)


def test_compute_player_dk_points_history_empty_input_returns_empty_frames():
    history = dfs_ceiling.compute_player_dk_points_history(pd.DataFrame(columns=["game_date", "batter", "pitcher", "events", "bat_score", "post_bat_score"]))
    assert history["hitters"].empty
    assert history["pitchers"].empty


def test_compute_ceiling_percentiles_exact_arithmetic_for_a_qualified_player():
    values = list(range(1, 21))  # 1..20, enough games to clear min_games
    history = pd.DataFrame({
        "key_mlbam": [1] * len(values),
        "Actual_DK_Points_Modeled": values,
    })

    result = dfs_ceiling.compute_ceiling_percentiles(history, percentile=90, min_games=10).set_index("key_mlbam")

    expected = pd.Series(values).quantile(0.90)
    assert result.loc[1, "Ceiling_DK_Points"] == pytest.approx(expected)
    assert result.loc[1, "n_games"] == 20
    assert result.loc[1, "Ceiling_Source"] == "player"


def test_compute_ceiling_percentiles_small_sample_falls_back_to_group_wide():
    # Player 1 has plenty of games; player 2 has only 2 (below min_games=10)
    # and must fall back to the GROUP-WIDE percentile across both players'
    # pooled history, not their own 2-game percentile.
    history = pd.DataFrame({
        "key_mlbam": [1] * 15 + [2, 2],
        "Actual_DK_Points_Modeled": list(range(1, 16)) + [100, 200],
    })

    result = dfs_ceiling.compute_ceiling_percentiles(history, percentile=90, min_games=10).set_index("key_mlbam")

    expected_group_ceiling = history["Actual_DK_Points_Modeled"].quantile(0.90)
    assert result.loc[2, "n_games"] == 2
    assert result.loc[2, "Ceiling_Source"] == "group_fallback"
    assert result.loc[2, "Ceiling_DK_Points"] == pytest.approx(expected_group_ceiling)
    assert result.loc[1, "Ceiling_Source"] == "player"


def test_compute_ceiling_percentiles_coerces_object_dtype_points_column():
    # compute_actual_pitcher_dk_points's real output can leave
    # Actual_DK_Points_Modeled as an object-dtype column (pd.NA upcasting
    # inside its FIP-safe-division step) - groupby().quantile() would raise
    # TypeError on that dtype without the defensive coercion.
    history = pd.DataFrame({
        "key_mlbam": [1] * 15,
        "Actual_DK_Points_Modeled": pd.array(list(range(1, 16)), dtype="object"),
    })

    result = dfs_ceiling.compute_ceiling_percentiles(history, percentile=90, min_games=10).set_index("key_mlbam")

    expected = pd.Series(range(1, 16)).quantile(0.90)
    assert result.loc[1, "Ceiling_DK_Points"] == pytest.approx(expected)


def test_compute_ceiling_percentiles_empty_history_returns_empty_with_expected_columns():
    result = dfs_ceiling.compute_ceiling_percentiles(pd.DataFrame(columns=["key_mlbam", "Actual_DK_Points_Modeled"]))
    assert result.empty
    assert list(result.columns) == ["key_mlbam", "Ceiling_DK_Points", "n_games", "Ceiling_Source"]


def test_backtest_ceiling_signal_no_persisted_data_returns_zero_n(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    result = dfs_ceiling.backtest_ceiling_signal(str(raw_dir), season=2026, days=20)
    assert result["hitters"]["n"] == 0
    assert result["pitchers"]["n"] == 0


def _game_rows(game_pk, date, events, pitcher=99, batter=1, home_team="NYY", away_team="BOS"):
    # Same minimal fixture shape as test_dfs_backtest.py's own _game_rows -
    # duplicated locally rather than cross-imported, matching this
    # project's self-contained-test-file convention.
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
            "bat_score": pre, "post_bat_score": away_runs,
        })
    return rows


def _multi_game_statcast(n_games=6, gap_days=5):
    events = ["strikeout"] * 5 + ["field_out"] * 6 + ["walk"] * 3 + ["single"] * 4 + ["double"] * 1 + ["home_run"] * 1
    rows = []
    for i in range(n_games):
        date = pd.Timestamp("2026-05-01") + pd.Timedelta(days=i * gap_days)
        rows.extend(_game_rows(i + 1, date, events))
    return pd.DataFrame(rows)


def test_backtest_ceiling_signal_returns_well_formed_metrics(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _multi_game_statcast(n_games=8).to_parquet(raw_dir / "statcast_2026.parquet", index=False)

    result = dfs_ceiling.backtest_ceiling_signal(str(raw_dir), season=2026, days=5)

    for player_type in ("hitters", "pitchers"):
        metrics = result[player_type]
        assert "n" in metrics
        if metrics["n"] >= 2 and "ceiling_correlation" in metrics:
            # A single repeating batter/pitcher in this synthetic fixture
            # can produce zero-variance Ceiling_DK_Points on some dates -
            # np.corrcoef legitimately returns nan there, not a bug.
            correlation = metrics["ceiling_correlation"]
            assert pd.isna(correlation) or -1.0 <= correlation <= 1.0
            if "ceiling_capture_rate" in metrics:
                assert 0.0 <= metrics["ceiling_capture_rate"] <= 1.0
                assert 0.0 <= metrics["mean_projection_capture_rate"] <= 1.0


def test_compute_upside_deviation_exact_arithmetic_for_a_qualified_player():
    # Player 1: [0, 0, 10, 10, 10] -> mean=6. Upside-only deviations:
    # 0 (below mean, clipped to 0), 0, 4, 4, 4 -> squared: 0,0,16,16,16.
    # mean of those / 5 games = 48/5 = 9.6 -> sqrt = 3.098...
    values = [0, 0, 10, 10, 10, 0, 0, 10, 10, 10]  # 10 games, clears min_games=10
    history = pd.DataFrame({"key_mlbam": [1] * len(values), "Actual_DK_Points_Modeled": values})

    result = dfs_ceiling.compute_upside_deviation(history, min_games=10).set_index("key_mlbam")

    mean = sum(values) / len(values)
    expected = (sum(max(v - mean, 0) ** 2 for v in values) / len(values)) ** 0.5
    assert result.loc[1, "Upside_Deviation"] == pytest.approx(expected)
    assert result.loc[1, "n_games"] == 10
    assert result.loc[1, "Upside_Deviation_Source"] == "player"


def test_compute_upside_deviation_ignores_downside_spread():
    # Two series with the SAME mean (9) and the SAME upside-side values
    # (10, 12, 14, 16, 18), but very different downside spread - a
    # symmetric bust pattern in one, a shallow one in the other. Only the
    # upside side is measured, so both must produce the SAME
    # Upside_Deviation despite very different overall variance.
    symmetric_bust = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]  # mean=9
    shallow_bust = [4, 4, 4, 4, 4, 10, 12, 14, 16, 18]  # mean=9 too
    assert sum(symmetric_bust) / len(symmetric_bust) == pytest.approx(sum(shallow_bust) / len(shallow_bust))

    history_symmetric = pd.DataFrame({"key_mlbam": [1] * len(symmetric_bust), "Actual_DK_Points_Modeled": symmetric_bust})
    history_shallow = pd.DataFrame({"key_mlbam": [1] * len(shallow_bust), "Actual_DK_Points_Modeled": shallow_bust})
    result_symmetric = dfs_ceiling.compute_upside_deviation(history_symmetric, min_games=10).set_index("key_mlbam")
    result_shallow = dfs_ceiling.compute_upside_deviation(history_shallow, min_games=10).set_index("key_mlbam")

    assert result_symmetric.loc[1, "Upside_Deviation"] == pytest.approx(result_shallow.loc[1, "Upside_Deviation"])


def test_compute_upside_deviation_small_sample_falls_back_to_group_wide():
    player_1_values = list(range(1, 16))  # 15 games, clears min_games=10
    player_2_values = [0, 20]  # only 2 games, below min_games=10
    history = pd.DataFrame({
        "key_mlbam": [1] * len(player_1_values) + [2] * len(player_2_values),
        "Actual_DK_Points_Modeled": player_1_values + player_2_values,
    })

    result = dfs_ceiling.compute_upside_deviation(history, min_games=10).set_index("key_mlbam")

    all_values = player_1_values + player_2_values
    group_mean = sum(all_values) / len(all_values)
    expected_group_deviation = (sum(max(v - group_mean, 0) ** 2 for v in all_values) / len(all_values)) ** 0.5

    assert result.loc[2, "n_games"] == 2
    assert result.loc[2, "Upside_Deviation_Source"] == "group_fallback"
    assert result.loc[2, "Upside_Deviation"] == pytest.approx(expected_group_deviation)
    assert result.loc[1, "Upside_Deviation_Source"] == "player"
    # Player 1's own value must differ from the group-wide fallback here -
    # otherwise this test couldn't distinguish "used its own value" from
    # "used the fallback by coincidence".
    assert result.loc[1, "Upside_Deviation"] != pytest.approx(expected_group_deviation)


def test_compute_upside_deviation_empty_history_returns_empty_with_expected_columns():
    result = dfs_ceiling.compute_upside_deviation(pd.DataFrame(columns=["key_mlbam", "Actual_DK_Points_Modeled"]))
    assert result.empty
    assert list(result.columns) == ["key_mlbam", "Upside_Deviation", "n_games", "Upside_Deviation_Source"]


def test_compute_boom_adjusted_score_exact_arithmetic():
    mean_points = pd.Series([5.0, 10.0])
    upside_deviation = pd.Series([2.0, 0.5])

    result = dfs_ceiling.compute_boom_adjusted_score(mean_points, upside_deviation, k=1.5)

    assert result.tolist() == pytest.approx([5.0 + 1.5 * 2.0, 10.0 + 1.5 * 0.5])


def test_compute_boom_adjusted_score_zero_k_reduces_to_mean():
    mean_points = pd.Series([5.0, 10.0])
    upside_deviation = pd.Series([2.0, 0.5])

    result = dfs_ceiling.compute_boom_adjusted_score(mean_points, upside_deviation, k=0.0)

    assert result.tolist() == pytest.approx(mean_points.tolist())


def test_compute_boom_adjusted_score_rewards_volatile_player_over_steady_one_with_same_or_higher_mean():
    # The user's exact scenario: player A scores a flat 5 every night
    # (zero deviation); player B averages 4.8 but with real upside swings
    # (nonzero Upside_Deviation). A positive k must rank B above A despite
    # B's slightly lower mean.
    mean_points = pd.Series({"A": 5.0, "B": 4.8})
    upside_deviation = pd.Series({"A": 0.0, "B": 6.0})

    result = dfs_ceiling.compute_boom_adjusted_score(mean_points, upside_deviation, k=0.5)

    assert result["B"] > result["A"]


def test_backtest_boom_adjusted_signal_no_persisted_data_returns_zero_n_per_k(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    result = dfs_ceiling.backtest_boom_adjusted_signal(str(raw_dir), season=2026, days=20, k_grid=[0.0, 1.0])

    assert result["hitters"][0.0]["n"] == 0
    assert result["hitters"][1.0]["n"] == 0
    assert result["pitchers"][0.0]["n"] == 0


def test_backtest_boom_adjusted_signal_returns_well_formed_metrics_per_k(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _multi_game_statcast(n_games=8).to_parquet(raw_dir / "statcast_2026.parquet", index=False)

    k_grid = [0.0, 0.5, 1.0]
    result = dfs_ceiling.backtest_boom_adjusted_signal(str(raw_dir), season=2026, days=5, k_grid=k_grid)

    for player_type in ("hitters", "pitchers"):
        for k in k_grid:
            metrics = result[player_type][k]
            assert "n" in metrics
            if metrics["n"] >= 2 and "correlation" in metrics:
                correlation = metrics["correlation"]
                assert pd.isna(correlation) or -1.0 <= correlation <= 1.0
                if "capture_rate" in metrics:
                    assert 0.0 <= metrics["capture_rate"] <= 1.0
