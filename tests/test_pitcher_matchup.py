import pandas as pd
import pytest

from mlb_metrics import config, data, pitcher_matchup


@pytest.fixture(autouse=True)
def _no_network_name_register(monkeypatch):
    # Same rationale as test_dfs_backtest.py's/test_dfs_ceiling.py's
    # identical fixture: chadwick_register() hits a network URL this
    # sandbox blocks, and names aren't used in any numeric computation
    # backtest_pitcher_matchup_signal's underlying pipeline.compute_outputs
    # call needs.
    monkeypatch.setattr(
        data, "get_name_register",
        lambda: pd.DataFrame(columns=["key_mlbam", "key_bbref", "name_first", "name_last"]),
    )


def test_compute_opponent_offense_ratio_exact_arithmetic():
    # opponent at 6.0 bases/game vs league average 5.0 -> raw ratio 1.2.
    # weight=0.5 blends halfway to neutral: 1 + 0.5*(1.2-1) = 1.1.
    ratio = pitcher_matchup.compute_opponent_offense_ratio(
        pd.Series([6.0]), league_bases_pg=5.0, weight=0.5
    )
    assert ratio.iloc[0] == pytest.approx(1.1)


def test_compute_opponent_offense_ratio_weight_zero_is_always_neutral():
    # weight=0.0 must return EXACTLY 1.0 regardless of how extreme the
    # opponent's own bases_pg is - the built-in null hypothesis.
    ratio = pitcher_matchup.compute_opponent_offense_ratio(
        pd.Series([0.1, 50.0, 5.0]), league_bases_pg=5.0, weight=0.0
    )
    assert (ratio == 1.0).all()


def test_compute_opponent_offense_ratio_clips_extremes():
    lo, hi = config.PITCHER_MATCHUP_OFFENSE_CLIP
    # A wildly strong/weak opponent at weight=1.0 (full unblended ratio)
    # must still land within the configured clip bounds, not blow past them.
    ratio = pitcher_matchup.compute_opponent_offense_ratio(
        pd.Series([100.0, 0.01]), league_bases_pg=5.0, weight=1.0
    )
    assert ratio.iloc[0] == pytest.approx(hi)
    assert ratio.iloc[1] == pytest.approx(lo)


def test_attach_opponent_offense_missing_opponent_falls_back_to_league_average():
    pitchers = pd.DataFrame({"key_mlbam": [1, 2], "opponent": ["NYY", "ZZZ"]})
    confidence = pd.DataFrame({"team": ["NYY", "BOS"], "team_bases_pg": [8.0, 4.0]})
    # league average = (8.0 + 4.0) / 2 = 6.0

    result = pitcher_matchup.attach_opponent_offense(pitchers, confidence, weight=1.0)

    nyy_row = result[result["opponent"] == "NYY"].iloc[0]
    assert nyy_row["Opponent_Bases_PG"] == pytest.approx(8.0)
    # Raw ratio 8.0/6.0=1.333 clips to config.PITCHER_MATCHUP_OFFENSE_CLIP's
    # upper bound at weight=1.0 - see test_compute_opponent_offense_ratio_
    # clips_extremes for the clip itself; this just confirms the merge
    # path applies it.
    _, hi = config.PITCHER_MATCHUP_OFFENSE_CLIP
    assert nyy_row["Opponent_Offense_Ratio"] == pytest.approx(hi)

    # "ZZZ" isn't in confidence at all - falls back to the league average
    # itself, which makes the ratio exactly 1.0 (neutral), not a dropped row.
    zzz_row = result[result["opponent"] == "ZZZ"].iloc[0]
    assert zzz_row["Opponent_Bases_PG"] == pytest.approx(6.0)
    assert zzz_row["Opponent_Offense_Ratio"] == pytest.approx(1.0)


def test_compute_opponent_adjusted_pitcher_points_stronger_offense_lowers_points():
    # Both H_Allowed and ER carry NEGATIVE DK point weights
    # (DFS_DK_PITCHER_H_POINTS/ER_POINTS), so scaling them UP by a >1.0
    # ratio (a stronger-than-average opponent offense) must LOWER the
    # final DK_Points_Pitcher, not raise it - the sign this whole module
    # exists to introduce (facing a great offense should hurt a
    # projection, not help it).
    pitchers = pd.DataFrame({
        "Expected_IP": [6.0], "Expected_K": [6.0], "Expected_BB": [2.0],
        "Expected_H_Allowed": [5.0], "Expected_ER": [2.0],
    })
    baseline = pitcher_matchup.compute_opponent_adjusted_pitcher_points(pitchers, offense_ratio=pd.Series([1.0]))
    tougher = pitcher_matchup.compute_opponent_adjusted_pitcher_points(pitchers, offense_ratio=pd.Series([1.15]))
    easier = pitcher_matchup.compute_opponent_adjusted_pitcher_points(pitchers, offense_ratio=pd.Series([0.85]))

    assert tougher["DK_Points_Pitcher"].iloc[0] < baseline["DK_Points_Pitcher"].iloc[0]
    assert easier["DK_Points_Pitcher"].iloc[0] > baseline["DK_Points_Pitcher"].iloc[0]

    # Exact arithmetic check on the tougher-opponent case.
    expected = (
        6.0 * config.DFS_DK_PITCHER_IP_POINTS
        + 6.0 * config.DFS_DK_PITCHER_K_POINTS
        + 2.0 * config.DFS_DK_PITCHER_BB_POINTS
        + (5.0 * 1.15) * config.DFS_DK_PITCHER_H_POINTS
        + (2.0 * 1.15) * config.DFS_DK_PITCHER_ER_POINTS
    )
    assert tougher["DK_Points_Pitcher"].iloc[0] == pytest.approx(expected)


def test_compute_opponent_adjusted_pitcher_points_does_not_mutate_input():
    pitchers = pd.DataFrame({
        "Expected_IP": [6.0], "Expected_K": [6.0], "Expected_BB": [2.0],
        "Expected_H_Allowed": [5.0], "Expected_ER": [2.0],
    })
    pitcher_matchup.compute_opponent_adjusted_pitcher_points(pitchers, offense_ratio=pd.Series([1.15]))
    assert pitchers["Expected_H_Allowed"].iloc[0] == pytest.approx(5.0)


def test_backtest_pitcher_matchup_signal_no_persisted_data_returns_zero_n(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    result = pitcher_matchup.backtest_pitcher_matchup_signal(str(raw_dir), season=2026, days=20)
    assert result["n"] == 0


def _two_sided_game_rows(game_pk, date, away_events, home_events, away_pitcher, home_pitcher, home_team, away_team):
    # Same shape (and same reasoning) as test_dfs_backtest.py's own
    # _two_sided_game_rows - duplicated locally rather than cross-imported,
    # matching this project's self-contained-test-file convention. BOTH
    # halves are needed so teams.compute_offensive_edge/
    # compute_home_run_stats get real bases-for/bases-against data for
    # both teams (a Top-half-only fixture leaves every team's
    # team_bases_pg permanently NaN, which is exactly what this module's
    # opponent-offense adjustment needs to NOT be).
    rows = []
    away_runs = 0
    home_runs = 0
    for i, e in enumerate(away_events):
        pre = away_runs
        if e in ("home_run", "single"):
            away_runs += 1
        rows.append({
            "game_pk": game_pk, "game_date": date, "pitcher": home_pitcher, "batter": 1,
            "events": e, "p_throws": "R", "inning_topbot": "Top",
            "home_team": home_team, "away_team": away_team,
            "at_bat_number": i + 1, "pitch_number": 1,
            "home_score": home_runs, "away_score": pre,
            "post_home_score": home_runs, "post_away_score": away_runs,
            "bat_score": pre, "post_bat_score": away_runs,
        })
    for i, e in enumerate(home_events):
        pre = home_runs
        if e in ("home_run", "single"):
            home_runs += 1
        rows.append({
            "game_pk": game_pk, "game_date": date, "pitcher": away_pitcher, "batter": 2,
            "events": e, "p_throws": "R", "inning_topbot": "Bot",
            "home_team": home_team, "away_team": away_team,
            "at_bat_number": len(away_events) + i + 1, "pitch_number": 1,
            "home_score": pre, "away_score": away_runs,
            "post_home_score": home_runs, "post_away_score": away_runs,
            "bat_score": pre, "post_bat_score": home_runs,
        })
    return rows


def _multi_game_statcast(n_games=8, gap_days=5):
    # Just two teams (BOS strong offense, NYY weak-but-nonzero offense),
    # alternating who's home each game - real dispersion in team_bases_pg
    # is needed for the opponent-offense ratio to do anything at all, and
    # both teams need to appear as home team at least once and hit at
    # least one real home run (not just a single) for
    # teams.assemble_team_metrics' inner joins to keep both teams at all -
    # see test_dfs_backtest.py's identically-named fixture for the full
    # reasoning. Pitchers are CONSTANT per team so each clears
    # config.DFS_PITCHER_MIN_STARTS across the sample.
    strong_offense = ["home_run"] * 3 + ["single"] * 4 + ["field_out"] * 3
    weak_offense = ["home_run"] * 1 + ["strikeout"] * 7 + ["field_out"] * 2
    rows = []
    for i in range(n_games):
        date = pd.Timestamp("2026-05-01") + pd.Timedelta(days=i * gap_days)
        if i % 2 == 0:
            away_team, away_events, away_pitcher = "BOS", strong_offense, 981
            home_team, home_events, home_pitcher = "NYY", weak_offense, 991
        else:
            away_team, away_events, away_pitcher = "NYY", weak_offense, 991
            home_team, home_events, home_pitcher = "BOS", strong_offense, 981
        rows.extend(_two_sided_game_rows(
            i + 1, date, away_events, home_events, away_pitcher=away_pitcher, home_pitcher=home_pitcher,
            home_team=home_team, away_team=away_team,
        ))
    return pd.DataFrame(rows)


def test_backtest_pitcher_matchup_signal_returns_well_formed_metrics(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _multi_game_statcast(n_games=10).to_parquet(raw_dir / "statcast_2026.parquet", index=False)

    result = pitcher_matchup.backtest_pitcher_matchup_signal(str(raw_dir), season=2026, days=8)

    # This fixture has real dispersion in opponent offense, so it should
    # produce actual scored rows, not just an empty-shape result.
    assert result["n"] >= 2
    assert set(result["by_weight"].keys()) == set(config.PITCHER_MATCHUP_WEIGHT_GRID)
    for weight in config.PITCHER_MATCHUP_WEIGHT_GRID:
        metrics = result["by_weight"][weight]
        assert metrics["correlation"] is None or -1.0 <= metrics["correlation"] <= 1.0
        assert metrics["mae"] >= 0.0
