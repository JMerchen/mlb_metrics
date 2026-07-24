import pandas as pd
import pytest

from mlb_metrics import teams


def test_compute_strength_metrics_survives_chained_groupby_apply():
    """Regression test: compute_strength_metrics chains two separate
    `.groupby("team").apply(...)` calls, and the helpers re-derive "team"
    from `group.name` rather than relying on it staying a column in `group`
    - pandas' groupby(...).apply() excludes the grouping column from what's
    passed to the applied function, and the second groupby("team") call
    would otherwise KeyError on a pandas version where that's enforced
    (this broke in production once pandas made this the only behavior)."""
    record = pd.DataFrame({
        "opp": ["BOS", "NYY", "BOS", "NYY"],
        "team": ["NYY", "BOS", "NYY", "BOS"],
        "game_date": pd.to_datetime(["2026-06-01", "2026-06-01", "2026-06-02", "2026-06-02"]),
        "game_id": [1, 1, 2, 2],
        "win": [1, 0, 0, 1],
        "loss": [0, 1, 1, 0],
        "rs": [5, 3, 2, 4],
        "ra": [3, 5, 4, 2],
    })

    current_strength, sos = teams.compute_strength_metrics(record)

    assert set(current_strength["team"]) == {"NYY", "BOS"}
    assert set(sos["team"]) == {"NYY", "BOS"}
    # NYY is 1-1 overall but its most recent game (game 2) was a win, so its
    # current (most-recent-game) rolling win rate should be higher than BOS's.
    nyy_current = current_strength.set_index("team").loc["NYY", "current_strength"]
    bos_current = current_strength.set_index("team").loc["BOS", "current_strength"]
    assert nyy_current > bos_current


def test_compute_park_factors_normalizes_to_league_average():
    data = pd.DataFrame([
        {"game_id": 1, "home_team": "A", "post_home_score": 6, "post_away_score": 4},  # combined 10
        {"game_id": 2, "home_team": "A", "post_home_score": 5, "post_away_score": 3},  # combined 8
        {"game_id": 3, "home_team": "B", "post_home_score": 4, "post_away_score": 2},  # combined 6
        {"game_id": 4, "home_team": "C", "post_home_score": 3, "post_away_score": 3},  # combined 6
        {"game_id": 5, "home_team": "C", "post_home_score": 4, "post_away_score": 2},  # combined 6
    ])

    park_factors = teams.compute_park_factors(data).set_index("team")

    # A avg=(10+8)/2=9, B avg=6, C avg=(6+6)/2=6 -> league avg=(9+6+6)/3=7.
    assert park_factors.loc["A", "Park_Factor"] == pytest.approx(9 / 7)
    assert park_factors.loc["B", "Park_Factor"] == pytest.approx(6 / 7)
    assert park_factors.loc["C", "Park_Factor"] == pytest.approx(6 / 7)


def test_compute_park_factors_uses_final_combined_score_not_every_pitch():
    # Two rows for the same game (score climbing pitch-by-pitch, as real
    # Statcast data does) - only the final (max) combined score should
    # count, same "max combined score = final state" pattern used by
    # build_team_record/data.extract_game_results, not a sum across rows.
    data = pd.DataFrame([
        {"game_id": 1, "home_team": "A", "post_home_score": 1, "post_away_score": 0},
        {"game_id": 1, "home_team": "A", "post_home_score": 3, "post_away_score": 2},  # final: combined 5
        {"game_id": 2, "home_team": "B", "post_home_score": 3, "post_away_score": 2},  # final: combined 5
    ])

    park_factors = teams.compute_park_factors(data).set_index("team")

    assert park_factors.loc["A", "Park_Factor"] == pytest.approx(1.0)
    assert park_factors.loc["B", "Park_Factor"] == pytest.approx(1.0)


def test_compute_home_run_stats_no_home_runs_yet_does_not_crash():
    # Regression test: early in a season (or any short as-of-date history
    # slice - see dfs_backtest.assemble_ml_training_rows, which replays
    # from day one), no home run may have occurred at all yet, so the
    # internal pivot table can be missing the "homer" column entirely -
    # this used to raise KeyError("homer") instead of just reporting 0.
    data = pd.DataFrame([
        {
            "game_id": 1, "home_team": "NYY", "away_team": "BOS", "inning_topbot": "Top",
            "events": "single", "home_score": 0, "away_score": 0, "post_home_score": 0, "post_away_score": 1,
        },
        {
            "game_id": 1, "home_team": "NYY", "away_team": "BOS", "inning_topbot": "Bot",
            "events": "field_out", "home_score": 0, "away_score": 1, "post_home_score": 0, "post_away_score": 1,
        },
    ])

    for_merge, sus = teams.compute_home_run_stats(data)  # must not raise

    assert list(for_merge.columns) == ["team", "home_run_reliance", "homer_per_game", "game_homer_rate"]
