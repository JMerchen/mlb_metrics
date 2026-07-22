import pandas as pd

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
