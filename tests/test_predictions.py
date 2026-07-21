import pandas as pd
import pytest

from mlb_metrics import predictions


def _hitters(rows):
    """rows: list of (key_mlbam, pa_l, pa_r, game_hit_prob)."""
    return pd.DataFrame(
        [
            {
                "key_mlbam": key, "name_first": f"F{key}", "name_last": f"L{key}", "team": "NYY",
                "PA_L": pa_l, "PA_R": pa_r,
                "probability_L": 0, "probability_R": 0, "probability": 0,
                "Game_Hit_Probability": ghp, "Consistency": 0, "Approach": 0, "Expected_Bases": 0,
            }
            for key, pa_l, pa_r, ghp in rows
        ]
    )


def test_select_picks_applies_plate_appearance_qualifier_and_ranks():
    hitters = _hitters([
        (1, 0, 5, 0.99),   # unqualified: only 5 PA, would otherwise rank first
        (2, 10, 20, 0.80),
        (3, 15, 15, 0.70),
        (4, 0, 40, 0.60),
    ])

    picks = predictions.select_picks(hitters, "2026-06-20", top_n=2, min_plate_appearances=30)

    assert list(picks["key_mlbam"]) == [2, 3]
    assert list(picks["rank"]) == [1, 2]
    assert (picks["date"] == pd.Timestamp("2026-06-20")).all()
    assert list(picks["predicted_probability"]) == [0.80, 0.70]
    assert (picks["metric"] == "Game_Hit_Probability").all()
    assert picks["actual_hit"].isna().all()
    assert picks.loc[0, "name"] == "F2 L2"


def test_append_predictions_dedupes_and_prefers_existing_row(tmp_path):
    log_path = str(tmp_path / "predictions.csv")

    day1 = predictions.select_picks(_hitters([(1, 0, 40, 0.9)]), "2026-06-19", top_n=1, min_plate_appearances=30)
    predictions.append_predictions(day1, log_path)

    # Simulate resolution: mark the 06-19 pick as a hit directly in the log.
    resolved = pd.read_csv(log_path, parse_dates=["date"])
    resolved.loc[0, "actual_hit"] = 1
    resolved.to_csv(log_path, index=False)

    # Re-logging the same (date, key_mlbam, metric) should NOT clobber the
    # already-resolved actual_hit, and a genuinely new day's pick should
    # simply be added.
    day1_again = predictions.select_picks(_hitters([(1, 0, 40, 0.9)]), "2026-06-19", top_n=1, min_plate_appearances=30)
    day2 = predictions.select_picks(_hitters([(1, 0, 40, 0.9)]), "2026-06-20", top_n=1, min_plate_appearances=30)
    combined = predictions.append_predictions(pd.concat([day1_again, day2]), log_path)

    assert len(combined) == 2
    row_19 = combined[combined["date"] == "2026-06-19"].iloc[0]
    assert row_19["actual_hit"] == 1
    row_20 = combined[combined["date"] == "2026-06-20"].iloc[0]
    assert pd.isna(row_20["actual_hit"])


def test_append_predictions_dedupes_within_a_single_fresh_batch(tmp_path):
    """Regression test: a `picks` batch that already contains duplicate
    (date, key_mlbam, metric) rows - e.g. from git_backtest reconstructing
    the same date via two different commits - must be deduped even on the
    very first write, when there's no existing log to merge against yet."""
    log_path = str(tmp_path / "predictions.csv")

    day = predictions.select_picks(_hitters([(1, 0, 40, 0.9)]), "2026-06-19", top_n=1, min_plate_appearances=30)
    duplicated_batch = pd.concat([day, day.copy()], ignore_index=True)

    combined = predictions.append_predictions(duplicated_batch, log_path)

    assert len(combined) == 1
    logged = pd.read_csv(log_path, parse_dates=["date"])
    assert len(logged) == 1


def test_resolve_predictions_fills_pending_and_leaves_resolved_rows_alone(tmp_path):
    log_path = str(tmp_path / "predictions.csv")
    log = pd.DataFrame([
        {"date": "2026-06-18", "key_mlbam": 1, "name": "A", "rank": 1, "predicted_probability": 0.9, "metric": "Game_Hit_Probability", "actual_hit": 1},
        {"date": "2026-06-19", "key_mlbam": 1, "name": "A", "rank": 1, "predicted_probability": 0.8, "metric": "Game_Hit_Probability", "actual_hit": None},
        {"date": "2026-06-19", "key_mlbam": 2, "name": "B", "rank": 2, "predicted_probability": 0.7, "metric": "Game_Hit_Probability", "actual_hit": None},
    ])
    log.to_csv(log_path, index=False)

    completed_events = pd.DataFrame([
        {"game_date": pd.Timestamp("2026-06-18"), "batter": 1, "events": "home_run"},  # already resolved, ignored
        {"game_date": pd.Timestamp("2026-06-19"), "batter": 1, "events": "single"},
        {"game_date": pd.Timestamp("2026-06-19"), "batter": 2, "events": "field_out"},
    ])

    result = predictions.resolve_predictions(log_path, completed_events).set_index(["date", "key_mlbam"])

    assert result.loc[(pd.Timestamp("2026-06-18"), 1), "actual_hit"] == 1  # untouched
    assert result.loc[(pd.Timestamp("2026-06-19"), 1), "actual_hit"] == 1  # single -> hit
    assert result.loc[(pd.Timestamp("2026-06-19"), 2), "actual_hit"] == 0  # field_out -> no hit


def test_resolve_predictions_missing_log_returns_empty():
    result = predictions.resolve_predictions("/nonexistent/path.csv", pd.DataFrame(columns=["game_date", "batter", "events"]))
    assert result.empty
