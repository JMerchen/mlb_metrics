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
    assert picks["at_bats"].isna().all()
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
        # Already resolved (at_bats already set) - must be left alone even
        # though completed_events below would compute a *different* result
        # for it (a miss instead of the stored hit), proving it's skipped.
        {"date": "2026-06-18", "key_mlbam": 1, "name": "A", "rank": 1, "predicted_probability": 0.9, "metric": "Game_Hit_Probability", "actual_hit": 1, "at_bats": 1},
        {"date": "2026-06-19", "key_mlbam": 1, "name": "A", "rank": 1, "predicted_probability": 0.8, "metric": "Game_Hit_Probability", "actual_hit": None, "at_bats": None},
        {"date": "2026-06-19", "key_mlbam": 2, "name": "B", "rank": 2, "predicted_probability": 0.7, "metric": "Game_Hit_Probability", "actual_hit": None, "at_bats": None},
        # Batter 3 had zero at-bats on 06-19 (not in completed_events at all
        # for that date) - should resolve to a confirmed no_game, not stay pending.
        {"date": "2026-06-19", "key_mlbam": 3, "name": "C", "rank": 3, "predicted_probability": 0.6, "metric": "Game_Hit_Probability", "actual_hit": None, "at_bats": None},
        # 06-20 is beyond the outcome data's coverage - must stay pending.
        {"date": "2026-06-20", "key_mlbam": 1, "name": "A", "rank": 1, "predicted_probability": 0.85, "metric": "Game_Hit_Probability", "actual_hit": None, "at_bats": None},
    ])
    log.to_csv(log_path, index=False)

    completed_events = pd.DataFrame([
        {"game_date": pd.Timestamp("2026-06-18"), "batter": 1, "events": "field_out"},  # would be a miss - must be ignored
        {"game_date": pd.Timestamp("2026-06-19"), "batter": 1, "events": "single"},
        {"game_date": pd.Timestamp("2026-06-19"), "batter": 2, "events": "field_out"},
        # note: no rows at all for batter 3 on 06-19, and 06-20 doesn't appear anywhere.
    ])

    result = predictions.resolve_predictions(log_path, completed_events).set_index(["date", "key_mlbam"])

    row_18 = result.loc[(pd.Timestamp("2026-06-18"), 1)]
    assert row_18["actual_hit"] == 1 and row_18["at_bats"] == 1  # untouched, not recomputed

    row_19_1 = result.loc[(pd.Timestamp("2026-06-19"), 1)]
    assert row_19_1["actual_hit"] == 1 and row_19_1["at_bats"] == 1  # single -> hit

    row_19_2 = result.loc[(pd.Timestamp("2026-06-19"), 2)]
    assert row_19_2["actual_hit"] == 0 and row_19_2["at_bats"] == 1  # field_out -> miss

    row_19_3 = result.loc[(pd.Timestamp("2026-06-19"), 3)]
    assert pd.isna(row_19_3["actual_hit"]) and row_19_3["at_bats"] == 0  # confirmed no_game

    row_20 = result.loc[(pd.Timestamp("2026-06-20"), 1)]
    assert pd.isna(row_20["actual_hit"]) and pd.isna(row_20["at_bats"])  # still genuinely pending


def test_resolve_predictions_missing_log_returns_empty():
    result = predictions.resolve_predictions("/nonexistent/path.csv", pd.DataFrame(columns=["game_date", "batter", "events"]))
    assert result.empty


def test_resolve_predictions_migrates_a_log_written_before_at_bats_existed(tmp_path):
    log_path = str(tmp_path / "predictions.csv")
    # No at_bats column at all - simulates a log written by an older version.
    legacy_log = pd.DataFrame([
        {"date": "2026-06-18", "key_mlbam": 1, "name": "A", "rank": 1, "predicted_probability": 0.9, "metric": "Game_Hit_Probability", "actual_hit": None},
    ])
    legacy_log.to_csv(log_path, index=False)

    completed_events = pd.DataFrame([
        {"game_date": pd.Timestamp("2026-06-18"), "batter": 1, "events": "single"},
    ])

    result = predictions.resolve_predictions(log_path, completed_events)

    assert result.loc[0, "actual_hit"] == 1
    assert result.loc[0, "at_bats"] == 1
