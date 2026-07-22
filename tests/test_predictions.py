import pandas as pd
import pytest

from mlb_metrics import config, predictions


def _hitters(rows):
    """rows: list of (key_mlbam, pa_l, pa_r, game_hit_prob). probability
    mirrors game_hit_prob (a reasonable synthetic stand-in) so these
    fixtures clear select_picks' joint HITTER_MIN_PROBABILITY gate on both
    columns unless a test is specifically exercising that gate."""
    return pd.DataFrame(
        [
            {
                "key_mlbam": key, "name_first": f"F{key}", "name_last": f"L{key}", "team": "NYY",
                "PA_L": pa_l, "PA_R": pa_r,
                "probability_L": 0, "probability_R": 0, "probability": ghp,
                "Game_Hit_Probability": ghp, "Consistency": 0, "Approach": ghp * ghp, "Expected_Bases": 0,
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
    assert (picks["model_version"] == config.HITTER_MODEL_VERSION).all()


def test_select_picks_logs_probability_and_defaults_matchup_hit_probability_to_na():
    hitters = _hitters([(1, 0, 40, 0.9)])

    picks = predictions.select_picks(hitters, "2026-06-20", top_n=1, min_plate_appearances=30)

    assert picks.iloc[0]["probability"] == 0.9  # passed through from the hitters table
    assert pd.isna(picks.iloc[0]["Matchup_Hit_Probability"])  # not merged into `hitters` this call


def test_select_picks_logs_matchup_hit_probability_when_present():
    hitters = _hitters([(1, 0, 40, 0.9)])
    hitters["Matchup_Hit_Probability"] = 0.75

    picks = predictions.select_picks(hitters, "2026-06-20", top_n=1, min_plate_appearances=30)

    assert picks.iloc[0]["Matchup_Hit_Probability"] == 0.75


def test_select_picks_model_version_is_configurable():
    hitters = _hitters([(1, 0, 40, 0.9)])

    picks = predictions.select_picks(hitters, "2026-06-20", top_n=1, min_plate_appearances=30, model_version="test-v")

    assert picks.loc[0, "model_version"] == "test-v"


def test_append_predictions_migrates_a_log_written_before_model_version_existed(tmp_path):
    log_path = str(tmp_path / "predictions.csv")
    legacy_log = pd.DataFrame([
        {"date": "2026-06-18", "key_mlbam": 1, "name": "A", "rank": 1, "predicted_probability": 0.9,
         "metric": "Game_Hit_Probability", "actual_hit": 1, "at_bats": 1},
    ])
    legacy_log.to_csv(log_path, index=False)

    new_pick = predictions.select_picks(_hitters([(2, 0, 40, 0.9)]), "2026-06-19", top_n=1, min_plate_appearances=30)
    combined = predictions.append_predictions(new_pick, log_path)

    row_18 = combined[combined["date"] == "2026-06-18"].iloc[0]
    assert row_18["model_version"] == predictions.LEGACY_MODEL_VERSION
    row_19 = combined[combined["date"] == "2026-06-19"].iloc[0]
    assert row_19["model_version"] == config.HITTER_MODEL_VERSION


def test_select_picks_applies_lineup_qualifiers_when_columns_present():
    hitters = _hitters([
        (1, 0, 40, 0.90),  # top-half slot, consistent starter -> qualifies
        (2, 0, 40, 0.85),  # bats bottom of the order -> excluded
        (3, 0, 40, 0.80),  # bats top-half, but inconsistent (low start_rate) -> excluded
        (4, 0, 40, 0.70),  # never started at all (null avg) -> excluded, not treated as slot 0
    ])
    hitters["avg_batting_order"] = [2.0, 7.0, 2.0, float("nan")]
    hitters["start_rate"] = [0.9, 0.9, 0.2, 0.0]

    picks = predictions.select_picks(hitters, "2026-06-20", top_n=5, min_plate_appearances=30)

    assert list(picks["key_mlbam"]) == [1]


def test_select_picks_lineup_qualifiers_are_noop_without_columns():
    # A batter who would fail the lineup qualifiers if they applied must
    # still be picked when the columns simply aren't present (old wave.csv
    # snapshots from before this feature existed).
    hitters = _hitters([(1, 0, 40, 0.90)])
    assert "avg_batting_order" not in hitters.columns

    picks = predictions.select_picks(hitters, "2026-06-20", top_n=5, min_plate_appearances=30)

    assert list(picks["key_mlbam"]) == [1]


def test_select_picks_requires_both_probability_and_game_hit_probability_above_threshold():
    hitters = _hitters([
        (1, 0, 40, 0.90),  # both signals strong -> qualifies
        (2, 0, 40, 0.85),  # high Game_Hit_Probability but a divergent, low probability -> excluded
        (3, 0, 40, 0.75),  # high probability but a divergent, low Game_Hit_Probability -> excluded
    ])
    hitters.loc[hitters["key_mlbam"] == 2, "probability"] = 0.5
    hitters.loc[hitters["key_mlbam"] == 3, "Game_Hit_Probability"] = 0.5

    picks = predictions.select_picks(hitters, "2026-06-20", top_n=5, min_plate_appearances=30)

    assert list(picks["key_mlbam"]) == [1]


def test_select_picks_min_probability_is_configurable():
    hitters = _hitters([(1, 0, 40, 0.65)])  # would fail the default 0.7 gate

    excluded = predictions.select_picks(hitters, "2026-06-20", top_n=5, min_plate_appearances=30)
    assert excluded.empty

    included = predictions.select_picks(
        hitters, "2026-06-20", top_n=5, min_plate_appearances=30, min_probability=0.6
    )
    assert list(included["key_mlbam"]) == [1]


def test_select_picks_rank_metric_chooses_differently_than_metric_but_reports_metric():
    # Player 2 has the higher Game_Hit_Probability, but player 1 has the
    # higher Approach (Game_Hit_Probability * probability) once ranked on a
    # combined signal - rank_metric picks among qualified hitters by
    # whichever column it names, while predicted_probability/metric always
    # still reflect `metric` (Game_Hit_Probability here), not rank_metric.
    hitters = _hitters([(1, 0, 40, 0.80), (2, 0, 40, 0.82)])
    hitters.loc[hitters["key_mlbam"] == 1, "probability"] = 0.95
    hitters.loc[hitters["key_mlbam"] == 1, "Approach"] = 0.80 * 0.95
    hitters.loc[hitters["key_mlbam"] == 2, "probability"] = 0.72
    hitters.loc[hitters["key_mlbam"] == 2, "Approach"] = 0.82 * 0.72

    by_metric = predictions.select_picks(hitters, "2026-06-20", top_n=1, min_plate_appearances=30)
    assert list(by_metric["key_mlbam"]) == [2]

    by_rank_metric = predictions.select_picks(
        hitters, "2026-06-20", top_n=1, min_plate_appearances=30, rank_metric="Approach"
    )
    assert list(by_rank_metric["key_mlbam"]) == [1]
    assert by_rank_metric.iloc[0]["predicted_probability"] == 0.80  # still Game_Hit_Probability
    assert by_rank_metric.iloc[0]["metric"] == "Game_Hit_Probability"


def test_select_picks_filters_by_teams_playing_today():
    hitters = _hitters([
        (1, 0, 40, 0.90),
        (2, 0, 40, 0.80),
    ])
    hitters.loc[hitters["key_mlbam"] == 1, "team"] = "NYY"
    hitters.loc[hitters["key_mlbam"] == 2, "team"] = "BOS"

    picks = predictions.select_picks(
        hitters, "2026-06-20", top_n=5, min_plate_appearances=30, teams_playing_today={"BOS"}
    )

    assert list(picks["key_mlbam"]) == [2]


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


def test_resolve_predictions_migrates_a_log_written_before_model_version_existed(tmp_path):
    log_path = str(tmp_path / "predictions.csv")
    legacy_log = pd.DataFrame([
        {"date": "2026-06-18", "key_mlbam": 1, "name": "A", "rank": 1, "predicted_probability": 0.9,
         "metric": "Game_Hit_Probability", "actual_hit": None, "at_bats": None},
    ])
    legacy_log.to_csv(log_path, index=False)

    completed_events = pd.DataFrame([
        {"game_date": pd.Timestamp("2026-06-18"), "batter": 1, "events": "single"},
    ])

    result = predictions.resolve_predictions(log_path, completed_events)

    assert result.loc[0, "model_version"] == predictions.LEGACY_MODEL_VERSION
