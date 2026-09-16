import pandas as pd
import pytest

from mlb_metrics import config, nfl_prop_predictions


def _top_props(rows):
    """rows: list of dicts in nfl_player_props.top_prop_bets' own output shape."""
    return pd.DataFrame(rows)


def _pick(player_id="wr1", category="Receptions", direction="Over", player_rate=5.0, **overrides):
    row = {
        "player_id": player_id, "player_name": "WR One", "team": "SF", "position": "WR",
        "opponent": "SEA", "games": 10, "category": category, "player_rate": player_rate,
        "opponent_allowed_rate": 6.0, "league_rate": 5.0, "ratio": 1.2, "direction": direction,
    }
    row.update(overrides)
    return row


def test_select_prop_picks_stamps_season_week_and_version():
    result = nfl_prop_predictions.select_prop_picks(_top_props([_pick()]), season=2026, week=3)

    assert result.loc[0, "season"] == 2026
    assert result.loc[0, "week"] == 3
    assert result.loc[0, "model_version"] == config.NFL_PROP_MODEL_VERSION
    # player_rate is logged as the real settle-against baseline.
    assert result.loc[0, "baseline_rate"] == 5.0
    assert pd.isna(result.loc[0, "actual_value"])
    assert pd.isna(result.loc[0, "hit"])


def test_select_prop_picks_handles_a_real_empty_week():
    # A real bye-heavy week (or one where nothing cleared the real
    # games/usage floors) must not raise - same graceful contract
    # nfl_player_props.write_prop_bets_csv already has for that case.
    result = nfl_prop_predictions.select_prop_picks(pd.DataFrame(), season=2026, week=3)

    assert result.empty
    assert list(result.columns) == nfl_prop_predictions.PROP_PREDICTION_COLUMNS


def test_append_prop_predictions_dedupes_on_player_week_and_category(tmp_path):
    log_path = str(tmp_path / "props.csv")
    week3 = nfl_prop_predictions.select_prop_picks(_top_props([_pick()]), season=2026, week=3)
    nfl_prop_predictions.append_prop_predictions(week3, log_path)

    # Same real player, same real week, but a DIFFERENT real category -
    # a legitimately separate pick, not a duplicate.
    other_category = nfl_prop_predictions.select_prop_picks(
        _top_props([_pick(category="Receiving Yards", player_rate=60.0)]), season=2026, week=3
    )
    combined = nfl_prop_predictions.append_prop_predictions(other_category, log_path)

    assert len(combined) == 2
    assert set(combined["category"]) == {"Receptions", "Receiving Yards"}


def test_resolve_prop_predictions_scores_a_real_over_hit_and_miss(tmp_path):
    log_path = str(tmp_path / "props.csv")
    picks = nfl_prop_predictions.select_prop_picks(
        _top_props([
            _pick(player_id="hit_over", direction="Over", player_rate=5.0),
            _pick(player_id="miss_over", direction="Over", player_rate=5.0),
        ]),
        season=2026, week=3,
    )
    nfl_prop_predictions.append_prop_predictions(picks, log_path)

    weekly = pd.DataFrame([
        {"player_id": "hit_over", "season": 2026, "week": 3, "receptions": 8.0},
        {"player_id": "miss_over", "season": 2026, "week": 3, "receptions": 2.0},
    ])

    resolved = nfl_prop_predictions.resolve_prop_predictions(log_path, weekly).set_index("player_id")

    assert resolved.loc["hit_over", "actual_value"] == 8.0
    assert resolved.loc["hit_over", "hit"] == 1
    assert resolved.loc["miss_over", "actual_value"] == 2.0
    assert resolved.loc["miss_over", "hit"] == 0


def test_resolve_prop_predictions_scores_a_real_under_pick_in_the_right_direction(tmp_path):
    log_path = str(tmp_path / "props.csv")
    picks = nfl_prop_predictions.select_prop_picks(
        _top_props([_pick(player_id="under_hit", direction="Under", player_rate=5.0)]), season=2026, week=3
    )
    nfl_prop_predictions.append_prop_predictions(picks, log_path)

    weekly = pd.DataFrame([{"player_id": "under_hit", "season": 2026, "week": 3, "receptions": 2.0}])

    resolved = nfl_prop_predictions.resolve_prop_predictions(log_path, weekly)

    # Coming in BELOW the baseline is a real hit for a real Under pick.
    assert resolved.loc[0, "hit"] == 1


def test_resolve_prop_predictions_records_an_exact_tie_as_a_real_push(tmp_path):
    log_path = str(tmp_path / "props.csv")
    picks = nfl_prop_predictions.select_prop_picks(
        _top_props([_pick(player_id="pusher", direction="Over", player_rate=4.0)]), season=2026, week=3
    )
    nfl_prop_predictions.append_prop_predictions(picks, log_path)

    weekly = pd.DataFrame([{"player_id": "pusher", "season": 2026, "week": 3, "receptions": 4.0}])

    resolved = nfl_prop_predictions.resolve_prop_predictions(log_path, weekly)

    assert resolved.loc[0, "actual_value"] == 4.0
    assert pd.isna(resolved.loc[0, "hit"])  # a real push, never silently a loss


def test_resolve_prop_predictions_leaves_a_real_dnp_pending_not_a_loss(tmp_path):
    # A real inactive/injured/benched player has no real row that week -
    # scoring that as a miss would understate a real hit rate for a
    # reason unrelated to the real signal being measured.
    log_path = str(tmp_path / "props.csv")
    picks = nfl_prop_predictions.select_prop_picks(
        _top_props([_pick(player_id="did_not_play", direction="Over", player_rate=5.0)]), season=2026, week=3
    )
    nfl_prop_predictions.append_prop_predictions(picks, log_path)

    weekly = pd.DataFrame([{"player_id": "someone_else", "season": 2026, "week": 3, "receptions": 9.0}])

    resolved = nfl_prop_predictions.resolve_prop_predictions(log_path, weekly)

    assert pd.isna(resolved.loc[0, "actual_value"])
    assert pd.isna(resolved.loc[0, "hit"])


def test_resolve_prop_predictions_leaves_an_unplayed_week_pending(tmp_path):
    log_path = str(tmp_path / "props.csv")
    picks = nfl_prop_predictions.select_prop_picks(
        _top_props([_pick(player_id="wr1")]), season=2026, week=5
    )
    nfl_prop_predictions.append_prop_predictions(picks, log_path)

    # Real weekly stats exist, but only for an EARLIER real week.
    weekly = pd.DataFrame([{"player_id": "wr1", "season": 2026, "week": 3, "receptions": 9.0}])

    resolved = nfl_prop_predictions.resolve_prop_predictions(log_path, weekly)

    assert pd.isna(resolved.loc[0, "actual_value"])


def test_resolve_prop_predictions_uses_the_right_stat_column_per_category(tmp_path):
    log_path = str(tmp_path / "props.csv")
    picks = nfl_prop_predictions.select_prop_picks(
        _top_props([
            _pick(player_id="p1", category="Receiving Yards", direction="Over", player_rate=50.0),
            _pick(player_id="p1", category="Receptions", direction="Over", player_rate=5.0),
        ]),
        season=2026, week=3,
    )
    nfl_prop_predictions.append_prop_predictions(picks, log_path)

    weekly = pd.DataFrame([
        {"player_id": "p1", "season": 2026, "week": 3, "receptions": 3.0, "receiving_yards": 90.0},
    ])

    resolved = nfl_prop_predictions.resolve_prop_predictions(log_path, weekly).set_index("category")

    # Same real player, same real week - each category must settle
    # against its OWN real stat, not share one.
    assert resolved.loc["Receiving Yards", "actual_value"] == 90.0
    assert resolved.loc["Receiving Yards", "hit"] == 1
    assert resolved.loc["Receptions", "actual_value"] == 3.0
    assert resolved.loc["Receptions", "hit"] == 0


def test_summarize_prop_results_reports_hit_rate_pushes_and_pending(tmp_path):
    log_path = str(tmp_path / "props.csv")
    picks = nfl_prop_predictions.select_prop_picks(
        _top_props([
            _pick(player_id="a", direction="Over", player_rate=5.0),
            _pick(player_id="b", direction="Over", player_rate=5.0),
            _pick(player_id="pusher", direction="Over", player_rate=5.0),
            _pick(player_id="pending", direction="Over", player_rate=5.0),
        ]),
        season=2026, week=3,
    )
    nfl_prop_predictions.append_prop_predictions(picks, log_path)
    weekly = pd.DataFrame([
        {"player_id": "a", "season": 2026, "week": 3, "receptions": 9.0},
        {"player_id": "b", "season": 2026, "week": 3, "receptions": 1.0},
        {"player_id": "pusher", "season": 2026, "week": 3, "receptions": 5.0},
    ])
    nfl_prop_predictions.resolve_prop_predictions(log_path, weekly)

    summary = nfl_prop_predictions.summarize_prop_results(log_path)
    overall = summary[summary["category"] == "all"].iloc[0]

    assert overall["n_resolved"] == 2
    assert overall["n_hits"] == 1
    assert overall["hit_rate"] == pytest.approx(0.5)
    assert overall["n_pushes"] == 1
    assert overall["n_pending"] == 1


def test_summarize_prop_results_with_no_log_yet_is_empty_not_a_crash(tmp_path):
    result = nfl_prop_predictions.summarize_prop_results(str(tmp_path / "nothing_here.csv"))

    assert result.empty
