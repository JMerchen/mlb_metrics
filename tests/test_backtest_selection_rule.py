"""Unit tests for scripts/backtest_selection_rule.py's own logic
(select_and_resolve, report) against a tiny hand-built `pools` list -
bypassing build_date_pools' real Statcast/model-artifact dependency, per
this project's "the backtest tool itself is under test, not just a
one-off unchecked script" discipline."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "backtest_selection_rule.py"


def _load_backtest_selection_rule_module():
    sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location("backtest_selection_rule", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pick_pool_row(key, name, ghp, probability, approach, matchup_hp, matchup_approach, model_hp):
    return {
        "key_mlbam": key, "name_first": name, "name_last": "L", "team": "NYY",
        "PA_L": 0, "PA_R": 40, "Game_Hit_Probability": ghp, "probability": probability,
        "Approach": approach, "Matchup_Hit_Probability": matchup_hp, "Matchup_Approach": matchup_approach,
        "Model_Hit_Probability": model_hp,
    }


def test_select_and_resolve_matchup_approach_ranks_by_approach_product():
    module = _load_backtest_selection_rule_module()
    pools = [
        {
            "date": "2026-06-20",
            "pick_pool": pd.DataFrame([
                # Player A: lower Model_Hit_Probability but higher Matchup_Approach.
                _pick_pool_row(1, "A", 0.85, 0.9, 0.765, 0.6, 0.459, 0.55),
                # Player B: higher Model_Hit_Probability but lower Matchup_Approach.
                _pick_pool_row(2, "B", 0.70, 0.8, 0.560, 0.9, 0.504, 0.92),
            ]),
            "got_hit": pd.DataFrame([{"key_mlbam": 1, "Got_Hit": 1}, {"key_mlbam": 2, "Got_Hit": 0}]),
            "has_model": True,
        }
    ]

    heuristic_picks = module.select_and_resolve(pools, "Matchup_Approach", top_n=1, min_plate_appearances=30)
    assert list(heuristic_picks["key_mlbam"]) == [2]  # higher Matchup_Approach (0.504 > 0.459)
    assert heuristic_picks.iloc[0]["actual_hit"] == 0
    assert heuristic_picks.iloc[0]["at_bats"] == 1

    model_picks = module.select_and_resolve(
        pools, "Model_Hit_Probability", top_n=1, min_plate_appearances=30, min_model_probability=0.0
    )
    assert list(model_picks["key_mlbam"]) == [2]  # higher Model_Hit_Probability (0.92 > 0.55)


def test_select_and_resolve_model_hit_probability_picks_differently_from_matchup_approach():
    module = _load_backtest_selection_rule_module()
    pools = [
        {
            "date": "2026-06-20",
            "pick_pool": pd.DataFrame([
                # Player A: HIGHEST Matchup_Approach, but LOWEST Model_Hit_Probability.
                # (Matchup_Hit_Probability=0.75 clears HITTER_MIN_PROBABILITY's 0.7 gate.)
                _pick_pool_row(1, "A", 0.85, 0.95, 0.8075, 0.75, 0.605625, 0.30),
                # Player B: lower Matchup_Approach, but HIGHEST Model_Hit_Probability.
                _pick_pool_row(2, "B", 0.72, 0.75, 0.5400, 0.75, 0.405000, 0.90),
            ]),
            "got_hit": pd.DataFrame([{"key_mlbam": 1, "Got_Hit": 0}, {"key_mlbam": 2, "Got_Hit": 1}]),
            "has_model": True,
        }
    ]

    heuristic_picks = module.select_and_resolve(pools, "Matchup_Approach", top_n=1, min_plate_appearances=30)
    assert list(heuristic_picks["key_mlbam"]) == [1]

    model_picks = module.select_and_resolve(
        pools, "Model_Hit_Probability", top_n=1, min_plate_appearances=30, min_model_probability=0.0
    )
    assert list(model_picks["key_mlbam"]) == [2]  # proves the two rules genuinely diverge


def test_select_and_resolve_skips_dates_with_no_usable_model_prediction():
    module = _load_backtest_selection_rule_module()
    pools = [
        {
            "date": "2026-06-20",
            "pick_pool": pd.DataFrame([_pick_pool_row(1, "A", 0.9, 0.9, 0.81, 0.8, 0.648, 0.9)]),
            "got_hit": pd.DataFrame([{"key_mlbam": 1, "Got_Hit": 1}]),
            "has_model": False,  # e.g. the model artifact wasn't loadable that date
        }
    ]

    model_picks = module.select_and_resolve(
        pools, "Model_Hit_Probability", top_n=1, min_plate_appearances=30, min_model_probability=0.0
    )
    assert model_picks.empty

    heuristic_picks = module.select_and_resolve(pools, "Matchup_Approach", top_n=1, min_plate_appearances=30)
    assert list(heuristic_picks["key_mlbam"]) == [1]  # heuristic tier is unaffected by has_model


def test_select_and_resolve_missing_from_got_hit_table_means_no_game():
    # A picked player absent from that date's real Got_Hit table (zero
    # at-bats that day) must resolve to at_bats=0/actual_hit=NaN, not be
    # silently dropped or treated as a miss.
    module = _load_backtest_selection_rule_module()
    pools = [
        {
            "date": "2026-06-20",
            "pick_pool": pd.DataFrame([_pick_pool_row(1, "A", 0.9, 0.9, 0.81, 0.8, 0.648, 0.9)]),
            "got_hit": pd.DataFrame(columns=["key_mlbam", "Got_Hit"]),
            "has_model": True,
        }
    ]

    picks = module.select_and_resolve(pools, "Matchup_Approach", top_n=1, min_plate_appearances=30)
    assert len(picks) == 1
    assert picks.iloc[0]["at_bats"] == 0
    assert pd.isna(picks.iloc[0]["actual_hit"])


def test_report_runs_without_error_on_empty_and_nonempty_input(capsys):
    module = _load_backtest_selection_rule_module()
    empty = pd.DataFrame(columns=["date", "key_mlbam", "rank", "predicted_probability", "actual_hit", "metric"])
    module.report("empty case", empty, total_dates=0)

    nonempty = pd.DataFrame([
        {"date": "2026-06-20", "key_mlbam": 1, "rank": 1, "predicted_probability": 0.8, "actual_hit": 1, "metric": "Game_Hit_Probability"},
    ])
    module.report("nonempty case", nonempty, total_dates=1)

    out = capsys.readouterr().out
    assert "empty case" in out
    assert "nonempty case" in out
