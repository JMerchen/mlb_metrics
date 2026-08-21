"""Unit tests for scripts/backtest_same_game_diversification.py's own
select_and_resolve/_count_same_game_pairs/report logic against tiny
hand-built pools - bypassing the real dfs_backtest._compute_date_outputs
Statcast dependency (the pools are built directly, not via
build_date_pools), mirroring tests/test_backtest_selection_rule.py's
established pattern for this same script's sibling."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "backtest_same_game_diversification.py"


def _load_module():
    sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location("backtest_same_game_diversification", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pool_row(key, ghp, game_pk, team="NYY"):
    # Matchup_Approach deliberately equals ghp directly (not Approach's
    # real ghp*ghp formula) so the tests' own margin arithmetic in their
    # comments matches the actual rank_metric values exactly.
    return {
        "key_mlbam": key, "name_first": f"F{key}", "name_last": f"L{key}", "team": team,
        "PA_L": 0, "PA_R": 40, "probability": ghp, "Game_Hit_Probability": ghp,
        "Consistency": 0, "Approach": ghp * ghp, "Matchup_Approach": ghp, "game_pk": game_pk,
    }


def _entry(date, rows, hit_by_key):
    pick_pool = pd.DataFrame([_pool_row(*r) for r in rows])
    got_hit = pd.DataFrame([{"key_mlbam": k, "Got_Hit": h} for k, h in hit_by_key.items()])
    return {"date": pd.Timestamp(date), "pick_pool": pick_pool, "got_hit": got_hit}


def test_select_and_resolve_applies_the_given_margin_and_resolves_real_outcomes():
    module = _load_module()
    # #1/#2 share game_pk=100; key=3 (game_pk=200) is a close alternative.
    rows = [(1, 0.90, 100), (2, 0.85, 100), (3, 0.80, 200)]
    entry = _entry("2026-06-20", rows, {1: 1, 2: 0, 3: 1})
    pools = [entry]

    picks_no_margin = module.select_and_resolve(pools, "Matchup_Approach", margin=0.0)
    assert list(picks_no_margin["key_mlbam"]) == [1, 2, 3]
    assert list(picks_no_margin["actual_hit"]) == [1, 0, 1]

    picks_diversified = module.select_and_resolve(pools, "Matchup_Approach", margin=0.06)
    # key=3 promoted into the #2 slot - the real outcome resolution still
    # correctly follows whichever key ends up in each rank.
    assert list(picks_diversified["key_mlbam"]) == [1, 3, 2]
    assert list(picks_diversified["actual_hit"]) == [1, 1, 0]


def test_count_same_game_pairs_counts_only_dates_with_a_real_shared_game_top_2():
    module = _load_module()
    same_game_entry = _entry("2026-06-20", [(1, 0.90, 100), (2, 0.85, 100)], {1: 1, 2: 1})
    different_game_entry = _entry("2026-06-21", [(1, 0.90, 100), (2, 0.85, 200)], {1: 1, 2: 0})

    count = module._count_same_game_pairs([same_game_entry, different_game_entry], margin=0.0, rank_metric="Matchup_Approach")

    assert count == 1


def test_report_computes_the_real_require_all_true_metric(capsys):
    module = _load_module()
    # Day 1: both picks hit (a real "both" success). Day 2: one miss (a
    # real "both" failure) - top_2_BOTH_hit_rate should be 0.5, not the
    # require_all=False ("any") rate, which would be higher.
    picks_df = pd.DataFrame([
        {"date": "2026-06-20", "key_mlbam": 1, "rank": 1, "predicted_probability": 0.9, "actual_hit": 1, "at_bats": 1},
        {"date": "2026-06-20", "key_mlbam": 2, "rank": 2, "predicted_probability": 0.8, "actual_hit": 1, "at_bats": 1},
        {"date": "2026-06-21", "key_mlbam": 1, "rank": 1, "predicted_probability": 0.9, "actual_hit": 1, "at_bats": 1},
        {"date": "2026-06-21", "key_mlbam": 2, "rank": 2, "predicted_probability": 0.8, "actual_hit": 0, "at_bats": 1},
    ])

    both_rate = module.report(0.0, picks_df, total_dates=2)

    assert both_rate == pytest.approx(0.5)
    out = capsys.readouterr().out
    assert "top_2_BOTH_hit_rate" in out
    assert "the real BTS rule" in out
