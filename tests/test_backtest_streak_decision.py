"""Unit tests for scripts/backtest_streak_decision.py's own day-level
aggregation, single-path simulation, and go/no-go reporting logic against
tiny hand-built fixtures - the real DP math itself is already covered by
tests/test_decision_theory.py, mirroring
tests/test_backtest_shrinkage.py's "the backtest tool itself is under
test" discipline."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "backtest_streak_decision.py"


def _load_module():
    sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location("backtest_streak_decision", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prediction_row(date, rank, predicted_probability, actual_hit, at_bats):
    return {
        "date": date, "rank": rank, "metric": "Game_Hit_Probability",
        "predicted_probability": predicted_probability, "actual_hit": actual_hit, "at_bats": at_bats,
    }


def test_load_day_level_aggregates_real_columns_correctly(tmp_path):
    module = _load_module()
    rows = [
        # Day 1: both picks hit - no_miss, n_hit=2, p=mean(0.9,0.8)=0.85
        _prediction_row("2026-06-01", 1, 0.9, 1, 4),
        _prediction_row("2026-06-01", 2, 0.8, 1, 4),
        # Day 2: one hit, one miss - any_miss=True, p=mean(0.6,0.5)=0.55
        _prediction_row("2026-06-02", 1, 0.6, 1, 4),
        _prediction_row("2026-06-02", 2, 0.5, 0, 4),
        # Day 3: still pending (at_bats unresolved) - must be dropped entirely
        _prediction_row("2026-06-03", 1, 0.9, None, None),
    ]
    csv_path = tmp_path / "predictions.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    day_level = module._load_day_level(str(csv_path))

    assert len(day_level) == 2  # the pending day is dropped
    day1 = day_level[day_level["date"] == pd.Timestamp("2026-06-01")].iloc[0]
    assert day1["p"] == pytest.approx(0.85)
    assert day1["n_hit"] == 2
    assert day1["any_miss"] == False  # noqa: E712
    day2 = day_level[day_level["date"] == pd.Timestamp("2026-06-02")].iloc[0]
    assert day2["p"] == pytest.approx(0.55)
    assert day2["any_miss"] == True  # noqa: E712


def test_simulate_always_play_applies_every_real_outcome():
    module = _load_module()
    day_level = pd.DataFrame([
        {"date": "2026-06-01", "p": 0.9, "n_hit": 2, "any_miss": False},
        {"date": "2026-06-02", "p": 0.5, "n_hit": 1, "any_miss": True},
        {"date": "2026-06-03", "p": 0.9, "n_hit": 1, "any_miss": False},
    ])
    # Day 1: streak 0 -> 2. Day 2: a real miss resets to 0 regardless of
    # p. Day 3: streak 0 -> 1.
    result = module._simulate(day_level, lambda streak, days_remaining, p: True)

    assert result["final_streak"] == 1
    assert result["longest_streak"] == 2
    assert result["n_played"] == 3


def test_simulate_sitting_out_a_day_ignores_its_real_outcome():
    module = _load_module()
    day_level = pd.DataFrame([
        {"date": "2026-06-01", "p": 0.9, "n_hit": 2, "any_miss": False},
        {"date": "2026-06-02", "p": 0.5, "n_hit": 1, "any_miss": True},  # would reset if played - sat out instead
        {"date": "2026-06-03", "p": 0.9, "n_hit": 1, "any_miss": False},
    ])
    result = module._simulate(day_level, lambda streak, days_remaining, p: p >= 0.8)

    # Day 1: streak 0 -> 2 (played). Day 2: sat out (p=0.5 < 0.8) - the
    # real miss that day is never applied, streak stays 2. Day 3: streak
    # 2 -> 3 (played).
    assert result["final_streak"] == 3
    assert result["longest_streak"] == 3
    assert result["n_played"] == 2


def test_main_runs_end_to_end_on_a_tiny_synthetic_log(tmp_path, capsys):
    module = _load_module()
    rows = []
    # A short real-shaped history: mostly strong days (both picks hit),
    # a couple of misses, so streak_progression has real non-reset days
    # to estimate a gain from and the DP has something to solve.
    for i, (p1, p2, hit1, hit2) in enumerate([
        (0.9, 0.85, 1, 1), (0.88, 0.84, 1, 1), (0.6, 0.55, 1, 0),
        (0.91, 0.86, 1, 1), (0.89, 0.83, 1, 1), (0.65, 0.5, 0, 1),
    ]):
        date = f"2026-06-{i + 1:02d}"
        rows.append(_prediction_row(date, 1, p1, hit1, 4))
        rows.append(_prediction_row(date, 2, p2, hit2, 4))
    csv_path = tmp_path / "predictions.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    sys.argv = ["backtest_streak_decision.py", "--predictions-path", str(csv_path), "--bootstrap-samples", "50"]
    module.main()

    out = capsys.readouterr().out
    assert "Real single-path replay" in out
    assert "Bootstrap (50 resamples" in out
    assert ("GO:" in out) or ("NO-GO" in out)
