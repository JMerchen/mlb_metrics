"""Unit tests for scripts/backtest_shrinkage.py's own sweep/scoring/go-no-go
logic against a tiny hand-built hitter-hit-log DataFrame - bypassing the
real dfs_backtest.assemble_hitter_hit_log Statcast dependency (monkeypatched),
per this project's "the backtest tool itself is under test, not just a
one-off unchecked script" discipline (see tests/test_backtest_selection_rule.py
for the same pattern)."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "backtest_shrinkage.py"


def _load_backtest_shrinkage_module():
    sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location("backtest_shrinkage", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hit_log_row(key, total_pa, probability, got_hit):
    return {"key_mlbam": key, "Total_PA": total_pa, "probability": probability, "Got_Hit": got_hit}


def test_sweep_restores_config_after_each_run(monkeypatch):
    module = _load_backtest_shrinkage_module()
    original_wave = module.config.WAVE_SHRINKAGE_STRENGTH
    seen_strengths = []

    def fake_assemble(raw_dir, season, days):
        seen_strengths.append(module.config.WAVE_SHRINKAGE_STRENGTH)
        return pd.DataFrame([_hit_log_row(1, 40, 0.3, 1), _hit_log_row(2, 40, 0.3, 0)])

    monkeypatch.setattr(module.dfs_backtest, "assemble_hitter_hit_log", fake_assemble)

    module._sweep("data/raw", 2026, None, [0.0, 25.0, 50.0], "wave")

    # The sweep really did vary the live config value across calls...
    assert seen_strengths == [0.0, 25.0, 50.0]
    # ...and restored it afterward, regardless of the sweep's own outcome -
    # a crash mid-sweep must never leave a stale monkeypatched value behind.
    assert module.config.WAVE_SHRINKAGE_STRENGTH == original_wave


def test_sweep_splits_full_vs_pa_gated_population(monkeypatch):
    module = _load_backtest_shrinkage_module()

    def fake_assemble(raw_dir, season, days):
        return pd.DataFrame([
            _hit_log_row(1, 10, 0.5, 1),   # below BACKTEST_MIN_PLATE_APPEARANCES (30) - full only
            _hit_log_row(2, 40, 0.5, 0),   # above the gate - counted in both
        ])

    monkeypatch.setattr(module.dfs_backtest, "assemble_hitter_hit_log", fake_assemble)

    results = module._sweep("data/raw", 2026, None, [0.0], "wave")

    assert results[0.0]["full"]["n"] == 2
    assert results[0.0]["gated"]["n"] == 1


def test_sweep_empty_hit_log_reports_no_scored_rows(monkeypatch):
    module = _load_backtest_shrinkage_module()
    monkeypatch.setattr(module.dfs_backtest, "assemble_hitter_hit_log", lambda raw_dir, season, days: pd.DataFrame())

    results = module._sweep("data/raw", 2026, None, [0.0, 10.0], "wave")

    assert results[0.0]["full"] is None
    assert results[0.0]["gated"] is None


def test_report_says_go_when_a_nonzero_strength_beats_baseline_on_gated_population(capsys):
    module = _load_backtest_shrinkage_module()
    baseline = {"log_loss": 0.70, "brier_score": 0.24, "roc_auc": 0.55, "accuracy": 0.55, "n": 100}
    better = {"log_loss": 0.65, "brier_score": 0.22, "roc_auc": 0.58, "accuracy": 0.57, "n": 100}
    results = {
        0.0: {"full": baseline, "gated": baseline},
        25.0: {"full": better, "gated": better},
    }

    module.report("test sweep", results)

    assert "GO: strength=25.0" in capsys.readouterr().out


def test_report_says_no_go_when_no_nonzero_strength_beats_baseline(capsys):
    module = _load_backtest_shrinkage_module()
    baseline = {"log_loss": 0.65, "brier_score": 0.22, "roc_auc": 0.58, "accuracy": 0.57, "n": 100}
    worse = {"log_loss": 0.70, "brier_score": 0.24, "roc_auc": 0.55, "accuracy": 0.55, "n": 100}
    results = {
        0.0: {"full": baseline, "gated": baseline},
        25.0: {"full": worse, "gated": worse},
    }

    module.report("test sweep", results)

    assert "NO-GO: strength=0" in capsys.readouterr().out
