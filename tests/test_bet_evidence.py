"""scripts/build_bet_evidence.py: the evidence file for the NFL Picks tab."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location("build_bet_evidence", REPO_ROOT / "scripts" / "build_bet_evidence.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_series_reports_record_bankroll_rates_and_a_path_by_season():
    script = _script()
    rng = np.random.default_rng(5)
    n = 400
    frame = pd.DataFrame({
        "season": np.repeat(np.arange(2015, 2025), n // 10), "date": pd.date_range("2015-09-10", periods=n, freq="7D").astype(str),
        "price": -110, "won": (rng.random(n) < 0.6).astype(float),
    })
    frame.loc[3, "won"] = np.nan                      # a push
    out = script.series("k", "Label", "replay", "note", frame)
    assert out["bets"] == n and out["won"] + out["lost"] == n - 1
    assert out["wealth"] > 20 and out["peak"] >= out["wealth"]
    assert out["rate_low"] < out["win_rate"] < out["rate_high"]
    assert out["fade_tripped"] is None
    assert [p["at"] for p in out["path"]] == [str(y) for y in range(2015, 2025)]
    assert script.series("k", "Label", "live", "note", pd.DataFrame()) is None


def test_build_reads_the_logged_bets_and_any_replay_files(tmp_path):
    script = _script()
    data, replays = tmp_path / "data", tmp_path / "replay"
    data.mkdir()
    replays.mkdir()
    pd.DataFrame({
        "date": ["2026-09-13", "2026-09-20", "2026-09-27", "2026-10-04"],
        "status": ["win", "loss", "win", "pending"],
        "spread_bet": ["KC -3", None, "BUF +4", "DAL +7"], "spread_bet_result": ["win", None, "loss", None],
        "total_lean": ["over", "under", None, None], "total_lean_result": ["win", "push", None, None],
        "bet_units": [1.5, 0, 2.0, 0], "bet_profit_units": [3.0, None, -2.0, None],
    }).to_csv(data / "nfl_game_picks_picks.csv", index=False)
    pd.DataFrame({"season": [2024, 2024, 2025], "date": ["2024-09-08", "2024-10-06", "2025-09-07"], "price": [-110, -105, -110],
                  "won": [1.0, 0.0, 1.0]}).to_csv(replays / script.REPLAY_FILES["total"], index=False)
    out = script.build(str(data), str(replays))
    by_key = {s["key"]: s for s in out["series"]}
    assert set(by_key) == {"nfl_spread_live", "nfl_total_replay", "nfl_total_live", "nfl_moneyline_live"}
    assert (by_key["nfl_spread_live"]["won"], by_key["nfl_spread_live"]["lost"]) == (1, 1)
    # A pushed lean is a bet with no result either way.
    assert (by_key["nfl_total_live"]["bets"], by_key["nfl_total_live"]["won"], by_key["nfl_total_live"]["lost"]) == (2, 1, 0)
    # Moneyline bets of different sizes: units are the real profit, the
    # bankroll works on the return per unit staked.
    money = by_key["nfl_moneyline_live"]
    assert money["units"] == 1.0 and "rate_low" not in money
    assert out["bar"] == 20.0 and round(out["break_even"], 4) == 0.5238
