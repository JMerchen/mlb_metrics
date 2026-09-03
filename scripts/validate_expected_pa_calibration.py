"""Directly tests whether Expected_PA-weighted `probability` is better
calibrated against real outcomes than the old flat-trials version - using
EVERY qualified hitter each date (not just the 2 that got picked), which
gives ~100x the statistical power of comparing top-2-picks-per-day hit
rates. This is the real test of "does weighting by real expected at-bats
actually make the probability estimate more accurate", isolated from
ranking-metric/qualifier noise.

For each of the 153 real historical dates, every PA-qualified hitter
(config.BACKTEST_MIN_PLATE_APPEARANCES) gets both a flat-trials
probability (the old constant WAVE_TRIALS_PER_GAME) and an Expected_PA-
weighted one (the new per-batter trials count), resolved against that
date's real Got_Hit outcome - no lookahead (both computed from
history strictly before the date, dfs_backtest._compute_date_outputs).
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from mlb_metrics import config, data, dfs_backtest

OUTPUT = "data/backtest/expected_pa_calibration.csv"


def run(raw_dir: str = "data/raw", season: int | None = None) -> pd.DataFrame:
    season = season or config.SEASON_START.year
    persisted = data.load_persisted_statcast(raw_dir, season)
    team_schedule = dfs_backtest.derive_historical_team_schedule(persisted)
    dates = sorted(team_schedule["date"].unique())
    print(f"{len(dates)} real dates: {dates[0]} to {dates[-1]}", flush=True)

    rows = []
    start = time.time()
    for i, current_date in enumerate(dates):
        day = dfs_backtest._compute_date_outputs(persisted, team_schedule, current_date)
        if day is None:
            continue
        wave = day["outputs"]["wave"]
        qualified = wave[(wave["PA_L"] + wave["PA_R"]) >= config.BACKTEST_MIN_PLATE_APPEARANCES].copy()
        if qualified.empty:
            continue

        qualified["old_probability"] = 1 - (1 - qualified["WAVE"]) ** config.WAVE_TRIALS_PER_GAME
        # `probability` on `wave` is already the Expected_PA-weighted version
        # (hitters.assemble_hitters recomputed it) - kept as-is here.

        day_events = persisted[persisted["game_date"] == current_date]
        actual = dfs_backtest.compute_actual_hitter_got_hit(
            data.completed_events(day_events, ["game_date", "batter", "events"])
        )
        scored = qualified.merge(actual, on="key_mlbam", how="inner")
        if scored.empty:
            continue

        scored["date"] = current_date
        rows.append(scored[["date", "key_mlbam", "WAVE", "Expected_PA", "probability", "old_probability",
                             "Game_Hit_Probability", "Got_Hit"]])

        if (i + 1) % 10 == 0 or i == len(dates) - 1:
            print(f"  [{i+1}/{len(dates)}] {current_date} done ({time.time()-start:.0f}s)", flush=True)

    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return result


if __name__ == "__main__":
    result = run()
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    print(f"Wrote {OUTPUT} ({len(result)} rows).")
