"""Real backtest for the 2026-09-08 Beat the Streak complaint: "matchup
isn't weighted enough" AND "the site has shown every batter as
Speculative for weeks."

Both trace to the same live formula (see matchup.compute_matchup_approach's
own docstring for the full real-numbers writeup): `Matchup_Approach =
Approach * Matchup_Hit_Probability ** weight`, where `Approach` already
multiplies together two overlapping, highly correlated hitter-only
signals (Game_Hit_Probability * probability) before the one real
opponent-adjusted signal (Matchup_Hit_Probability) gets a single
multiplicative say. `weight=1.0` (today's live default,
config.MATCHUP_APPROACH_WEIGHT) reproduces the original bare-
multiplication formula exactly.

No-lookahead, reusing the exact per-date recompute technique every other
backtest in this project uses (dfs_backtest._compute_date_outputs) and
dfs_backtest.derive_historical_team_schedule (reconstructs a REAL
historical schedule - including real probable starters - directly from
persisted Statcast, not a live-only API), so the full persisted season is
available, not just a couple of weeks.

Two real questions, answered together (in this order - see module-level
plan) because recalibrating DAILY_PICK_MIN_PROBABILITY against the OLD
formula first would just go stale again the moment the weight changes:

1. Does a real weight > 1.0 produce a better realized selection (higher
   resolved hit rate / lower Brier) than today's live weight=1.0, on a
   REAL, walk-forward-validated split (the weight is picked on an early
   block of dates, confirmed on an untouched final holdout block - this
   project already caught and corrected exactly this leakage mistake
   once this session, NFL's market-tiebreak weight)?
2. Given the WINNING weight's real resulting combined_probability
   distribution, what DAILY_PICK_MIN_PROBABILITY bar achieves the same
   "full day coverage while matching the best resolved hit-rate/Brier
   plateau" standard the original 0.77 was derived with
   (config.DAILY_PICK_MIN_PROBABILITY's own docstring) - on real,
   accumulated live-shaped data instead of a 42-day mostly-NaN-matchup
   replay?

Needs data/raw/statcast_<season>_<month>.parquet (see data.persist_raw_statcast)
and, if present, the saved hit-probability model artifact
(config.HITTER_HIT_PROBABILITY_MODEL_PATH) - a day without a usable model
prediction still gets picks via the plain Matchup_Approach heuristic tier,
same real graceful-degradation `predictions.select_picks` already has.

Usage:
    python scripts/backtest_matchup_weight.py
    python scripts/backtest_matchup_weight.py --season 2026 --holdout-frac 0.2
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from mlb_metrics import config, data, dfs_backtest, dfs_ml, evaluation, matchup, predictions

WEIGHT_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
BAR_GRID = [0.65, 0.68, 0.70, 0.72, 0.75, 0.77, 0.80]


def build_date_pools(dates, persisted: pd.DataFrame, team_schedule: pd.DataFrame) -> list[dict]:
    """One pass over `dates`: for each real historical date, builds the
    real Matchup_Approach ingredient columns (Approach, Matchup_Hit_Probability
    - NOT yet combined, so every weight candidate can be applied cheaply
    afterward) merged with Model_Hit_Probability (when the model predicts
    something for that date), plus that date's real outcomes. Mirrors
    backtest_selection_rule.build_date_pools exactly, minus the
    Matchup_Approach assignment itself (deferred - see select_and_resolve)."""
    pools = []
    for date in dates:
        day = dfs_backtest._compute_date_outputs(persisted, team_schedule, date)
        if day is None:
            continue

        pick_pool = day["outputs"]["wave"].merge(day["matchup_probability"], on="key_mlbam", how="inner")

        hitter_features = dfs_ml.build_hitter_features(
            day["outputs"]["wave"], day["outputs"]["pave"], day["outputs"]["confidence"],
            day["todays_schedule"], day["matchup_probability"],
        )
        model_predictions = dfs_ml.predict_hitter_hit_probability(hitter_features)
        has_model = not model_predictions.empty
        if has_model:
            pick_pool = pick_pool.merge(model_predictions, on="key_mlbam", how="left")

        day_events = persisted[persisted["game_date"] == date]
        got_hit = dfs_backtest.compute_actual_hitter_got_hit(
            data.completed_events(day_events, ["game_date", "batter", "events"])
        )
        pools.append({"date": date, "pick_pool": pick_pool, "got_hit": got_hit, "has_model": has_model})
    return pools


def select_and_resolve(pools: list[dict], weight: float) -> pd.DataFrame:
    """Same real selection predictions.select_picks makes live (the
    Model_Hit_Probability shortlist gate when available, Matchup_Approach
    ranking the survivors), at a given real matchup weight - one real
    resolved-pick row per (date, rank), directly consumable by
    evaluation.py's scoring functions AND `evaluation._combined_probability`."""
    rows = []
    for entry in pools:
        pick_pool = entry["pick_pool"].copy()
        pick_pool["Matchup_Approach"] = matchup.compute_matchup_approach(
            pick_pool["Approach"], pick_pool["Matchup_Hit_Probability"], weight=weight
        )
        picks = predictions.select_picks(pick_pool, entry["date"], rank_metric="Matchup_Approach")
        if picks.empty:
            continue

        picks = picks.merge(
            entry["got_hit"].rename(columns={"Got_Hit": "resolved_hit"}), on="key_mlbam", how="left"
        )
        picks["at_bats"] = picks["resolved_hit"].notna().astype(int)
        picks["actual_hit"] = picks["resolved_hit"]
        picks = picks.drop(columns="resolved_hit")
        rows.append(picks)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=predictions.PREDICTION_COLUMNS)


def report_metrics(label: str, picks_df: pd.DataFrame, total_dates: int) -> dict:
    resolved = evaluation.resolved_only(picks_df)
    dates_with_a_pick = picks_df["date"].nunique() if not picks_df.empty else 0
    top1 = evaluation.top_k_hit_rate(picks_df, 1, require_all=False)
    top2 = evaluation.top_k_hit_rate(picks_df, 2, require_all=False)
    brier = evaluation.brier_score(picks_df)
    ll = evaluation.log_loss(picks_df)
    print(f"  {label}")
    print(f"    n_scored={len(resolved)}, dates_with_a_pick={dates_with_a_pick}/{total_dates}")
    print(f"    any_of_top_1_hit_rate={top1:.4f}" if top1 == top1 else "    any_of_top_1_hit_rate=n/a")
    print(f"    any_of_top_2_hit_rate={top2:.4f}" if top2 == top2 else "    any_of_top_2_hit_rate=n/a")
    print(f"    brier_score={brier:.4f}" if brier == brier else "    brier_score=n/a")
    print(f"    log_loss={ll:.4f}" if ll == ll else "    log_loss=n/a")
    return {"top1": top1, "top2": top2, "brier": brier, "log_loss": ll, "n": len(resolved)}


def bar_report(label: str, picks_df: pd.DataFrame, bar: float) -> dict:
    """The exact real methodology config.DAILY_PICK_MIN_PROBABILITY was
    originally derived with (see that constant's own docstring): day
    coverage (fraction of resolved days with >=1 "recommended"-grade pick
    at this bar) and the resolved hit rate/Brier among ONLY those
    recommended-grade picks - the real streak-counting subset."""
    resolved = evaluation.resolved_only(picks_df)
    if resolved.empty:
        return {"bar": bar, "day_coverage": float("nan"), "hit_rate": float("nan"), "brier": float("nan"), "n": 0}
    combined = evaluation._combined_probability(resolved)
    recommended = resolved[combined >= bar]
    total_days = resolved["date"].nunique()
    days_with_a_recommendation = recommended["date"].nunique() if not recommended.empty else 0
    day_coverage = days_with_a_recommendation / total_days if total_days else float("nan")
    hit_rate = recommended["actual_hit"].astype(float).mean() if not recommended.empty else float("nan")
    brier = ((recommended["actual_hit"].astype(float) - combined[combined >= bar]) ** 2).mean() if not recommended.empty else float("nan")
    return {"bar": bar, "day_coverage": day_coverage, "hit_rate": hit_rate, "brier": brier, "n": len(recommended)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--holdout-frac", type=float, default=0.25)
    args = parser.parse_args()

    season = args.season or config.SEASON_START.year
    persisted = data.load_persisted_statcast(args.raw_dir, season)
    if persisted is None:
        print(f"No persisted Statcast in {args.raw_dir} for season {season} - nothing to backtest.")
        return

    team_schedule = dfs_backtest.derive_historical_team_schedule(persisted)
    all_dates = sorted(team_schedule["date"].unique())
    n_holdout = max(1, int(round(len(all_dates) * args.holdout_frac)))
    train_dates, holdout_dates = all_dates[:-n_holdout], all_dates[-n_holdout:]
    print(f"{len(all_dates)} real dates total: {len(train_dates)} train (pick the weight/bar), "
          f"{len(holdout_dates)} untouched holdout (confirm only).")

    print("\nBuilding real per-date pools once (expensive step, reused for every weight candidate)...")
    train_pools = build_date_pools(train_dates, persisted, team_schedule)
    holdout_pools = build_date_pools(holdout_dates, persisted, team_schedule)
    print(f"{len(train_pools)} usable train dates, {len(holdout_pools)} usable holdout dates.")

    print("\n=== Step 1: real weight sweep (picked on train, confirmed on holdout) ===")
    train_results = {}
    for weight in WEIGHT_GRID:
        picks = select_and_resolve(train_pools, weight)
        train_results[weight] = report_metrics(f"weight={weight}", picks, len(train_pools))

    baseline_top2 = train_results[1.0]["top2"]
    # NaN-safe: a weight with no resolved picks at all (top2 is NaN) sorts
    # last rather than crashing the comparison.
    scored_weights = [w for w in WEIGHT_GRID if train_results[w]["top2"] == train_results[w]["top2"]]
    best_weight = max(scored_weights, key=lambda w: (train_results[w]["top2"], -train_results[w]["brier"]))
    print(f"\nBest weight on TRAIN by any_of_top_2_hit_rate (tiebreak: lower Brier): {best_weight} "
          f"(baseline weight=1.0 top2={baseline_top2:.4f})")

    print(f"\n=== Confirming weight={best_weight} vs. weight=1.0 baseline on the UNTOUCHED holdout ===")
    holdout_baseline_picks = select_and_resolve(holdout_pools, 1.0)
    holdout_baseline = report_metrics("weight=1.0 (live baseline)", holdout_baseline_picks, len(holdout_pools))
    holdout_best_picks = select_and_resolve(holdout_pools, best_weight)
    holdout_best = report_metrics(f"weight={best_weight} (candidate)", holdout_best_picks, len(holdout_pools))

    weight_clears_gate = (
        holdout_best["top2"] == holdout_best["top2"] and holdout_baseline["top2"] == holdout_baseline["top2"]
        and holdout_best["top2"] >= holdout_baseline["top2"] and holdout_best["brier"] <= holdout_baseline["brier"]
    )
    if weight_clears_gate:
        print(f"\n-> weight={best_weight} clears the real save-gate (>= baseline hit rate AND <= baseline Brier "
              f"on the untouched holdout) - adopting it for step 2.")
        final_weight = best_weight
        final_picks_for_bar = pd.concat([
            select_and_resolve(train_pools, final_weight), holdout_best_picks
        ], ignore_index=True)
    else:
        print(f"\n-> weight={best_weight} does NOT clear the real save-gate on the untouched holdout - "
              f"reporting this honestly and keeping MATCHUP_APPROACH_WEIGHT=1.0.")
        final_weight = 1.0
        final_picks_for_bar = pd.concat([
            select_and_resolve(train_pools, 1.0), holdout_baseline_picks
        ], ignore_index=True)

    print(f"\n=== Step 2: real DAILY_PICK_MIN_PROBABILITY bar sweep at weight={final_weight} "
          f"(all {len(all_dates)} real dates, same methodology the original 0.77 was derived with) ===")
    for bar in BAR_GRID:
        result = bar_report(f"bar={bar}", final_picks_for_bar, bar)
        coverage_str = f"{result['day_coverage']:.3f}" if result["day_coverage"] == result["day_coverage"] else "n/a"
        hit_rate_str = f"{result['hit_rate']:.4f}" if result["hit_rate"] == result["hit_rate"] else "n/a"
        brier_str = f"{result['brier']:.4f}" if result["brier"] == result["brier"] else "n/a"
        print(f"  bar={bar}: day_coverage={coverage_str}, n_recommended={result['n']}, "
              f"hit_rate={hit_rate_str}, brier={brier_str}")

    print(f"\nFinal recommendation: MATCHUP_APPROACH_WEIGHT={final_weight} "
          f"(vs. current live 1.0) - see the bar sweep above for a real "
          f"DAILY_PICK_MIN_PROBABILITY candidate at that weight, whichever "
          f"achieves reasonable day coverage without giving up the hit-rate/Brier plateau.")


if __name__ == "__main__":
    main()
