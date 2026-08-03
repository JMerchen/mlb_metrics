"""Real backtest of predictions.select_picks' SELECTION RULE - not just
the hit-probability model's own already-known calibration accuracy
(scripts/train_hitter_hit_model.py's holdout log_loss/Brier/ROC AUC
answers that question). This answers a different, real one: does
ranking-and-gating by Model_Hit_Probability at the top-N granularity
select_picks actually uses produce a better realized selection than
today's Matchup_Approach heuristic - a question about discriminative
power at the extreme top of the distribution, not average calibration.

No-lookahead, reusing dfs_backtest._compute_date_outputs's per-date
recompute (the same technique every other backtest in this project uses)
and the REAL predictions.select_picks/evaluation.py functions directly -
not a reimplementation of either.

Restricted to the SAME final holdout date block
scripts/train_hitter_hit_model.py validated the saved model artifact on
(config.ML_FINAL_HOLDOUT_DATES) - evaluating on training dates would be
leakage. Uses the LAST ML_FINAL_HOLDOUT_DATES dates of whatever's
currently persisted, so any real Statcast history accumulated since the
model's original training run naturally widens the window.

Sweeps a small grid of candidate min_model_probability thresholds and
reports the same hit-rate/Brier-score-plateau-without-zero-pick-days
discipline config.HITTER_MIN_PROBABILITY/config.DAILY_PICK_MIN_PROBABILITY
were each derived with - reported honestly even if the model does NOT
beat Matchup_Approach at this granularity (this project's "report
honestly either way" standard).

Needs data/raw/statcast_<season>.parquet (see scripts/wave.py) and the
saved model artifact (config.HITTER_HIT_PROBABILITY_MODEL_PATH,
scripts/train_hitter_hit_model.py).

Usage:
    python scripts/backtest_selection_rule.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from mlb_metrics import config, data, dfs_backtest, dfs_ml, evaluation, predictions

MIN_MODEL_PROBABILITY_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def build_date_pools(dates, persisted: pd.DataFrame, team_schedule: pd.DataFrame) -> list[dict]:
    """One pass over `dates`: for each real historical date, builds the
    Matchup_Approach pick_pool (today's live heuristic tier) merged with
    Model_Hit_Probability (when the model predicts something for that
    date), plus that date's real outcomes. Reused by both the fixed-
    heuristic comparison and the threshold sweep below, so the expensive
    per-date pipeline recompute + model inference happens exactly once
    per date regardless of how many min_model_probability candidates get
    evaluated afterward."""
    pools = []
    for date in dates:
        day = dfs_backtest._compute_date_outputs(persisted, team_schedule, date)
        if day is None:
            continue

        pick_pool = day["outputs"]["wave"].merge(day["matchup_probability"], on="key_mlbam", how="inner")
        pick_pool["Matchup_Approach"] = pick_pool["Approach"] * pick_pool["Matchup_Hit_Probability"]

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


def select_and_resolve(pools: list[dict], rank_metric: str, **select_kwargs) -> pd.DataFrame:
    """Runs predictions.select_picks on each cached date pool (skipping a
    date the Model_Hit_Probability tier has no usable prediction for, when
    `rank_metric="Model_Hit_Probability"`), then resolves each returned
    pick against that date's REAL Got_Hit outcome. Returns a
    predictions.PREDICTION_COLUMNS-shaped frame with actual_hit/at_bats
    already filled in, directly consumable by evaluation.py's scoring
    functions.

    select_picks' gate activates off Model_Hit_Probability's mere
    presence as a column, regardless of rank_metric - so when simulating
    a non-model tier, the column is dropped first. Otherwise a pool built
    for the model-tier simulation (which also carries Model_Hit_
    Probability for the OTHER simulation on the same date) would wrongly
    apply the new model gate to the historical heuristic-tier simulation,
    which never had a model gate in real production."""
    rows = []
    for entry in pools:
        if rank_metric == "Model_Hit_Probability" and not entry["has_model"]:
            continue
        pick_pool = entry["pick_pool"]
        if rank_metric != "Model_Hit_Probability":
            pick_pool = pick_pool.drop(columns="Model_Hit_Probability", errors="ignore")
        picks = predictions.select_picks(pick_pool, entry["date"], rank_metric=rank_metric, **select_kwargs)
        if picks.empty:
            continue

        picks = picks.merge(
            entry["got_hit"].rename(columns={"Got_Hit": "resolved_hit"}), on="key_mlbam", how="left"
        )
        # Present in that date's real completed-events table => had a real
        # at-bat that date (at_bats=1, don't need the exact count here);
        # absent => a picked player with zero at-bats that day (no_game).
        picks["at_bats"] = picks["resolved_hit"].notna().astype(int)
        picks["actual_hit"] = picks["resolved_hit"]
        picks = picks.drop(columns="resolved_hit")
        rows.append(picks)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=predictions.PREDICTION_COLUMNS)


def report(label: str, picks_df: pd.DataFrame, total_dates: int) -> None:
    resolved = evaluation.resolved_only(picks_df)
    dates_with_a_pick = picks_df["date"].nunique() if not picks_df.empty else 0
    zero_pick_dates = total_dates - dates_with_a_pick
    print(f"\n{label}")
    print(f"  n_scored={len(resolved)}, dates_with_a_pick={dates_with_a_pick}/{total_dates} (zero_pick_dates={zero_pick_dates})")
    for k in (1, 2, config.BACKTEST_TOP_N):
        rate = evaluation.top_k_hit_rate(picks_df, k, require_all=False)
        rate_str = f"{rate:.4f}" if rate == rate else "n/a"
        print(f"  any_of_top_{k}_hit_rate={rate_str}")
    brier = evaluation.brier_score(picks_df)
    ll = evaluation.log_loss(picks_df)
    print(f"  brier_score={brier:.4f}" if brier == brier else "  brier_score=n/a")
    print(f"  log_loss={ll:.4f}" if ll == ll else "  log_loss=n/a")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--holdout-dates", type=int, default=config.ML_FINAL_HOLDOUT_DATES)
    args = parser.parse_args()

    season = args.season or config.SEASON_START.year
    persisted = data.load_persisted_statcast(args.raw_dir, season)
    if persisted is None:
        print(f"No persisted Statcast in {args.raw_dir} for season {season} - nothing to backtest.")
        return

    team_schedule = dfs_backtest.derive_historical_team_schedule(persisted)
    all_dates = sorted(team_schedule["date"].unique())
    dates = all_dates[-args.holdout_dates:] if len(all_dates) > args.holdout_dates else all_dates
    print(f"Backtesting the selection rule over the final {len(dates)} real dates (of {len(all_dates)} total) - "
          f"the same holdout block scripts/train_hitter_hit_model.py validates the saved model artifact on.")

    pools = build_date_pools(dates, persisted, team_schedule)
    total_dates = len(pools)
    dates_with_model = sum(1 for p in pools if p["has_model"])
    print(f"{total_dates} dates have usable prior history; {dates_with_model} of those have a usable Model_Hit_Probability prediction.")

    heuristic_picks = select_and_resolve(pools, "Matchup_Approach")
    report("Matchup_Approach (today's live heuristic tier)", heuristic_picks, total_dates)

    print("\n=== min_model_probability grid sweep (Model_Hit_Probability tier) ===")
    for threshold in MIN_MODEL_PROBABILITY_GRID:
        model_picks = select_and_resolve(pools, "Model_Hit_Probability", min_model_probability=threshold)
        report(f"Model_Hit_Probability (min_model_probability={threshold})", model_picks, dates_with_model)


if __name__ == "__main__":
    main()
